# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen SDFG signature -- snapshotted at build time, verified at codegen.

Captured when the kernel SDFG leaves ``SDFGBuilder.build()`` and pinned
on the SDFG (``sdfg._frozen_signature``).  The binding emitter uses this
snapshot, not the live SDFG, so later transformations can't silently
invalidate a generated wrapper.

Drift gate lives in ``build_fortran_library``: before emit/link it calls
``fs.verify_against(sdfg)``, raising ``SignatureDriftError`` on
divergence.  dace-core contributes only the opaque ``SDFG.frontend_metadata``
dict this rides in and never reads it -- the contract is dace-fortran-only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import dace
from dace.data import Data

if TYPE_CHECKING:
    from dace import SDFG

# Where a Fortran caller's buffers live unless a pass says otherwise.  Anything a
# transformation moves into a ``GPU_*`` storage is a device relocation the binding has
# to bridge; everything else is still reachable through the pointer the caller passed.
HOST_STORAGE = "CPU_Heap"
DEVICE_STORAGE_PREFIX = "GPU_"


class SignatureDriftError(RuntimeError):
    """Raised when the live SDFG's arglist / free_symbols disagrees with
    a ``FrozenSignature`` attached to it."""


class ModuleOrigin(NamedTuple):
    """The module and entity a module-global SDFG name reads."""

    module: str
    entity: str


class FrozenArgKind(Enum):
    """How an SDFG argument crosses the binding."""

    ARRAY = "array"
    SCALAR = "scalar"
    SYMBOL = "symbol"
    #: An integer communicator; the wrapper converts it via ``MPI_Comm_f2c``.
    MPI_COMM = "mpi_comm"


@dataclass(frozen=True, slots=True)
class FrozenArg:
    """One argument in the frozen signature.

    sdfg_name: name DaCe sees, may differ from fortran_name after struct
        flattening (``st%u`` -> ``st_u``).
    kind: how the argument crosses the binding (:class:`FrozenArgKind`).
    from_struct_member: original Fortran expr (``st%u``) if extracted by
        hlfir-flatten-structs, else None.
    is_written: True if this is a module-scope global the kernel WRITES;
        binding copies the final value back to the host module var.
    """

    fortran_name: str
    sdfg_name: str
    kind: FrozenArgKind
    dtype: str
    rank: int
    shape: tuple[str, ...] = field(default_factory=tuple)
    intent: str = ""
    from_struct_member: str | None = None
    is_written: bool = False
    # Provenance for a flattened component of a MODULE-LEVEL array-of-structs
    # global (QE ``becxx(ikq)%k``, TYPE(bec_type) ALLOCATABLE).  This arg is
    # the SoA image (``becxx_k``); binding sources it via an AoS<->SoA copy
    # loop instead of a direct assign.  Empty for ordinary args.
    #   aos_origin_mod/struct -- owning module / AoS global name.
    #   aos_member_path       -- '%'-joined component path.
    #   aos_outer_rank        -- leading record-array (element) dim count.
    #   aos_root_rank         -- dims of a root that is itself an array of
    #                            records (``flux_n(j) % vf(i) % sf``), ahead
    #                            of the aos_outer_rank ones.
    #   global_alloc_inside   -- kernel ALLOCATEs the component: binding
    #                            allocates the host global, skips copy-in.
    aos_origin_mod: str = ""
    aos_origin_struct: str = ""
    aos_member_path: str = ""
    aos_outer_rank: int = 0
    aos_root_rank: int = 0
    aos_struct_static: bool = False  # the root record array has a fixed size: no presence guard
    global_alloc_inside: bool = False
    aos_struct_pointer: bool = False
    aos_member_pointer: bool = False
    # Module-origin global storage class: copy-in guarded with allocated()/
    # associated() so an unallocated host isn't read; both false = static.
    module_origin_allocatable: bool = False
    module_origin_pointer: bool = False
    # Where the buffer lives.  ``storage`` is the CALLER's location, frozen at build
    # time and never rewritten -- a Fortran dummy is host memory, which is why the
    # default is a host storage rather than empty.  Offloading relocates the kernel's
    # copy, and :func:`refreeze` records that in ``device_storage``; the binding then
    # brackets the call in OpenACC data clauses instead of handing the host pointer
    # straight through.  Back on the host, ``device_storage`` clears again.
    storage: str = HOST_STORAGE
    device_storage: str = ""
    # The packed companion of an array of records this argument is one element of (``element_index``, 0-based):
    # dace_fortran.array_split split the companion, and the binding passes that element's member itself.
    element_of: str = ""
    element_index: int = 0

    @property
    def acc_data_clause(self) -> str:
        """``copyin`` / ``copyout`` / ``copy`` for this arg, '' if it never leaves the host.

        Direction is the Fortran intent: an ``in`` dummy the kernel doesn't write needs
        only the host->device leg, an ``out`` dummy only the device->host one, anything
        else both.
        """
        if self.kind is not FrozenArgKind.ARRAY or not self.device_storage or self.device_storage == self.storage:
            return ""
        intent = self.intent.lower()
        if intent == "in" and not self.is_written:
            return "copyin"
        if intent == "out":
            return "copyout"
        return "copy"

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (``shape`` tuple becomes a list)."""
        d = asdict(self)
        d["shape"] = list(self.shape)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FrozenArg:
        """Rebuild from a :meth:`to_dict` mapping (list back to tuple)."""
        d = dict(d)
        d["shape"] = tuple(d.get("shape", []))
        d["kind"] = FrozenArgKind(d["kind"])
        return cls(**d)


@dataclass(frozen=True, slots=True)
class FrozenSignature:
    """Full snapshot of one entry subroutine's SDFG signature.

    ``args`` order matches ``__program_<entry>``'s C params: data args
    sorted, then scalars, then free symbols (DaCe's generate_headers order).
    """

    entry: str  # 'compute_tendencies'
    mangled: str  # '_QPcompute_tendencies'
    args: tuple[FrozenArg, ...]
    free_symbols: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = 1
    # Auto-detected module-global provenance for SDFG names that aren't
    # outer dummies.  Maps sdfg_name -> (module, entity).  Binding emitter
    # merges with hand-authored OriginalInterface.module_symbol_sources
    # (explicit map wins on conflict).
    module_symbol_origins: dict[str, ModuleOrigin] = field(default_factory=dict)
    # Integer communicator dummy the wrapper feeds (via MPI_Comm_f2c +
    # MPI_Comm_size) into __user_comm/__user_comm_size at dace_init_<entry>
    # time.  None if no runtime MPI comm.
    user_comm_source: str | None = None
    # The dtype (as :func:`dtype_string` renders it) DaCe gives each free symbol it passes by value: the
    # C entry takes a shape-only symbol in the width of the SDFG's symbol table (int64 for an inferred
    # extent), and a wrapper declaring another width hands it a half-written register. Derived from the SDFG
    # (filled in from it when a snapshot lacks it), so it takes no part in comparing snapshots.
    symbol_dtypes: dict[str, str] = field(default_factory=dict, compare=False)

    # ----- I/O ---------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (every tuple becomes a list).

        Also the on-SDFG storage form -- see :func:`attach_to_sdfg`.
        """
        return {
            "entry": self.entry,
            "mangled": self.mangled,
            "args": [a.to_dict() for a in self.args],
            "free_symbols": list(self.free_symbols),
            "schema_version": self.schema_version,
            "module_symbol_origins": {k: list(v) for k, v in self.module_symbol_origins.items()},
            "user_comm_source": self.user_comm_source,
            "symbol_dtypes": dict(self.symbol_dtypes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FrozenSignature:
        """Rebuild from a :meth:`to_dict` mapping (lists back to tuples)."""
        return cls(
            entry=d["entry"],
            mangled=d["mangled"],
            args=tuple(FrozenArg.from_dict(a) for a in d["args"]),
            free_symbols=tuple(d.get("free_symbols", [])),
            schema_version=d.get("schema_version", 1),
            module_symbol_origins={k: ModuleOrigin(*v) for k, v in d.get("module_symbol_origins", {}).items()},
            user_comm_source=d.get("user_comm_source"),
            symbol_dtypes=dict(d.get("symbol_dtypes", {})),
        )

    def to_json(self, path: str) -> None:
        """Write the snapshot to ``path`` as indented JSON."""
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> FrozenSignature:
        """Load a snapshot previously written by :meth:`to_json`."""
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    # ----- Drift check -------------------------------------------------

    def verify_against(self, sdfg: SDFG) -> None:
        """Compare live ``sdfg.arglist()`` + free symbols against this
        snapshot; raise ``SignatureDriftError`` on divergence (arg name
        set/order, dtype per arg, free-symbol set).

        Doesn't check dimensionality past order/dtype -- symbolic shapes
        may canonicalise; codegen catches concrete mismatches later.
        """
        live_arglist = sdfg.arglist()
        live_fs = {str(s) for s in sdfg.free_symbols}
        snap_fs = set(self.free_symbols)

        # arglist() folds free symbols into the arg list; the snapshot
        # models them separately, so validate as two partitions here.
        live_names = [k for k in live_arglist if k not in live_fs]
        snap_names = [a.sdfg_name for a in self.args if a.sdfg_name not in snap_fs]
        if live_names != snap_names:
            raise SignatureDriftError(
                f"signature drift on {self.entry!r}: expected args {snap_names}, got {live_names}"
            )

        # dtype per data/scalar arg -- skip free-symbol args (checked
        # below) and any snapshot arg the live arglist no longer carries.
        for a in self.args:
            if a.sdfg_name in snap_fs or a.sdfg_name not in live_arglist:
                continue
            live_dtype = dtype_string(live_arglist[a.sdfg_name])
            if live_dtype != a.dtype:
                raise SignatureDriftError(
                    f"signature drift on {self.entry!r}: arg {a.sdfg_name!r} "
                    f"dtype {a.dtype!r} in snapshot but {live_dtype!r} now"
                )

        if live_fs != snap_fs:
            raise SignatureDriftError(
                f"signature drift on {self.entry!r}: expected free symbols {sorted(snap_fs)}, got {sorted(live_fs)}"
            )


# ----- On-SDFG storage ----------------------------------------------------
#
# The snapshot lives inside ``sdfg.frontend_metadata``, a dace Property, so it
# survives ``sdfg.save()`` / ``SDFG.from_file()``.  It is still reached through
# the historical ``sdfg._frozen_signature`` name, installed below as a
# descriptor on ``SDFG``; before this it was a plain Python attribute, which
# every serialization round-trip silently dropped.

SDFG_METADATA_KEY = "frozen_signature"
_CACHE_ATTR = "_frozen_signature_cache"

# Argument kinds an optimization pass is allowed to delete; see :func:`refreeze`.
_MAY_SHRINK = frozenset({FrozenArgKind.SCALAR, FrozenArgKind.SYMBOL})


def get_frozen_signature(sdfg: SDFG) -> FrozenSignature | None:
    """Deserialise the snapshot stored on ``sdfg``; None if it carries none."""
    raw = _metadata(sdfg).get(SDFG_METADATA_KEY)
    if raw is None:
        return None
    # Hand back the same object while the stored dict is untouched, so repeated
    # reads don't rebuild several hundred FrozenArgs apiece.
    cached = vars(sdfg).get(_CACHE_ATTR)
    if cached is not None and cached[0] is raw:
        return cached[1]
    frozen = FrozenSignature.from_dict(raw)
    if frozen.free_symbols and not frozen.symbol_dtypes:
        # A snapshot taken before the widths were recorded: read them from the live SDFG.
        frozen = replace(frozen, symbol_dtypes=symbol_dtypes_of(sdfg, frozen.free_symbols))
    vars(sdfg)[_CACHE_ATTR] = (raw, frozen)
    return frozen


def symbol_dtypes_of(sdfg: SDFG, names: Iterable[str]) -> dict[str, str]:
    """The dtype DaCe passes each of the free symbols ``names`` with: its argument list's, which is the symbol
    table's entry or DaCe's default for an undeclared symbol."""
    arglist = sdfg.arglist()
    out: dict[str, str] = {}
    for name in names:
        if name in arglist:
            out[name] = dtype_string(arglist[name])
        elif name in sdfg.symbols:
            out[name] = str(sdfg.symbols[name])
    return out


def _metadata(sdfg: SDFG) -> dict[str, Any]:
    """The SDFG's mutable ``frontend_metadata`` dict (dace's property is annotated as a read-only mapping)."""
    return cast(dict[str, Any], sdfg.frontend_metadata)


def attach_to_sdfg(sdfg: SDFG, frozen: FrozenSignature | None) -> None:
    """Store ``frozen`` on ``sdfg`` in serialized form; None clears it."""
    if frozen is None:
        _metadata(sdfg).pop(SDFG_METADATA_KEY, None)
        vars(sdfg).pop(_CACHE_ATTR, None)
        return
    raw = frozen.to_dict()
    _metadata(sdfg)[SDFG_METADATA_KEY] = raw
    vars(sdfg)[_CACHE_ATTR] = (raw, frozen)


def _install_sdfg_accessor() -> None:
    """Make ``sdfg._frozen_signature`` a view onto ``sdfg.frontend_metadata``."""
    from dace.sdfg import SDFG

    if isinstance(SDFG.__dict__.get("_frozen_signature"), property):
        return
    if "frontend_metadata" not in cast(Any, SDFG).__properties__:  # set by dace's make_properties
        raise RuntimeError(
            "this dace has no SDFG.frontend_metadata property, so a frozen "
            "signature could not survive save/load; update dace"
        )
    cast(Any, SDFG)._frozen_signature = property(get_frozen_signature, attach_to_sdfg)  # installed accessor


_install_sdfg_accessor()


def drop_unread_module_inputs(sdfg: SDFG) -> list[str]:
    """Remove the inputs ``sdfg`` takes from a module variable but no longer reads, and return their names.

    The frontend hands an entry every module variable its code may read; once optimisation has folded the branches
    that read them, most are dead weight the binding would still copy in on every call.  Only module variables go:
    the caller's own buffers are the ABI.  Re-snapshot with :func:`refreeze` afterwards.
    """
    from dace_fortran.pipelines import read_data

    frozen = get_frozen_signature(sdfg)
    if frozen is None:
        return []
    read = read_data(sdfg)
    unread = [
        name
        for name in frozen.module_symbol_origins
        if name in sdfg.arrays and not sdfg.arrays[name].transient and name not in read
    ]
    for name in unread:
        sdfg.remove_data(name, validate=False)
    return unread


def refreeze(sdfg: SDFG) -> FrozenSignature:
    """Re-snapshot after a DELIBERATE transformation of the built SDFG (e.g. an optimization
    pipeline run between ``build()`` and ``build_fortran_library``), so the bindings regenerate
    against the live signature instead of tripping the drift check.

    Contract -- optimization must not change the BUFFER side of the Fortran-facing ABI:

    * array (and MPI communicator) args must match the original snapshot exactly -- names,
      order, dtypes.  The caller's buffers are the ABI, so one going missing means the
      wrapper would stop passing memory the kernel still expects, or vice versa;
    * scalar args, the free-symbol set and the args the binding reads from a module may
      SHRINK, which is what specialization does: folding a value to a constant deletes its
      argument, and the binding simply derives and passes fewer. A scalar the kernel no
      longer reads costs the caller nothing, and a module variable is no buffer of the
      caller's (a module scalar crosses as a one-element array, ABI-preserved);
    * anything NEW is refused, symbol or scalar -- the binding has no value derivation
      for an argument that was not in the Fortran interface.

    An array's STORAGE may change, which is how offloading shows up here: the caller-side
    location stays pinned in ``FrozenArg.storage`` and the kernel's new one is recorded in
    ``device_storage``, so the binding can bracket the call in OpenACC data clauses rather
    than pass a host pointer to a device kernel.

    Returns the new snapshot and attaches it to ``sdfg._frozen_signature``.
    """
    frozen = get_frozen_signature(sdfg)
    if frozen is None:
        raise RuntimeError(
            f"refreeze: SDFG {sdfg.name!r} carries no _frozen_signature; it must come from SDFGBuilder.build()"
        )
    live_arglist = sdfg.arglist()
    live_fs = {str(s) for s in sdfg.free_symbols}
    snap_fs = set(frozen.free_symbols)

    added = sorted(live_fs - snap_fs)
    if added:
        raise SignatureDriftError(
            f"refreeze on {frozen.entry!r}: optimization introduced free "
            f"symbols {added} the binding cannot derive values for"
        )

    dropped_buffers = [
        a.sdfg_name
        for a in frozen.args
        if a.kind not in _MAY_SHRINK
        and a.sdfg_name not in frozen.module_symbol_origins
        and a.sdfg_name not in live_arglist
    ]
    if dropped_buffers:
        raise SignatureDriftError(
            f"refreeze on {frozen.entry!r}: optimization removed "
            f"args {dropped_buffers}; only scalars, free symbols and module variables may shrink"
        )

    def _survives(a: FrozenArg) -> bool:
        # Anything gone from both the arglist and the free symbols was folded to a constant:
        # drop it so the regenerated binding stops deriving and passing it.  Only scalars,
        # symbols and module variables reach here -- the rest are refused above.
        return a.sdfg_name in live_arglist or a.sdfg_name in live_fs

    def _relocate(a: FrozenArg) -> FrozenArg:
        # Host-side storage churn (Register, Pinned) still hands the caller's pointer
        # straight through, so only a move into device memory is worth recording.
        live = sdfg.arrays.get(a.sdfg_name)
        name = live.storage.name if live is not None else ""
        return replace(a, device_storage=name if name.startswith(DEVICE_STORAGE_PREFIX) else "")

    kept_symbols = tuple(s for s in frozen.free_symbols if s in live_fs)
    new = replace(
        frozen,
        args=tuple(_relocate(a) for a in frozen.args if _survives(a)),
        free_symbols=kept_symbols,
        symbol_dtypes=symbol_dtypes_of(sdfg, kept_symbols),
    )
    # Full re-validation (arg partition, per-arg dtypes, symbol set) against the live SDFG.
    new.verify_against(sdfg)
    attach_to_sdfg(sdfg, new)
    return new


def dtype_string(desc: Data) -> str:
    """Stringify a DaCe data descriptor's dtype for comparison."""
    t = desc.dtype
    if isinstance(t, dace.dtypes.opaque):
        # opaque.to_string() is unimplemented here; ctype is its identity.
        return t.ctype
    # typeclass instances have to_string; fall back to str otherwise.
    return t.to_string() if isinstance(t, dace.dtypes.typeclass) else str(t)
