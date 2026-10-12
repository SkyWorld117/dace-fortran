# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thin coordinator -- turns ``(FrozenSignature, OriginalInterface,
FlattenPlan)`` into a ``<entry>_bindings.f90`` file.

Real work lives in sibling modules (``flatten_plan.py`` data model,
``loop_copy.py`` renderers, ``block_builders.py`` builders +
``assemble_module``); this file is pure orchestration.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from dace_fortran.bindings.acc_transfers import AccTransferPlan, Directive
from dace_fortran.bindings.block_builders import (
    assemble_module,
    build_c_interface,
    build_finalize,
    build_handle_state,
    build_wrapper_body,
    build_wrapper_head,
    build_wrapper_tail,
    check_init_symbols,
    splice_acc_staging,
)
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenSignature
from dace_fortran.bindings.module_access import ExposedVariable
from dace_fortran.bindings.specialization import (
    Specialization,
    build_specialization_guard,
    decline_after_record,
    guard_module_sources,
    render_symbol_record,
)
from dace_fortran.bindings.storage_guard import STORAGE_GUARD_HELPERS, build_storage_guard


def emit_bindings(
    frozen: FrozenSignature,
    iface: OriginalInterface,
    plan: FlattenPlan,
    out_path: str | Path,
    dace_arglist: tuple[str, ...] = (),
    enum_maps: dict | None = None,
    acc_residency: AccTransferPlan | None = None,
    init_symbols: tuple[str, ...] | None = None,
    directive: Directive = Directive.OPENACC,
    module_allocatables_are_targets: bool = True,
    specialization: Specialization | None = None,
    record_symbols_to: str | Path | None = None,
    record_only: bool = False,
    private_access: Mapping[str, Mapping[str, ExposedVariable]] | None = None,
    private_imports: Sequence[str] = (),
) -> Path:
    """Emit a Fortran binding module for the built SDFG.

    Creates ``out_path``'s parent dir if missing; overwrites any existing
    file.  ``dace_arglist`` is live codegen output (``CompiledSDFG._sig``),
    not snapshotted in ``FrozenSignature`` -- empty falls back to
    ``frozen.args`` order.  ``enum_maps`` (from
    :func:`rewrite_string_enum_to_integer`) makes the binding accept a
    ``CHARACTER`` dummy and ``SELECT CASE``-translate it to the integer
    the SDFG expects; the SDFG itself only ever sees the integer.

    ``init_symbols`` (optional) is the parameter list the compiled ``__dace_init`` declares, from
    :func:`dace_fortran.bindings.build_fortran_library.compiled_init_symbols`; ``None`` assumes every free symbol.
    A list that cannot be this SDFG's (read from the header of another build of an SDFG with the same name) raises
    :class:`~dace_fortran.bindings.frozen_signature.SignatureDriftError`, see :func:`block_builders.check_init_symbols`.

    ``acc_residency`` (optional) is an
    :class:`dace_fortran.bindings.acc_transfers.AccTransferPlan` -- the sidecar
    x SDFG-storage crossing from :func:`plan_acc_transfers` -- and makes the
    wrapper stage ICON's device-resident arguments itself: ``UPDATE HOST
    ASYNC(1)`` before the copy-in, ``UPDATE DEVICE ASYNC(1)`` + ``WAIT(1)``
    after the copy-out, ``HOST_DATA USE_DEVICE`` around the SDFG invocation
    (see :func:`block_builders.splice_acc_staging`).  A plan from
    :func:`plan_device_resident_caller` instead hands a GPU-storage SDFG the caller's own device copies: module
    arrays through ``acc_deviceptr`` (no host copy), the companions of arrays of records packed on the device.
    ``None`` (the default)
    falls back to the standalone direction: if ``frozen`` records arguments an
    offload pass moved to the device, the wrapper gets an ``!$ACC DATA`` region
    staging the host caller's buffers up; if it records none, the emitted module
    is byte-identical to today's CPU-only one.

    ``directive`` picks the language the staging is written in: OpenACC (nvfortran, ICON) or OpenMP offload
    (e.g. ROCm amdflang); the plan is the same.

    ``module_allocatables_are_targets=False`` is for a caller whose modules are compiled as they are, without the
    TARGET attribute :func:`build_fortran_library` patches into their deferred arrays: the binding then copies an
    allocatable module array in and out instead of aliasing it, which pointer association would not allow.

    ``specialization`` (optional): the SDFG was specialised to these symbol and scalar values; the wrapper checks
    them before it does any work and declines the call otherwise (:mod:`dace_fortran.bindings.specialization`).
    ``record_symbols_to`` (optional): a file the wrapper appends the integer symbols and scalar inputs of every call
    to, the input of :func:`dace_fortran.bindings.specialization.recorded_invariants`.
    ``record_only`` (with ``record_symbols_to`` and ``specialization``): the wrapper declines every call once it has
    recorded it, so the caller runs its own code -- a sizing run that never executes the general SDFG.
    Unless ``record_only``, a wrapper handing the SDFG members of records apart from them first checks that none of
    them shares storage with another array it takes, and declines the call (or stops) otherwise
    (:mod:`dace_fortran.bindings.storage_guard`).
    ``private_access`` / ``private_imports``: module variables their modules keep private, reached through the
    accessors of the modules' copies, and the ``use`` lines their declarations need (see :func:`assemble_module`).
    """
    from dace_fortran.bindings.acc_transfers import plan_frozen_transfers

    check_init_symbols(frozen, init_symbols, dace_arglist)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    enum_maps = enum_maps or {}
    if acc_residency is None:
        acc_residency = plan_frozen_transfers(frozen)

    blocks = {
        "c_interface": build_c_interface(frozen, iface, dace_arglist, init_symbols),
        "handle_state": build_handle_state(iface),
        "wrapper_head": build_wrapper_head(
            frozen,
            iface,
            plan,
            enum_maps=enum_maps,
            module_allocatables_are_targets=module_allocatables_are_targets,
            acc_residency=acc_residency,
            declinable=specialization is not None,
        ),
        "wrapper_body": build_wrapper_body(
            frozen,
            iface,
            plan,
            enum_maps=enum_maps,
            module_allocatables_are_targets=module_allocatables_are_targets,
            acc_residency=acc_residency,
            declinable=specialization is not None,
        ),
        "wrapper_tail": build_wrapper_tail(
            frozen,
            iface,
            plan,
            dace_arglist,
            enum_maps=enum_maps,
            init_symbols=init_symbols,
            module_allocatables_are_targets=module_allocatables_are_targets,
            acc_residency=acc_residency,
        ),
        "finalize": build_finalize(iface),
    }
    blocks = splice_acc_staging(blocks, iface.entry, acc_residency, directive)
    # A sizing run declines every call once it has recorded it: the SDFG never runs, whatever the members share.
    storage = [] if record_only else build_storage_guard(frozen, iface, plan, declinable=specialization is not None)
    if storage:
        blocks["wrapper_body"] = "\n".join([*storage, "", blocks["wrapper_body"]])
        blocks["helpers"] = STORAGE_GUARD_HELPERS
    if specialization is not None:
        guard = build_specialization_guard(specialization, iface, plan)
        blocks["wrapper_body"] = "\n".join([*guard, "", blocks["wrapper_body"]])
    if record_only and (record_symbols_to is None or specialization is None):
        raise ValueError("record_only needs record_symbols_to and a specialization (the wrapper declines through it)")
    if record_symbols_to is not None:
        record = render_symbol_record(frozen, iface, plan, record_symbols_to)
        if record_only:
            record += decline_after_record()
        blocks["wrapper_body"] = "\n".join([blocks["wrapper_body"], "", *record])
    extra = guard_module_sources(specialization, iface, plan) if specialization is not None else None
    out_file.write_text(
        assemble_module(
            iface,
            frozen,
            blocks,
            plan,
            acc_residency,
            extra_sources=extra,
            private_access=private_access,
            private_imports=private_imports,
        )
    )
    return out_file
