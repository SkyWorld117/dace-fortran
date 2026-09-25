# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turning an SDFG into a compiled library: what to CONSTANT-FOLD, and what to stamp.

NOT `dace_fortran.build`, WHICH ALREADY EXISTS AND DOES SOMETHING ELSE.  That module builds an SDFG
*from source* through flang and HLFIR (`build_sdfg`, `build_sdfg_from_files`).  This one starts where
that finishes: an SDFG in hand, a set of free symbols to either fold into the kernel or leave as
runtime arguments, and a closure of module sources the unit is compiled against.

WHAT IS HERE AND WHAT IS NOT.  The mechanisms are here; the VALUES are not, and they arrive as
arguments.  Which symbols to keep at runtime is delivered as ``keep_runtime`` rather than being a list
in this file, because it names things only a consumer knows -- a project's ghost-offset symbols and
plane offsets are subscript arithmetic, and folding one to 0 does not mis-size a loop, it copies the
wrong cells with no error at all.

``closure_hash`` EXISTS HERE BECAUSE IT EXISTED TWICE.  A consumer's staleness checker had its own
copy of the digest with a comment saying it MUST match this one -- which is a correctness requirement
(a drift makes every library read as STALE) resting on a docstring.  Two implementations of one
digest are one implementation too many.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

__all__ = [
    "bake_scalars",
    "bake_split",
    "closure_files",
    "closure_hash",
    "kernel_only",
]

#: A runtime EXTENT, in the spelling DaCe writes for one: `<name>_d<k>`.
_EXTENT_RE = re.compile(r"^(offset_)?[A-Za-z_]\w*_d\d+$")


def closure_files(directory) -> list:
    """The compile closure as FILES, sorted -- `*.f90` directly under ``directory``.

    A unit built this way is not self-contained: the builder resolves the USE graph from these files
    at compile time.  That is what makes a closure regeneration FREE -- it rewrites no unit, so it
    forces no re-bake -- which embedding the closure in every unit destroys.
    """
    return sorted(Path(directory).glob("*.f90"))


def closure_hash(directory) -> str:
    """A stable digest of a compile closure, for stamping what a library was baked against.

    WHAT THIS ANSWERS, and it is the question nothing else can.  A "was this baked for THAT?"
    marker is needed because a mismatch is SILENT: a library whose hardcoded offsets assume one
    configuration looks fine in another until the kernel reads the wrong cells.  The closure has the
    same hole and it is worse, because there are two independent ways into it -- the closure on disk
    moves while a library keeps an old bake, or one family is regenerated while its siblings are not.
    Recorded per library, the answer is a comparison rather than a recollection.

    The NAME is part of the digest as well as the bytes, so adding or removing a closure module
    changes it.  Truncated to 16 hex characters: this is a stamp, not a security boundary.
    """
    h = hashlib.sha256()
    for f in closure_files(directory):
        h.update(f.name.encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def kernel_only(text: str, module_patterns: Sequence[str]) -> str:
    """Everything from the first KERNEL module on -- the closure prefix dropped.

    ``module_patterns`` are regexes the kernel module names must match, because they are a project's
    own naming convention and not derivable from the text.  Passing them rather than listing them is
    the same rule as everywhere else in this package: the mechanism is general, the names are not.

    A generator that composes closure+prologue+kernel can be split here rather than have its
    internals changed in the same pass; generators that emit kernel-only directly do not need this.
    """
    alt = "|".join(module_patterns)
    pat = re.compile(rf"^module ({alt})\s*$")
    lines = text.split("\n")
    first = next((i for i, l in enumerate(lines) if pat.match(l)), None)
    if first is None:
        raise SystemExit(
            f"kernel_only: no module matching {module_patterns!r} in the composed unit")
    return "\n".join(lines[first:])


def bake_scalars(sdfg, *, values: Sequence[Mapping] = (), keep_runtime: Iterable[str] = (),
                 extra: Sequence[Mapping] = ()) -> Dict[str, object]:
    """Bake EVERY free symbol that is neither an EXTENT nor a RUNTIME-ONLY one.

    ``values`` are consulted in order and the first one naming a symbol wins; a value of ``None`` is a
    SENTINEL meaning "keep this one at runtime".  Anything named in ``keep_runtime`` is left alone, as
    is any symbol used as a LOOP BOUND -- which is DERIVED FROM THE GRAPH rather than listed, because
    a list only approximates it.  Baking a loop bound does not merely mis-size the loop: an index that
    subtracts from it becomes an out-of-bounds memlet and the build fails, and a symbol that only
    appears in index arithmetic instead yields a kernel that reads the wrong cells SILENTLY.

    Whatever is left is folded to a dtype-appropriate zero: module-closure state that the baked-off
    configuration never reads.
    """
    scalars: Dict[str, object] = {}
    loop_bound_syms = set()
    for st in sdfg.all_states():
        for node in st.nodes():
            rng = getattr(getattr(node, "map", None), "range", None)
            if rng is None:
                continue
            for r in rng.ranges:
                for part in r:
                    try:
                        loop_bound_syms |= {str(x) for x in part.free_symbols}
                    except AttributeError:
                        pass
    keep = set(keep_runtime)
    for sym in sdfg.free_symbols:
        s = str(sym)
        if _EXTENT_RE.match(s):
            continue
        if s in loop_bound_syms:
            continue
        for src in tuple(values) + tuple(extra):
            if s in src:
                if src[s] is None:
                    break                       # None sentinel: keep it at runtime
                scalars[s] = src[s]
                break
        else:
            if s in keep:
                continue
            try:
                dt = sdfg.symbols[s]
            except (KeyError, TypeError):
                dt = None
            if dt is None:
                scalars[s] = 0
            else:
                try:
                    base = dt.base_type if hasattr(dt, "base_type") else dt
                    if "int" in str(base) or "bool" in str(base):
                        scalars[s] = 0
                    else:
                        scalars[s] = 0.0
                except Exception:
                    scalars[s] = 0
    return scalars


def bake_split(sdfg, *, values: Sequence[Mapping] = (), keep_runtime: Iterable[str] = (),
               extra: Sequence[Mapping] = ()) -> Tuple[Dict, Dict]:
    """`bake_scalars` split into FREE SYMBOLS vs scalar data CONTAINERS.

    A DA-SDFG's proxy API takes those as two dicts, and the difference is not cosmetic: a symbol is a
    value folded into the kernel, while a `Scalar` container is a one-element buffer that stays an
    argument.  Choosing wrong for a name that exists as both is how a baked value turns into an
    untouched memory location.
    """
    from dace import data as _data

    syms, conts = {}, {}
    raw = bake_scalars(sdfg, values=values, keep_runtime=keep_runtime, extra=extra)
    for k, v in raw.items():
        if isinstance(sdfg.arrays.get(k), _data.Scalar):
            conts[k] = v
        else:
            syms[k] = v
    return syms, conts
