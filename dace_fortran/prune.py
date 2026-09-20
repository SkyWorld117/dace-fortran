# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prune arguments the SDFG never reads.

The bridge materialises a Fortran module's closure -- the derived types, parameters and module
variables a translation unit declares -- and every one of them lands in the SDFG as a NON-TRANSIENT
array, which dace then reports as an argument.  Measured on MFC's sweeps kernel: **305 of 378
arguments are never accessed anywhere in the graph.**

That is not dead dataflow.  `SimplifyPass` already runs `DeadDataflowElimination`, which removes
nodes and edges; an argument that has no access node *at all* survives it untouched, because nothing
in the pass considers the argument list.  So the ABI ends up being the closure's variable set rather
than the kernel's needs, and it grows whenever the closure gains a field.

Why it matters beyond tidiness: the argument list is what every shim and every call site is written
against.  A pruning pass makes the ABI track the kernel, so a closure that gains an unused field stops
moving the interface.

**Why this is written defensively, and the three ways it was wrong first.**  Deciding "is this name
used?" by inspecting the graph is a trap, and each attempt that got it wrong would have produced a
silently wrong kernel or an unusable SDFG:

1. *Data nodes only* selected **`re_idx`** for removal.  A symbol read only in INDEX ARITHMETIC has no
   data node; pruning it makes the kernel read the wrong cells with no error anywhere.
2. *Deferring to `sdfg.used_symbols()`* was closer but still not enough: a symbol appears in the graph
   in more places than a state's nodes and edges -- notably an INTER-STATE edge's condition and
   assignments -- and that scan does not walk `sdfg.edges()`.
3. Even with both, dace's `used_symbols` **excludes names that are registered as arrays**, so an array
   that is also used as an index symbol looks unused until the array is removed and the name
   resurfaces as a free symbol -- at which point `arglist()` raises `KeyError` looking it up in a
   `symbols` table that never had it.

So the pass does not trust its own analysis for the decision that can corrupt a graph.  It asks first
(:func:`unused_arguments`), and then **verifies the one invariant that matters after each removal --
that the signature is still computable** (:func:`prune_unused_arguments`).  A name whose removal
breaks `arglist()` is put back.  That makes the failure mode impossible rather than unlikely, and it
costs one `arglist()` call per candidate at build time.
"""
from __future__ import annotations

from typing import List, Set

from dace import SDFG


def _free_symbols_of(obj) -> Set[str]:
    """Names appearing in an expression-like object, defensively (layouts and slices vary)."""
    out: Set[str] = set()
    for attr in ("free_symbols",):
        fs = getattr(obj, attr, None)
        if fs is None:
            continue
        try:
            out |= {str(s) for s in fs}
        except TypeError:
            pass
    return out


def _used(sdfg: SDFG) -> Set[str]:
    """Every name the graph USES, at any nesting depth.

    "Used" is deliberately wider than "has an access node": a symbol that appears only in index
    arithmetic -- a memlet subset, a map range, a tasklet's code, an inter-state edge's condition --
    has no data node, and pruning it yields a kernel that reads the wrong cells with no error.

    Nested SDFGs are walked explicitly, for the same reason: a closure variable read only inside an
    inlined helper lives in a nested graph.
    """
    names: Set[str] = set()
    for state in sdfg.all_states():
        for node in state.data_nodes():
            names.add(node.data)
        for node in state.nodes():
            names |= _free_symbols_of(node)                     # tasklets: code, and `free_symbols`
            rng = getattr(getattr(node, "map", None), "range", None)
            if rng is not None:
                for r in rng.ranges:
                    for part in r:
                        names |= _free_symbols_of(part)
            inner = getattr(node, "sdfg", None)
            if inner is not None and inner is not sdfg:
                names |= _used(inner)
        for edge in state.edges():
            for attr in ("data", "subset", "other_subset"):
                names |= _free_symbols_of(getattr(edge, attr, None))
    # Inter-state edges carry conditions and assignments -- code that names symbols and is not in any
    # state.  `re_idx` lives here, which is why a state-only scan missed it.
    for edge in sdfg.edges():
        for attr in ("condition", "assignments"):
            names |= _free_symbols_of(getattr(edge, attr, None))
        us = getattr(edge, "used_symbols", None)
        if callable(us):
            try:
                names |= {str(s) for s in us(all_symbols=False)}
            except Exception:  # noqa: BLE001
                pass
    return names


def unused_arguments(sdfg: SDFG) -> List[str]:
    """Argument names in ``sdfg.arglist()`` the graph never uses, sorted.

    A REPORT, not a licence to remove: see the module docstring for why the answer alone is not
    enough.  ``prune_unused_arguments`` treats this as a candidate list and still verifies each
    removal.
    """
    used = _used(sdfg)
    out = []
    for name in sdfg.arglist():
        n = str(name)
        if n in used:
            continue
        if n in sdfg.arrays and sdfg.arrays[n].transient:
            continue
        out.append(n)
    return sorted(out)


def prune_unused_arguments(sdfg: SDFG, *, verify: bool = True) -> int:
    """Drop every unused argument from the signature.  Returns how many were removed.

    Only the ARGUMENT is removed -- the `sdfg.arrays` entry and its argument status -- never an
    access, since by construction a name with no access node has none to remove.

    With ``verify`` (the default) each removal is checked against the one invariant that matters:
    **the signature must still be computable**.  `arglist()` resolves free symbols through
    `sdfg.symbols`, and removing an array can make a name resurface as a free symbol -- so a removal
    that breaks it is undone.  Without this the pass raises `KeyError` on real MFC kernels; the
    analysis cannot see the case, so the pass does not get to assume it away.

    CAVEAT, and it is the reason this is not wired into any pipeline: the argument list is an
    INTERFACE.  Every shim and every call site in the consuming project is written against the current
    arity and order, so pruning a real kernel's signature means regenerating them.  Turn it on as part
    of a re-port, not as a cleanup.
    """
    gone = 0
    for name in unused_arguments(sdfg):
        arr = sdfg.arrays.pop(name, None)
        sym = sdfg.symbols.pop(name, None)
        if verify:
            try:
                sdfg.arglist()
            except Exception:  # noqa: BLE001 -- the removal broke the signature: put it back
                if arr is not None:
                    sdfg.arrays[name] = arr
                if sym is not None:
                    sdfg.symbols[name] = sym
                continue
        gone += 1
    return gone
