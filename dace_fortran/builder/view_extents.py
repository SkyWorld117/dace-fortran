# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Define the extents of section views from the section each one views.

An inlined callee's dummy bound to a section of an array (``call f(a(:, lo:hi))``) becomes a view of ``a``.  The
bridge sizes the view from the section's extents, and an extent it cannot trace (bounds read from a record the
routine sets just before the call, ``iv%beg = ...; call f(a(:, iv%beg:iv%end))``) is left as the placeholder
symbol ``<view>_d<k>``.  That symbol is free, and nothing outside the SDFG can supply it: its value only exists once
the routine has run up to the call.  The section itself is known -- it is the source subset of the memlet linking
the view to its source in every state that accesses it -- so the placeholder is assigned the element count of the
section's triplet on an interstate edge just before the first block that uses it, where the call takes place.
Reading the section bounds there also makes DaCe keep a scalar bound alive across states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
from dace import data, subsets, symbolic
from dace.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion


def _section_sizes(view_subset: Sequence[str], source_shape: Sequence[object]) -> list[object] | None:
    """Element count of every dimension of the source subset a view links to (``:`` spans the source dimension)."""
    parts = [f"0:{source_shape[d]}" if str(s).strip() == ":" else str(s) for d, s in enumerate(view_subset)]
    try:
        return subsets.Range.from_string(", ".join(parts)).size()
    except (SyntaxError, TypeError, ValueError):
        return None


def _uses(block: ControlFlowBlock, sym: str) -> bool:
    return sym in {str(s) for s in block.used_symbols(all_symbols=True, with_contents=True)}


def _header_uses(block: ControlFlowBlock, sym: str) -> bool:
    """Whether a region's own header (loop bounds, branch conditions) uses ``sym``, its contents aside."""
    if isinstance(block, ConditionalBlock):
        return any(cond is not None and sym in {str(s) for s in cond.get_free_symbols()} for cond, _ in block.branches)
    return sym in {str(s) for s in block.used_symbols(all_symbols=True, with_contents=False)}


def _insertion_point(region: ControlFlowRegion, sym: str) -> tuple[ControlFlowRegion, ControlFlowBlock] | None:
    """The region and block before which ``sym`` must be defined: descend while one block holds every use."""
    while True:
        users = [b for b in region.nodes() if _uses(b, sym)]
        for e in region.edges():
            if sym in {str(s) for s in e.data.free_symbols} and e.src not in users:
                users.append(e.src)
        if not users:
            return None
        if len(users) == 1 and not _header_uses(users[0], sym):
            block = users[0]
            if isinstance(block, ConditionalBlock):
                inner = [b for _, b in block.branches if _uses(b, sym)]
                if len(inner) == 1:
                    region = inner[0]
                    continue
                return region, block
            if isinstance(block, ControlFlowRegion):
                region = block
                continue
            return region, block
        try:
            order = list(nx.topological_sort(region.nx))
        except nx.NetworkXUnfeasible:
            return None
        first = min(users, key=order.index)
        # The definition must precede every use: refuse a first user that does not reach the others.
        reach = nx.descendants(region.nx, first)
        if any(u is not first and u not in reach for u in users):
            return None
        return region, first


def define_section_view_extents(sdfg: SDFG, views: Mapping[str, tuple[str, Sequence[str]]]) -> dict[str, str]:
    """Assign every placeholder extent ``<view>_d<k>`` of the section views ``views`` before its first use.

    ``views`` maps a view to its source and the source-side subset of the linking memlet (one entry per source
    dimension: a scalar index for a dimension the section drops, ``:`` for a whole one, else ``start:stop[:step]``).
    Returns the placeholders defined, with the expression each one is assigned.
    """
    defined: dict[str, str] = {}
    for name, (source, view_subset) in views.items():
        desc = sdfg.arrays.get(name)
        src = sdfg.arrays.get(source)
        if not isinstance(desc, data.View) or src is None:
            continue
        triplets = [d for d, s in enumerate(view_subset) if ":" in str(s)]
        if len(triplets) != len(desc.shape) or len(view_subset) != len(src.shape):
            continue
        placeholders = {k: f"{name}_d{k}" for k, s in enumerate(desc.shape) if str(s) == f"{name}_d{k}"}
        if not placeholders:
            continue
        sizes = _section_sizes(view_subset, src.shape)
        if sizes is None:
            continue
        for k, sym in placeholders.items():
            extent = sizes[triplets[k]]
            if any(str(s) in placeholders.values() for s in symbolic.symlist(extent)):
                continue
            where = _insertion_point(sdfg, sym)
            if where is None:
                continue
            region, block = where
            value = symbolic.symstr(extent, cpp_mode=False)
            region.add_state_before(
                block,
                label=f"extent_{sym}",
                is_start_block=region.start_block is block,
                assignments={sym: value},
            )
            defined[sym] = value
    return defined
