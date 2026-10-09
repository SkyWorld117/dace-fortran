# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Give each loop its own copy of a transient array that is local to every iteration of it.

``LoopToMap`` privatises a transient only when no state outside the loop touches it.  A Fortran work array
(``real(8) :: w(0:2)``) is typically shared by several loop nests of one routine -- one per direction, per scheme --
so each nest sees the others' accesses, treats the array as shared and stays sequential, although no value ever flows
from one nest to another.

:func:`privatize_loop_local_arrays` renames such an array per outermost loop when, in every loop using it, each read
is covered by a write earlier in the same iteration: then the loops cannot exchange values through it, and each copy
is local to its loop for ``LoopToMap`` to privatise.  Scalars are covered too: ``ScalarFission`` splits a scalar only
where one write dominates its reads and leaves it whole otherwise.  The check is conservative -- a write counts once
every path through the iteration makes it and names constant elements, a read only if it names constant elements
already written -- and an array it cannot decide is left as it is.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from math import prod

from dace import SDFG, subsets
from dace.sdfg import nodes
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.utils import dfs_topological_sort

#: Element sets larger than this are not enumerated: such an access is treated as undecidable.
_MAX_ELEMENTS = 4096


def privatize_loop_local_arrays(sdfg: SDFG) -> dict[str, list[str]]:
    """Rename every transient array that is local to each iteration of the loops using it, one copy per outermost
    loop, in ``sdfg`` and its nested SDFGs.  Returns ``{original name: [names of the new copies]}``."""
    renamed: dict[str, list[str]] = {}
    for sd in sdfg.all_sdfgs_recursive():
        renamed.update(_privatize(sd))
    return renamed


def _privatize(sdfg: SDFG) -> dict[str, list[str]]:
    users: dict[str, dict[LoopRegion, list[tuple[SDFGState, nodes.AccessNode]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    outside: set[str] = set()
    for state in sdfg.states():
        loop = _outermost_loop(state, sdfg)
        for node in state.data_nodes():
            if not sdfg.arrays[node.data].transient:
                continue
            if loop is None:
                outside.add(node.data)
            else:
                users[node.data][loop].append((state, node))
    interstate_reads = {str(s) for e in sdfg.all_interstate_edges() for s in e.data.read_symbols()}

    renamed: dict[str, list[str]] = {}
    for name, loops in users.items():
        if name in outside or name in interstate_reads or len(loops) < 2:
            continue
        if not all(_iteration_local(name, loop, accesses) for loop, accesses in loops.items()):
            continue
        copies = []
        for loop in list(loops)[1:]:
            new_name = sdfg.add_datadesc(name, sdfg.arrays[name].clone(), find_new_name=True)
            for state in loop.all_states():
                for node in state.data_nodes():
                    if node.data == name:
                        node.data = new_name
                for edge in state.edges():
                    if edge.data.data == name:
                        edge.data.data = new_name
            copies.append(new_name)
        renamed[name] = copies
    return renamed


def _outermost_loop(block: ControlFlowBlock, top: SDFG) -> LoopRegion | None:
    found, region = None, block.parent_graph
    while region is not None and region is not top:
        if isinstance(region, LoopRegion):
            found = region
        region = region.parent_graph
    return found


def _loop_chain(block: ControlFlowBlock, stop: LoopRegion) -> list[LoopRegion]:
    """The loop regions enclosing ``block``, innermost first, up to and including ``stop``."""
    chain, region = [], block.parent_graph
    while region is not None:
        if isinstance(region, LoopRegion):
            chain.append(region)
        if region is stop:
            break
        region = region.parent_graph
    return chain


def _elements(subset: subsets.Subset | None) -> set[tuple[int, ...]] | None:
    """The elements ``subset`` names, when its bounds are constants; ``None`` otherwise."""
    if subset is None:
        return None
    ranges = []
    try:
        for begin, end, step in subset.ndrange():
            ranges.append(range(int(begin), int(end) + 1, int(step)))
    except TypeError:
        return None
    if prod(len(r) for r in ranges) > _MAX_ELEMENTS:
        return None
    return set(product(*ranges))


def _iteration_local(name: str, loop: LoopRegion, accesses) -> bool:
    """Is every read of ``name`` in ``loop`` covered by a write earlier in the same iteration of the innermost loop
    that holds all its accesses?"""
    chains = [_loop_chain(state, loop) for state, _node in accesses]
    common = set(chains[0])
    for chain in chains[1:]:
        common &= set(chain)
    body = next(region for region in chains[0] if region in common)  # innermost loop holding every access
    return _written_after(body, name, set()) is not None


def _written_after(region: ControlFlowRegion, name: str, written: set) -> set | None:
    """The elements of ``name`` certainly written once ``region`` has run, given those written before it; ``None``
    when some read in it is not covered.  A conditional's branches each start from ``written``, and only what every
    path writes counts after it (all of them with an ``else``, otherwise none beyond ``written``); an inner loop's
    reads must be covered, its writes do not count after it, as it may run no iteration."""
    if any(not e.data.is_unconditional() for e in region.edges()):
        return None  # branches as conditional edges: which blocks run is not plain to see
    for block in cfg_analysis.blockorder_topological_sort(region, recursive=False, ignore_nonstate_blocks=False):
        if isinstance(block, SDFGState):
            if not _state_covered(block, name, written):
                return None
        elif isinstance(block, ConditionalBlock):
            after = []
            for _condition, branch in block.branches:
                branch_written = _written_after(branch, name, set(written))
                if branch_written is None:
                    return None
                after.append(branch_written)
            if any(condition is None for condition, _branch in block.branches):
                written = set.intersection(*after)
        elif isinstance(block, LoopRegion):
            if _written_after(block, name, set(written)) is None:
                return None
        elif isinstance(block, ControlFlowRegion):
            inner = _written_after(block, name, written)
            if inner is None:
                return None
            written = inner
        else:
            return None  # a block kind the walk does not know
    return written


def _state_covered(state: SDFGState, name: str, written: set) -> bool:
    """Walk ``state``'s accesses to ``name`` in dataflow order: adds the elements it certainly writes to ``written``,
    and says whether each read is covered."""
    for node in dfs_topological_sort(state):
        if not isinstance(node, nodes.AccessNode) or node.data != name:
            continue
        for edge in state.in_edges(node):
            if edge.data.is_empty():
                continue
            elems = _elements(edge.data.get_dst_subset(edge, state))
            if edge.data.wcr is not None and (elems is None or not elems <= written):
                return False  # an accumulation reads the element it updates
            # A nested SDFG's memlet may cover more than it writes: only a direct write is certain.
            if not edge.data.dynamic and not isinstance(edge.src, nodes.NestedSDFG) and elems is not None:
                written |= elems
        for edge in state.out_edges(node):
            if edge.data.is_empty():
                continue
            elems = _elements(edge.data.get_src_subset(edge, state))
            if elems is None or not elems <= written:
                return False
    return True
