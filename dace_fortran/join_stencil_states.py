# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fuse a stencil's producer and consumer states that only symbol assignments keep apart.

A producer map writes a transient, a later state's map reads it, and between them a source sets a symbol (a loop
bound that is only known at run time -- a constant would have been propagated). ``LoopToMap``/map fusion cannot see
across that state boundary, so the pair stays two kernels. This pass moves the assignments past the consumer when
that is sound, fuses the two states, and lets the existing fusion passes see the pair.

Lifted from ``dace_fortran/substitute/fuse_stencils.py`` on ``mfc-substitute-20261021`` (commits ``6f7fd538``,
``079c57c9``, ``f64a4035``) so the existing region/shim pipeline can use it without importing the substitute tree.
"""

from __future__ import annotations

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.transformation.interstate import StateFusion


def _top_maps(state: dace.SDFGState) -> list[nodes.MapEntry]:
    scopes = state.scope_dict()
    return [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scopes[n] is None]


def _movable_past(edge, sdfg: dace.SDFG) -> bool:
    """Whether ``edge``'s symbol assignments can move onto the edges out of its destination state: the edge is
    unconditional and the state's only way in (in the state each symbol then holds its assigned value: a state changes
    no symbol), the state has a way out, and the values read no data (which the state may change) and none of the
    symbols the edge assigns (whose old values they read)."""
    state, graph = edge.dst, edge.dst.parent_graph
    assigned = set(edge.data.assignments)
    if not assigned or not edge.data.is_unconditional() or not isinstance(state, dace.SDFGState):
        return False
    if graph.in_degree(state) != 1 or not graph.out_degree(state):
        return False
    reads = {str(x) for v in edge.data.assignments.values() for x in symbolic.free_symbols_and_functions(v)}
    return not reads & (set(sdfg.arrays) | assigned)


def join_stencil_states(sdfg: dace.SDFG) -> int:
    """Fuse each state holding a stencil's consumer into its producer's state where only symbol assignments keep them
    apart (loop bounds of later code a source sets between the two, say: a value known only at run time stays an
    assignment, a constant would have been propagated): the values replace the symbols in the consumer's state and in
    the conditions and values of the edges out of it, the assignments move onto those edges (one that assigns a symbol
    again keeps its own value), the two states fuse, and the stencil fusion sees the pair. Returns the number of
    states joined."""
    joined = 0
    for state in list(sdfg.states()):
        graph = state.parent_graph
        tops = set(_top_maps(state))
        written = {n.data for n in state.data_nodes() if sdfg.arrays[n.data].transient and state.in_degree(n) == 1
                   and isinstance(state.in_edges(n)[0].src, nodes.MapExit)
                   and state.entry_node(state.in_edges(n)[0].src) in tops}
        if not written or graph.out_degree(state) != 1:
            continue
        edge = graph.out_edges(state)[0]
        after = edge.dst
        if not isinstance(after, dace.SDFGState) or not any(n.data in written and after.out_degree(n)
                                                            for n in after.data_nodes()):
            continue
        if not _movable_past(edge, sdfg):
            continue  # states nothing is assigned between are the simplification's to fuse
        values = dict(edge.data.assignments)
        after.replace_dict(values)
        for out in graph.out_edges(after):
            out.data.replace_dict(values, replace_keys=False)
            out.data.assignments = {**values, **out.data.assignments}
        edge.data.assignments = {}
        fusion = StateFusion()
        fusion.first_state, fusion.second_state = state, after
        if fusion.can_be_applied(graph, 0, sdfg):
            StateFusion.apply_to(sdfg, first_state=state, second_state=after, verify=False)
            joined += 1
    return joined
