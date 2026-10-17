"""Collapse a map with the map its nested body holds, so that a kernel parallelises over every dimension of a loop nest.

Turning an outer loop into a map wraps its body in a nested SDFG when the body had several states; once those fuse
into one state holding nothing but the inner loop's map, the two maps form a perfect nest, but MapCollapse cannot see
through the nested SDFG and InlineSDFG refuses it as soon as the body writes an argument element and reads it back
in the same iteration (an output connector's access node with readers, inside the inner map). On a GPU the result is
one thread per outer iteration, each running the whole inner loop: in a two-dimensional domain, a few thousand threads
for millions of cells.

``collapse_through_bodies`` takes every top-level map of the SDFG's states whose scope is exactly one nested SDFG
holding one state with one top-level map (and its access nodes), whose range does not depend on the outer map's
parameters; it routes each in-scope write-then-read of an argument element through a transient scalar (the readers
take the value just computed, the element is written once from it), inlines the body and collapses the two maps.
"""
from __future__ import annotations

import dace
from dace import nodes, symbolic
from dace.transformation.dataflow import MapCollapse
from dace.transformation.interstate import InlineSDFG


def _forward_in_scope_reads(nsdfg: dace.SDFG, state: dace.SDFGState) -> int:
    """Route every write-then-read of one argument element inside a map scope through a transient scalar. Returns the
    number of access nodes rerouted."""
    count = 0
    for node in list(state.data_nodes()):
        desc = nsdfg.arrays[node.data]
        if desc.transient or state.entry_node(node) is None:
            continue
        ins, outs = state.in_edges(node), state.out_edges(node)
        if len(ins) != 1 or not outs:
            continue
        write = ins[0]
        if write.data.wcr is not None or write.data.subset is None or write.data.subset.num_elements() != 1:
            continue
        if write.data.data != node.data or any(e.data.data != node.data or e.data.subset != write.data.subset
                                               for e in outs):
            continue
        # The element leaves the map through the exit connector the map already writes the container through.
        exit_node = state.exit_node(state.entry_node(node))
        conns = {e.dst_conn for e in state.in_edges(exit_node) if e.data.data == node.data and e.dst_conn}
        if len(conns) != 1:
            continue
        name, _ = nsdfg.add_scalar(f"{node.data}_fwd", desc.dtype, transient=True, find_new_name=True)
        tmp = state.add_access(name)
        state.remove_edge(write)
        state.add_edge(write.src, write.src_conn, tmp, None, dace.Memlet(data=name, subset="0"))
        state.add_edge(tmp, None, exit_node, conns.pop(),
                       dace.Memlet(data=node.data, subset=write.data.subset, other_subset="0"))
        for e in outs:
            state.remove_edge(e)
            state.add_edge(tmp, None, e.dst, e.dst_conn, dace.Memlet(data=name, subset="0"))
        state.remove_node(node)
        count += 1
    return count


def _drop_unread_inputs(state: dace.SDFGState, body: nodes.NestedSDFG) -> None:
    """Remove the input connectors of ``body`` whose data the body no longer reads, with their outer memlet paths."""
    nstate = body.sdfg.nodes()[0]
    read = {n.data for n in nstate.data_nodes() if nstate.out_degree(n) > 0}
    for conn in list(body.in_connectors):
        if conn in read:
            continue
        for e in list(state.in_edges_by_connector(body, conn)):
            state.remove_memlet_path(e)
        body.remove_in_connector(conn)


def _perfect_body(state: dace.SDFGState, entry: nodes.MapEntry):
    """The nested SDFG node that is ``entry``'s whole body, when it holds one state with one top-level map whose range
    does not depend on ``entry``'s parameters; else None."""
    inner = [n for n in state.scope_children()[entry] if not isinstance(n, nodes.MapExit)]
    if len(inner) != 1 or not isinstance(inner[0], nodes.NestedSDFG):
        return None
    body = inner[0]
    if len(body.sdfg.nodes()) != 1 or not isinstance(body.sdfg.nodes()[0], dace.SDFGState):
        return None
    nstate = body.sdfg.nodes()[0]
    top = [n for n in nstate.nodes() if nstate.entry_node(n) is None]
    maps = [n for n in top if isinstance(n, nodes.MapEntry)]
    others = [n for n in top if not isinstance(n, (nodes.MapEntry, nodes.MapExit, nodes.AccessNode))]
    if len(maps) != 1 or others:
        return None
    params = set(entry.map.params)
    inner_free = {str(s) for r in maps[0].map.range for e in r for s in symbolic.pystr_to_symbolic(str(e)).free_symbols}
    if params & inner_free:
        return None
    # The inner range is written in the body's symbols; they must be the outer ones (an identity symbol mapping).
    for s in inner_free:
        if s in body.symbol_mapping and str(body.symbol_mapping[s]) != s:
            return None
    return body


def collapse_through_bodies(sdfg: dace.SDFG) -> int:
    """Collapse every top-level map with the map its nested body holds (see the module docstring). Returns the number
    of map pairs collapsed."""
    collapsed = 0
    for state in sdfg.states():
        for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]:
            body = _perfect_body(state, entry)
            if body is None:
                continue
            nstate = body.sdfg.nodes()[0]
            if _forward_in_scope_reads(body.sdfg, nstate):
                _drop_unread_inputs(state, body)
            if not InlineSDFG.can_be_applied_to(sdfg, nested_sdfg=body):
                continue
            InlineSDFG.apply_to(sdfg, nested_sdfg=body, verify=False)
            inner = [n for n in state.scope_children()[entry] if isinstance(n, nodes.MapEntry)]
            if len(inner) == 1 and MapCollapse.can_be_applied_to(sdfg, outer_map_entry=entry, inner_map_entry=inner[0]):
                MapCollapse.apply_to(sdfg, outer_map_entry=entry, inner_map_entry=inner[0], verify=False)
                collapsed += 1
    return collapsed
