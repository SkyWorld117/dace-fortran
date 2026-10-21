# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Splitting an array along a dimension that every access subscripts with a constant.

An array ``a`` read and written only as ``a[k, ...]`` with constant ``k`` holds independent arrays ``a_k[...]``, one
per index.  Replacing it by them lets a caller hand each one over on its own -- the members of an array of records,
for instance, which live apart in memory and would otherwise have to be packed into one buffer on every call -- and
lets a kernel keep the elements of a small scratch array in scalars of their own.

An access node orders what passes through it: every read from it follows every write into it, whichever elements
they touch.  Once its elements are nodes of their own, a read of one element no longer follows the writes of the
others, and a later read of an element can lose the writes that reached it only through the shared node.  The split
keeps that order with empty edges (see :func:`_keep_order`).
"""

from __future__ import annotations

import copy
import math

import dace
import networkx as nx
import sympy
from dace import subsets
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion

#: An edge that touches the array without telling which part of it.
_UNKNOWN = object()


def _subset_of(edge, state: dace.SDFGState, name: str):
    """The subset of ``name`` an edge reads or writes, None if the edge does not touch it, :data:`_UNKNOWN` if it does
    but the memlet does not say where."""
    if isinstance(edge.src, nodes.AccessNode) and edge.src.data == name:
        subset = edge.data.get_src_subset(edge, state)
    elif isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == name:
        subset = edge.data.get_dst_subset(edge, state)
    elif edge.data.data == name:
        subset = edge.data.subset
    else:
        return None
    return _UNKNOWN if subset is None else subset


def _constant(rng) -> int | None:
    """The index of a single-element range with a literal bound, or None."""
    begin, end, _step = rng
    if begin == end and isinstance(sympy.sympify(begin), sympy.Integer):
        return int(begin)
    return None


def constant_indices(sdfg: dace.SDFG, name: str, dim: int = 0) -> set[int] | None:
    """The indices of dimension ``dim`` the accesses of the array ``name`` use, or None unless each of them is one
    literal.

    Only the outermost SDFG is looked at: an array a nested SDFG takes, or one an interstate edge or a loop or branch
    condition reads, does not qualify.
    """
    if not isinstance(sdfg.arrays.get(name), dace.data.Array):
        return None
    indices: set[int] = set()
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG) and any(
                e.data.data == name for e in [*state.in_edges(node), *state.out_edges(node)]
            ):
                return None
        for edge in state.edges():
            if edge.data.is_empty():
                continue  # an ordering edge: it touches no element
            subset = _subset_of(edge, state, name)
            if subset is None:
                continue
            index = None if subset is _UNKNOWN else _constant(subset[dim])
            if index is None:
                return None
            indices.add(index)
    for edge in sdfg.all_interstate_edges():
        if any(m.data == name for m in edge.data.get_read_memlets(sdfg.arrays)):
            return None
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, AbstractControlFlowRegion) and any(m.data == name for m in block.get_meta_read_memlets()):
            return None
    return indices


def _without(rng: subsets.Range, dim: int, scalar: bool = False) -> subsets.Range:
    if scalar:
        return subsets.Range([(0, 0, 1)])
    return subsets.Range([r for i, r in enumerate(rng) if i != dim])


def _written(state: dace.SDFGState, node: nodes.AccessNode) -> bool:
    return any(not e.data.is_empty() for e in state.in_edges(node))


def _keep_order(state: dace.SDFGState, parts: list[nodes.AccessNode]) -> None:
    """Restore the order one access node gave the accesses now split over ``parts``: every read from the node
    followed every write into it.  An empty edge goes from each part something writes into to each other part only
    read there (so that analyses which look for a write reaching a read node still find one through it), and to the
    readers of each other part written there too (an edge between two written parts could close a cycle), unless a
    path already orders the two."""
    for part in parts:
        if not _written(state, part):
            continue
        for other in parts:
            if other is part:
                continue
            if not _written(state, other):
                if not nx.has_path(state.nx, part, other):
                    state.add_nedge(part, other, dace.Memlet())
                continue
            for edge in state.out_edges(other):
                if edge.data.is_empty() or nx.has_path(state.nx, part, edge.dst):
                    continue
                state.add_nedge(part, edge.dst, dace.Memlet())


def split_array(sdfg: dace.SDFG, name: str, dim: int = 0) -> dict[int, str] | None:
    """Replace the array ``name`` by one array per index of dimension ``dim`` its accesses use, named
    ``<name>_<index>``, and return them by index; None (and ``sdfg`` unchanged) unless :func:`constant_indices` finds
    every index a literal.

    Each new array has the remaining dimensions of ``name``, laid out column-major on their own: a caller hands it
    over as the contiguous array it is, not as a slice of the original.  A transient with no dimension left becomes a
    scalar.  Edges that carry no data (they only order their ends) may touch the array's nodes: each part of a node
    keeps them.
    """
    indices = constant_indices(sdfg, name, dim)
    if not indices:
        return None
    names = {k: f"{name}_{k}" for k in sorted(indices)}
    if any(n in sdfg.arrays or n in sdfg.symbols for n in names.values()):
        return None
    desc = sdfg.arrays[name]
    shape = [s for i, s in enumerate(desc.shape) if i != dim]
    strides = [math.prod(shape[:i]) for i in range(len(shape))]
    offset = [o for i, o in enumerate(desc.offset) if i != dim]
    scalar = not shape and desc.transient
    for new_name in names.values():
        if scalar:
            part = dace.data.Scalar(
                desc.dtype, transient=True, storage=desc.storage, lifetime=desc.lifetime, debuginfo=desc.debuginfo
            )
        else:
            part = copy.deepcopy(desc)
            part.set_shape(shape, strides=strides, total_size=math.prod(shape), offset=offset)
        sdfg.add_datadesc(new_name, part)

    for state in sdfg.states():
        # Access nodes: each edge moves to the node of its own index; a copy's subset on the array loses the split
        # dimension here, where it is known which end the array is.
        for node in [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == name]:
            replacements: dict[int, nodes.AccessNode] = {}
            ordering = []
            for edge in [*state.in_edges(node), *state.out_edges(node)]:
                if edge.data.is_empty():
                    ordering.append(edge)
                    continue
                index = _constant(_subset_of(edge, state, name)[dim])
                if index not in replacements:
                    replacements[index] = state.add_access(names[index])
                new = replacements[index]
                memlet = edge.data
                if memlet.data != name:
                    memlet.other_subset = _without(memlet.other_subset, dim, scalar)
                if edge.dst is node:
                    state.add_edge(edge.src, edge.src_conn, new, edge.dst_conn, memlet)
                else:
                    state.add_edge(new, edge.src_conn, edge.dst, edge.dst_conn, memlet)
                state.remove_edge(edge)
            parts = list(replacements.values())
            # What the node was ordered after (before), each part is; a node of ordering edges only passes the order
            # on from its predecessors to its successors.
            before = [e for e in ordering if e.dst is node]
            after = [e for e in ordering if e.src is node]
            for b in before:
                for target, conn in [(part, None) for part in parts] or [(a.dst, a.dst_conn) for a in after]:
                    state.add_edge(b.src, b.src_conn, target, conn, dace.Memlet())
            for a in after:
                for part in parts:
                    state.add_edge(part, None, a.dst, a.dst_conn, dace.Memlet())
            state.remove_node(node)
            _keep_order(state, parts)
        # Memlets naming the array: its index's array, without the split dimension.
        for edge in state.edges():
            memlet = edge.data
            if memlet.data == name:
                memlet.data = names[_constant(memlet.subset[dim])]
                memlet.subset = _without(memlet.subset, dim, scalar)
    sdfg.remove_data(name, validate=False)
    return names
