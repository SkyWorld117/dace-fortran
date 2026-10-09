# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Splitting an array along a dimension that every access subscripts with a constant.

An array ``a`` read and written only as ``a[k, ...]`` with constant ``k`` holds independent arrays ``a_k[...]``, one
per index.  Replacing it by them lets a caller hand each one over on its own -- the members of an array of records,
for instance, which live apart in memory and would otherwise have to be packed into one buffer on every call.
"""

from __future__ import annotations

import copy
import math

import dace
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


def _without(rng: subsets.Range, dim: int) -> subsets.Range:
    return subsets.Range([r for i, r in enumerate(rng) if i != dim])


def split_array(sdfg: dace.SDFG, name: str, dim: int = 0) -> dict[int, str] | None:
    """Replace the array ``name`` by one array per index of dimension ``dim`` its accesses use, named
    ``<name>_<index>``, and return them by index; None (and ``sdfg`` unchanged) unless :func:`constant_indices` finds
    every index a literal.

    Each new array has the remaining dimensions of ``name``, laid out column-major on their own: a caller hands it
    over as the contiguous array it is, not as a slice of the original.
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
    for new_name in names.values():
        part = copy.deepcopy(desc)
        part.set_shape(shape, strides=strides, total_size=math.prod(shape), offset=offset)
        sdfg.add_datadesc(new_name, part)

    for state in sdfg.states():
        # Access nodes: each edge moves to the node of its own index; a copy's subset on the array loses the split
        # dimension here, where it is known which end the array is.
        for node in [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == name]:
            replacements: dict[int, nodes.AccessNode] = {}
            for edge in [*state.in_edges(node), *state.out_edges(node)]:
                index = _constant(_subset_of(edge, state, name)[dim])
                if index not in replacements:
                    replacements[index] = state.add_access(names[index])
                new = replacements[index]
                memlet = edge.data
                if memlet.data != name:
                    memlet.other_subset = _without(memlet.other_subset, dim)
                if edge.dst is node:
                    state.add_edge(edge.src, edge.src_conn, new, edge.dst_conn, memlet)
                else:
                    state.add_edge(new, edge.src_conn, edge.dst, edge.dst_conn, memlet)
                state.remove_edge(edge)
            state.remove_node(node)
        # Memlets naming the array: its index's array, without the split dimension.
        for edge in state.edges():
            memlet = edge.data
            if memlet.data == name:
                memlet.data = names[_constant(memlet.subset[dim])]
                memlet.subset = _without(memlet.subset, dim)
    sdfg.remove_data(name, validate=False)
    return names
