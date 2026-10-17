"""Unroll the small maps inside kernel bodies, so that a body becomes straight-line code over constant subscripts.

A kernel body (the nested SDFG of a top-level map, or the scope of a map nested in another one) often holds maps over
a handful of elements: an array statement ``alpha(:) = 0`` or ``omega = alpha / s`` over the weights of a stencil.
DaCe's ``MapUnroll`` only takes top-level maps of an SDFG and replicates the whole connected component around them; a
body needs the map's own scope replicated in place. Once unrolled, every access subscripts the small arrays with a
constant, which is what splitting them into scalars (scalarise.py) and fusing the body's states then need.

``unroll_small_maps`` unrolls every map with constant ranges of at most ``limit`` iterations that is not a top-level
map of the outermost SDFG (those are the kernels), whose scope holds only tasklets and access nodes of containers no
node outside the scope touches (each copy gets its own container).
"""
from __future__ import annotations

import copy
import itertools
import re

import dace
from dace import nodes, symbolic
from dace.properties import CodeBlock


def _constant_ranges(entry: nodes.MapEntry) -> list[range] | None:
    out = []
    for begin, end, step in entry.map.range:
        try:
            b, e, s = (int(symbolic.evaluate(x, {})) for x in (begin, end, step))
        except (TypeError, ValueError, KeyError):
            return None
        out.append(range(b, e + 1, s))
    return out


def _local_containers(sdfg: dace.SDFG, state: dace.SDFGState, inner: list) -> set[str] | None:
    """The containers of the access nodes in ``inner``, or None unless each is a transient no node outside the scope
    touches."""
    names = {n.data for n in inner if isinstance(n, nodes.AccessNode)}
    inner_set = set(inner)
    for name in names:
        if not sdfg.arrays[name].transient:
            return None
    for st in sdfg.states():
        for n in st.data_nodes():
            if n.data in names and n not in inner_set:
                return None
    for e in sdfg.all_interstate_edges():
        if names & {str(s) for s in e.data.free_symbols}:
            return None
    return names


def _substituted(memlet: dace.Memlet, values: dict) -> dace.Memlet:
    m = copy.deepcopy(memlet)
    repl = {symbolic.symbol(p): v for p, v in values.items()}
    for subset in (m.subset, m.other_subset):
        if subset is not None:
            subset.replace(repl)
    return m


def _unroll(sdfg: dace.SDFG, state: dace.SDFGState, entry: nodes.MapEntry, ranges: list[range],
            local: set[str]) -> None:
    exit_ = state.exit_node(entry)
    inner = [n for n in state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()]
    outer_scope = state.scope_dict()[entry]
    outer_exit = state.exit_node(outer_scope) if outer_scope is not None else None
    ins = {e.dst_conn: e for e in state.in_edges(entry) if e.dst_conn}
    outs = {e.src_conn: e for e in state.out_edges(exit_) if e.src_conn}
    before = [e for e in state.in_edges(entry) if e.data.is_empty()]
    after = [e for e in state.out_edges(exit_) if e.data.is_empty()]
    params = entry.map.params
    for values in itertools.product(*ranges):
        vals = dict(zip(params, values))
        renamed = {name: sdfg.add_datadesc(f"{name}_u{'_'.join(map(str, values))}".replace("-", "m"),
                                           copy.deepcopy(sdfg.arrays[name]), find_new_name=True) for name in local}
        copies = {}
        for n in inner:
            if isinstance(n, nodes.Tasklet):
                new = copy.deepcopy(n)
                code = new.code.as_string
                for p, v in vals.items():
                    code = re.sub(rf"\b{re.escape(p)}\b", f"({v})", code)
                new.code = CodeBlock(code, new.code.language)
            else:
                new = nodes.AccessNode(renamed.get(n.data, n.data))
            state.add_node(new)
            copies[n] = new

        def memlet_of(e):
            m = _substituted(e.data, vals)
            if m.data in renamed:
                m.data = renamed[m.data]
            return m

        for n in inner:
            for e in state.out_edges(n):
                if e.dst in copies:
                    state.add_edge(copies[n], e.src_conn, copies[e.dst], e.dst_conn, memlet_of(e))
        for e in state.out_edges(entry):
            dst = copies[e.dst]
            if e.data.is_empty():
                for b in before:
                    state.add_edge(b.src, b.src_conn, dst, None, dace.Memlet())
                if outer_scope is not None and not before:
                    state.add_nedge(outer_scope, dst, dace.Memlet())
                continue
            outer = ins["IN_" + e.src_conn[len("OUT_"):]]
            state.add_edge(outer.src, outer.src_conn, dst, e.dst_conn, memlet_of(e))
        for e in state.in_edges(exit_):
            src = copies[e.src]
            if e.data.is_empty():
                for a in after:
                    state.add_edge(src, None, a.dst, a.dst_conn, dace.Memlet())
                if outer_exit is not None and not after:
                    state.add_nedge(src, outer_exit, dace.Memlet())
                continue
            outer = outs["OUT_" + e.dst_conn[len("IN_"):]]
            state.add_edge(src, e.src_conn, outer.dst, outer.dst_conn, memlet_of(e))
    state.remove_nodes_from([entry, exit_, *inner])
    for name in local:
        sdfg.remove_data(name, validate=False)


def unroll_small_maps(sdfg: dace.SDFG, limit: int = 8) -> int:
    """See the module docstring. Returns the number of maps unrolled."""
    count = 0
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            while True:
                scopes = state.scope_dict()
                found = None
                for entry in state.nodes():
                    if not isinstance(entry, nodes.MapEntry):
                        continue
                    if sd is sdfg and scopes[entry] is None:
                        continue  # a kernel
                    if entry.map.schedule in dace.dtypes.GPU_SCHEDULES:
                        continue
                    ranges = _constant_ranges(entry)
                    if ranges is None or len(list(itertools.product(*ranges))) > limit:
                        continue
                    inner = state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()
                    if not all(isinstance(n, (nodes.Tasklet, nodes.AccessNode)) for n in inner):
                        continue
                    local = _local_containers(sd, state, inner)
                    if local is None:
                        continue
                    found = (entry, ranges, local)
                    break
                if found is None:
                    break
                _unroll(sd, state, *found)
                count += 1
    return count
