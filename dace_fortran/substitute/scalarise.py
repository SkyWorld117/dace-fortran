"""Split small scratch arrays into scalars, so that the states a reused one keeps apart can fuse.

A kernel body that works on a few-element scratch array (stencil weights, candidate values) overwrites it more than
once: one normalisation writes it, another rewrites it from the first result. After the short loops are unrolled,
every state that touches it conflicts with the next on the same container, so state fusion stops and the body stays
dozens of states long; passes that look for a pattern within one state (the normalisation rewrite in numerics.py,
for instance) then find nothing. Element by element, though, the writes and reads are independent values.

``scalarise_small_arrays`` takes every transient one-dimensional array of at most ``limit`` elements in the SDFG and
its nested SDFGs whose every access is a tasklet's, at a constant subscript, or a reduction summing the whole array:

* each such reduction becomes one tasklet adding the elements, so that every access subscripts the array with a
  constant (a reduction over an array that is not split stays a reduction: numerics.py matches that form too);
* the array is split into one scalar per element (``dace_fortran.array_split.split_array``, which keeps the order its
  access nodes gave the reads and writes of different elements);
* in the SDFGs that held a split array only: writes a later write overwrites before any read go (the array's
  initialisation), scalar fission gives each value its own scalar, the order edges into nodes that are only read move
  onto their readers and the source nodes of containers the state does not write merge (the shapes state fusion
  refuses otherwise), and state fusion merges the states the old reuse kept apart.

Every step keeps the order of the accesses: a node removed or rewired passes the order it carried on with empty
edges. The split itself once lost it, and dead-dataflow elimination then dropped the writes a read still needed.
"""
from __future__ import annotations

import dace
from dace import nodes
from dace.libraries.standard import Reduce


def _is_whole_sum(reduce_node: Reduce, state: dace.SDFGState, limit: int) -> bool:
    from dace_fortran.substitute import numerics

    if not numerics._is_sum(reduce_node) or state.in_degree(reduce_node) != 1 or state.out_degree(reduce_node) != 1:
        return False
    edge = state.in_edges(reduce_node)[0]
    if edge.data.data is None or edge.data.data not in state.sdfg.arrays:
        return False
    desc = state.sdfg.arrays[edge.data.data]
    try:
        size = int(desc.total_size)
    except (TypeError, ValueError):
        return False
    return len(desc.shape) == 1 and size <= limit and edge.data.subset.num_elements() == size


def _expand_sum(state: dace.SDFGState, red: Reduce) -> None:
    """Replace the whole-array sum ``red`` by one tasklet adding the elements."""
    sd = state.sdfg
    inp, out = state.in_edges(red)[0], state.out_edges(red)[0]
    size = int(sd.arrays[inp.data.data].total_size)
    lo = inp.data.subset[0][0]
    tasklet = state.add_tasklet("element_sum", {f"_i{k}": None for k in range(size)}, {"_o": None},
                                "_o = " + " + ".join(f"_i{k}" for k in range(size)))
    for k in range(size):
        state.add_edge(inp.src, inp.src_conn, tasklet, f"_i{k}", dace.Memlet(data=inp.data.data, subset=str(lo + k)))
    state.add_edge(tasklet, "_o", out.dst, out.dst_conn, dace.Memlet.from_memlet(out.data))
    state.remove_node(red)


def expand_small_sums(sdfg: dace.SDFG, limit: int = 8, only: set[str] | None = None) -> int:
    """Replace every reduction summing a whole array of at most ``limit`` elements (of ``only``, if given) by one
    tasklet adding the elements, in ``sdfg`` and its nested SDFGs. Returns the number replaced."""
    count = 0
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for red in [n for n in state.nodes() if isinstance(n, Reduce) and _is_whole_sum(n, state, limit)]:
                if only is not None and state.in_edges(red)[0].data.data not in only:
                    continue
                _expand_sum(state, red)
                count += 1
    return count


def _splittable(sdfg: dace.SDFG, name: str, limit: int) -> bool:
    """Whether every access of ``name`` in ``sdfg`` is a tasklet's at a constant subscript or a whole-array sum of
    it: what is left once the sums are expanded splits into scalars. Edges that carry no data (they only order their
    ends) do not count; no copy, map scope boundary, other library node or nested SDFG may see the array, and no
    interstate edge or condition may read it (``constant_indices`` checks that part)."""
    from dace_fortran.array_split import _constant  # noqa: PLC0415

    found = False
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data != name:
                continue
            found = True
            for e in state.in_edges(node) + state.out_edges(node):
                if e.data.is_empty():
                    continue
                other = e.src if e.dst is node else e.dst
                if isinstance(other, Reduce) and e.src is node and _is_whole_sum(other, state, limit):
                    continue
                if not isinstance(other, nodes.Tasklet) or e.data.data != name or e.data.subset is None:
                    return False
                if len(e.data.subset) != 1 or _constant(e.data.subset[0]) is None:
                    return False
    if not found:
        return False
    for e in sdfg.all_interstate_edges():
        if name in e.data.free_symbols or any(name in str(v) for v in e.data.assignments.values()):
            return False
    from dace.sdfg.state import AbstractControlFlowRegion  # noqa: PLC0415

    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, AbstractControlFlowRegion) and any(m.data == name for m in block.get_meta_read_memlets()):
            return False
    return True


def _candidates(sdfg: dace.SDFG, limit: int) -> list[str]:
    out = []
    for name, desc in sdfg.arrays.items():
        if not desc.transient or type(desc) is not dace.data.Array or len(desc.shape) != 1:
            continue
        try:
            size = int(desc.total_size)
        except (TypeError, ValueError):
            continue
        if size <= limit and _splittable(sdfg, name, limit):
            out.append(name)
    return out


def remove_nodes_keeping_order(state: dace.SDFGState, doomed: list) -> None:
    """Remove the nodes ``doomed`` from ``state`` such that whatever ran before them still runs before whatever ran
    after them: an empty edge from each outside predecessor to each outside successor, unless a path orders them."""
    import networkx as nx  # noqa: PLC0415

    gone = set(doomed)
    preds = {e.src for n in doomed for e in state.in_edges(n) if e.src not in gone}
    succs = {e.dst for n in doomed for e in state.out_edges(n) if e.dst not in gone}
    for n in doomed:
        state.remove_node(n)
    for p in preds:
        for q in succs:
            if not nx.has_path(state.nx, p, q):
                state.add_nedge(p, q, dace.Memlet())
    for p in preds:
        if isinstance(p, nodes.AccessNode) and state.degree(p) == 0:
            state.remove_node(p)


def _data_edges(edges) -> list:
    return [e for e in edges if not e.data.is_empty()]


def _overwritten_write(sdfg: dace.SDFG, state: dace.SDFGState):
    """One write of a transient scalar in ``state`` that a later write there overwrites before anything reads it,
    with the tasklets that only compute it, or None."""
    import networkx as nx  # noqa: PLC0415

    scopes = state.scope_dict()
    by_name: dict[str, list] = {}
    for n in state.data_nodes():
        desc = sdfg.arrays[n.data]
        if isinstance(desc, dace.data.Scalar) and desc.transient and scopes[n] is None:
            by_name.setdefault(n.data, []).append(n)
    for accesses in by_name.values():
        writes = [n for n in accesses if _data_edges(state.in_edges(n))]
        for x in writes:
            if _data_edges(state.out_edges(x)):
                continue  # read here
            later = [w for w in writes if w is not x and nx.has_path(state.nx, x, w)]
            if not later:
                continue  # its value may live on after the state
            unread = True
            for y in accesses:
                if y is x or not _data_edges(state.out_edges(y)) or _data_edges(state.in_edges(y)):
                    continue  # not a reader, or one that reads its own write
                if nx.has_path(state.nx, y, x):
                    continue  # read before the write
                if nx.has_path(state.nx, x, y) and any(nx.has_path(state.nx, w, y) for w in later):
                    continue  # a later write comes between
                unread = False
                break
            if not unread:
                continue
            writers = [e.src for e in _data_edges(state.in_edges(x))]
            if all(isinstance(t, nodes.Tasklet) and not t.has_side_effects(sdfg)
                   and len(_data_edges(state.out_edges(t))) == 1 for t in writers):
                return [x, *writers]
    return None


def remove_overwritten_scalar_writes(sdfg: dace.SDFG) -> int:
    """Drop the writes of transient scalars that a later write in the same state overwrites before anything reads
    them (an array's initialisation ``a(:) = 0`` before ``a = ...``, once ``a`` is split into scalars), with the
    side-effect free tasklets that only compute them. Such a write keeps an order the dataflow does not need, and
    passes that merge the uses of a name (scalar fission) give its node the name of the previous value, which then
    blocks state fusion. Returns the number of writes removed; the SDFG's own states only."""
    removed = 0
    for state in sdfg.states():
        while True:
            doomed = _overwritten_write(sdfg, state)
            if doomed is None:
                break
            remove_nodes_keeping_order(state, doomed)
            removed += 1
    return removed


def order_readers_not_reads(sdfg: dace.SDFG) -> int:
    """Move the ordering edges into a node that is only read onto its readers: ``p -> (x) -> r`` with an empty
    ``p -> x`` becomes ``(x) -> r`` with an empty ``p -> r``. Same order, but the node is a source again: state fusion
    takes a node with any in-edge for a write, and refuses to fuse a state that writes what the previous one reads.
    Returns the number of nodes changed; the SDFG's own states only."""
    import networkx as nx  # noqa: PLC0415

    changed = 0
    for state in sdfg.states():
        scopes = state.scope_dict()
        for x in list(state.data_nodes()):
            ins = state.in_edges(x)
            if scopes[x] is not None or not ins or _data_edges(ins) or not _data_edges(state.out_edges(x)):
                continue
            readers = {e.dst for e in state.out_edges(x)}
            for e in list(ins):
                state.remove_edge(e)
                for r in readers:
                    if not nx.has_path(state.nx, e.src, r):
                        state.add_nedge(e.src, r, dace.Memlet())
            changed += 1
    return changed


def merge_unwritten_sources(sdfg: dace.SDFG) -> int:
    """Merge the source nodes of a container that no node of the state writes into one: every read there sees the
    same value. State fusion refuses a second state that reads one container through two source nodes. Returns the
    number of nodes merged away; the SDFG's own states only."""
    merged = 0
    for state in sdfg.states():
        scopes = state.scope_dict()
        written = {n.data for n in state.data_nodes() if _data_edges(state.in_edges(n))}
        groups: dict[str, list] = {}
        for n in state.source_nodes():
            if isinstance(n, nodes.AccessNode) and scopes[n] is None and n.data not in written:
                groups.setdefault(n.data, []).append(n)
        for group in groups.values():
            keep = group[0]
            for other in group[1:]:
                for e in list(state.out_edges(other)):
                    state.add_edge(keep, e.src_conn, e.dst, e.dst_conn, e.data)
                    state.remove_edge(e)
                state.remove_node(other)
                merged += 1
    return merged


def tidy_after_split(sdfg: dace.SDFG) -> None:
    """Overwritten writes removed, scalar fission and state fusion on ``sdfg`` (one nested SDFG): each value its own
    scalar, initialisations that are overwritten before any read gone, and the states the reuse of one container kept
    apart fused."""
    from dace_fortran import pipelines as P  # noqa: PLC0415

    remove_overwritten_scalar_writes(sdfg)
    P.fission_scalars(sdfg)
    order_readers_not_reads(sdfg)
    merge_unwritten_sources(sdfg)
    sdfg.apply_transformations_repeated(P.StateFusionExtended)


def scalarise_small_arrays(sdfg: dace.SDFG, limit: int = 8) -> dict[str, int]:
    """See the module docstring. Returns the arrays split, by ``<SDFG label>#<cfg id>/<name>``, with their element
    counts."""
    from dace_fortran import array_split  # noqa: PLC0415

    split: dict[str, int] = {}
    for sd in list(sdfg.all_sdfgs_recursive()):
        names = _candidates(sd, limit)
        if not names:
            continue
        for state in sd.states():
            for red in [n for n in state.nodes() if isinstance(n, Reduce) and _is_whole_sum(n, state, limit)]:
                if state.in_edges(red)[0].data.data in names:
                    _expand_sum(state, red)
        done = False
        for name in names:
            parts = array_split.split_array(sd, name, 0)
            if parts:
                split[f"{sd.label}#{sd.cfg_id}/{name}"] = len(parts)
                done = True
        if done:
            tidy_after_split(sd)
    return split
