"""Generic stencil fusion and copy forwarding on a specialised SDFG, as the substitute pipeline applies them.

Three steps, each driven by the dataflow alone:

1. ``fold_control_flow_constants``: reads of known scalars and of elements of constant arrays in conditions, loop
   statements and interstate assignments are replaced by their values; constant propagation and dead-branch
   elimination then drop the branches the case never takes (producer and consumer of a stencil end up in one state).
2. ``fuse_stencil_pairs``: every transient array written by one top-level map and read by another whose per-iteration
   reads the producer can satisfy (``OTFStencilFusion``) is recomputed inside the consumer; the producer goes once
   none of its outputs has a reader left.
3. ``forward_copies``: a transient that only mirrors a non-transient array through plain copy maps (an input pack, an
   output unpack) is removed and its users address the array directly, through the copy's element relation.

The slice harness (slice/fused/fuse_dev.py) exercises the same functions on one direction of the benchmark.
"""

from __future__ import annotations

import ast
import re
from typing import Iterable

import dace
from dace import symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.subsets import Range
from dace.transformation.dataflow.otf_stencil_fusion import OTFStencilFusion
from dace.transformation.dataflow.tile_stencil_fusion import TileStencilFusion
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.dead_state_elimination import DeadStateElimination


# --- 1. control flow constants -----------------------------------------------------------------------------------------


def _fold_code(code: CodeBlock | None, subs: dict[str, str], elements: dict[str, list]) -> CodeBlock | None:
    if code is None:
        return None
    text = code.as_string
    if elements:
        # The index may be a constant expression rather than a literal (``a[(1 - 1)]`` after a loop was unrolled and
        # its iterator propagated): evaluate it.
        def element(m: re.Match) -> str:
            index = symbolic.simplify(symbolic.pystr_to_symbolic(m.group(2)))
            values = elements[m.group(1)]
            if getattr(index, "is_Integer", False) and 0 <= int(index) < len(values) and values[int(index)] is not None:
                return str(values[int(index)])
            return m.group(0)

        pattern = r"\b(" + "|".join(map(re.escape, elements)) + r")\[([^\[\]]+)\]"
        text = re.sub(pattern, element, text)
    for name, value in subs.items():
        text = re.sub(rf"\b{re.escape(name)}\b", f"({value})", text)
    if " if " in text and code.language == dace.Language.Python:
        text = _fold_conditional_expressions(text)
    return CodeBlock(text, language=code.language)


def _fold_conditional_expressions(text: str) -> str:
    """``a if c else b`` with a condition of literals only (``1 == 1``, after an element read was folded) replaced by
    the branch it takes; the text unchanged when there is none or it does not parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text

    class Fold(ast.NodeTransformer):
        changed = False

        def visit_IfExp(self, node: ast.IfExp):
            self.generic_visit(node)
            if all(isinstance(n, (ast.Constant, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.operator,
                                  ast.cmpop, ast.boolop, ast.unaryop, ast.Load)) for n in ast.walk(node.test)):
                try:
                    taken = eval(compile(ast.Expression(node.test), "<condition>", "eval"), {"__builtins__": {}})
                except Exception:  # noqa: BLE001
                    return node
                self.changed = True
                return node.body if taken else node.orelse
            return node

    fold = Fold()
    tree = fold.visit(tree)
    return ast.unparse(tree) if fold.changed else text


def fold_control_flow_constants(sdfg: dace.SDFG, scalars: dict[str, object], elements: dict[str, list]) -> None:
    """Replace reads of known scalars and of elements of constant arrays in every condition, loop statement and
    interstate assignment, recursively; then propagate the constants and drop the branches they decide. Folding and
    propagation alternate: an element read through an index expression of symbols (``a[i - 1]`` after a loop was
    unrolled) becomes a literal subscript only once the propagation has substituted the symbols."""
    subs = {k: str(v) for k, v in scalars.items()}

    def fold_once() -> bool:
        changed = False
        for sd in sdfg.all_sdfgs_recursive():
            for e in sd.all_interstate_edges(recursive=True):
                cond = _fold_code(e.data.condition, subs, elements)
                if cond is not None and cond.as_string != e.data.condition.as_string:
                    e.data.condition, changed = cond, True
                assignments = {k: _fold_code(CodeBlock(v), subs, elements).as_string for k, v in e.data.assignments.items()}
                if assignments != {k: str(v) for k, v in e.data.assignments.items()}:
                    e.data.assignments, changed = assignments, True
            for blk in sd.all_control_flow_blocks(recursive=True):
                if isinstance(blk, ConditionalBlock):
                    folded = [(_fold_code(c, subs, elements), b) for c, b in blk.branches]
                    if [(c.as_string if c else None) for c, _ in folded] != [(c.as_string if c else None) for c, _ in blk.branches]:
                        blk.branches[:], changed = folded, True
                elif isinstance(blk, LoopRegion):
                    for attr in ("loop_condition", "init_statement", "update_statement"):
                        code = getattr(blk, attr)
                        folded = _fold_code(code, subs, elements)
                        if folded is not None and folded.as_string != code.as_string:
                            setattr(blk, attr, folded)
                            changed = True
        return changed

    for round_ in range(4):
        changed = fold_once()
        ConstantPropagation().apply_pass(sdfg, {})
        DeadStateElimination().apply_pass(sdfg, {})
        if not changed and round_ >= 1:
            break


# --- 2. stencil fusion -------------------------------------------------------------------------------------------------


def _top_maps(state: dace.SDFGState) -> list[nodes.MapEntry]:
    scopes = state.scope_dict()
    return [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scopes[n] is None]


def stencil_pairs(sdfg: dace.SDFG, fusion: type = OTFStencilFusion
                  ) -> list[tuple[dace.SDFGState, nodes.MapExit, nodes.AccessNode, nodes.MapEntry]]:
    """``(state, producer exit, array, consumer entry)`` for every transient written by one top-level map and read
    by another top-level map of the same state, where the fusion applies."""
    pairs = []
    for state in sdfg.states():
        tops = set(_top_maps(state))
        for node in state.data_nodes():
            if not sdfg.arrays[node.data].transient or state.in_degree(node) != 1:
                continue
            producer = state.in_edges(node)[0].src
            if not isinstance(producer, nodes.MapExit) or state.entry_node(producer) not in tops:
                continue
            for e in state.out_edges(node):
                consumer = e.dst
                if isinstance(consumer, nodes.MapEntry) and consumer in tops and consumer is not state.entry_node(
                    producer
                ):
                    xform = fusion()
                    xform.first_map_exit, xform.array, xform.second_map_entry = producer, node, consumer
                    if xform.can_be_applied(state, 0, sdfg):
                        pairs.append((state, producer, node, consumer))
    return pairs


def fuse_stencil_pairs(sdfg: dace.SDFG, tile: bool = False) -> list[str]:
    """Apply :class:`OTFStencilFusion` (``tile``: :class:`TileStencilFusion`, the producer once per tile of the
    consumer into shared memory, every output of it the consumer reads at once) to every stencil pair until none is
    left; returns the fused arrays' names."""
    propagate_memlets_sdfg(sdfg)
    fused = []
    # The tile fusion takes the pairs it can (constant offsets along the parameters, no range reads); the
    # recomputing fusion takes what is left.
    for fusion in ([TileStencilFusion, OTFStencilFusion] if tile else [OTFStencilFusion]):
        while True:
            pairs = stencil_pairs(sdfg, fusion)
            if not pairs:
                break
            state, pexit, node, centry = pairs[0]
            fusion.apply_to(sdfg, first_map_exit=pexit, array=node, second_map_entry=centry, verify=False)
            fused.append(f"{node.data} ({'tile' if fusion is TileStencilFusion else 'recompute'})")
    return fused


# --- 3. copy forwarding ------------------------------------------------------------------------------------------------


def copy_tasklets(sdfg: dace.SDFG, src: str, dst: str) -> list:
    """Every tasklet in a top-level map that only copies one element of ``src`` into ``dst`` (with its state, map)."""
    found = []
    for state in sdfg.states():
        for entry in _top_maps(state):
            exit_ = state.exit_node(entry)
            for t in state.scope_subgraph(entry).nodes():
                if not isinstance(t, nodes.Tasklet):
                    continue
                ins, outs = state.in_edges(t), state.out_edges(t)
                if (
                    len(ins) == 1
                    and len(outs) == 1
                    and ins[0].data.data == src
                    and outs[0].data.data == dst
                    and ins[0].src is entry
                    and outs[0].dst is exit_
                    and t.code.as_string.strip() == f"{outs[0].src_conn} = {ins[0].dst_conn}"
                ):
                    found.append((state, entry, t))
    return found


def _copy_dims(state, entry, tasklet, name: str) -> list[tuple]:
    """Per dim of ``name`` in the copy ``tasklet``: the map parameter subscripting it (or None) and the shift added
    to it (a constant or an expression of symbols)."""
    params = [symbolic.symbol(p) for p in entry.map.params]
    edges = state.in_edges(tasklet) + state.out_edges(tasklet)
    subset = next(e.data.subset for e in edges if e.data.data == name)
    out = []
    for begin, _end, _step in subset.ranges:
        expr = symbolic.pystr_to_symbolic(str(begin))
        p = next((q for q in params if q in expr.free_symbols), None)
        shift = symbolic.simplify(expr - p if p is not None else expr)
        if p is not None and any(q in shift.free_symbols for q in params):
            raise LookupError(f"{name}: a subscript mixes map parameters")
        out.append((p, shift))
    return out


def _same(values) -> bool:
    first = values[0]
    return all(symbolic.simplify(v - first) == 0 for v in values[1:])


def _relation_candidates(copies, scratch: str, real: str) -> list[list[tuple[int | None, object]]]:
    """The element relations one group of copies (the copies of one state) is consistent with, between ``scratch``
    and ``real``: per dim of ``real``, the dim of ``scratch`` that supplies its index and the shift to add, or
    ``None`` with the fixed index the real array has there (a record index the scratch lacks).

    Dims subscripted by a map parameter pair by that parameter. Dims subscripted by constants (the copies copy one
    component each) pair by the one assignment under which every copy of the group shows the same shift; a real
    constant dim left unpaired must be the same constant in every copy. Raises LookupError when no or several
    assignments fit, or when the copies are not element-wise. A single copy can fit several assignments."""
    import itertools

    dims = [(_copy_dims(st, en, tk, scratch), _copy_dims(st, en, tk, real)) for st, en, tk in copies]
    t0, r0 = dims[0]
    if any(len(t) != len(t0) or len(r) != len(r0) for t, r in dims):
        raise LookupError(f"{scratch} and {real}: copies of different ranks")
    scratch_consts = [d for d, (p, _c) in enumerate(t0) if p is None]
    real_consts = [i for i, (pe, _ce) in enumerate(r0) if pe is None]
    mapping: dict[int, tuple] = {}
    for i, (pe, _ce) in enumerate(r0):
        if pe is None:
            continue
        d = next((d for d, (p, _c) in enumerate(t0) if p == pe), None)
        if d is None:
            raise LookupError(f"{scratch} and {real} are not copied element by element")
        shifts = [r[i][1] - t[d][1] for t, r in dims]
        if not _same(shifts):
            raise LookupError(f"{scratch} and {real}: the copies disagree on dim {i}")
        mapping[i] = (d, shifts[0])
    if len(scratch_consts) > len(real_consts):
        raise LookupError(f"{scratch} and {real} are not copied element by element")
    fits = []
    for chosen in itertools.permutations(real_consts, len(scratch_consts)):
        candidate: dict[int, tuple] = {}
        for d, i in zip(scratch_consts, chosen):
            shifts = [r[i][1] - t[d][1] for t, r in dims]
            if not _same(shifts):
                break
            candidate[i] = (d, shifts[0])
        else:
            for i in real_consts:
                if i in candidate:
                    continue
                consts = [r[i][1] for _t, r in dims]
                if not _same(consts):
                    break
                candidate[i] = (None, consts[0])
            else:
                fits.append(candidate)
    if not fits:
        raise LookupError(f"{scratch} and {real}: no way to pair the constant dims")
    return [[{**mapping, **fit}[i] for i in range(len(r0))] for fit in fits]


def _numeric_extent(expr, sdfg: dace.SDFG, known: dict | None = None):
    """The integer an array extent evaluates to under the SDFG's constants and ``known`` symbol values (the extents
    the recorded calls had: the configuration the binding checks implies them), or None."""
    try:
        return int(symbolic.evaluate(expr, {**(known or {}), **sdfg.constants}))
    except Exception:  # noqa: BLE001 -- a symbol without a value
        return None


def _extents_agree(sdfg: dace.SDFG, scratch: str, real: str, relation, known: dict | None = None) -> bool:
    """A scratch dim with more elements than the real dim it would supply cannot be its index (a 2D case's z slot
    of one element, which the variable dim of a copy per variable would otherwise fit)."""
    sshape = sdfg.arrays[scratch].shape
    rshape = sdfg.arrays[real].shape
    for i, (d, _shift) in enumerate(relation):
        if d is None:
            continue
        se, re_ = _numeric_extent(sshape[d], sdfg, known), _numeric_extent(rshape[i], sdfg, known)
        if se is not None and re_ is not None and se > re_:
            return False
    return True


def _structure(relation) -> tuple:
    """Which scratch dim (or none) supplies each real dim: what a relation shares across states, whose fixed
    indices differ (one direction's slot of the real array, say)."""
    return tuple(d for d, _sh in relation)


def _relation(copies, scratch: str, real: str) -> list[tuple[int | None, object]]:
    """The unique element relation ``copies`` establish (see :func:`_relation_candidates`); LookupError if several
    fit."""
    fits = _relation_candidates(copies, scratch, real)
    if len(fits) != 1:
        raise LookupError(f"{scratch} and {real}: {len(fits)} ways to pair the constant dims")
    return fits[0]


def _remove_copy(state: dace.SDFGState, entry: nodes.MapEntry, tasklet: nodes.Tasklet) -> None:
    exit_ = state.exit_node(entry)
    body = [n for n in state.scope_subgraph(entry).nodes() if isinstance(n, nodes.Tasklet)]
    if len(body) == 1:
        sub = state.scope_subgraph(entry, include_entry=True, include_exit=True)
        outer = {e.src for e in state.in_edges(entry)} | {e.dst for e in state.out_edges(exit_)}
        state.remove_nodes_from(list(sub.nodes()))
        for n in outer:
            if n in state.nodes() and state.degree(n) == 0:
                state.remove_node(n)
        return
    for e in state.in_edges(tasklet):
        conn = e.src_conn
        state.remove_edge(e)
        if not any(x.src_conn == conn for x in state.out_edges(entry)):
            entry.remove_out_connector(conn)
            in_conn = "IN_" + conn[4:]
            for x in list(state.in_edges(entry)):
                if x.dst_conn == in_conn:
                    state.remove_edge(x)
                    if state.degree(x.src) == 0:
                        state.remove_node(x.src)
            entry.remove_in_connector(in_conn)
    for e in state.out_edges(tasklet):
        conn = e.dst_conn
        state.remove_edge(e)
        if not any(x.dst_conn == conn for x in state.in_edges(exit_)):
            exit_.remove_in_connector(conn)
            out_conn = "OUT_" + conn[3:]
            for x in list(state.out_edges(exit_)):
                if x.src_conn == out_conn:
                    state.remove_edge(x)
                    if state.degree(x.dst) == 0:
                        state.remove_node(x.dst)
            exit_.remove_out_connector(out_conn)
    state.remove_node(tasklet)


def _remap_subset(subset, full_scratch, real_desc, mapping):
    if subset == full_scratch:  # a whole-array subset (a nested connector) stays the whole array
        return Range.from_array(real_desc)
    return Range([
        (subset.ranges[d][0] + sh, subset.ranges[d][1] + sh, subset.ranges[d][2]) if d is not None else (sh, sh, 1)
        for d, sh in mapping
    ])


def _remap_code(code: str, name: str, mapping) -> str:
    """``code`` (an interstate assignment or condition) with every subscript ``name[...]`` addressed through the
    element relation, as :func:`_remap_subset` does for memlets."""
    if f"{name}[" not in code:
        return code

    class Remap(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == name:
                old = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                new = [str(symbolic.pystr_to_symbolic(ast.unparse(old[d])) + sh) if d is not None else str(sh)
                       for d, sh in mapping]
                node.slice = ast.parse(", ".join(new), mode="eval").body
            return node

    return ast.unparse(Remap().visit(ast.parse(code)))


def _define_symbols(node: nodes.NestedSDFG, parent: dace.SDFG, symbols) -> None:
    """Make the parent's ``symbols`` available inside ``node``'s SDFG: declared there and mapped to themselves."""
    for sym in symbols:
        name = str(sym)
        if name in node.sdfg.arrays:
            continue
        if name not in node.sdfg.symbols:
            node.sdfg.add_symbol(name, parent.symbols.get(name, dace.int64))
        node.symbol_mapping.setdefault(name, symbolic.pystr_to_symbolic(name))


def _remap_nested(node: nodes.NestedSDFG, parent: dace.SDFG, conn: str, full_scratch, real_desc, mapping, symbols,
                  seen: set) -> None:
    """Address the real array inside a nested SDFG that took the scratch array through ``conn``: the connector's
    descriptor becomes the real one, every subset and every subscript in an interstate edge goes through the element
    relation, and the symbols the real descriptor and the relation use are defined in the nested SDFG
    (recursively)."""
    nsdfg = node.sdfg
    _define_symbols(node, parent, symbols)
    if (id(nsdfg), conn) in seen:
        return
    seen.add((id(nsdfg), conn))
    inner = nsdfg.arrays[conn]
    inner.shape, inner.strides, inner.total_size = real_desc.shape, real_desc.strides, real_desc.total_size
    inner.offset = real_desc.offset
    # MUSCL's slopes read the scratch array in interstate assignments (3D_shockdroplet_muscl): left in the scratch's
    # order of dims, they addressed the real array's dims in the wrong order, out of its bounds.
    for e in nsdfg.all_interstate_edges():
        e.data.assignments = {k: _remap_code(v, conn, mapping) for k, v in e.data.assignments.items()}
        e.data.condition = CodeBlock(_remap_code(e.data.condition.as_string, conn, mapping))
    for st in nsdfg.all_states():
        for e in st.edges():
            if e.data.data == conn and e.data.subset is not None:
                e.data.subset = _remap_subset(e.data.subset, full_scratch, real_desc, mapping)
        for n in st.nodes():
            if isinstance(n, nodes.NestedSDFG):
                for e in st.in_edges(n) + st.out_edges(n):
                    if e.data.data == conn:
                        _remap_nested(n, nsdfg, e.dst_conn if e.dst is n else e.src_conn, full_scratch, real_desc,
                                      mapping, symbols, seen)


def _states_using(sdfg: dace.SDFG, name: str) -> set:
    """The top-level states in which ``name`` is accessed (through an access node or a nested connector)."""
    return {st for st in sdfg.states() if any(isinstance(n, nodes.AccessNode) and n.data == name for n in st.nodes())}



def _nearest_state(start, targets: set, forward: bool):
    """The first state of ``targets`` reached from ``start`` along the control flow (forward: successors, else
    predecessors), through nested regions by their first/last states; None if the nearest blocks hold several."""
    from collections import deque

    def states_of(block):
        if isinstance(block, dace.SDFGState):
            return [block]
        return list(block.all_states()) if hasattr(block, "all_states") else []

    block = start
    while block is not None:
        graph = block.parent_graph
        if graph is None:
            return None
        seen, queue = {block}, deque([block])
        while queue:
            b = queue.popleft()
            nxt = [e.dst for e in graph.out_edges(b)] if forward else [e.src for e in graph.in_edges(b)]
            for n in nxt:
                if n in seen:
                    continue
                seen.add(n)
                hits = [st for st in states_of(n) if st in targets]
                if len(hits) == 1:
                    return hits[0]
                if len(hits) > 1:
                    return None
                queue.append(n)
        # Nothing in this region after (before) the block: continue after (before) the region itself.
        block = graph if isinstance(graph, dace.sdfg.state.ControlFlowRegion) and not isinstance(graph, dace.SDFG) else None
    return None

def forward_copy(sdfg: dace.SDFG, scratch: str, real: str, reads: bool, known: dict | None = None) -> int:
    """Remove the copy maps between ``scratch`` and ``real`` (``reads``: real -> scratch) and address ``real``
    wherever ``scratch`` was used, through the element relation the copies of the same state establish (the states
    may relate the two differently: a sweep per direction packs with its own permutation). Returns the number of
    copies removed. Raises LookupError when a state uses ``scratch`` without copies of its own to tell the relation."""
    src, dst = (real, scratch) if reads else (scratch, real)
    copies = copy_tasklets(sdfg, src, dst)
    if not copies:
        raise LookupError(f"no copy {src} -> {dst}")
    by_state: dict = {}
    for state, entry, tasklet in copies:
        by_state.setdefault(state, []).append((state, entry, tasklet))
    users = _states_using(sdfg, scratch)
    # A dim of the scratch fixed at one index everywhere (a 2D sweep's z) supplies the same elements whichever
    # constant dim of the real array it pairs with: folded to that index, the pairings that differ there are one.
    pinned = _pinned_dims(sdfg, scratch)
    candidates = {state: _decide([f for f in _relation_candidates(group, scratch, real)
                                  if _extents_agree(sdfg, scratch, real, f, known)], pinned)
                  for state, group in by_state.items()}
    # The pairing of dims is one layout relation; only the fixed indices differ between states. The states whose
    # copies determine it decide it for those whose copies fit several pairings.
    decided = {_structure(fits[0]) for fits in candidates.values() if len(fits) == 1}
    if len(decided) > 1:
        raise LookupError(f"{scratch} and {real}: the states pair the dims differently")
    # When no state's copies decide the pairing, forwarding refuses: a guess (fewest index shifts, say) picked the
    # wrong pairing for some solvers and produced out-of-bounds subsets.
    relations = {}
    for state, fits in candidates.items():
        chosen = [f for f in fits if not decided or _structure(f) in decided]
        if len(chosen) != 1:
            raise LookupError(f"{scratch} and {real}: {len(chosen)} ways to pair the constant dims")
        relations[state] = chosen[0]
    # A state that uses the scratch but has no copies of its own: a writer takes the relation of the copy that
    # consumes what it wrote (the first copy state after it), a reader the one of the copy that filled it (the last
    # before it).
    for state in users - set(by_state):
        acc = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == scratch]
        writes = any(state.in_degree(n) for n in acc)
        reads_here = any(state.out_degree(n) for n in acc)
        if writes == reads_here:
            raise LookupError(f"{scratch}: state {state.label} both reads and writes it without copies")
        found = _nearest_state(state, set(by_state), forward=writes)
        if found is None:
            raise LookupError(f"{scratch}: no copy {'after' if writes else 'before'} state {state.label}")
        relations[state] = relations[found]
    for state, group in by_state.items():
        for st, entry, tasklet in group:
            _remove_copy(st, entry, tasklet)
    for st, mapping in relations.items():
        _rewrite_state(sdfg, st, scratch, real, mapping)
    del sdfg.arrays[scratch]
    return len(copies)


def _rewrite_state(sdfg: dace.SDFG, st: dace.SDFGState, scratch: str, real: str, mapping) -> None:
    """Address ``real`` wherever ``st`` used ``scratch``, through the element relation ``mapping`` (per dim of
    ``real``: the scratch dim supplying it and the shift, or ``None`` and the fixed index).  A read of the renamed
    array is ordered after every writer of ``real`` the state already holds."""
    full_scratch, real_desc = Range.from_array(sdfg.arrays[scratch]), sdfg.arrays[real]
    symbols = set(real_desc.free_symbols)
    for _d, sh in mapping:
        symbols |= set(getattr(sh, "free_symbols", ()))
    seen: set = set()
    for node in st.nodes():
        if isinstance(node, nodes.NestedSDFG):
            for e in st.in_edges(node) + st.out_edges(node):
                if e.data.data == scratch:
                    _remap_nested(node, sdfg, e.dst_conn if e.dst is node else e.src_conn, full_scratch,
                                  real_desc, mapping, symbols, seen)
    writers = [n for n in st.nodes() if isinstance(n, nodes.AccessNode) and n.data == real
               and any(not e.data.is_empty() for e in st.in_edges(n))]
    renamed = []
    for node in st.nodes():
        if isinstance(node, nodes.AccessNode) and node.data == scratch:
            node.data = real
            renamed.append(node)
    for e in st.edges():
        if e.data.data == scratch:
            e.data.data = real
            e.data.subset = _remap_subset(e.data.subset, full_scratch, real_desc, mapping)
    import networkx as nx  # noqa: PLC0415

    for node in renamed:
        if not st.out_degree(node):
            continue
        for w in writers:
            if w is node or any(e.dst is node for e in st.out_edges(w)):
                continue
            if nx.has_path(st.nx, node, w):
                continue  # the writer runs after this read, which takes the value the copies took
            st.add_nedge(w, node, dace.Memlet())


def mirrored_scratch(sdfg: dace.SDFG) -> list[tuple[str, str, bool]]:
    """``(scratch, real, reads)`` for every transient whose only writers are copy maps from one non-transient array
    (``reads``), or whose only readers are copy maps into one (``not reads``)."""
    writers: dict[str, set] = {}
    readers: dict[str, set] = {}
    copies: dict[tuple[str, str], int] = {}
    for state in sdfg.states():
        for entry in _top_maps(state):
            exit_ = state.exit_node(entry)
            for t in state.scope_subgraph(entry).nodes():
                if isinstance(t, nodes.Tasklet):
                    ins, outs = state.in_edges(t), state.out_edges(t)
                    if len(ins) == 1 and len(outs) == 1 and ins[0].src is entry and outs[0].dst is exit_ \
                            and t.code.as_string.strip() == f"{outs[0].src_conn} = {ins[0].dst_conn}":
                        copies[(ins[0].data.data, outs[0].data.data)] = copies.get(
                            (ins[0].data.data, outs[0].data.data), 0) + 1
        for node in state.data_nodes():
            for e in state.in_edges(node):
                writers.setdefault(node.data, set()).add(e.src)
            for e in state.out_edges(node):
                readers.setdefault(node.data, set()).add(e.dst)
    out = []
    for name, desc in sdfg.arrays.items():
        if not desc.transient or not isinstance(desc, dace.data.Array):
            continue
        sources = [s for (s, d) in copies if d == name and not sdfg.arrays[s].transient]
        sinks = [d for (s, d) in copies if s == name and not sdfg.arrays[d].transient]
        if len(set(sources)) == 1 and all(
                isinstance(w, nodes.MapExit) for w in writers.get(name, ())):
            out.append((name, sources[0], True))
        elif len(set(sinks)) == 1 and all(isinstance(r, nodes.MapEntry) for r in readers.get(name, ())):
            out.append((name, sinks[0], False))
    return out


def forward_copies(sdfg: dace.SDFG, pairs: Iterable[tuple[str, str, bool]] | None = None,
                   known: dict | None = None, skip: Iterable[str] = ()) -> list[str]:
    """Forward every mirrored scratch array (or the given ``(scratch, real, reads)`` triples), then every staging
    array (:func:`forward_staging`). ``known``: symbol values (the recorded extents) that decide between pairings of
    the constant dims a copy per variable leaves open. ``skip``: scratch arrays left as they are (an A/B of one
    forwarding)."""
    done = []
    skip = set(skip)
    for scratch, real, reads in (pairs if pairs is not None else mirrored_scratch(sdfg)):
        if scratch in skip:
            done.append(f"{scratch} skipped")
            continue
        try:
            n = forward_copy(sdfg, scratch, real, reads, known)
        except LookupError as e:
            done.append(f"{scratch} kept: {e}")
            continue
        done.append(f"{scratch} -> {real} ({n} copies)")
    if pairs is None:
        done += forward_staging(sdfg, known, skip)
    return done


def _fills(state: dace.SDFGState, scratch: str) -> list[tuple[nodes.MapEntry, nodes.Tasklet, str]]:
    """The copy maps of ``state`` that fill ``scratch`` from a non-transient array: (map entry, tasklet, source)."""
    out = []
    for entry in _top_maps(state):
        exit_ = state.exit_node(entry)
        for t in state.scope_subgraph(entry).nodes():
            if not isinstance(t, nodes.Tasklet):
                continue
            ins, outs = state.in_edges(t), state.out_edges(t)
            if (
                len(ins) == 1
                and len(outs) == 1
                and outs[0].data.data == scratch
                and ins[0].data.data
                and not state.sdfg.arrays[ins[0].data.data].transient
                and ins[0].src is entry
                and outs[0].dst is exit_
                and t.code.as_string.strip() == f"{outs[0].src_conn} = {ins[0].dst_conn}"
            ):
                out.append((entry, t, ins[0].data.data))
    return out


def _covered(read: Range, fills: list[Range]) -> bool:
    """Whether every element of ``read`` lies in one of the ``fills`` (integer bounds only): along a dim the fills
    agree on, the read must stay within it; along a dim they differ in, each is a point and the read's points must
    all be among them."""
    try:
        r = [(int(b), int(e)) for b, e, _s in read.ranges]
        f = [[(int(b), int(e)) for b, e, _s in fill.ranges] for fill in fills]
    except (TypeError, ValueError):
        return False
    if not f or any(len(x) != len(r) for x in f):
        return False
    for d, (rb, re_) in enumerate(r):
        bounds = {x[d] for x in f}
        if len(bounds) == 1:
            (b, e), = bounds
            if rb < b or re_ > e:
                return False
        else:
            if any(b != e for b, e in bounds) or any(k not in {b for b, _e in bounds} for k in range(rb, re_ + 1)):
                return False
    return True


def _pinned_dims(sdfg: dace.SDFG, scratch: str) -> dict[int, object]:
    """The dims of ``scratch`` every memlet of the SDFG fixes at one index (a sweep's degenerate direction): dim ->
    that index."""
    pinned: dict = {}
    seen = False
    for st in sdfg.all_states():
        for e in st.edges():
            if e.data.data != scratch or e.data.subset is None:
                continue
            ranges = e.data.subset.ranges
            if not seen:
                pinned = {d: b for d, (b, en, _s) in enumerate(ranges) if symbolic.simplify(en - b) == 0}
                seen = True
                continue
            for d in list(pinned):
                b, en, _s = ranges[d]
                if symbolic.simplify(en - b) != 0 or symbolic.simplify(b - pinned[d]) != 0:
                    del pinned[d]
    return pinned


def _canonical(fit, pinned: dict) -> tuple:
    """A fit with its pairings of pinned scratch dims replaced by the fixed index they amount to: fits that differ
    only there address the same elements."""
    return tuple((None, symbolic.simplify(pinned[d] + sh)) if d in pinned else (d, sh) for d, sh in fit)


def _decide(fits: list, pinned: dict) -> list:
    """The distinct fits once the pinned dims are folded (the first of each kind, folded)."""
    out: dict = {}
    for f in fits:
        c = _canonical(f, pinned)
        out.setdefault(_structure(c), list(c))
    return list(out.values())


def _staging_state(state: dace.SDFGState, scratch: str):
    """``(fills, source)`` when ``scratch`` is pure staging in ``state``: filled by copy maps from one non-transient
    array, every read of it after those fills and within what they wrote; ``None`` otherwise."""
    acc = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == scratch]
    fills = _fills(state, scratch)
    sources = {src for _e, _t, src in fills}
    if not fills or len(sources) != 1:
        return None
    exits = {state.exit_node(entry): entry for entry, _t, _s in fills}
    written = []
    for ex in exits:
        written += [e.data.subset for e in state.out_edges(ex) if e.data.data == scratch]
    filled = set()
    for n in acc:
        for e in state.in_edges(n):
            if e.src in exits:
                filled.add(n)
            elif not (isinstance(e.src, nodes.AccessNode) and e.src.data == scratch and e.data.is_empty()):
                return None
    changed = True
    while changed:
        changed = False
        for n in acc:
            if n not in filled and any(e.src in filled and e.data.is_empty() for e in state.in_edges(n)):
                filled.add(n)
                changed = True
    for n in acc:
        for e in state.out_edges(n):
            if e.data.is_empty():
                continue
            if n not in filled or not _covered(e.data.subset, written):
                return None
    return fills, sources.pop()


def forward_staging(sdfg: dace.SDFG, known: dict | None = None, skip: Iterable[str] = ()) -> list[str]:
    """Forward the staging transients: a transient every state fills by copy maps from one non-transient array
    before reading it, and reads only within what the fills wrote, never carries a value between states -- each
    state's readers take the source array through the element relation its own copies establish (a state whose
    copies leave the pairing of the constant dims open follows the states of the same source that decide it).  The
    copies go; the transient goes once no state uses it."""
    done = []
    skip = set(skip)
    for scratch, desc in list(sdfg.arrays.items()):
        if not desc.transient or not isinstance(desc, dace.data.Array) or scratch in skip:
            continue
        states = _states_using(sdfg, scratch)
        if not states:
            continue
        info = {}
        for st in states:
            found = _staging_state(st, scratch)
            if found is None:
                info = None
                break
            info[st] = found
        if not info:
            continue
        by_source: dict = {}
        for st, (_fills_, src) in info.items():
            by_source.setdefault(src, []).append(st)
        pinned = _pinned_dims(sdfg, scratch)
        relations = {}
        undecided = []
        for src, sts in by_source.items():
            fits = {}
            for st in sts:
                try:
                    fits[st] = _decide(
                        [f for f in _relation_candidates([(st, entry, t) for entry, t, _s in info[st][0]], scratch, src)
                         if _extents_agree(sdfg, scratch, src, f, known)], pinned)
                except LookupError:
                    fits[st] = []
            decided = {_structure(f[0]) for f in fits.values() if len(f) == 1}
            if len(decided) > 1:
                undecided.append(f"{src}: the states pair the dims differently")
                continue
            for st in sts:
                chosen = [f for f in fits[st] if not decided or _structure(f) in decided]
                if len(chosen) == 1:
                    relations[st] = (src, chosen[0])
                else:
                    undecided.append(f"{src}: {len(chosen)} ways to pair the constant dims in {st.label}")
        if not relations:
            done.append(f"{scratch} kept: {'; '.join(undecided[:2])}")
            continue
        for st, (src, mapping) in relations.items():
            for entry, t, _s in info[st][0]:
                _remove_copy(st, entry, t)
            _rewrite_state(sdfg, st, scratch, src, mapping)
        if len(relations) == len(info):
            del sdfg.arrays[scratch]
        done.append(f"{scratch} staged: {len(relations)} of {len(info)} states read {sorted(by_source)} directly"
                    + (f" ({'; '.join(undecided[:2])})" if undecided else ""))
    return done
