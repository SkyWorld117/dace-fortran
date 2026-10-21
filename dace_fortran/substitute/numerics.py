"""Opt-in numerics rewrites: each changes rounding, so none runs unless asked for.

``combine_normalised_quotients``: a normalisation ``w_k = a_k / sum_j a_j`` of values that are themselves quotients
``a_k = n_k / d_k`` costs ``2 m`` divisions for ``m`` terms. Over a common denominator it costs one:
``p_k = n_k * prod_{j != k} d_j``, ``w_k = p_k * (1 / sum_j p_j)``. Weighted averages of this shape (nonlinear
reconstruction weights, mixture fractions) are division-bound on GPUs.

The rewrite matches the pattern in a state's dataflow:

* a ``Reduce`` node summing every element of a small one-dimensional array ``a`` into a scalar ``s`` (or a tasklet
  ``s = a0 + a1 + ...`` adding values each held in its own container, as scalarise.py leaves them);
* each element ``a[k]`` written by its own tasklet whose expression is a quotient (``n / d``, or a product with one
  quotient factor);
* each element divided by the sum, ``a[k] / s``, by its own tasklet -- or, with every term in its own container,
  each term divided by the sum written out in its dividing tasklet, ``a_k / (a_0 + a_1 + ...)``.

The dividing tasklets are replaced by one tasklet that reads the quotients' inputs; the quotient tasklets, the
reduction and the intermediate containers go when nothing else reads them.
"""
from __future__ import annotations

import ast
import copy

import dace
from dace import nodes as nd
from dace.libraries.standard import Reduce


def _rhs(tasklet: nd.Tasklet) -> tuple[str, ast.expr] | None:
    """``(output connector, expression)`` of a tasklet that is one assignment ``out = expr``."""
    if tasklet.code.language != dace.Language.Python:
        return None
    try:
        tree = ast.parse(tasklet.code.as_string.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    target = tree.body[0].targets
    if len(target) != 1 or not isinstance(target[0], ast.Name):
        return None
    return target[0].id, tree.body[0].value


def _quotient(expr: ast.expr) -> tuple[ast.expr, ast.expr] | None:
    """``(numerator, denominator)`` of ``n / d`` or of a product with one quotient factor ``x * (n / d)``."""
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        return expr.left, expr.right
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult):
        for this, other in ((expr.right, expr.left), (expr.left, expr.right)):
            if isinstance(this, ast.BinOp) and isinstance(this.op, ast.Div):
                return ast.BinOp(left=other, op=ast.Mult(), right=this.left), this.right
    return None


def _is_sum(reduce_node: Reduce) -> bool:
    """A reduction over every element that adds its operands, starting from zero."""
    if reduce_node.axes or reduce_node.identity is None or float(reduce_node.identity) != 0.0:
        return False
    try:
        fn = ast.parse(str(reduce_node.wcr).strip(), mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(fn, ast.Lambda) or len(fn.args.args) != 2:
        return False
    a, b = (arg.arg for arg in fn.args.args)
    body = fn.body
    return (isinstance(body, ast.BinOp) and isinstance(body.op, ast.Add) and isinstance(body.left, ast.Name)
            and isinstance(body.right, ast.Name) and {body.left.id, body.right.id} == {a, b})


def _renamed(expr: ast.expr, names: dict[str, str]) -> str:
    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            return ast.Name(id=names.get(node.id, node.id), ctx=node.ctx)

    return ast.unparse(Rename().visit(copy.deepcopy(expr)))


def _element(memlet: dace.Memlet) -> int | None:
    """The constant index of a one-element memlet on a one-dimensional container."""
    if memlet.subset is None or len(memlet.subset) != 1 or memlet.subset.num_elements() != 1:
        return None
    try:
        return int(memlet.subset[0][0])
    except (TypeError, ValueError):
        return None


def _match(state: dace.SDFGState, red: Reduce):
    """The pattern around the reduction ``red``: ``(array node, {k: quotient tasklet}, sum node, {k: dividing
    tasklet})``, or None."""
    if not _is_sum(red) or state.in_degree(red) != 1 or state.out_degree(red) != 1:
        return None
    src = state.in_edges(red)[0].src
    dst = state.out_edges(red)[0].dst
    if not isinstance(src, nd.AccessNode) or not isinstance(dst, nd.AccessNode):
        return None
    sdfg = state.sdfg
    desc = sdfg.arrays[src.data]
    try:
        m = int(desc.total_size)
    except (TypeError, ValueError):
        return None
    if len(desc.shape) != 1 or not 2 <= m <= 8 or state.in_edges(red)[0].data.subset.num_elements() != m:
        return None
    writers: dict[int, nd.Tasklet] = {}
    for e in state.in_edges(src):
        k = _element(e.data)
        if not isinstance(e.src, nd.Tasklet) or k is None or k in writers or state.out_degree(e.src) != 1:
            return None
        rhs = _rhs(e.src)
        if rhs is None or rhs[0] != e.src_conn or _quotient(rhs[1]) is None:
            return None
        writers[k] = e.src
    if sorted(writers) != list(range(m)):
        return None
    dividers: dict[int, nd.Tasklet] = {}
    for e in state.out_edges(src):
        if e.dst is red:
            continue
        t = e.dst
        k = _element(e.data)
        if not isinstance(t, nd.Tasklet) or k is None or k in dividers or state.in_degree(t) != 2 or state.out_degree(t) != 1:
            continue
        rhs = _rhs(t)
        sum_edges = [x for x in state.in_edges(t) if x.src is dst]
        if rhs is None or len(sum_edges) != 1:
            continue
        expr = rhs[1]
        if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div)
                and isinstance(expr.left, ast.Name) and expr.left.id == e.dst_conn
                and isinstance(expr.right, ast.Name) and expr.right.id == sum_edges[0].dst_conn):
            continue
        dividers[k] = t
    if sorted(dividers) != list(range(m)):
        return None
    return src, writers, dst, dividers


def _is_sum_of_inputs(tasklet: nd.Tasklet) -> bool:
    """A tasklet ``out = i0 + i1 + ...`` adding each of its (at least two) inputs once."""
    rhs = _rhs(tasklet)
    if rhs is None:
        return False
    terms, stack = [], [rhs[1]]
    while stack:
        e = stack.pop()
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
            stack += [e.left, e.right]
        elif isinstance(e, ast.Name):
            terms.append(e.id)
        else:
            return False
    return len(terms) >= 2 and sorted(terms) == sorted(tasklet.in_connectors) and len(set(terms)) == len(terms)


def _data(edges) -> list:
    """The edges that carry data (an empty memlet only orders its ends)."""
    return [e for e in edges if not e.data.is_empty()]


def _match_scalars(state: dace.SDFGState, adder: nd.Tasklet):
    """The pattern around a tasklet adding values each held in its own container (``s = a0 + a1 + ...``): ``(term
    access nodes, their quotient tasklets, the sum's access node, a dividing tasklet per term)``, or None. Edges that
    only order their ends do not count."""
    if not _is_sum_of_inputs(adder) or len(_data(state.out_edges(adder))) != 1 or len(adder.in_connectors) > 8:
        return None
    dst = _data(state.out_edges(adder))[0].dst
    if not isinstance(dst, nd.AccessNode):
        return None
    terms, writers, dividers = [], [], []
    for e in _data(state.in_edges(adder)):
        node = e.src
        if not isinstance(node, nd.AccessNode) or len(_data(state.in_edges(node))) != 1 or node in terms:
            return None
        w = _data(state.in_edges(node))[0]
        if not isinstance(w.src, nd.Tasklet) or len(_data(state.out_edges(w.src))) != 1:
            return None
        rhs = _rhs(w.src)
        if rhs is None or rhs[0] != w.src_conn or _quotient(rhs[1]) is None:
            return None
        found = None
        for d in _data(state.out_edges(node)):
            t = d.dst
            if t is adder or not isinstance(t, nd.Tasklet) or len(_data(state.in_edges(t))) != 2 \
                    or len(_data(state.out_edges(t))) != 1:
                continue
            drhs = _rhs(t)
            sum_edges = [x for x in _data(state.in_edges(t)) if x.src is dst]
            if drhs is None or len(sum_edges) != 1:
                continue
            expr = drhs[1]
            if (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div) and isinstance(expr.left, ast.Name)
                    and expr.left.id == d.dst_conn and isinstance(expr.right, ast.Name)
                    and expr.right.id == sum_edges[0].dst_conn):
                if found is not None:
                    return None
                found = t
        if found is None:
            return None
        terms.append(node)
        writers.append(w.src)
        dividers.append(found)
    return terms, writers, dst, dividers


def _inline_normalisation(tasklet: nd.Tasklet) -> tuple[str, list[str]] | None:
    """``(numerator connector, sum connectors)`` of a tasklet ``out = x / (y0 + y1 + ...)`` over plain connectors,
    each summed once and the numerator among them."""
    rhs = _rhs(tasklet)
    if rhs is None:
        return None
    expr = rhs[1]
    if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div) and isinstance(expr.left, ast.Name)):
        return None
    terms, stack = [], [expr.right]
    while stack:
        e = stack.pop()
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
            stack += [e.left, e.right]
        elif isinstance(e, ast.Name):
            terms.append(e.id)
        else:
            return None
    if len(terms) < 2 or len(set(terms)) != len(terms):
        return None
    return expr.left.id, terms


def _match_inline(state: dace.SDFGState, divider: nd.Tasklet):
    """The normalisation a tasklet ``a_k / (a_0 + a_1 + ...)`` belongs to, with each term in its own container and the
    sum written out in every dividing tasklet: ``(term access nodes, their quotient tasklets, a dividing tasklet per
    term)``, or None."""
    form = _inline_normalisation(divider)
    if form is None or len(_data(state.out_edges(divider))) != 1:
        return None

    def sources(t: nd.Tasklet) -> dict[str, nd.AccessNode] | None:
        by_conn = {}
        for e in _data(state.in_edges(t)):
            if not isinstance(e.src, nd.AccessNode) or e.dst_conn in by_conn:
                return None
            by_conn[e.dst_conn] = e.src
        return by_conn

    src = sources(divider)
    num, summed = form
    if src is None or set(src) != set(summed) | {num} or len({id(src[c]) for c in summed}) != len(summed):
        return None
    terms = [src[c] for c in summed]
    if len(terms) > 8:
        return None
    dividers: dict[int, nd.Tasklet] = {}
    for node in terms:
        for e in _data(state.out_edges(node)):
            t = e.dst
            if not isinstance(t, nd.Tasklet) or len(_data(state.out_edges(t))) != 1:
                continue
            f = _inline_normalisation(t)
            ts = sources(t) if f is not None else None
            if ts is None or set(ts) != set(f[1]) | {f[0]} or {id(ts[c]) for c in f[1]} != {id(n) for n in terms}:
                continue
            k = next(i for i, n in enumerate(terms) if n is ts[f[0]])
            if dividers.get(k, t) is not t:
                return None  # two tasklets divide the same term: not one normalisation
            dividers[k] = t
    if sorted(dividers) != list(range(len(terms))):
        return None
    writers = []
    for node in terms:
        ins = _data(state.in_edges(node))
        if len(ins) != 1 or not isinstance(ins[0].src, nd.Tasklet) or len(_data(state.out_edges(ins[0].src))) != 1:
            return None
        rhs = _rhs(ins[0].src)
        if rhs is None or rhs[0] != ins[0].src_conn or _quotient(rhs[1]) is None:
            return None
        writers.append(ins[0].src)
    return terms, writers, [dividers[k] for k in range(len(terms))]


def _apply_inline(state: dace.SDFGState, terms, writers, dividers) -> bool:
    """Rewrite the matched inline normalisation; False (and nothing changed) when the order of the removed nodes
    cannot pass to the combined tasklet."""
    doomed = list(dividers)
    for node, w in zip(terms, writers):
        if not [e for e in _data(state.out_edges(node)) if e.dst not in doomed]:
            doomed += [node, w]
    inputs = [(e.src, e.src_conn, e.data) for w in writers for e in _data(state.in_edges(w))]
    if not _ordering_is_acyclic(state, doomed, inputs, set(dividers)):
        return False
    new = _rewrite(state, writers, dividers, remove=False)
    _take_over_order(state, doomed, new)
    return True


def _doomed_scalars(state: dace.SDFGState, terms, writers, dst: nd.AccessNode, dividers, adder: nd.Tasklet) -> list:
    """The nodes the scalar rewrite removes: the dividers always; the sum (and its adder) and each term (and its
    quotient) once nothing else reads them."""
    doomed = list(dividers)
    readers_left = lambda n: [e for e in _data(state.out_edges(n)) if e.dst not in doomed]  # noqa: E731
    if not readers_left(dst):
        doomed += [dst, adder]
    for node, w in zip(terms, writers):
        if not readers_left(node):
            doomed += [node, w]
    return doomed


def _ordering_is_acyclic(state: dace.SDFGState, doomed: list, inputs, dividers) -> bool:
    """Whether the combined tasklet can take over the order of the ``doomed`` nodes (each empty edge into one of them
    comes into the new tasklet, each one out of them leaves it) without a cycle: no node it would have to precede may
    lie on a path to one of its inputs."""
    import networkx as nx  # noqa: PLC0415

    gone = set(doomed)
    after = {e.dst for n in doomed for e in state.out_edges(n) if e.data.is_empty() and e.dst not in gone}
    # What the new tasklet writes comes after it as well.
    after |= {e.dst for n in doomed if n in dividers for e in _data(state.out_edges(n))}
    before = {src for src, _conn, _memlet in inputs} | {
        e.src for n in doomed for e in state.in_edges(n) if e.data.is_empty() and e.src not in gone}
    return not any(a is b or nx.has_path(state.nx, a, b) for a in after for b in before)


def _take_over_order(state: dace.SDFGState, doomed: list, new: nd.Tasklet) -> None:
    """Give ``new`` the ordering edges of the ``doomed`` nodes it replaces (they computed what it computes), then
    remove those nodes and the access nodes nothing touches any more."""
    import networkx as nx  # noqa: PLC0415

    gone = set(doomed)
    before = {e.src for n in doomed for e in state.in_edges(n) if e.data.is_empty() and e.src not in gone}
    after = {e.dst for n in doomed for e in state.out_edges(n) if e.data.is_empty() and e.dst not in gone}
    inputs = {e.src for n in doomed for e in _data(state.in_edges(n)) if e.src not in gone}
    for n in doomed:
        state.remove_node(n)
    for b in before:
        if not nx.has_path(state.nx, b, new):
            state.add_nedge(b, new, dace.Memlet())
    for a in after:
        if not nx.has_path(state.nx, new, a):
            state.add_nedge(new, a, dace.Memlet())
    for n in inputs | before:
        if isinstance(n, nd.AccessNode) and n in state.nodes() and state.degree(n) == 0:
            state.remove_node(n)


def _apply_scalars(state: dace.SDFGState, terms, writers, dst: nd.AccessNode, dividers, adder: nd.Tasklet) -> bool:
    """Rewrite the matched scalar normalisation; False (and nothing changed) when the order of the removed nodes
    cannot pass to the combined tasklet."""
    doomed = _doomed_scalars(state, terms, writers, dst, dividers, adder)
    inputs = [(e.src, e.src_conn, e.data) for w in writers for e in _data(state.in_edges(w))]
    if not _ordering_is_acyclic(state, doomed, inputs, set(dividers)):
        return False
    new = _rewrite(state, writers, dividers, remove=False)
    _take_over_order(state, doomed, new)
    return True


def _rewrite(state: dace.SDFGState, writers, dividers, remove: bool = True) -> nd.Tasklet:
    """One tasklet computing every ``dividers[k]`` result over a common denominator from the inputs of the quotient
    tasklets ``writers[k]``; the dividing tasklets go (unless ``remove`` is false)."""
    m = len(writers)
    inputs: dict[str, tuple] = {}
    numerators, denominators = [], []
    for k in range(m):
        names = {}
        for e in _data(state.in_edges(writers[k])):
            conn = f"_q{k}_{e.dst_conn}"
            names[e.dst_conn] = conn
            inputs[conn] = (e.src, e.src_conn, copy.deepcopy(e.data))
        _out, expr = _rhs(writers[k])
        n, d = _quotient(expr)
        numerators.append(_renamed(n, names))
        denominators.append(_renamed(d, names))
    lines = [f"_d{k} = {denominators[k]}" for k in range(m)]
    for k in range(m):
        others = " * ".join(f"_d{j}" for j in range(m) if j != k)
        lines.append(f"_p{k} = ({numerators[k]}) * {others}")
    lines.append("_r = 1.0 / (" + " + ".join(f"_p{k}" for k in range(m)) + ")")
    outputs = {}
    for k in range(m):
        e = _data(state.out_edges(dividers[k]))[0]
        lines.append(f"_w{k} = _p{k} * _r")
        outputs[f"_w{k}"] = (e.dst, e.dst_conn, copy.deepcopy(e.data))
    new = state.add_tasklet("combined_normalisation", set(inputs), set(outputs), "\n".join(lines))
    for conn, (node, node_conn, memlet) in inputs.items():
        state.add_edge(node, node_conn, new, conn, memlet)
    for conn, (node, node_conn, memlet) in outputs.items():
        state.add_edge(new, conn, node, node_conn, memlet)
    if remove:
        for t in dividers:
            state.remove_node(t)
    return new


def _apply(state: dace.SDFGState, src: nd.AccessNode, writers, dst: nd.AccessNode, dividers) -> None:
    m = len(writers)
    inputs: dict[str, tuple] = {}
    lines, numerators, denominators = [], [], []
    for k in range(m):
        t = writers[k]
        names = {}
        for e in state.in_edges(t):
            conn = f"_q{k}_{e.dst_conn}"
            names[e.dst_conn] = conn
            inputs[conn] = (e.src, e.src_conn, copy.deepcopy(e.data))
        out_conn, expr = _rhs(t)
        n, d = _quotient(expr)
        numerators.append(_renamed(n, names))
        denominators.append(_renamed(d, names))
    for k in range(m):
        lines.append(f"_d{k} = {denominators[k]}")
    for k in range(m):
        others = " * ".join(f"_d{j}" for j in range(m) if j != k)
        lines.append(f"_p{k} = ({numerators[k]}) * {others}")
    lines.append("_r = 1.0 / (" + " + ".join(f"_p{k}" for k in range(m)) + ")")
    outputs = {}
    for k in range(m):
        e = state.out_edges(dividers[k])[0]
        lines.append(f"_w{k} = _p{k} * _r")
        outputs[f"_w{k}"] = (e.dst, e.dst_conn, copy.deepcopy(e.data))
    new = state.add_tasklet("combined_normalisation", set(inputs), set(outputs), "\n".join(lines))
    for conn, (node, node_conn, memlet) in inputs.items():
        state.add_edge(node, node_conn, new, conn, memlet)
    for conn, (node, node_conn, memlet) in outputs.items():
        state.add_edge(new, conn, node, node_conn, memlet)
    for t in dividers.values():
        state.remove_node(t)
    # The reduction and the quotients go when nothing else reads them.
    if state.out_degree(dst) == 0:
        red = state.in_edges(dst)[0].src
        state.remove_node(dst)
        state.remove_node(red)
    if state.out_degree(src) == 0:
        for t in writers.values():
            for e in state.in_edges(t):
                state.remove_edge(e)
                if state.degree(e.src) == 0 and isinstance(e.src, nd.AccessNode):
                    state.remove_node(e.src)
            state.remove_node(t)
        state.remove_node(src)


def combine_normalised_quotients(sdfg: dace.SDFG) -> int:
    """Rewrite every normalisation of quotients in ``sdfg`` and its nested SDFGs (see the module docstring).
    Returns the number of normalisations rewritten."""
    count = 0
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            while True:
                found = None
                for node in state.nodes():
                    if isinstance(node, Reduce):
                        found = _match(state, node)
                        if found is not None:
                            break
                if found is None:
                    break
                _apply(state, *found)
                count += 1
            # The same normalisation over values each in its own container (scalarise.py splits small arrays so).
            refused = set()
            while True:
                found = None
                for node in state.nodes():
                    if isinstance(node, nd.Tasklet) and node not in refused:
                        found = _match_scalars(state, node)
                        if found is not None:
                            adder = node
                            break
                if found is None:
                    break
                if _apply_scalars(state, *found, adder):
                    count += 1
                else:
                    refused.add(adder)
            # ... and with the sum written out in each dividing tasklet.
            refused = set()
            while True:
                found = None
                for node in state.nodes():
                    if isinstance(node, nd.Tasklet) and node not in refused:
                        found = _match_inline(state, node)
                        if found is not None:
                            first = node
                            break
                if found is None:
                    break
                if _apply_inline(state, *found):
                    count += 1
                else:
                    refused.add(first)
    return count
