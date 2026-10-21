"""Tests of splitting the small arrays of kernel bodies into scalars (scalarise.py) and of the normalisation rewrite
on the form that leaves (numerics.py), CPU. Run: python -m pytest test_scalarise.py. Needs dace-fortran's
order-keeping array split (array_split.split_array turning a one-dimensional transient into ordered scalars)."""
import numpy as np

import dace
from dace import nodes
from dace.libraries.standard import Reduce

from dace_fortran.substitute.scalarise import (
    merge_unwritten_sources,
    order_readers_not_reads,
    remove_overwritten_scalar_writes,
    scalarise_small_arrays,
)

N = 16


def _weights_body(m: int = 3) -> dace.SDFG:
    """A kernel body as a frontend leaves it: per point i, ``a(:) = 0`` then ``a[k] = c[k] / b[i, k]``, ``s = sum(a)``,
    ``w[k] = a[k] / s`` -- the array ``a`` written twice, read through one node, summed by a reduction -- in a nested
    SDFG of two states, and out[i] = sum_k w[k] * v[i, k]."""
    body = dace.SDFG("body")
    body.add_symbol("i", dace.int64)
    for name, shape in (("b", [N, m]), ("v", [N, m]), ("c", [m]), ("o", [N])):
        body.add_array(name, shape, dace.float64)
    body.add_array("a", [m], dace.float64, transient=True)
    body.add_array("w", [m], dace.float64, transient=True)
    body.add_scalar("s", dace.float64, transient=True)
    init = body.add_state("init")
    a0 = init.add_access("a")
    for k in range(m):
        t = init.add_tasklet(f"zero{k}", {}, {"_o": None}, "_o = 0.0")
        init.add_edge(t, "_o", a0, None, dace.Memlet(f"a[{k}]"))
    main = body.add_state_after(init, "main")
    a, w, s = main.add_access("a"), main.add_access("w"), main.add_access("s")
    b, v, c, o = main.add_read("b"), main.add_read("v"), main.add_read("c"), main.add_write("o")
    for k in range(m):
        t = main.add_tasklet(f"quot{k}", {"_c": None, "_b": None}, {"_o": None}, "_o = (_c / _b)")
        main.add_edge(c, None, t, "_c", dace.Memlet(f"c[{k}]"))
        main.add_edge(b, None, t, "_b", dace.Memlet(f"b[i, {k}]"))
        main.add_edge(t, "_o", a, None, dace.Memlet(f"a[{k}]"))
    red = Reduce("sum", "lambda a, b: a + b", None, identity=0)
    main.add_node(red)
    main.add_edge(a, None, red, None, dace.Memlet(f"a[0:{m}]"))
    main.add_edge(red, None, s, None, dace.Memlet("s[0]"))
    for k in range(m):
        t = main.add_tasklet(f"div{k}", {"_a": None, "_s": None}, {"_w": None}, "_w = (_a / _s)")
        main.add_edge(a, None, t, "_a", dace.Memlet(f"a[{k}]"))
        main.add_edge(s, None, t, "_s", dace.Memlet("s[0]"))
        main.add_edge(t, "_w", w, None, dace.Memlet(f"w[{k}]"))
    dot = main.add_tasklet("dot", {f"_w{k}": None for k in range(m)} | {f"_v{k}": None for k in range(m)}, {"_o": None},
                           "_o = " + " + ".join(f"_w{k} * _v{k}" for k in range(m)))
    for k in range(m):
        main.add_edge(w, None, dot, f"_w{k}", dace.Memlet(f"w[{k}]"))
        main.add_edge(v, None, dot, f"_v{k}", dace.Memlet(f"v[i, {k}]"))
    main.add_edge(dot, "_o", o, None, dace.Memlet("o[i]"))

    sdfg = dace.SDFG(f"weights_body_{m}")
    for name, shape in (("B", [N, m]), ("V", [N, m]), ("C", [m]), ("OUT", [N])):
        sdfg.add_array(name, shape, dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("points", {"i": f"0:{N}"})
    nsdfg = state.add_nested_sdfg(body, {"b", "v", "c"}, {"o"}, symbol_mapping={"i": "i"})
    for outer, inner in (("B", "b"), ("V", "v"), ("C", "c")):
        state.add_memlet_path(state.add_read(outer), me, nsdfg, dst_conn=inner, memlet=dace.Memlet.from_array(
            outer, sdfg.arrays[outer]))
    state.add_memlet_path(nsdfg, mx, state.add_write("OUT"), src_conn="o", memlet=dace.Memlet("OUT[0:16]"))
    return sdfg


def _run_weights(sdfg: dace.SDFG, m: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.zeros(N)
    sdfg(B=rng.uniform(0.5, 2.0, (N, m)), V=rng.uniform(-1, 1, (N, m)), C=rng.uniform(0.1, 1.0, m), OUT=out)
    return out


def test_small_arrays_of_a_body_become_scalars_in_one_state():
    sdfg = _weights_body()
    expected = _run_weights(_weights_body())
    split = scalarise_small_arrays(sdfg)
    assert sorted(name.split("/")[-1] for name in split) == ["a", "w"]
    sdfg.validate()
    body = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    assert len(list(body.states())) == 1
    assert not any(isinstance(n, Reduce) for n in body.start_state.nodes())
    assert all(isinstance(d, dace.data.Scalar) for name, d in body.arrays.items() if d.transient)
    np.testing.assert_allclose(_run_weights(sdfg), expected, rtol=0, atol=0)


def test_the_normalisation_rewrite_matches_the_scalar_form():
    from dace_fortran.substitute.numerics import combine_normalised_quotients

    sdfg = _weights_body()
    expected = _run_weights(_weights_body())
    scalarise_small_arrays(sdfg)
    assert combine_normalised_quotients(sdfg) == 1
    sdfg.validate()
    np.testing.assert_allclose(_run_weights(sdfg), expected, rtol=1e-13)


def test_an_unsplit_array_keeps_its_reduction():
    """``a`` also read at a runtime subscript cannot be split: its sum stays a reduction (the form the array matcher
    of numerics.py takes)."""
    sdfg = _weights_body()
    body = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    body.add_symbol("j", dace.int64)
    main = [s for s in body.states() if s.label == "main"][0]
    a = next(n for n in main.data_nodes() if n.data == "a")
    o = next(n for n in main.data_nodes() if n.data == "o")
    peek = main.add_tasklet("peek", {"_a": None}, {}, "pass")
    main.add_edge(a, None, peek, "_a", dace.Memlet("a[j]"))
    main.add_nedge(peek, o, dace.Memlet())
    split = scalarise_small_arrays(sdfg)
    assert not any(name.endswith("/a") for name in split)
    assert any(isinstance(n, Reduce) for n in main.nodes())


def _ordering_state():
    sdfg = dace.SDFG("ordering")
    sdfg.add_array("x", [1], dace.float64)
    sdfg.add_array("y", [1], dace.float64)
    sdfg.add_scalar("t", dace.float64, transient=True)
    state = sdfg.add_state()
    return sdfg, state


def test_an_overwritten_scalar_write_goes():
    sdfg, state = _ordering_state()
    t1, t2 = state.add_access("t"), state.add_access("t")
    zero = state.add_tasklet("zero", {}, {"_o": None}, "_o = 0.0")
    state.add_edge(zero, "_o", t1, None, dace.Memlet("t[0]"))
    copy_in = state.add_tasklet("copy_in", {"_i": None}, {"_o": None}, "_o = _i")
    state.add_edge(state.add_read("x"), None, copy_in, "_i", dace.Memlet("x[0]"))
    state.add_edge(copy_in, "_o", t2, None, dace.Memlet("t[0]"))
    state.add_nedge(t1, copy_in, dace.Memlet())  # the frontend's order: the initialisation first
    out = state.add_tasklet("out", {"_t": None}, {"_o": None}, "_o = 3 * _t")
    state.add_edge(t2, None, out, "_t", dace.Memlet("t[0]"))
    state.add_edge(out, "_o", state.add_write("y"), None, dace.Memlet("y[0]"))
    assert remove_overwritten_scalar_writes(sdfg) == 1
    assert zero not in state.nodes() and t1 not in state.nodes()
    y = np.zeros(1)
    sdfg(x=np.array([2.0]), y=y)
    assert y[0] == 6.0


def test_a_scalar_write_read_later_stays():
    sdfg, state = _ordering_state()
    t1, t2 = state.add_access("t"), state.add_access("t")
    zero = state.add_tasklet("zero", {}, {"_o": None}, "_o = 1.0")
    state.add_edge(zero, "_o", t1, None, dace.Memlet("t[0]"))
    reader = state.add_access("t")  # reads the first value through a node of its own, ordered after the write
    state.add_nedge(t1, reader, dace.Memlet())
    out = state.add_tasklet("out", {"_t": None}, {"_o": None}, "_o = _t")
    state.add_edge(reader, None, out, "_t", dace.Memlet("t[0]"))
    state.add_edge(out, "_o", state.add_write("y"), None, dace.Memlet("y[0]"))
    later = state.add_tasklet("later", {}, {"_o": None}, "_o = 5.0")
    state.add_nedge(out, later, dace.Memlet())
    state.add_edge(later, "_o", t2, None, dace.Memlet("t[0]"))
    state.add_nedge(t1, later, dace.Memlet())
    assert remove_overwritten_scalar_writes(sdfg) == 0


def test_ordering_moves_from_a_read_node_to_its_readers_and_unwritten_sources_merge():
    sdfg, state = _ordering_state()
    x1, x2 = state.add_read("x"), state.add_read("x")
    first = state.add_tasklet("first", {"_i": None}, {"_o": None}, "_o = _i")
    state.add_edge(x1, None, first, "_i", dace.Memlet("x[0]"))
    t = state.add_access("t")
    state.add_edge(first, "_o", t, None, dace.Memlet("t[0]"))
    second = state.add_tasklet("second", {"_i": None, "_t": None}, {"_o": None}, "_o = _i + _t")
    state.add_nedge(t, x2, dace.Memlet())
    state.add_edge(x2, None, second, "_i", dace.Memlet("x[0]"))
    state.add_edge(t, None, second, "_t", dace.Memlet("t[0]"))
    state.add_edge(second, "_o", state.add_write("y"), None, dace.Memlet("y[0]"))
    assert order_readers_not_reads(sdfg) == 1
    assert state.in_degree(x2) == 0
    assert merge_unwritten_sources(sdfg) == 1
    assert len([n for n in state.data_nodes() if n.data == "x"]) == 1
    sdfg.validate()
    y = np.zeros(1)
    sdfg(x=np.array([2.0]), y=y)
    assert y[0] == 4.0


def _accumulate_body() -> dace.SDFG:
    """Per point i, in a kernel body: ``d(:) = 0``, then ``d[0] += x[i]`` and ``d[1] += y[i]`` twice, each step
    reading the node the previous one wrote, and ``o = d[0] - d[1]`` read from the node the last ``d[1]`` update
    wrote: the read of ``d[0]`` follows its last write only through the nodes of ``d`` (a one-sided difference
    accumulated per component, as a boundary condition computes its gradients)."""
    body = dace.SDFG("acc")
    body.add_symbol("i", dace.int64)
    for name in ("x", "y", "o"):
        body.add_array(name, [N], dace.float64)
    body.add_array("d", [2], dace.float64, transient=True)
    state = body.add_state()
    prev = state.add_access("d")
    for k in range(2):
        t = state.add_tasklet(f"zero{k}", {}, {"_o": None}, "_o = 0.0")
        state.add_edge(t, "_o", prev, None, dace.Memlet(f"d[{k}]"))
    for step in range(2):
        for k, src in ((0, "x"), (1, "y")):
            nxt = state.add_access("d")
            t = state.add_tasklet(f"acc{step}{k}", {"_d": None, "_v": None}, {"_o": None}, "_o = _d + _v")
            state.add_edge(prev, None, t, "_d", dace.Memlet(f"d[{k}]"))
            state.add_edge(state.add_read(src), None, t, "_v", dace.Memlet(f"{src}[i]"))
            state.add_edge(t, "_o", nxt, None, dace.Memlet(f"d[{k}]"))
            prev = nxt
    diff = state.add_tasklet("diff", {"_p": None, "_q": None}, {"_o": None}, "_o = _p - 3 * _q")
    state.add_edge(prev, None, diff, "_p", dace.Memlet("d[0]"))
    state.add_edge(prev, None, diff, "_q", dace.Memlet("d[1]"))
    state.add_edge(diff, "_o", state.add_write("o"), None, dace.Memlet("o[i]"))

    sdfg = dace.SDFG("accumulate")
    for name in ("X", "Y", "OUT"):
        sdfg.add_array(name, [N], dace.float64)
    outer = sdfg.add_state()
    me, mx = outer.add_map("points", {"i": f"0:{N}"})
    nsdfg = outer.add_nested_sdfg(body, {"x", "y"}, {"o"}, symbol_mapping={"i": "i"})
    for o, inner in (("X", "x"), ("Y", "y")):
        outer.add_memlet_path(outer.add_read(o), me, nsdfg, dst_conn=inner, memlet=dace.Memlet(f"{o}[0:{N}]"))
    outer.add_memlet_path(nsdfg, mx, outer.add_write("OUT"), src_conn="o", memlet=dace.Memlet(f"OUT[0:{N}]"))
    return sdfg


def test_a_read_through_a_later_node_survives_dead_dataflow_elimination():
    """Split without the order, ``d_0``'s last write lands in a sink node and its read in a separate source node:
    dead-dataflow elimination (in any later simplification) then drops the writes and the read sees garbage."""
    x, y = np.arange(N, dtype=np.float64), np.arange(N, dtype=np.float64) ** 2
    expected = 2 * x - 3 * 2 * y
    sdfg = _accumulate_body()
    assert sorted(name.split("/")[-1] for name in scalarise_small_arrays(sdfg)) == ["d"]
    sdfg.simplify()
    out = np.zeros(N)
    sdfg(X=x.copy(), Y=y.copy(), OUT=out)
    np.testing.assert_array_equal(out, expected)
