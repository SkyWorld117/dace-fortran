"""Tests of the opt-in numerics rewrites (CPU).  Run: python -m pytest test_numerics.py"""
import numpy as np

import dace
from dace.libraries.standard import Reduce

from dace_fortran.substitute.numerics import combine_normalised_quotients


def _weights_sdfg(m: int, mapped: bool = False) -> dace.SDFG:
    """For every point i: a[k] = c[k] / (b[i, k] ** 2) (or a product with a quotient factor), s = sum(a), w[k] = a[k] / s,
    out[i] = sum_k w[k] * v[i, k]. The reduction and the divisions are in one state of the map's body, as a frontend
    lowers a small weighted average."""
    sdfg = dace.SDFG(f"weights_{m}_{int(mapped)}")
    sdfg.add_array("b", [16, m], dace.float64)
    sdfg.add_array("v", [16, m], dace.float64)
    sdfg.add_array("c", [m], dace.float64)
    sdfg.add_array("out", [16], dace.float64)
    sdfg.add_array("a", [m], dace.float64, transient=True)
    sdfg.add_array("w", [m], dace.float64, transient=True)
    sdfg.add_scalar("s", dace.float64, transient=True)
    state = sdfg.add_state()
    # Sequential: the small transients live outside the map, as in a kernel body each thread owns its copy.
    me, mx = state.add_map("points", {"i": "0:16"}, schedule=dace.ScheduleType.Sequential)
    a_node, w_node, s_node = state.add_access("a"), state.add_access("w"), state.add_access("s")
    b, v, c, out = state.add_read("b"), state.add_read("v"), state.add_read("c"), state.add_write("out")
    for k in range(m):
        if mapped:
            code = "_out = (_c * (1.0 + _c)) * (_b / (_c * _c + _b * (1.0 - 2.0 * _c)))"
        else:
            code = "_out = (_c / (_b ** 2))"
        t = state.add_tasklet(f"quot{k}", {"_c", "_b"}, {"_out"}, code)
        state.add_memlet_path(c, me, t, dst_conn="_c", memlet=dace.Memlet(f"c[{k}]"))
        state.add_memlet_path(b, me, t, dst_conn="_b", memlet=dace.Memlet(f"b[i, {k}]"))
        state.add_edge(t, "_out", a_node, None, dace.Memlet(f"a[{k}]"))
    red = Reduce("sum", "lambda a, b: a + b", None, identity=0)
    state.add_node(red)
    state.add_edge(a_node, None, red, None, dace.Memlet(f"a[0:{m}]"))
    state.add_edge(red, None, s_node, None, dace.Memlet("s[0]"))
    for k in range(m):
        t = state.add_tasklet(f"div{k}", {"_a", "_s"}, {"_w"}, "_w = (_a / _s)")
        state.add_edge(a_node, None, t, "_a", dace.Memlet(f"a[{k}]"))
        state.add_edge(s_node, None, t, "_s", dace.Memlet("s[0]"))
        state.add_edge(t, "_w", w_node, None, dace.Memlet(f"w[{k}]"))
    dot = state.add_tasklet("dot", {"_w", "_v"}, {"_o"}, "_o = " + " + ".join(f"_w[{k}] * _v[{k}]" for k in range(m)))
    state.add_edge(w_node, None, dot, "_w", dace.Memlet(f"w[0:{m}]"))
    state.add_memlet_path(v, me, dot, dst_conn="_v", memlet=dace.Memlet(f"v[i, 0:{m}]"))
    state.add_memlet_path(dot, mx, out, src_conn="_o", memlet=dace.Memlet("out[i]"))
    return sdfg


def _run(sdfg, b, v, c):
    out = np.zeros(16)
    sdfg(b=b.copy(), v=v.copy(), c=c.copy(), out=out)
    return out


def _check(m: int, mapped: bool):
    rng = np.random.default_rng(m)
    b = rng.uniform(0.1, 2.0, (16, m))
    v = rng.uniform(-1.0, 1.0, (16, m))
    c = rng.uniform(0.1, 0.9, m)
    reference = _run(_weights_sdfg(m, mapped), b, v, c)
    sdfg = _weights_sdfg(m, mapped)
    sdfg.name = sdfg.name + "_combined"  # its own build: a same-named library loaded earlier in the process would be reused
    assert combine_normalised_quotients(sdfg) == 1
    sdfg.validate()
    divisions = sum(n.code.as_string.count("/") for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert divisions == 1  # one reciprocal of the common denominator
    assert not any(isinstance(n, Reduce) for n, _ in sdfg.all_nodes_recursive())
    np.testing.assert_allclose(_run(sdfg, b, v, c), reference, rtol=1e-13, atol=1e-15)


def test_two_quotients_share_one_reciprocal():
    _check(2, mapped=False)


def test_three_quotients_share_one_reciprocal():
    _check(3, mapped=False)


def test_a_product_with_a_quotient_factor_counts_as_a_quotient():
    _check(2, mapped=True)


def test_an_array_read_elsewhere_keeps_its_writers():
    sdfg = _weights_sdfg(2)
    state = sdfg.states()[0]
    a_node = next(n for n in state.data_nodes() if n.data == "a")
    sdfg.add_array("a_copy", [2], dace.float64)
    sdfg.arrays["a_copy"].transient = False
    keep = state.add_write("a_copy")
    state.add_edge(a_node, None, keep, None, dace.Memlet("a[0:2]"))
    assert combine_normalised_quotients(sdfg) == 1
    assert sum(1 for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label.startswith("quot")) == 2


def test_normalisations_of_scalarised_values_share_one_reciprocal():
    """After scalarise.py has split the small arrays into one container per element, the normalisation is a tasklet
    adding scalars and a division per scalar; it is rewritten the same way."""
    from dace_fortran.substitute.scalarise import scalarise_small_arrays

    for m, mapped in ((2, False), (3, False), (3, True)):
        rng = np.random.default_rng(10 + m)
        b = rng.uniform(0.1, 2.0, (16, m))
        v = rng.uniform(-1.0, 1.0, (16, m))
        c = rng.uniform(0.1, 0.9, m)
        reference = _run(_weights_sdfg(m, mapped), b, v, c)
        sdfg = _weights_sdfg(m, mapped)
        sdfg.name = sdfg.name + "_scalars"
        split = scalarise_small_arrays(sdfg)
        assert any(key.endswith("/a") for key in split), split
        assert combine_normalised_quotients(sdfg) == 1
        sdfg.validate()
        divisions = sum(n.code.as_string.count("/") for n, _ in sdfg.all_nodes_recursive()
                        if isinstance(n, dace.nodes.Tasklet))
        assert divisions == 1
        np.testing.assert_allclose(_run(sdfg, b, v, c), reference, rtol=1e-13, atol=1e-15)


def _inline_sdfg(m: int, mapped: bool) -> dace.SDFG:
    """For every point i: a_k = c[k] / (b[i, k] ** 2) (or a product with a quotient factor), each in its own scalar,
    w_k = a_k / (a_0 + ... + a_{m-1}) with the sum written out in each dividing tasklet, out[i] = sum_k w_k v[i, k]:
    a normalisation as scalarise.py leaves it once small sums are expanded."""
    sdfg = dace.SDFG(f"inline_{m}_{int(mapped)}")
    sdfg.add_array("b", [16, m], dace.float64)
    sdfg.add_array("v", [16, m], dace.float64)
    sdfg.add_array("c", [m], dace.float64)
    sdfg.add_array("out", [16], dace.float64)
    for k in range(m):
        sdfg.add_scalar(f"a{k}", dace.float64, transient=True)
        sdfg.add_scalar(f"w{k}", dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map("points", {"i": "0:16"}, schedule=dace.ScheduleType.Sequential)
    b, v, c, out = state.add_read("b"), state.add_read("v"), state.add_read("c"), state.add_write("out")
    a_nodes = [state.add_access(f"a{k}") for k in range(m)]
    w_nodes = [state.add_access(f"w{k}") for k in range(m)]
    for k in range(m):
        code = ("_out = (_c * (1.0 + _c)) * (_b / (_c * _c + _b * (1.0 - 2.0 * _c)))" if mapped
                else "_out = (_c / (_b ** 2))")
        t = state.add_tasklet(f"quot{k}", {"_c", "_b"}, {"_out"}, code)
        state.add_memlet_path(c, me, t, dst_conn="_c", memlet=dace.Memlet(f"c[{k}]"))
        state.add_memlet_path(b, me, t, dst_conn="_b", memlet=dace.Memlet(f"b[i, {k}]"))
        state.add_edge(t, "_out", a_nodes[k], None, dace.Memlet(f"a{k}[0]"))
    for k in range(m):
        conns = {f"_s{j}" for j in range(m)} | {"_x"}
        t = state.add_tasklet(f"div{k}", conns, {"_w"}, "_w = (_x / (" + " + ".join(f"_s{j}" for j in range(m)) + "))")
        state.add_edge(a_nodes[k], None, t, "_x", dace.Memlet(f"a{k}[0]"))
        for j in range(m):
            state.add_edge(a_nodes[j], None, t, f"_s{j}", dace.Memlet(f"a{j}[0]"))
        state.add_edge(t, "_w", w_nodes[k], None, dace.Memlet(f"w{k}[0]"))
    dot = state.add_tasklet("dot", {f"_w{k}" for k in range(m)} | {"_v"}, {"_o"},
                            "_o = " + " + ".join(f"_w{k} * _v[{k}]" for k in range(m)))
    for k in range(m):
        state.add_edge(w_nodes[k], None, dot, f"_w{k}", dace.Memlet(f"w{k}[0]"))
    state.add_memlet_path(v, me, dot, dst_conn="_v", memlet=dace.Memlet(f"v[i, 0:{m}]"))
    state.add_memlet_path(dot, mx, out, src_conn="_o", memlet=dace.Memlet("out[i]"))
    return sdfg


def test_a_sum_written_out_in_each_division_shares_one_reciprocal():
    for m, mapped in ((2, False), (3, False), (3, True)):
        rng = np.random.default_rng(20 + m)
        b = rng.uniform(0.1, 2.0, (16, m))
        v = rng.uniform(-1.0, 1.0, (16, m))
        c = rng.uniform(0.1, 0.9, m)
        reference = _run(_inline_sdfg(m, mapped), b, v, c)
        sdfg = _inline_sdfg(m, mapped)
        sdfg.name = sdfg.name + "_combined"
        assert combine_normalised_quotients(sdfg) == 1
        sdfg.validate()
        divisions = sum(n.code.as_string.count("/") for n, _ in sdfg.all_nodes_recursive()
                        if isinstance(n, dace.nodes.Tasklet))
        assert divisions == 1
        np.testing.assert_allclose(_run(sdfg, b, v, c), reference, rtol=1e-13, atol=1e-15)


def test_an_inline_sum_over_other_terms_is_not_a_normalisation():
    """Dividing tasklets whose sums differ (one leaves a term out) are left alone."""
    sdfg = _inline_sdfg(3, False)
    state = sdfg.states()[0]
    div0 = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label == "div0")
    div0.code = dace.properties.CodeBlock("_w = (_x / (_s0 + _s1 + _s2 + 1.0))")
    assert combine_normalised_quotients(sdfg) == 0
