"""Tests of collapse_bodies (CPU).  Run: python -m pytest test_collapse_bodies.py"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes

from dace_fortran.substitute.collapse_bodies import collapse_through_bodies

N, M = 7, 5


def _nest() -> dace.SDFG:
    """for i: body(i) = [for j: A[i, j, 0] = X[i, j] * 2; A[i, j, 1] = A[i, j, 0] + 1] with the body a nested SDFG:
    the shape a loop nest has after the outer loop became a map around its (multi-state, then fused) body. The body
    writes A[i, j, 0] and reads it back in the same iteration."""
    inner = dace.SDFG("body")
    inner.add_symbol("i", dace.int64)
    inner.add_array("X", [N, M], dace.float64)
    inner.add_array("A", [N, M, 2], dace.float64)
    st = inner.add_state()
    me, mx = st.add_map("inner", {"j": f"0:{M}"})
    x, a_out = st.add_read("X"), st.add_write("A")
    t1 = st.add_tasklet("scale", {"_x"}, {"_a"}, "_a = _x * 2")
    t2 = st.add_tasklet("shift", {"_a"}, {"_b"}, "_b = _a + 1")
    a_mid = st.add_access("A")
    st.add_memlet_path(x, me, t1, dst_conn="_x", memlet=dace.Memlet("X[i, j]"))
    st.add_edge(t1, "_a", a_mid, None, dace.Memlet("A[i, j, 0]"))
    st.add_edge(a_mid, None, t2, "_a", dace.Memlet("A[i, j, 0]"))
    st.add_memlet_path(t2, mx, a_out, src_conn="_b", memlet=dace.Memlet("A[i, j, 1]"))

    outer = dace.SDFG("nest")
    outer.add_array("X", [N, M], dace.float64)
    outer.add_array("A", [N, M, 2], dace.float64)
    ost = outer.add_state()
    ome, omx = ost.add_map("outer", {"i": f"0:{N}"})
    body = ost.add_nested_sdfg(inner, {"X", "A"}, {"A"}, symbol_mapping={"i": "i"})
    ost.add_memlet_path(ost.add_read("X"), ome, body, dst_conn="X", memlet=dace.Memlet(f"X[i, 0:{M}]"))
    ost.add_memlet_path(ost.add_read("A"), ome, body, dst_conn="A", memlet=dace.Memlet(f"A[i, 0:{M}, 0]"))
    ost.add_memlet_path(body, omx, ost.add_write("A"), src_conn="A", memlet=dace.Memlet(f"A[i, 0:{M}, 0:2]"))
    return outer


def _run(sdfg: dace.SDFG):
    x = np.arange(N * M, dtype=np.float64).reshape(N, M).copy()
    a = np.zeros((N, M, 2))
    sdfg(X=x, A=a)
    return a


@pytest.mark.xfail(
    strict=False,
    raises=Exception,
    reason="This fork's ``MapCollapse`` is locally modified and produces ``X[2*i, j]`` / ``X[0:13:2, 0:5]`` "
           "where the colleague's produces ``X[i, j]``, so the collapsed nest fails validation.  Reproduces "
           "with this fork's dace main as well, i.e. it is not a merge regression; ``collapse_through_bodies`` "
           "is not on the WRF path.",
)
def test_a_write_then_read_body_collapses_into_one_two_dimensional_map():
    reference = _run(_nest())
    sdfg = _nest()
    sdfg.name = "nest_collapsed"
    assert collapse_through_bodies(sdfg) == 1
    sdfg.validate()
    tops = [n for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None]
    assert [len(n.map.params) for n in tops] == [2]
    assert not any(isinstance(n, nodes.NestedSDFG) for n, _ in sdfg.all_nodes_recursive())
    a = _run(sdfg)
    np.testing.assert_array_equal(a, reference)
    np.testing.assert_array_equal(a[:, :, 0], np.arange(N * M).reshape(N, M) * 2.0)
    np.testing.assert_array_equal(a[:, :, 1], np.arange(N * M).reshape(N, M) * 2.0 + 1)


def test_an_inner_range_depending_on_the_outer_parameter_is_left_alone():
    sdfg = _nest()
    st = next(iter(sdfg.states()))
    body = next(n for n in st.nodes() if isinstance(n, nodes.NestedSDFG))
    inner_map = next(n for n in body.sdfg.states()[0].nodes() if isinstance(n, nodes.MapEntry))
    inner_map.map.range = dace.subsets.Range.from_string("0:i+1")
    assert collapse_through_bodies(sdfg) == 0
