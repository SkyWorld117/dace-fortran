"""Tests of unrolling the small maps inside kernels (straight.py, CPU). Run: python -m pytest test_straight.py"""
import numpy as np

import dace
from dace import nodes

from dace_fortran.substitute.straight import unroll_small_maps

N = 16


def _maps(sdfg: dace.SDFG) -> list:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def _inner_map_sdfg(m: int = 3, local: bool = False) -> dace.SDFG:
    """For every point i (a kernel map): a[k] = 2 * b[i, k] for k < m (an inner map; with ``local``, through a
    scratch scalar the inner scope alone uses), then out[i] = sum_k a[k] * (k + 1)."""
    sdfg = dace.SDFG(f"inner_{m}_{int(local)}")
    sdfg.add_array("b", [N, m], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_array("a", [m], dace.float64, transient=True)
    if local:
        sdfg.add_scalar("tmp", dace.float64, transient=True)
    state = sdfg.add_state()
    b, out, a = state.add_read("b"), state.add_write("out"), state.add_access("a")
    me, mx = state.add_map("points", {"i": f"0:{N}"})
    ie, ix = state.add_map("weights", {"k": f"0:{m}"}, schedule=dace.ScheduleType.Sequential)
    if local:
        t1 = state.add_tasklet("half", {"_b": None}, {"_t": None}, "_t = _b")
        tmp = state.add_access("tmp")
        t2 = state.add_tasklet("double", {"_t": None}, {"_a": None}, "_a = 2 * _t")
        state.add_memlet_path(b, me, ie, t1, dst_conn="_b", memlet=dace.Memlet("b[i, k]"))
        state.add_edge(t1, "_t", tmp, None, dace.Memlet("tmp[0]"))
        state.add_edge(tmp, None, t2, "_t", dace.Memlet("tmp[0]"))
        state.add_memlet_path(t2, ix, a, src_conn="_a", memlet=dace.Memlet("a[k]"))
    else:
        t = state.add_tasklet("double", {"_b": None}, {"_a": None}, "_a = 2 * _b")
        state.add_memlet_path(b, me, ie, t, dst_conn="_b", memlet=dace.Memlet("b[i, k]"))
        state.add_memlet_path(t, ix, a, src_conn="_a", memlet=dace.Memlet("a[k]"))
    code = "_o = " + " + ".join(f"_a[{k}] * {k + 1}" for k in range(m))
    dot = state.add_tasklet("dot", {"_a": None}, {"_o": None}, code)
    state.add_edge(a, None, dot, "_a", dace.Memlet(f"a[0:{m}]"))
    state.add_memlet_path(dot, mx, out, src_conn="_o", memlet=dace.Memlet("out[i]"))
    return sdfg


def _run_inner(sdfg: dace.SDFG, m: int) -> np.ndarray:
    b = np.arange(N * m, dtype=np.float64).reshape(N, m).copy()
    out = np.zeros(N)
    sdfg(b=b, out=out)
    return out


def test_a_small_map_inside_a_kernel_unrolls():
    sdfg = _inner_map_sdfg()
    expected = _run_inner(_inner_map_sdfg(), 3)
    assert unroll_small_maps(sdfg) == 1
    sdfg.validate()
    assert [e.map.label for e in _maps(sdfg)] == ["points"]
    np.testing.assert_array_equal(_run_inner(sdfg, 3), expected)


def test_a_scope_local_container_gets_one_copy_per_iteration():
    sdfg = _inner_map_sdfg(local=True)
    expected = _run_inner(_inner_map_sdfg(local=True), 3)
    assert unroll_small_maps(sdfg) == 1
    sdfg.validate()
    assert "tmp" not in sdfg.arrays and len([n for n in sdfg.arrays if n.startswith("tmp_u")]) == 3
    np.testing.assert_array_equal(_run_inner(sdfg, 3), expected)


def test_kernels_and_long_maps_stay():
    sdfg = _inner_map_sdfg(m=9)
    assert unroll_small_maps(sdfg, limit=8) == 0
    assert len(_maps(sdfg)) == 2
    top = _inner_map_sdfg(m=3)
    unroll_small_maps(top, limit=64)
    assert [e.map.label for e in _maps(top)] == ["points"]  # the kernel over 16 points is no candidate
