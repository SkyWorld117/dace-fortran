"""wcr_accumulations: a loop that adds to the cells on both sides of each face becomes a map with atomic sums, with
the sequential loop's result; a loop that also reads the array elsewhere is left alone. The loops are built in the
form dace-fortran emits: one tasklet reads and writes the array element itself.
Run: PYTHONPATH=../../dace:../../dace-fortran python -m pytest test_wcr_accumulations.py"""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap

from dace_fortran.substitute.wcr_accumulations import accumulations_to_wcr

N = 16


def _update(state, index, op):
    """``rhs[index] = rhs[index] op f[j]``."""
    rhs_in, f, rhs_out = state.add_read("rhs"), state.add_read("f"), state.add_write("rhs")
    t = state.add_tasklet("update", {"r", "g"}, {"o"}, f"o = r {op} g")
    state.add_edge(rhs_in, None, t, "r", dace.Memlet(f"rhs[{index}]"))
    state.add_edge(f, None, t, "g", dace.Memlet("f[j]"))
    state.add_edge(t, "o", rhs_out, None, dace.Memlet(f"rhs[{index}]"))


def _face_loop(read_elsewhere: bool) -> tuple[dace.SDFG, LoopRegion]:
    sdfg = dace.SDFG("faces")
    sdfg.add_array("rhs", [N + 1], dace.float64)
    sdfg.add_array("f", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    loop = LoopRegion("faces", f"j < {N}", "j", "j = 0", "j = j + 1")
    sdfg.add_node(loop, is_start_block=True)
    left = loop.add_state("left", is_start_block=True)
    _update(left, "j", "+")
    right = loop.add_state_after(left, "right")
    _update(right, "j + 1", "-")
    if read_elsewhere:
        use = loop.add_state_after(right, "use")
        t = use.add_tasklet("use", {"r"}, {"o"}, "o = r")
        use.add_edge(use.add_read("rhs"), None, t, "r", dace.Memlet("rhs[j]"))
        use.add_edge(t, "o", use.add_write("out"), None, dace.Memlet("out[j]"))
    sdfg.validate()
    return sdfg, loop


def test_the_face_loop_becomes_a_map_with_the_sequential_result():
    rng = np.random.default_rng(0)
    rhs0, f = rng.random(N + 1), rng.random(N)
    expected = rhs0.copy()
    for j in range(N):
        expected[j] += f[j]
        expected[j + 1] -= f[j]
    sdfg, loop = _face_loop(read_elsewhere=False)
    assert not LoopToMap.can_be_applied_to(sdfg, loop=loop)
    assert len(accumulations_to_wcr(sdfg)) == 2
    assert LoopToMap.can_be_applied_to(sdfg, loop=loop)
    LoopToMap.apply_to(sdfg, loop=loop)
    rhs = rhs0.copy()
    sdfg(rhs=rhs, f=f, out=np.zeros(N))
    np.testing.assert_allclose(rhs, expected)


def test_a_loop_that_also_reads_the_array_is_left_alone():
    sdfg, _loop = _face_loop(read_elsewhere=True)
    assert accumulations_to_wcr(sdfg) == []


def _scatter_loop() -> tuple[dace.SDFG, LoopRegion]:
    """``c = idx[j]`` on an interstate edge, then ``rhs[c] = rhs[c] + f[j]`` (the support points of a source term)."""
    sdfg = dace.SDFG("scatter")
    sdfg.add_array("rhs", [N + 1], dace.float64)
    sdfg.add_array("f", [N], dace.float64)
    sdfg.add_array("idx", [N], dace.int64)
    sdfg.add_symbol("c", dace.int64)
    loop = LoopRegion("points", f"j < {N}", "j", "j = 0", "j = j + 1")
    sdfg.add_node(loop, is_start_block=True)
    pick = loop.add_state("pick", is_start_block=True)
    add = loop.add_state("add")
    loop.add_edge(pick, add, dace.InterstateEdge(assignments={"c": "idx[j]"}))
    _update(add, "c", "+")
    sdfg.validate()
    return sdfg, loop


def test_a_scatter_through_a_loaded_index_becomes_a_map_with_atomic_sums():
    rng = np.random.default_rng(1)
    rhs0, f = rng.random(N + 1), rng.random(N)
    idx = rng.integers(0, N + 1, N)  # with repeats
    expected = rhs0.copy()
    np.add.at(expected, idx, f)
    sdfg, loop = _scatter_loop()
    assert not LoopToMap.can_be_applied_to(sdfg, loop=loop)
    assert accumulations_to_wcr(sdfg) == ["rhs[c]"]
    assert LoopToMap.can_be_applied_to(sdfg, loop=loop)
    LoopToMap.apply_to(sdfg, loop=loop)
    rhs = rhs0.copy()
    sdfg(rhs=rhs, f=f, idx=idx)
    np.testing.assert_allclose(rhs, expected)
