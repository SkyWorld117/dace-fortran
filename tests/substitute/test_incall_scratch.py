"""Tests of the in-call scratch analysis (CPU, no compilation).  Run: python -m pytest test_incall_scratch.py"""
import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.properties import CodeBlock

from dace_fortran.substitute.incall_scratch import uncovered_reads

N = 10


def _sdfg(name: str, *arrays: str, size: int = N) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    for a in arrays:
        sdfg.add_array(a, [size], dace.float64)
    return sdfg


def _write(state, array: str, subset: str, value: str = "1.0"):
    """A tasklet writing ``array[subset]`` (one element)."""
    t = state.add_tasklet(f"w_{array}", {}, {"o"}, f"o = {value}")
    state.add_edge(t, "o", state.add_write(array), None, dace.Memlet(f"{array}[{subset}]"))


def _copy(state, src: str, dst: str, src_subset: str, dst_subset: str):
    """A tasklet reading ``src[src_subset]`` and writing ``dst[dst_subset]`` (elements)."""
    t = state.add_tasklet(f"c_{src}_{dst}", {"i"}, {"o"}, "o = i")
    state.add_edge(state.add_read(src), None, t, "i", dace.Memlet(f"{src}[{src_subset}]"))
    state.add_edge(t, "o", state.add_write(dst), None, dace.Memlet(f"{dst}[{dst_subset}]"))


def _mapped(state, src: str, dst: str, rng: str, src_index: str = "i", dst_index: str = "i"):
    """A map over ``rng`` copying ``src[src_index]`` to ``dst[dst_index]``."""
    state.add_mapped_tasklet("m", {"i": rng}, {"x": dace.Memlet(f"{src}[{src_index}]")}, "y = x",
                             {"y": dace.Memlet(f"{dst}[{dst_index}]")}, external_edges=True)


def _loop(sdfg_or_region, var: str, lo: str, end: str, label: str = "loop") -> LoopRegion:
    loop = LoopRegion(label, f"{var} < {end}", var, f"{var} = {lo}", f"{var} = {var} + 1")
    sdfg_or_region.add_node(loop)
    return loop


def _chain(region, *blocks):
    for a, b in zip(blocks, blocks[1:]):
        region.add_edge(a, b, dace.InterstateEdge())


def test_write_then_read_is_covered():
    sdfg = _sdfg("wr", "a", "b", "out")
    s1 = sdfg.add_state(is_start_block=True)
    _mapped(s1, "b", "a", f"0:{N}")
    s2 = sdfg.add_state_after(s1)
    _mapped(s2, "a", "out", f"0:{N}")
    found = uncovered_reads(sdfg, ["a", "b"])
    assert found["a"] == []
    assert found["b"], "b is read and never written"


def test_partial_write_leaves_the_rest_uncovered():
    sdfg = _sdfg("partial", "a", "b", "out")
    s1 = sdfg.add_state(is_start_block=True)
    _mapped(s1, "b", "a", f"0:{N - 1}")
    s2 = sdfg.add_state_after(s1)
    _mapped(s2, "a", "out", f"0:{N}")
    assert uncovered_reads(sdfg, ["a"])["a"]


def test_union_of_writes_covers_a_read():
    """An interior and two halos written separately cover the whole array."""
    sdfg = _sdfg("halos", "a", "b", "out")
    s1 = sdfg.add_state(is_start_block=True)
    _mapped(s1, "b", "a", f"1:{N - 1}")
    s2 = sdfg.add_state_after(s1)
    _write(s2, "a", "0")
    _write(s2, "a", f"{N - 1}")
    s3 = sdfg.add_state_after(s2)
    _mapped(s3, "a", "out", f"0:{N}")
    assert uncovered_reads(sdfg, ["a"])["a"] == []


def test_read_before_write_in_one_state():
    sdfg = _sdfg("rbw", "a", "out")
    s = sdfg.add_state()
    _copy(s, "a", "out", "0", "0")
    _write(s, "a", "0")
    assert uncovered_reads(sdfg, ["a"])["a"]


def test_write_reaching_the_read_in_one_state():
    sdfg = _sdfg("chain", "a", "b", "out")
    s = sdfg.add_state()
    t1 = s.add_tasklet("t1", {"i"}, {"o"}, "o = i")
    t2 = s.add_tasklet("t2", {"i"}, {"o"}, "o = i")
    a = s.add_access("a")
    s.add_edge(s.add_read("b"), None, t1, "i", dace.Memlet("b[0]"))
    s.add_edge(t1, "o", a, None, dace.Memlet("a[0]"))
    s.add_edge(a, None, t2, "i", dace.Memlet("a[0]"))
    s.add_edge(t2, "o", s.add_write("out"), None, dace.Memlet("out[0]"))
    assert uncovered_reads(sdfg, ["a"])["a"] == []


def test_loop_writes_expand_over_the_iteration_space():
    sdfg = _sdfg("loop", "a", "b", "out")
    loop = _loop(sdfg, "i", "0", str(N))
    body = loop.add_state("body", is_start_block=True)
    _copy(body, "b", "a", "i", "i")
    after = sdfg.add_state("after")
    sdfg.add_edge(loop, after, dace.InterstateEdge())
    _mapped(after, "a", "out", f"0:{N}")
    assert uncovered_reads(sdfg, ["a"])["a"] == []


def test_loop_that_may_not_run_writes_nothing_certain():
    sdfg = _sdfg("emptyloop", "a", "b", "out")
    sdfg.add_symbol("n", dace.int64)
    loop = _loop(sdfg, "i", "0", "n")
    body = loop.add_state("body", is_start_block=True)
    _copy(body, "b", "a", "0", "0")
    after = sdfg.add_state("after")
    sdfg.add_edge(loop, after, dace.InterstateEdge())
    _copy(after, "a", "out", "0", "0")
    assert uncovered_reads(sdfg, ["a"])["a"]


def test_loop_carried_read_is_uncovered():
    """a[i] = a[i - 1]: the first iteration reads an element from before (and others are not proven either)."""
    sdfg = _sdfg("carried", "a", "out")
    loop = _loop(sdfg, "i", "1", str(N))
    body = loop.add_state("body", is_start_block=True)
    _copy(body, "a", "a", "i - 1", "i")
    assert uncovered_reads(sdfg, ["a"])["a"]


def test_read_in_the_same_iteration_is_covered():
    sdfg = _sdfg("sameiter", "a", "b", "out")
    loop = _loop(sdfg, "i", "0", str(N))
    w = loop.add_state("w", is_start_block=True)
    _copy(w, "b", "a", "i", "i")
    r = loop.add_state("r")
    loop.add_edge(w, r, dace.InterstateEdge())
    _copy(r, "a", "out", "i", "i")
    assert uncovered_reads(sdfg, ["a"])["a"] == []


def _branches(sdfg, with_else: bool):
    sdfg.add_symbol("c", dace.int64)
    cond = ConditionalBlock("if")
    sdfg.add_node(cond, is_start_block=True)
    then = ControlFlowRegion("then", sdfg=sdfg)
    _write(then.add_state("t", is_start_block=True), "a", "0")
    cond.add_branch(CodeBlock("c > 0"), then)
    if with_else:
        other = ControlFlowRegion("else", sdfg=sdfg)
        _write(other.add_state("e", is_start_block=True), "a", "0", "2.0")
        cond.add_branch(None, other)
    after = sdfg.add_state("after")
    sdfg.add_edge(cond, after, dace.InterstateEdge())
    _copy(after, "a", "out", "0", "0")


def test_write_in_one_branch_only_is_not_certain():
    sdfg = _sdfg("ifonly", "a", "out")
    _branches(sdfg, with_else=False)
    assert uncovered_reads(sdfg, ["a"])["a"]


def test_write_in_every_branch_is_certain():
    sdfg = _sdfg("ifelse", "a", "out")
    _branches(sdfg, with_else=True)
    assert uncovered_reads(sdfg, ["a"])["a"] == []


def test_symbols_assigned_on_edges_take_their_value():
    """a[k] with k = 3 covers a read of a[3], also after k is reassigned."""
    sdfg = _sdfg("sym", "a", "out")
    sdfg.add_symbol("k", dace.int64)
    s0 = sdfg.add_state(is_start_block=True)
    s1 = sdfg.add_state("w")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "3"}))
    _write(s1, "a", "k")
    s2 = sdfg.add_state("r")
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"k": "5"}))
    _copy(s2, "a", "out", "3", "0")
    assert uncovered_reads(sdfg, ["a"])["a"] == []


def test_unknown_symbol_values_are_not_trusted():
    """k assigned from data: a[k] proves nothing about a[3]."""
    sdfg = _sdfg("dsym", "a", "out", "d")
    sdfg.add_symbol("k", dace.int64)
    s0 = sdfg.add_state(is_start_block=True)
    s1 = sdfg.add_state("w")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "d[0]"}))
    _write(s1, "a", "k")
    s2 = sdfg.add_state("r")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    _copy(s2, "a", "out", "3", "0")
    assert uncovered_reads(sdfg, ["a"])["a"]


def test_elements_of_a_constant_copied_in_are_known():
    """idx = (/3, 1/) copied from a constant; k = idx[0]; a[k] then covers a read of a[3]."""
    sdfg = _sdfg("const", "a", "out")
    sdfg.add_array("idx", [2], dace.int64)
    sdfg.add_constant("lit", np.array([3, 1], dtype=np.int64), dace.data.Array(dace.int64, [2]))
    sdfg.add_symbol("k", dace.int64)
    s0 = sdfg.add_state(is_start_block=True)
    s0.add_edge(s0.add_read("lit"), None, s0.add_write("idx"), None, dace.Memlet("lit[0:2]", other_subset="0:2"))
    s1 = sdfg.add_state("w")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "idx[0]"}))
    _write(s1, "a", "k")
    s2 = sdfg.add_state("r")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    _copy(s2, "a", "out", "3", "0")
    assert uncovered_reads(sdfg, ["a"])["a"] == []


def test_conditions_reading_an_array_are_reads():
    sdfg = _sdfg("condread", "a", "out")
    s0 = sdfg.add_state(is_start_block=True)
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(condition="a[0] > 0"))
    assert uncovered_reads(sdfg, ["a"])["a"]


def test_map_reading_a_neighbour_of_its_own_write_is_uncovered():
    """Inside one map, a[i + 1] read after a[i] was written in the same iteration: another iteration's element."""
    sdfg = _sdfg("mapnbr", "a", "b", "out", size=N + 1)
    s = sdfg.add_state()
    me, mx = s.add_map("m", {"i": f"0:{N}"})
    t1 = s.add_tasklet("t1", {"x"}, {"y"}, "y = x")
    t2 = s.add_tasklet("t2", {"x"}, {"y"}, "y = x")
    inner = s.add_access("a")
    b, out, a_out = s.add_read("b"), s.add_write("out"), s.add_write("a")
    s.add_memlet_path(b, me, t1, dst_conn="x", memlet=dace.Memlet("b[i]"))
    s.add_edge(t1, "y", inner, None, dace.Memlet("a[i]"))
    s.add_edge(inner, None, t2, "x", dace.Memlet("a[i + 1]"))
    s.add_memlet_path(t2, mx, out, src_conn="y", memlet=dace.Memlet("out[i]"))
    s.add_memlet_path(inner, mx, a_out, memlet=dace.Memlet("a[i]"))
    assert uncovered_reads(sdfg, ["a"])["a"]


def test_map_writes_cover_a_later_map():
    sdfg = _sdfg("maps", "a", "b", "out")
    s = sdfg.add_state()
    s.add_mapped_tasklet("w", {"i": f"0:{N}"}, {"x": dace.Memlet("b[i]")}, "y = x", {"y": dace.Memlet("a[i]")},
                         external_edges=True)
    a = next(n for n in s.data_nodes() if n.data == "a")
    me, mx = s.add_map("r", {"j": f"0:{N}"})
    t = s.add_tasklet("t", {"x"}, {"y"}, "y = x")
    s.add_memlet_path(a, me, t, dst_conn="x", memlet=dace.Memlet("a[j]"))
    s.add_memlet_path(t, mx, s.add_write("out"), src_conn="y", memlet=dace.Memlet("out[j]"))
    assert uncovered_reads(sdfg, ["a"])["a"] == []


def test_write_conflict_resolution_reads_its_output():
    sdfg = _sdfg("wcr", "a", "b")
    s = sdfg.add_state()
    t = s.add_tasklet("t", {"x"}, {"y"}, "y = x")
    s.add_edge(s.add_read("b"), None, t, "x", dace.Memlet("b[0]"))
    s.add_edge(t, "y", s.add_write("a"), None, dace.Memlet("a[0]", wcr="lambda p, q: p + q"))
    assert uncovered_reads(sdfg, ["a"])["a"]


def _nested(write_first: bool) -> dace.SDFG:
    inner = dace.SDFG("inner")
    inner.add_array("x", [N], dace.float64)
    inner.add_array("y", [N], dace.float64)
    st = inner.add_state()
    if write_first:
        st.add_mapped_tasklet("w", {"i": f"0:{N}"}, {}, "o = 1.0", {"o": dace.Memlet("x[i]")}, external_edges=True)
        xa = next(n for n in st.data_nodes() if n.data == "x")
        me, mx = st.add_map("r", {"j": f"0:{N}"})
        t = st.add_tasklet("t", {"v"}, {"o"}, "o = v")
        st.add_memlet_path(xa, me, t, dst_conn="v", memlet=dace.Memlet("x[j]"))
        st.add_memlet_path(t, mx, st.add_write("y"), src_conn="o", memlet=dace.Memlet("y[j]"))
    else:
        st.add_mapped_tasklet("r", {"i": f"0:{N}"}, {"v": dace.Memlet("x[i]")}, "o = v", {"o": dace.Memlet("y[i]")},
                              external_edges=True)
    outer = _sdfg("outer", "a", "out")
    s = outer.add_state()
    ns = s.add_nested_sdfg(inner, [] if write_first else ["x"], ["y", "x"] if write_first else ["y"])
    if write_first:
        s.add_edge(ns, "x", s.add_write("a"), None, dace.Memlet(f"a[0:{N}]"))
    else:
        s.add_edge(s.add_read("a"), None, ns, "x", dace.Memlet(f"a[0:{N}]"))
    s.add_edge(ns, "y", s.add_write("out"), None, dace.Memlet(f"out[0:{N}]"))
    if write_first:
        s2 = outer.add_state_after(s)
        _mapped(s2, "a", "out", f"0:{N}")
    return outer


def test_nested_sdfg_writes_count_outside():
    assert uncovered_reads(_nested(write_first=True), ["a"])["a"] == []


def test_nested_sdfg_reads_are_checked_inside():
    assert uncovered_reads(_nested(write_first=False), ["a"])["a"]


def test_arrays_behind_a_view_are_not_tracked():
    sdfg = _sdfg("view", "a", "b", "out")
    sdfg.add_view("v", [N], dace.float64)
    s1 = sdfg.add_state(is_start_block=True)
    _mapped(s1, "b", "a", f"0:{N}")
    s2 = sdfg.add_state_after(s1)
    a, v = s2.add_read("a"), s2.add_access("v")
    s2.add_edge(a, None, v, None, dace.Memlet(f"a[0:{N}]"))
    t = s2.add_tasklet("t", {"x"}, {"y"}, "y = x")
    s2.add_edge(v, None, t, "x", dace.Memlet("v[0]"))
    s2.add_edge(t, "y", s2.add_write("out"), None, dace.Memlet("out[0]"))
    assert "view" in uncovered_reads(sdfg, ["a"])["a"][0]


def test_unstructured_cycle_does_not_trust_its_counter():
    """i = 0; a[i] = ...; then a region cycling over a[i] with i incremented: only a[0] was written."""
    sdfg = _sdfg("cycle", "a", "out")
    sdfg.add_symbol("i", dace.int64)
    s0 = sdfg.add_state("init", is_start_block=True)
    s1 = sdfg.add_state("w")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"i": "0"}))
    _write(s1, "a", "i")
    region = ControlFlowRegion("cyc", sdfg=sdfg)
    sdfg.add_node(region)
    sdfg.add_edge(s1, region, dace.InterstateEdge())
    head = region.add_state("head", is_start_block=True)
    body = region.add_state("body")
    _copy(body, "a", "out", "i", "0")
    region.add_edge(head, body, dace.InterstateEdge(condition=f"i < {N}"))
    region.add_edge(body, head, dace.InterstateEdge(assignments={"i": "i + 1"}))
    region.add_edge(head, region.add_state("exit"), dace.InterstateEdge(condition=f"i >= {N}"))
    assert uncovered_reads(sdfg, ["a"])["a"]
