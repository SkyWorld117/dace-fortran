"""Soundness of the loop-local scalar fission predicate.

The rename cannot change a value on its own -- every access moves to a copy, and the copy is
confined to the loop by construction.  The one way this pass miscompiles is by deciding a scalar is
loop-local when it is not, so that decision is what these cases pin: the two patterns where a
scalar legitimately CARRIES a value must be refused, and the pattern this pass exists for must
split.  They build their CFGs directly (no Fortran source), so they need neither flang nor
gfortran and run anywhere.
"""
import dace
from dace.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

from dace_fortran.loop_scalar_fission import fission_loop_local_scalars


def _base(n=8):
    sdfg = dace.SDFG("t")
    sdfg.add_array("dst", [n], dace.float32)
    sdfg.add_array("src", [n], dace.float32)
    sdfg.add_scalar("x", dace.float32, transient=True)
    return sdfg


def _write(state, name, arr, idx=0):
    """``name = arr[idx]`` -- returns the access node, so a co-located reader can reuse it."""
    t = state.add_tasklet("w", {"inp"}, {"out"}, "out = inp")
    a = state.add_access(arr)
    node = state.add_access(name)
    state.add_edge(a, None, t, "inp", dace.Memlet(f"{arr}[{idx}]"))
    state.add_edge(t, "out", node, None, dace.Memlet(f"{name}[0]"))
    return node


def _read(state, name, arr, idx=0, node=None):
    """``arr[idx] = name``.  ``node`` reuses the writer's access node when both live in one state --
    which is the shape the builder emits for ``x = ...`` followed by ``y = x``, and one node is the
    only well-formed way to have both in a single state."""
    t = state.add_tasklet("r", {"inp"}, {"out"}, "out = inp")
    node = node if node is not None else state.add_access(name)
    a = state.add_access(arr)
    state.add_edge(node, None, t, "inp", dace.Memlet(f"{name}[0]"))
    state.add_edge(t, "out", a, None, dace.Memlet(f"{arr}[{idx}]"))
    return node


def test_writes_then_reads_in_one_iteration_splits():
    """The pattern this pass exists for: every read is fed by a write in the same iteration."""
    sdfg = _base()
    start = sdfg.add_state("start", is_start_block=True)
    loop = LoopRegion("L", "i = 0", "i < 8", "i = i + 1")
    sdfg.add_node(loop)
    body = loop.add_state("body")
    node = _write(body, "x", "src", "i")
    _read(body, "x", "dst", "i", node=node)
    sdfg.add_edge(start, loop, InterstateEdge())

    assert fission_loop_local_scalars(sdfg), "a provably iteration-local scalar must be split"
    sdfg.validate()


def test_read_before_write_is_refused():
    """CARRIED IN: the read takes the previous iteration's value, which the copy does not have."""
    sdfg = _base()
    start = sdfg.add_state("start", is_start_block=True)
    loop = LoopRegion("L", "i = 0", "i < 8", "i = i + 1")
    sdfg.add_node(loop)
    body = loop.add_state("body")
    node = _read(body, "x", "dst", "i")
    _write(body, "x", "src", "i")
    sdfg.add_edge(start, loop, InterstateEdge())

    assert not fission_loop_local_scalars(sdfg)


def test_conditional_write_is_refused():
    """CARRIED IN: the branch may not run, so a later read can still see a stale value."""
    sdfg = _base()
    start = sdfg.add_state("start", is_start_block=True)
    loop = LoopRegion("L", "i = 0", "i < 8", "i = i + 1")
    sdfg.add_node(loop)
    cond = ConditionalBlock("if")
    branch = ControlFlowRegion("branch")
    _write(branch.add_state("write", is_start_block=True), "x", "src", "i")
    cond.add_branch("i < 4", branch)
    loop.add_node(cond)
    after = loop.add_state("after")
    _read(after, "x", "dst", "i")
    loop.add_edge(cond, after, InterstateEdge())
    sdfg.add_edge(start, loop, InterstateEdge())

    assert not fission_loop_local_scalars(sdfg)


def test_loop_value_read_afterwards_is_refused():
    """CARRIED OUT: something after the loop expects what the loop last computed."""
    sdfg = _base()
    start = sdfg.add_state("start", is_start_block=True)
    loop = LoopRegion("L", "i = 0", "i < 8", "i = i + 1")
    sdfg.add_node(loop)
    body = loop.add_state("body")
    node = _write(body, "x", "src", "i")
    _read(body, "x", "dst", "i", node=node)
    end = sdfg.add_state("end")
    _read(end, "x", "dst", 0)
    sdfg.add_edge(start, loop, InterstateEdge())
    sdfg.add_edge(loop, end, InterstateEdge())

    assert not fission_loop_local_scalars(sdfg)


def test_scalar_written_before_the_loop_is_refused():
    """A top-level write means the loop reads a value defined outside it."""
    sdfg = _base()
    start = sdfg.add_state("start", is_start_block=True)
    _write(start, "x", "src", 0)
    loop = LoopRegion("L", "i = 0", "i < 8", "i = i + 1")
    sdfg.add_node(loop)
    body = loop.add_state("body")
    _read(body, "x", "dst", "i")
    sdfg.add_edge(start, loop, InterstateEdge())

    assert not fission_loop_local_scalars(sdfg)
