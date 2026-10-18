"""Folding of known values in control flow (fuse_stencils.fold_control_flow_constants), CPU.
Run: python -m pytest test_fold_constants.py"""
import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock

from dace_fortran.substitute.fuse_stencils import _fold_conditional_expressions, fold_control_flow_constants


def test_a_conditional_expression_with_a_literal_condition_takes_its_branch():
    assert _fold_conditional_expressions("(_i < ((a if (1 == 1) else b) + 1))") == "_i < a + 1"
    assert _fold_conditional_expressions("(a if (2 == 1) else b)") == "b"
    assert _fold_conditional_expressions("(a if (n == 1) else b)") == "(a if (n == 1) else b)"  # unknown: kept
    assert _fold_conditional_expressions("x = 1") == "x = 1"


def _branch_on_flag() -> dace.SDFG:
    """out[i] = x[i] * (2 if flags[d] != 0 else 3) for every i, d fixed by an interstate assignment: the branch reads
    an element of a module array of flags at a subscript known only after propagation."""
    sdfg = dace.SDFG("branch_on_flag")
    sdfg.add_array("x", [8], dace.float64)
    sdfg.add_array("out", [8], dace.float64)
    sdfg.add_array("flags", [3], dace.int32)
    sdfg.add_symbol("d", dace.int64)
    init = sdfg.add_state("init", is_start_block=True)
    branch = ConditionalBlock("on_flag")
    sdfg.add_node(branch)
    sdfg.add_edge(init, branch, dace.InterstateEdge(assignments={"d": "1"}))
    for code, factor in (("flags[d] != 0", 2.0), (None, 3.0)):
        body = dace.sdfg.state.ControlFlowRegion(f"b{int(factor)}", sdfg=sdfg)
        st = body.add_state(f"s{int(factor)}", is_start_block=True)
        st.add_mapped_tasklet("scale", {"i": "0:8"}, {"_x": dace.Memlet("x[i]")}, f"_o = _x * {factor}",
                              {"_o": dace.Memlet("out[i]")}, external_edges=True)
        branch.add_branch(dace.properties.CodeBlock(code) if code else None, body)
    return sdfg


def test_a_branch_on_a_recorded_element_folds_once_its_subscript_is_known():
    sdfg = _branch_on_flag()
    fold_control_flow_constants(sdfg, {}, {"flags": [1, 0, 1]})
    sdfg.simplify()
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks(recursive=True))
    x, out = np.arange(8, dtype=np.float64), np.zeros(8)
    sdfg(x=x, out=out, flags=np.array([1, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(out, 3.0 * x)  # flags[1] == 0: the else branch


def _branch_on_symbol(condition: str) -> dace.SDFG:
    """out[i] = x[i] * (2 if <condition> else 3), the condition reading the symbol ``bc``."""
    sdfg = dace.SDFG("branch_on_symbol")
    sdfg.add_array("x", [8], dace.float64)
    sdfg.add_array("out", [8], dace.float64)
    sdfg.add_symbol("bc", dace.int64)
    sdfg.add_state("init", is_start_block=True)
    branch = ConditionalBlock("on_bc")
    sdfg.add_node(branch)
    sdfg.add_edge(sdfg.start_state, branch, dace.InterstateEdge())
    for code, factor in ((condition, 2.0), (None, 3.0)):
        body = dace.sdfg.state.ControlFlowRegion(f"b{int(factor)}", sdfg=sdfg)
        st = body.add_state(f"s{int(factor)}", is_start_block=True)
        st.add_mapped_tasklet("scale", {"i": "0:8"}, {"_x": dace.Memlet("x[i]")}, f"_o = _x * {factor}",
                              {"_o": dace.Memlet("out[i]")}, external_edges=True)
        branch.add_branch(dace.properties.CodeBlock(code) if code else None, body)
    return sdfg


def test_a_condition_the_same_for_every_value_of_a_set_folds_and_one_that_differs_stays():
    from dace_fortran.substitute.fuse_stencils import fold_conditions_over_value_sets

    sdfg = _branch_on_symbol("bc <= -5")  # a boundary condition no rank has: bc is 0 on one rank, -3 on the other
    assert fold_conditions_over_value_sets(sdfg, {"bc": [0, -3]}) == 1
    sdfg.simplify()
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks(recursive=True))
    x, out = np.arange(8, dtype=np.float64), np.zeros(8)
    sdfg(x=x, out=out, bc=0)
    np.testing.assert_array_equal(out, 3.0 * x)

    kept = _branch_on_symbol("bc >= 0")  # the exchange with a neighbour: one rank has it, the other has not
    assert fold_conditions_over_value_sets(kept, {"bc": [0, -3]}) == 0
    kept.simplify()
    assert any(isinstance(b, ConditionalBlock) for b in kept.all_control_flow_blocks(recursive=True))
    out = np.zeros(8)
    kept(x=x, out=out, bc=-3)
    np.testing.assert_array_equal(out, 3.0 * x)
    kept(x=x, out=out, bc=0)
    np.testing.assert_array_equal(out, 2.0 * x)
