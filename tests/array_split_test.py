"""Splitting an array along a dimension every access subscripts with a constant: ``a[k, ...]`` becomes the arrays
``a_k[...]``, and anything else leaves the SDFG as it was."""

import numpy as np
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

from dace_fortran.array_split import constant_indices, split_array


def _rows_sdfg(index_of_second: str = "2") -> dace.SDFG:
    """A map writing ``a[0, i] = x[i]`` and ``a[<index_of_second>, i] = 2 * x[i]``."""
    sdfg = dace.SDFG("rows")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("x", ["n"], dace.float64)
    sdfg.add_array("a", [3, "n"], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "first",
        {"i": "0:n"},
        {"_x": dace.Memlet("x[i]")},
        "_a = _x",
        {"_a": dace.Memlet("a[0, i]")},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "second",
        {"i": "0:n"},
        {"_x": dace.Memlet("x[i]")},
        "_a = 2 * _x",
        {"_a": dace.Memlet(f"a[{index_of_second}, i]")},
        external_edges=True,
    )
    return sdfg


def test_constant_rows_become_arrays_of_their_own():
    sdfg = _rows_sdfg()
    assert constant_indices(sdfg, "a") == {0, 2}
    assert split_array(sdfg, "a") == {0: "a_0", 2: "a_2"}
    assert "a" not in sdfg.arrays
    assert sdfg.arrays["a_0"].shape == (dace.symbol("n"),) and sdfg.arrays["a_2"].strides == (1,)
    sdfg.validate()
    x = np.arange(5, dtype=np.float64)
    a_0, a_2 = np.zeros(5), np.zeros(5)
    sdfg(x=x, a_0=a_0, a_2=a_2, n=5)
    np.testing.assert_array_equal(a_0, x)
    np.testing.assert_array_equal(a_2, 2 * x)


def test_a_non_constant_index_leaves_the_array():
    sdfg = _rows_sdfg(index_of_second="i % 3")
    assert constant_indices(sdfg, "a") is None
    assert split_array(sdfg, "a") is None
    assert "a" in sdfg.arrays


def test_a_condition_reading_the_array_leaves_it():
    sdfg = _rows_sdfg()
    first = next(iter(sdfg.states()))
    branch = ConditionalBlock("if_a")
    sdfg.add_node(branch)
    sdfg.add_edge(first, branch, dace.InterstateEdge())
    body = ControlFlowRegion("then", sdfg=sdfg)
    body.add_state("noop", is_start_block=True)
    branch.add_branch(CodeBlock("a[0, 0] > 0"), body)
    assert split_array(sdfg, "a") is None
