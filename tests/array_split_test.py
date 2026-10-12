"""Splitting an array along a dimension every access subscripts with a constant: ``a[k, ...]`` becomes the arrays
``a_k[...]``, and anything else leaves the SDFG as it was."""

import dace
import networkx as nx
import numpy as np
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


def _shared_node_sdfg() -> dace.SDFG:
    """``a[0] = x`` and ``a[1] = y`` into one node of the transient ``a``, ``a[1] = 2 * a[1]`` into the next one, and
    ``out = a[0] + a[1]`` read from that second node: the read of ``a[0]`` follows its write only through the nodes
    of ``a``."""
    sdfg = dace.SDFG("shared_node")
    for name in ("x", "y", "out"):
        sdfg.add_array(name, [1], dace.float64)
    sdfg.add_transient("a", [2], dace.float64)
    state = sdfg.add_state()
    first, second = state.add_access("a"), state.add_access("a")
    for name, index in (("x", 0), ("y", 1)):
        copy_in = state.add_tasklet(f"set_{index}", {"_i"}, {"_o"}, "_o = _i")
        state.add_edge(state.add_read(name), None, copy_in, "_i", dace.Memlet(f"{name}[0]"))
        state.add_edge(copy_in, "_o", first, None, dace.Memlet(f"a[{index}]"))
    double = state.add_tasklet("double", {"_i"}, {"_o"}, "_o = 2 * _i")
    state.add_edge(first, None, double, "_i", dace.Memlet("a[1]"))
    state.add_edge(double, "_o", second, None, dace.Memlet("a[1]"))
    add = state.add_tasklet("add", {"_p", "_q"}, {"_o"}, "_o = _p + _q")
    state.add_edge(second, None, add, "_p", dace.Memlet("a[0]"))
    state.add_edge(second, None, add, "_q", dace.Memlet("a[1]"))
    state.add_edge(add, "_o", state.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


def test_a_read_of_one_element_still_follows_the_writes_into_its_node():
    sdfg = _shared_node_sdfg()
    assert split_array(sdfg, "a") == {0: "a_0", 1: "a_1"}
    sdfg.validate()
    state = sdfg.start_state
    add = next(n for n in state.nodes() if getattr(n, "label", None) == "add")
    written = next(n for n in state.data_nodes() if n.data == "a_0" and state.in_degree(n) > 0)
    assert nx.has_path(state.nx, written, add)


def test_dead_dataflow_elimination_keeps_a_write_read_through_another_node():
    """Without the order, ``a_0`` written into a sink node and read from a separate source node looks dead to
    dead-dataflow elimination, which drops the write: ``out`` would be ``2 * y`` plus whatever ``a_0`` held."""
    sdfg = _shared_node_sdfg()
    split_array(sdfg, "a")
    sdfg.simplify()
    out = np.zeros(1)
    sdfg(x=np.array([3.0]), y=np.array([5.0]), out=out)
    assert out[0] == 13.0


def test_the_elements_of_a_one_dimensional_transient_become_scalars():
    sdfg = _shared_node_sdfg()
    split_array(sdfg, "a")
    assert all(isinstance(sdfg.arrays[f"a_{k}"], dace.data.Scalar) for k in (0, 1))
    assert all(
        e.data.subset.ranges == [(0, 0, 1)] for e in sdfg.start_state.edges() if e.data.data in ("a_0", "a_1")
    )


def test_ordering_edges_of_a_node_hold_for_every_part():
    """An empty edge into (out of) a node of the array orders each of its parts; a node holding only empty edges
    passes the order from its predecessors to its successors."""
    sdfg = _shared_node_sdfg()
    state = sdfg.start_state
    first = next(n for n in state.data_nodes() if n.data == "a" and state.in_degree(n) == 2)
    out = next(n for n in state.data_nodes() if n.data == "out")
    x = next(n for n in state.data_nodes() if n.data == "x")
    state.add_nedge(first, out, dace.Memlet())
    relay = state.add_access("a")
    state.add_nedge(x, relay, dace.Memlet())
    state.add_nedge(relay, out, dace.Memlet())
    assert constant_indices(sdfg, "a") == {0, 1}
    split_array(sdfg, "a")
    sdfg.validate()
    for k in (0, 1):
        writer = f"set_{k}"
        part = next(
            n
            for n in state.data_nodes()
            if n.data == f"a_{k}" and any(e.src.label == writer for e in state.in_edges(n))
        )
        assert any(e.dst is out and e.data.is_empty() for e in state.out_edges(part))
    assert any(e.src is x and e.dst is out and e.data.is_empty() for e in state.edges())
    assert nx.is_directed_acyclic_graph(state.nx)
