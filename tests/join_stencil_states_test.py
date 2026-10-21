"""join_stencil_states: a stencil's producer and consumer in two states that only symbol assignments keep apart (loop
bounds of later code a source sets between them) fuse into one state, the assignments moving past the consumer, so
the stencil fusion sees the pair.

Adapted from ``tests/substitute/test_join_stencil_states.py`` on ``mfc-substitute-20261021``:
- the producer/consumer bodies are mapped tasklets, not nested SDFGs.  The original's nested bodies relied on the
  No-View-Nested-SDFG memlet composition this fork has not taken (validation rejects the composed subset);
  ``join_stencil_states`` sees the same top-level MapExit / transient / MapEntry shape either way.
- the final ``fuse_stencil_pairs`` assertion is dropped: that needs the OTF/Tile stencil fusion this fork has not
  ported.  The join itself is what this file tests.
"""
import dace
import numpy as np
from dace.sdfg import nodes

from dace_fortran.join_stencil_states import join_stencil_states


def _split_pair(consumer_range: str = "1:14") -> dace.SDFG:
    """tmp[i] = 2 * a[i] + 1 over 1..14, then iend = p, then b[j] = tmp[j + 1] - 1 over ``consumer_range``, then
    c[0] = iend."""
    sdfg = dace.SDFG("split_pair")
    sdfg.add_symbol("p", dace.int64)
    sdfg.add_symbol("iend", dace.int64)
    for name, size in (("a", 16), ("b", 16), ("c", 1)):
        sdfg.add_array(name, [size], dace.float64)
    sdfg.add_transient("tmp", [16], dace.float64)
    first = sdfg.add_state("produce_state", is_start_block=True)
    first.add_mapped_tasklet("produce", {"i": "1:15"}, {"x": dace.Memlet("a[i]")}, "y = 2 * x + 1",
                             {"y": dace.Memlet("tmp[i]")}, external_edges=True)
    second = sdfg.add_state("consume_state")
    second.add_mapped_tasklet("consume", {"j": consumer_range}, {"r": dace.Memlet("tmp[j + 1]")}, "o = r - 1",
                              {"o": dace.Memlet("b[j]")}, external_edges=True)
    third = sdfg.add_state("use_state")
    tasklet = third.add_tasklet("use", {}, {"o"}, "o = iend")
    third.add_edge(tasklet, "o", third.add_write("c"), None, dace.Memlet("c[0]"))
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={"iend": "p"}))
    sdfg.add_edge(second, third, dace.InterstateEdge())
    return sdfg


def test_an_assignment_the_consumer_does_not_read_moves_past_it_and_the_pair_fuses(tmp_path):
    sdfg = _split_pair()
    assert join_stencil_states(sdfg) == 1
    assert len(list(sdfg.states())) == 2
    (edge,) = sdfg.edges()
    assert edge.data.assignments == {"iend": "p"}  # now after the consumer, before the state that reads it
    # The joined SDFG computes what the split one does.
    sdfg.build_folder = str(tmp_path / "build")
    a = np.arange(16, dtype=np.float64)
    b = np.zeros(16)
    c = np.zeros(1)
    sdfg(a=a, b=b, c=c, p=7)
    expected = np.zeros(16)
    expected[1:14] = 2 * a[2:15] + 1 - 1
    np.testing.assert_array_equal(b, expected)
    assert c[0] == 7


def test_an_assignment_the_consumer_reads_takes_its_value_there(tmp_path):
    """The consumer's map runs to iend: in its state iend is p, which replaces it, and the states join."""
    sdfg = _split_pair(consumer_range="1:iend")
    assert join_stencil_states(sdfg) == 1
    consume = next(n for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry)
                   and n.map.label == "consume_map")
    assert str(consume.map.range[0][1]) == "p - 1"
    sdfg.build_folder = str(tmp_path / "build")
    a, b, c = np.arange(16, dtype=np.float64), np.zeros(16), np.zeros(1)
    sdfg(a=a, b=b, c=c, p=14)
    expected = np.zeros(16)
    expected[1:14] = 2 * a[2:15]
    np.testing.assert_array_equal(b, expected)
    assert c[0] == 14


def test_an_assignment_of_a_value_read_from_data_keeps_the_states_apart():
    """The value reads a container the consumer's state might write: the assignment stays before it."""
    sdfg = _split_pair()
    sdfg.add_scalar("n", dace.int64)
    first, second = list(sdfg.states())[:2]
    sdfg.edges_between(first, second)[0].data.assignments = {"iend": "n"}
    assert join_stencil_states(sdfg) == 0
    assert sdfg.edges_between(first, second)[0].data.assignments == {"iend": "n"}


def test_a_state_before_the_producer_with_nothing_assigned_between_stays_apart():
    """A state filling what the producer's state reads, nothing assigned between them, is not joined with it: that
    join would take the producer's state out of the list the pass walks, and the pair would stay apart."""
    sdfg = _split_pair()
    produce_state = next(s for s in sdfg.states() if s.label == "produce_state")
    sdfg.add_transient("pre", [16], dace.float64)
    sdfg.add_array("d", [16], dace.float64)
    fill = sdfg.add_state_before(produce_state, "fill_state", is_start_block=True)
    fill.add_mapped_tasklet("fill", {"i": "0:16"}, {"x": dace.Memlet("a[i]")}, "y = x", {"y": dace.Memlet("pre[i]")},
                            external_edges=True)
    produce_state.add_mapped_tasklet("use_pre", {"i": "0:16"}, {"x": dace.Memlet("pre[i]")}, "y = x",
                                     {"y": dace.Memlet("d[i]")}, external_edges=True)
    assert join_stencil_states(sdfg) == 1
    maps = {n.map.label: s for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry)}
    assert maps["produce_map"] is maps["consume_map"]
    assert maps["fill_map"] is not maps["produce_map"]
