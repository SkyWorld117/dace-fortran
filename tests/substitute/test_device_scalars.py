"""Tests of device_scalars (CPU: structure only).  Run: python -m pytest test_device_scalars.py"""
import dace
from dace.sdfg import nodes

from dace_fortran.substitute.device_scalars import device_copies_of_kernel_read_scalars


def _kernel_reading_scalars(write_t: bool) -> dace.SDFG:
    """A map whose nested body reads the scalars ``s`` and ``t``; a free tasklet on the host reads ``s`` too; with
    ``write_t`` a tasklet writes ``t`` before the map."""
    sdfg = dace.SDFG("kernel_scalars" + ("_written" if write_t else ""))
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    sdfg.add_scalar("s", dace.float64)
    sdfg.add_scalar("t", dace.float64)
    sdfg.add_array("h", [1], dace.float64)
    scalar = {n: dace.Memlet.from_array(n, sdfg.arrays[n]) for n in ("s", "t")}
    state = sdfg.add_state(is_start_block=True)
    body = dace.SDFG("body")
    body.add_array("a", [8], dace.float64)
    body.add_array("b", [8], dace.float64)
    body.add_scalar("s", dace.float64)
    body.add_scalar("t", dace.float64)
    body.add_symbol("i", dace.int32)
    bstate = body.add_state()
    bt = bstate.add_tasklet("bt", {"x", "cs", "ct"}, {"y"}, "y = x * cs + ct")
    bstate.add_edge(bstate.add_read("a"), None, bt, "x", dace.Memlet("a[i]"))
    bstate.add_edge(bstate.add_read("s"), None, bt, "cs", dace.Memlet.from_array("s", body.arrays["s"]))
    bstate.add_edge(bstate.add_read("t"), None, bt, "ct", dace.Memlet.from_array("t", body.arrays["t"]))
    bstate.add_edge(bt, "y", bstate.add_write("b"), None, dace.Memlet("b[i]"))
    if write_t:
        wt = state.add_tasklet("set_t", {}, {"o"}, "o = 2.0")
        state.add_edge(wt, "o", state.add_write("t"), None, scalar["t"])
    nested = state.add_nested_sdfg(body, {"a", "s", "t"}, {"b"}, symbol_mapping={"i": "i"})
    me, mx = state.add_map("m", {"i": "0:8"})
    state.add_memlet_path(state.add_read("A"), me, nested, dst_conn="a", memlet=dace.Memlet("A[0:8]"))
    s_node = state.add_read("s")
    state.add_memlet_path(s_node, me, nested, dst_conn="s", memlet=scalar["s"])
    state.add_memlet_path(state.add_access("t"), me, nested, dst_conn="t", memlet=scalar["t"])
    state.add_memlet_path(nested, mx, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[0:8]"))
    host = state.add_tasklet("host", {"x"}, {"y"}, "y = x")
    state.add_edge(s_node, None, host, "x", scalar["s"])
    state.add_edge(host, "y", state.add_write("h"), None, dace.Memlet("h[0]"))
    sdfg.validate()
    return sdfg


def _map_inputs(sdfg: dace.SDFG) -> dict[str, dace.data.Data]:
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                return {e.data.data: sdfg.arrays[e.data.data] for e in state.in_edges(node)}
    raise AssertionError("no map")


def test_scalars_a_kernel_reads_get_device_copies():
    sdfg = _kernel_reading_scalars(write_t=False)
    assert device_copies_of_kernel_read_scalars(sdfg) == ["s", "t"]
    sdfg.validate()
    inputs = _map_inputs(sdfg)
    assert "s" not in inputs and "t" not in inputs
    copies = {n: d for n, d in inputs.items() if isinstance(d, dace.data.Scalar)}
    assert all(d.transient and d.storage == dace.StorageType.GPU_Global for d in copies.values())
    first = sdfg.start_block
    assert first.label == "scalars_to_device" and len(first.nodes()) == 4
    # The host tasklet keeps reading the argument itself.
    host_reads = [e.data.data for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.Tasklet)
                  and n.label == "host" for e in st.in_edges(n)]
    assert host_reads == ["s"]
    # The nested body names the copy.
    nested = next(n for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.NestedSDFG))
    assert set(nested.in_connectors) == {"a", "s", "t"}


def test_a_scalar_the_sdfg_writes_keeps_its_host_copy():
    sdfg = _kernel_reading_scalars(write_t=True)
    assert device_copies_of_kernel_read_scalars(sdfg) == ["s"]
    sdfg.validate()
    inputs = _map_inputs(sdfg)
    assert "t" in inputs and "s" not in inputs
