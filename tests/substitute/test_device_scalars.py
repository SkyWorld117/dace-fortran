"""Tests of device_scalars (CPU: structure only).  Run: python -m pytest test_device_scalars.py"""
import dace
from dace.sdfg import nodes

from dace_fortran.substitute.device_scalars import device_copies_of_kernel_read_scalars

GPU = dace.StorageType.GPU_Global


def _kernel_reading_scalars(write_t: bool) -> dace.SDFG:
    """A GPU map whose nested body reads the scalars ``s`` and ``t``; a free tasklet on the host reads ``s`` too;
    with ``write_t`` a host tasklet writes ``t`` first (a specialised value)."""
    sdfg = dace.SDFG("kernel_scalars" + ("_written" if write_t else ""))
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    sdfg.add_scalar("s", dace.float64)
    sdfg.add_scalar("t", dace.float64, transient=write_t)
    sdfg.add_array("h", [1], dace.float64)

    def scalar(n):  # a fresh memlet per edge
        return dace.Memlet.from_array(n, sdfg.arrays[n])

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
    t_node = state.add_access("t")
    if write_t:
        wt = state.add_tasklet("set_t", {}, {"o"}, "o = 2.0")
        state.add_edge(wt, "o", t_node, None, scalar("t"))
    nested = state.add_nested_sdfg(body, {"a", "s", "t"}, {"b"}, symbol_mapping={"i": "i"})
    me, mx = state.add_map("m", {"i": "0:8"}, schedule=dace.ScheduleType.GPU_Device)
    state.add_memlet_path(state.add_read("A"), me, nested, dst_conn="a", memlet=dace.Memlet("A[0:8]"))
    s_node = state.add_read("s")
    state.add_memlet_path(s_node, me, nested, dst_conn="s", memlet=scalar("s"))
    state.add_memlet_path(t_node, me, nested, dst_conn="t", memlet=scalar("t"))
    state.add_memlet_path(nested, mx, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[0:8]"))
    host = state.add_tasklet("host", {"x"}, {"y"}, "y = x")
    state.add_edge(s_node, None, host, "x", scalar("s"))
    state.add_edge(host, "y", state.add_write("h"), None, dace.Memlet("h[0]"))
    return sdfg


def _map_inputs(sdfg: dace.SDFG) -> dict[str, dace.data.Data]:
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                return {e.data.data: sdfg.arrays[e.data.data] for e in state.in_edges(node)}
    raise AssertionError("no map")


def _copies(state: dace.SDFGState) -> dict[str, str]:
    """Host-to-device copies in ``state``: source name -> device name."""
    return {e.src.data: e.dst.data for e in state.edges() if isinstance(e.src, nodes.AccessNode)
            and isinstance(e.dst, nodes.AccessNode) and state.parent.arrays[e.dst.data].storage == GPU}


def test_scalars_a_kernel_reads_get_device_copies():
    sdfg = _kernel_reading_scalars(write_t=False)
    assert device_copies_of_kernel_read_scalars(sdfg) == ["s", "t"]
    sdfg.validate()
    inputs = _map_inputs(sdfg)
    assert "s" not in inputs and "t" not in inputs
    assert all(d.transient and d.storage == GPU for n, d in inputs.items() if isinstance(d, dace.data.Scalar))
    assert set(_copies(sdfg.start_block)) == {"s", "t"}
    # The host tasklet keeps reading the argument itself.
    host_reads = [e.data.data for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.Tasklet)
                  and n.label == "host" for e in st.in_edges(n)]
    assert host_reads == ["s"]


def test_a_scalar_a_host_tasklet_writes_is_copied_after_the_write():
    sdfg = _kernel_reading_scalars(write_t=True)
    assert device_copies_of_kernel_read_scalars(sdfg) == ["s", "t"]
    sdfg.validate()
    state = sdfg.start_block
    copies = _copies(state)
    assert set(copies) == {"s", "t"}
    writer = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == "set_t")
    written = state.out_edges(writer)[0].dst
    assert any(e.dst.data == copies["t"] for e in state.out_edges(written)), "the copy reads the written node"


def test_a_scalar_read_from_inside_the_map_scope_gets_a_device_copy():
    sdfg = dace.SDFG("inside_scope")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    sdfg.add_scalar("s", dace.float64)
    state = sdfg.add_state(is_start_block=True)
    me, mx = state.add_map("m", {"i": "0:8"}, schedule=dace.ScheduleType.GPU_Device)
    t = state.add_tasklet("t", {"x", "cs"}, {"y"}, "y = x * cs")
    state.add_memlet_path(state.add_read("A"), me, t, dst_conn="x", memlet=dace.Memlet("A[i]"))
    inner = state.add_read("s")  # inside the scope: no edge through the map entry
    state.add_edge(inner, None, t, "cs", dace.Memlet.from_array("s", sdfg.arrays["s"]))
    state.add_nedge(me, inner, dace.Memlet())
    state.add_memlet_path(t, mx, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i]"))
    sdfg.validate()
    assert device_copies_of_kernel_read_scalars(sdfg) == ["s"]
    sdfg.validate()
    assert inner.data != "s" and sdfg.arrays[inner.data].storage == GPU
    copies = _copies(state)
    assert copies == {"s": inner.data}
    dev_node = next(n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == inner.data
                    and state.in_degree(n))
    assert any(e.dst is me for e in state.out_edges(dev_node)), "the copy is ordered before the kernel"


def test_a_host_side_nested_sdfg_gets_its_own_copies():
    """The kernel sits in a nested SDFG outside any map; the scalar reaches it as a connector."""
    outer = dace.SDFG("outer")
    outer.add_array("A", [8], dace.float64)
    outer.add_array("B", [8], dace.float64)
    outer.add_scalar("s", dace.float64)
    inner = _kernel_reading_scalars(write_t=True)
    inner.arrays["h"].transient = True
    state = outer.add_state(is_start_block=True)
    node = state.add_nested_sdfg(inner, {"A", "s"}, {"B"})
    state.add_edge(state.add_read("A"), None, node, "A", dace.Memlet("A[0:8]"))
    state.add_edge(state.add_read("s"), None, node, "s", dace.Memlet.from_array("s", outer.arrays["s"]))
    state.add_edge(node, "B", state.add_write("B"), None, dace.Memlet("B[0:8]"))
    outer.validate()
    assert device_copies_of_kernel_read_scalars(outer) == [f"{inner.label}.s", f"{inner.label}.t"]
    outer.validate()
