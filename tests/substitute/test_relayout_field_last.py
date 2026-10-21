"""relayout_field_last: a transient image of a record array gets its field dimensions contiguous, its small ones (a
record's index, a component) slowest; an extent that differs between the ranks of a run (a symbol) counts as a field
dimension."""
import dace
import numpy as np

from dace_fortran.substitute.make_substitute import relayout_field_last


def _copy_through(shape: list) -> dace.SDFG:
    """b = a through the transient img: one state fills img, the next copies it to b in a map whose body is a nested
    SDFG taking img and b whole, with its own name for the extent ``n`` (``n_in``)."""
    sdfg = dace.SDFG("copy_through")
    sdfg.add_symbol("n", dace.int64)
    for name in ("a", "b"):
        sdfg.add_array(name, shape, dace.float64)
    sdfg.add_transient("img", shape, dace.float64)
    params = {f"i{d}": f"0:{s}" for d, s in enumerate(shape)}
    index = ", ".join(params)
    fill = sdfg.add_state("fill", is_start_block=True)
    fill.add_mapped_tasklet("fill", params, {"x": dace.Memlet(f"a[{index}]")}, "y = x",
                            {"y": dace.Memlet(f"img[{index}]")}, external_edges=True)
    copy = sdfg.add_state_after(fill, "copy")
    inner_shape = [str(s).replace("n", "n_in") for s in shape]
    body = dace.SDFG("body")
    body.add_symbol("n_in", dace.int64)
    for name in ("img", "b"):
        body.add_array(name, inner_shape, dace.float64)
    for p in params:
        body.add_symbol(p, dace.int64)
    st = body.add_state(is_start_block=True)
    tasklet = st.add_tasklet("copy", {"x"}, {"y"}, "y = x")
    st.add_edge(st.add_read("img"), None, tasklet, "x", dace.Memlet(f"img[{index}]"))
    st.add_edge(tasklet, "y", st.add_write("b"), None, dace.Memlet(f"b[{index}]"))
    entry, exit_ = copy.add_map("copy", params)
    nested = copy.add_nested_sdfg(body, {"img"}, {"b"}, {**{p: p for p in params}, "n_in": "n"})
    whole = ", ".join(f"0:{s}" for s in shape)
    copy.add_memlet_path(copy.add_read("img"), entry, nested, dst_conn="img", memlet=dace.Memlet(f"img[{whole}]"))
    copy.add_memlet_path(nested, exit_, copy.add_write("b"), src_conn="b", memlet=dace.Memlet(f"b[{whole}]"))
    return sdfg


def test_a_constant_shape_gets_its_field_dimensions_first():
    sdfg = _copy_through([3, 40, 24])
    assert relayout_field_last(sdfg, ["img"]) == {"img": (960, 1, 40)}


def test_a_symbolic_extent_is_a_field_dimension(tmp_path):
    sdfg = _copy_through([2, "n", 20])
    n = dace.symbolic.pystr_to_symbolic("n")
    assert relayout_field_last(sdfg, ["img"]) == {"img": (20 * n, 1, n)}
    body = next(node for st in sdfg.states() for node in st.nodes() if isinstance(node, dace.nodes.NestedSDFG))
    assert "n" in body.sdfg.symbols and body.sdfg.arrays["img"].strides == (20 * n, 1, n)
    sdfg.build_folder = str(tmp_path / "build")
    a = np.random.default_rng(0).random((2, 5, 20))
    b = np.zeros_like(a)
    sdfg(a=a, b=b, n=5)
    np.testing.assert_array_equal(b, a)
