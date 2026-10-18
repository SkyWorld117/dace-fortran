"""``fold_aliases``: the image of a one-element module array of records folds onto the module array it is, in the
nested SDFGs too; the frozen signature loses the image.   Run: PYTHONPATH=../../dace:../../dace-fortran pytest"""
import numpy as np
import dace
from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenArgKind, FrozenSignature, attach_to_sdfg, \
    get_frozen_signature

from dace_fortran.substitute.fold_aliases import fold_aliases, fold_member_aliases

N = 8


def _sdfg() -> dace.SDFG:
    """``jac`` (module array, (N,)) and the image ``jac_sf_sf`` ((1, d1), offsets d1) of ``jac_sf(1)%sf``: a map
    doubles ``jac`` in place, a nested SDFG adds one to the image's element (a routine that fills ghost cells)."""
    sdfg = dace.SDFG("igr")
    sdfg.add_symbol("jac_sf_sf_d1", dace.int64)
    sdfg.add_symbol("offset_jac_sf_sf_d1", dace.int64)
    sdfg.add_array("jac", [N], dace.float64)
    sdfg.add_array("jac_sf_sf", [1, dace.symbol("jac_sf_sf_d1")], dace.float64)
    a = sdfg.add_state("double")
    a.add_mapped_tasklet("dbl", {"i": f"0:{N}"}, {"v": dace.Memlet("jac[i]")}, "w = 2 * v",
                         {"w": dace.Memlet("jac[i]")}, external_edges=True)
    b = sdfg.add_state_after(a, "fill")
    inner = dace.SDFG("fill_body")
    inner.add_symbol("off", dace.int64)
    inner.add_array("jac_sf_sf", [1, dace.symbol("jac_sf_sf_d1")], dace.float64)
    st = inner.add_state("s")
    st.add_mapped_tasklet("inc", {"k": "0:2"}, {"v": dace.Memlet("jac_sf_sf[0, k - off]")}, "w = v + 1",
                          {"w": dace.Memlet("jac_sf_sf[0, k - off]")}, external_edges=True)
    nsdfg = b.add_nested_sdfg(inner, {"jac_sf_sf"}, {"jac_sf_sf"}, {"off": "offset_jac_sf_sf_d1", "jac_sf_sf_d1": "jac_sf_sf_d1"})
    rd, wr = b.add_read("jac_sf_sf"), b.add_write("jac_sf_sf")
    b.add_edge(rd, None, nsdfg, "jac_sf_sf", dace.Memlet("jac_sf_sf[0, 0:jac_sf_sf_d1]"))
    b.add_edge(nsdfg, "jac_sf_sf", wr, None, dace.Memlet("jac_sf_sf[0, 0:jac_sf_sf_d1]"))
    arr = FrozenArgKind.ARRAY
    frozen = FrozenSignature("igr", "_QMmPigr", (
        FrozenArg("jac", "jac", arr, "float64", 1, (str(N),), is_written=True),
        FrozenArg("jac_sf_sf", "jac_sf_sf", arr, "float64", 2, ("1", "jac_sf_sf_d1"), aos_origin_mod="m_igr",
                  aos_origin_struct="jac_sf", aos_member_path="sf", aos_outer_rank=1, aos_member_pointer=True,
                  is_written=True),
        FrozenArg("jac_sf_sf_d1", "jac_sf_sf_d1", FrozenArgKind.SYMBOL, "int64", 0),
        FrozenArg("offset_jac_sf_sf_d1", "offset_jac_sf_sf_d1", FrozenArgKind.SYMBOL, "int64", 0),
    ), free_symbols=("jac_sf_sf_d1", "offset_jac_sf_sf_d1"))
    attach_to_sdfg(sdfg, frozen)
    sdfg.validate()
    return sdfg


def test_the_image_folds_onto_the_array_it_is():
    sdfg = _sdfg()
    assert fold_aliases(sdfg, [("jac_sf_sf", 0, "jac")], {"offset_jac_d0": -1}) == ["jac_sf_sf[0] -> jac"]
    sdfg.validate()
    assert "jac_sf_sf" not in sdfg.arrays and "jac_sf_sf_d1" not in sdfg.symbols
    frozen = get_frozen_signature(sdfg)
    assert [a.sdfg_name for a in frozen.args] == ["jac"] and frozen.free_symbols == ()
    jac = np.arange(N, dtype=np.float64)
    sdfg(jac=jac)
    expected = 2 * np.arange(N, dtype=np.float64)
    expected[1:3] += 1  # k - off for k in 0:2 and off = -1: elements 1 and 2
    np.testing.assert_allclose(jac, expected)


def test_an_image_of_several_elements_is_kept():
    sdfg = _sdfg()
    sdfg.arrays["jac_sf_sf"].shape = (2, dace.symbol("jac_sf_sf_d1"))
    assert fold_aliases(sdfg, [("jac_sf_sf", 1, "jac")], {}) == [
        "jac_sf_sf[1] kept apart from jac: element 1 of 2: only a one-element array of records folds"]
    assert "jac_sf_sf" in sdfg.arrays


def _members(ranged: bool = False) -> dace.SDFG:
    """The images ``p_sf`` of ``p(:)%sf`` and ``c_sf`` of ``c(:)%sf`` (three elements each), where the program points
    ``p(1)%sf`` at ``c(1)%sf``: c = x; p(1) = c(1) (a copy onto itself there), p(2:3) = 2 c(2:3); p(1)'s first
    point = its second (a boundary); y = c(1) + p(1) + p(2). ``ranged``: p(1:3) is also halved over all elements
    (no element a literal)."""
    sdfg = dace.SDFG("members")
    for name, shape in (("x", [3, N]), ("p_sf", [3, N]), ("c_sf", [3, N]), ("y", [N])):
        sdfg.add_array(name, shape, dace.float64)
    copy = sdfg.add_state("copy")
    for k in range(3):
        copy.add_mapped_tasklet(f"c{k}", {"i": f"0:{N}"}, {"v": dace.Memlet(f"x[{k}, i]")}, "w = v",
                                {"w": dace.Memlet(f"c_sf[{k}, i]")}, external_edges=True)
    convert = sdfg.add_state_after(copy, "convert")
    for k, factor in ((0, 1), (1, 2), (2, 2)):
        convert.add_mapped_tasklet(f"p{k}", {"i": f"0:{N}"}, {"v": dace.Memlet(f"c_sf[{k}, i]")}, f"w = {factor} * v",
                                   {"w": dace.Memlet(f"p_sf[{k}, i]")}, external_edges=True)
    last = convert
    if ranged:
        last = sdfg.add_state_after(convert, "halve")
        last.add_mapped_tasklet("h", {"k": "0:3", "i": f"0:{N}"}, {"v": dace.Memlet("p_sf[k, i]")}, "w = v / 2",
                                {"w": dace.Memlet("p_sf[k, i]")}, external_edges=True)
    boundary = sdfg.add_state_after(last, "boundary")
    t = boundary.add_tasklet("b", {"v"}, {"w"}, "w = v")
    boundary.add_edge(boundary.add_read("p_sf"), None, t, "v", dace.Memlet("p_sf[0, 1]"))
    boundary.add_edge(t, "w", boundary.add_write("p_sf"), None, dace.Memlet("p_sf[0, 0]"))
    # One map per element of p read: a map reading two takes them through one connector, over both elements.
    use = sdfg.add_state_after(boundary, "use")
    use.add_mapped_tasklet("u", {"i": f"0:{N}"}, {"a": dace.Memlet("c_sf[0, i]"), "d": dace.Memlet("p_sf[1, i]")},
                           "w = a + d", {"w": dace.Memlet("y[i]")}, external_edges=True)
    more = sdfg.add_state_after(use, "use_more")
    more.add_mapped_tasklet("v", {"i": f"0:{N}"}, {"a": dace.Memlet("y[i]"), "b": dace.Memlet("p_sf[0, i]")},
                            "w = a + b", {"w": dace.Memlet("y[i]")}, external_edges=True)
    sdfg.validate()
    return sdfg


def _aliased(x: np.ndarray, ranged: bool = False) -> np.ndarray:
    """y as the program computes it, ``p(1)%sf`` and ``c(1)%sf`` one array."""
    c = x.copy()
    p1 = 2 * c[1]
    if ranged:
        c[0] /= 2
        p1 /= 2
    c[0][0] = c[0][1]
    return c[0] + c[0] + p1


def test_an_element_of_one_image_folds_onto_the_element_of_the_other_it_is():
    sdfg = _members()
    assert fold_member_aliases(sdfg, [("p_sf", 0, "c_sf", 0)]) == [("p_sf", 0, "c_sf", 0)]
    sdfg.validate()
    x = np.random.default_rng(1).random((3, N))
    y = np.zeros(N)
    sdfg(x=x, p_sf=np.zeros((3, N)), c_sf=np.zeros((3, N)), y=y)
    np.testing.assert_allclose(y, _aliased(x))
    unfolded = np.zeros(N)
    _members()(x=x, p_sf=np.zeros((3, N)), c_sf=np.zeros((3, N)), y=unfolded)
    assert not np.allclose(unfolded, _aliased(x))  # two arrays: c(1) misses the boundary written through p(1)


def test_an_image_accessed_over_its_elements_folds_the_other_way():
    sdfg = _members(ranged=True)
    assert fold_member_aliases(sdfg, [("p_sf", 0, "c_sf", 0)]) == [("c_sf", 0, "p_sf", 0)]
    sdfg.validate()
    x = np.random.default_rng(2).random((3, N))
    y = np.zeros(N)
    sdfg(x=x, p_sf=np.zeros((3, N)), c_sf=np.zeros((3, N)), y=y)
    np.testing.assert_allclose(y, _aliased(x, ranged=True))


def test_images_of_different_extents_are_kept_apart():
    sdfg = _members()
    sdfg.arrays["c_sf"].set_shape([3, N + 1])
    assert fold_member_aliases(sdfg, [("p_sf", 0, "c_sf", 0)]) == []


def test_an_interstate_read_of_a_moved_element_reads_the_other_image():
    """``s = p(1)%sf(2)``, assigned on an interstate edge after the boundary and added to y: once p(1)'s accesses are
    c(1)'s, the assignment reads c(1) too."""
    sdfg = _members()
    use = next(s for s in sdfg.states() if s.label == "use")
    edge = sdfg.in_edges(use)[0]
    edge.data.assignments = {"s": "p_sf[0, 1]"}
    sdfg.add_symbol("s", dace.float64)
    tasklet = next(n for n in use.nodes() if isinstance(n, dace.nodes.Tasklet))
    tasklet.code = dace.properties.CodeBlock("w = a + d + s")
    sdfg.validate()
    assert fold_member_aliases(sdfg, [("p_sf", 0, "c_sf", 0)]) == [("p_sf", 0, "c_sf", 0)]
    assert sdfg.in_edges(use)[0].data.assignments == {"s": "c_sf[0, 1]"}
    sdfg.validate()
    x = np.random.default_rng(3).random((3, N))
    y = np.zeros(N)
    sdfg(x=x, p_sf=np.zeros((3, N)), c_sf=np.zeros((3, N)), y=y)
    c0 = x[0].copy()
    c0[0] = c0[1]
    np.testing.assert_allclose(y, c0 + 2 * x[1] + c0 + c0[1])
