"""``fold_aliases``: the image of a one-element module array of records folds onto the module array it is, in the
nested SDFGs too; the frozen signature loses the image.   Run: PYTHONPATH=../../dace:../../dace-fortran pytest"""
import numpy as np
import dace
from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenArgKind, FrozenSignature, attach_to_sdfg, \
    get_frozen_signature

from dace_fortran.substitute.fold_aliases import fold_aliases

N = 8


def _sdfg() -> dace.SDFG:
    """``jac`` (module array, (N,)) and the image ``jac_sf_sf`` ((1, d1), offsets d1) of ``jac_sf(1)%sf``: a map
    doubles ``jac`` in place, a nested SDFG adds one to the image's element (the ghost-fill routine of MFC)."""
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
