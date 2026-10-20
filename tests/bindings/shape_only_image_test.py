"""A record image the SDFG allocates itself (scratch: finish_scratch_images) leaves the arguments, but an extent of
it without a recorded value -- one that differs between the ranks of a run -- stays a free symbol: the binding reads
it, and a lower bound, off the record on the host, as the image's leading dims off the record array and the others
off a member of its first element."""
from dace_fortran.bindings import FlattenPlan, FrozenArg, FrozenSignature, OriginalInterface
from dace_fortran.bindings.block_builders import _build_symbol_assigns
from dace_fortran.bindings.frozen_signature import FrozenArgKind


def test_shape_only_image_symbols_read_the_record():
    image = FrozenArg(
        fortran_name="q_cons_qp%vf%sf",
        sdfg_name="q_cons_qp_vf_sf",
        kind=FrozenArgKind.ARRAY,
        dtype="float64",
        rank=4,
        shape=tuple(f"q_cons_qp_vf_sf_d{i}" for i in range(4)),
        intent="inout",
        aos_origin_mod="m_global_parameters",
        aos_origin_struct="q_cons_qp%vf",
        aos_member_path="sf",
        aos_outer_rank=1,
    )
    symbols = ("q_cons_qp_vf_sf_d0", "q_cons_qp_vf_sf_d3", "offset_q_cons_qp_vf_sf_d3")
    frozen = FrozenSignature(entry="kernel", mangled="_QPkernel", args=(), free_symbols=symbols,
                             shape_only_args=(image,))
    iface = OriginalInterface(entry="kernel", args=(), struct_types={}, used_modules={})
    text = "\n".join(_build_symbol_assigns(frozen, FlattenPlan(entries=()), set(), iface, passed=set(symbols)))
    assert "q_cons_qp_vf_sf_d0 = int(size(q_cons_qp%vf, dim=1), c_int)" in text
    assert "q_cons_qp_vf_sf_d3 = int(size(q_cons_qp%vf(lbound(q_cons_qp%vf, 1))%sf, dim=3), c_int)" in text
    assert "offset_q_cons_qp_vf_sf_d3 = int(lbound(q_cons_qp%vf(lbound(q_cons_qp%vf, 1))%sf, dim=3), c_int)" in text
    assert "allocated(q_cons_qp%vf)" in text and "TODO" not in text


def test_shape_only_image_survives_the_snapshot():
    image = FrozenArg(fortran_name="qp%vf%sf", sdfg_name="qp_vf_sf", kind=FrozenArgKind.ARRAY, dtype="float64",
                      rank=2, shape=("qp_vf_sf_d0", "qp_vf_sf_d1"), aos_origin_mod="m", aos_origin_struct="qp%vf",
                      aos_member_path="sf", aos_outer_rank=1)
    frozen = FrozenSignature(entry="kernel", mangled="_QPkernel", args=(), shape_only_args=(image,))
    assert FrozenSignature.from_dict(frozen.to_dict()) == frozen


def test_shape_only_image_under_a_root_array_of_records_reads_its_first_elements():
    """MFC's flux_n(dir)%vf(var)%sf: the root's dims, then the segment's, then the member's, each read off the first
    element of the level above (on ranks of unequal shares the member's extent differs between the ranks)."""
    image = FrozenArg(
        fortran_name="flux_n%vf%sf",
        sdfg_name="flux_n_vf_sf",
        kind=FrozenArgKind.ARRAY,
        dtype="float64",
        rank=5,
        shape=tuple(f"flux_n_vf_sf_d{i}" for i in range(5)),
        intent="inout",
        aos_origin_mod="m_rhs",
        aos_origin_struct="flux_n%vf",
        aos_member_path="sf",
        aos_outer_rank=1,
        aos_root_rank=1,
    )
    symbols = ("flux_n_vf_sf_d0", "flux_n_vf_sf_d1", "flux_n_vf_sf_d2", "offset_flux_n_vf_sf_d2")
    frozen = FrozenSignature(entry="kernel", mangled="_QPkernel", args=(), free_symbols=symbols,
                             shape_only_args=(image,))
    iface = OriginalInterface(entry="kernel", args=(), struct_types={}, used_modules={})
    text = "\n".join(_build_symbol_assigns(frozen, FlattenPlan(entries=()), set(), iface, passed=set(symbols)))
    assert "flux_n_vf_sf_d0 = int(size(flux_n, dim=1), c_int)" in text
    assert "flux_n_vf_sf_d1 = int(size(flux_n(lbound(flux_n, 1))%vf, dim=1), c_int)" in text
    member = "flux_n(lbound(flux_n, 1))%vf(lbound(flux_n(lbound(flux_n, 1))%vf, 1))%sf"
    assert f"flux_n_vf_sf_d2 = int(size({member}, dim=1), c_int)" in text
    assert f"offset_flux_n_vf_sf_d2 = int(lbound({member}, dim=1), c_int)" in text
    assert "allocated(flux_n)" in text and "TODO" not in text
