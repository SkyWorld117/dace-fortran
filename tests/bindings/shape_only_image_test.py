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
