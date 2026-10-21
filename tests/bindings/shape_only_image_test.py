"""A record image the SDFG allocates itself (scratch: finish_scratch_images) leaves the arguments, but an extent of
it without a recorded value -- one that differs between the ranks of a run -- stays a free symbol: the binding reads
it, and a lower bound, off the record on the host, as a packed image is laid out: the leading dims off the record
array, the others spanning the bounds of every element's member."""
import subprocess

from dace_fortran.bindings import FlattenPlan, FrozenArg, FrozenSignature, OriginalInterface
from dace_fortran.bindings.block_builders import _aos_loop_pieces, _build_symbol_assigns
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
    assert "min(aos_q_cons_qp_vf_sf_l2, lbound(q_cons_qp%vf(aos_q_cons_qp_vf_sf_i0)%sf, 3))" in text
    assert "q_cons_qp_vf_sf_d3 = int(aos_q_cons_qp_vf_sf_u2 - aos_q_cons_qp_vf_sf_l2 + 1, c_int)" in text
    assert "offset_q_cons_qp_vf_sf_d3 = int(aos_q_cons_qp_vf_sf_l2, c_int)" in text
    assert "allocated(q_cons_qp%vf)" in text and "TODO" not in text


def test_shape_only_image_survives_the_snapshot():
    image = FrozenArg(fortran_name="qp%vf%sf", sdfg_name="qp_vf_sf", kind=FrozenArgKind.ARRAY, dtype="float64",
                      rank=2, shape=("qp_vf_sf_d0", "qp_vf_sf_d1"), aos_origin_mod="m", aos_origin_struct="qp%vf",
                      aos_member_path="sf", aos_outer_rank=1)
    frozen = FrozenSignature(entry="kernel", mangled="_QPkernel", args=(), shape_only_args=(image,))
    assert FrozenSignature.from_dict(frozen.to_dict()) == frozen


def test_shape_only_extents_span_every_element_under_a_root_array_of_records(tmp_path):
    """MFC's flux_n(dir)%vf(var)%sf: the root's dims, then the largest segment, then each member dim spanning every
    element's bounds -- the directions' elements differ (each keeps its own axis first), so the first element's
    would not hold the others. Compiled and run against such records."""
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
    symbols = ("flux_n_vf_sf_d0", "flux_n_vf_sf_d1", "flux_n_vf_sf_d2", "flux_n_vf_sf_d3", "offset_flux_n_vf_sf_d3")
    frozen = FrozenSignature(entry="kernel", mangled="_QPkernel", args=(), free_symbols=symbols,
                             shape_only_args=(image,))
    iface = OriginalInterface(entry="kernel", args=(), struct_types={}, used_modules={})
    lines = _build_symbol_assigns(frozen, FlattenPlan(entries=()), set(), iface, passed=set(symbols))
    pieces = _aos_loop_pieces(image)
    names = [*symbols, *pieces.loop_vars, *pieces.lower_vars, *pieces.upper_vars, *pieces.segment_caps]
    newline = "\n"
    (tmp_path / "probe.f90").write_text(f"""
module m_rhs
  implicit none
  type :: field
    real(8), allocatable :: sf(:, :, :)
  end type
  type :: fields
    type(field), allocatable :: vf(:)
  end type
  type(fields), allocatable :: flux_n(:)
end module
program probe
  use iso_c_binding
  use m_rhs
  implicit none
  integer(c_int) :: {", ".join(names)}
  integer :: v
  allocate(flux_n(2))
  allocate(flux_n(1)%vf(3), flux_n(2)%vf(3))
  do v = 1, 3
    allocate(flux_n(1)%vf(v)%sf(-1:8, 0:3, 0:0), flux_n(2)%vf(v)%sf(0:3, -1:8, 0:0))
  end do
{newline.join(lines)}
  print *, {", ".join(symbols)}
end program
""")
    subprocess.check_call(["gfortran", "-ffree-line-length-none", "-o", "probe", "probe.f90"], cwd=tmp_path)
    values = [int(v) for v in subprocess.check_output([str(tmp_path / "probe")]).split()]
    assert values == [2, 3, 10, 10, -1]  # the first element alone: 4 and 0 in the second member dim
