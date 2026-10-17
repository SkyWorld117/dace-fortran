# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The device-resident direction: the caller keeps its arrays on the GPU with OpenACC, the SDFG's arrays are GPU
storage, and the binding hands over device addresses without staging anything through the host.  A dummy goes
through ``HOST_DATA USE_DEVICE`` as in the ICON lane; a module array is passed as the device address of the caller's
module variable (``acc_deviceptr``); the packed companion of an array of records is packed on the device.
"""

import shutil
import subprocess
from pathlib import Path

import dace
import pytest

from dace_fortran.bindings import FlattenPlan, emit_bindings
from dace_fortran.bindings.acc_transfers import DEVICE, HOST, plan_device_resident_caller
from dace_fortran.bindings.flatten_plan import FlattenEntry, FlattenRecipe, SyntheticGlobal
from dace_fortran.bindings.fortran_interface import (
    DerivedType,
    Member,
    OriginalArg,
    OriginalInterface,
    build_auto_interface,
)
from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenArgKind, FrozenSignature, ModuleOrigin
from dace_fortran.bindings.loop_copy import render_aos_alloc_pack_in, render_aos_alloc_pack_out
from tests._util import build_sdfg

# ----- the pack loops ----------------------------------------------------------


def _rank3_pointer_member() -> FlattenRecipe:
    return FlattenRecipe(
        flat_names=("q_sf",),
        read_exprs=("q($i1)%sf($i2, $i3, $i4)",),
        rank=4,
        shape_exprs=("size(q, dim=1)", "cap_q_sf_d0", "cap_q_sf_d1", "cap_q_sf_d2"),
        aos_alloc=True,
        cap_symbols=("cap_q_sf_d0", "cap_q_sf_d1", "cap_q_sf_d2"),
        lb_symbols=("lb_q_sf_d0", "lb_q_sf_d1", "lb_q_sf_d2"),
        member_pointer=True,
    )


def test_device_pack_in_runs_openacc_loops_over_present_members():
    text = "\n".join(render_aos_alloc_pack_in(_rank3_pointer_member(), "q", device=True))
    assert "allocate(q_sf(size(q, dim=1), cap_q_sf_d0, cap_q_sf_d1, cap_q_sf_d2))" in text
    assert "!$acc enter data create(q_sf)" in text
    assert "!$acc parallel loop collapse(3) default(present)" in text
    assert ("q_sf(i1, q_sf_off1 + i2, q_sf_off2 + i3, q_sf_off3 + i4) = q(i1)%sf(lb_q_sf_d0 + q_sf_off1 + i2 - 1, "
            "lb_q_sf_d1 + q_sf_off2 + i3 - 1, lb_q_sf_d2 + q_sf_off3 + i4 - 1)") in text
    # The trip counts come from host scalars, never from the member's descriptor in the loop bounds.
    assert "q_sf_ext3 = size(q(i1)%sf, 3)" in text and "do i4 = 1, q_sf_ext3" in text
    assert text.index("q_sf_ext1 = size(q(i1)%sf, 1)") < text.index("!$acc parallel loop collapse(3)")
    assert "if (associated(q(i1)%sf))" in text
    # The buffer is zeroed on the device only when an element leaves padding behind.
    assert "if (q_sf_padded) then" in text
    assert "!$acc kernels default(present)" in text
    assert "q_sf = 0" in text
    # No host-side copy of the data.
    assert "= q(i1)%sf\n" not in text


def test_device_pack_out_mirrors_the_pack_in_and_keeps_the_device_copy():
    text = "\n".join(render_aos_alloc_pack_out(_rank3_pointer_member(), "q", device=True))
    assert ("q(i1)%sf(lb_q_sf_d0 + q_sf_off1 + i2 - 1, lb_q_sf_d1 + q_sf_off2 + i3 - 1, lb_q_sf_d2 + q_sf_off3 + i4 - 1)"
            " = q_sf(i1, q_sf_off1 + i2, q_sf_off2 + i3, q_sf_off3 + i4)") in text
    assert "exit data delete" not in text and "deallocate(q_sf)" not in text


def test_device_pack_in_sets_the_image_up_only_when_its_layout_changes():
    """The image and its device copy are kept from one call to the next: allocation, device creation and zeroing
    sit in a branch taken when the saved layout differs from the current one; the packing runs on every call."""
    text = "\n".join(render_aos_alloc_pack_in(_rank3_pointer_member(), "q", device=True))
    setup = text[text.index("if (q_sf_fresh) then") : text.index("deallocate(q_sf_layout_now)")]
    for line in ("allocate(q_sf(size(q, dim=1), cap_q_sf_d0, cap_q_sf_d1, cap_q_sf_d2))",
                 "!$acc enter data create(q_sf)", "q_sf = 0", "!$acc exit data delete(q_sf)", "q_sf_layout = q_sf_layout_now"):
        assert line in setup, line
    assert "q_sf_layout_now(q_sf_layout_k + 2) = lbound(q(i1)%sf, 1)" in text
    assert "q_sf_layout_now(q_sf_layout_k + 7) = ubound(q(i1)%sf, 3)" in text
    assert text.index("deallocate(q_sf_layout_now)") < text.index("!$acc parallel loop collapse(3)")


_KEPT_IMAGE_HARNESS = """module kept_image
  use iso_c_binding
  implicit none
  type :: rec
    real(c_double), pointer :: sf(:, :) => null()
  end type
contains
  subroutine pack(q, probe)
    type(rec), intent(in) :: q(:)
    real(c_double), intent(out) :: probe(3)
    real(c_double), allocatable, target, save :: q_sf(:, :, :)
    integer(c_int64_t), allocatable, save :: q_sf_layout(:)
    integer(c_int64_t), allocatable :: q_sf_layout_now(:)
    integer(c_int64_t) :: q_sf_layout_k
    logical :: q_sf_fresh, q_sf_lb_taken, q_sf_padded
    integer(c_int64_t) :: q_sf_ext1, q_sf_ext2, q_sf_off1, q_sf_off2
    integer(c_int64_t) :: cap_q_sf_d0, cap_q_sf_d1, lb_q_sf_d0, lb_q_sf_d1
    integer :: i1, i2, i3
{body}
    ! the corner of element 2, padding as long as its member is the smaller one; a value of element 1
    probe(1) = q_sf(2, cap_q_sf_d0, cap_q_sf_d1)
    probe(2) = q_sf(1, 1, 1)
    probe(3) = merge(1.0_c_double, 0.0_c_double, q_sf_fresh)
    ! mark the padding: a kept image must still hold the mark in the next call
    q_sf(2, cap_q_sf_d0, cap_q_sf_d1) = -7.0_c_double
  end subroutine
end module

program main
  use kept_image
  implicit none
  type(rec) :: q(2)
  real(c_double) :: p(3)
  allocate(q(1)%sf(4, 3), q(2)%sf(2, 3))
  q(1)%sf = 1.0_c_double
  q(2)%sf = 2.0_c_double
  call pack(q, p)
  print '(3f6.1)', p
  q(1)%sf = 5.0_c_double
  call pack(q, p)
  print '(3f6.1)', p
  deallocate(q(2)%sf)
  allocate(q(2)%sf(3, 3))
  q(2)%sf = 2.0_c_double
  call pack(q, p)
  print '(3f6.1)', p
end program
"""


def _gfortran_compiles_openacc() -> bool:
    """gfortran accepts OpenACC directives (they run on the host without an offload target)."""
    if shutil.which("gfortran") is None:
        return False
    probe = subprocess.run(
        ["gfortran", "-fopenacc", "-fsyntax-only", "-x", "f95", "-"],
        check=False,
        input="program p\n!$acc kernels\nprint *, 1\n!$acc end kernels\nend program p\n",
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def test_a_kept_image_is_repacked_and_set_up_again_only_on_a_new_layout(tmp_path: Path):
    """Compiled and run (OpenACC on the host): the first call sets the image up (padding zero), the second packs the
    new values into the kept image (the mark left in the padding survives), the third sees an element's member
    grow and sets the image up again (padding zeroed, the mark gone)."""
    if not _gfortran_compiles_openacc():
        pytest.skip("needs gfortran with OpenACC directives (-fopenacc)")
    recipe = FlattenRecipe(
        flat_names=("q_sf",),
        read_exprs=("q($i1)%sf($i2, $i3)",),
        rank=3,
        shape_exprs=("size(q, dim=1)", "cap_q_sf_d0", "cap_q_sf_d1"),
        aos_alloc=True,
        cap_symbols=("cap_q_sf_d0", "cap_q_sf_d1"),
        lb_symbols=("lb_q_sf_d0", "lb_q_sf_d1"),
        member_pointer=True,
    )
    body = "\n".join(render_aos_alloc_pack_in(recipe, "q", device=True))
    (tmp_path / "kept.f90").write_text(_KEPT_IMAGE_HARNESS.format(body=body))
    subprocess.run(["gfortran", "-fopenacc", "-ffree-line-length-none", "kept.f90", "-o", "kept"], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    out = subprocess.run(["./kept"], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.split()
    assert [float(x) for x in out] == [0.0, 1.0, 1.0, -7.0, 5.0, 0.0, 0.0, 5.0, 1.0]


def test_host_pack_is_unchanged():
    text = "\n".join(render_aos_alloc_pack_in(_rank3_pointer_member(), "q"))
    assert "!$acc" not in text
    assert ("q_sf(i1, lbound(q(i1)%sf, 1) - lb_q_sf_d0 + 1:ubound(q(i1)%sf, 1) - lb_q_sf_d0 + 1, "
            "lbound(q(i1)%sf, 2) - lb_q_sf_d1 + 1:ubound(q(i1)%sf, 2) - lb_q_sf_d1 + 1, "
            "lbound(q(i1)%sf, 3) - lb_q_sf_d2 + 1:ubound(q(i1)%sf, 3) - lb_q_sf_d2 + 1) = q(i1)%sf") in text


# ----- planning ----------------------------------------------------------------


def _array(name: str, intent: str = "in", shape: tuple[str, ...] = ("n",)) -> FrozenArg:
    return FrozenArg(
        fortran_name=name,
        sdfg_name=name,
        kind=FrozenArgKind.ARRAY,
        dtype="float64",
        rank=1,
        shape=shape,
        intent=intent,
    )


def _synthetic_sdfg(storage: dict[str, dace.StorageType]) -> dace.SDFG:
    sdfg = dace.SDFG("k")
    sdfg.add_symbol("n", dace.int64)
    for name, st in storage.items():
        sdfg.add_array(name, shape=(dace.symbol("n"),), dtype=dace.float64, transient=False, storage=st)
    return sdfg


def test_each_kind_of_array_takes_its_own_route():
    gpu, cpu = dace.StorageType.GPU_Global, dace.StorageType.CPU_Heap
    sdfg = _synthetic_sdfg({"x": gpu, "gam": gpu, "nb": cpu, "q_sf": gpu, "t_sf": gpu, "s_w": gpu, "fp_g": gpu})
    frozen = FrozenSignature(
        entry="k",
        mangled="_QPk",
        args=(
            _array("x", "inout"),
            _array("gam"),
            _array("nb", shape=("1",)),
            _array("q_sf", "inout"),
            _array("t_sf", "inout"),
            _array("s_w", "in"),
            _array("fp_g", "in"),
        ),
        free_symbols=("n",),
        module_symbol_origins={"gam": ModuleOrigin("m_par", "gam"), "nb": ModuleOrigin("m_par", "nb")},
    )
    iface = OriginalInterface(
        entry="k",
        args=(
            OriginalArg(name="x", fortran_type="real(c_double)", rank=1, shape=("n",), intent="inout"),
            OriginalArg(name="q", fortran_type="type(field)", rank=1, shape=("n",), intent="inout"),
            OriginalArg(name="t", fortran_type="type(field)", rank=0, intent="inout", struct_type="field"),
            OriginalArg(name="s", fortran_type="type(rec)", rank=0, intent="in"),
        ),
        struct_types={
            "field": DerivedType(
                name="field",
                module="field_types",
                members=(Member(name="sf", fortran_type="real(c_double)", rank=3, alloc="pointer"),),
            )
        },
    )
    plan = FlattenPlan(
        synthetic_globals=(SyntheticGlobal(symbol="fp_g", module="m_par", entity="fp", member="g"),),
        entries=(
            FlattenEntry(
                outer_expr="q", outer_type="type(field)", writeback_intent="inout", recipe=_rank3_pointer_member()
            ),
            FlattenEntry(
                outer_expr="t",
                outer_type="type(field)",
                writeback_intent="inout",
                recipe=FlattenRecipe(flat_names=("t_sf",), read_exprs=("t%sf",), rank=3, aliasable=True),
            ),
            FlattenEntry(
                outer_expr="s",
                outer_type="type(rec)",
                writeback_intent="",
                recipe=FlattenRecipe(flat_names=("s_w",), read_exprs=("s%w($i1)",), rank=1, shape_exprs=("size(s%w)",)),
            ),
        ),
    )
    acc = plan_device_resident_caller(sdfg, frozen, iface, plan)
    assert acc.use_device == ("x", "q_sf", "t_sf", "s_w", "fp_g")  # dummy, device-packed, alias, deep copy, lifted
    assert acc.deviceptr == ("gam",)  # a module array: the caller's device copy
    assert acc.device_pack == ("q_sf",)
    # The deep copy and the lifted member read are wrapper-side host buffers; t%sf may be disassociated, and the
    # scratch its alias is then bound to needs a device address.
    assert acc.create == ("presence_scratch_float64",)
    assert acc.data_region == (("COPYIN", "s_w"), ("COPYIN", "fp_g"), ("CREATE", "presence_scratch_float64"))
    assert acc.storage["nb"] is HOST and acc.storage["gam"] is DEVICE  # the lifted module scalar stays on the host
    assert acc.active


def test_module_record_array_images_pack_on_the_device():
    """The image of a deferred-shape member of a module array of records is the caller's device data; the image of a
    fixed-shape value member is host data."""
    gpu = dace.StorageType.GPU_Global
    sdfg = _synthetic_sdfg({"qp_vf_sf": gpu, "idw_beg": gpu})
    deferred = FrozenArg(
        fortran_name="qp_vf_sf", sdfg_name="qp_vf_sf", kind=FrozenArgKind.ARRAY, dtype="float64", rank=4,
        shape=("qp_vf_sf_d0", "qp_vf_sf_d1", "qp_vf_sf_d2", "qp_vf_sf_d3"), is_written=True,
        aos_origin_mod="m_fields", aos_origin_struct="qp % vf", aos_member_path="sf", aos_outer_rank=1,
        aos_member_pointer=True,
    )
    static = FrozenArg(
        fortran_name="idw_beg", sdfg_name="idw_beg", kind=FrozenArgKind.ARRAY, dtype="float64", rank=2,
        shape=("idw_beg_d0", "3"), aos_origin_mod="m_bounds", aos_origin_struct="idw", aos_member_path="beg",
        aos_outer_rank=1, aos_struct_static=True,
    )
    frozen = FrozenSignature(entry="k", mangled="_QPk", args=(deferred, static), free_symbols=("n",))
    iface = OriginalInterface(entry="k", args=())
    acc = plan_device_resident_caller(sdfg, frozen, iface, FlattenPlan(entries=()))
    assert acc.device_pack == ("qp_vf_sf",)
    assert acc.use_device == ("qp_vf_sf", "idw_beg")
    assert acc.data_region == (("COPY", "idw_beg"),)  # host data, staged as before


def test_a_member_of_a_module_record_takes_the_device_lane():
    """A deferred-shape member of one module record (``divu%sf``) is the caller's device data too: it is copied into
    the buffer's device copy and back into the member's on the device -- no transfer through the host, and the
    caller's device copy has the result."""
    from dace_fortran.bindings.block_builders import _render_aos_copy_in, _render_aos_copy_out

    sdfg = _synthetic_sdfg({"divu_sf": dace.StorageType.GPU_Global})
    member = FrozenArg(
        fortran_name="divu_sf", sdfg_name="divu_sf", kind=FrozenArgKind.ARRAY, dtype="float64", rank=3,
        shape=("divu_sf_d0", "divu_sf_d1", "divu_sf_d2"), is_written=True,
        aos_origin_mod="m_bubbles", aos_origin_struct="divu", aos_member_path="sf", aos_outer_rank=0,
        aos_member_pointer=True,
    )
    frozen = FrozenSignature(entry="k", mangled="_QPk", args=(member,), free_symbols=("n",))
    acc = plan_device_resident_caller(sdfg, frozen, OriginalInterface(entry="k", args=()), FlattenPlan(entries=()))
    assert acc.device_pack == ("divu_sf",)
    assert acc.data_region == ()  # no COPY through the host
    copy_in = "\n".join(_render_aos_copy_in(member, device=True))
    assert "!$acc enter data create(divu_sf)" in copy_in
    assert copy_in.index("!$acc kernels default(present)") < copy_in.index("divu_sf = divu%sf")
    copy_out = "\n".join(_render_aos_copy_out(member, device=True))
    assert copy_out.index("!$acc kernels default(present)") < copy_out.index("divu%sf = divu_sf")
    assert "!$acc exit data delete(divu_sf)" in copy_out
    # A host caller keeps the host copies.
    host = "\n".join([*_render_aos_copy_in(member), *_render_aos_copy_out(member)])
    assert "!$acc" not in host and "divu_sf = divu%sf" in host


def test_a_host_sdfg_plans_nothing():
    cpu = dace.StorageType.CPU_Heap
    sdfg = _synthetic_sdfg({"x": cpu})
    frozen = FrozenSignature(entry="k", mangled="_QPk", args=(_array("x", "inout"),), free_symbols=("n",))
    iface = OriginalInterface(
        entry="k", args=(OriginalArg(name="x", fortran_type="real(c_double)", rank=1, shape=("n",)),)
    )
    assert not plan_device_resident_caller(sdfg, frozen, iface, FlattenPlan(entries=())).active


# ----- the emitted wrapper -----------------------------------------------------

_KERNEL = """
module field_types
  implicit none
  type field
    real(8), pointer, dimension(:, :, :) :: sf => null()
  end type field
end module field_types

module params
  implicit none
  real(8), allocatable :: gam(:)
  integer :: nb = 2
end module params

module kern
  implicit none
contains
  subroutine convert(n, q, p, t)
    use field_types, only: field
    use params, only: gam, nb
    integer, intent(in) :: n
    type(field), dimension(n), intent(in) :: q
    type(field), dimension(n), intent(inout) :: p
    type(field), intent(inout) :: t
    integer :: i, j, k, l
    do i = 1, n
      do l = lbound(q(i)%sf, 3), ubound(q(i)%sf, 3)
        do k = lbound(q(i)%sf, 2), ubound(q(i)%sf, 2)
          do j = lbound(q(i)%sf, 1), ubound(q(i)%sf, 1)
            p(i)%sf(j, k, l) = gam(i)*q(i)%sf(j, k, l) + nb
            t%sf(j, k, l) = p(i)%sf(j, k, l)
          end do
        end do
      end do
    end do
  end subroutine convert
end module kern
"""


def _device_resident_binding(tmp_path: Path) -> str:
    """Build the kernel, move every non-trivial array to the GPU, emit the binding for a device-resident caller."""
    builder = build_sdfg(_KERNEL, tmp_path / "sdfg", name="convert", entry="kern::convert")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "convert"
    for desc in sdfg.arrays.values():
        if not desc.transient and isinstance(desc, dace.data.Array) and desc.total_size != 1:
            desc.storage = dace.StorageType.GPU_Global
    frozen = sdfg._frozen_signature
    iface = build_auto_interface(sdfg._fortran_interface_raw, "convert")
    acc = plan_device_resident_caller(sdfg, frozen, iface, plan)
    out = tmp_path / "convert_bindings.f90"
    emit_bindings(frozen, iface, plan, out, acc_residency=acc)
    return out.read_text()


def test_the_wrapper_passes_device_copies_and_packs_on_the_device(tmp_path: Path):
    text = _device_resident_binding(tmp_path)
    assert "use openacc, only: acc_deviceptr" in text
    # The module array: the caller's device copy, no host copy; null while gam is unallocated.
    assert "if (allocated(gam__mod)) gam__dev = transfer(acc_deviceptr(gam__mod), c_null_ptr)" in text
    assert "gam__dev = c_null_ptr" in text
    assert "gam = gam__mod" not in text and "allocate(gam(" not in text
    # The lifted module scalar still comes from the host.
    assert "nb = nb__mod" in text or "nb(1) = nb__mod" in text or "nb = int(nb__mod" in text
    # The records' members are packed on the device and handed over inside HOST_DATA.
    assert "!$acc enter data create(q_sf)" in text and "!$acc enter data create(p_sf)" in text
    assert "!$acc parallel loop collapse(3) default(present)" in text
    assert "!$acc exit data delete(p_sf)" in text
    use_device = text.split("USE_DEVICE(")[1].split(")")[0]
    assert "!$ACC HOST_DATA USE_DEVICE(" in text and "q_sf" in use_device and "t_sf" in use_device
    # Nothing is staged through the host; the only device copy the wrapper makes is the scratch a disassociated
    # t%sf's alias is bound to.
    assert "t_sf(1:1, 1:1, 1:1) => presence_scratch_float64" in text
    data_region = text[text.index("!$ACC DATA") :].splitlines()
    assert data_region[1].strip() == "!$ACC   CREATE(presence_scratch_float64)"
    assert not any(clause in text for clause in ("COPYIN(", "COPYOUT(", "COPY("))


def _gfortran_has_openacc() -> bool:
    if shutil.which("gfortran") is None:
        return False
    probe = subprocess.run(
        ["gfortran", "-fopenacc", "-fsyntax-only", "-"],
        check=False,
        input="program p\nuse openacc\nend program p\n",
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def test_the_wrapper_compiles_with_openacc(tmp_path: Path):
    if not _gfortran_has_openacc():
        pytest.skip("needs gfortran with OpenACC support (the openacc module)")
    _device_resident_binding(tmp_path)
    (tmp_path / "kernel.f90").write_text(_KERNEL)
    subprocess.run(
        ["gfortran", "-fopenacc", "-ffree-line-length-none", "-c", "kernel.f90", "convert_bindings.f90"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
