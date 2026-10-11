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


def test_device_pack_out_mirrors_the_pack_in_and_frees_the_device_copy():
    text = "\n".join(render_aos_alloc_pack_out(_rank3_pointer_member(), "q", device=True))
    assert ("q(i1)%sf(lb_q_sf_d0 + q_sf_off1 + i2 - 1, lb_q_sf_d1 + q_sf_off2 + i3 - 1, lb_q_sf_d2 + q_sf_off3 + i4 - 1)"
            " = q_sf(i1, q_sf_off1 + i2, q_sf_off2 + i3, q_sf_off3 + i4)") in text
    assert text.index("!$acc exit data delete(q_sf)") < text.index("deallocate(q_sf)")


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
