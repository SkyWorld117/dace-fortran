"""A module array of records the routine alone carries from one call to the next (``acc % vf(i) % sf``), its buffer
kept by the binding between calls (``emit_bindings(kept_images=...)``): packed once, never copied back, so the state
lives in the buffer.  Over several calls the routine's results match plain gfortran, the members themselves keep the
values they had before the first call, and members reallocated to another shape are packed afresh."""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from tests.bindings.module_global_aos_struct_test import _link_pair, _var_meta

_SRC = """
module kept_state
  implicit none
  type scalar_field
    real(8), pointer :: sf(:, :, :) => null()
  end type
  type vector_field
    type(scalar_field), allocatable :: vf(:)
  end type
  type(vector_field) :: acc
contains
  subroutine accumulate(x, y, n)
    integer, intent(in) :: n
    real(8), intent(in) :: x(0:n, 0:n, 0:n)
    real(8), intent(out) :: y(0:n, 0:n, 0:n)
    integer :: i, j, k, l
    do i = 1, 2
      do l = 0, n
        do k = 0, n
          do j = 0, n
            acc%vf(i)%sf(j, k, l) = acc%vf(i)%sf(j, k, l) + i*x(j, k, l)
          end do
        end do
      end do
    end do
    do l = 0, n
      do k = 0, n
        do j = 0, n
          y(j, k, l) = acc%vf(1)%sf(j, k, l) - acc%vf(2)%sf(j, k, l)
        end do
      end do
    end do
  end subroutine accumulate
end module kept_state
"""

# Calls the routine ``calls`` times on the state the driver sets up (for grid size n), every call's result in out.
# reset=1 (re)allocates the state with the initial values first; the members' values after the calls go to state.
_DRIVER = """
subroutine run_accumulate{suffix}(n, calls, reset, x, out, state) bind(c, name='run_accumulate{suffix}')
  use iso_c_binding
  use kept_state, only: acc
  {use_callee}
  implicit none
  integer(c_int), value :: n, calls, reset
  real(c_double), intent(in) :: x(0:n, 0:n, 0:n)
  real(c_double), intent(out) :: out(0:n, 0:n, 0:n, calls), state(0:n, 0:n, 0:n, 2)
  integer :: i, c
  if (reset /= 0) then
    if (allocated(acc%vf)) then
      do i = 1, 2
        deallocate(acc%vf(i)%sf)
      end do
      deallocate(acc%vf)
    end if
    allocate(acc%vf(2))
    do i = 1, 2
      allocate(acc%vf(i)%sf(0:n, 0:n, 0:n))
      acc%vf(i)%sf = 10*i
    end do
  end if
  do c = 1, calls
    call {call}(x, out(:, :, :, c), n)
  end do
  do i = 1, 2
    state(:, :, :, i) = acc%vf(i)%sf
  end do
end subroutine run_accumulate{suffix}
"""


def _build(tmp_path: Path, kept: tuple[str, ...]):
    """Build the SDFG, emit the binding keeping ``kept``, return ``(builder, so_path, binding_path, frozen)``."""
    from dace_fortran.bindings import FlattenPlan, emit_bindings
    from dace_fortran.bindings.fortran_interface import build_auto_interface
    from tests._util import build_sdfg

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_SRC, sdfg_dir, name="accumulate", entry="kept_state::accumulate")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "accumulate"
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)
    binding = tmp_path / "accumulate_bindings.f90"
    emit_bindings(
        sdfg._frozen_signature,
        iface,
        plan,
        str(binding),
        tuple(getattr(compiled, "_sig", None) or ()),
        kept_images=kept,
    )
    return builder, so_path, binding, sdfg._frozen_signature


def _run(fn, n: int, calls: int, reset: bool, x: np.ndarray):
    out = np.zeros((n + 1, n + 1, n + 1, calls), order="F")
    state = np.zeros((n + 1, n + 1, n + 1, 2), order="F")
    fn(
        ctypes.c_int(n),
        ctypes.c_int(calls),
        ctypes.c_int(int(reset)),
        x.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
        state.ctypes.data_as(ctypes.c_void_p),
    )
    return out, state


def test_kept_record_buffer_carries_state_e2e(tmp_path: Path):
    from dace_fortran.bindings.block_builders import keepable

    builder, so_path, binding, frozen = _build(tmp_path, ("acc_vf_sf",))
    assert [a.sdfg_name for a in frozen.args if keepable(a)] == ["acc_vf_sf"]
    meta = _var_meta(builder, "acc_vf_sf")
    assert meta.aos_origin_mod == "kept_state"
    assert meta.is_written is True
    text = binding.read_text()
    assert "allocatable, target, save :: acc_vf_sf" in text
    assert "if (.not. allocated(acc_vf_sf)) then" in text
    assert "SoA->AoS copy-out" not in text  # never copied back
    assert "deallocate(acc_vf_sf)" in text  # only to repack members of another shape

    driver = _DRIVER.format(
        suffix="", use_callee="use accumulate_dace_bindings, only: accumulate_dace", call="accumulate_dace"
    )
    reference = _DRIVER.format(suffix="_ref", use_callee="use kept_state, only: accumulate", call="accumulate")
    sdfg_lib, ref_lib = _link_pair(_SRC, "accumulate", driver, reference, so_path, binding, tmp_path)

    rng = np.random.default_rng(0)
    n, calls = 3, 4
    x = np.asfortranarray(rng.standard_normal((n + 1,) * 3))
    expected = np.stack([-10 - c * x for c in range(1, calls + 1)], axis=-1)  # (10 + c x) - (20 + 2c x)

    ref_out, _ = _run(ref_lib.run_accumulate_ref, n, calls, True, x)
    np.testing.assert_allclose(ref_out, expected, rtol=1e-14)
    out, state = _run(sdfg_lib.run_accumulate, n, calls, True, x)
    np.testing.assert_allclose(out, expected, rtol=1e-14)
    # The routine's state lives in the kept buffer: the members still hold what the driver gave them.
    np.testing.assert_array_equal(state[..., 0], 10)
    np.testing.assert_array_equal(state[..., 1], 20)

    # More calls continue from the buffer, not from the (unchanged) members.
    out, _ = _run(sdfg_lib.run_accumulate, n, 2, False, x)
    np.testing.assert_allclose(out, np.stack([-10 - c * x for c in (5, 6)], axis=-1), rtol=1e-14)

    # Members reallocated to another shape: the buffer is packed afresh from them.
    m = 5
    y = np.asfortranarray(rng.standard_normal((m + 1,) * 3))
    out, _ = _run(sdfg_lib.run_accumulate, m, 2, True, y)
    ref_out, _ = _run(ref_lib.run_accumulate_ref, m, 2, True, y)
    np.testing.assert_allclose(out, ref_out, rtol=1e-14)


def test_kept_images_must_be_module_record_buffers(tmp_path: Path):
    with pytest.raises(ValueError, match="kept_images"):
        _build(tmp_path, ("x",))
