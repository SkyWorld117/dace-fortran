"""An array of records that is a component of a module-global record (``qp % vf(i) % sf``, ``qp`` a module variable,
``vf`` an allocatable array of records with a pointer member), written by the kernel directly and through an inlined
callee, marshalled through the generated Fortran binding and compared with plain gfortran."""

import ctypes
from pathlib import Path

import numpy as np

from tests.bindings.module_global_aos_struct_test import _build_module, _link_pair, _var_meta

_SRC = """
module vm
  implicit none
  type scalar_field
    real(8), pointer :: sf(:, :, :) => null()
  end type
  type vector_field
    type(scalar_field), allocatable :: vf(:)
  end type
  type(vector_field) :: qp
contains
  subroutine twice(v, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(2), intent(inout) :: v
    integer :: i, j, k, l
    do i = 1, 2
      do l = 0, n
        do k = 0, n
          do j = 0, n
            v(i)%sf(j, k, l) = 2 * v(i)%sf(j, k, l) + i
          end do
        end do
      end do
    end do
  end subroutine twice

  subroutine copy_twice(q, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(2), intent(in) :: q
    integer :: i, j, k, l
    do i = 1, 2
      do l = 0, n
        do k = 0, n
          do j = 0, n
            qp%vf(i)%sf(j, k, l) = q(i)%sf(j, k, l)
          end do
        end do
      end do
    end do
    call twice(qp%vf, n)
  end subroutine copy_twice
end module vm
"""

_DRIVER = """
subroutine run_copy_twice{suffix}(n, src, out) bind(c, name='run_copy_twice{suffix}')
  use iso_c_binding
  use vm, only: qp, scalar_field
  {use_callee}
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(in), target :: src(0:n, 0:n, 0:n, 2)
  real(c_double), intent(out) :: out(0:n, 0:n, 0:n, 2)
  type(scalar_field) :: q(2)
  integer :: i
  if (allocated(qp%vf)) deallocate(qp%vf)
  allocate(qp%vf(2))
  do i = 1, 2
    allocate(qp%vf(i)%sf(0:n, 0:n, 0:n))
    qp%vf(i)%sf = -1
    q(i)%sf(0:, 0:, 0:) => src(:, :, :, i)
  end do
  call {call}(q, n)
  do i = 1, 2
    out(:, :, :, i) = qp%vf(i)%sf
  end do
end subroutine run_copy_twice{suffix}
"""


def test_module_record_array_component_written_e2e(tmp_path: Path):
    builder, _sdfg, so_path, binding = _build_module(_SRC, "copy_twice", "vm::copy_twice", tmp_path)

    meta = _var_meta(builder, "qp_vf_sf")
    assert meta.aos_origin_mod == "vm"
    assert meta.aos_origin_struct == "qp % vf"
    assert meta.aos_member_path == "sf"
    assert meta.aos_outer_rank == 1
    assert meta.is_written is True

    driver = _DRIVER.format(
        suffix="", use_callee="use copy_twice_dace_bindings, only: copy_twice_dace", call="copy_twice_dace"
    )
    reference = _DRIVER.format(suffix="_ref", use_callee="use vm, only: copy_twice", call="copy_twice")
    sdfg_lib, ref_lib = _link_pair(_SRC, "copy_twice", driver, reference, so_path, binding, tmp_path)

    n = 3
    src = np.asfortranarray(np.random.default_rng(0).standard_normal((n + 1, n + 1, n + 1, 2)))

    def run(fn):
        out = np.zeros_like(src, order="F")
        fn(ctypes.c_int(n), src.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p))
        return out

    expected = 2 * src + np.array([1.0, 2.0])
    np.testing.assert_allclose(run(ref_lib.run_copy_twice_ref), expected, rtol=1e-14)
    np.testing.assert_allclose(run(sdfg_lib.run_copy_twice), expected, rtol=1e-14)
