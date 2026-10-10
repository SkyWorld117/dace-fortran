"""Record subscripts, end to end: kernels that reach array members of arrays of records through several record levels,
constant and varying subscripts, rank-N record arrays, sections of record arrays and members passed to dummies with
their own bounds.

Each kernel is built to an SDFG, called through its generated Fortran binding, and compared with the same kernel
compiled by gfortran. The patterns the bridge does not handle yet are marked xfail (strict) with the exception they
raise today.
"""

import ctypes

import numpy as np
import pytest

from tests.bindings.module_global_aos_struct_test import _build_module, _link_pair

_TYPES = """
  type scalar_field
    real(8), pointer :: sf(:) => null()
  end type
  type vector_field
    type(scalar_field), allocatable :: vf(:)
  end type
"""

# Fills a two-level module or local `fl(2)%vf(2)%sf(n)` with distinct values.
_FILL_TWO_LEVEL = """
  do i = 1, 2
    allocate(fl(i)%vf(2))
    do k = 1, 2
      allocate(fl(i)%vf(k)%sf(n))
      do j = 1, n
        fl(i)%vf(k)%sf(j) = 1000*i + 100*k + j
      end do
    end do
  end do
"""
_OUT_TWO_LEVEL = "do i = 1, 2; do k = 1, 2; out((2*(i-1)+k-1)*n+1:(2*(i-1)+k)*n) = fl(i)%vf(k)%sf; end do; end do"

_KERNELS = {
    # Two different constant elements of a module array of records in one routine.
    "constant_elements": dict(
        src=f"""
module ra_constant_elements
  implicit none
{_TYPES}
  type(scalar_field), allocatable :: q(:)
contains
  subroutine kern(n)
    integer, intent(in) :: n
    integer :: j
    do j = 1, n
      q(3)%sf(j) = q(4)%sf(j) + 10*q(1)%sf(j)
    end do
  end subroutine kern
end module ra_constant_elements
""",
        setup="""
  if (allocated(q)) deallocate(q)
  allocate(q(4))
  do i = 1, 4
    allocate(q(i)%sf(n))
    do j = 1, n
      q(i)%sf(j) = 100*i + j
    end do
  end do
""",
        call="kern(n)",
        out="out(1:n) = q(3)%sf; out(n+1:2*n) = q(4)%sf",
        nout=2,
    ),
    # Two record levels of a module variable, varying subscripts at both, then constant ones.
    "two_levels": dict(
        src=f"""
module ra_two_levels
  implicit none
{_TYPES}
  type(vector_field), allocatable :: fl(:)
contains
  subroutine kern(n)
    integer, intent(in) :: n
    integer :: id, i, j
    do id = 1, 2
      do i = 1, 2
        do j = 1, n
          fl(id)%vf(i)%sf(j) = fl(id)%vf(i)%sf(j) + 10*id + i
        end do
      end do
    end do
    do j = 1, n
      fl(1)%vf(1)%sf(j) = fl(1)%vf(1)%sf(j) + fl(2)%vf(2)%sf(j)
    end do
  end subroutine kern
end module ra_two_levels
""",
        setup="  if (allocated(fl)) deallocate(fl)\n  allocate(fl(2))\n" + _FILL_TWO_LEVEL,
        call="kern(n)",
        out=_OUT_TWO_LEVEL,
        nout=4,
        xfail=(KeyError, "two record levels of a module variable are not registered"),
    ),
    # A constant outer subscript over a varying inner one: fl(1)%vf(i) and fl(2)%vf(i) must stay apart.
    "constant_over_varying": dict(
        src=f"""
module ra_constant_over_varying
  implicit none
{_TYPES}
  type(vector_field), allocatable :: fl(:)
contains
  subroutine kern(n)
    integer, intent(in) :: n
    integer :: i, j
    do i = 1, 2
      do j = 1, n
        fl(1)%vf(i)%sf(j) = fl(1)%vf(i)%sf(j) + fl(2)%vf(i)%sf(j)
      end do
    end do
  end subroutine kern
end module ra_constant_over_varying
""",
        setup="  if (allocated(fl)) deallocate(fl)\n  allocate(fl(2))\n" + _FILL_TWO_LEVEL,
        call="kern(n)",
        out=_OUT_TWO_LEVEL,
        nout=4,
        xfail=(KeyError, "two record levels of a module variable are not registered"),
    ),
    # The same with a dummy root.
    "constant_over_varying_dummy": dict(
        src=f"""
module ra_constant_over_varying_dummy
  implicit none
{_TYPES}
contains
  subroutine kern(fl, n)
    integer, intent(in) :: n
    type(vector_field), dimension(2), intent(inout) :: fl
    integer :: i, j
    do i = 1, 2
      do j = 1, n
        fl(1)%vf(i)%sf(j) = fl(1)%vf(i)%sf(j) + fl(2)%vf(i)%sf(j)
      end do
    end do
  end subroutine kern
end module ra_constant_over_varying_dummy
""",
        local="type(vector_field) :: fl(2)",
        setup=_FILL_TWO_LEVEL,
        call="kern(fl, n)",
        out=_OUT_TWO_LEVEL,
        nout=4,
        xfail=(KeyError, "two record levels of a dummy are not registered"),
    ),
    # A section of a dummy array of records, not starting at its lower bound, passed to a dummy declared from 1.
    "section_argument": dict(
        src=f"""
module ra_section_argument
  implicit none
{_TYPES}
contains
  subroutine inner(v, m, n)
    integer, intent(in) :: m, n
    type(scalar_field), dimension(1:), intent(inout) :: v
    integer :: i, j
    do i = 1, m
      do j = 1, n
        v(i)%sf(j) = 2*v(i)%sf(j) + i
      end do
    end do
  end subroutine inner

  subroutine kern(q, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(4), intent(inout) :: q
    call inner(q(2:3), 2, n)
  end subroutine kern
end module ra_section_argument
""",
        local="type(scalar_field) :: q(4)",
        setup="""
  do i = 1, 4
    allocate(q(i)%sf(n))
    do j = 1, n
      q(i)%sf(j) = 100*i + j
    end do
  end do
""",
        call="kern(q, n)",
        out="do i = 1, 4; out((i-1)*n+1:i*n) = q(i)%sf; end do",
        nout=4,
    ),
    # A strided section of a record-array dummy.
    "strided_section_argument": dict(
        src=f"""
module ra_strided_section_argument
  implicit none
{_TYPES}
contains
  subroutine inner(v, m, n)
    integer, intent(in) :: m, n
    type(scalar_field), dimension(:), intent(inout) :: v
    integer :: i, j
    do i = 1, m
      do j = 1, n
        v(i)%sf(j) = 3*v(i)%sf(j) - i
      end do
    end do
  end subroutine inner

  subroutine kern(q, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(5), intent(inout) :: q
    call inner(q(1:5:2), 3, n)
  end subroutine kern
end module ra_strided_section_argument
""",
        local="type(scalar_field) :: q(5)",
        setup="""
  do i = 1, 5
    allocate(q(i)%sf(n))
    do j = 1, n
      q(i)%sf(j) = 100*i + j
    end do
  end do
""",
        call="kern(q, n)",
        out="do i = 1, 5; out((i-1)*n+1:i*n) = q(i)%sf; end do",
        nout=5,
    ),
    # A section passed to a dummy whose lower bound is 0.
    "section_into_zero_based_dummy": dict(
        src=f"""
module ra_section_into_zero_based_dummy
  implicit none
{_TYPES}
contains
  subroutine inner(v, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(0:), intent(inout) :: v
    integer :: j
    do j = 1, n
      v(0)%sf(j) = v(0)%sf(j) + v(1)%sf(j)
    end do
  end subroutine inner

  subroutine kern(q, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(4), intent(inout) :: q
    call inner(q(3:4), n)
  end subroutine kern
end module ra_section_into_zero_based_dummy
""",
        local="type(scalar_field) :: q(4)",
        setup="""
  do i = 1, 4
    allocate(q(i)%sf(n))
    do j = 1, n
      q(i)%sf(j) = 100*i + j
    end do
  end do
""",
        call="kern(q, n)",
        out="do i = 1, 4; out((i-1)*n+1:i*n) = q(i)%sf; end do",
        nout=4,
    ),
    # A rank-2 dummy array of records with a pointer member.
    "rank2_dummy": dict(
        src=f"""
module ra_rank2_dummy
  implicit none
{_TYPES}
contains
  subroutine kern(b, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(1:2, 1:2), intent(inout) :: b
    integer :: i, k, j
    do k = 1, 2
      do i = 1, 2
        do j = 1, n
          b(i, k)%sf(j) = b(i, k)%sf(j) + 10*i + k
        end do
      end do
    end do
  end subroutine kern
end module ra_rank2_dummy
""",
        local="type(scalar_field) :: b(2, 2)",
        setup="""
  do k = 1, 2
    do i = 1, 2
      allocate(b(i, k)%sf(n))
      do j = 1, n
        b(i, k)%sf(j) = 1000*i + 100*k + j
      end do
    end do
  end do
""",
        call="kern(b, n)",
        out="do k = 1, 2; do i = 1, 2; out((2*(k-1)+i-1)*n+1:(2*(k-1)+i)*n) = b(i, k)%sf; end do; end do",
        nout=4,
    ),
    # A rank-2 dummy array of records with an integer allocatable member of rank 3, read and written.
    "rank2_integer_member": dict(
        src="""
module ra_rank2_integer_member
  implicit none
  type integer_field
    integer, allocatable :: sf(:, :, :)
  end type
contains
  subroutine kern(bc, n)
    integer, intent(in) :: n
    type(integer_field), dimension(1:3, 1:2), intent(inout) :: bc
    integer :: d, s, j
    do s = 1, 2
      do d = 1, 3
        do j = 1, n
          bc(d, s)%sf(j, 0, 1) = bc(d, s)%sf(j, 0, 1) + bc(1, 1)%sf(j, 1, 1) * s + d
        end do
      end do
    end do
  end subroutine kern
end module ra_rank2_integer_member
""",
        local="type(integer_field) :: bc(3, 2)",
        setup="""
  do k = 1, 2
    do i = 1, 3
      allocate(bc(i, k)%sf(n, 0:1, 1))
      bc(i, k)%sf = 0
      do j = 1, n
        bc(i, k)%sf(j, 0, 1) = 100*i + 10*k + j
        bc(i, k)%sf(j, 1, 1) = j
      end do
    end do
  end do
""",
        call="kern(bc, n)",
        out="do k = 1, 2; do i = 1, 3; out((3*(k-1)+i-1)*n+1:(3*(k-1)+i)*n) = bc(i, k)%sf(:, 0, 1); end do; end do",
        nout=6,
    ),
    # A rank-3 dummy array of records with lower bounds other than 1, every element read and written.
    "rank3_dummy_with_bounds": dict(
        src=f"""
module ra_rank3_dummy_with_bounds
  implicit none
{_TYPES}
contains
  subroutine kern(c, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(0:1, 2:3, -1:0), intent(inout) :: c
    integer :: a, b, e, j
    do e = -1, 0
      do b = 2, 3
        do a = 0, 1
          do j = 1, n
            c(a, b, e)%sf(j) = c(a, b, e)%sf(j) * 2 + a + 10*b + 100*e
          end do
        end do
      end do
    end do
  end subroutine kern
end module ra_rank3_dummy_with_bounds
""",
        local="type(scalar_field) :: c(0:1, 2:3, -1:0)",
        setup="""
  do k = -1, 0
    do i = 2, 3
      do m = 0, 1
        allocate(c(m, i, k)%sf(n))
        do j = 1, n
          c(m, i, k)%sf(j) = 1000*m + 100*i + 10*k + j
        end do
      end do
    end do
  end do
""",
        call="kern(c, n)",
        out="""l = 0
  do k = -1, 0; do i = 2, 3; do m = 0, 1; out(l*n+1:(l+1)*n) = c(m, i, k)%sf; l = l + 1; end do; end do; end do""",
        nout=8,
    ),
    # A whole member passed to a dummy with its own lower bound.
    "member_with_lower_bound": dict(
        src=f"""
module ra_member_with_lower_bound
  implicit none
{_TYPES}
contains
  subroutine grad(f, n)
    integer, intent(in) :: n
    real(8), dimension(0:), intent(inout) :: f
    integer :: j
    do j = 0, n - 1
      f(j) = f(j) + j
    end do
  end subroutine grad

  subroutine kern(q, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(2), intent(inout) :: q
    call grad(q(2)%sf, n)
  end subroutine kern
end module ra_member_with_lower_bound
""",
        local="type(scalar_field) :: q(2)",
        setup="""
  do i = 1, 2
    allocate(q(i)%sf(n))
    do j = 1, n
      q(i)%sf(j) = 100*i + j
    end do
  end do
""",
        call="kern(q, n)",
        out="do i = 1, 2; out((i-1)*n+1:i*n) = q(i)%sf; end do",
        nout=2,
    ),
    # A section of a member passed to an assumed-shape dummy that asks for its size and lower bound.
    "member_section_argument": dict(
        src=f"""
module ra_member_section_argument
  implicit none
{_TYPES}
contains
  subroutine scale(f)
    real(8), dimension(:), intent(inout) :: f
    integer :: j
    do j = 1, size(f)
      f(j) = 2*f(j) + lbound(f, 1) + 10*size(f)
    end do
  end subroutine scale

  subroutine kern(q, n)
    integer, intent(in) :: n
    type(scalar_field), dimension(2), intent(inout) :: q
    call scale(q(2)%sf(3:n))
  end subroutine kern
end module ra_member_section_argument
""",
        local="type(scalar_field) :: q(2)",
        setup="""
  do i = 1, 2
    allocate(q(i)%sf(n))
    do j = 1, n
      q(i)%sf(j) = 100*i + j
    end do
  end do
""",
        call="kern(q, n)",
        out="do i = 1, 2; out((i-1)*n+1:i*n) = q(i)%sf; end do",
        nout=2,
    ),
}

_DRIVER = """
subroutine run{suffix}(n, out) bind(c, name='run{suffix}')
  use iso_c_binding
  use {module}
  {use_binding}
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(out) :: out({nout}*n)
  integer :: i, j, k, l, m
  {local}
{setup}
  call {call}
  {out}
end subroutine run{suffix}
"""


def _params():
    for name, kernel in _KERNELS.items():
        marks = []
        if "xfail" in kernel:
            raises, reason = kernel["xfail"]
            marks.append(pytest.mark.xfail(strict=True, raises=raises, reason=reason))
        yield pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize("name", _params())
def test_kernel_matches_gfortran(name, tmp_path):
    kernel = _KERNELS[name]
    module = f"ra_{name}"
    # One SDFG name per kernel: DaCe builds an SDFG in a folder of its name, which parallel tests would share.
    _, _, so_path, binding = _build_module(kernel["src"], module, f"{module}::kern", tmp_path)
    common = dict(
        module=module, nout=kernel["nout"], local=kernel.get("local", ""), setup=kernel["setup"], out=kernel["out"]
    )
    driver = _DRIVER.format(
        suffix="",
        use_binding=f"use {module}_dace_bindings, only: {module}_dace",
        call=kernel["call"].replace("kern(", f"{module}_dace("),
        **common,
    )
    reference = _DRIVER.format(suffix="_ref", use_binding="", call=kernel["call"], **common)
    sdfg_lib, ref_lib = _link_pair(kernel["src"], module, driver, reference, so_path, binding, tmp_path)
    n = 5

    def run(entry):
        out = np.zeros(kernel["nout"] * n)
        entry(ctypes.c_int(n), out.ctypes.data_as(ctypes.c_void_p))
        return out

    np.testing.assert_allclose(run(sdfg_lib.run), run(ref_lib.run_ref), rtol=1e-14, atol=0)
