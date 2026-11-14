"""Constant-pool tests -- array/scalar literals on the RHS.

Flang lowers array literals to read-only globals (``_QQro.<count>x<dtype>.<n>``)
carrying dense init data on a ``fir.global`` op.  The bridge must detect the
``hlfir.declare``/``parameter`` -> ``fir.address_of`` chain, extract the dense
values, and synthesise an SDFG init state -- otherwise the kernel's reads
silently return zero (transient registered but uninitialised).
"""

from pathlib import Path

import numpy as np

from tests._util import build_sdfg, f2py_compile


def _build(src: str, tmp: Path, name: str = "main", entry: str | None = None):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


def test_scalar_literal_assigned_then_used(tmp_path: Path):
    """Scalar literal RHS baseline -- no constant pool involved (lowers to
    ``arith.constant``)."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out
  real(8) :: x
  x = 2.0d0
  out = x + 1.0d0
end subroutine
"""
    mod = f2py_compile(src, tmp_path / "ref", "scalar_lit_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float64)

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    assert out[0] == 3.0


def test_array_literal_assigned_then_used(tmp_path: Path):
    """Array literal RHS exercises the ``_QQro.NxrK.M`` constant pool
    (``fir.address_of`` + ``hlfir.declare parameter`` + ``hlfir.assign``); bridge must
    surface the global's dense init data before the assign fires."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(3)
  real(8) :: x(3)
  x = (/ 2.0d0, 3.0d0, 4.0d0 /)
  out(1) = x(1) + 1.0d0
  out(2) = x(2) + 1.0d0
  out(3) = x(3) + 1.0d0
end subroutine
"""
    mod = f2py_compile(src, tmp_path / "ref", "array_lit_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float64)

    sdfg = _build(src, tmp_path)
    out = np.zeros(3, order="F", dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [3.0, 4.0, 5.0])


def test_integer_array_literal_used_as_index_source(tmp_path: Path):
    """Integer array literal feeding a downstream gather pins the int32
    constant-pool path (``_QQro.3xi4.N``)."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out(3)
  integer :: idx(3)
  idx = (/ 10, 20, 30 /)
  out(1) = idx(1) * 2
  out(2) = idx(2) * 2
  out(3) = idx(3) * 2
end subroutine
"""
    mod = f2py_compile(src, tmp_path / "ref", "int_array_lit_ref")
    out_ref = np.asarray(mod.main(), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(3, order="F", dtype=np.int32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [20, 40, 60])


def test_indirect_index_with_symbol_index(tmp_path: Path):
    """``a(idx(j))`` with runtime symbol ``j``: indirect load, handled by
    ``collect_indirect`` (``<arr>_at<gid>``), not the ``__sym_<arr>_<n>`` constant-pool
    path used for ``idx(1)``."""
    src = """
subroutine main(out, j)
  implicit none
  integer, intent(in) :: j
  real(8), intent(out) :: out
  real(8) :: a(10)
  integer :: idx(5)
  integer :: i
  do i = 1, 10
    a(i) = real(i, 8)
  end do
  do i = 1, 5
    idx(i) = i * 2
  end do
  out = a(idx(j))
end subroutine
"""
    mod = f2py_compile(src, tmp_path / "ref", "indirect_sym_idx_ref")
    out_ref = np.asarray(mod.main(3), dtype=np.float64)

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out, j=3)
    np.testing.assert_array_equal(out, out_ref)
    # idx(3) = 6, a(6) = 6.0.
    assert out[0] == 6.0


def test_indirect_index_with_local_scalar_symbol(tmp_path: Path):
    """Same as above but the symbol is a LOCAL scalar, not a dummy: a scalar feeding an
    ``hlfir.designate`` index is promoted to a SYMBOL, so ``sval = 3 -> idx(sval) ->
    idx_at<gid>`` flows through the symbol-side machinery."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out
  real(8) :: a(10)
  integer :: idx(5)
  integer :: sval
  integer :: i
  do i = 1, 10
    a(i) = real(i, 8)
  end do
  do i = 1, 5
    idx(i) = i * 2
  end do
  sval = 3
  out = a(idx(sval))
end subroutine
"""
    mod = f2py_compile(src, tmp_path / "ref", "indirect_local_sym_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float64)

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    assert out[0] == 6.0


def test_indirect_index_with_struct_member_symbol(tmp_path: Path):
    """Same as above but the index source is a flat companion of a struct member
    (post-flatten ``s_idx``)."""
    src = """
module lib
  implicit none
  type t
    integer :: idx(5)
  end type t
end module lib

subroutine main(out, j)
  use lib
  implicit none
  integer, intent(in) :: j
  real(8), intent(out) :: out
  real(8) :: a(10)
  type(t) :: s
  integer :: i
  do i = 1, 10
    a(i) = real(i, 8)
  end do
  do i = 1, 5
    s%idx(i) = i * 2
  end do
  out = a(s%idx(j))
end subroutine
"""
    mod = f2py_compile(src, tmp_path / "ref", "indirect_sym_struct_idx_ref")
    out_ref = np.asarray(mod.main(3), dtype=np.float64)

    sdfg = _build(src, tmp_path, entry="main")
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out, j=3)
    np.testing.assert_array_equal(out, out_ref)
    assert out[0] == 6.0


def test_two_distinct_array_literals(tmp_path: Path):
    """Two array literals get sequential flang counters (``_QQro.3xr8.0``/``.1``); bridge
    must handle both side-by-side."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(3)
  real(8) :: a(3), b(3)
  a = (/ 1.0d0, 2.0d0, 3.0d0 /)
  b = (/ 10.0d0, 20.0d0, 30.0d0 /)
  out(1) = a(1) + b(1)
  out(2) = a(2) + b(2)
  out(3) = a(3) + b(3)
end subroutine
"""
    mod = f2py_compile(src, tmp_path / "ref", "two_lit_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float64)

    sdfg = _build(src, tmp_path)
    out = np.zeros(3, order="F", dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [11.0, 22.0, 33.0])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))


def test_multidimensional_pool_and_parameter_keep_fortran_indices(tmp_path):
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(2,3)
  real(8), parameter :: p(2,3) = reshape([1d0,2d0,3d0,4d0,5d0,6d0],[2,3])
  real(8) :: x(2,3)
  integer :: i,j
  x = reshape([10d0,20d0,30d0,40d0,50d0,60d0],[2,3])
  do j=1,3
    do i=1,2
      out(i,j) = x(i,j) + p(3-i,j)
    enddo
  enddo
end subroutine
"""
    sdfg = _build(src, tmp_path)
    out = np.zeros((2,3), dtype=np.float64, order="F")
    sdfg(out=out)
    np.testing.assert_array_equal(out, [[12,34,56],[21,43,65]])


def test_rank_three_data_keeps_readonly_and_written_layouts_equal(tmp_path):
    src = """
subroutine main(out)
  implicit none
  real, intent(out) :: out(2,3,2)
  real :: r(2,3,2), w(2,3,2)
  integer :: i,j,k
  data r /1,2,3,4,5,6,7,8,9,10,11,12/
  data w /101,102,103,104,105,106,107,108,109,110,111,112/
  w(1,1,1) = 999.
  do k=1,2
    do j=1,3
      do i=1,2
        out(i,j,k) = r(i,j,k) + w(i,j,k)
      enddo
    enddo
  enddo
end subroutine
"""
    sdfg = _build(src, tmp_path)
    out = np.zeros((2,3,2), dtype=np.float32, order="F")
    sdfg(out=out)
    expected = (100 + 2*np.arange(1,13,dtype=np.float32)).reshape(2,3,2,order="F")
    expected[0,0,0] = 1000
    np.testing.assert_array_equal(out, expected)
