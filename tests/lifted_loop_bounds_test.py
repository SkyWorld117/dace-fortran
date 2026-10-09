"""A DO that Flang lowers unstructured -- a CYCLE inside an IF -- is lifted to a while loop whose start and trip count
are plain scalar stores.  Bounds that read array elements (``lo(2)``, ``ib(3)%beg``) must keep their subscripts there,
like the structured loop's bounds do."""

import numpy as np

from tests._util import build_sdfg


def _expected(x: np.ndarray, jlo: int, jhi: int, klo: int, khi: int) -> float:
    sub = x[jlo - 1 : jhi, klo - 1 : khi]
    return float(np.where(sub > 50.0, sub, 2.0 * sub).sum())


def test_lifted_loop_bounds_from_array_elements(tmp_path):
    src = """
subroutine kb(lo, hi, x, out)
  implicit none
  integer, intent(in) :: lo(2), hi(2)
  real(8), intent(in) :: x(10, 10)
  real(8), intent(out) :: out
  integer :: j, k
  out = 0
  do k = lo(2), hi(2)
    do j = lo(1), hi(1)
      if (x(j, k) > 50.0d0) then
        out = out + x(j, k)
        cycle
      end if
      out = out + 2.0d0*x(j, k)
    end do
  end do
end subroutine kb
"""
    sdfg = build_sdfg(src, tmp_path, name="kb").build()
    x = np.asfortranarray(np.arange(100, dtype=np.float64).reshape(10, 10))
    out = np.zeros(1)
    sdfg(lo=np.array([2, 3], dtype=np.int32), hi=np.array([6, 8], dtype=np.int32), x=x, out=out)
    assert out[0] == _expected(x, 2, 6, 3, 8)


def test_lifted_loop_bounds_from_record_array_members(tmp_path):
    src = """
module bm
  implicit none
  type bounds_info
    integer :: beg, end
  end type
contains
  subroutine kb(ib, x, out)
    type(bounds_info), dimension(1:2), intent(in) :: ib
    real(8), intent(in) :: x(10, 10)
    real(8), intent(out) :: out
    integer :: j, k
    out = 0
    do k = ib(2)%beg, ib(2)%end
      do j = ib(1)%beg, ib(1)%end
        if (x(j, k) > 50.0d0) then
          out = out + x(j, k)
          cycle
        end if
        out = out + 2.0d0*x(j, k)
      end do
    end do
  end subroutine kb
end module bm
"""
    sdfg = build_sdfg(src, tmp_path, name="kb", entry="bm::kb").build()
    x = np.asfortranarray(np.arange(100, dtype=np.float64).reshape(10, 10))
    out = np.zeros(1)
    sdfg(ib_beg=np.array([2, 3], dtype=np.int32), ib_end=np.array([6, 8], dtype=np.int32), x=x, out=out)
    assert out[0] == _expected(x, 2, 6, 3, 8)
