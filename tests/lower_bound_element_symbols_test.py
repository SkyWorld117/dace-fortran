"""An assumed-shape dummy whose declared lower bounds read array elements (``a(lo(1):, lo(2):)``): the bounds are
values the routine computes from its inputs at entry, not the caller's descriptor bounds, so the offsets come from
the element reads instead of staying free symbols."""

import numpy as np

from tests._util import build_sdfg

SRC = """
module bnd
  implicit none
  integer :: lo(2) = [0, -2]
end module bnd

module kern
contains
  subroutine shift(a, n)
    use bnd
    implicit none
    integer, intent(in) :: n
    real(8), intent(inout) :: a(lo(1):, lo(2):)
    integer :: i, j
    do j = lo(2), lo(2) + 1
      do i = lo(1), lo(1) + n - 1
        a(i, j) = a(i, j) + i + 10*j
      end do
    end do
  end subroutine shift
end module kern
"""


def test_lower_bounds_from_array_elements(tmp_path):
    sdfg = build_sdfg(SRC, tmp_path / "sdfg", name="shift", entry="kern::shift").build()
    free = {str(s) for s in sdfg.free_symbols}
    assert not any(s.startswith("offset_a_") for s in free), f"offsets left to the caller: {free}"
    lo = np.array([3, -2], dtype=np.int32)
    n = 4
    a = np.zeros((n, 2), order="F")
    sdfg(a=a, n=n, lo=lo)
    k = np.arange(n)[:, None]
    m = np.arange(2)[None, :]
    np.testing.assert_array_equal(a, (lo[0] + k) + 10.0 * (lo[1] + m))
