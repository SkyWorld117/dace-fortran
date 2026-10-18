"""An array element read in one statement both as an inline indirect index and as a value (a bubble's cell:
``s = cell(1) + (pos - x_cb(cell(1) - 1))/dx(cell(1))``): the index reads become one lifted symbol, and the value
read keeps its subscript -- it was rendered as the bare array (``float64(cell)``), which does not compile."""
import numpy as np

from tests._util import build_sdfg

SRC = """
module m_cellpos
  implicit none
contains
  subroutine kern(pos, x_cb, dx, s, n)
    integer, intent(in) :: n
    real(8), intent(in) :: pos(n), x_cb(0:n), dx(n)
    real(8), intent(out) :: s(n)
    integer :: cell(3), k
    do k = 1, n
      cell(1) = int(pos(k))
      s(k) = real(cell(1), 8) + (pos(k) - x_cb(cell(1) - 1))/dx(cell(1))
    end do
  end subroutine
end module
"""


def test_element_read_as_index_and_as_value(tmp_path):
    sdfg = build_sdfg(SRC, tmp_path, name="cellpos", entry="m_cellpos::kern").build()
    n = 4
    pos = np.array([1.5, 2.25, 3.5, 1.0])
    x_cb = np.linspace(0.0, n, n + 1)
    dx = np.full(n, 1.0)
    s = np.zeros(n)
    sdfg(pos=pos, x_cb=x_cb, dx=dx, s=s, n=n)
    cell = pos.astype(int)
    np.testing.assert_allclose(s, cell + (pos - x_cb[cell - 1]) / dx[cell - 1])
