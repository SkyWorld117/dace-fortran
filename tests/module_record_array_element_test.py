"""An element of a static module array of records handed to an inlined callee and assigned whole: every member of
the element -- scalars and arrays -- is read from the array, not only the scalar ones."""

import numpy as np

from tests._util import build_sdfg

_SRC = """
module bm
  implicit none
  type bounds_info
    integer :: beg, end
    real(8) :: vel_in(3)
  end type
  type(bounds_info) :: idw(3)
  type(bounds_info) :: cur
contains
  subroutine use_bounds(d, x, out)
    type(bounds_info), intent(in) :: d
    real(8), intent(in) :: x(10)
    real(8), intent(out) :: out
    integer :: j
    cur = d
    out = 0
    do j = cur%beg, cur%end
      out = out + x(j)
    end do
  end subroutine use_bounds

  subroutine kb(x, out)
    real(8), intent(in) :: x(10)
    real(8), intent(out) :: out
    call use_bounds(idw(2), x, out)
  end subroutine kb
end module bm
"""


def test_a_whole_element_assignment_reads_its_array_members_too(tmp_path):
    sdfg = build_sdfg(_SRC, tmp_path, name="kb", entry="bm::kb").build()
    # The array member of the elements, laid out element-major like the scalar ones.
    assert sdfg.arrays["idw_vel_in"].shape[0] == 3
    x = np.arange(1, 11, dtype=np.float64)
    out = np.zeros(1)
    vel = np.asfortranarray(np.arange(9, dtype=np.float64).reshape(3, 3))
    cur_vel = np.zeros(3)
    args = dict(
        x=x,
        out=out,
        idw_beg=np.array([1, 2, 3], np.int32),
        idw_end=np.array([4, 5, 6], np.int32),
        idw_vel_in=vel,
        cur_vel_in=cur_vel,
    )
    sdfg(**{k: v for k, v in args.items() if k in sdfg.arglist()})
    assert out[0] == x[1:5].sum()  # idw(2): beg 2, end 5
    np.testing.assert_array_equal(cur_vel, vel[1, :])
