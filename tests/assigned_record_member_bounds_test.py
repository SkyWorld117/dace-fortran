"""A scalar member of a record read as a loop bound is read where the loop starts when the kernel assigns the record:
``b = d`` ahead of ``do j = b%beg, b%end`` runs over ``d``'s bounds, not the ones ``b`` held on entry."""

import numpy as np

from tests._util import build_sdfg

_MODULE = """
module bm
  implicit none
  type bounds_info
    integer :: beg, end
    real(8) :: v(3)
  end type
  type(bounds_info) :: mb
contains
  subroutine set_bounds(d)
    type(bounds_info), intent(in) :: d
    mb = d
  end subroutine set_bounds

  subroutine sum_over(b, x, out)
    type(bounds_info), intent(in) :: b
    real(8), intent(in) :: x(10)
    real(8), intent(out) :: out
    integer :: j
    out = 0
    do j = b%beg, b%end
      out = out + x(j)
    end do
  end subroutine sum_over

  subroutine kb(d, x, out)
    type(bounds_info), intent(in) :: d
    real(8), intent(in) :: x(10)
    real(8), intent(out) :: out
    integer :: j
    {assign}
    {loop}
  end subroutine kb
end module bm
"""


_LOOP = """out = 0
    do j = mb%beg, mb%end
      out = out + x(j)
    end do"""


def _run(tmp_path, assign: str, loop: str = _LOOP) -> float:
    sdfg = build_sdfg(_MODULE.format(assign=assign, loop=loop), tmp_path, name="kb", entry="bm::kb").build()
    # The bounds come from the assignment, not from the module variable's value on entry.
    assert not {"mb_beg", "mb_end"} & {str(s) for s in sdfg.free_symbols}
    x = np.arange(1, 11, dtype=np.float64)
    out = np.zeros(1)
    args = dict(d_beg=np.array([2], np.int32), d_end=np.array([5], np.int32), d_v=np.zeros(3), x=x, out=out)
    args.update({name: np.zeros(3) for name in sdfg.arglist() if name == "mb_v"})
    sdfg(**args)
    return out[0]


def test_loop_bounds_from_a_module_record_the_kernel_assigns(tmp_path):
    assert _run(tmp_path, "mb = d") == 2 + 3 + 4 + 5


def test_loop_bounds_from_a_module_record_an_inlined_callee_assigns(tmp_path):
    # The callee declares the module variable again; its assignment still counts.
    assert _run(tmp_path, "call set_bounds(d)") == 2 + 3 + 4 + 5


def test_loop_bounds_read_through_an_inlined_callees_dummy(tmp_path):
    # The callee's dummy is declared on the module variable: its members are the record the kernel assigned.
    assert _run(tmp_path, "mb = d", loop="call sum_over(mb, x, out)") == 2 + 3 + 4 + 5
