"""A scalar assigned to a whole array without extents in its declaration fills every element.

``v = s`` with ``v`` an assumed-shape dummy (or an allocatable) declares no shape, so the broadcast found no extents,
fell back to a plain scalar assignment and wrote the first element only (an inlined copy of the dummy) or failed to
build (the entry's own dummy). The extents now come from the variable an inlined copy is bound to, else from the
array's extent symbols. A constant takes another path and always worked; the cases here assign a value read from
data, one of them an element of another array.
"""

from pathlib import Path

import numpy as np

from tests._util import build_sdfg

INLINED = """
module m_broadcast_inlined
  implicit none
contains
  subroutine fill(v, s)
    double precision, intent(inout) :: v(:)
    double precision, intent(in) :: s
    v = s * 2d0 + 1d0
  end subroutine fill

  subroutine driver(x, y, s)
    double precision, intent(inout) :: x(3), y(4)
    double precision, intent(in) :: s
    call fill(x, s)
    call fill(y, s + 1d0)
  end subroutine driver
end module m_broadcast_inlined
"""

ENTRY = """
module m_broadcast_entry
  implicit none
contains
  subroutine driver(v, t)
    double precision, intent(inout) :: v(:)
    double precision, intent(in) :: t(3)
    v = t(2) * 2d0
  end subroutine driver
end module m_broadcast_entry
"""

ALLOCATABLE = """
module m_broadcast_allocatable
  implicit none
contains
  subroutine driver(out, s, n)
    integer, intent(in) :: n
    double precision, intent(out) :: out(n)
    double precision, intent(in) :: s
    double precision, allocatable :: a(:)
    integer :: i
    allocate(a(n))
    a = s * 3d0
    do i = 1, n
      out(i) = a(i) + i
    end do
    deallocate(a)
  end subroutine driver
end module m_broadcast_allocatable
"""


def test_broadcast_through_inlined_assumed_shape_dummy(tmp_path: Path):
    sdfg = build_sdfg(INLINED, tmp_path, name="broadcast_inlined", entry="m_broadcast_inlined::driver").build()
    x, y = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0, 7.0])
    sdfg(x=x, y=y, s=0.5)
    np.testing.assert_array_equal(x, np.full(3, 0.5 * 2 + 1.0))
    np.testing.assert_array_equal(y, np.full(4, 1.5 * 2 + 1.0))


def test_broadcast_to_entry_assumed_shape_dummy(tmp_path: Path):
    sdfg = build_sdfg(ENTRY, tmp_path, name="broadcast_entry", entry="m_broadcast_entry::driver").build()
    v = np.zeros(5)
    sdfg(v=v, t=np.array([0.25, 0.75, 1.25]), v_d0=np.int64(5))
    np.testing.assert_array_equal(v, np.full(5, 1.5))


def test_broadcast_to_allocatable(tmp_path: Path):
    sdfg = build_sdfg(ALLOCATABLE, tmp_path, name="broadcast_allocatable", entry="m_broadcast_allocatable::driver")
    sdfg = sdfg.build()
    out = np.zeros(4)
    sdfg(out=out, s=0.5, n=4)
    np.testing.assert_array_equal(out, 1.5 + np.arange(1, 5))
