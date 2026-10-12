"""An elementwise expression over a strided section must read every ``st``-th element from the section's start.

Position ``i`` of ``w(lo:hi:st)`` is ``w(lo + (i - 1) * st)``.  The element designate of a section was rebased by the
section's start only, so ``2*w(1:5:2)`` read ``w(1), w(2), w(3)`` -- silently wrong values.  Covered: a positive and a
negative stride, a start other than 1, and strided dims of rank-2 sections that keep or drop a scalar dim."""

import numpy as np

from tests._util import build_sdfg

_SRC = """
subroutine strided_reads(w, a, o1, o2, o3, o4)
  real(8), intent(in) :: w(7), a(4, 5)
  real(8), intent(out) :: o1(3), o2(3), o3(2), o4(2, 2)
  o1 = 2 * w(2:7:2)
  o2 = w(7:1:-3) + 1
  o3 = 3 * a(2, 1:5:3)
  o4 = a(1:4:3, 2:5:2) - 1
end subroutine strided_reads
"""


def test_elementwise_reads_of_strided_sections(tmp_path):
    sdfg = build_sdfg(_SRC, tmp_path, name="strided_reads").build()
    sdfg.validate()
    w = np.arange(1.0, 8.0)
    a = np.array(np.arange(1.0, 21.0).reshape((4, 5), order="F"), order="F")
    o1, o2, o3 = np.zeros(3), np.zeros(3), np.zeros(2)
    o4 = np.asfortranarray(np.zeros((2, 2)))
    sdfg(w=w, a=a, o1=o1, o2=o2, o3=o3, o4=o4)
    np.testing.assert_array_equal(o1, 2 * w[1:7:2])
    np.testing.assert_array_equal(o2, w[6::-3] + 1)
    np.testing.assert_array_equal(o3, 3 * a[1, 0:5:3])
    np.testing.assert_array_equal(o4, a[0:4:3, 1:5:2] - 1)
