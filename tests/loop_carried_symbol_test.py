"""A symbol a loop body advances after a statement that reads it keeps the value that statement read.

A loop body holding only assignments is emitted with every symbol assignment on the interstate edge ahead of the
compute states.  For a counter that advances after the store that uses it (``v(k) = x; k = k + 1``) that is wrong:
the store would read the advanced counter, every value would land one position later and the last one past the end
of ``v``.  Such a body is emitted in source order instead.  Two symbol assignments where one reads what the other
writes (``j = k; k = k + 1``) take two interstate edges: DaCe gives the assignments on one edge no order.

Every destination is longer than what the code writes, and its tail is checked to be untouched."""

import numpy as np

from tests._util import build_sdfg

_COUNTER_SRC = """
subroutine counter_fill(a, v)
  real(8), intent(in) :: a
  real(8), intent(inout) :: v(6)
  integer :: i, k
  k = 1
  v(k) = -1
  k = k + 1
  do i = 1, 3
    v(k) = i * a
    k = k + 1
  end do
end subroutine counter_fill
"""

_TWO_LOOPS_SRC = """
subroutine counter_two_loops(n, a, v)
  integer, intent(in) :: n
  real(8), intent(in) :: a
  real(8), intent(inout) :: v(2 * n + 2)
  integer :: i, k
  k = 1
  do i = 1, n
    v(k) = i * a
    k = k + 1
  end do
  do i = 1, n
    v(k) = -i * a
    k = k + 1
  end do
end subroutine counter_two_loops
"""

_READ_AROUND_SRC = """
subroutine counter_read_around(a, v, w)
  real(8), intent(in) :: a
  real(8), intent(inout) :: v(5), w(5)
  integer :: i, k
  k = 1
  do i = 1, 3
    v(k) = i * a
    k = k + 1
    w(k) = i * a + k
  end do
end subroutine counter_read_around
"""

_COPY_THEN_ADVANCE_SRC = """
subroutine counter_copy_then_advance(a, v, w)
  real(8), intent(in) :: a
  real(8), intent(inout) :: v(5), w(6)
  integer :: i, j, k
  k = 2
  do i = 1, 3
    j = k
    k = k + 1
    v(j) = i * a
    w(k) = -i * a
  end do
end subroutine counter_copy_then_advance
"""


def _sentinels(n):
    return 100.0 + np.arange(n, dtype=np.float64)


def test_counter_advanced_after_each_store(tmp_path):
    sdfg = build_sdfg(_COUNTER_SRC, tmp_path, name="counter_fill").build()
    sdfg.validate()
    v = _sentinels(6)
    sdfg(a=0.5, v=v)
    np.testing.assert_array_equal(v, [-1.0, 0.5, 1.0, 1.5, 104.0, 105.0])


def test_counter_carried_through_two_loops(tmp_path):
    sdfg = build_sdfg(_TWO_LOOPS_SRC, tmp_path, name="counter_two_loops").build()
    sdfg.validate()
    n, a = 3, 0.25
    v = _sentinels(2 * n + 2)
    sdfg(n=n, a=a, v=v)
    i = np.arange(1, n + 1)
    np.testing.assert_array_equal(v, np.concatenate([i * a, -i * a, [106.0, 107.0]]))


def test_counter_read_before_and_after_it_advances(tmp_path):
    sdfg = build_sdfg(_READ_AROUND_SRC, tmp_path, name="counter_read_around").build()
    sdfg.validate()
    a = 0.5
    v, w = _sentinels(5), _sentinels(5)
    sdfg(a=a, v=v, w=w)
    np.testing.assert_array_equal(v, [a, 2 * a, 3 * a, 103.0, 104.0])
    np.testing.assert_array_equal(w, [100.0, a + 2, 2 * a + 3, 3 * a + 4, 104.0])


def test_symbol_copied_before_it_advances(tmp_path):
    """``j = k; k = k + 1`` with both used as subscripts: two symbol assignments that need two edges."""
    sdfg = build_sdfg(_COPY_THEN_ADVANCE_SRC, tmp_path, name="counter_copy_then_advance").build()
    sdfg.validate()
    a = 0.5
    v, w = _sentinels(5), _sentinels(6)
    sdfg(a=a, v=v, w=w)
    np.testing.assert_array_equal(v, [100.0, a, 2 * a, 3 * a, 104.0])
    np.testing.assert_array_equal(w, [100.0, 101.0, -a, -2 * a, -3 * a, 105.0])
