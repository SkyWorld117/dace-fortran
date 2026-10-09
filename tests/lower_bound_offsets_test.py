"""Per-axis offset symbols make non-1 Fortran lower bounds work uniformly.

The bridge declares ``offset_<arr>_d<i>`` symbols for every array (one
per dim) and emits memlets in the form ``A[(idx) - offset_A_d<i>]``.
``SDFGBuilder.build()`` calls ``sdfg.specialize`` at the end with the
known compile-time values (default ``1``; ``dimension(20:24)`` gets
``20``; etc.).  Sympy folds the substitution everywhere automatically.

Tests pin the Fortran shapes the design unblocks:

  * ``dimension(20:24)``   --  constant non-1 lower bound.
  * ``dimension(0:4)``     --  constant zero lower bound.
  * Whole-array section assignment with a non-1 lb on both sides  --
    section-parent + array offset interact through the symbol layer.
  * Whole-array assignments to an array with non-1 lower bounds
    (elemental expression, scalar broadcast, section): the writes run
    over 1-based positions and must index the array from its own
    lower bounds, constant, assumed-shape or runtime.
  * An inlined callee whose dummy declares lower bounds of its own: its
    subscripts are rebased from the dummy's bounds to the caller's.
"""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg


def test_constant_nonone_lower_bound(tmp_path: Path):
    """``real :: a(20:24)``  --  lbound 20.  Writing ``a(22) = 7`` must
    land on the third element (offset 2 from the start of storage)."""
    src = """
subroutine main(a)
  real(8), intent(inout) :: a(20:24)
  a(22) = 7.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    a = np.zeros(5, dtype=np.float64, order="F")
    sdfg(a=a)
    np.testing.assert_array_equal(a, [0.0, 0.0, 7.0, 0.0, 0.0])


def test_constant_zero_lower_bound(tmp_path: Path):
    """``real :: a(0:4)``  --  lbound 0.  Writing ``a(0) = 1`` and
    ``a(4) = 5`` lands on the first and last element of storage."""
    src = """
subroutine main(a)
  real(8), intent(inout) :: a(0:4)
  a(0) = 1.0d0
  a(4) = 5.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    a = np.zeros(5, dtype=np.float64, order="F")
    sdfg(a=a)
    np.testing.assert_array_equal(a, [1.0, 0.0, 0.0, 0.0, 5.0])


def test_section_assignment_with_default_lb(tmp_path: Path):
    """Section assign over default-lb arrays  --  ``res(2:4) = src(2:4)``
    exercises the section-parent's ``(lo - 1)`` contribution alongside
    the array's offset symbol."""
    src = """
subroutine main(src, res)
  real(8), intent(in)    :: src(5)
  real(8), intent(inout) :: res(5)
  res(2:4) = src(2:4) * 2.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    src = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64, order="F")
    res = np.full(5, -1.0, dtype=np.float64, order="F")
    sdfg(src=src, res=res)
    # Only res[1..3] should change; res[0] and res[4] stay at -1.0.
    np.testing.assert_array_equal(res, [-1.0, 40.0, 60.0, 80.0, -1.0])


def test_whole_array_expression_with_nonone_lower_bounds(tmp_path: Path):
    """``a = a * 3`` and ``t = a + t`` on ``a(0:2, -2:1)``: each element is written where it was read."""
    src = """
subroutine main(a, t)
  real(8), intent(inout) :: a(0:2, -2:1), t(0:2, -2:1)
  a = a*3.0d0
  t = a + t
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    rng = np.random.default_rng(0)
    a0 = np.asfortranarray(rng.standard_normal((3, 4)))
    t0 = np.asfortranarray(rng.standard_normal((3, 4)))
    a, t = a0.copy(order="F"), t0.copy(order="F")
    sdfg(a=a, t=t)
    np.testing.assert_allclose(a, 3.0 * a0, rtol=1e-14)
    np.testing.assert_allclose(t, 3.0 * a0 + t0, rtol=1e-14)


def test_whole_array_broadcast_with_nonone_lower_bounds(tmp_path: Path):
    """``a = 7`` on ``a(20:24)`` and ``b = 5`` on ``b(0:2, -2:1)`` fill exactly their own storage."""
    src = """
subroutine main(a, b)
  real(8), intent(inout) :: a(20:24), b(0:2, -2:1)
  a = 7.0d0
  b = 5.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    a = np.zeros(5, dtype=np.float64, order="F")
    b = np.zeros((3, 4), dtype=np.float64, order="F")
    sdfg(a=a, b=b)
    np.testing.assert_array_equal(a, np.full(5, 7.0))
    np.testing.assert_array_equal(b, np.full((3, 4), 5.0))


def test_section_assignment_with_nonone_lower_bound(tmp_path: Path):
    """``res(1:3) = src(1:3) * 2`` on ``res(0:4)`` / ``src(0:4)``: the section starts one past the lower bound."""
    src = """
subroutine main(src, res)
  real(8), intent(in)    :: src(0:4)
  real(8), intent(inout) :: res(0:4)
  res(1:3) = src(1:3) * 2.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    src = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64, order="F")
    res = np.full(5, -1.0, dtype=np.float64, order="F")
    sdfg(src=src, res=res)
    np.testing.assert_array_equal(res, [-1.0, 40.0, 60.0, 80.0, -1.0])


def test_whole_array_assignment_to_assumed_shape_with_lower_bounds(tmp_path: Path):
    """``a = a * 2`` on an assumed-shape ``a(0:, -2:)``: the lower bounds come from the dummy's declaration."""
    src = """
subroutine main(a)
  real(8), intent(inout) :: a(0:, -2:)
  a = a*2.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    a0 = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    a = a0.copy(order="F")
    sdfg(a=a)
    np.testing.assert_array_equal(a, 2.0 * a0)


def test_whole_array_assignment_with_runtime_lower_bound(tmp_path: Path):
    """``a = a + 1`` on ``a(m:m + 4)`` with a runtime ``m``."""
    src = """
subroutine main(m, a)
  integer, intent(in) :: m
  real(8), intent(inout) :: a(m:m + 4)
  a = a + 1.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    a0 = np.arange(5, dtype=np.float64)
    a = a0.copy()
    sdfg(m=np.int32(-3), a=a)
    np.testing.assert_array_equal(a, a0 + 1.0)


_CALLEE_SRC = """
module m
  implicit none
contains
  subroutine main(n, a, t)
    integer, intent(in) :: n
    real(8), intent(in) :: a(0:n, -2:1)
    real(8), intent(out) :: t(0:n, -2:1)
    call double_into(n, a, t)
  end subroutine main
  subroutine double_into(n, x, y)
    integer, intent(in) :: n
    real(8), intent(in) :: x({lb}n{ub}, -2:1)
    real(8), intent(out) :: y({lb}n{ub}, -2:1)
    integer :: i, j
    do j = -2, 1
      do i = {lo}, {hi}
        y(i, j) = 2.0d0*x(i, j)
      end do
    end do
  end subroutine double_into
end module m
"""


@pytest.mark.parametrize(
    ("bounds", "fmt"),
    [
        ("same as the caller's", {"lb": "0:", "ub": "", "lo": "0", "hi": "n"}),
        ("starting at 1", {"lb": "", "ub": " + 1", "lo": "1", "hi": "n + 1"}),
        ("assumed shape", {"lb": "0:", "ub": "", "lo": "0", "hi": "n"}),
    ],
)
def test_inlined_callee_dummy_with_own_lower_bounds(tmp_path: Path, bounds: str, fmt: dict):
    """An inlined callee indexes ``x`` / ``y`` from its dummies' own lower bounds, not from 1."""
    src = _CALLEE_SRC.format(**fmt)
    if bounds == "assumed shape":
        src = src.replace("x(0:n, -2:1)", "x(0:, -2:)").replace("y(0:n, -2:1)", "y(0:, -2:)")
    sdfg = build_sdfg(src, tmp_path, name="main", entry="m::main").build()
    n = 3
    a = np.asfortranarray(np.arange((n + 1) * 4, dtype=np.float64).reshape(n + 1, 4))
    t = np.zeros_like(a, order="F")
    sdfg(n=np.int32(n), a=a, t=t)
    np.testing.assert_array_equal(t, 2.0 * a)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
