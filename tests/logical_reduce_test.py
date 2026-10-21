"""Logical reductions -- ``ANY``/``ALL``/``COUNT`` on whole arrays.  In ``kRedTable``,
works for the whole-array shape Flang emits without section slicing.  Dynamic sections
(``ANY(mask(lo:hi, jk))``) hit a separate gap in ``emit_reduce``, xfailed via
``test_loopnest_6_sdfg_matches_f2py``.
"""

from pathlib import Path

import numpy as np

from tests._util import build_sdfg


def test_any_whole_array(tmp_path: Path):
    """``result = ANY(mask)`` on a 1D logical array."""
    src = """
subroutine kernel_any(mask, result, n)
  implicit none
  integer, intent(in)  :: n
  logical, intent(in)  :: mask(n)
  logical, intent(out) :: result
  result = ANY(mask)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="kernel_any", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(42)
    mask = np.asfortranarray(rng.random(8) > 0.5)
    result = np.zeros(1, dtype=np.uint32)
    sdfg(mask=mask, result=result, n=8)
    expected = bool(mask.any())
    assert bool(result[0]) == expected, f"ANY({mask.tolist()}) -> {bool(result[0])}, want {expected}"


def test_all_whole_array(tmp_path: Path):
    """``result = ALL(mask)`` on a 1D logical array."""
    src = """
subroutine kernel_all(mask, result, n)
  implicit none
  integer, intent(in)  :: n
  logical, intent(in)  :: mask(n)
  logical, intent(out) :: result
  result = ALL(mask)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="kernel_all", pipeline="hlfir-propagate-shapes").build()

    # All-true case
    mask_all = np.asfortranarray(np.ones(5, dtype=np.uint32))
    res_all = np.zeros(1, dtype=np.uint32)
    sdfg(mask=mask_all, result=res_all, n=5)
    assert bool(res_all[0]) is True

    # One-false case
    mask_mixed = np.asfortranarray(np.array([True, True, False, True, True], dtype=np.uint32))
    res_mixed = np.zeros(1, dtype=np.uint32)
    sdfg(mask=mask_mixed, result=res_mixed, n=5)
    assert bool(res_mixed[0]) is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))


def test_any_of_inline_comparison_as_actual_argument(tmp_path: Path):
    """``pick(any(i == idx), ...)``: the elemental mask is reduced and the result stored into the inlined callee's
    association scalar for its ``logical`` dummy."""
    src = """
module m
  implicit none
contains
  pure real(8) function pick(flag, a, b)
    logical, intent(in) :: flag
    real(8), intent(in) :: a, b
    if (flag) then
      pick = a
    else
      pick = b
    end if
  end function pick

  subroutine kernel(idx, n, out)
    integer, intent(in) :: idx(3), n
    real(8), intent(out) :: out(n)
    integer :: i
    do i = 1, n
      out(i) = pick(any(i == idx), 1.0d0, 2.0d0)
    end do
  end subroutine kernel
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name="kernel", entry="m::kernel").build()
    idx = np.array([2, 5, 7], dtype=np.int32)
    out = np.zeros(8, dtype=np.float64)
    sdfg(idx=idx, n=np.int32(8), out=out)
    np.testing.assert_array_equal(out, [2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0])
