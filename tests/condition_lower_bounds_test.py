"""Control-flow array reads use the same lower bounds as dataflow memlets."""

import numpy as np
import pytest

from tests._util import build_sdfg


_SRC = """
subroutine guarded(lo, hi, lj, uj, its, ite, jts, jte, mask, out)
  implicit none
  integer, value :: lo, hi, lj, uj, its, ite, jts, jte
  integer, intent(in) :: mask(lo:hi,lj:uj)
  integer, intent(inout) :: out(lo:hi,lj:uj)
  integer :: i, j
  do j = jts, jte
    do i = its, ite
      if (mask(i,j) > 0 .and. mask(i,j) < 3) then
        out(i,j) = 7
      end if
    end do
  end do
end subroutine guarded
"""


@pytest.mark.parametrize("lo,lj", [(-3, -2), (2, 3), (1, 1)])
def test_repeated_array_condition_respects_lower_bounds(tmp_path, lo, lj):
    # Repeated identical reads take the interstate-expression fallback after
    # access deduplication. Keep even the old incorrect indices in allocated
    # storage so a failure is deterministic rather than an out-of-bounds read.
    hi, uj = lo + 10, lj + 10
    its, jts = max(lo, 1) + 1, max(lj, 1) + 1
    ite, jte = its + 2, jts + 2
    mask = np.asfortranarray(np.arange(121, dtype=np.int32).reshape(11, 11) % 5)
    out = np.full_like(mask, -9, order="F")
    expected = out.copy(order="F")
    for j in range(jts, jte + 1):
        for i in range(its, ite + 1):
            if 0 < mask[i - lo, j - lj] < 3:
                expected[i - lo, j - lj] = 7
    sdfg = build_sdfg(_SRC, tmp_path, name="guarded").build()
    sdfg(mask=mask, out=out, lo=lo, hi=hi, lj=lj, uj=uj,
         its=its, ite=ite, jts=jts, jte=jte)
    np.testing.assert_array_equal(out, expected)


def test_repeated_condition_on_inlined_rank_reduced_section(tmp_path):
    src = """
module section_conditions
contains
subroutine section_guard(a, out)
  integer, intent(in) :: a(-2:8,3:6)
  integer, intent(out) :: out(11)
  call inner(a(:,4), out)
end subroutine
subroutine inner(v, out)
  integer, intent(in) :: v(:)
  integer, intent(out) :: out(11)
  integer :: i
  out = -9
  do i = 1, 11
    if (v(i) > 0 .and. v(i) < 3) out(i) = 7
  end do
end subroutine
end module
"""
    a = np.asfortranarray(np.arange(44, dtype=np.int32).reshape(11, 4) % 5)
    out = np.zeros(11, dtype=np.int32)
    sdfg = build_sdfg(src, tmp_path, name="section_guard",
                      entry="section_conditions::section_guard").build()
    sdfg(a=a, out=out)
    np.testing.assert_array_equal(out, np.where((a[:, 1] > 0) & (a[:, 1] < 3), 7, -9))


def test_do_while_repeated_condition_with_nonunit_bound(tmp_path):
    src = """
subroutine scan(mask, out)
  integer, intent(in) :: mask(-3:7)
  integer, intent(out) :: out
  integer :: i
  i = 1
  do while (mask(i) > 0 .and. mask(i) < 3)
    i = i + 1
  end do
  out = i
end subroutine
"""
    mask = np.array([0, 0, 0, 0, 1, 2, 1, 3, 0, 0, 0], dtype=np.int32)
    out = np.zeros(1, dtype=np.int32)
    sdfg = build_sdfg(src, tmp_path, name="scan").build()
    sdfg(mask=mask, out=out)
    assert out[0] == 4


def test_select_case_with_nonunit_bound(tmp_path):
    src = """
subroutine select_values(a, out)
  integer, intent(in) :: a(-3:7)
  integer, intent(out) :: out(-3:7)
  integer :: i
  do i = -3, 7
    select case (a(i))
    case (1)
      out(i) = 11
    case (2:3)
      out(i) = 23
    case default
      out(i) = -1
    end select
  end do
end subroutine
"""
    a = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 4, 2], dtype=np.int32)
    out = np.zeros_like(a)
    sdfg = build_sdfg(src, tmp_path, name="select_values").build()
    sdfg(a=a, out=out)
    np.testing.assert_array_equal(out, np.where(a == 1, 11, np.where((a >= 2) & (a <= 3), 23, -1)))


def test_nonunit_condition_read_matches_writes_for_parallel_mapping(tmp_path):
    from dace_fortran.loop_mapping import loop_to_map_type

    # The dependence check must see the same cell in writes and control-flow
    # reads; subtracting one only in the condition falsely creates a hazard.
    src = _SRC.replace("intent(in) :: mask", "intent(inout) :: mask")
    src = src.replace("      if (mask", "      mask(i,j) = mod(i+j,5)\n      if (mask")
    lo, hi, lj, uj = -3, 7, -2, 8
    mask = np.zeros((11, 11), dtype=np.int32, order="F")
    out = np.full_like(mask, -9)
    expected = out.copy(order="F")
    expected_mask = mask.copy(order="F")
    for j in range(2, 5):
        for i in range(2, 5):
            value = (i + j) % 5
            expected_mask[i - lo, j - lj] = value
            if 0 < value < 3:
                expected[i - lo, j - lj] = 7
    sdfg = build_sdfg(src, tmp_path, name="guarded").build()
    mapped = sdfg.apply_transformations_repeated(loop_to_map_type(["loop_[ij]_.*"]), validate=False)
    assert mapped > 0
    sdfg.validate()
    sdfg(mask=mask, out=out, lo=lo, hi=hi, lj=lj, uj=uj,
         its=2, ite=4, jts=2, jte=4)
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(out, expected)


def test_named_search_exit_preserves_array_read_bounds(tmp_path):
    src = """
subroutine first_match(lo, hi, lj, uj, a, threshold, out)
  integer, value :: lo, hi, lj, uj
  real(8), intent(in) :: a(lo:hi,1:8,lj:uj), threshold(lo:hi,lj:uj)
  integer, intent(inout) :: out(lo:hi,lj:uj)
  integer :: i, j, k
  do j = 2, 4
    do i = 2, 4
      search: do k = 1, 5
        if (a(i,k,j) >= threshold(i,j)) then
          out(i,j) = k
          exit search
        endif
      enddo search
    enddo
  enddo
end subroutine
"""
    lo, hi, lj, uj = -3, 7, -2, 8
    ii, jj = np.indices((11, 11))
    threshold = np.asfortranarray(((ii + 2 * jj) % 4).astype(np.float64) + 1.5)
    a = np.broadcast_to(np.arange(1., 9.)[None, :, None], (11, 8, 11)).copy(order="F")
    out = np.full((11, 11), -9, dtype=np.int32, order="F")
    expected = out.copy(order="F")
    for j in range(2, 5):
        for i in range(2, 5):
            expected[i - lo, j - lj] = int(np.ceil(threshold[i - lo, j - lj]))
    sdfg = build_sdfg(src, tmp_path, name="first_match").build()
    sdfg(a=a, threshold=threshold, out=out, lo=lo, hi=hi, lj=lj, uj=uj)
    np.testing.assert_array_equal(out, expected)
