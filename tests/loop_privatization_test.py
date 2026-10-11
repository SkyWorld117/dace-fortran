"""``privatize_loop_local_arrays``: a work array several loop nests share without passing values through it gets one
copy per nest, so every nest becomes a map; an array that does carry a value from one loop to the next keeps its
single container."""

import numpy as np
import pytest
from dace.sdfg.state import LoopRegion

from dace_fortran.loop_privatization import privatize_loop_local_arrays
from dace_fortran.pipelines import optimize
from tests._util import build_sdfg, f2py_compile

SHARED_SRC = """
subroutine nests(n, idir, a, b)
  implicit none
  integer, intent(in) :: n, idir
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  real(8) :: w(0:1)
  integer :: i
  if (idir == 1) then
    do i = 1, n
      w(0) = a(i)
      w(1) = 2.0d0*a(i)
      b(i) = w(0) + w(1)
    end do
  end if
  if (idir == 2) then
    do i = 1, n
      w(1) = a(i) + 1.0d0
      w(0) = a(i) - 1.0d0
      b(i) = w(0)*w(1)
    end do
  end if
end subroutine nests
"""

BRANCHED_SRC = """
subroutine branched(n, idir, a, b)
  implicit none
  integer, intent(in) :: n, idir
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  real(8) :: t, w(0:1)
  integer :: i
  if (idir == 1) then
    do i = 1, n
      if (a(i) > 0.0d0) then
        t = a(i)
        w(0) = 1.0d0
      else
        t = -a(i)
        w(0) = 2.0d0
      end if
      b(i) = t*w(0)
    end do
  end if
  if (idir == 2) then
    do i = 1, n
      if (a(i) > 1.0d0) then
        t = a(i) - 1.0d0
      else
        t = 1.0d0 - a(i)
      end if
      w(0) = 3.0d0
      b(i) = t + w(0)
    end do
  end if
end subroutine branched
"""

CARRIED_SRC = """
subroutine carried(n, a, b)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  real(8) :: w(0:1)
  integer :: i
  do i = 1, n
    w(0) = a(i)
    w(1) = a(i)*a(i)
  end do
  do i = 1, n
    b(i) = w(0) + w(1) + a(i)
  end do
end subroutine carried
"""


INDEXED_SRC = """
subroutine indexed(n, idir, perm, a, b)
  implicit none
  integer, intent(in) :: n, idir
  integer, intent(in) :: perm(3)
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  real(8) :: w(3)
  integer :: i
  if (idir == 1) then
    do i = 1, n
      w(1) = a(i)
      w(2) = 2.0d0*a(i)
      w(3) = 3.0d0*a(i)
      b(i) = w(perm(1)) - w(perm(2))
    end do
  end if
  if (idir == 2) then
    do i = 1, n
      w(3) = a(i) + 1.0d0
      w(2) = a(i) - 1.0d0
      w(1) = a(i)*a(i)
      b(i) = w(perm(3))*w(perm(1))
    end do
  end if
end subroutine indexed
"""


def _loops(sdfg) -> list[LoopRegion]:
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


def test_each_nest_gets_its_own_copy(tmp_path):
    sdfg = build_sdfg(SHARED_SRC, tmp_path / "sdfg", name="nests").build()
    renamed = privatize_loop_local_arrays(sdfg)
    assert list(renamed) == ["w"] and len(renamed["w"]) == 1
    sdfg.validate()


@pytest.mark.parametrize("direction", [1, 2])
def test_shared_work_array_no_longer_keeps_the_nests_sequential(tmp_path, direction):
    sdfg = build_sdfg(SHARED_SRC, tmp_path / "sdfg", name="nests").build()
    optimize(sdfg)
    assert _loops(sdfg) == [], "a nest stayed a loop"
    ref = f2py_compile(SHARED_SRC, tmp_path / "ref", f"nests_ref_{direction}")
    a = np.linspace(-2.0, 3.0, 9).copy()  # linspace returns a view, which DaCe does not take
    b, b_ref = np.zeros(9), np.zeros(9)
    sdfg(n=9, idir=direction, a=a, b=b)
    ref.nests(direction, a, b_ref)
    np.testing.assert_array_equal(b, b_ref)


def test_a_value_carried_between_loops_keeps_one_container(tmp_path):
    sdfg = build_sdfg(CARRIED_SRC, tmp_path / "sdfg", name="carried").build()
    assert privatize_loop_local_arrays(sdfg) == {}
    sdfg = build_sdfg(CARRIED_SRC, tmp_path / "sdfg2", name="carried").build()
    optimize(sdfg)
    ref = f2py_compile(CARRIED_SRC, tmp_path / "ref", "carried_ref")
    a = np.linspace(1.0, 2.0, 5).copy()
    b, b_ref = np.zeros(5), np.zeros(5)
    sdfg(n=5, a=a, b=b)
    ref.carried(a, b_ref)
    np.testing.assert_array_equal(b, b_ref)
    np.testing.assert_allclose(b, a[-1] + a[-1] ** 2 + a)


@pytest.mark.parametrize("direction", [1, 2])
def test_a_temporary_written_on_every_branch_is_local(tmp_path, direction):
    """``t`` and ``w(0)`` are assigned on both branches of an IF before being read, in each nest."""
    sdfg = build_sdfg(BRANCHED_SRC, tmp_path / "sdfg", name="branched").build()
    optimize(sdfg)
    assert _loops(sdfg) == [], "a nest stayed a loop"
    ref = f2py_compile(BRANCHED_SRC, tmp_path / "ref", f"branched_ref_{direction}")
    a = np.linspace(-2.0, 3.0, 9).copy()
    b, b_ref = np.zeros(9), np.zeros(9)
    sdfg(n=9, idir=direction, a=a, b=b)
    ref.branched(direction, a, b_ref)
    np.testing.assert_array_equal(b, b_ref)


def test_a_read_at_a_runtime_index_is_local_once_every_element_is_written(tmp_path):
    """``w(perm(k))`` names no constant element, but each nest has written all of ``w`` before it."""
    sdfg = build_sdfg(INDEXED_SRC, tmp_path / "sdfg", name="indexed").build()
    renamed = privatize_loop_local_arrays(sdfg)
    assert list(renamed) == ["w"] and len(renamed["w"]) == 1
    sdfg = build_sdfg(INDEXED_SRC, tmp_path / "sdfg2", name="indexed").build()
    optimize(sdfg)
    assert _loops(sdfg) == [], "a nest stayed a loop"
    ref = f2py_compile(INDEXED_SRC, tmp_path / "ref", "indexed_ref")
    a = np.linspace(-2.0, 3.0, 9).copy()
    perm = np.array([3, 1, 2], dtype=np.int32)
    for direction in (1, 2):
        b, b_ref = np.zeros(9), np.zeros(9)
        sdfg(n=9, idir=direction, perm=perm, a=a, b=b)
        ref.indexed(direction, perm, a, b_ref)
        np.testing.assert_array_equal(b, b_ref)
