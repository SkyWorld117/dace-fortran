"""``hlfir-restore-counted-loops``: a DO whose body has a CYCLE inside an IF is lowered by flang as unstructured
control flow around a trip counter and lifted to an ``scf.while``; the pass restores the counted ``fir.do_loop``, so
the loop can become a map like any other DO."""

import numpy as np
import pytest

from dace_fortran.pipelines import num_maps, optimize
from tests._util import build_sdfg, f2py_compile, run_passes_dump

LIFT_AND_RESTORE = "lift-cf-to-scf,hlfir-restore-counted-loops"

CYCLE_SRC = """
subroutine scale(n, x, y, last)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: y(n)
  integer, intent(out) :: last
  integer :: j
  do j = 1, n
    if (x(j) < 0.0d0) then
      y(j) = 0.0d0
      cycle
    end if
    y(j) = 2.0d0*x(j)
  end do
  last = j
end subroutine scale
"""

NEST_SRC = """
subroutine cells(m, n, p, q, r)
  implicit none
  integer, intent(in) :: m, n, p
  real(8), intent(in) :: q(0:m, -1:n, p)
  real(8), intent(inout) :: r(0:m, -1:n, p)
  integer :: j, k, l
  do l = 1, p
    do k = -1, n
      do j = 0, m
        if (q(j, k, l) > 100.0d0) then
          r(j, k, l) = 100.0d0
          cycle
        end if
        r(j, k, l) = q(j, k, l) + j + k + l
      end do
    end do
  end do
end subroutine cells
"""

EXIT_SRC = """
subroutine first_negative(n, x, pos)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n)
  integer, intent(out) :: pos
  integer :: j
  pos = 0
  do j = 1, n
    if (x(j) < 0.0d0) then
      pos = j
      exit
    end if
  end do
end subroutine first_negative
"""


def test_a_do_with_a_cycle_is_a_counted_loop_again(tmp_path):
    ir = run_passes_dump(CYCLE_SRC, tmp_path, "scale", LIFT_AND_RESTORE)
    assert "scf.while" not in ir
    assert "fir.do_loop" in ir


def test_a_do_with_an_exit_stays_a_while_loop(tmp_path):
    """Its condition tests the exit flag, not just the trip counter: not a counted loop."""
    ir = run_passes_dump(EXIT_SRC, tmp_path, "first_negative", LIFT_AND_RESTORE)
    assert "scf.while" in ir


def test_the_restored_loop_runs_like_the_original(tmp_path):
    sdfg = build_sdfg(CYCLE_SRC, tmp_path / "sdfg", name="scale").build()
    ref = f2py_compile(CYCLE_SRC, tmp_path / "ref", "scale_ref")
    x = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    y, last = np.zeros(5), np.zeros(1, dtype=np.int32)
    y_ref = np.zeros(5)
    sdfg(n=5, x=x, y=y, last=last)
    last_ref = ref.scale(x, y_ref)  # f2py returns the intent(out) scalar
    np.testing.assert_array_equal(y, [2.0, 0.0, 6.0, 0.0, 10.0])
    np.testing.assert_array_equal(y, y_ref)
    assert last[0] == 6 == last_ref  # the loop variable's value after the DO


@pytest.mark.parametrize("n", [0, 3])
def test_the_loop_variable_after_an_empty_or_short_loop(tmp_path, n):
    sdfg = build_sdfg(CYCLE_SRC, tmp_path / "sdfg", name="scale").build()
    x = np.ones(max(n, 1))
    y, last = np.zeros(max(n, 1)), np.zeros(1, dtype=np.int32)
    sdfg(n=n, x=x, y=y, last=last)
    assert last[0] == n + 1


def test_a_restored_cell_loop_nest_becomes_maps(tmp_path):
    """The CYCLE sits in the innermost loop, so all three loops were lifted; restored, they parallelise."""
    sdfg = build_sdfg(NEST_SRC, tmp_path / "sdfg", name="cells").build()
    optimize(sdfg)
    assert num_maps(sdfg) > 0
    m, n, p = 4, 3, 2
    rng = np.random.default_rng(0)
    q = np.asfortranarray(rng.uniform(0.0, 200.0, (m + 1, n + 2, p)))
    r = np.zeros_like(q)
    sdfg(m=m, n=n, p=p, q=q, r=r)
    jj = np.arange(0, m + 1)[:, None, None]
    kk = np.arange(-1, n + 1)[None, :, None]
    ll = np.arange(1, p + 1)[None, None, :]
    expected = np.where(q > 100.0, 100.0, q + jj + kk + ll)
    np.testing.assert_allclose(r, expected, rtol=1e-14)
