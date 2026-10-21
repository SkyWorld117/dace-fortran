"""An EXIT that tests a value the loop body has just computed.  Flang lifts such a DO to an scf.while whose break
condition is computed inside the body (for a counted DO, it is the result of the scf.if guarding the body): it must be
taken where it is computed, not before the body, where it reads the previous iteration's values -- or uninitialised
ones on the first iteration -- and leaves the loop early."""

import numpy as np
import pytest

from tests._util import build_sdfg, f2py_compile

NEWTON_SRC = """
subroutine newton(n, maxit, c, w)
  implicit none
  integer, intent(in) :: n, maxit
  real(8), intent(in) :: c(n)
  real(8), intent(inout) :: w(n)
  real(8) :: x, dx
  integer :: j, iter
  do j = 1, n
    x = c(j) + 1.0d0
    do iter = 1, maxit
      dx = -(x*x - c(j))/(2.0d0*x)
      x = x + dx
      if (abs(dx) < 1.0d-12*x) exit
    end do
    w(j) = x
  end do
end subroutine newton
"""

HALVE_SRC = """
subroutine halve(n, maxit, a, steps)
  implicit none
  integer, intent(in) :: n, maxit
  real(8), intent(inout) :: a(n)
  integer, intent(inout) :: steps(n)
  integer :: j, iter
  do j = 1, n
    steps(j) = 0
    do iter = 1, maxit
      if (a(j) > 1.0d0) then
        a(j) = a(j)/2
        steps(j) = steps(j) + 1
      end if
      if (a(j) <= 1.0d0) exit
    end do
  end do
end subroutine halve
"""


@pytest.mark.parametrize("maxit", [1, 2, 3, 50])
def test_exit_on_the_step_the_body_computed(tmp_path, maxit):
    """Newton iteration per element: the EXIT tests the step of the current iteration."""
    sdfg = build_sdfg(NEWTON_SRC, tmp_path / "sdfg", name="newton").build()
    ref = f2py_compile(NEWTON_SRC, tmp_path / "ref", f"newton_ref_{maxit}")
    c = np.linspace(1.0, 4.0, 16).copy()  # linspace returns a view, which DaCe does not take
    w = np.zeros(16)
    w_ref = np.zeros(16)
    sdfg(n=16, maxit=maxit, c=c, w=w)
    ref.newton(maxit, c, w_ref)
    np.testing.assert_allclose(w, w_ref, rtol=1e-13)  # rounding only: DaCe and gfortran contract differently


def test_exit_on_an_element_the_body_updated(tmp_path):
    """The EXIT reads an array element an IF in the body has just updated."""
    sdfg = build_sdfg(HALVE_SRC, tmp_path / "sdfg", name="halve").build()
    ref = f2py_compile(HALVE_SRC, tmp_path / "ref", "halve_ref")
    a0 = np.array([0.5, 1.0, 1.5, 3.0, 7.0, 40.0, 1000.0, 2.0])
    a, steps = a0.copy(), np.zeros(8, dtype=np.int32)
    a_ref, steps_ref = a0.copy(), np.zeros(8, dtype=np.int32)
    sdfg(n=8, maxit=6, a=a, steps=steps)
    ref.halve(6, a_ref, steps_ref)
    np.testing.assert_array_equal(steps, steps_ref)
    np.testing.assert_array_equal(a, a_ref)
