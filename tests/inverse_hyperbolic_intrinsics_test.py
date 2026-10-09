"""ASINH / ACOSH / ATANH: flang emits them as ``math.asinh`` / ``math.acosh`` / ``math.atanh`` ops, inside a scalar
expression and in a condition alike."""

import numpy as np

from tests._util import build_sdfg

SRC = """
subroutine hyp(n, x, y, flags)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n)
  real(8), intent(out) :: y(n, 3)
  integer, intent(out) :: flags(n)
  integer :: i
  do i = 1, n
    y(i, 1) = 2.0d0*atanh(x(i))
    y(i, 2) = asinh(x(i)) + 1.0d0
    y(i, 3) = acosh(1.0d0 + x(i)*x(i))
    flags(i) = 0
    if (atanh(x(i)) > 0.5d0) flags(i) = 1
  end do
end subroutine hyp
"""


def test_inverse_hyperbolic_functions(tmp_path):
    sdfg = build_sdfg(SRC, tmp_path / "sdfg", name="hyp").build()
    x = np.linspace(-0.9, 0.9, 7).copy()  # linspace returns a view, which DaCe does not take
    y = np.zeros((7, 3), order="F")
    flags = np.zeros(7, dtype=np.int32)
    sdfg(n=7, x=x, y=y, flags=flags)
    np.testing.assert_allclose(y[:, 0], 2.0 * np.arctanh(x), rtol=1e-14)
    np.testing.assert_allclose(y[:, 1], np.arcsinh(x) + 1.0, rtol=1e-14)
    np.testing.assert_allclose(y[:, 2], np.arccosh(1.0 + x * x), rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(flags, (np.arctanh(x) > 0.5).astype(np.int32))
