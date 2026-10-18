"""A scalar zeroed and then overwritten under a module flag the build specialises: once the branches fold, the two
assignments are straight-line code, and the second must still win (a Riemann solver's low-Mach pressure correction:
``pcorr = 0`` followed by ``if (low_mach == 1) pcorr = ...``, inside ``if (low_mach == 1) ... else pcorr = 0``)."""

import numpy as np
import pytest

from dace_fortran.builder import SDFGBuilder
from tests._util import compile_to_hlfir

_SRC = """
module cfg
  implicit none
  integer :: low_mach = 0
end module cfg

module kern
  use cfg
  implicit none
contains
  subroutine flux(x, y, n)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(inout) :: y(n)
    real(8) :: zc, p
    integer :: j
    do j = 1, n
      if (low_mach == 1) then
        zc = min(1d0, x(j))
        p = 0d0
        if (low_mach == 1) then
          p = -(x(j) - 3d0)*zc
        end if
      else
        p = 0d0
      end if
      y(j) = y(j) + 2d0*p
    end do
  end subroutine flux
end module kern
"""


@pytest.mark.parametrize("values", [None, {"cfg::low_mach": 1}], ids=["general", "specialised"])
def test_the_overwrite_of_a_zeroed_scalar_wins(tmp_path, values):
    hlfir = compile_to_hlfir(_SRC, tmp_path / "hlfir", name="kern")
    sdfg = SDFGBuilder(str(hlfir), entry="kern::flux", specialized_globals=values).build()
    sdfg.name = f"flux_{'spec' if values else 'general'}"
    sdfg.build_folder = str(tmp_path / sdfg.name)
    x = np.linspace(0.25, 2.0, 8).copy()
    y = np.zeros(8)
    args = {"x": x, "y": y, "n": 8}
    if values is None:
        args["low_mach"] = np.array([1], dtype=np.int32)
    sdfg(**{k: v for k, v in args.items() if k in sdfg.arglist() or k in {str(s) for s in sdfg.free_symbols}})
    np.testing.assert_allclose(y, 2 * (-(x - 3) * np.minimum(1, x)))
