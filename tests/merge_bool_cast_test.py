"""A ``merge`` of two values that specialisation makes the constants 1 and 0 (or -1 and 0) is a select the MLIR
canonicaliser turns into a boolean widened to an integer (``arith.extui`` / ``arith.extsi`` of the condition). Used as
an integer -- in a comparison of an interstate condition, as a loop bound -- it must stay one: the boolean alone
(``(i == 1) > 0``) is rejected by the symbolic parser, and a sign-extended true is -1."""

import numpy as np

from dace_fortran.builder import SDFGBuilder
from tests._util import compile_to_hlfir

_SRC = """
module sizes_cfg
  implicit none
  integer :: sizes(2) = [1, 0]
end module sizes_cfg

module sizes_work
  use sizes_cfg
  implicit none
contains
  subroutine pick_sum(a, y)
    real(8), intent(in) :: a(2, 3)
    real(8), intent(out) :: y(2)
    integer :: i, q, n1, n2
    n1 = sizes(1)
    n2 = sizes(2)
    do i = 1, 2
      y(i) = -1
      if (merge(n1, n2, i == 1) > 0) y(i) = 0
      do q = 1, merge(n1, n2, i == 1)
        y(i) = y(i) + a(i, q)
      end do
      y(i) = y(i) + 10 * merge(-1, 0, i == 2)
    end do
  end subroutine pick_sum
end module sizes_work
"""


def test_a_merge_of_specialised_one_and_zero_stays_an_integer(tmp_path):
    hlfir = compile_to_hlfir(_SRC, tmp_path / "hlfir", name="sizes_work")
    sdfg = SDFGBuilder(str(hlfir), entry="sizes_work::pick_sum",
                       specialized_globals={"sizes_cfg::sizes(1)": 1, "sizes_cfg::sizes(2)": 0}).build()
    sdfg.name = "pick_sum_spec"
    sdfg.build_folder = str(tmp_path / sdfg.name)
    a = np.array(np.arange(1, 7, dtype=np.float64).reshape(2, 3, order="F"), order="F")  # owns its data
    y = np.zeros(2)
    sdfg(a=a, y=y)
    # i = 1: one term a(1, 1); i = 2: no term, y stays -1, and merge(-1, 0, .true.) adds -10.
    np.testing.assert_array_equal(y, [a[0, 0], -1 - 10])


def test_a_merge_of_a_constant_and_zero_as_a_loop_bound_stays_a_merge(tmp_path):
    """Specialised to 2 and 0 the merge stays a select, ``select(i == 1, 2, 0)``: not the extent clamp
    ``select(x > 0, x, 0)`` it resembles, so the bound is not the constant 2 -- the loop of i = 2 has no trips (as MFC's
    interface Reynolds numbers with shear but no bulk viscosity, whose second row of indices is never set)."""
    hlfir = compile_to_hlfir(_SRC, tmp_path / "hlfir", name="sizes_work")
    sdfg = SDFGBuilder(str(hlfir), entry="sizes_work::pick_sum",
                       specialized_globals={"sizes_cfg::sizes(1)": 2, "sizes_cfg::sizes(2)": 0}).build()
    sdfg.name = "pick_sum_spec_two"
    sdfg.build_folder = str(tmp_path / sdfg.name)
    a = np.array(np.arange(1, 7, dtype=np.float64).reshape(2, 3, order="F"), order="F")
    y = np.zeros(2)
    sdfg(a=a, y=y)
    # i = 1: two terms a(1, 1) + a(1, 2); i = 2: no term, y stays -1, and merge(-1, 0, .true.) adds -10.
    np.testing.assert_array_equal(y, [a[0, 0] + a[0, 1], -1 - 10])
