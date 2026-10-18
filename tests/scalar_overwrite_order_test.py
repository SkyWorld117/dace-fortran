"""A scalar zeroed and then overwritten under a module flag the build specialises: once the branches fold, the two
assignments are straight-line code, and the second must still win (a Riemann solver's low-Mach pressure correction:
``pcorr = 0`` followed by ``if (low_mach == 1) pcorr = ...``, inside ``if (low_mach == 1) ... else pcorr = 0``)."""

import dace
import networkx as nx
import numpy as np
import pytest

from dace_fortran.builder import SDFGBuilder
from dace_fortran.builder.statement_order import order_state
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
        args["low_mach"] = 1
    sdfg(**{k: v for k, v in args.items() if k in sdfg.arglist() or k in {str(s) for s in sdfg.free_symbols}})
    np.testing.assert_allclose(y, 2 * (-(x - 3) * np.minimum(1, x)))


def test_two_stores_into_one_node_run_in_statement_order():
    """``p = 0`` then ``p = 5`` into one access node, then ``y = p``: the second store follows the first."""
    sdfg = dace.SDFG("two_stores")
    sdfg.add_scalar("p", dace.float64, transient=True)
    sdfg.add_array("y", [1], dace.float64)
    state = sdfg.add_state()
    p = state.add_access("p")
    zero = state.add_tasklet("zero", {}, {"o"}, "o = 0.0")
    state.add_edge(zero, "o", p, None, dace.Memlet("p[0]"))
    five = state.add_tasklet("five", {}, {"o"}, "o = 5.0")
    state.add_edge(five, "o", p, None, dace.Memlet("p[0]"))
    use = state.add_tasklet("use", {"i"}, {"o"}, "o = i")
    state.add_edge(p, None, use, "i", dace.Memlet("p[0]"))
    state.add_edge(use, "o", state.add_write("y"), None, dace.Memlet("y[0]"))
    assert order_state(state) == []
    assert nx.has_path(state.nx, zero, five) and not nx.has_path(state.nx, five, zero)
    y = np.zeros(1)
    sdfg(y=y)
    assert y[0] == 5.0


def test_stores_into_different_elements_stay_unordered():
    """``b(1) = 1`` and ``b(2) = 2`` into one node need no order: no edge is added."""
    sdfg = dace.SDFG("two_elements")
    sdfg.add_array("b", [2], dace.float64)
    state = sdfg.add_state()
    b = state.add_access("b")
    one = state.add_tasklet("one", {}, {"o"}, "o = 1.0")
    state.add_edge(one, "o", b, None, dace.Memlet("b[0]"))
    two = state.add_tasklet("two", {}, {"o"}, "o = 2.0")
    state.add_edge(two, "o", b, None, dace.Memlet("b[1]"))
    assert order_state(state) == []
    assert not nx.has_path(state.nx, one, two) and not nx.has_path(state.nx, two, one)
