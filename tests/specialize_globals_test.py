"""Module variables an entry only reads, specialised to given values before inlining
(``SDFGBuilder(specialized_globals=...)``, the ``hlfir-specialize-globals`` pass): the branches they decide fold, and
a procedure only a dead branch calls is never inlined -- nor is anything it reads."""

import numpy as np

from dace_fortran.builder import SDFGBuilder
from tests._util import compile_to_hlfir

_SRC = """
module cfg
  implicit none
  logical :: use_extra = .false.
  integer :: nsweeps = 2
  real(8) :: scale = 1.0d0
  real(8) :: extra_field(4) = 0
end module cfg

module kern
  use cfg
  implicit none
contains
  subroutine extra(y)
    real(8), intent(inout) :: y(:)
    y = y + extra_field
  end subroutine extra

  subroutine step(x, y)
    real(8), intent(in) :: x(:)
    real(8), intent(inout) :: y(:)
    integer :: s
    do s = 1, nsweeps
      y = y + scale*x
    end do
    if (use_extra) then
      if (nsweeps > 1) call extra(y)
    end if
  end subroutine step

  subroutine rescale(x, y)
    real(8), intent(in) :: x(:)
    real(8), intent(inout) :: y(:)
    scale = 2*scale
    y = y + scale*x
  end subroutine rescale
end module kern
"""


def _build(tmp_path, entry: str, values=None):
    hlfir = compile_to_hlfir(_SRC, tmp_path / "hlfir", name="kern")
    sdfg = SDFGBuilder(str(hlfir), entry=entry, specialized_globals=values).build()
    sdfg.name = f"{entry.split('::')[-1]}_{'spec' if values else 'general'}"
    sdfg.build_folder = str(tmp_path / sdfg.name)
    return sdfg


def test_a_branch_on_a_specialised_flag_takes_its_callee_away(tmp_path):
    general = _build(tmp_path, "kern::step")
    assert "extra_field" in general.arrays  # the callee's module array, reached through the flag's branch

    values = {"cfg::use_extra": False, "cfg::nsweeps": 2, "cfg::scale": 1.5}
    sdfg = _build(tmp_path, "kern::step", values)
    taken = set(sdfg.arglist()) | {str(s) for s in sdfg.free_symbols}
    assert "extra_field" not in taken and not {"use_extra", "nsweeps", "scale"} & taken
    x, y = np.arange(4, dtype=np.float64), np.ones(4)
    sdfg(x=x, y=y)
    np.testing.assert_array_equal(y, 1 + 2 * 1.5 * x)


def test_the_candidates_are_the_module_scalars_read_and_never_written(tmp_path):
    hlfir = compile_to_hlfir(_SRC, tmp_path / "hlfir", name="kern")
    assert SDFGBuilder(str(hlfir), entry="kern::step").configuration_candidates == {
        "cfg::use_extra": "logical(4)",
        "cfg::nsweeps": "integer(4)",
        "cfg::scale": "real(8)",
    }
    # Recorded before the values apply; a scalar the entry writes is none.
    rescale = SDFGBuilder(str(hlfir), entry="kern::rescale", specialized_globals={"cfg::scale": 1.5})
    assert "cfg::scale" not in rescale.configuration_candidates


def test_a_global_the_entry_writes_keeps_its_value_open(tmp_path):
    sdfg = _build(tmp_path, "kern::rescale", {"cfg::scale": 1.5})
    assert "scale" in set(sdfg.arglist())


_SELECT_SRC = """
module tables
  implicit none
  logical :: shifted = .false.
  real(8) :: ya(4) = [1d0, 2d0, 3d0, 4d0]
  real(8) :: yb(4) = [10d0, 20d0, 30d0, 40d0]
end module tables

module pick_mod
  use tables
  implicit none
contains
  subroutine pick(d, k, r)
    integer, intent(in) :: d, k
    real(8), intent(inout) :: r(:)
    select case (d)
    case (1)
      r(1) = ya(k)
    case (2)
      r(1) = yb(k)
    case default
      r(1) = ya(k)
    end select
    if (shifted) r(1) = r(1) + 100
  end subroutine pick
end module pick_mod
"""


def test_cases_that_read_different_arrays_stay_apart(tmp_path):
    # Cases alike but for the array they read must not be merged into one block that takes the array as an argument.
    hlfir = compile_to_hlfir(_SELECT_SRC, tmp_path / "hlfir", name="pick")
    sdfg = SDFGBuilder(str(hlfir), entry="pick_mod::pick", specialized_globals={"tables::shifted": False}).build()
    sdfg.build_folder = str(tmp_path / "pick")
    ya, yb = np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0])
    for d, expected in ((1, 3.0), (2, 30.0), (3, 3.0)):
        r = np.zeros(1)
        sdfg(d=d, k=3, r=r, ya=ya, yb=yb)
        assert r[0] == expected
