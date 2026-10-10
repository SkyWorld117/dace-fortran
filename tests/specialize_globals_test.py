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


def test_a_global_the_entry_writes_keeps_its_value_open(tmp_path):
    sdfg = _build(tmp_path, "kern::rescale", {"cfg::scale": 1.5})
    assert "scale" in set(sdfg.arglist())
