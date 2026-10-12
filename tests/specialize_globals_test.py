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


_ELEMENTS_SRC = """
module opts
  implicit none
  logical :: flags(3) = [.true., .false., .true.]
  real(8) :: weights(2) = [2d0, 3d0]
  integer :: counts(3) = 0
  integer :: levels(2) = [1, 2]
  integer :: steps(0:2) = [10, 20, 30]
end module opts

module recon
  use opts
  implicit none
contains
  subroutine blend(x, y)
    real(8), intent(in) :: x(:)
    real(8), intent(inout) :: y(:)
    if (flags(1)) then
      y = y + weights(1)*x
    else
      y = y - weights(2)*x
    end if
  end subroutine blend

  subroutine tally(x, y, d)
    real(8), intent(in) :: x(:)
    real(8), intent(inout) :: y(:)
    integer, intent(in) :: d
    if (flags(d)) y = y + x
    if (flags(2)) y = 2*y
    counts(d) = counts(d) + 1
    y(1) = y(1) + counts(1)
    levels(2) = levels(1)
  end subroutine tally

  subroutine pick(f, y)
    logical, intent(in) :: f(3)
    real(8), intent(inout) :: y(:)
    if (f(3)) y = -y
  end subroutine pick

  subroutine chain(y)
    real(8), intent(inout) :: y(:)
    call pick(flags, y)
  end subroutine chain

  subroutine take(s, y)
    integer, intent(in) :: s(3)
    real(8), intent(inout) :: y(:)
    y = y + s(1)
  end subroutine take

  subroutine shifted(y)
    real(8), intent(inout) :: y(:)
    y = y*steps(2)
    call take(steps, y)
  end subroutine shifted
end module recon
"""


def _branches(sdfg) -> int:
    """The conditional blocks and conditional interstate edges of ``sdfg``, nested ones included."""
    from dace.sdfg.state import ConditionalBlock

    blocks = sum(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks(recursive=True))
    return blocks + sum(not e.data.is_unconditional() for e in sdfg.all_interstate_edges(recursive=True))


def _elements_build(tmp_path, entry: str, values=None):
    hlfir = compile_to_hlfir(_ELEMENTS_SRC, tmp_path / "hlfir", name="recon")
    sdfg = SDFGBuilder(str(hlfir), entry=entry, specialized_globals=values).build()
    sdfg.name = f"{entry.split('::')[-1]}_{'spec' if values else 'general'}"
    sdfg.build_folder = str(tmp_path / sdfg.name)
    return sdfg


def test_elements_of_a_scalar_array_read_at_literal_subscripts_are_candidates(tmp_path):
    hlfir = compile_to_hlfir(_ELEMENTS_SRC, tmp_path / "hlfir", name="recon")
    assert SDFGBuilder(str(hlfir), entry="recon::blend").configuration_candidates == {
        "opts::flags(1)": "logical(4)",
        "opts::weights(1)": "real(8)",
        "opts::weights(2)": "real(8)",
    }
    # A read at a runtime subscript changes no element (flags(2) stays one); a write at one may change any (no element
    # of counts is); a write at another literal subscript is other storage (levels(1) stays, levels(2) goes).
    assert SDFGBuilder(str(hlfir), entry="recon::tally").configuration_candidates == {
        "opts::flags(2)": "logical(4)",
        "opts::levels(1)": "integer(4)",
    }


def test_a_branch_on_a_specialised_element_folds_and_the_array_goes(tmp_path):
    general = _elements_build(tmp_path, "recon::blend")
    assert {"flags", "weights"} <= set(general.arrays)

    sdfg = _elements_build(tmp_path, "recon::blend", {"opts::flags(1)": False, "opts::weights(2)": 3.0})
    assert not {"flags", "weights"} & set(sdfg.arrays)
    assert _branches(sdfg) < _branches(general)
    x, y = np.arange(4, dtype=np.float64), np.ones(4)
    sdfg(x=x, y=y)
    np.testing.assert_array_equal(y, 1 - 3 * x)


def test_an_element_the_entry_may_write_keeps_its_value_open(tmp_path):
    sdfg = _elements_build(tmp_path, "recon::tally", {"opts::flags(2)": False, "opts::counts(1)": 5})
    assert "counts" in sdfg.arrays  # written at a runtime subscript: counts(1) is not baked
    x, y = np.arange(4, dtype=np.float64), np.ones(4)
    flags = np.array([1, 1, 1], dtype=sdfg.arrays["flags"].dtype.type)  # flags(2) baked false, flags(d) read
    counts = np.array([5, 0, 0], dtype=np.int32)
    sdfg(x=x, y=y, d=1, flags=flags, counts=counts, levels=np.array([1, 2], dtype=np.int32))
    assert counts.tolist() == [6, 0, 0]
    np.testing.assert_array_equal(y, [1 + 0 + 6, 2, 3, 4])


def test_an_element_read_through_an_inlined_dummy_is_baked_after_inlining(tmp_path):
    hlfir = compile_to_hlfir(_ELEMENTS_SRC, tmp_path / "hlfir", name="recon")
    # Passed whole to a call before inlining; read at a literal subscript of the inlined dummy after it.
    assert "opts::flags(3)" in SDFGBuilder(str(hlfir), entry="recon::chain").configuration_candidates
    sdfg = _elements_build(tmp_path, "recon::chain", {"opts::flags(3)": True})
    assert "flags" not in sdfg.arrays
    y = np.arange(4, dtype=np.float64)
    sdfg(y=y)
    np.testing.assert_array_equal(y, -np.arange(4))


def test_an_inlined_dummy_numbers_the_elements_from_its_own_lower_bound(tmp_path):
    hlfir = compile_to_hlfir(_ELEMENTS_SRC, tmp_path / "hlfir", name="recon")
    # s(1) of the dummy s(3) bound to steps(0:2) is steps(0).
    candidates = SDFGBuilder(str(hlfir), entry="recon::shifted").configuration_candidates
    assert {"opts::steps(0)", "opts::steps(2)"} <= set(candidates) and "opts::steps(1)" not in candidates
    sdfg = _elements_build(tmp_path, "recon::shifted", {"opts::steps(0)": 10, "opts::steps(2)": 30})
    assert "steps" not in sdfg.arrays
    y = np.arange(4, dtype=np.float64)
    sdfg(y=y)
    np.testing.assert_array_equal(y, 30 * np.arange(4) + 10)
