"""Constant subscripts outside constant bounds in code a constant condition rules out: flang-21 rejects the source,
the codebase emitter clamps them (:func:`dace_fortran.flang_codebase.clamp_out_of_bounds_constant_subscripts`)."""

from pathlib import Path

import pytest

import dace_fortran
from dace_fortran.flang_codebase import clamp_out_of_bounds_constant_subscripts
from tests._util import flang_binary

_SOURCE = """# 1 "/src/m_grid.f90"
module m_grid
  implicit none
  integer, parameter :: num_dims = 2
contains
  subroutine neighbours(base, k, coords)
    integer, intent(in) :: base(num_dims), k
    integer, intent(out) :: coords(num_dims)
    coords(1) = base(1) + k
    if (num_dims > 1) coords(2) = base(2) + k
    if (num_dims > 2) coords(3) = base(3) + k
  end subroutine neighbours
end module m_grid
"""


def test_the_reported_subscripts_are_clamped_to_their_bounds():
    errors = (
        "/src/m_grid.f90:10:23: error: subscript 3 is greater than upper bound 2 for dimension 1 of array\n"
        "/src/m_grid.f90:10:37: error: subscript 3 is greater than upper bound 2 for dimension 1 of array\n"
    )
    patched, clamped = clamp_out_of_bounds_constant_subscripts(_SOURCE, errors)
    assert clamped == 2
    assert "if (num_dims > 2) coords(2) = base(2) + k" in patched
    assert "coords(1) = base(1) + k" in patched  # untouched lines stay as they are


def test_a_location_that_is_not_a_literal_subscript_is_left_alone():
    errors = "/src/m_grid.f90:10:23: error: subscript 4 is greater than upper bound 2 for dimension 1 of array\n"
    patched, clamped = clamp_out_of_bounds_constant_subscripts(_SOURCE, errors)
    assert clamped == 0
    assert patched == _SOURCE


def test_flang_lowers_a_module_with_a_dead_out_of_bounds_subscript(tmp_path: Path):
    try:
        flang = flang_binary()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"needs flang: {e}")
    entry = _SOURCE.split("\n", 1)[1]  # without the line marker: the emitter composes its own
    out = dace_fortran.emit_hlfir_from_codebase(entry_source=entry, out_path=tmp_path / "tu.hlfir", flang_program=flang)
    text = Path(out).read_text()
    assert "neighbours" in text


def test_a_line_expanded_more_than_once_is_clamped_where_the_subscript_is():
    """A preprocessor loop repeats a source line with different literals; the error names the source line, the clamp
    patches the expansion that holds the reported subscript."""
    source = (
        '# 1 "/src/m_dirs.f90"\n'
        "module m_dirs\n"
        "  implicit none\n"
        "contains\n"
        "  subroutine pick(c, out)\n"
        "    real(8), intent(in) :: c(0:0, 2)\n"
        "    real(8), intent(out) :: out(2)\n"
        '# 7 "/src/m_dirs.f90"\n'
        "    out(1) = c(0, 1)\n"
        '# 7 "/src/m_dirs.f90"\n'
        "    out(2) = c(1, 2)\n"
        "  end subroutine pick\n"
        "end module m_dirs\n"
    )
    errors = "/src/m_dirs.f90:7:14: error: subscript 1 is greater than upper bound 0 for dimension 1 of array\n"
    patched, clamped = clamp_out_of_bounds_constant_subscripts(source, errors)
    assert clamped == 1
    assert "out(1) = c(0, 1)" in patched and "out(2) = c(0, 2)" in patched


def test_cpp_directives_and_fypp_markers_before_a_dead_subscript(tmp_path: Path):
    source = _SOURCE.replace("    if (num_dims > 2)", "#if 0\n    coords(1) = 0\n#endif\n    if (num_dims > 2)")
    out = dace_fortran.emit_hlfir_from_codebase(entry_source=source, out_path=tmp_path / "mixed.hlfir",
                                               flang_program=flang_binary())
    assert "neighbours" in out.read_text()
