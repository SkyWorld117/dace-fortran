"""Tests of the accessor copies of a build's modules.  Run: python -m pytest test_module_copies.py"""
from pathlib import Path

from dace_fortran.substitute.module_copies import access_entries, module_files, write_access_copies

_KINDS = """
module kinds
    implicit none
    integer, parameter :: rk = 8
end module kinds
"""

_OPTS = """
module opts
    use kinds
    implicit none
    private
    public :: s_init
    logical :: flags(3)
    real(rk) :: tol
contains
    subroutine s_init()
        flags = .true.
        tol = 0.5_rk
    end subroutine s_init
end module opts
"""


def test_one_copy_per_module_holds_the_variables_of_every_user(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "kinds.f90").write_text(_KINDS)
    (first / "opts.f90").write_text(_OPTS)
    (second / "opts.f90").write_text(_OPTS.replace("tol", "tolerance"))
    files = module_files([first, second])
    assert files == {"kinds": first / "kinds.f90", "opts": first / "opts.f90"}  # the first directory wins

    # The variables a binding needs and those a substitute entry checks, named twice and in any case: one copy.
    access = write_access_copies({"opts": ["flags", "TOL", "flags"]}, files, tmp_path / "access")
    assert [a["module"] for a in access] == ["opts"]
    assert sorted(access[0]["variables"]) == ["flags", "tol"]
    assert access[0]["imports"] == ["use kinds, only: rk"]
    copy = Path(access[0]["copy"])
    assert copy == tmp_path / "access" / "opts.f90"
    assert "public :: dace_access_flags, dace_access_tol" in copy.read_text()
    assert access_entries(access) == [
        {"module": "opts", "source": str(first / "opts.f90"), "copy": str(copy), "variables": ["flags", "tol"]}
    ]


def test_a_procedure_the_binding_imports_needs_no_accessor(tmp_path: Path):
    """A binding importing a module procedure next to a private variable (the call of a prologue, say): the procedure
    is reached as it is, only the variable gets an accessor in the module's copy."""
    from dace_fortran.substitute.make_substitute import expose_private_imports

    (tmp_path / "kinds.f90").write_text(_KINDS)
    (tmp_path / "opts.f90").write_text(_OPTS)
    bindings = tmp_path / "bindings.f90"
    bindings.write_text("module b\n  use opts, only: s_init, tol\nend module b\n")
    access = expose_private_imports(bindings, [tmp_path], tmp_path / "access")
    assert [(a["module"], sorted(a["variables"])) for a in access] == [("opts", ["tol"])]
