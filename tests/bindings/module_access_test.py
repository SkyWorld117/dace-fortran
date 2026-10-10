"""Accessors to a module's private variables (dace_fortran.bindings.module_access): the copy of the module's source
they are appended to, and linking that copy in place of the original object."""

import subprocess
from pathlib import Path

import pytest

from dace_fortran.bindings.module_access import (
    ModuleAccessError,
    changed_global_data_symbols,
    expose_module_variables,
)

# The declaration forms a large code base uses for module state: several entities per statement, attribute lists with
# a dimension, scalar records, trailing doc comments, and private-by-default with a public list on one line. The
# record types come from a module of their own, so a binding can declare pointers of them.
_FIELDS = """
module fields
    implicit none
    type scalar_field
        real(8), pointer :: sf(:, :, :) => null()
    end type scalar_field
    type vector_field
        type(scalar_field), allocatable :: vf(:)
    end type vector_field
end module fields
"""

_OWNER = """
module owner
    use fields
    implicit none
    private; public :: s_init, n_public
    integer, parameter :: nmax = 4
    integer :: n_public = 0
    type(vector_field) :: q_cons, q_prim  !< state at the quadrature points
    type(vector_field), allocatable, dimension(:) :: flux_n, flux_src_n
    real(8), allocatable, dimension(:, :, :, :) :: ql_rsx, &
        qr_rsx
    real(8), pointer :: view(:) => null()
contains
    subroutine s_init(n)
        integer, intent(in) :: n
        integer :: i
        allocate(q_prim%vf(2), flux_n(3))
        do i = 1, 2
            allocate(q_prim%vf(i)%sf(-3:n, 0:n, 0:n))
            q_prim%vf(i)%sf = 10 * i
        end do
        allocate(ql_rsx(n, n, n, 2))
        ql_rsx = 7
        n_public = n
    end subroutine s_init
end module owner
"""


def test_accessors_are_appended_and_the_declarations_kept():
    copy, exposed = expose_module_variables(_OWNER, "owner", ["q_prim", "flux_n", "QR_RSX", "view"])
    for line in _OWNER.splitlines():
        assert line in copy.splitlines()  # nothing of the original changes
    assert "    target :: q_prim, flux_n, qr_rsx" in copy  # the pointer needs no target attribute
    assert "    public :: dace_access_q_prim, dace_access_flux_n, dace_access_qr_rsx, dace_access_view" in copy
    assert copy.index("public :: dace_access_q_prim") < copy.index("contains")
    assert (exposed["q_prim"].type_spec, exposed["q_prim"].rank, exposed["q_prim"].allocatable) == (
        "type(vector_field)",
        0,
        False,
    )
    assert (exposed["qr_rsx"].type_spec, exposed["qr_rsx"].rank) == ("real(8)", 4)
    assert exposed["view"].pointer and exposed["view"].rank == 1
    assert "real(8), pointer :: p(:, :, :, :)" in copy
    assert "if (allocated(flux_n)) then" in copy


def test_constants_unknown_names_and_taken_accessor_names_are_refused():
    with pytest.raises(ModuleAccessError, match="named constant"):
        expose_module_variables(_OWNER, "owner", ["nmax"])
    with pytest.raises(ModuleAccessError, match="no variable"):
        expose_module_variables(_OWNER, "owner", ["q_missing"])
    with pytest.raises(ModuleAccessError, match="already declares"):
        taken = _OWNER.replace("    integer :: n_public = 0\n", "    integer :: n_public = 0\n    integer :: dace_access_q_prim\n")
        expose_module_variables(taken, "owner", ["q_prim"])
    with pytest.raises(ModuleAccessError, match="no module"):
        expose_module_variables(_OWNER, "other", ["q_prim"])


_USER = """
subroutine user_report(n) bind(c, name="user_report")
    use owner, only: n_public
    integer, intent(out) :: n
    n = n_public
end subroutine user_report
"""

# The binding points at the state through pointer variables: ``associate`` on an accessor's result is not safe, gfortran
# copies a record selector with allocatable components and deallocates the copy's components, which are the module's.
_BINDING = """
program binding
    use fields, only: vector_field
    use owner, only: s_init, dace_access_q_prim, dace_access_ql_rsx
    implicit none
    interface
        subroutine user_report(n) bind(c, name="user_report")
            integer, intent(out) :: n
        end subroutine user_report
    end interface
    type(vector_field), pointer :: q
    real(8), pointer :: ql(:, :, :, :)
    integer :: n
    call s_init(4)
    q => dace_access_q_prim()
    print '(3i4, f8.1)', lbound(q%vf(2)%sf), q%vf(2)%sf(-3, 0, 0)
    ql => dace_access_ql_rsx()
    call user_report(n)
    print '(f8.1, i4)', ql(4, 4, 4, 2), n
end program binding
"""


def _compile(fc: str, src: Path, obj: Path, mod_dir: Path) -> None:
    subprocess.run([fc, "-c", str(src), "-o", str(obj), f"-J{mod_dir}", f"-I{mod_dir}"], check=True)


def test_the_copy_links_in_place_of_the_original_object(tmp_path: Path):
    orig, copy = tmp_path / "orig", tmp_path / "copy"
    orig.mkdir()
    copy.mkdir()
    for d in (orig, copy):
        (d / "fields.f90").write_text(_FIELDS)
        _compile("gfortran", d / "fields.f90", d / "fields.o", d)
    (orig / "owner.f90").write_text(_OWNER)
    (orig / "user.f90").write_text(_USER)
    _compile("gfortran", orig / "owner.f90", orig / "owner.o", orig)
    _compile("gfortran", orig / "user.f90", orig / "user.o", orig)  # another object, built against the original

    source, _ = expose_module_variables(_OWNER, "owner", ["q_prim", "ql_rsx"])
    (copy / "owner.f90").write_text(source)
    (copy / "binding.f90").write_text(_BINDING)
    _compile("gfortran", copy / "owner.f90", copy / "owner.o", copy)
    _compile("gfortran", copy / "binding.f90", copy / "binding.o", copy)
    assert changed_global_data_symbols(orig / "owner.o", copy / "owner.o") == []

    exe = tmp_path / "run"
    objects = [copy / "binding.o", orig / "user.o", copy / "owner.o", orig / "fields.o"]
    subprocess.run(["gfortran", *map(str, objects), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["-3", "0", "0", "20.0", "7.0", "4"]


def test_a_changed_data_layout_is_reported(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "m.f90").write_text("module m\n    real(8), public :: table(4)\nend module m\n")
    (b / "m.f90").write_text("module m\n    real(8), public :: table(5)\nend module m\n")
    _compile("gfortran", a / "m.f90", a / "m.o", a)
    _compile("gfortran", b / "m.f90", b / "m.o", b)
    assert changed_global_data_symbols(a / "m.o", b / "m.o") == ["__m_MOD_table"]
