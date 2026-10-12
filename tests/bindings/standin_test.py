"""Stand-ins for a procedure (dace_fortran.bindings.standin), linked in its place with --wrap and --defsym: a recording
stand-in that writes module variables on every call, and a substitute entry that falls back to the original when the
binding declines."""

import subprocess
from pathlib import Path

from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.specialization import recorded_invariants
from dace_fortran.bindings.standin import RecordedValue, emit_recording_standin, emit_substitute_entry
from tests._util import build_sdfg

_MODULES = """
module cfg
    implicit none
    logical :: use_extra = .false.
    integer :: nsweeps = 2
    real(8) :: scale = 1.5d0
end module cfg

module fields
    implicit none
    type scalar_field
        real(8), pointer :: sf(:) => null()
    end type scalar_field
end module fields

module kern
    use cfg
    use fields
    implicit none
contains
    subroutine step(q, n, x, y, k, t)
        integer, intent(in) :: n
        type(scalar_field), dimension(2), intent(inout) :: q
        real(8), dimension(0:), intent(in) :: x
        real(8), dimension(n), intent(inout) :: y
        integer, value :: k
        real(8), optional, intent(in) :: t
        integer :: i
        do i = 1, n
            y(i) = y(i) + scale * x(i - 1) * k
        end do
        q(1)%sf = q(1)%sf + nsweeps
        if (present(t)) y(1) = y(1) + t
    end subroutine step
end module kern
"""

_MAIN = """
program main
    use fields, only: scalar_field
    use kern, only: step
    implicit none
    type(scalar_field) :: q(2)
    real(8), target :: a(3), b(3)
    real(8) :: x(0:3), y(4)
    a = 1
    b = 2
    q(1)%sf => a
    q(2)%sf => b
    x = [1d0, 2d0, 3d0, 4d0]
    y = 0
    call step(q, 4, x, y, 3, 0.5d0)
    call step(q, 4, x, y, 3)
    print '(4f8.2, f8.2)', y, a(1)
end program main
"""

# A binding that always takes the call and does nothing.
_BINDING_TAKES = """
module step_dace_bindings
    use fields, only: scalar_field
    implicit none
contains
    subroutine step_dace(q, n, x, y, k, t, dace_declined)
        integer, intent(in) :: n
        type(scalar_field), dimension(2), intent(inout) :: q
        real(8), dimension(0:), intent(in) :: x
        real(8), dimension(n), intent(inout) :: y
        integer, value :: k
        real(8), optional, intent(in) :: t
        logical, optional, intent(out) :: dace_declined
        if (present(dace_declined)) dace_declined = .false.
    end subroutine step_dace
end module step_dace_bindings
"""

# A binding that always declines, counting its calls.
_BINDING = """
module step_dace_bindings
    use fields, only: scalar_field
    implicit none
    integer :: calls = 0
contains
    subroutine step_dace(q, n, x, y, k, t, dace_declined)
        integer, intent(in) :: n
        type(scalar_field), dimension(2), intent(inout) :: q
        real(8), dimension(0:), intent(in) :: x
        real(8), dimension(n), intent(inout) :: y
        integer, value :: k
        real(8), optional, intent(in) :: t
        logical, optional, intent(out) :: dace_declined
        calls = calls + 1
        if (present(dace_declined)) dace_declined = .true.
    end subroutine step_dace
end module step_dace_bindings
"""


def _interface(tmp_path: Path):
    builder = build_sdfg(_MODULES, tmp_path / "sdfg", name="step", entry="kern::step")
    return build_auto_interface(builder._fortran_interface_raw, "step")


def _run(tmp_path: Path, sources: list[Path], standin_symbol: str | None) -> str:
    """Compile ``sources`` in order and link them, with the stand-in in place of kern::step when given."""
    objects = []
    for src in sources:
        obj = src.with_suffix(".o")
        subprocess.run(["gfortran", "-c", str(src), "-o", str(obj), f"-J{tmp_path}", f"-I{tmp_path}"], check=True)
        objects.append(str(obj))
    wrap = []
    if standin_symbol:
        sym = "__kern_MOD_step"  # gfortran's name of the module procedure
        wrap = [
            f"-Wl,--wrap={sym}",
            f"-Wl,--defsym=__wrap_{sym}={standin_symbol}_",
            f"-Wl,--defsym=dace_original_step_=__real_{sym}",
        ]
    exe = tmp_path / ("run_wrapped" if standin_symbol else "run_plain")
    subprocess.run(["gfortran", *objects, *wrap, "-o", str(exe)], check=True)
    return subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout


def test_dummies_pass_through_unchanged_and_the_recording_has_the_values(tmp_path: Path):
    iface = _interface(tmp_path)
    assert [(a.name, a.passing) for a in iface.args] == [
        ("q", "address"),
        ("n", "address"),
        ("x", "descriptor"),
        ("y", "address"),
        ("k", "value"),
        ("t", "address"),
    ]
    (tmp_path / "modules.f90").write_text(_MODULES)
    (tmp_path / "main.f90").write_text(_MAIN)
    plain = _run(tmp_path, [tmp_path / "modules.f90", tmp_path / "main.f90"], None)

    # A path longer than a Fortran line, with a quote in it.
    folder = tmp_path / ("it's " + "a" * 150)
    folder.mkdir()
    record = folder / "calls.record"
    values = [
        RecordedValue("cfg", "use_extra", "logical(4)"),
        RecordedValue("cfg", "nsweeps", "integer(4)"),
        RecordedValue("cfg", "scale", "real(8)"),
    ]
    standin = emit_recording_standin(iface, values, record, tmp_path / "record_step.f90")
    wrapped = _run(tmp_path, [tmp_path / "modules.f90", standin, tmp_path / "main.f90"], "dace_record_step")
    assert wrapped == plain
    assert record.read_text().count("\n\n") == 2  # two calls, each ended by an empty line
    assert recorded_invariants(record) == {"cfg::use_extra": 0, "cfg::nsweeps": 2, "cfg::scale": 1.5}


def test_a_declining_binding_leaves_the_call_to_the_original(tmp_path: Path):
    iface = _interface(tmp_path)
    (tmp_path / "modules.f90").write_text(_MODULES)
    (tmp_path / "main.f90").write_text(_MAIN)
    (tmp_path / "binding.f90").write_text(_BINDING)
    plain = _run(tmp_path, [tmp_path / "modules.f90", tmp_path / "main.f90"], None)
    entry = emit_substitute_entry(iface, "step_dace_bindings", "step_dace", tmp_path / "sub_step.f90")
    sources = [tmp_path / "modules.f90", tmp_path / "binding.f90", entry, tmp_path / "main.f90"]
    assert _run(tmp_path, sources, "dace_sub_step") == plain


def _with_baked(tmp_path: Path, baked) -> tuple[str, str]:
    """The output of the plain program and of the one whose step goes through a substitute entry that checks
    ``baked`` in front of a binding that takes every call and does nothing."""
    iface = _interface(tmp_path)
    (tmp_path / "modules.f90").write_text(_MODULES)
    (tmp_path / "main.f90").write_text(_MAIN)
    (tmp_path / "binding.f90").write_text(_BINDING_TAKES)
    plain = _run(tmp_path, [tmp_path / "modules.f90", tmp_path / "main.f90"], None)
    entry = emit_substitute_entry(iface, "step_dace_bindings", "step_dace", tmp_path / "sub_step.f90", baked)
    sources = [tmp_path / "modules.f90", tmp_path / "binding.f90", entry, tmp_path / "main.f90"]
    return plain, _run(tmp_path, sources, "dace_sub_step")


_BAKED = [
    (RecordedValue("cfg", "use_extra", "logical(4)"), 0),
    (RecordedValue("cfg", "nsweeps", "integer(4)"), 2),
    (RecordedValue("cfg", "scale", "real(8)"), 1.5),
]


def test_baked_values_that_hold_let_the_binding_take_the_call(tmp_path: Path):
    plain, substituted = _with_baked(tmp_path, _BAKED)
    assert substituted != plain  # the binding ran (and did nothing)


def test_a_baked_value_that_differs_sends_the_call_to_the_original(tmp_path: Path):
    for i, value in ((0, 1), (1, 3), (2, 1.5000000000000002)):
        baked = list(_BAKED)
        baked[i] = (baked[i][0], value)
        sub = tmp_path / f"case{i}"
        sub.mkdir()
        plain, substituted = _with_baked(sub, baked)
        assert substituted == plain, baked[i]
