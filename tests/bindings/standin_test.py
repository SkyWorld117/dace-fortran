"""Stand-ins for a procedure (dace_fortran.bindings.standin), linked in its place with --wrap and --defsym: a recording
stand-in that writes module variables on every call, and a substitute entry that falls back to the original when the
binding declines."""

import subprocess
import os
from dataclasses import replace
from pathlib import Path

import pytest

from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.module_access import declaration_imports, expose_module_variables
from dace_fortran.bindings.specialization import recorded_invariants
from dace_fortran.bindings.standin import (
    RecordedValue,
    UnsupportedStandinError,
    emit_recording_standin,
    emit_substitute_entry,
)
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
    """Compile ``sources`` in order and link them, with the stand-in in place of kern::step when given. gfortran runs
    in ``tmp_path``: it looks for module files in its working directory first, where flang, run by other tests in the
    repository root, leaves its own of the same names."""
    objects = []
    for src in sources:
        obj = src.with_suffix(".o")
        flags = [f"-J{tmp_path}", f"-I{tmp_path}"]
        subprocess.run(["gfortran", "-c", str(src), "-o", str(obj), *flags], check=True, cwd=tmp_path)
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


def test_trace_reports_an_accepted_substitute_once_per_process(tmp_path: Path):
    _with_baked(tmp_path, _BAKED)
    env = {**os.environ, "DACE_SUBSTITUTE_TRACE": "1"}
    run = subprocess.run([str(tmp_path / "run_wrapped")], env=env, capture_output=True, text=True, check=True)
    assert run.stderr.count("DACE_SUBSTITUTE_ENGAGED step") == 1  # two accepted calls
    env.pop("DACE_SUBSTITUTE_TRACE")
    silent = subprocess.run([str(tmp_path / "run_wrapped")], env=env, capture_output=True, text=True, check=True)
    assert "DACE_SUBSTITUTE_ENGAGED" not in silent.stderr


def test_a_baked_value_that_differs_sends_the_call_to_the_original(tmp_path: Path):
    for i, value in ((0, 1), (1, 3), (2, 1.5000000000000002)):
        baked = list(_BAKED)
        baked[i] = (baked[i][0], value)
        sub = tmp_path / f"case{i}"
        sub.mkdir()
        plain, substituted = _with_baked(sub, baked)
        assert substituted == plain, baked[i]


# Configuration a module keeps private: an array of flags, a real of a kind from another module, an array of records and
# a record of a type from another module; one public variable beside them.
_OPTS_KINDS = """
module opts_kinds
    implicit none
    integer, parameter :: rk = 8
    type span
        integer :: lo, hi
    end type span
end module opts_kinds
"""

_OPTS = """
module opts
    use opts_kinds
    implicit none
    private
    public :: n_steps
    integer :: n_steps = 3
    logical :: flags(3) = [.true., .false., .true.]
    real(rk) :: tol = 0.25_rk
    type(span) :: spans(2) = [span(1, 4), span(-2, 7)]
    type(span) :: whole = span(5, 6)
end module opts
"""

_PRIVATE = ["flags", "tol", "spans", "whole"]


def _compile(src: Path, mod_dir: Path, *include: Path) -> Path:
    """``src`` compiled to an object beside it, its module files in ``mod_dir`` (gfortran runs there, see ``_run``)."""
    obj = src.with_suffix(".o")
    flags = [f"-J{mod_dir}", f"-I{mod_dir}", *(f"-I{d}" for d in include)]
    subprocess.run(["gfortran", "-c", str(src), "-o", str(obj), *flags], check=True, cwd=mod_dir)
    return obj


def _link_and_run(exe: Path, objects: list[Path], standin_symbol: str | None) -> str:
    wrap = []
    if standin_symbol:
        sym = "__kern_MOD_step"
        wrap = [
            f"-Wl,--wrap={sym}",
            f"-Wl,--defsym=__wrap_{sym}={standin_symbol}_",
            f"-Wl,--defsym=dace_original_step_=__real_{sym}",
        ]
    subprocess.run(["gfortran", *map(str, objects), *wrap, "-o", str(exe)], check=True)
    return subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout


class _PrivateProgram:
    """The program of ``_MAIN`` with the modules of ``_OPTS`` beside it, compiled against the original modules, and a
    copy of ``opts`` with accessors to its private variables, compiled with its module file aside (as a stand-in never
    sees it) to link in place of the original object."""

    def __init__(self, tmp_path: Path):
        self.dir = tmp_path
        for name, text in (("kinds", _OPTS_KINDS), ("opts", _OPTS), ("modules", _MODULES), ("main", _MAIN)):
            (tmp_path / f"{name}.f90").write_text(text)
        self.kinds, self.opts, self.modules, self.main = (
            _compile(tmp_path / f"{name}.f90", tmp_path) for name in ("kinds", "opts", "modules", "main")
        )
        copy_dir = tmp_path / "copy"
        copy_dir.mkdir()
        source, self.exposed = expose_module_variables(_OPTS, "opts", _PRIVATE)
        (copy_dir / "opts.f90").write_text(source)
        self.copy = _compile(copy_dir / "opts.f90", copy_dir, tmp_path)
        self.imports = declaration_imports(
            list(self.exposed.values()), "opts", {"opts": _OPTS, "opts_kinds": _OPTS_KINDS}
        )
        self.plain = _link_and_run(tmp_path / "run_plain", [self.kinds, self.opts, self.modules, self.main], None)

    def run(self, standin: Path, symbol: str, *extra: Path) -> str:
        """Output of the program with ``standin`` (compiled against the original modules) in place of kern::step and
        the copy in place of opts."""
        objects = [self.kinds, self.copy, self.modules, *(_compile(e, self.dir) for e in extra)]
        objects += [_compile(standin, self.dir), self.main]
        return _link_and_run(self.dir / f"run_{standin.stem}", objects, symbol)


_OPTS_VALUES = [
    RecordedValue("opts", "n_steps", "integer(4)"),
    RecordedValue("opts", "flags(1)", "logical(4)"),
    RecordedValue("opts", "flags(2)", "logical(4)"),
    RecordedValue("opts", "tol", "real(8)"),
    RecordedValue("opts", "spans(2)%hi", "integer(4)"),
    RecordedValue("opts", "whole%lo", "integer(4)"),
]


def test_private_variables_are_recorded_through_the_accessors_of_a_copy(tmp_path: Path):
    iface = _interface(tmp_path)
    program = _PrivateProgram(tmp_path)
    assert program.imports == ["use opts_kinds, only: rk, span"]
    record = tmp_path / "calls.record"
    standin = emit_recording_standin(
        iface, _OPTS_VALUES, record, tmp_path / "record_step.f90", {"opts": program.exposed}, program.imports
    )
    text = standin.read_text()
    assert "use opts, only: dace_value_0 => n_steps" in text  # public: imported as before
    assert "use opts, only: dace_value_1" not in text  # private: no import, which would not compile
    assert program.run(standin, "dace_record_step") == program.plain
    assert recorded_invariants(record) == {
        "opts::n_steps": 3,
        "opts::flags(1)": 1,
        "opts::flags(2)": 0,
        "opts::tol": 0.25,
        "opts::spans(2)%hi": 7,
        "opts::whole%lo": 5,
    }


_OPTS_BAKED = [
    (RecordedValue("opts", "flags(2)", "logical(4)"), 0),
    (RecordedValue("opts", "spans(2)%hi", "integer(4)"), 7),
    (RecordedValue("opts", "tol", "real(8)"), 0.25),
    (RecordedValue("opts", "n_steps", "integer(4)"), 3),
]


def test_baked_private_values_are_checked_through_the_accessors_of_a_copy(tmp_path: Path):
    iface = _interface(tmp_path)
    program = _PrivateProgram(tmp_path)
    (tmp_path / "binding.f90").write_text(_BINDING_TAKES)
    for i, value in ((None, None), (0, 1), (1, 4), (2, 0.5), (3, 2)):
        baked = list(_OPTS_BAKED)
        if i is not None:
            baked[i] = (baked[i][0], value)
        entry = emit_substitute_entry(
            iface,
            "step_dace_bindings",
            "step_dace",
            tmp_path / f"sub_step_{i}.f90",
            baked,
            {"opts": program.exposed},
            program.imports,
        )
        substituted = program.run(entry, "dace_sub_step", tmp_path / "binding.f90")
        if i is None:
            assert substituted != program.plain  # every value holds: the binding ran (and did nothing)
        else:
            assert substituted == program.plain, baked[i]


def test_a_private_allocatable_is_refused(tmp_path: Path):
    iface = _interface(tmp_path)
    _, exposed = expose_module_variables(_OPTS, "opts", ["flags"])
    allocatable = {"opts": {"flags": replace(exposed["flags"], allocatable=True)}}
    with pytest.raises(UnsupportedStandinError, match="allocatable"):
        emit_recording_standin(iface, _OPTS_VALUES[1:2], tmp_path / "r", tmp_path / "s.f90", allocatable)


def test_a_handover_after_taken_calls_stops_when_the_binding_keeps_buffers(tmp_path: Path):
    """The first call is taken, then a baked value changes, so the second call goes to the original. With
    ``refuse_handover`` (a binding with buffers kept between calls, which the members no longer follow) the program
    stops there instead; without it, the original takes over."""
    iface = _interface(tmp_path)
    main = _MAIN.replace("    use kern, only: step\n", "    use kern, only: step\n    use cfg, only: nsweeps\n")
    main = main.replace("    call step(q, 4, x, y, 3)\n", "    nsweeps = 5\n    call step(q, 4, x, y, 3)\n")
    (tmp_path / "modules.f90").write_text(_MODULES)
    (tmp_path / "main.f90").write_text(main)
    (tmp_path / "binding.f90").write_text(_BINDING_TAKES)
    for refuse in (False, True):
        entry = emit_substitute_entry(iface, "step_dace_bindings", "step_dace", tmp_path / "sub_step.f90", _BAKED,
                                      refuse_handover=refuse)
        sources = [tmp_path / "modules.f90", tmp_path / "binding.f90", entry, tmp_path / "main.f90"]
        if refuse:
            with pytest.raises(subprocess.CalledProcessError) as stopped:
                _run(tmp_path, sources, "dace_sub_step")
            assert "after it took calls" in stopped.value.stderr
        else:
            _run(tmp_path, sources, "dace_sub_step")


def test_a_recording_stand_in_per_rank_writes_the_file_of_its_rank(tmp_path: Path, monkeypatch):
    from dace_fortran.bindings.specialization import RANK_VARIABLES

    iface = _interface(tmp_path)
    (tmp_path / "modules.f90").write_text(_MODULES)
    (tmp_path / "main.f90").write_text(_MAIN)
    record = tmp_path / "calls.record"
    values = [RecordedValue("cfg", "nsweeps", "integer(4)")]
    standin = emit_recording_standin(iface, values, record, tmp_path / "record_step.f90", per_rank=True)
    for var in RANK_VARIABLES:  # the test itself may run under a launcher
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    _run(tmp_path, [tmp_path / "modules.f90", standin, tmp_path / "main.f90"], "dace_record_step")
    assert recorded_invariants(tmp_path / "calls.record.2") == {"cfg::nsweeps": 2}
    assert not record.exists()


def test_an_extent_reached_through_a_re_exporting_module_is_imported_from_its_own():
    """``bc(num_dims, 2)`` with ``num_dims`` a variable of m_common that the routine's module reaches through
    m_params (which re-exports it): the routine's imports name no module for it, so the stand-in takes the module
    the recorded value of ``num_dims`` names."""
    from dace_fortran.bindings.fortran_interface import OriginalArg, OriginalInterface
    from dace_fortran.bindings.standin import render_dummies

    iface = OriginalInterface(
        entry="step",
        args=(OriginalArg(name="bc", fortran_type="integer", rank=2, shape=("num_dims", "2"), intent="in",
                          passing="address"),),
        struct_types={},
        used_modules={"m_params": ["sys_size"]},
    )
    uses, decls = render_dummies(iface, {"num_dims": "m_common"})
    assert "use m_common, only: num_dims" in uses
    assert "integer, intent(in) :: bc(num_dims, 2)" in decls
    assert "use m_common, only: num_dims" not in render_dummies(iface)[0]
