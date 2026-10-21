"""A module array the SDFG allocates itself (a transient in the module array's place) is not a frozen argument, yet
the kernel's subscripts into it still take the module array's lower bounds and extents as free symbols
(``offset_<array>_d<i>`` and ``<array>_d<i>``).  The wrapper must read them from the module variable's import alias;
before it did, each of them fell through to ``! TODO: no plan entry gives size`` and stayed unset.

Binding-side plumbing only: no HLFIR pipeline, bridge or Fortran compiler is involved.
"""

from pathlib import Path

from dace_fortran.bindings import (
    FlattenPlan,
    FrozenArg,
    FrozenSignature,
    OriginalArg,
    OriginalInterface,
    emit_bindings,
)
from dace_fortran.bindings.block_builders import _build_symbol_assigns
from dace_fortran.bindings.frozen_signature import FrozenArgKind, ModuleOrigin

_WORK_SYMBOLS = ("offset_work_d0", "offset_work_d1", "work_d0", "work_d1")


def _out_arg() -> FrozenArg:
    return FrozenArg(
        fortran_name="out",
        sdfg_name="out",
        kind=FrozenArgKind.ARRAY,
        dtype="float64",
        rank=2,
        shape=("n", "m"),
        intent="inout",
    )


def _frozen(origins: dict[str, ModuleOrigin]) -> FrozenSignature:
    """``out(n, m)`` is the only argument; ``work`` is a transient the SDFG allocates, so it appears among the free
    symbols (its lower bounds and extents) and in ``origins`` but not among the arguments."""
    return FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(_out_arg(),),
        free_symbols=("m", "n", *_WORK_SYMBOLS),
        module_symbol_origins=origins,
    )


def _iface() -> OriginalInterface:
    return OriginalInterface(
        entry="kernel",
        args=(
            OriginalArg(name="out", fortran_type="real(c_double)", rank=2, shape=("n", "m"), intent="inout"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )


_STATE_WORK = {"work": ModuleOrigin("m_state", "work")}


def test_transient_module_array_bounds_are_read_from_the_module_variable():
    lines = _build_symbol_assigns(_frozen(_STATE_WORK), FlattenPlan(entries=()), {"out", "n", "m"}, _iface())
    text = "\n".join(lines)
    assert "    offset_work_d0 = int(lbound(work__mod, dim=1), c_int)" in lines
    assert "    offset_work_d1 = int(lbound(work__mod, dim=2), c_int)" in lines
    assert "    work_d0 = int(size(work__mod, dim=1), c_int)" in lines
    assert "    work_d1 = int(size(work__mod, dim=2), c_int)" in lines
    for sym in _WORK_SYMBOLS:
        assert f"TODO: no plan entry gives size for free symbol '{sym}'" not in text
    assert "TODO: no plan entry" not in text


def test_transient_module_array_bounds_are_not_guessed_without_a_module_origin():
    """Only a name the bridge traced to a module variable is read from one: a transient of the kernel's own has no
    caller-side storage to measure, and keeps the TODO rather than reading an import that does not exist."""
    text = "\n".join(_build_symbol_assigns(_frozen({}), FlattenPlan(entries=()), {"out", "n", "m"}, _iface()))
    assert "work__mod" not in text
    for sym in _WORK_SYMBOLS:
        assert f"! TODO: no plan entry gives size for free symbol '{sym}'" in text


def test_the_assembled_wrapper_imports_the_alias_and_assigns_the_bounds(tmp_path: Path):
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(_frozen(_STATE_WORK), _iface(), FlattenPlan(entries=()), str(out))
    src = out.read_text()
    assert "use m_state, only: work__mod => work" in src
    assert "offset_work_d0 = int(lbound(work__mod, dim=1), c_int)" in src
    assert "offset_work_d1 = int(lbound(work__mod, dim=2), c_int)" in src
    assert "work_d0 = int(size(work__mod, dim=1), c_int)" in src
    assert "work_d1 = int(size(work__mod, dim=2), c_int)" in src
    assert "no plan entry gives size" not in src
    # The SDFG allocates `work`: the wrapper holds no local copy of it and passes none.
    assert "allocate(work(" not in src and "work = work__mod" not in src


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
