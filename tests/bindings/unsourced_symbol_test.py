"""The wrapper refuses to hand the SDFG a free symbol it has no value for.

A free symbol no argument, flatten-plan entry, module variable or record member gives a value used to get a
``! TODO: no plan entry gives size for free symbol`` comment, a declared but unassigned local, and that
uninitialised local as the argument of both entry points (and as a recorded value in a sizing run).  The emitter now
raises instead; a symbol the SDFG does not take keeps the comment, since nothing reads it.

Binding-side plumbing only: no HLFIR pipeline, bridge or Fortran compiler is involved.
"""

from pathlib import Path

import pytest

from dace_fortran.bindings import FlattenPlan, FrozenArg, FrozenSignature, OriginalArg, OriginalInterface, emit_bindings
from dace_fortran.bindings.block_builders import UnsourcedSymbolError
from dace_fortran.bindings.frozen_signature import FrozenArgKind


def _frozen() -> FrozenSignature:
    """``out(n, m)`` is the only argument; ``w_d1`` is the extent of a view nothing outside the SDFG knows."""
    out = FrozenArg(
        fortran_name="out", sdfg_name="out", kind=FrozenArgKind.ARRAY, dtype="float64", rank=2, shape=("n", "m")
    )
    return FrozenSignature(entry="kernel", mangled="_QPkernel", args=(out,), free_symbols=("m", "n", "w_d1"))


def _iface() -> OriginalInterface:
    return OriginalInterface(
        entry="kernel",
        args=(
            OriginalArg(name="out", fortran_type="real(c_double)", rank=2, shape=("n", "m"), intent="inout"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )


def _emit(tmp_path: Path, dace_arglist: tuple[str, ...], init_symbols: tuple[str, ...] | None) -> Path:
    return emit_bindings(
        _frozen(),
        _iface(),
        FlattenPlan(entries=()),
        str(tmp_path / "kernel_bindings.f90"),
        dace_arglist,
        init_symbols=init_symbols,
    )


def test_a_passed_symbol_without_a_source_is_refused(tmp_path: Path):
    with pytest.raises(UnsourcedSymbolError, match=r"no source for the free symbols \['w_d1'\]") as err:
        _emit(tmp_path, ("out", "m", "n", "w_d1"), ("m", "n", "w_d1"))
    assert err.value.symbols == ["w_d1"]
    assert not (tmp_path / "kernel_bindings.f90").exists()


def test_without_the_program_arguments_every_free_symbol_counts_as_passed(tmp_path: Path):
    with pytest.raises(UnsourcedSymbolError, match="w_d1"):
        _emit(tmp_path, (), None)


def test_a_symbol_the_sdfg_does_not_take_keeps_its_comment(tmp_path: Path):
    """``w_d1`` only shapes an array no state touches: DaCe passes it to neither entry point."""
    src = _emit(tmp_path, ("out", "m", "n"), ("m", "n")).read_text()
    assert "! TODO: no plan entry gives size for free symbol 'w_d1'" in src
    calls = [line for line in src.splitlines() if "dace_init_kernel(" in line and "=" in line]
    assert calls and all("w_d1" not in line for line in calls)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
