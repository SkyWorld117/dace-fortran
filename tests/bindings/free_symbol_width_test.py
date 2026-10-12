"""A free symbol that is not one of the snapshot's arguments (an extent DaCe inferred, named in no Fortran dummy)
is still passed to the C entry by value, in the width of the SDFG's symbol table (int64 for an inferred extent).
The bindings declared it ``integer(c_int)`` regardless, so the entry read a 64-bit parameter of which only 32 bits
had been written.  The snapshot now records the dtype of each free symbol (``FrozenSignature.symbol_dtypes``) and
the three declarations of such a symbol -- both C interfaces and the wrapper's local -- take it from there.

Binding-side plumbing only: no HLFIR pipeline, bridge or Fortran compiler is involved.
"""

from dataclasses import replace
from pathlib import Path

import dace

from dace_fortran.bindings import (
    FlattenPlan,
    FrozenArg,
    FrozenSignature,
    OriginalArg,
    OriginalInterface,
    emit_bindings,
)
from dace_fortran.bindings.frozen_signature import FrozenArgKind, symbol_dtypes_of

_NARROW = "integer(c_int), value :: n_d2"
_WIDE = "integer(c_int64_t), value :: n_d2"
_WIDE_LOCAL = "integer(c_int64_t) :: n_d2"


def _frozen(symbol_dtypes: dict[str, str]) -> FrozenSignature:
    """``out(n, m)`` is the only argument; ``n_d2`` is a free symbol no argument carries."""
    out = FrozenArg(
        fortran_name="out",
        sdfg_name="out",
        kind=FrozenArgKind.ARRAY,
        dtype="float64",
        rank=2,
        shape=("n", "m"),
        intent="inout",
    )
    return FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(out,),
        free_symbols=("m", "n", "n_d2"),
        symbol_dtypes=symbol_dtypes,
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


def _emit(frozen: FrozenSignature, tmp_path: Path) -> str:
    """The emitted module's source, with the symbols in the order DaCe's generated code takes them."""
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(
        frozen,
        _iface(),
        FlattenPlan(entries=()),
        str(out),
        dace_arglist=("out", "m", "n", "n_d2"),
        init_symbols=("m", "n", "n_d2"),
    )
    return out.read_text()


def _between(src: str, start: str, end: str) -> list[str]:
    """The stripped lines from the one starting with ``start`` up to the one starting with ``end``."""
    lines = [line.strip() for line in src.splitlines()]
    first = next(i for i, line in enumerate(lines) if line.startswith(start))
    last = next(i for i, line in enumerate(lines) if i > first and line.startswith(end))
    return lines[first : last + 1]


def test_a_shape_only_symbol_is_declared_in_the_width_the_snapshot_records(tmp_path: Path):
    src = _emit(_frozen({"m": "int32", "n": "int32", "n_d2": "int64"}), tmp_path)
    program = _between(src, "subroutine dace_program_kernel", "end subroutine")
    init = _between(src, "function dace_init_kernel", "end function")
    assert _WIDE in program
    assert _WIDE in init
    # The wrapper's local is the actual argument of both calls, so it has the same width.
    assert _WIDE_LOCAL in [line.strip() for line in src.splitlines()]
    assert _NARROW not in src
    # The symbols that are Fortran dummies keep the default integer.
    assert "integer(c_int), value :: n" in program and "integer(c_int), value :: m" in program


def test_a_snapshot_without_widths_keeps_the_default_integer(tmp_path: Path):
    """An old snapshot records no widths and gets the historical declaration: this is what the test above changes."""
    src = _emit(_frozen({}), tmp_path)
    assert _between(src, "subroutine dace_program_kernel", "end subroutine").count(_NARROW) == 1
    assert _between(src, "function dace_init_kernel", "end function").count(_NARROW) == 1
    assert "integer(c_int) :: n_d2" in src
    assert "c_int64_t" not in src


def test_symbol_dtypes_come_from_the_sdfgs_argument_list_and_do_not_affect_equality():
    sdfg = dace.SDFG("free_symbol_width")
    sdfg.add_symbol("n_d1", dace.int32)
    sdfg.add_symbol("n_d2", dace.int64)
    sdfg.add_array("src", [dace.symbol("n_d1"), dace.symbol("n_d2")], dace.float64)
    sdfg.add_array("dst", [dace.symbol("n_d1"), dace.symbol("n_d2")], dace.float64)
    state = sdfg.add_state()
    state.add_nedge(state.add_read("src"), state.add_write("dst"), dace.Memlet("src[0:n_d1, 0:n_d2]"))
    assert symbol_dtypes_of(sdfg, ["n_d1", "n_d2"]) == {"n_d1": "int32", "n_d2": "int64"}

    # The field is derived from the SDFG, so a snapshot that lacks it is the same snapshot.
    with_widths = _frozen({"n_d2": "int64"})
    without = replace(with_widths, symbol_dtypes={})
    assert with_widths == without
    assert with_widths.symbol_dtypes != without.symbol_dtypes


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
