"""The ``__dace_init`` parameter list the binding passes is read from the header DaCe generates into the build
folder.  Two specialisations of one routine compile to SDFGs of the same name, so when they are built at the same
time in one build folder, the header one binding reads can be the other's.  An inlined callee called once more in
the other specialisation gives its assumed-shape dummy one more extent symbol (``x_call1_d1`` beside
``x_call0_d1``), and an array only the other one reads brings its own extents; the binding then passed those symbols
to ``__dace_init`` without declaring them and did not compile, and a header with fewer symbols would have bound the
rest in the wrong positions.  The emitter now refuses an init list that is not exactly the symbols the SDFG's
``__program`` takes.

Binding-side plumbing only: no HLFIR pipeline, bridge or Fortran compiler is involved.
"""

import re
from pathlib import Path

import pytest

from dace_fortran.bindings import (
    FlattenPlan,
    FrozenArg,
    FrozenSignature,
    OriginalArg,
    OriginalInterface,
    SignatureDriftError,
    emit_bindings,
)
from dace_fortran.bindings.frozen_signature import FrozenArgKind

#: The live ``__program`` argument list: the array, then the free symbols in DaCe's order.
_ARGLIST = ("out", "m", "n", "x_call0_d1")
_INIT = ("m", "n", "x_call0_d1")


def _frozen(free_symbols: tuple[str, ...] = ("m", "n", "x_call0_d1")) -> FrozenSignature:
    """``out(n, m)`` is the only argument; ``x_call0_d1`` is the extent of an inlined callee's dummy."""
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
        free_symbols=free_symbols,
        symbol_dtypes={s: "int64" if "call" in s else "int32" for s in free_symbols},
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


def _emit(tmp_path: Path, init_symbols, dace_arglist=_ARGLIST, frozen: FrozenSignature | None = None) -> Path:
    out = tmp_path / "kernel_bindings.f90"
    return emit_bindings(
        frozen or _frozen(),
        _iface(),
        FlattenPlan(entries=()),
        str(out),
        dace_arglist=dace_arglist,
        init_symbols=init_symbols,
    )


def _init_call_args(src: str) -> list[str]:
    """The actual arguments of the wrapper's ``dace_init_kernel`` call."""
    call = src[src.index("dace_handle = dace_init_kernel(") :]
    call = call[call.index("(") + 1 : call.index(")")]
    return [a.strip() for a in call.replace("&", "").split(",")]


def test_the_header_of_this_build_is_passed_through(tmp_path: Path):
    src = _emit(tmp_path, _INIT).read_text()
    assert _init_call_args(src) == list(_INIT)
    assert "integer(c_int64_t) :: x_call0_d1" in [line.strip() for line in src.splitlines()]


@pytest.mark.parametrize(
    ("other_header", "foreign"),
    [
        # The other specialisation inlines the callee twice: its header also takes the second call's extent.
        (("m", "n", "x_call0_d1", "x_call1_d1"), ["x_call1_d1"]),
        # It reads an array this one never touches: its header takes that array's extents, sorted among ours.
        (("grad_d0", "grad_d1", "m", "n", "x_call0_d1"), ["grad_d0", "grad_d1"]),
    ],
)
def test_a_header_naming_symbols_of_another_build_is_refused(tmp_path: Path, other_header, foreign):
    with pytest.raises(SignatureDriftError, match=re.escape(f"takes {foreign}, which are not symbols of this SDFG")):
        _emit(tmp_path, other_header)
    # Refused before anything is written: no binding that cannot compile is left behind.
    assert not (tmp_path / "kernel_bindings.f90").exists()


def test_a_header_lacking_a_symbol_the_program_takes_is_refused(tmp_path: Path):
    # The other specialisation does not call the callee: every later init argument would shift by one.
    with pytest.raises(SignatureDriftError, match=r"lacks \['x_call0_d1'\], which __program takes"):
        _emit(tmp_path, ("m", "n"))


def test_a_header_taking_a_symbol_the_program_does_not_is_refused(tmp_path: Path):
    # ``k`` is a free symbol of this SDFG its code no longer uses: DaCe passes it to neither entry point.
    frozen = _frozen(("k", "m", "n", "x_call0_d1"))
    with pytest.raises(SignatureDriftError, match=r"takes \['k'\], which __program does not take"):
        _emit(tmp_path, ("k", "m", "n", "x_call0_d1"), frozen=frozen)


def test_a_header_in_another_order_is_refused(tmp_path: Path):
    with pytest.raises(SignatureDriftError, match="another order"):
        _emit(tmp_path, ("n", "m", "x_call0_d1"))


def test_an_unused_free_symbol_is_not_expected_in_the_header(tmp_path: Path):
    """The case the header is read for: a free symbol only the shape of an untouched array uses is in neither list."""
    src = _emit(tmp_path, _INIT, frozen=_frozen(("k", "m", "n", "x_call0_d1"))).read_text()
    assert _init_call_args(src) == list(_INIT)


def test_without_the_program_arguments_only_foreign_symbols_are_refused(tmp_path: Path):
    with pytest.raises(SignatureDriftError, match="x_call1_d1"):
        _emit(tmp_path, ("m", "x_call1_d1"), dace_arglist=())
    src = _emit(tmp_path, ("m", "n"), dace_arglist=()).read_text()
    assert _init_call_args(src) == ["m", "n"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
