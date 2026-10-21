"""Sympy-collision rename pass -- direct coverage.

The bridge renames Fortran identifiers matching a sympy module-level
``LazyFunction`` attribute (``test``, ``doctest``) to ``program_<name>``; without
it, any interstate-edge expression mentioning the array crashes sympy with
``SympifyError: cannot sympify object of type LazyFunction``.  Identifiers naming
a Python builtin callable (``iter``, ``max``, ...) are renamed for the same reason:
sympy's parser evaluates them to the builtin instead of a symbol.
"""

from pathlib import Path

import numpy as np

from dace_fortran.bindings.frozen_signature import FrozenArgKind
from tests._util import build_sdfg


def _build(src: str, tmp: Path, name: str = "main", entry: str | None = None):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


def test_reserved_name_test_as_local_array(tmp_path: Path):
    """Local array named ``test``  --  the indexed-load case that
    triggered the original ``LazyFunction`` SympifyError."""
    src = """
subroutine main(d)
  double precision d(4)
  integer test(3, 3, 3)
  integer indices(3, 3, 3)
  indices(1, 1, 1) = 2
  indices(1, 1, 2) = 3
  indices(1, 1, 3) = 1
  test(indices(1, 1, 1), indices(1, 1, 2), indices(1, 1, 3)) = 2
  d(test(2, 3, 1)) = 5.5
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.5, 42, 42])


def test_reserved_name_doctest_as_local_array(tmp_path: Path):
    """Same as above but for ``doctest`` (the other sympy ``LazyFunction`` collision)
    -- the rename map must cover the full reserved set."""
    src = """
subroutine main(d)
  double precision d(4)
  integer doctest(3, 3)
  doctest(1, 1) = 2
  doctest(2, 3) = 3
  d(doctest(1, 1)) = 5.5
  d(doctest(2, 3)) = 7.5
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.5, 7.5, 42])


def test_reserved_name_is_renamed_in_sdfg(tmp_path: Path):
    """The SDFG advertises ``program_test`` and the builder records the
    ``test -> program_test`` mapping for the binding emitter."""
    src = """
subroutine main(d)
  double precision d(2)
  integer test(2)
  test(1) = 1
  test(2) = 2
  d(1) = test(1)
  d(2) = test(2)
end subroutine main
"""
    from tests._util import build_sdfg as _bs

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = _bs(src, sdfg_dir, name="main")
    sdfg = builder.build()

    assert "program_test" in sdfg.arrays
    assert "test" not in sdfg.arrays
    assert builder.dace_name_map.get("test") == "program_test"


def test_reserved_name_test_as_dummy_argument(tmp_path: Path):
    """``test`` as a dummy argument -- exercises the SDFG signature + FrozenSignature.
    The SDFG kwarg uses the renamed ``program_test``; FrozenSignature maps it back
    to the original ``test`` for the binding wrapper."""
    src = """
subroutine main(d, test)
  double precision, intent(inout) :: d(2)
  integer, intent(in) :: test(2)
  d(1) = test(1)
  d(2) = test(2)
end subroutine main
"""
    from tests._util import build_sdfg as _bs

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = _bs(src, sdfg_dir, name="main")
    sdfg = builder.build()

    # SDFG-internal name renamed; user-facing Fortran name preserved on the FrozenSignature.
    assert "program_test" in sdfg.arrays
    fs = sdfg._frozen_signature
    test_arg = next((a for a in fs.args if a.fortran_name == "test"), None)
    assert test_arg is not None, f"no 'test' arg in FrozenSignature: {[a.fortran_name for a in fs.args]}"
    assert test_arg.sdfg_name == "program_test"

    # Numerical: pass via the SDFG-internal kwarg, verify data flows through the renamed identifier.
    d = np.zeros(2, dtype=np.float64)
    test_arr = np.array([7, 9], dtype=np.int32)
    sdfg(d=d, program_test=test_arr)
    assert np.allclose(d, [7.0, 9.0])


def test_reserved_name_binding_wrapper_emits(tmp_path: Path):
    """The binding emitter consumes the renamed FrozenSignature and produces a
    wrapper file -- the surface downstream Fortran callers see; without it the
    SDFG side is fixed but the binding side stays broken."""
    from dace_fortran.bindings import emit_bindings
    from dace_fortran.bindings.flatten_plan import FlattenPlan
    from dace_fortran.bindings.fortran_interface import OriginalArg, OriginalInterface
    from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenSignature

    fs = FrozenSignature(
        entry="main",
        mangled="_QPmain",
        args=(
            FrozenArg(
                fortran_name="d",
                sdfg_name="d",
                kind=FrozenArgKind.ARRAY,
                dtype="float64",
                rank=1,
                shape=("2",),
                intent="inout",
            ),
            FrozenArg(
                fortran_name="test",
                sdfg_name="program_test",
                kind=FrozenArgKind.ARRAY,
                dtype="int32",
                rank=1,
                shape=("2",),
                intent="in",
            ),
        ),
        free_symbols=(),
    )
    iface = OriginalInterface(
        entry="main",
        args=(
            OriginalArg(name="d", fortran_type="real(c_double)", rank=1, intent="inout"),
            OriginalArg(name="test", fortran_type="integer(c_int)", rank=1, intent="in"),
        ),
    )
    plan = FlattenPlan(entries=())
    out = tmp_path / "main_bindings.f90"
    emit_bindings(fs, iface, plan, str(out))
    text = out.read_text()
    # Wrapper exposes the original ``test`` to the caller; renamed ``program_test`` is the bind(C) param to the SDFG.
    assert "test" in text
    assert "program_test" in text


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))


def test_python_builtin_name_as_loop_counter_survives_simplify(tmp_path: Path):
    """A counted DO with an EXIT whose counter is called ``iter`` is lifted to a loop that updates the counter on an
    interstate edge; ``simplify`` parses that expression, which fails on the builtin unless the name is renamed."""
    src = """
subroutine main(n, maxit, c, w)
  implicit none
  integer, intent(in) :: n, maxit
  real(8), intent(in) :: c(n)
  real(8), intent(inout) :: w(n)
  real(8) :: x, dx
  integer :: j, iter
  do j = 1, n
    x = c(j) + 1.0d0
    do iter = 1, maxit
      dx = -(x*x - c(j))/(2.0d0*x)
      x = x + dx
      if (abs(dx) < 1.0d-12*x) exit
    end do
    w(j) = x
  end do
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    assert "iter" not in sdfg.arrays and "iter" not in sdfg.symbols
    sdfg.simplify()
    c = np.linspace(1.0, 4.0, 8).copy()
    w = np.zeros(8)
    sdfg(n=8, maxit=50, c=c, w=w)
    np.testing.assert_allclose(w, np.sqrt(c), rtol=1e-12)


def test_python_builtin_name_as_dummy_argument(tmp_path: Path):
    """A dummy called ``input`` is ``program_input`` on the SDFG and stays ``input`` for the binding.  (A name the
    bridge renders as an intrinsic, ``max``, is refused by the bridge itself.)"""
    src = """
subroutine main(d, input)
  double precision, intent(inout) :: d(2)
  integer, intent(in) :: input(2)
  d(1) = input(1)
  d(2) = input(2) + d(2)
end subroutine main
"""
    from tests._util import build_sdfg as _bs

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = _bs(src, sdfg_dir, name="main")
    sdfg = builder.build()
    assert "program_input" in sdfg.arrays and "input" not in sdfg.arrays
    sdfg.simplify()
    d = np.array([0.0, 0.5])
    sdfg(d=d, program_input=np.array([3, 4], dtype=np.int32))
    assert d.tolist() == [3.0, 4.5]
    frozen = sdfg._frozen_signature
    arg = next(a for a in frozen.args if a.sdfg_name == "program_input")
    assert arg.fortran_name == "input"


def test_sympy_function_name_as_local_survives_simplify(tmp_path: Path):
    """A local called ``denom`` that an IF tests: the condition is an interstate-edge expression.

    DaCe's parser reads ``denom`` as a plain SYMBOL in this fork (its eval namespace does not expose
    ``sympy.denom``), so nothing is misread and the local keeps its name -- what matters is that the
    condition computes the same values as Fortran, checked below.  A name the parser DOES misread is
    renamed instead (``test_names_the_symbolic_parser_misreads``)."""
    src = """
subroutine main(n, a, b)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(out) :: b(n)
  real(8) :: denom
  integer :: j
  do j = 1, n
    denom = a(j) - 1.0d0
    if (denom > 0.0d0) then
      b(j) = 1.0d0 / denom
    else
      b(j) = 0.0d0
    end if
  end do
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    sdfg.simplify()
    a = np.array([0.5, 2.0, 3.0, 1.0])
    b = np.zeros(4)
    sdfg(n=4, a=a, b=b)
    np.testing.assert_allclose(b, [0.0, 1.0, 0.5, 0.0])
    # Not renamed here (no collision), so the local is still a transient -- and still not a
    # free symbol leaking out of the SDFG.
    assert "denom" not in sdfg.symbols


def test_names_the_symbolic_parser_misreads():
    """A name the parser resolves to something other than a symbol of that name is a hazard (the builder
    renames it); a plain identifier is not.  WHICH names are hazards is a property of the parser's eval
    namespace, not a fixed list: this fork exposes only ``nan`` and ``iter`` among the ones this test used
    to name, and parses ``denom`` / ``sign`` / ``exp`` / ``field`` / ``test`` as plain symbols."""
    from dace_fortran.builder import misread_by_symbolic_parser

    for name in ("nan", "iter"):
        assert misread_by_symbolic_parser(name), name
    for name in ("q_cons_vf", "i", "n", "x1", "program_denom", "_loop_it_3", "denom", "sign", "exp", "field",
                 "test"):
        assert not misread_by_symbolic_parser(name), name
