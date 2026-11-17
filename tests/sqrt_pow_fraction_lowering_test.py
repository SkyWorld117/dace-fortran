# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``SQRT`` must never reach the generated code as ``pow(x, 1/2)``.

The bridge lowers ``SQRT(x)`` to ``math.sqrt`` -> ``sqrt(x)`` and ``x ** 0.5d0`` to
``pow(x, 0.5)``; both are correct and are pinned here (the WRF nest's
``rr = NINT(SQRT(FLOAT(nf)))`` is exactly this shape, and a ``pow(x, 0)`` there
collapses ``rr`` to 1 and wrecks every interpolation offset).

The remaining hole is NOT in this frontend: once the expression becomes a SYMBOLIC
value -- a map range or a memlet subset, e.g. ``DO i = 1, NINT(SQRT(FLOAT(n)))``
after ``loop-to-map`` -- DaCe's own sympy printer renders it.  ``sympy.Half``
prints as the string ``1/2``, ``float('1/2')`` raises, so
``sympy.SymPyPrinter._print_Pow`` (``dace/symbolic.py``, the ``float(exponent) ==
0.5`` test -- it tests the PRINTED string, not the sympy exponent) misses the
``dace::math::sqrt`` fast path and emits ``dace::math::pow(x, 1/2)``, i.e. integer
division in C++: ``pow(x, 0) == 1``.

``test_sqrt_in_a_loop_bound_lowers_to_sqrt`` therefore xfails until that printer
is fixed (strict: it will XPASS the moment it is, flagging this marker for
removal).  Minimal reproducer for the DaCe fix::

    from dace.symbolic import pystr_to_symbolic, symstr
    symstr(pystr_to_symbolic('sqrt(x)'), cpp_mode=True)
    # -> '(dace::math::pow(x, 1/2))'   (wrong; should be '(dace::math::sqrt(x))')
"""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

_DACE_PRINTER_BUG = "DaCe dace/symbolic.py _print_Pow renders sympy.Half as pow(x, 1/2)"

_SRC_RR = """
subroutine nearest_root(n, rr)
  integer, intent(in) :: n
  integer, intent(out) :: rr
  rr = nint(sqrt(float(n)))
end subroutine nearest_root

subroutine half_power(x, y)
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = x ** 0.5d0
end subroutine half_power
"""

_SRC_BOUND = """
subroutine scaled(n, a, out)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(out) :: out(n)
  integer :: i
  do i = 1, nint(sqrt(float(n)))
    out(i) = 2.0d0 * a(i)
  end do
end subroutine scaled
"""


def _tasklet_codes(sdfg):
    return [
        str(node.code.as_string)
        for state in sdfg.all_states()
        for node in state.nodes()
        if getattr(node, "code", None) is not None
    ]


def test_bridge_emits_a_real_sqrt(tmp_path):
    """``NINT(SQRT(...))`` keeps a typed ``sqrt`` and ``x**0.5d0`` a REAL ``0.5``.

    Guards the frontend half of the WRF nest's ``rr`` bug: no ``pow(x, ...)`` may
    appear for a SQRT, and the exponent of a square-root power must be a REAL
    literal (``0.5``), never an integer ``1/2``.
    """
    sdfg = build_sdfg(_SRC_RR, tmp_path / "sdfg", name="sqrt_rr", entry="nearest_root").build()
    codes = _tasklet_codes(sdfg)
    assert any("sqrt(" in code for code in codes), codes
    assert not any("pow(" in code for code in codes), codes

    rr = np.zeros(1, dtype=np.int32)
    for n in range(1, 37):
        sdfg(n=np.int32(n), rr=rr)
        assert rr[0] == int(np.round(np.sqrt(n))), (n, rr[0])

    sdfg2 = build_sdfg(_SRC_RR, tmp_path / "sdfg2", name="sqrt_half", entry="half_power").build()
    (code,) = [c for c in _tasklet_codes(sdfg2) if "pow(" in c or "sqrt(" in c]
    assert "pow(" in code and "0.5" in code and "1/2" not in code, code
    y = np.zeros(1)
    sdfg2(x=np.float64(9.0), y=y)
    assert y[0] == 3.0


@pytest.mark.xfail(strict=True, reason=_DACE_PRINTER_BUG)
def test_sqrt_in_a_loop_bound_lowers_to_sqrt(tmp_path):
    """A SQRT used as a loop bound must survive the symbolic round trip.

    ``DO i = 1, NINT(SQRT(FLOAT(n)))`` becomes a map range; DaCe renders the
    symbolic ``sqrt`` as ``dace::math::pow(x, 1/2)`` == ``pow(x, 0)`` == 1, so the
    loop runs once instead of ``NINT(SQRT(n))`` times.  Compares against the
    direct numpy computation (and the gfortran-identical frontend lowering).
    """
    sdfg = build_sdfg(_SRC_BOUND, tmp_path / "sdfg", name="sqrt_bound", entry="scaled").build()
    from dace_fortran.pipelines import optimize

    optimized = optimize(sdfg)
    optimized.compile()
    for cpp in Path(optimized.build_folder).rglob("*.cpp"):
        text = cpp.read_text()
        if "sqrt" in text or "pow(" in text:
            assert "1/2" not in text, f"{cpp.name} renders a symbolic sqrt as pow(x, 1/2):\n{text}"

    n = 16
    a = np.arange(1.0, n + 1.0)
    out = np.zeros(n)
    optimized(n=np.int32(n), a=a, out=out)
    expected = np.zeros(n)
    expected[: int(np.round(np.sqrt(n)))] = 2.0 * a[: int(np.round(np.sqrt(n)))]
    np.testing.assert_array_equal(out, expected)
