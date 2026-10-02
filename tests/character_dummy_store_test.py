"""Regression: a ``CHARACTER`` dummy assigned a string literal must be DROPPED, not misbuilt.

The WRF physics idiom ``character(len=*),intent(out):: errmsg; errmsg = '...OK'`` appears in
~36 physics files.  Flang lowers it to an ``hlfir.assign`` whose destination is a
``!fir.boxchar<1>`` and whose RHS is a ``!fir.ref<!fir.char<1,N>>`` pointing at the
literal-pool global ``_QQclX<hex>``.  The bridge does not model character data (see
``passes/StripCharacterRuntime.cpp``), so before the fix the assign survived to
``buildAssignNode``, which read the literal global as a scalar operand and either:

* raised ``KeyError '_QQclX6B204F4B'`` -- the access node's container was never registered
  as an SDFG array (the MINIMAL repro), or
* emitted a tasklet ``_out = 204F4B`` -- the mangled name's byte tail, rejected by DaCe's
  ``ast.parse`` with ``SyntaxError: invalid decimal literal`` (the full ``mp_wsm6_run``).

``hlfir-strip-character-runtime`` now erases character-valued ``hlfir.assign`` stores, so
the error-message plumbing is dropped and the numerical contract is untouched.

FALSIFIABILITY: revert the character-assign walk in ``StripCharacterRuntime.cpp`` and
``test_character_dummy_string_literal_store_builds`` fails with ``KeyError '_QQcl...'``
while ``test_character_store_is_absent_from_the_sdfg`` would find the leaked global.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

# The minimal P-b repro: a numerical subroutine that also writes an error message.
_MINIMAL = """
subroutine k(a, errmsg, errflg)
  implicit none
  real, intent(inout) :: a
  character(len=*), intent(out) :: errmsg
  integer, intent(out) :: errflg
  a = a + 1.0
  errmsg = 'k OK'
  errflg = 0
end subroutine k
"""

# A character-to-character store into a local buffer, plus a literal store, to pin the rule
# as "any character-valued destination" rather than only the literal shape.
_CHAR_VAR_STORE = """
subroutine kk(n, a, errmsg)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  character(len=*), intent(out) :: errmsg
  character(len=32) :: buf
  integer :: i
  do i = 1, n
    a(i) = a(i) * 2.0
  end do
  buf = 'scratch'
  errmsg = buf
end subroutine kk
"""


def _build(source: str, tmp_path: Path, name: str):
    d = tmp_path / "sdfg"
    d.mkdir(parents=True, exist_ok=True)
    return build_sdfg(source, d, name=name, entry=None).build()


def test_character_dummy_string_literal_store_builds(tmp_path: Path):
    """The minimal mp_wsm6-shaped repro builds instead of raising ``KeyError '_QQclX...'``."""
    sdfg = _build(_MINIMAL, tmp_path, "char_store_min")
    # The numerical surface survives; the error-message plumbing is gone.
    assert "a" in sdfg.arrays
    assert "errflg" in sdfg.arrays


def test_character_store_is_absent_from_the_sdfg(tmp_path: Path):
    """No flang literal-pool container, and no tasklet referencing the mangled ``_QQcl`` global."""
    sdfg = _build(_MINIMAL, tmp_path, "char_store_absent")
    leaked = [str(d) for d in sdfg.arrays if "_QQcl" in str(d)]
    assert not leaked, f"character literal global leaked into the SDFG: {leaked}"
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                code = getattr(node, "code", None)
                assert code is None or "_QQcl" not in code.as_string, \
                    f"tasklet references the character literal global: {code.as_string!r}"


def test_character_variable_store_also_dropped(tmp_path: Path):
    """A character-to-character store (``errmsg = buf``) is dropped by the same rule."""
    sdfg = _build(_CHAR_VAR_STORE, tmp_path, "char_store_var")
    assert "a" in sdfg.arrays
    assert not [d for d in sdfg.arrays if "_QQcl" in str(d)]
