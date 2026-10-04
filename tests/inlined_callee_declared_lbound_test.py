"""An inlined callee that declares its OWN dummy lower bound must rebase by that bound, not by 1.

``buildDesignateIndexExpr``'s assumed-shape-alias rebase computed
``adjust = lb_outer - 1`` -- the literal ``1`` being the Fortran default
assumed-shape lower bound.  That is right for a bare ``b(:)`` callee dummy
(covered by ``assumed_shape_shift_alias_offset_test``), but wrong the moment the
callee dummy carries an explicit lower bound: ``dimension(0:1000)`` -- exactly
the stability-table dummies the WRF surface-layer scheme interpolates through
inlined helper functions.  Measured there: ``tab(nzol)`` inside the helper
lowered to ``tab[nzol - 1]``, so ``nzol == 0`` read ``tab[-1]`` (a dace
``InvalidSDFGEdgeError``, and -- were validation skipped -- the wrong element).

The fix uses the callee declare's own lower bound: ``adjust = lb_outer -
lb_callee``, which collapses to the previous ``lb_outer - 1`` for the default
and to ``lb_outer - 0`` for ``(0:)``/``(0:N)``.

This pins the explicit-bound case: ``tab(3)`` on a ``(0:4)`` dummy must read the
element at Fortran index 3 (the 4th), regardless of which bound the caller's
array happens to declare.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")


def _src(callee_lb: int) -> str:
    # Internal (contained) helper, matching the WRF slices' shape: a function
    # nested in the routine, taking a table whose dummy declares an explicit
    # non-default lower bound, and indexing it by a literal.  The caller and
    # callee dummies span the SAME 5 elements, so the accessed element is the
    # 4th (storage index 3, value 300.0) for every ``callee_lb``.
    return f"""
subroutine outer(tab, out)
  real(8), intent(in)  :: tab(0:4)
  real(8), intent(out) :: out
  out = pick(tab)
contains
  real(8) function pick(tab)
    real(8), intent(in) :: tab({callee_lb}:{callee_lb + 4})
    pick = tab({callee_lb + 3})
  end function pick
end subroutine outer
"""


@pytest.mark.parametrize("callee_lb", [0, -2, 1])
def test_inlined_callee_declared_lbound(tmp_path: Path, callee_lb: int):
    """The helper reads the caller's 4th element (300.0) for every declared bound.

    The callee view's element ``k`` maps to storage index ``k - callee_lb``; the
    fixture accesses ``tab(callee_lb + 3)``, i.e. storage index 3, in all cases.
    A ``-1`` shift makes the reported element the 3rd (200.0) instead.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_src(callee_lb), sdfg_dir, name="outer", entry="outer").build()

    # Fortran indices 0..4 -> values 0,100,200,300,400.
    tab = np.asfortranarray(np.array([0.0, 100.0, 200.0, 300.0, 400.0], dtype=np.float64))
    out = np.zeros(1, dtype=np.float64, order="F")
    sdfg(tab=tab, out=out)
    assert out[0] == 300.0, (
        f"callee_lb={callee_lb}: the inlined helper must read storage index 3 (300.0); got "
        f"{out[0]}.  A 200.0 read means the callee's declared lower bound was ignored and the "
        f"literal 1 assumed, shifting the access one element low.")


def test_inlined_callee_declared_lbound_is_not_off_by_one(tmp_path: Path):
    """The literal-index memlet for the helper must equal the inline memlet.

    Direct structural guard: the same ``tab(3)`` access written inline and through
    the inlined helper must yield the same memlet subset.  Before the fix the
    helper subset was ``2`` (``3 - 1``) while inline was ``3``.
    """
    from dace.sdfg import nodes as _n

    def _tab_subsets(sdfg):
        found = set()

        def walk(s):
            for st in s.all_states():
                for e in st.edges():
                    d = e.data
                    if d is None:
                        continue
                    src = str(getattr(e.src, "data", ""))
                    dst = str(getattr(e.dst, "data", ""))
                    if src == "tab" or dst == "tab":
                        found.add(repr(d.src_subset if src == "tab" else d.dst_subset))
                for nd in [x for x in st.nodes() if isinstance(x, _n.NestedSDFG)]:
                    walk(nd.sdfg)

        walk(sdfg)
        return found

    inline = """
subroutine outer(tab, out)
  real(8), intent(in)  :: tab(0:4)
  real(8), intent(out) :: out
  out = tab(3)
end subroutine outer
"""
    d1 = tmp_path / "inline"
    d1.mkdir()
    s_inline = build_sdfg(inline, d1, name="outer", entry="outer").build()
    d2 = tmp_path / "helper"
    d2.mkdir()
    s_helper = build_sdfg(_src(0), d2, name="outer", entry="outer").build()
    assert "Range (3)" in _tab_subsets(s_inline)
    assert "Range (3)" in _tab_subsets(s_helper), (
        f"helper lowered tab(3) to {sorted(_tab_subsets(s_helper))}, not Range(3)")
