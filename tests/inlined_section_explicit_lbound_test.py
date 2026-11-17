# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A section actual bound to an explicit-shape dummy keeps the CALLEE's declared lower bound.

``hlfir-fold-copy-in-out`` reparents the inlined callee's accesses from the contiguous
copy-in temp onto the parent section with ``parent_idx = alias_idx + (lo - 1)`` -- the
``1`` being the default lower bound of the section VIEW.  That is only right when the
callee's dummy is default-bound.  An explicit-shape dummy that repeats the caller's
bounds (``xf(ims:ime, jms:jme, nf)`` with ``ims = cims`` -- the WRF nest's
``nest_sint(XF(ims:ime,jms:jme,NF))`` reached with ``psca(k,cims:cime,cjms:cjme,:)``)
indexes the view as ``alias_idx - ims``, so the fold shifted every access of the sweep
by ``ims - 1``: the generated CUDA read/wrote ``(II-1)`` / ``(JJ-1)`` where the gather
used ``(II-cims)`` / ``(JJ-cjms)``, and at ``cims=cjms=13`` the sweep wrote 12 elements
past the gather slab (out of the SDFG allocation).

``d8fb6866`` rebased the ASSUMED-SHAPE alias path (``buildDesignateIndexExpr``) through
symbolic caller bounds; this is the OTHER path -- the section reparent inside the
copy-in fold -- which bypasses that rebase entirely and hardcoded ``lo - 1``.

The baseline is the same arithmetic evaluated directly in numpy, per the e2e policy:
never hand-tuned literal expectations.
"""

import re

import numpy as np
import pytest

from tests._util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

_SRC = """
module m_sec_lb
contains
  ! Explicit-shape dummy repeating the CALLER's bounds: the callee frame is
  ! `ims..ime`, not `1..ime-ims+1`.
  subroutine sweep3(xf, ims, ime, jms, jme, nf)
    integer, intent(in) :: ims, ime, jms, jme, nf
    real(8), intent(inout) :: xf(ims:ime, jms:jme, nf)
    integer :: i, j
    do j = jms, jme
      do i = ims, ime
        xf(i, j, 1) = 2.0d0 * xf(i, j, 1) + dble(i) - dble(j)
      end do
    end do
  end subroutine sweep3

  ! 2-D sibling: a rank-2 explicit-shape dummy fed a section of a rank-3 workspace.
  subroutine sweep2(xy, ims, ime, jms, jme)
    integer, intent(in) :: ims, ime, jms, jme
    real(8), intent(inout) :: xy(ims:ime, jms:jme)
    integer :: i, j
    do j = jms, jme
      do i = ims, ime
        xy(i, j) = xy(i, j) + 10.0d0 * dble(i + 2 * j)
      end do
    end do
  end subroutine sweep2

  subroutine driver(out3, out2, cims, cime, cjms, cjme, nf)
    integer, intent(in) :: cims, cime, cjms, cjme, nf
    real(8), intent(out) :: out3(cims:cime, cjms:cjme)
    real(8), intent(out) :: out2(cims:cime, cjms:cjme)
    ! Local workspaces whose 2nd/3rd dims carry the caller's (symbolic, non-1)
    ! lower bounds -- the WRF nest's k-first gather slab.
    real(8) :: w3(1:1, cims:cime, cjms:cjme, nf)
    real(8) :: w2(1:nf, cims:cime, cjms:cjme)
    integer :: i, j
    do j = cjms, cjme
      do i = cims, cime
        w3(1, i, j, 1) = dble(i * 1000 + j)
        w2(1, i, j) = dble(i - 3 * j)
      end do
    end do
    ! Section actuals: scalar first dim + triplets on the bounded dims.
    call sweep3(w3(1, cims:cime, cjms:cjme, 1:nf), cims, cime, cjms, cjme, nf)
    call sweep2(w2(1, cims:cime, cjms:cjme), cims, cime, cjms, cjme)
    do j = cjms, cjme
      do i = cims, cime
        out3(i, j) = w3(1, i, j, 1)
        out2(i, j) = w2(1, i, j)
      end do
    end do
  end subroutine driver
end module m_sec_lb
"""


def _run(sdfg, cims: int, cime: int, cjms: int, cjme: int):
    ni, nj = cime - cims + 1, cjme - cjms + 1
    out3 = np.full((ni, nj), np.nan, order="F")
    out2 = np.full((ni, nj), np.nan, order="F")
    scalars = dict(cime=cime, cjme=cjme, nf=1)
    sdfg(
        out3=out3,
        out2=out2,
        cims=np.int64(cims),
        cjms=np.int64(cjms),
        **{k: np.int32(v) for k, v in scalars.items()},
    )
    return out3, out2


def _expected(cims: int, cime: int, cjms: int, cjme: int):
    ii, jj = np.meshgrid(
        np.arange(cims, cime + 1, dtype=np.float64),
        np.arange(cjms, cjme + 1, dtype=np.float64),
        indexing="ij",
    )
    w3 = 2.0 * (ii * 1000.0 + jj) + ii - jj
    w2 = (ii - 3.0 * jj) + 10.0 * (ii + 2.0 * jj)
    return w3, w2


@pytest.mark.parametrize(
    "cims, cime, cjms, cjme",
    [
        (13, 41, 13, 41),  # the nest's d02 shape: non-1, cims = cjms = 13
        (1, 9, 1, 7),  # 1-based caller: the fold's original assumption
        (-4, 6, 2, 11),  # mixed/negative bounds: no sign-dependent shortcut
    ],
)
def test_section_actual_keeps_callee_explicit_lower_bound(tmp_path, cims, cime, cjms, cjme):
    """The inlined sweeps must touch the same elements the caller's gather wrote.

    Before the fix the ``w3``/``w2`` accesses emitted ``II - 1`` (the view's default
    1-based frame) instead of ``II - cims``; at ``cims=13`` the 3-D sweep wrote 12
    elements past the allocation (heap corruption) and every value read back was the
    untouched gather plane.
    """
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="m_sec_lb_driver", entry="m_sec_lb::driver").build()
    sdfg.validate()
    out3, out2 = _run(sdfg, cims, cime, cjms, cjme)
    exp3, exp2 = _expected(cims, cime, cjms, cjme)
    np.testing.assert_array_equal(out3, exp3)
    np.testing.assert_array_equal(out2, exp2)


def _workspace_subsets(sdfg, arr: str):
    """Normalized memlet subsets of every edge touching ``arr`` (loop iterators erased)."""
    patterns = set()
    for state in sdfg.all_states():
        for edge in state.edges():
            memlet = edge.data
            if memlet is None:
                continue
            src, dst = str(getattr(edge.src, "data", "")), str(getattr(edge.dst, "data", ""))
            if arr not in (src, dst):
                continue
            subset = memlet.src_subset if src == arr else memlet.dst_subset
            patterns.add(re.sub(r"_loop_it_\d+", "IT", str(subset)))
    return patterns


def test_inlined_section_access_matches_the_caller_gather(tmp_path):
    """Structural guard: the gather and the inlined callee index the workspace identically.

    Both loops touch ``w3``/``w2`` with the same expression, so after normalizing the
    loop iterators each workspace has exactly ONE subset -- ``(0, IT - cims, IT - cjms,
    0)``.  The pre-fix fold emitted ``(0, IT - 1, IT - 1, 0)`` for the callee side, i.e.
    a second pattern shifted by ``cims - 1`` (out of bounds at ``cims = 13``).
    """
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="m_sec_lb_struct", entry="m_sec_lb::driver").build()

    w3 = _workspace_subsets(sdfg, "w3")
    w2 = _workspace_subsets(sdfg, "w2")
    assert w3 == {"0, IT - cims, IT - cjms, 0"}, w3
    assert w2 == {"0, IT - cims, IT - cjms"}, w2
