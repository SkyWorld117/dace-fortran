# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""`prune_unused_arguments`: drop the arguments the graph never reads.

The bridge materialises a module's closure, so a Fortran translation unit's module variables all
arrive as non-transient arrays and therefore as arguments.  Measured on MFC's sweeps kernel, **305 of
378** are never accessed, and pruning takes it to **74**.

The tests that matter are the ways this went wrong while being written: pruning something the kernel
READS (a silently wrong kernel), and leaving the signature uncomputable (`arglist()` resolves free
symbols through `sdfg.symbols`, and removing an array can make a name resurface as one).
"""
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.prune import prune_unused_arguments, unused_arguments

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

# `used` is read by the kernel; `unused` is a module variable it never touches.  Both are in the
# closure, so both reach the ABI.
SRC = """
module m
  integer, parameter :: n = 8
  real(kind=8), dimension(8) :: used
  real(kind=8), dimension(8) :: unused
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    out = used(1)
  end subroutine
end module
"""


def _build(tmp_path):
    return build_sdfg(SRC, tmp_path / "sdfg", name="driver", entry="m::driver").build()


def test_the_signature_survives_pruning(tmp_path):
    """THE INVARIANT, and the one that was violated first.  `arglist()` resolves free symbols
    through `sdfg.symbols`; removing an array whose name is also used as a symbol makes that name
    resurface and the lookup raise `KeyError`.  A pass that can leave an SDFG whose own signature
    cannot be computed is worse than no pass."""
    sdfg = _build(tmp_path)
    prune_unused_arguments(sdfg)
    sdfg.arglist()  # must not raise
    assert len(sdfg.arglist()) > 0


def test_the_question_and_the_action_agree(tmp_path):
    """`unused_arguments` reports candidates and `prune_unused_arguments` removes a subset of them --
    never more.  A reporter and an actor that can disagree is how a "0 unused" line means nothing."""
    sdfg = _build(tmp_path)
    reported = set(unused_arguments(sdfg))
    before = len(sdfg.arglist())
    removed = prune_unused_arguments(sdfg)
    assert 0 <= removed <= len(reported), f"removed {removed} of {len(reported)} reported"
    assert len(sdfg.arglist()) == before - removed


def test_a_read_argument_is_never_pruned(tmp_path):
    """The one that matters: pruning what the kernel reads produces a wrong kernel, silently."""
    sdfg = _build(tmp_path)
    assert "used" in [str(a) for a in sdfg.arglist()], "the read array is not even an argument"
    prune_unused_arguments(sdfg)
    assert "used" in [str(a) for a in sdfg.arglist()], (
        "an array the kernel READS was pruned -- a silently wrong kernel, not a tidier signature")
    assert "used" in sdfg.arrays


def test_an_unread_argument_is_pruned(tmp_path):
    sdfg = _build(tmp_path)
    assert "unused" in [str(a) for a in sdfg.arglist()]
    prune_unused_arguments(sdfg)
    assert "unused" not in [str(a) for a in sdfg.arglist()]
    assert "unused" not in sdfg.arrays


def test_pruning_leaves_the_arithmetic_alone(tmp_path):
    """No access node is touched, so the computation cannot change -- the property that makes this
    safe without a differential gate.  Asserted structurally."""
    sdfg = _build(tmp_path)

    def accesses(g):
        return {(st.label, nd.data) for st in g.all_states() for nd in st.data_nodes()}

    before = accesses(sdfg)
    prune_unused_arguments(sdfg)
    assert accesses(sdfg) == before
