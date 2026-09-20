# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The union nest: the one transformation that decides whether a generated kernel coalesces."""
import re

import pytest

from dace_fortran import tu


# The bounds a REAL pair has: the arms of a stencil share their bases and differ by one cell at each
# end, in the same direction.  `("ulb", "chi")`/`("clo", "ure")` -- which these tests used to use --
# is NOT that shape, and the symbolic check in `union_range` now rejects it (see
# `test_unrelated_arms_are_refused_...`), because accepting it is what let a mis-paired arm emit.
UPPER: tu.Bounds = ("clo + 1", "chi")        # the L arm, one cell higher
LOWER: tu.Bounds = ("clo", "chi - 1")        # the R arm


def test_a_pair_becomes_ONE_nest_not_two_siblings():
    """The whole point.  Two sibling loops lower to two SIBLING maps, and a map-collapse pass fuses a
    CHAIN -- so the device map covers only the outer loop and the unit-stride dimension ends up serial
    per thread.  Measured: 2.0 ms/launch against 5.4 us for the union form."""
    body = tu.union_arms("c", UPPER, LOWER,
                         ["oL(c, b, a) = f(c, b, a)"], ["oR(c, b, a) = g(c, b, a)"])
    out = tu.nest(["a", "b", "c"], [("alo", "ahi"), ("blo", "bhi"), ("clo", "chi")], body)
    assert out.count("do ") == 3                  # one nest, three axes
    assert out.count("end do") == 3
    assert out.count("if (") == 2                 # the two arms are guards INSIDE that nest


def test_each_arm_is_guarded_by_its_own_bound():
    body = tu.union_arms("c", UPPER, LOWER, ["L"], ["R"])
    assert "if (c >= clo + 1) then" in body
    assert "if (c <= chi - 1) then" in body


def test_swapping_which_end_the_left_arm_occupies_swaps_the_guards():
    """Getting this backwards is SILENT -- both guards compile and both arms run, they just write the
    wrong cells -- so it is an explicit flag rather than an inference from argument order."""
    up = tu.union_arms("c", UPPER, LOWER, ["L"], ["R"], left_is_upper=True)
    dn = tu.union_arms("c", UPPER, LOWER, ["L"], ["R"], left_is_upper=False)
    assert "c >= clo + 1" in up and "c <= chi - 1" in up
    assert "c <= chi" in dn and "c >= clo" in dn
    assert up != dn


def test_a_non_overlapping_pair_is_refused_not_emitted():
    """A union nest whose guards do not cover both arms computes the WRONG CELLS silently.  Refusing
    is the only safe outcome."""
    with pytest.raises(tu.ArmsNotFormable):
        tu.union_arms("c", (0, 4), (7, 9), ["L"], ["R"])
    # ... and ADJACENT is fine: [0,4] and [5,9] tile with no gap
    tu.union_arms("c", (0, 4), (5, 9), ["L"], ["R"])


def test_the_union_of_two_overlapping_arms_is_their_span():
    assert tu.union_range((0, 10), (0, 4)) == (0, 10)
    # SYMBOLIC, and the answer is the real span: the mirrored pair [-1..hi]/[lo..+1] covers [lo, hi],
    # NOT `(lo_l, hi_r)` -- that is the INTERSECTION, and an earlier version returned it unchecked.
    assert tu.union_range(("clo + 1", "chi"), ("clo", "chi - 1")) == ("clo", "chi")
    assert tu.union_range(("clo", "chi - 1"), ("clo + 1", "chi")) == ("clo", "chi")


def test_unrelated_arms_are_refused_rather_than_emitted():
    """THE CHECK THAT WAS MISSING, and the reason it is worth having: the symbolic case used to return
    without validating anything, so a caller that paired the WRONG two arms -- which is exactly what a
    non-injective shape key does, crossing the dx pair with the dy pair -- emitted a nest whose guards
    cover the wrong axis, silently.

    Unrelated bounds: nothing to order, but also nothing to RENAME -- and that is checkable.
    """
    with pytest.raises(tu.ArmsNotFormable, match="do not share their bounds' bases"):
        tu.union_arms("c", ("ulb", "chi"), ("clo", "ure"), ["L"], ["R"])


def test_a_pair_that_is_not_a_one_cell_mirror_is_refused():
    """Same bases, wrong offsets: contiguous, overlapping, and NOT a stencil pair.  One nest over the
    union of their guards would not cover both ranges exactly once."""
    with pytest.raises(tu.ArmsNotFormable, match="one-cell mirror"):
        tu.union_range(("clo + 3", "chi"), ("clo", "chi - 5"))
    with pytest.raises(tu.ArmsNotFormable, match="one-cell mirror"):
        # offsets one apart but in OPPOSITE directions -- the arms overlap in the middle
        tu.union_range(("clo + 1", "chi + 1"), ("clo", "chi - 1"))


def test_module_declares_the_rank_the_caller_gives():
    """A hand-written emitter that assumed rank 3 declared the one 1-D array as 3-D until someone
    special-cased its NAME; the rank is an argument here for exactly that reason."""
    src = tu.module("m", "k",
                    dummies=[("a", 3), ("cc", 1)], scalars=["n"],
                    body="      a(0, 0, 0) = cc(0)",
                    nest_vars=["c"], nest_bounds=[("0", "n")], doc="test")
    assert "a(0:, 0:, 0:)" in src
    assert "cc(0:)" in src
    assert "cc(0:, 0:, 0:)" not in src
    # the nest is present and closed
    assert src.count("do c = 0, n") == 1
    assert re.search(r"end do\s*\n\s*end subroutine", src)
