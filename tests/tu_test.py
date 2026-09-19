# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The union nest: the one transformation that decides whether a generated kernel coalesces."""
import re

import pytest

from dace_fortran import tu


def test_a_pair_becomes_ONE_nest_not_two_siblings():
    """The whole point.  Two sibling loops lower to two SIBLING maps, and a map-collapse pass fuses a
    CHAIN -- so the device map covers only the outer loop and the unit-stride dimension ends up serial
    per thread.  Measured: 2.0 ms/launch against 5.4 us for the union form."""
    body = tu.union_arms("c", ("ulb", "chi"), ("clo", "ure"),
                         ["oL(c, b, a) = f(c, b, a)"], ["oR(c, b, a) = g(c, b, a)"])
    out = tu.nest(["a", "b", "c"], [("alo", "ahi"), ("blo", "bhi"), ("clo", "chi")], body)
    assert out.count("do ") == 3                  # one nest, three axes
    assert out.count("end do") == 3
    assert out.count("if (") == 2                 # the two arms are guards INSIDE that nest


def test_each_arm_is_guarded_by_its_own_bound():
    body = tu.union_arms("c", ("ulb", "chi"), ("clo", "ure"),
                         ["L"], ["R"])
    assert "if (c >= ulb) then" in body
    assert "if (c <= ure) then" in body


def test_swapping_which_end_the_left_arm_occupies_swaps_the_guards():
    """Getting this backwards is SILENT -- both guards compile and both arms run, they just write the
    wrong cells -- so it is an explicit flag rather than an inference from argument order."""
    up = tu.union_arms("c", ("ulb", "chi"), ("clo", "ure"), ["L"], ["R"], left_is_upper=True)
    dn = tu.union_arms("c", ("ulb", "chi"), ("clo", "ure"), ["L"], ["R"], left_is_upper=False)
    assert "c >= ulb" in up and "c <= ure" in up
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
    assert tu.union_range(("ulb", "chi"), ("clo", "ure")) == ("ulb", "ure")


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
