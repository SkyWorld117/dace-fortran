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


# ---------------------------------------------------------------------------------------------------
# Analysing a shape's nests.  These came from a porting project's emitter, where each one had already
# cost a measured defect; they are here because none of them is knowledge about THAT project.
# ---------------------------------------------------------------------------------------------------


def test_the_difference_axis_is_looked_up_by_DIM_not_hardcoded_innermost():
    """The house convention is outermost = LAST storage dimension, so position `dim` is `loops[-1-dim]`.

    Getting this wrong guards the difference on the wrong loop for 4 of 6 shapes and NOTHING RAISES:
    the bounds are read off the wrong axis, so the union is taken over the wrong axis and the emitted
    nest writes the wrong cells.  An emitter with the bug reported "6 shape(s) emitted, 0 blocked" the
    whole time."""
    assert tu.axis_index(tu.Shape("AVG", 0, ("l", "k", "j"))) == 2      # innermost
    assert tu.axis_letter(tu.Shape("AVG", 0, ("l", "k", "j"))) == "j"
    assert tu.axis_index(tu.Shape("GRAD", 2, ("j", "l", "k"))) == 0     # outermost
    assert tu.axis_letter(tu.Shape("GRAD", 2, ("j", "l", "k"))) == "j"


def test_an_undetermined_dim_is_treated_as_zero_rather_than_refused():
    assert tu.axis_index(tu.Shape("AVG", None, ("l", "k", "j"))) == 2


def test_a_shape_with_no_loops_RAISES_instead_of_returning_minus_one():
    """`-1` is a valid Python index, so `bounds[axis_index(shape)]` read the LAST bound silently.  That
    is the same silent-wrong-answer shape as the hardcoded innermost loop, one step further out."""
    empty = tu.Shape("AVG", 0, ())
    with pytest.raises(ValueError, match="no loops"):
        tu.axis_index(empty)
    assert tu.axis_letter(empty) is None             # the letter is still just an absent answer


def test_nest_bounds_sees_a_loop_variable_longer_than_one_letter():
    """`[a-z]` was too narrow and it was found four separate times in one emitter: a nest spelling its
    loops `k_loop`/`l_loop` yielded NO loops, and the caller then emitted a unit with no `do` at all
    whose body referenced `j`/`k`/`l` undeclared."""
    body = "do i = 0, n\n  do k_loop = a + 1, b - 1\n    x = 1\n  end do\nend do"
    assert tu.nest_bounds(body) == [("i", "0", "n"), ("k_loop", "a + 1", "b - 1")]


def test_nest_bounds_returns_nothing_for_a_body_with_no_do():
    assert tu.nest_bounds("x = y + 1") == []


def test_two_units_differing_only_in_NAME_and_LINE_are_the_same_shape():
    """One library per SHAPE serves every loop of that shape, so the check that two pairs of a shape
    agree must not be defeated by the rename it is checking for."""
    a = tu.module("avg_d0_l189_mod", "avg_d0_l189_mod_kernel", dummies=[("a1", 3)], scalars=["n"],
                  body="do c = 1, n\n  a1(c, 1, 1) = 1\nend do",
                  nest_vars=["c"], nest_bounds=[("1", "n")], doc="extracted from 189/204")
    b = tu.module("visc_avg_d0_mod", "visc_avg_d0_mod_kernel", dummies=[("a1", 3)], scalars=["n"],
                  body="do c = 1, n\n  a1(c, 1, 1) = 1\nend do",
                  nest_vars=["c"], nest_bounds=[("1", "n")], doc="extracted from 189/204")
    assert tu.shape_signature(a) == tu.shape_signature(b)
    assert a != b                                     # they really do differ as text


def test_a_shape_signature_notices_a_changed_assignment():
    """The falsifying control: if only names were normalised, this would compare equal too."""
    a = tu.module("u_mod", "u_mod_kernel", dummies=[("a1", 3)], scalars=["n"],
                  body="do c = 1, n\n  a1(c, 1, 1) = 1\nend do",
                  nest_vars=["c"], nest_bounds=[("1", "n")])
    b = tu.module("u_mod", "u_mod_kernel", dummies=[("a1", 3)], scalars=["n"],
                  body="do c = 1, n\n  a1(c, 1, 1) = 2\nend do",
                  nest_vars=["c"], nest_bounds=[("1", "n")])
    assert tu.shape_signature(a) != tu.shape_signature(b)


def test_a_shape_signature_ignores_the_wrap_point_the_renamed_length_moves():
    """`module` breaks the argument list at 100 columns, so a longer NAME moves the `&`.  MEASURED
    before this normalisation: two identical units reported a mismatch at
    `& b_lo, b_hi, c_lo, ...` against `& b_hi, c_lo, c_hi, ...`."""
    long_args = [f"a{i}" for i in range(14)]
    long = tu.module("a_very_long_unit_name_here_mod", "k", dummies=[("a1", 3)],
                     scalars=long_args, body="do c = 1, n\n  a1(c, 1, 1) = 1\nend do",
                     nest_vars=["c"], nest_bounds=[("1", "n")])
    short = tu.module("s_mod", "k", dummies=[("a1", 3)], scalars=long_args,
                      body="do c = 1, n\n  a1(c, 1, 1) = 1\nend do",
                      nest_vars=["c"], nest_bounds=[("1", "n")])
    assert long != short                              # the wrap really did move
    assert tu.shape_signature(long) == tu.shape_signature(short)


def test_aliases_binds_the_three_assignments_a_source_writes_on_ONE_line():
    """A line-anchored regex matches none of `is1 = ix; is2 = iy; is3 = iz`, so the map comes out empty
    and a pair that DOES run over the same cells is refused for a reason that is no longer true.

    The result is `alias_map`'s CANONICAL map, so BOTH sides of each assignment appear, each resolved
    to its representative.  Asserting only the three left-hand names would pass on a map that had lost
    `iy`'s own entry -- which is the side the bounds actually spell."""
    m = tu.aliases_of("  is1_viscous = ix; is2_viscous = iy; is3_viscous = iz\n  x = 1\n  y = x + 1\n")
    assert m["is1_viscous"] == m["ix"] == "ix"
    assert m["is2_viscous"] == m["iy"] == "iy"
    assert m["is3_viscous"] == m["iz"] == "iz"


def test_an_expression_on_the_right_is_not_an_alias():
    """An alias that is not a pure rename is not an alias, and neither is a self-assignment."""
    assert tu.aliases_of("a = b + 1\nc = c\nd = -e\n") == {}


def test_aliases_is_empty_for_a_routine_with_no_pure_renames():
    """The falsifying control for the test above: the map comes from a SCAN, so an empty answer has to
    be reachable rather than being what a broken scan always returns."""
    assert tu.aliases_of("subroutine s\n  x = 1\nend subroutine\n") == {}


# ---------------------------------------------------------------------------------------------------
# The mechanical half: turning the source's own accesses into dummies, and reading a copy's
# subscripts.  Both take the ACCESS IDIOM as a parameter -- it is a property of the code being
# ported, and assuming one project's spelling is the one thing this module must not do.
# ---------------------------------------------------------------------------------------------------


def test_dummies_binds_every_component_access_and_keeps_the_subscripts():
    dummies, subst = tu.dummies(["oa(c, b, a) = qa%vf(i)%sf(c, b, a)"])
    assert [d for d, _, _ in dummies] == ["a1", "a2"]
    assert subst["qa%vf(i)"] == "a1" and subst["oa"] == "a2"
    # the SUBSCRIPTS stay verbatim -- rewriting them is where arithmetic gets composed by accident
    assert dummies[0][1] == 3 and dummies[1][1] == 3


def test_dummies_also_binds_a_PLAIN_module_array():
    """The component idiom is not the only way Fortran reads memory.  MEASURED: a shape that divides by
    a coordinate spacing carried as a module array reached the compiler undeclared -- `Function 'y_cc'
    at (1) has no IMPLICIT type` -- because the component pattern never saw it."""
    dummies, subst = tu.dummies(["oa(c, b, a) = qa%sf(c, b, a) / y_cc(b)"])
    assert "y_cc" in subst and dummies[2][1] == 1


def test_dummies_does_not_bind_the_INNER_part_of_a_component_chain():
    """`q%vf(i)%sf(...)` contains `q%vf(i)`, which the plain-array pattern matches on its own; binding
    it too gives the SAME array two dummies of different ranks, and the two units then disagree about
    which is which (`Rank mismatch in argument 'a5'`)."""
    _, subst = tu.dummies(["oa(c, b, a) = q%vf(i)%sf(c, b, a)"])
    assert set(subst) == {"q%vf(i)", "oa"}


def test_dummies_skips_intrinsics_and_keywords():
    _, subst = tu.dummies(["oa(c, b, a) = max(size(qa%sf(c, b, a), 1), 2)"])
    assert "max" not in subst and "size" not in subst


def test_the_access_idiom_is_a_PARAMETER():
    """Asserted with a spelling that is NOT the default: if the idiom were baked in, this would find
    nothing and return an empty binding -- which is exactly the silent failure mode."""
    dummies, subst = tu.dummies(["oa(c) = qa%field(c)"], access="%field")
    assert subst["qa"] == "a1" and len(dummies) == 2
    # ... and with the default, the COMPONENT access is missed while the PLAIN one is still found --
    # which is the two passes behaving independently, and the reason this shows up as a missing
    # substitution rather than as an empty result.
    _, default = tu.dummies(["oa(c) = qa%field(c)"])
    assert "qa" not in default and default["oa"] == "a1"


def test_subscripts_splits_on_the_MATCHING_paren_not_the_first_comma():
    """A naive `split(',')` cuts `m - (j - 1)` in half, and the cut expression parses as a DIFFERENT
    affine function rather than failing."""
    s = "qa%sf(m - (j - 1), k, l) = qb%sf(m - (j - 1), k, l)"
    assert tu.subscripts(s, 1) == ("m - (j - 1)", "m - (j - 1)")
    assert tu.subscripts(s, 2) == ("k", "k")
    assert tu.subscripts(s, 3) == ("l", "l")


def test_subscripts_refuses_what_it_cannot_read_rather_than_returning_a_fragment():
    with pytest.raises(tu.NotAnAccess, match="not an assignment"):
        tu.subscripts("qa%sf(1, 2, 3)", 1)
    with pytest.raises(tu.NotAnAccess, match="no `%sf\\(` access"):
        tu.subscripts("oa(1, 2, 3) = qb(1, 2, 3)", 1)
    with pytest.raises(tu.NotAnAccess, match="wanted 4"):
        tu.subscripts("qa%sf(1, 2, 3) = qb%sf(1, 2, 3)", 4)
