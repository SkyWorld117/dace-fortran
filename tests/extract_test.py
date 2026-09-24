# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extraction: the preprocessor pass, the nest slice, and the statement folding."""
from pathlib import Path

import pytest

from dace_fortran import discovery, extract

FIXTURE = Path(__file__).parent / "fixtures" / "nest_shapes.f90"


def test_a_nest_is_returned_with_its_closing_ends():
    """The slice must be self-contained: a nest handed on without its `end do`s is not a nest, and a
    depth counter downstream (the discovery one included) then runs to end-of-file."""
    text = FIXTURE.read_text()
    first = discovery.blocks(text)[0][0]
    got = extract.nest(text, first)
    assert got.count('do ') >= 3
    assert got.rstrip().endswith('end do')
    # and it is exactly the block discovery reported
    assert got.split('\n') == discovery.blocks(text)[0][1]


def test_nest_index_agrees_with_line_lookup():
    """Two ways to the same nest must not disagree -- a caller iterating must get what a caller
    locating by line gets."""
    text = FIXTURE.read_text()
    for i, (line, body) in enumerate(discovery.blocks(text)):
        assert extract.nest_at_nest_index(text, i) == extract.nest(text, line)


def test_routine_selects_one_procedure_and_refuses_to_guess():
    text = ("subroutine a\n  x = 1\nend subroutine a\n"
            "function b()\n  b = 2\nend function b\n")
    assert extract.routine(text, "a").startswith("subroutine a")
    assert extract.routine(text, "b").startswith("function b")
    with pytest.raises(KeyError):
        extract.routine(text, "c")


def test_statement_folding_joins_continuations():
    """These lines are wrapped at ~120 columns, so reading them one at a time truncates every
    right-hand side -- which is how a first version of the discovery classified nineteen averaging
    loops as unknown."""
    body = ["        a = b + &", "            c", "        d = e"]
    st = discovery.statements(body)
    assert st == ["a = b + c", "d = e"]     # stripped, then concatenated


def test_cpp_is_a_real_pass_not_a_strip():
    """`-C` is load bearing: the anchors that locate a nest ARE comments (`!$acc parallel loop`), so
    a pass that dropped them would make extraction impossible."""
    src = "#define N 4\n!$acc parallel loop collapse(3)\ndo j = 0, N\nend do\n"
    out = extract.cpp(src, defines=[])
    assert "!$acc parallel loop" in out
    assert "do j = 0, 4" in out
    assert "#define" not in out


def test_cpp_defines_reach_the_source():
    src = "#ifdef KEEP\nkept\n#endif\n#ifdef DROP\ndropped\n#endif\n"
    out = extract.cpp(src, defines=["KEEP"])
    assert "kept" in out and "dropped" not in out


# ---------------------------------------------------------------------------------------------------
# The branch a nest sits in.  A source may hold TWO nests of one shape with DIFFERENT arithmetic, one
# per arm of a branch pair, and which one is live is not a choice the emitter gets to make.
# ---------------------------------------------------------------------------------------------------

BRANCHED = """subroutine s
  if (mode /= dual_pass) then
    !$acc parallel loop
    do j = 1, n
      x = flux(j - 1) - flux(j)
    end do
  else
    !$acc parallel loop
    do j = 1, n
      x = one_face(j)
    end do
  end if
end subroutine
"""


def test_a_nest_in_the_if_arm_reports_that_guard():
    assert extract.branch_of(BRANCHED, 4) == "mode /= dual_pass"


def test_a_nest_in_the_ELSE_arm_reports_the_NEGATED_guard():
    """Scanning back for the nearest `if (...) then` alone returns the SAME text for an `if` arm and
    its `else` arm -- MEASURED on a real source, where both x-direction flux nests reported
    `hypo_nc_mode /= hypo_nc_mode_dual_pass`.  A caller selecting "the nest inside guard G" then keeps
    BOTH, emits two different arithmetic forms under one library name, and the second silently
    overwrites the first on the way to the bake."""
    assert extract.branch_of(BRANCHED, 9) == "!.(mode /= dual_pass)"
    assert extract.branch_of(BRANCHED, 4) != extract.branch_of(BRANCHED, 9)


def test_a_nest_in_no_branch_returns_the_empty_string():
    """The empty string is the "not in a branch" answer, which is what makes a guard filter able to say
    'this shape is not covered by the guard' instead of silently dropping it."""
    assert extract.branch_of(BRANCHED, 1) == ""            # the `subroutine` line, above the `if`
    assert extract.branch_of("subroutine s\n  do j = 1, n\n  end do\nend subroutine\n", 2) == ""


def test_every_arm_of_an_else_if_chain_gets_a_DISTINCT_answer():
    """THE COLLISION THIS CLOSES, and the shape of the answer.

    With only the nearest `if` negated, the `else if` arm and the plain `else` arm BOTH answered
    `!.(a)` -- so a caller selecting "the nests inside guard `!.(a)`" kept two arms that compute
    different things, which is exactly what the `else` negation was added to stop, one level in.

    The convention the function settles on: an arm answers the negation of the NEAREST enclosing
    condition, conjoined with its own when it is an `else if`.  That makes the `else` arm `!.(b)`
    rather than the full `!.(a) && !.(b)` -- incomplete as a predicate, and the right thing for what
    this is FOR, which is telling arms apart.  DISTINCTNESS is therefore the property asserted, not
    the three literal strings; a future change that made them distinctive in another way would still
    satisfy it, and one that re-introduced the collision could not."""
    src = ("subroutine s\n"
           "  if (a) then\n    x = 1\n"
           "  else if (b) then\n    y = 2\n"
           "  else\n    z = 3\n"
           "  end if\nend subroutine\n")
    arms = [extract.branch_of(src, n) for n in (2, 4, 6)]
    assert arms == ["a", "!.(a) && (b)", "!.(b)"]
    assert len(set(arms)) == 3
