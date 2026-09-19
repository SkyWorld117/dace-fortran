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
