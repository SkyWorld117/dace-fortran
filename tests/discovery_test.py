# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shape discovery over the packaged fixture of real loop nests."""
from pathlib import Path

from dace_fortran import discovery

FIXTURE = Path(__file__).parent / "fixtures" / "nest_shapes.f90"

# The fixture is a verbatim excerpt, so these are the counts a human reading it arrives at: three
# averaging shapes of two pairs each, and three difference shapes of one pair each.
EXPECTED = {("AVG", 0): 4, ("AVG", 1): 4, ("AVG", 2): 4,
            ("GRAD", 0): 2, ("GRAD", 1): 2, ("GRAD", 2): 2}


def test_every_nest_of_the_fixture_is_found():
    nests = discovery.blocks(FIXTURE.read_text())
    assert len(nests) == sum(EXPECTED.values()) == 18


def test_the_shapes_match_the_hand_derived_table():
    shapes = discovery.discover(FIXTURE.read_text(), "shapes")
    got = {(s.op, s.dim): len(lns) for s, lns in shapes.items()}
    assert got == EXPECTED


def test_an_unrecognised_nest_is_reported_not_classified():
    """`None` must be an outcome, not a fallback: a nest that does not fit the signature has to be
    sayable, or a caller ends up silently approximating it."""
    text = "!$acc parallel loop collapse(3)\n  do j = 1, n\n    x%sf(j) = 1.0\n  end do\n"
    assert discovery.canonical(discovery.blocks(text)[0][1]) is None
    assert "UNRECOGNISED" in discovery.report(text)


def test_several_loop_ends_on_one_line_still_delimit_the_nest():
    """House styles pack the ends together (`end do; end do; end do`).  A depth counter that only
    recognises a lone `end do` never returns to zero, the scan runs to end-of-file, and every nest in
    the file collapses into one -- which is exactly what a first version of this did."""
    text = ("!$acc parallel loop collapse(3)\n"
            "  do l = 0, 2\n  do k = 0, 2\n  do j = 0, 2\n"
            "    x%sf(j, k, l) = 0.25*(a%sf(j, k, l) + a%sf(j - 1, k, l))\n"
            "  end do; end do; end do\n"
            "!$acc parallel loop collapse(3)\n"
            "  do l = 0, 2\n  do k = 0, 2\n  do j = 0, 2\n"
            "    y%sf(j, k, l) = 0.25*(a%sf(j + 1, k, l) + a%sf(j, k, l))\n"
            "  end do; end do; end do\n")
    nests = discovery.blocks(text)
    assert len(nests) == 2
    # both nests are one shape, at their own lines -- the L and R arms of a single family
    shapes = discovery.discover(text)
    assert len(shapes) == 1
    assert list(shapes.values())[0] == [1, 7]


def test_scoping_selects_one_routine():
    text = ("subroutine a\n!$acc parallel loop collapse(3)\n  do l = 0, 2\n  do k = 0, 2\n"
            "  do j = 0, 2\n    x%sf(j, k, l) = a1%sf(j, k, l)\n  end do\n  end do\n  end do\n"
            "end subroutine a\n"
            "subroutine b\n!$acc parallel loop collapse(3)\n  do l = 0, 2\n  do k = 0, 2\n"
            "  do j = 0, 2\n    x%sf(j, k, l) = a1%sf(j, k, l)\n  end do\n  end do\n  end do\n"
            "!$acc parallel loop collapse(3)\n  do l = 0, 2\n  do k = 0, 2\n"
            "  do j = 0, 2\n    x%sf(j, k, l) = a1%sf(j, k, l)\n  end do\n  end do\n  end do\n"
            "end subroutine b\n")
    assert len(discovery.blocks(discovery.scope_to_routine(text, "a"))) == 1
    assert len(discovery.blocks(discovery.scope_to_routine(text, "b"))) == 2


def test_the_access_idiom_is_a_parameter_not_a_literal():
    """`%sf(...)` is a derived-type component access -- generic Fortran, but an IDIOM.  A parser that
    hard-codes one house style silently finds NOTHING in the next codebase (zero shapes, not an
    error), so the token is a parameter."""
    text = ("!$acc parallel loop collapse(3)\n"
            "  do l = 0, 2\n  do k = 0, 2\n  do j = 0, 2\n"
            "    x%val(j, k, l) = 0.25*(a%val(j, k, l) + a%val(j - 1, k, l))\n"
            "  end do\n  end do\n  end do\n")
    assert discovery.discover(text) == {}                     # the default idiom finds nothing
    shapes = discovery.discover(text, access="%val")
    assert list(shapes) == [discovery.Shape("AVG", 0, ("l", "k", "j"))]
