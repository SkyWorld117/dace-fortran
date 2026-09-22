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


# --------------------------------------------------------------------------- the ANCHOR
# `discovery.blocks` used to key on the literal `collapse(3)`: a fixed depth, blind to a project's
# macro, and silent about both -- the anchor lines simply stop matching, so the failure is zero
# shapes rather than an error.  It anchors on the directive now and READS the depth from it.

NEST = """  do l = 1, n
    do j = 1, n
      do k = 1, n
        x%sf(k, j, l) = y%sf(k, j, l) - y%sf(k, j - 1, l)
      end do
    end do
  end do
"""


def test_the_macro_form_is_an_anchor():
    """A project's `$:GPU_PARALLEL_LOOP(...)` is the anchor to prefer where it exists -- it is
    semantic, it brackets the nest, and it survives the case where the accelerator directives were
    never defined (the macro may expand to NOTHING, which is exactly when a post-cpp scan goes
    silently blind)."""
    text = "$:GPU_PARALLEL_LOOP(collapse=3, private='[i, j, k, l]')\n" + NEST + "  $:END_GPU_PARALLEL_LOOP()\n"
    assert len(discovery.blocks(text)) == 1


def test_the_closer_is_not_an_anchor():
    """`$:END_GPU_PARALLEL_LOOP()` matches a naive `\\w*PARALLEL_LOOP` pattern, and each nest is then
    counted TWICE -- once at its opener and once at its closer, where there is no `do` and the body
    scan runs to end-of-file.  Measured on a real routine: 36 anchors for 18 loops."""
    text = "$:GPU_PARALLEL_LOOP(collapse=3)\n" + NEST + "  $:END_GPU_PARALLEL_LOOP()\n"
    starts = [ln for ln, _ in discovery.blocks(text)]
    assert len(starts) == 1
    assert len(discovery.blocks(text)[0][1]) < 20      # the nest, not to end-of-file


def test_the_depth_is_read_from_the_directive():
    """A nest the source collapses to FOUR is invisible to a `collapse(3)` key -- and `m_viscous.fpp`
    really does carry `collapse=2`, `collapse=3` and `collapse=4`."""
    four = "$:GPU_PARALLEL_LOOP(collapse=4)\n" + NEST + "  $:END_GPU_PARALLEL_LOOP()\n"
    assert len(discovery.blocks(four)) == 1                       # accepted by default
    assert len(discovery.blocks(four, collapse=3)) == 0           # and still filterable
    assert len(discovery.blocks(four, collapse=4)) == 1


def test_multi_character_loop_variables_are_LOOPS():
    """`do k_loop = ...` is a loop, and a single-letter class silently says it is not.

    MFC's x-direction flux nest spells its loops `j`, `q_loop`, `l_loop`, `k_loop` while the SAME
    construct in y/z uses `q`, `l`, `k`.  `canonical` matched loops with `([a-z])`, so it matched ONE
    of the four, hit its `len(loops) < 3` guard, and returned `None` for the whole nest.

    The consequence was not a wrong shape but a MISSING one: with no FLUX shape there is no axis
    letter, so the arm classifier reports `both/neither` for every arm and the group is refused with
    "could not pair the arms by the direction their stencil reaches" -- a message that names the
    symptom and points nowhere near the class.  MEASURED: the x direction emitted 0 units, and all
    three directions emit after widening the class to `[a-z_]\\w*`.
    """
    body = [
        "do j = 1, sys_size",
        "  do q_loop = 0, p",
        "    do l_loop = 0, n",
        "      do k_loop = 0, m",
        "        inv_ds = 1._wp/dx(k_loop)",
        "        flux_face1 = flux_n(1)%vf(j)%sf(k_loop - 1, l_loop, q_loop)",
        "        flux_face2 = flux_n(1)%vf(j)%sf(k_loop, l_loop, q_loop)",
        "        rhs_vf(j)%sf(k_loop, l_loop, q_loop) = inv_ds*(flux_face1 - flux_face2)",
        "      end do",
        "    end do",
        "  end do",
        "end do",
    ]
    shape = discovery.canonical(body)
    assert shape is not None, "a nest with multi-character loop variables is still not a shape"
    assert shape.op == "FLUX", shape
    assert shape.dim == 0, f"the difference is on subscript 0 (`k_loop - 1`), got {shape.dim}"
    assert "k_loop" in shape.loops or "q_loop" in shape.loops, shape.loops
