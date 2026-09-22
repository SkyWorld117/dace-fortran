# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The offset-copy reshape: one kernel for several CALL INSTANCES, with the offsets derived.

The oracle is MFC's `s_periodic`, whose six (direction, face) cases a single kernel per direction
serves today.  Its call sites pass `(0, nsweep + 1)` for a `%beg` face and `(nsweep + bb + 1, bb)` for
`%end`, and NOTHING in the tree links those integers to the Fortran they implement -- which is what
these tests close: the values are re-derived from the source's own subscripts and compared.

The workspace's `scripts/tu_from_source_gate.sh` carries the other half, against the real committed
units; this file stays free of any MFC path so the library's own tests do not need a checkout.
"""
import pytest

from dace_fortran import reshape as R

# The six faces' swept subscripts, verbatim from `s_periodic`.  `m`/`n`/`p` are MFC's extents for the
# x/y/z directions and `bb` is its ghost width -- all as text, because the answer is symbolic.
FACES = {
    ("x", "beg"): ("-j", "m - (j - 1)", "m", ("0", "m + 1")),
    ("x", "end"): ("m + j", "j - 1", "m", ("m + bb + 1", "bb")),
    ("y", "beg"): ("-j", "n - (j - 1)", "n", ("0", "n + 1")),
    ("y", "end"): ("n + j", "j - 1", "n", ("n + bb + 1", "bb")),
    ("z", "beg"): ("-j", "p - (j - 1)", "p", ("0", "p + 1")),
    ("z", "end"): ("p + j", "j - 1", "p", ("p + bb + 1", "bb")),
}


@pytest.mark.parametrize("key", sorted(FACES))
def test_the_shipped_offsets_are_DERIVED_from_the_source(key):
    """All six faces, against the values the shipped shim hardcodes."""
    dst, src, ext, want = FACES[key]
    assert R.derive_offsets(dst, src, ext) == want


def test_a_minus_sign_distributes_and_does_not_just_disappear():
    """`m - (j - 1)` is `m - j + 1`, NOT `m - j - 1`.

    The first version of this parser rewrote `(j-1)` textually, which is off by two in the constant
    term and INVISIBLE in the slope -- and it made a working implementation report three mismatches.
    A probe that is wrong about arithmetic cannot testify about arithmetic.
    """
    assert R.parse_affine("-j", "m") == (-1, 0, 0)
    assert R.parse_affine("m - (j - 1)", "m") == (-1, 1, 1)
    assert R.parse_affine("m + j", "m") == (1, 0, 1)
    assert R.parse_affine("j - 1", "m") == (1, -1, 0)


def test_the_ghost_width_CANCELS_on_the_reversed_face():
    """`%beg` is `(0, EXT + 1)` -- no `bb` in it -- and that is a fact worth pinning.

    With a slope of -1 the normalising constant IS the ghost width, so the frame's own `+bb` cancels.
    Substituting 6 for `bb` would produce the same integers and hide the cancellation; the symbolic
    form is what makes the derivation comparable to a call site written in symbols.
    """
    doff, soff = R.derive_offsets("-j", "m - (j - 1)", "m")
    assert "bb" not in doff and "bb" not in soff


def test_the_offset_relations_hold_for_EVERY_face():
    """A cross-check that does not go through the derivation: the two planes are `EXT + 1` apart.

    For `%beg` the source is HIGHER by `EXT+1`; for `%end` the destination is.  This is what makes one
    kernel serve both -- the face flips which plane is written, not the spacing.
    """
    for (_, face), (dst, src, ext, _) in FACES.items():
        ad, bd, nd = R.parse_affine(dst, ext)
        as_, bs, ns = R.parse_affine(src, ext)
        assert ad == as_, f"{dst!r}/{src!r} have different slopes"
        # in RAW coordinates: src - dst is +(n*EXT + b) for %beg and negative for %end
        delta = (ns - nd, bs - bd)
        assert delta == ((1, 1) if face == "beg" else (-1, -1)), (face, delta)


def test_faces_that_are_not_the_same_copy_are_REFUSED():
    """`-j` and `j - 1` have opposite slopes, so no single ascending iterate serves both.

    Emitting them anyway is a kernel that copies the right planes over the wrong rectangle -- the kind
    of thing no value check sees.  Refusing is the whole reason this is a function and not a lambda.
    """
    with pytest.raises(R.NotAnOffsetCopy, match="different slopes"):
        R.derive_offsets("-j", "j - 1", "m")


def test_a_non_unit_slope_is_refused():
    with pytest.raises(R.NotAnOffsetCopy, match="slope"):
        R.derive_offsets("2*j", "2*j", "m")


def test_the_nest_puts_the_UNIT_STRIDE_axis_innermost():
    """`gg` last.  The device map's innermost loop is what lands on the thread axis, and `gg` is the
    unit-stride storage subscript -- with it serial, every warp strides across the row.  MEASURED for
    this family: `gg` innermost is a 3-D coalesced map."""
    vars_, bounds = R.nest_vars_and_bounds([("kb", "ke", "kk"), ("lb", "le", "ll")], "bb")
    assert vars_ == ["ll", "kk", "gg"]
    assert bounds == [("0", "le - lb"), ("0", "ke - kb"), ("0", "bb - 1")]


@pytest.mark.parametrize("swept,expected", [
    (1, "q1d(doff + gg, kb + kk, lb + ll) = q1s(soff + gg, kb + kk, lb + ll)"),
    (2, "q1d(kb + kk, doff + gg, lb + ll) = q1s(kb + kk, soff + gg, lb + ll)"),
    (3, "q1d(kb + kk, lb + ll, doff + gg) = q1s(kb + kk, lb + ll, soff + gg)"),
])
def test_the_swept_axis_moves_and_the_transverse_pair_FOLLOWS(swept, expected):
    """The only difference between the x, y and z units.  Placed by iterating STORAGE POSITION and
    consuming the transverse list in order -- a hand-built permutation is where this gets crossed."""
    text = R.render_offset_copy(module="m", kernel="k", swept=swept,
                                transverse=[("kb", "ke", "kk"), ("lb", "le", "ll")],
                                nrows=1, scalars=["bb", "kb", "ke", "lb", "le", "doff", "soff"])
    assert expected in text


def test_the_row_count_drives_BOTH_the_dummies_and_the_body():
    """The eqn dimension is unrolled into one dummy per row, and the two must agree.

    A row left as a LOOP is a dynamic map dimension and the host wrapper then launches once per row --
    MEASURED at 576 launches per direction against 72.  A dummy count that drifted from the body count
    would be an ABI mismatch or an uninitialised row; tying them to one parameter is what prevents it.
    """
    text = R.render_offset_copy(module="m", kernel="k", swept=1,
                                transverse=[("kb", "ke", "kk"), ("lb", "le", "ll")],
                                nrows=4, scalars=["bb", "kb", "ke", "lb", "le", "doff", "soff"])
    # one ASSIGNMENT per row, so count the body form rather than the bare name -- the name also
    # appears in the declaration, and counting it there was the first version's bug.
    body = "\n".join(l for l in text.split("\n") if l.strip().startswith("q"))
    assert body.count(" = ") == 4
    assert body.count("q1d(") == 1 and body.count("q4d(") == 1
    assert "q5d(" not in text and "q5s(" not in text
    # the scalars line is a FIFTH `intent(in)`, so scope the count to the array declarations
    assert text.count("real(8), intent(inout) ::") == 4
    assert text.count("real(8), intent(in) ::") == 4


def test_the_destination_and_the_source_are_DIFFERENT_dummies():
    """Not decoration.  With one dummy the innermost loop cannot become a Map -- `q(i + mm1)` and
    `q(i)` are not provably disjoint for a symbolic offset, and a Map carries no cross-iteration
    ordering -- so the unit-stride iterate stays serial per thread.  Two dummies take the device map
    from 2 dims to 3."""
    text = R.render_offset_copy(module="m", kernel="k", swept=1,
                                transverse=[("kb", "ke", "kk"), ("lb", "le", "ll")],
                                nrows=2, scalars=["bb", "kb", "ke", "lb", "le", "doff", "soff"])
    # row 1 reads the SOURCE plane through `q1s` and writes the destination through `q1d`: if these
    # were one dummy the kernel would be the form the bridge refuses to map.
    assert "q1d(doff + gg, kb + kk, lb + ll) = q1s(soff + gg, kb + kk, lb + ll)" in text
    assert "q2d(doff + gg, kb + kk, lb + ll) = q2s(soff + gg, kb + kk, lb + ll)" in text
    assert "real(8), intent(inout) :: q1d" in text and "real(8), intent(in) :: q1s" in text
