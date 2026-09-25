# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The TU binding record: a record read, a record APPLIED, and the refusals between the two.

Fixtures rather than a real project's records on purpose -- the module's contract is a FORMAT, and a
test that needed one project's JSON would not be shipping with the format.

The one thing a fixture must get right: an ACCESS is `<base>%sf(...)`.  A record's `base` is a PREFIX
of that, and the completeness guard looks for a surviving `%sf(`.  A statement with no `%sf` at all is
bound by EVERY table vacuously -- so a test written that way passes for the wrong reason, which is how
the first version of this file was wrong.
"""
import json

import pytest

from dace_fortran import tu_bindings as tb


def fam(shapes, **fam_level):
    raw = {"family": "fam", "source_routine": "s_routine", "shapes": shapes}
    raw.update(fam_level)
    return tb.Family(name="fam", raw=raw)


def acc(base, dummy, intent="in", rank=3):
    return {"base": base, "dummy": dummy, "intent": intent, "rank": rank}


# Two tables whose ONLY difference is the arrays they read -- the case selection-by-matching exists
# for: no key over (op, dim) can name them, so a table is chosen by whether it BINDS.
TWO_SHAPES = {
    "OP:x": {"accesses": [acc("qa", "q{n}"), acc("oa", "o{n}", "inout")]},
    "OP:y": {"accesses": [acc("rb", "r{n}"), acc("ob", "p{n}", "inout")]},
}
NEST_X = "oa(c, b, a) = qa%sf(c, b, a)"
NEST_Y = "ob(c, b, a) = rb%sf(c, b, a)"
# the two-access nest the partial-vs-full test needs
NEST_TWO = "oc(c, b, a) = qa%sf(c, b, a) + rb%sf(c, b, a)"


def test_the_table_that_BINDS_the_statements_is_the_one_selected():
    """Selection is by MATCHING, not by key.  `OP:x`'s table leaves a `rb%sf(` standing, so it is
    refused as an option rather than chosen by a key that does not distinguish the two."""
    assert fam(TWO_SHAPES).for_statements([NEST_Y]).key == "OP:y"
    assert fam(TWO_SHAPES).for_statements([NEST_X]).key == "OP:x"


def test_no_table_that_binds_is_a_REFUSAL_and_not_a_guess():
    """The whole point of the record: a shape the emitter has not been generalised for must be
    REPORTED.  A table chosen by approximation is a kernel reading the wrong array with a green build."""
    with pytest.raises(tb.ShapeNotRecorded, match="no binding table binds this nest"):
        fam(TWO_SHAPES).for_statements(["oo(c, b, a) = zz%sf(c, b, a)"])


def test_a_PARTIAL_table_is_a_FALLBACK_and_never_beats_a_table_that_binds():
    """MEASURED on the project this came from: returning the first partial table picked the x table
    for a y nest, because a partial table fails on a SUBSET of every nest it is tried against.  A
    table that binds FULLY always wins; a partial one is used only when none does.

    The nest carries TWO `%sf` accesses on purpose -- see the note below on what the guard can see.
    """
    shapes = {"OP:part": {"partial": True, "accesses": [acc("qa", "q{n}")]},
              "OP:full": {"accesses": [acc("qa", "q{n}"), acc("rb", "r{n}")]}}
    assert fam(shapes).for_statements([NEST_TWO]).key == "OP:full"
    # ... and the partial table is still reachable when nothing binds fully
    assert fam({"OP:part": {"partial": True, "accesses": [acc("qa", "q{n}")]}}) \
        .for_statements([NEST_TWO]).key == "OP:part"


def test_the_completeness_guard_only_sees_ACCESSES_that_carry_the_idiom():
    """WHAT THIS GUARD CANNOT SEE, asserted so it is a known limit rather than a surprise.

    An access is `<base>%sf(...)`, and the refusal fires on a surviving `%sf(`.  A PLAIN array access
    the record never declares is therefore left standing and NOT reported: `oa(...)` has no `%sf`, so
    a table binding only `qa` "binds" this nest completely as far as the guard is concerned.

    That is not silent in the end -- the emitted unit then references an undeclared name and fails to
    COMPILE, which the emitting gate checks -- but it is a failure two stages later than it looks, and
    a record whose plain-array access is simply wrong (rather than missing) still compiles.
    """
    only_qa = fam({"OP:x": {"accesses": [acc("qa", "q{n}")]}}).by_key("OP:x")
    # the `%sf` access binds and the nest is accepted
    assert "q1(" in only_qa.substitute(NEST_X, None)
    # ... while an unbound `%sf` access in the same nest is refused
    with pytest.raises(tb.BindingError):
        only_qa.substitute("oa(c, b, a) = qa%sf(c, b, a) + zz%sf(c, b, a)", None)


def test_an_uncovered_access_is_refused_so_an_undeclared_dummy_is_never_emitted():
    """`substitute` raises for a surviving `%sf(` in BOTH cases; honouring `partial` is the CALLER's
    job (`for_statements` above, and the emitter that drops and reports).  Asserted as it is, because
    a test that expected `partial` to silence this would be describing a different design."""
    full = fam({"OP:x": {"accesses": [acc("qa", "q{n}")]}}).for_statements([NEST_X])
    for b in (full, fam({"OP:x": {"partial": True, "accesses": [acc("qa", "q{n}")]}})
              .for_statements([NEST_X])):
        with pytest.raises(tb.BindingError, match="does not bind these access"):
            b.substitute("zz = unbound%sf(c, b, a)", None)


def test_the_refusal_NAMES_the_statement_and_not_just_the_access():
    """The same access can be bound in one arm and not the other, and a bare list of names gives no
    way to tell which -- worse, the names of the copies that DID match are already rewritten, so the
    list reads as a contradiction."""
    b = fam({"OP:x": {"accesses": [acc("qa", "q{n}")]}}).for_statements([NEST_X])
    with pytest.raises(tb.BindingError) as e:
        b.substitute("zz = unbound%sf(c, b, a)", None)
    assert "in:" in str(e.value) and "unbound" in str(e.value)


def test_by_key_lifts_the_blocks_that_are_per_SHAPE():
    """`library`, `loop_vars`, `loop_bounds` and `locals` describe THIS shape, and `from_dict` reads
    them at FAMILY level.  MEASURED twice on the project this came from: without the lift the rename
    never ran (a unit written under its per-pair name where `visc_avg_x_mod` was wanted) and the loop
    override never applied -- both silent no-ops that looked like the feature working."""
    f = fam({"OP:x": {"loop_vars": {"c": "a"},
                      "library": {"module": "m_mod", "rename": {"dL1": "dxL1"}},
                      "loop_bounds": {"a": ["jb", "je"]}, "locals": ["inv_ds"],
                      "accesses": [acc("qa", "q{n}")]}})
    b = f.by_key("OP:x")
    assert b.key == "OP:x"
    assert b.loop_vars == {"c": "a"}
    assert b.library["module"] == "m_mod"
    assert b.loop_bounds == {"a": ["jb", "je"]}
    assert b.locals == ["inv_ds"]


def test_an_unknown_key_is_refused():
    with pytest.raises(tb.ShapeNotRecorded, match="no binding table"):
        fam(TWO_SHAPES).by_key("OP:zzz")


def test_a_family_with_NO_shapes_is_refused_rather_than_returning_nothing():
    """A family with several call sites cannot be described by one access table, so a record without a
    `shapes` map is a record that cannot say what it covers."""
    with pytest.raises(tb.ShapeNotRecorded, match="no `shapes` tables"):
        fam({}).for_statements([NEST_X])


def test_load_family_takes_the_DIRECTORY_because_where_records_live_is_the_consumers():
    """The FORMAT is general; the layout is not."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        Path(td, "fam.json").write_text(json.dumps({"family": "fam", "shapes": TWO_SHAPES}))
        assert tb.load_family("fam", td).name == "fam"
        with pytest.raises(tb.BindingError, match="no TU binding record"):
            tb.load_family("nope", td)
        Path(td, "flat.json").write_text(json.dumps({"family": "flat"}))
        with pytest.raises(tb.BindingError, match="needs a `shapes` map"):
            tb.load_family("flat", td)


def test_rows_used_is_false_without_a_row_loop():
    """The row loop indexes a COMPONENT (`%vf(i)`), so whether a statement uses it decides whether the
    loop may be unrolled -- and unrolling one that is a real SUBSCRIPT drops the `do` header while the
    subscript still names it, leaving `i` undeclared."""
    b = fam({"OP:x": {"accesses": [acc("qa", "q{n}")]}},
            rows={"var": "i", "from": "iv%beg", "count": 3}).by_key("OP:x")
    assert tb.rows_used(["oa(c, b, a) = qa%vf(i)%sf(c, b, a)"], b)
    assert not tb.rows_used(["oa(c, b, a) = 0.d0"], b)
    assert not tb.rows_used(["oa(c, b, a) = qa%vf(i)%sf(c, b, a)"],
                            fam(TWO_SHAPES).by_key("OP:x"))          # no `rows` in this record


def test_a_malformed_access_or_row_block_is_refused_with_the_FIELD_it_is_missing():
    """The record is DATA, so a missing key is a fact to name rather than a `KeyError` three frames
    down inside a substitution."""
    with pytest.raises(tb.BindingError, match="has no `intent`"):
        fam({"OP:x": {"accesses": [{"base": "qa", "dummy": "q{n}", "rank": 3}]}}).by_key("OP:x")
    with pytest.raises(tb.BindingError, match="bad intent"):
        fam({"OP:x": {"accesses": [acc("qa", "q{n}", "sideways")]}}).by_key("OP:x")
    with pytest.raises(tb.BindingError, match="no `accesses`"):
        fam({"OP:x": {}}).by_key("OP:x")
    with pytest.raises(tb.BindingError, match="`rows` has no `count`"):
        fam({"OP:x": {"accesses": [acc("qa", "q{n}")]}}, rows={"var": "i", "from": "iv%beg"}) \
            .by_key("OP:x")


def test_a_row_loop_that_was_unrolled_needs_a_record_that_has_one():
    """`substitute(stmt, 0)` with no `rows` is a caller asking for something the record cannot answer,
    and the alternative is indexing an empty list."""
    b = fam(TWO_SHAPES).by_key("OP:x")
    with pytest.raises(tb.BindingError, match="no `rows`"):
        b.substitute(NEST_X, 0)
