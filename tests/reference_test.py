# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The kernel-validation core: binding the ABI's extents, and the bit-exact comparison."""
import numpy as np
import pytest

from dace_fortran import reference as ref


def test_extents_are_bound_from_the_arrays_passed():
    """A compiled SDFG carries `<array>_d<k>` arguments for each array's leading dimensions.  Filling
    one wrong is not a crash -- it is a plausible-looking wrong answer, which is why this is
    library code rather than a line in each validator."""
    class FakeSDFG:
        def arglist(self):
            return ["a", "a_d0", "a_d1", "b", "b_d0", "offset_a_d0"]

    kw = {"a": np.zeros((4, 5, 6)), "b": np.zeros((7, 8, 9))}
    ref.bind_extents(FakeSDFG(), kw)
    assert kw["a_d0"] == 4 and kw["a_d1"] == 5
    assert kw["b_d0"] == 7
    assert kw["offset_a_d0"] == 1        # an OFFSET is 1, not the shape


def test_an_extent_beyond_the_array_rank_is_refused():
    """The array handed over does not match the kernel's assumption.  Saying so beats indexing a
    shape that is not there and binding whatever happens to be next."""
    class FakeSDFG:
        def arglist(self):
            return ["a", "a_d3"]

    with pytest.raises(ValueError, match="dimension"):
        ref.bind_extents(FakeSDFG(), {"a": np.zeros((2, 2))})


def test_an_unknown_base_array_is_left_alone():
    """An argument whose base array is not in the mapping is not ours to invent."""
    class FakeSDFG:
        def arglist(self):
            return ["x_d0"]

    kw = {}
    ref.bind_extents(FakeSDFG(), kw)
    assert kw == {}


def test_a_bit_difference_is_a_mismatch():
    """The project's whole bar: the pipeline reorders statements but never reassociates arithmetic,
    so anything short of bit-identical is a bug, not rounding."""
    a = np.array([1.0, 2.0, 3.0])
    b = a.copy()
    b[1] = np.nextafter(2.0, 3.0)          # one ulp
    r = ref.compare({"f": a}, {"f": b})[0]
    assert not r.equal and r.mismatches == 1
    assert ref.compare({"f": a}, {"f": a})[0].equal


def test_nan_is_a_mismatch_even_against_itself():
    """`nan != nan`, so a NaN would otherwise be counted only by luck of the comparison used."""
    a = np.array([1.0, np.nan])
    r = ref.compare({"f": a}, {"f": a})[0]
    assert not r.equal and not r.finite


def test_a_shape_difference_is_a_mismatch_not_a_broadcast():
    r = ref.compare({"f": np.zeros(3)}, {"f": np.zeros(4)})[0]
    assert not r.equal


def test_an_array_absent_from_either_side_is_skipped():
    """A kernel's argument list and a reference file's contents rarely coincide exactly; inventing a
    mismatch for a missing reference would be worse than saying nothing."""
    r = ref.compare({"f": np.zeros(2), "g": np.zeros(2)}, {"f": np.zeros(2)})
    assert [c.name for c in r] == ["f"]


def test_the_verdict_reduces_to_one_answer():
    ok = ref.compare({"f": np.ones(2)}, {"f": np.ones(2)})
    assert ref.all_equal(ok)
    assert not ref.all_equal(ref.compare({"f": np.ones(2)}, {"f": np.zeros(2)}))
    assert not ref.all_equal([])          # comparing nothing is not a pass


def test_a_reference_record_round_trips(tmp_path):
    p = tmp_path / "ref.dat"
    from dace_fortran import unformatted
    unformatted.write_record(p, np.arange(6, dtype=np.float64))
    assert ref.reference_from_record(p, shape=(2, 3)).shape == (2, 3)
