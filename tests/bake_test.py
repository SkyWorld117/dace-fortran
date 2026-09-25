# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turning an SDFG into a library: what gets CONSTANT-FOLDED, and what gets STAMPED.

`closure_hash` is tested as a PROPERTY and not as a value: it is a stamp whose only job is to change
when the thing it describes changes, and a test asserting a literal would pass on an implementation
that ignored its input entirely.
"""
import tempfile
from pathlib import Path

import dace
import pytest

from dace_fortran import bake


@pytest.fixture
def closure(tmp_path: Path) -> Path:
    d = tmp_path / "closure"
    d.mkdir()
    (d / "m_b.f90").write_text("module m_b\nend module\n")
    (d / "m_a.f90").write_text("module m_a\nend module\n")
    return d


# ---------------------------------------------------------------------------------------------------
# The closure: as files, and as a stamp
# ---------------------------------------------------------------------------------------------------


def test_closure_files_is_SORTED_so_the_digest_is_order_independent(closure):
    assert [f.name for f in bake.closure_files(closure)] == ["m_a.f90", "m_b.f90"]


def test_closure_files_on_a_missing_or_empty_directory_is_an_EMPTY_list(closure):
    """It reports; whether an empty closure is fatal is the CALLER's judgement, because the message
    that helps names a project's own regeneration command."""
    assert bake.closure_files(closure / "nope") == []
    empty = closure / "empty"
    empty.mkdir()
    assert bake.closure_files(empty) == []


def test_the_digest_changes_when_a_module_is_ADDED(closure):
    before = bake.closure_hash(closure)
    (closure / "m_c.f90").write_text("module m_c\nend module\n")
    assert bake.closure_hash(closure) != before


def test_the_digest_covers_the_NAMES_and_not_only_the_bytes(closure):
    """A rename with identical contents must change it: the closure is a SET OF MODULES, and swapping
    one for another with the same text is a different closure."""
    before = bake.closure_hash(closure)
    (closure / "m_b.f90").rename(closure / "m_z.f90")
    assert bake.closure_hash(closure) != before


def test_the_digest_is_stable_for_unchanged_input(closure):
    """The falsifying control for the three above: a digest that changed every call would satisfy all
    of them and be useless as a stamp."""
    assert bake.closure_hash(closure) == bake.closure_hash(closure)
    assert len(bake.closure_hash(closure)) == 16


# ---------------------------------------------------------------------------------------------------
# Dropping the closure prefix from a composed unit
# ---------------------------------------------------------------------------------------------------


def test_kernel_only_drops_everything_before_the_first_matching_module():
    text = "module clo\nend module\n! prologue\nmodule sweep_fused_x_mod\nbody\nend module\n"
    assert bake.kernel_only(text, [r"sweep_fused_\w+_mod"]).startswith("module sweep_fused_x_mod")


def test_the_kernel_module_PATTERNS_are_the_callers(closure):
    """They are a naming convention, not derivable from the text -- so a unit whose kernel is spelled
    differently is handled by passing the pattern, and by passing NOTHING it is refused rather than
    silently returning the whole composed text."""
    text = "module kk_mod\nbody\nend module\n"
    assert bake.kernel_only(text, [r"kk_\w+"]) == text
    with pytest.raises(SystemExit, match="no module matching"):
        bake.kernel_only(text, [r"sweep_fused_\w+_mod"])


# ---------------------------------------------------------------------------------------------------
# Constant-folding the free symbols
# ---------------------------------------------------------------------------------------------------


@pytest.fixture
def sdfg():
    s = dace.SDFG("t")
    for n in ("nVar", "jlb", "kb_d0"):
        s.add_symbol(n, dace.int32)
    return s


def test_an_EXTENT_is_never_baked(sdfg):
    """`<name>_d<k>` is a runtime extent by construction -- it is a SHAPE, not a configuration value,
    and folding one would size the kernel for one grid."""
    assert "kb_d0" not in bake.bake_scalars(sdfg)


def test_a_value_wins_and_an_unknown_symbol_folds_to_a_TYPED_zero(sdfg):
    got = bake.bake_scalars(sdfg, values=({"nVar": 7},))
    assert got["nVar"] == 7
    assert got["jlb"] == 0                       # nothing names it: dtype-appropriate zero, int


def test_a_None_VALUE_is_the_sentinel_for_leave_it_at_runtime(sdfg):
    """Distinct from simply omitting the key, which folds it to zero.  A caller needs to say "I know
    about this symbol and it must stay an argument"."""
    assert "nVar" not in bake.bake_scalars(sdfg, values=({"nVar": None},))
    assert bake.bake_scalars(sdfg, values=({"nVar": 7},))["nVar"] == 7


def test_keep_runtime_names_symbols_a_project_knows_and_the_mechanism_does_not(sdfg):
    """The list is the CALLER's because it encodes subscript arithmetic -- folding one to 0 copies the
    wrong planes with no error at all, which is not something a general rule can see."""
    assert "nVar" not in bake.bake_scalars(sdfg, keep_runtime=("nVar",))
    assert bake.bake_scalars(sdfg)["nVar"] == 0          # without the list it is folded


def test_a_symbol_used_as_a_LOOP_BOUND_stays_runtime_and_is_derived_from_the_GRAPH(sdfg):
    """Not from a list, which only approximates it.  Folding a loop bound does not merely mis-size the
    loop: an index subtracting from it becomes an out-of-bounds memlet and the build fails, and one
    that only appears in index arithmetic instead yields a kernel reading the wrong cells silently."""
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("B", [4], dace.float64)
    st = sdfg.add_state()
    m, mx = st.add_map("m", {"i": "0:jlb"})           # `jlb` is now a map range
    r = st.add_access("A")
    w = st.add_access("B")
    t = st.add_tasklet("c", {"_i"}, {"_o"}, "_o = _i")
    st.add_memlet_path(r, m, t, dst_conn="_i", memlet=dace.Memlet("A[i]"))
    st.add_memlet_path(t, mx, w, src_conn="_o", memlet=dace.Memlet("B[i]"))
    assert "jlb" not in bake.bake_scalars(sdfg)        # derived, with nothing naming it


def test_bake_split_returns_the_two_dicts_the_optimize_API_takes(sdfg):
    """A free symbol is a value folded into the kernel; a `Scalar` CONTAINER is a one-element buffer
    that stays an argument, which is why the caller wants them apart.

    THE SCALAR SIDE IS NOT ASSERTED HERE, and the reason is measured rather than convenient: a
    `Scalar` container is NOT in `sdfg.free_symbols` -- not even when a map's range is `0:cnt` -- so
    nothing `bake_scalars` returns can ever reach `isinstance(..., Scalar)`, and the branch is
    unreachable through this path on the DaCe version in use.  The bake is what would exercise it;
    writing a fixture that forced it would be testing a branch the caller cannot reach.
    """
    syms, conts = bake.bake_split(sdfg, values=({"nVar": 3},), keep_runtime=("jlb",))
    assert syms["nVar"] == 3 and "jlb" not in syms
    assert conts == {}



def test_a_SCALAR_CONTAINER_the_kernel_reads_stays_runtime(sdfg):
    """The primitive that cost a full re-bake, and the ONLY of the five that a unit test can pin.

    A scalar container is read through a memlet whose SUBSET is `0`, so its name appears NOWHERE in
    the index arithmetic: `jb` is `mlt(data='jb', subset='0')`, not `Q(j + jb, ...)`.  A rule that
    looks only at map bounds and subsets therefore folds it to zero -- and MEASURED on a real port,
    that turned the family's loop bounds into zeros, the subsets went NEGATIVE, and the build failed
    with `Memlet subset negative out-of-bounds` and `StorageType.GPU_Global but accessed on host`.
    It failed LOUDLY there; the silent version of the same mistake is a kernel reading a neighbouring
    cell, which every differential gate would then read equally wrong.

    THE OTHER FOUR PRIMITIVES ARE NOT UNIT-TESTABLE AND ARE NOT FAKED HERE.  A `SymExpr` bound, a CFG
    branch condition and a control-flow code block all need a graph that DaCe's public API will not
    build in a test -- MEASURED: `add_map(..., {"i": "0:min(jlb, p)"})` keeps every symbol as one that
    HAS `free_symbols`, so a test written around it passes against the broken rule too, and a
    standalone `LoopRegion` never reaches the SDFG it is meant to belong to.  Those are covered where
    they can be: end to end, by the consumer's bake, which is what found the defect.
    """
    sdfg.add_symbol("p", dace.int32)
    sdfg.add_scalar("jb", dace.int32)
    sdfg.add_array("A", [4], dace.float64)
    st = sdfg.add_state()
    m, mx = st.add_map("m", {"i": "0:p"})
    a, jb, w = st.add_access("A"), st.add_access("jb"), st.add_access("A")
    t = st.add_tasklet("c", {"_i", "_j"}, {"_o"}, "_o = _i + _j")
    st.add_memlet_path(a, m, t, dst_conn="_i", memlet=dace.Memlet("A[i]"))
    st.add_memlet_path(jb, m, t, dst_conn="_j", memlet=dace.Memlet("jb[0]"))   # a READ of the container
    st.add_memlet_path(t, mx, w, src_conn="_o", memlet=dace.Memlet("A[i]"))
    assert "jb" in bake.graph_runtime_syms(sdfg)
    assert "jb" not in bake.bake_scalars(sdfg)
