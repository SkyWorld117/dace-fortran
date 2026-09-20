# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A read of a NESTED component of a module-scope struct.

`hlfir-flatten-global-scalar-reads` rewrites `mod%entity%member` into a read of a synthesized
per-member global, and the flattened leaf is named by concatenating the levels.  Nothing covered the
case where the component path has more than one step.

That is not a hypothetical: MFC's `eqn_idx` is `eqn_idx%cont%beg`, `eqn_idx%adv%end`, and so on, and
the flat leaf names (`eqn_idx_cont_beg`) are the KEYS THE BAKE SUPPLIES VALUES FOR.  Emitting a leaf
named after only the inner level -- `eqn_idx_beg` -- leaves a symbol nothing binds; `_bake_scalars`
then bakes every unbound free symbol to 0, and an access of the form `eqn_idx%beg + i - 1` reads
`index -1`.  It surfaced as `Memlet subset negative out-of-bounds` in four kernel families at once.

The mechanism that caused it, for whoever is tempted to repeat it: `traceToGlobalSym` peels a
`hlfir.designate` hop, and a peel that is not restricted to a COMPONENT-LESS element select walks an
inner component designate straight to the global.  The scan already peels the element select, so the
extra peel is not needed for the array-of-records case it was written for.

This test asserts the *names*, because that is what the pipeline has to agree on: the number is right
either way when the value happens to be baked correctly, and silently wrong when it is not.
"""
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

# MFC's `eqn_idx%cont%beg` shape, reduced: a module-scope struct whose member is itself a struct.
# The reader indexes a module ARRAY by the nested member, which is what turns a wrong leaf name into
# a negative index rather than merely a wrong value.
NESTED = """
module m
  type :: rng
    integer :: beg
    integer :: fin
  end type
  type :: eqns
    type(rng) :: cont
    type(rng) :: adv
    integer :: e
  end type
  type(eqns) :: eqn_idx
  real(kind=8), dimension(8) :: q
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    out = q(eqn_idx%cont%beg) + q(eqn_idx%adv%fin)
  end subroutine
end module
"""


def _leaves(sdfg):
    """Every name the SDFG knows about, in the lowered spelling."""
    return sorted(set(str(s) for s in sdfg.free_symbols) | set(sdfg.arrays))


def test_a_nested_component_keeps_both_levels_in_its_leaf_name(tmp_path):
    sdfg = build_sdfg(NESTED, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    names = _leaves(sdfg)
    both = [n for n in names if "cont_beg" in n or "adv_fin" in n]
    assert both, (f"no leaf named for both levels; the pipeline binds `eqn_idx_cont_beg`-shaped "
                  f"names, so a one-level leaf is a symbol nothing supplies. Got: {names}")
    # And the inner-level-only spelling must NOT be what was emitted instead: `eqn_idx%cont%beg`
    # collapsing to `..._beg` is the regression, and it is invisible in a name list that also
    # contains the correct leaf.
    collapsed = [n for n in names if n.endswith("_beg") and "cont_" not in n]
    assert not collapsed, f"a component path collapsed to its inner level only: {collapsed}"


def test_the_nested_leaves_are_what_a_case_bake_supplies(tmp_path):
    """The names are a CONTRACT with the bake, not a cosmetic detail, so it is worth stating which
    side depends on which.

    `_bake_scalars` bakes every free symbol it has a value for and **every other free symbol to 0**
    (`mfc_dace_build.BAKE_COMMON`, and the `baked symbols:` line it prints).  Its keys are the flat
    leaf spellings.  So a graph that asks for `eqn_idx_beg` gets 0 -- not an error, not a warning,
    just a wrong constant -- while the value that was written for it sits unused under
    `eqn_idx_cont_beg`.

    This asserts the two spellings agree on the form `sym_member_member`, which is the only thing a
    test here can pin down.  The numeric half cannot be a unit test: the companion global is
    synthesised with no initializer, so a compiled graph reads it uninitialized and dace's own
    codegen check (`-Wuninitialized`) correctly refuses it.  The end-to-end check is the bake gate.
    """
    sdfg = build_sdfg(NESTED, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    names = _leaves(sdfg)
    for leaf in ("eqn_idx_cont_beg", "eqn_idx_adv_fin"):
        assert leaf in names, (f"{leaf} missing -- a bake keyed on the two-level spelling would "
                               f"leave this graph's symbol unbound and zero. Got: {names}")
