# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The permissive ``LoopToMap`` escalation must be OPT-IN PER KERNEL, and defaulting it OFF must be
LOUD, not silently skipping the mapping.

The escalation (``offload._mapify``) is a STRUCTURAL rewrite with no conservative soundness check
behind it: it maps loop bodies the per-iteration affine-write check refused.  Measured (W9-localise,
2026-10-03) on the fused acoustic-substep window it restructured ~24 nests (52 -> 28 maps) and the
resulting dataflow is NOT arithmetically equivalent to stock -- it reproduces the shipped GPU
kernel's wild divergence (``t_2``/``ph`` maxrel ~ 2) -- while disabling ONLY the escalation makes
``wrfout_d01`` byte-identical (md5 ``b18a9d02c0cfa8c72ee41f1bf52fe445``).

The tests below pin both halves of the fix, each from the side that can fail:

  * the fixture's loop body really is refused by the conservative check and accepted by the
    permissive one (so "off" and "on" are genuinely different -- an empty difference would make the
    rest vacuous);
  * the default leaves the loop unmapped (falsifiable: a regression to the unconditional escalation
    would map it);
  * an explicit ``permissive_ltm=True`` still maps it (the opt-in is not dead);
  * a kernel that the conservative check refuses and that permissive could have scheduled FAILS
    LOUDLY by default -- never a silently restructured, wrong kernel.
"""
import dace
import pytest

from dace.sdfg import nodes as N
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap

from dace_fortran.offload import _mapify, offload_device_resident
from dace_fortran.pipelines import optimize


def refused_loop_sdfg(name: str, trip: int = 100) -> dace.SDFG:
    """A LoopRegion whose write index does not depend on the iterator -- the refusal, minimally.

    Conservative ``LoopToMap`` requires every write to be a per-iteration-injective affine
    ``a*i + b`` (``|a| >= 1``).  Here ``B[3]`` is loop-invariant, so the check refuses and the loop
    survives; permissive skips the check and maps it.  ``trip > unroll_limit`` (8) keeps
    ``optimize``'s ShortLoopUnroll from unrolling the loop out from under the test.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', (trip,), dace.float64)
    sdfg.add_array('B', (8,), dace.float64)
    body = sdfg.add_state('body', is_start_block=True)
    t = body.add_tasklet('t', {'a'}, {'b'}, 'b = a')
    a = body.add_access('A')
    b = body.add_access('B')
    body.add_edge(a, None, t, 'a', dace.Memlet('A[i]'))
    body.add_edge(t, 'b', b, None, dace.Memlet('B[3]'))
    sdfg.add_loop(None, body, None, 'i', '0', f'i < {trip}', 'i + 1')
    return sdfg


def _maps(sdfg) -> int:
    return sum(1 for st in sdfg.all_states() for nd in st.nodes() if isinstance(nd, N.MapEntry))


def _loops(sdfg) -> int:
    return sum(1 for nd in sdfg.nodes() if isinstance(nd, LoopRegion))


def test_the_fixture_is_genuinely_refused_by_the_conservative_check():
    """Falsifiability anchor: if conservative and permissive agreed on this body, every other test
    here would pass vacuously.  Conservative fires 0 times; permissive fires 1."""
    c = refused_loop_sdfg('ltm_conservative')
    assert c.apply_transformations_repeated(LoopToMap, validate=False) == 0
    assert _loops(c) == 1 and _maps(c) == 0

    p = refused_loop_sdfg('ltm_permissive')
    assert p.apply_transformations_repeated(LoopToMap, validate=False, permissive=True) == 1
    assert _loops(p) == 0 and _maps(p) == 1


def test_mapify_default_does_not_escalate():
    """The default is conservative-only: the refused loop stays a loop and no Map appears."""
    sdfg = refused_loop_sdfg('ltm_mapify_default')
    _mapify(sdfg)
    assert _loops(sdfg) == 1, "default silently mapped a loop the conservative check refused"
    assert _maps(sdfg) == 0


def test_mapify_opt_in_still_escalates():
    """The opt-in is live: with ``permissive_ltm=True`` the same body is mapped."""
    sdfg = refused_loop_sdfg('ltm_mapify_optin')
    _mapify(sdfg, permissive_ltm=True)
    assert _loops(sdfg) == 0
    assert _maps(sdfg) >= 1


def test_offload_fails_loudly_when_the_default_cannot_schedule():
    """No map to schedule -> RuntimeError, and the message points at the opt-in.  The point of the
    fix: a kernel the conservative check refuses fails loudly rather than being silently
    restructured into something the gates would pass."""
    with pytest.raises(RuntimeError, match='permissive_ltm'):
        offload_device_resident(refused_loop_sdfg('ltm_loud'))


def test_offload_opt_in_schedules_a_kernel_that_needs_it():
    """An explicit opt-in enables the escalation and yields a device map."""
    sdfg, n_gpu, _ = offload_device_resident(refused_loop_sdfg('ltm_sched'), permissive_ltm=True)
    assert n_gpu >= 1
    assert _loops(sdfg) == 0


def test_optimize_threads_the_flag_end_to_end():
    """The flag reaches the offload through ``optimize``: default raises, opt-in schedules."""
    with pytest.raises(RuntimeError, match='permissive_ltm'):
        optimize(refused_loop_sdfg('ltm_opt_default'), gpu=True, validate=False)

    sdfg = refused_loop_sdfg('ltm_opt_on')
    optimize(sdfg, gpu=True, validate=False, permissive_ltm=True)
    assert _maps(sdfg) >= 1
    assert any(nd.map.schedule == dace.dtypes.ScheduleType.GPU_Device
               for st in sdfg.all_states() for nd in st.nodes()
               if isinstance(nd, N.MapEntry))
