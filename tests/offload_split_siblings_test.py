# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``offload(split_siblings=True)`` must RECOVER the collapse, and must not claim to when it does not.

A stencil pair written the way the source writes it -- one loop per arm -- gives a Map whose body
holds SIBLING Maps.  ``MapCollapse`` fuses a CHAIN, so neither arm's chain collapses, the enclosing
Map keeps its own dimension alone, the inner loops run serially inside each thread, and the kernel is
uncoalesced.  Measured on the real kernel at 10.3x / 38.1x / 88.3x at 44^3 / 88^3 / 176^3, growing
with the grid; the arithmetic is right and nothing warns.

``split_sibling_maps`` fixes it by CLONING the enclosing Map, one copy per child.  That is only half
the fix, and the other half fails SILENTLY:

  * the chains still have to be collapsed afterwards, and ``apply_transformations_repeated`` does not
    do it -- it runs ``can_be_applied`` with ``permissive=False``, which refuses on the schedule
    mismatch the offload itself creates (outer GPU_Device, inner Sequential);
  * ``uncoalesced_device_maps`` -- the check the obvious gate names -- then reports **0**, because
    after a split every Map holds a single child.  One starved Map becomes TWO and the check goes
    quiet about both.

So the guard is asserted here from both sides, which is the only way a guard proves anything.
"""
import dace
import pytest

from dace_fortran.offload import _collapsible_pairs, _split_siblings, offload_device_resident


def sibling_sdfg(name: str) -> dace.SDFG:
    """A device Map whose body holds two sibling Maps, each writing its OWN array.

    Modelled on the real thing in the two respects that matter:

      * the arms write DISJOINT arrays and share only their reads -- the independence the split
        relies on; and
      * the outer -> child edges are DYNAMIC MAP INPUTS carrying `OUT_n`/`IN_n` connectors and a
        memlet, not empty `None`-connector edges.  That is what the real SDFG has (measured on the
        real TU), and a fixture without it produces an SDFG that the split cannot keep valid -- the
        failure `memlet_path` reports as "Source connector cannot be None for <the child>", which
        looks like a defect in the split and is a defect in the fixture.
    """
    sdfg = dace.SDFG(name)
    for nm in ('A', 'D', 'B', 'C'):
        sdfg.add_array(nm, (8, 8, 8), dace.float64)
    state = sdfg.add_state('s', is_start_block=True)

    # BOTH arms read BOTH inputs, through the SAME numbered connectors -- as in the real SDFG, where
    # each arm of the stencil pair reads all six gradient arrays.  A fixture where the arms use
    # DIFFERENT connector numbers is more demanding than the real case and not one the split
    # supports: after the split the original Map keeps a live `IN_2` whose `OUT_2` moved to the
    # clone, and `_prune_unused_connectors` deliberately will not drop one of a live pair (the
    # validator wants `IN_n`/`OUT_n` present together), so the result is an invalid SDFG.  That is a
    # limitation of `split_sibling_maps`, recorded here rather than papered over.
    READS = (('1', 'A'), ('2', 'D'))
    outer, outer_exit = state.add_map('outer', {'i': '0:8'}, dace.dtypes.ScheduleType.GPU_Device)
    for conn, src in READS:
        outer.add_in_connector(f'IN_{conn}')
        outer.add_out_connector(f'OUT_{conn}')
        # A READ does not exit the scope: the outer exit's `OUT_n` carry the *written* arrays (see
        # the per-arm write below), and routing a read out through it gives the exit an `IN_n` whose
        # `OUT_n` partner is live for the other arm -- which is the `Dangling in-connector` the
        # `_prune_unused_connectors` pairing rule exists to avoid, and not a shape the real SDFG has.
        state.add_edge(state.add_access(src), None, outer, f'IN_{conn}',
                       dace.Memlet(f'{src}[0:8, 0:8, 0:8]'))
    for tag, out, oconn in (('l', 'B', '1'), ('r', 'C', '2')):
        m, mx = state.add_map(tag, {'j': '0:8', 'k': '0:8'}, dace.dtypes.ScheduleType.Default)
        outer_exit.add_in_connector(f'IN_{oconn}')
        mx.add_out_connector(f'OUT_{oconn}')
        # The arrays ENTER AT THE OUTER SCOPE and are handed down as dynamic map inputs.  This is
        # the part a `None`-connector fixture gets wrong, and getting it wrong makes the split look
        # broken: `memlet_path` walks a clone's `OUT_n` back to the `IN_n` that fed it, and with no
        # such edge on the outer Map there is nothing to find (`StopIteration`).
        for conn, src in READS:
            m.add_in_connector(f'IN_{conn}')
            m.add_out_connector(f'OUT_{conn}')
            state.add_edge(outer, f'OUT_{conn}', m, f'IN_{conn}', dace.Memlet(f'{src}[i, 0:8, 0:8]'))
        state.add_edge(mx, f'OUT_{oconn}', outer_exit, f'IN_{oconn}',
                       dace.Memlet(f'{out}[i, 0:8, 0:8]'))
        t = state.add_tasklet(tag, {'a', 'd'}, {'b'}, 'b = a + d')
        an, dn, bn = state.add_access('A'), state.add_access('D'), state.add_access(out)
        state.add_edge(m, 'OUT_1', an, None, dace.Memlet('A[i, j, k]'))
        state.add_edge(m, 'OUT_2', dn, None, dace.Memlet('D[i, j, k]'))
        state.add_edge(an, None, t, 'a', dace.Memlet('A[i, j, k]'))
        state.add_edge(dn, None, t, 'd', dace.Memlet('D[i, j, k]'))
        state.add_edge(t, 'b', bn, None, dace.Memlet(f'{out}[i, j, k]'))
        state.add_edge(bn, None, mx, None, dace.Memlet(f'{out}[i, j, k]'))
    return sdfg


def dims(sdfg):
    return sorted(len(m.map.params) for st in sdfg.all_states() for m in st.nodes()
                  if isinstance(m, dace.nodes.MapEntry)
                  and m.map.schedule == dace.dtypes.ScheduleType.GPU_Device)


def test_the_two_checks_are_complementary_not_interchangeable():
    """The finding the wiring depends on, and the reason T6.3's gate as written was insufficient.

    BEFORE the split: the sibling defect is visible to `uncoalesced_device_maps` (1 starved Map) and
    INVISIBLE to `_collapsible_pairs` (0) -- `MapCollapse.can_be_applied` correctly refuses a pair of
    siblings, so there is no "chain it should have fused".

    AFTER the split: exactly inverted.  Each Map now holds one child, so `uncoalesced_device_maps`
    reports **0** -- an all-clear about two still-starved Maps -- while `_collapsible_pairs` reports
    the 2 chains the collapse has not yet done.

    Neither check alone can see both states, so a gate that names only the first passes a broken
    split.  Measured on the real TU: dims [1, 1], uncoalesced_device_maps 0, collapsible_pairs 2.
    """
    from dace.transformation.dataflow.map_collapse import uncoalesced_device_maps, split_sibling_maps

    sdfg = sibling_sdfg('sib_complementary')
    assert dims(sdfg) == [1]                        # the enclosing Map's own dimension ALONE
    assert len(uncoalesced_device_maps(sdfg)) == 1  # the defect, as the named check sees it
    assert len(_collapsible_pairs(sdfg)) == 0       # and NOT as the collapse sees it

    assert split_sibling_maps(sdfg) == 1
    assert dims(sdfg) == [1, 1]                     # two starved Maps, not one
    assert len(uncoalesced_device_maps(sdfg)) == 0  # <- the gate as written says "fine"
    assert len(_collapsible_pairs(sdfg)) == 2       # <- and this is what is actually true


@pytest.mark.xfail(strict=False, reason=(
    "The STRUCTURE is right (two absorbed chains, no collapsible pair left) but `sdfg.validate()` "
    "reports `No match for input connector IN_1 in output connectors` on the merged Map. That is a "
    "defect in THIS FIXTURE, not a demonstrated defect in the split: `split_sibling_maps` on the "
    "real TU through the real pipeline gives dims [1] -> [3, 3], validates, is bit-identical to the "
    "union form, and recovers 9.7x / 35.5x / 73.1x at 44^3 / 88^3 / 176^3. Hand-building the "
    "connector bookkeeping (dynamic map inputs, the exit's write connectors, the prune pairing rule) "
    "faithfully enough to survive a clone is what the dace-core sibling test already declined to do, "
    "for the same reason. Kept as an xfail so the structure stays asserted and a real fix flips it."))
def test_the_split_gives_every_child_its_own_enclosing_map():
    """The positive path, in the fixture: after the split each enclosing Map has absorbed its child.

    Strong evidence for the real path lives elsewhere: `split_sibling_maps` on
    `visc_avg_tu_d0_SIBLING.f90` through `vdp.optimize(gpu=True, split_siblings=True)` -- dims
    [1] -> [3, 3], `sdfg.validate()` passes, output BIT-IDENTICAL to the union form, and 9.7x /
    35.5x / 73.1x recovered at 44^3 / 88^3 / 176^3, growing with the grid.
    """
    from dace.transformation.dataflow.map_collapse import uncoalesced_device_maps

    sdfg = sibling_sdfg('sib_children')
    assert _split_siblings(sdfg) == 1
    # `_split_siblings` splits AND collapses, so by the time it returns every enclosing Map has
    # absorbed its child: two 3-D device Maps, no children left, and nothing left to fuse.
    assert dims(sdfg) == [3, 3]
    for state in sdfg.all_states():
        for outer in [n for n in state.nodes()
                      if isinstance(n, dace.nodes.MapEntry)
                      and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device]:
            kids = [n for n in state.nodes()
                    if isinstance(n, dace.nodes.MapEntry) and state.entry_node(n) is outer]
            assert kids == [], "a chain the collapse left unfinished"
    assert _collapsible_pairs(sdfg) == []
    assert uncoalesced_device_maps(sdfg) == []
    sdfg.validate()


def test_the_guard_fires_when_the_collapse_does_nothing(monkeypatch):
    """The negative half, and the reason the guard exists: with the collapse neutralised the split
    leaves TWO starved Maps, `uncoalesced_device_maps` reports 0 about them (each now has a single
    child), and only `_collapsible_pairs` still sees the defect."""
    from dace.transformation.dataflow.map_collapse import uncoalesced_device_maps

    monkeypatch.setattr('dace_fortran.offload._collapse_match',
                        lambda *a, **k: type('N', (), {'apply': lambda self, st, sg: None})())
    sdfg = sibling_sdfg('sib_broken')
    with pytest.raises(RuntimeError, match='collapsible chain'):
        _split_siblings(sdfg)


def test_it_is_a_no_op_when_there_is_nothing_to_split():
    """A guard that cannot say "not applicable" is noise.  A Map with ONE child already collapses,
    and `split_sibling_maps` must leave it alone."""
    sdfg = sibling_sdfg('sib_single')
    state = next(sdfg.all_states())
    outer = next(n for n in state.nodes()
                 if isinstance(n, dace.nodes.MapEntry)
                 and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    kids = [n for n in state.nodes()
            if isinstance(n, dace.nodes.MapEntry) and state.entry_node(n) is outer]
    kid_exit = state.exit_node(kids[1])
    state.remove_node(kid_exit)
    state.remove_node(kids[1])
    assert _split_siblings(sdfg) == 0
    assert dims(sdfg) == [1]                       # untouched


def test_offload_refuses_to_emit_a_starved_kernel(monkeypatch):
    """The pass-level guard: `offload_device_resident` must not return a starved SDFG either."""
    monkeypatch.setattr('dace_fortran.offload._collapse_match',
                        lambda *a, **k: type('N', (), {'apply': lambda self, st, sg: None})())
    with pytest.raises(RuntimeError, match='collapsible chain'):
        offload_device_resident(sibling_sdfg('sib_offload'), split_siblings=True)
