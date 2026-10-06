# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Give each loop its own copy of the size-1 scalars it writes, so the loop can map.

THE PROBLEM, measured (ysu_prod, W27).  ``LoopToMap`` refuses a nest when a written element's index
is not ``a*i+b`` with ``|a| >= 1`` (``loop_to_map.py:459``, via ``_check_range``).  A SIZE-1
container's subset is ``"0"``, which can never match -- so a single Fortran scalar temporary
blocks an entire nest.  That was 2833 of ysu_prod's 2947 refusals: ``brint``, ``bfx0``, ``rlamdz``.

WHY A RENAME IS ENOUGH.  The check is skipped for any container not in ``other_access_nodes``
(``loop_to_map.py:398-405``), which is "transients touched OUTSIDE this loop" plus "non-transients
touched INSIDE it".  A size-1 TRANSIENT confined to a single loop is therefore invisible to it --
verified both ways on ysu_prod: ``if_cond_137``, used only inside ``loop_j_133``, never blocks,
while ``brint`` is touched in FOUR nests (``loop_j_133/200/283/363``) and blocks all four.
``ScalarFission`` already mints some per-scope copies (``brint_0``, ``rvls_0``, ``gamfac_0``); it
does not cover the rest.  So this pass mints one copy per loop nest and rewrites the accesses --
no reshape, no new index.  The mapped body then holds an ordinary per-thread local, which is what
the 157 already-mapped nests do with their ``if_cond_*`` containers.

SOUNDNESS, and why it is checked rather than assumed.  A rename silently miscompiles when the
scalar carries a value:
  * IN  -- an iteration reads before writing, taking the previous iteration's value; the copy
           starts undefined, so the read is garbage.
  * OUT -- a later loop (or top-level code) reads the scalar expecting what this loop left; the
           copy's value never reaches the base container, so the reader gets a stale one.
Both are covered by one condition, checked per group: every uncovered read inside the region must
be reached by a write on every path from the region's entry.  Carried-in fails it directly;
carried-out shows up as a read-before-write in the receiving region.  If any group fails, the
scalar is left alone WHOLE -- a partially fissioned scalar is exactly the silent-miscompile shape
this pass exists to avoid, and this port has already been burned once by a permissive
transformation restructuring 24 nests while every gate stayed green.

A read inside a state is covered when its access node has an IN-EDGE: within one state the
dataflow graph runs as a unit, so the node's writer has executed and the readers read that value.
That is what lets `brint = ...` followed by `hgamu(i,j) = brint*...` in one fused state count as
defined, and it is why the fix reaches the common case at all.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState

#: Enough rounds for any nesting we emit; the iteration is monotone and converges in 2-3.
_FIXPOINT_ROUNDS = 8


def _state_access(state: SDFGState, name: str) -> Tuple[bool, bool]:
    """``(uncovered_read, writes)`` for one state.

    An access node with an in-edge is written by something in this state, and anything reading it
    reads that value -- so the node covers itself.  A reading node with NO in-edge needs the value
    to have been defined before the state is entered.
    """
    uncovered_read = False
    writes = False
    for n in state.nodes():
        if not isinstance(n, nodes.AccessNode) or n.data != name:
            continue
        if state.in_degree(n) > 0:
            writes = True
        elif state.out_degree(n) > 0:
            uncovered_read = True
    return uncovered_read, writes


def _block_exit(block, name: str, sdfg: SDFG, entry: bool) -> Tuple[bool, bool]:
    """``(ok, defined_at_exit)`` for one block, given whether ``name`` is defined on entry.

    ``ok`` is False when the block reads ``name`` without it being defined.
    """
    if isinstance(block, SDFGState):
        uncovered, writes = _state_access(block, name)
        return (not uncovered) or entry, entry or writes

    if isinstance(block, ConditionalBlock):
        ok = True
        exits = []
        has_else = False
        for cond, body in block.branches:
            if cond is None:
                has_else = True
            ok_b, exit_b = _scan(body, name, sdfg, entry)
            ok = ok and ok_b
            exits.append(exit_b)
        # Without an else the fall-through path writes nothing and keeps the entry value.
        exit_v = all(exits) if (has_else and exits) else (entry and all(exits))
        return ok, exit_v

    if isinstance(block, LoopRegion):
        # A loop bound reads the scalar on EVERY iteration, including the first -- so the value is
        # live on entry, which is exactly what a per-loop copy cannot provide.  Refuse outright
        # rather than analyse it; `do k = kts, klpbl` (klpbl a local INTEGER scalar) is a real
        # pattern in these slices.
        if _mentions(block, name):
            return False, entry
        # A loop may run zero times, so it cannot be relied on to define anything on exit.  Its
        # body, though, may be entered with the value a PREVIOUS iteration left -- that is the
        # carried-in case, and folding it into the fixpoint is what proves it absent.
        seen_entry = entry
        ok = True
        exit_v = entry
        for _ in range(_FIXPOINT_ROUNDS):
            ok_i, exit_i = _scan(block, name, sdfg, seen_entry)
            ok = ok_i
            exit_v = entry or exit_i
            nxt = entry or exit_i
            if nxt == seen_entry:
                break
            seen_entry = nxt
        return ok, exit_v

    # Break/Continue/Return and anything else: no accesses to account for.
    return True, entry


def _scan(region: ControlFlowRegion, name: str, sdfg: SDFG,
          entry_defined: bool) -> Tuple[bool, bool]:
    """``(ok, defined_at_exit)`` for a whole control-flow region."""
    blocks = list(region.nodes())
    if not blocks:
        return True, entry_defined

    def entry_of(b, out):
        preds = [e.src for e in region.in_edges(b)]
        return entry_defined if not preds else all(out[p] for p in preds)

    # Must-defined is monotone increasing, so start optimistic and iterate down to the fixpoint.
    out = {b: True for b in blocks}
    for _ in range(_FIXPOINT_ROUNDS):
        changed = False
        for b in blocks:
            _, nv = _block_exit(b, name, sdfg, entry_of(b, out))
            if nv != out[b]:
                out[b] = nv
                changed = True
        if not changed:
            break

    ok = True
    for b in blocks:
        ok_b, _ = _block_exit(b, name, sdfg, entry_of(b, out))
        ok = ok and ok_b

    # An interstate edge that READS the scalar needs it defined where the edge leaves.  Nothing
    # else in this pass looks at edge accesses, and `if (radflux) then` is one.
    for e in region.edges():
        if e.data is None or name not in {m.data for m in e.data.get_read_memlets(sdfg.arrays)}:
            continue
        if e.src in out and not out[e.src]:
            ok = False

    sinks = [b for b in blocks if not list(region.out_edges(b))]
    exit_v = entry_defined if not sinks else all(out[b] for b in sinks)
    return ok, exit_v


def _regions(sdfg: SDFG) -> List[ControlFlowRegion]:
    """The SDFG itself plus every nested control-flow region."""
    return [sdfg] + [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ControlFlowRegion)]


def _outermost_loop(node, sdfg: SDFG) -> Optional[LoopRegion]:
    """The outermost LoopRegion containing a state or a region, or None if there is none."""
    found = None
    parent = node.parent_graph
    while parent is not None and parent is not sdfg:
        if isinstance(parent, LoopRegion):
            found = parent
        parent = getattr(parent, "parent_graph", None)
    return found


def _touched_states(sdfg: SDFG, name: str) -> List[SDFGState]:
    hit = []
    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == name:
                hit.append(state)
                break
    return hit


def _touched_edges(sdfg: SDFG, name: str) -> List[ControlFlowRegion]:
    """Regions holding an interstate edge that READS ``name``.

    A scalar in an edge condition (`if (radflux) then`) is a real access, and one the state walk
    cannot see -- renaming it in the body while the edge still says `radflux` leaves the edge
    reading a container that no longer exists, which is exactly what `validate()` caught here.
    """
    hit = []
    for region in _regions(sdfg):
        for e in region.edges():
            if e.data is None:
                continue
            if name in {m.data for m in e.data.get_read_memlets(sdfg.arrays)}:
                hit.append(region)
                break
    return hit


def _mentions(block, name: str) -> bool:
    """Whether a control-flow block's own condition/loop statements name ``name``.

    A ``ConditionalBlock`` reads its condition through a CodeBlock (`if_279` reads `if_cond_277`),
    so a materialised condition is a container the STATE walk sees once and the block reads again.
    A ``LoopRegion`` evaluates its bounds every iteration, so a scalar there is read per iteration
    and cannot be treated as loop-local.
    """
    if isinstance(block, ConditionalBlock):
        return any(c is not None and name in c.as_string for c, _ in block.branches)
    if isinstance(block, LoopRegion):
        for attr in ("loop_condition", "init_statement", "update_statement"):
            v = getattr(block, attr, None)
            if v is not None and name in getattr(v, "as_string", ""):
                return True
    return False


def _rewrite_codeblock(cb, old: str, new: str) -> None:
    """Rename ``old`` to ``new`` inside a CodeBlock, leaving everything else spelled as it was."""
    if cb is None or old not in getattr(cb, "as_string", ""):
        return
    import ast

    from dace.frontend.python import astutils
    replacer = astutils.ASTFindReplace({old: new})
    try:
        tree = replacer.visit(ast.parse(cb.as_string))
    except SyntaxError:
        return
    if replacer.replace_count > 0:
        cb.as_string = astutils.unparse(tree)


def _rename_in_region(region: ControlFlowRegion, old: str, new: str) -> None:
    """Rename ``old`` to ``new`` throughout ``region`` -- states, their memlets, and the interstate
    edges of the region and everything nested in it.

    ``replace_dict(..., replace_keys=False)``: the assignment KEYS are symbol names, and a data
    container is never one of them, so renaming keys would only corrupt the edge.
    """
    for state in region.all_states():
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == old:
                n.data = new
        for e in state.edges():
            if e.data is not None and e.data.data == old:
                e.data.data = new
    for r in [region] + [b for b in region.all_control_flow_blocks()
                         if isinstance(b, ControlFlowRegion)]:
        for e in r.edges():
            if e.data is not None:
                e.data.replace_dict({old: new}, replace_keys=False)
    # Branch conditions: `if_279` reads `if_cond_277` by name, and a state access to the same
    # container is the only other trace of it.  Renaming the state alone leaves the condition
    # reading a container that no longer exists -- the KeyError this pass hit first.
    for b in region.all_control_flow_blocks():
        if isinstance(b, ConditionalBlock):
            for cond, _ in b.branches:
                _rewrite_codeblock(cond, old, new)


def fission_loop_local_scalars(sdfg: SDFG, verbose: bool = False) -> List[str]:
    """Mint a per-loop copy of every size-1 transient that a loop writes, when it is provably safe.

    Returns the names of the containers created.  Scalars that fail the liveness check, or that are
    touched outside any loop, are left exactly as they were.
    """
    made: List[str] = []
    for name in list(sdfg.arrays):
        desc = sdfg.arrays[name]
        # Only size-1 transients: an ABI container's storage is the caller's, and the multi-element
        # case has an injective index already and is not what refuses the check.
        if not desc.transient or desc.total_size != 1:
            continue

        states = _touched_states(sdfg, name)
        edge_regions = _touched_edges(sdfg, name)
        if not states and not edge_regions:
            continue

        groups: Dict[LoopRegion, List[SDFGState]] = {}
        outside = False
        for site in states + edge_regions:
            loop = _outermost_loop(site, sdfg)
            if loop is None:
                outside = True
                break
            groups.setdefault(loop, [])
            if isinstance(site, SDFGState):
                groups[loop].append(site)
        if outside or not groups:
            # Touched at top level: the loop's value would have to reach back out, which a copy
            # cannot do.  Leave it whole; a read-only scalar here does not refuse the check anyway.
            continue

        # Every group must prove the scalar is dead on entry; one failure disqualifies the scalar.
        safe = True
        for loop in groups:
            if _mentions(loop, name):
                safe = False
                break
            ok, _ = _scan(loop, name, sdfg, False)
            if not ok:
                safe = False
                break
        if not safe:
            if verbose:
                print(f"[loop_scalar_fission] {name}: read-before-write in some nest -- left whole")
            continue

        for loop in groups:
            new = f"{name}_f{loop.label}"
            if new in sdfg.arrays:
                # Never overwrite a live container: the name is derived from a label, and a
                # collision would silently rebind an unrelated array.
                raise RuntimeError(f"loop_scalar_fission: {new} already exists")
            sdfg.arrays[new] = copy.deepcopy(desc)
            _rename_in_region(loop, name, new)
            made.append(new)
        # Nothing reads or writes the base any more; drop it rather than carry a dead container.
        # Edges count: `_touched_states` alone would call a condition-only scalar dead.
        if not _touched_states(sdfg, name) and not _touched_edges(sdfg, name):
            del sdfg.arrays[name]
        if verbose:
            print(f"[loop_scalar_fission] {name}: {len(groups)} loop-local copy(ies)")

    return made
