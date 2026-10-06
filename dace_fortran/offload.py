#!/usr/bin/env python3
"""The GPU offload pass: schedule + storage assignment for device-resident offload.

`offload_device_resident(sdfg)` turns a FaCe-optimized (CPU-mapped) SDFG into
a device-resident GPU kernel:

  * outermost map of every top-level scope -> GPU_Device schedule
    (inner maps stay Sequential = per-thread loops inside the kernel);
  * every non-scalar signature array -> GPU_Global storage, so the generated
    C ABI takes DEVICE pointers and the codegen emits no per-call staging
    copies (the host-staged mirror path measured 3-4 orders slower at scale);
  * scalar/length-1 containers (dt, igr, ...) stay host-side: they cross as
    by-value kernel parameters, staged per call (tiny, and per-call-fresh).

This is the layer the pipeline lacked: `optimize()` maps loops into DaCe maps but leaves them
CPU-scheduled, so a device kernel needs the schedule/storage assignment below on top of it.

 `force_inline` is deliberately a CALLER parameter rather than a default: it is not generally
sound (applied library-wide it made a real kernel produce NaN), so it is enabled per kernel and
paid for by that kernel's differential gate.
"""
import re

import dace
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState


def _mapify(sdfg, permissive_ltm=False):
    """LoopToMap + MapCollapse at this level, then recurse into nested SDFGs.

    dp.optimize's LoopToMap runs on the top-level SDFG only: for the pack
    kernel it mapped just the outer loop while the spatial nest stayed
    control-flow inside a NestedSDFG - inlining that yields a serial tasklet
    and a GPU grid of nvar threads.  Mapping every level before inlining is
    what lets the collapse produce one full-domain map.

    ``permissive_ltm`` gates the PERMISSIVE ``LoopToMap`` escalation and is OFF by default, turned on
    per kernel exactly like ``force_inline`` and ``split_siblings``.  The conservative check is
    sufficient for every kernel whose loop bodies carry a per-iteration affine write (``a*i+b``,
    ``|a| >= 1``); the escalation exists only for bodies that check refuses (MFC's scatter writes,
    and a loop body living entirely inside a NestedSDFG whose connector memlet is the loop UNION).

    IT IS NOT SOUND IN GENERAL, AND THE FAILURE IS SILENT.  Measured (W9-localise, 2026-10-03): on
    the fused acoustic-substep window the conservative ``LoopToMap`` fires ZERO times while the
    permissive escalation restructures ~24 nests (52 -> 28 maps), and the resulting dataflow is not
    arithmetically equivalent to the stock sequence -- running it reproduces the shipped GPU
    kernel's wild divergence (``t_2``/``ph`` maxrel ~ 2).  With only this escalation disabled the
    window is byte-identical to stock (``wrfout_d01`` md5 ``b18a9d02c0cfa8c72ee41f1bf52fe445``).
    The escalation had no per-kernel bit-exact differential behind it; per the pass's own contract
    it must have one, and defaulting to conservative-only is the safe half of that contract.

    So: conservative only by default.  A kernel whose loops the conservative check refuses keeps
    those loops as control flow and, if it then has no map to schedule, ``offload_device_resident``
    FAILS LOUDLY (see the ``n_gpu == 0`` guard) -- loud failure beats a silently restructured, wrong
    kernel.  A caller that has a differential for one kernel opts in with ``permissive_ltm=True``."""
    from dace.transformation.interstate.loop_to_map import LoopToMap
    from dace.transformation.dataflow import MapCollapse

    def _apply_ltm(sdfg):
        """LoopToMap, escalating to permissive only when the caller opted in.

        The main refusal for MFC's kernels: a loop body that lives entirely inside a NestedSDFG
        carries a loop-UNION subset on the connector memlet, so the per-iteration affine write check
        sees no iterator and refuses.  MFC's scatter writes (r = ...; buf(r) = ...) are
        per-iteration-unique by construction; but that is a claim about MFC, and the W9 window
        (which is not a scatter kernel) is the measured counterexample, so the escalation is opt-in
        and must be paid for by a per-kernel bit-exact differential downstream."""
        from dace.sdfg import nodes as _nodes

        def _maps(s):
            return sum(1 for st in s.all_states() for nd in st.nodes()
                       if isinstance(nd, _nodes.MapEntry))

        before = _maps(sdfg)
        sdfg.apply_transformations_repeated(LoopToMap, validate=False)
        if _maps(sdfg) == before and permissive_ltm:
            sdfg.apply_transformations_repeated(LoopToMap, validate=False,
                                                permissive=True)

    _apply_ltm(sdfg)
    for state in sdfg.all_states():
        for node in list(state.nodes()):
            if isinstance(node, nodes.NestedSDFG):
                _mapify(node.sdfg, permissive_ltm=permissive_ltm)
    sdfg.apply_transformations_repeated(MapCollapse, validate=False)


def _force_inline(sdfg):
    """Inline the NestedSDFGs `inline_sdfgs` declines when it declines only its OWN checks.

    `InlineSDFG.can_be_applied` is conservative: on the M3 kernels it returns False for the
    innermost loop-body NestedSDFG, and `InlineSDFG.apply()` on that same node then succeeds.  A
    surviving NestedSDFG keeps its map OUT of the collapse, and that is not a cosmetic loss: the M3
    kernels baked with a 2-D map (l, k) and their innermost loop `j` -- the UNIT-STRIDE one --
    serial per thread, i.e. the strided dimension on the thread axis and no coalescing at all,
    which is what made them an order of magnitude slower than the stock loops they replace.
    Forcing the inline yields the full (l, k, j) chain and the collapse fuses all three.

    This is the same bargain the rest of this pass makes (validate=False, permissive=True): force
    past the check and let the per-kernel bit-exact differential catch an unsound result.  Every
    kernel here is gated that way, and a wrong collapse shows up as a divergence, not as a
    plausible-looking number.

    BUT ONLY THE CONSERVATIVE CHECKS ARE BYPASSED.  `InlineSDFG.can_be_applied` opens with two
    STRUCTURAL guards -- `no_inline`, and "the nested SDFG is a SINGLE-state SDFG" -- and they are
    re-checked here, deliberately, because they are not conservatism: they decide whether this
    transformation is applicable at all.  Forcing past them hands the SINGLE-state inliner a
    MULTI-state region, and `apply()` does not re-check its own guards, so it returns successfully
    having produced a WRONG SDFG rather than raising.

    That is not hypothetical -- it is measured.  Applying this pass to the whole library set made
    the default dispatch path produce NaN at step 1, and re-instrumenting `can_be_applied` per
    kernel shows why: `mfc_dace_conv`'s surviving nested SDFG is refused at `sdfg_nesting.py:183`
    (multi-state) while `mfc_dace_fdiff_src_*`'s is refused at `:218` (a conservative connector
    check, with the in/out memlets actually AGREEING).  Two different refusals, one justified and
    one not, and only the second is safe to force.
    """
    from dace.transformation.interstate import InlineSDFG

    moved, skipped = 0, set()
    for _ in range(20):
        remaining = [(n, st) for st in sdfg.all_states() for n in st.nodes()
                     if isinstance(n, nodes.NestedSDFG)]
        if not remaining:
            break
        progressed = 0
        for node, state in remaining:
            # The structural guards, re-checked (see the docstring): bypassing these is what made a
            # library-wide application silently wrong.
            if node.no_inline or len(node.sdfg.nodes()) != 1 or not isinstance(
                    node.sdfg.nodes()[0], SDFGState):
                skipped.add(id(node))
                continue
            inliner = InlineSDFG()
            inliner.setup_match(sdfg=sdfg, cfg_id=state.parent_graph.cfg_id,
                                state_id=state.block_id,
                                subgraph={InlineSDFG.nested_sdfg: node},
                                expr_index=0, override=True)
            try:
                inliner.apply(state, sdfg)
                moved += 1
                progressed += 1
            except Exception:  # noqa: BLE001 -- a refusal here is not fatal; leave the node
                pass
        # Inlining one node can make another inlinable, so iterate -- but a round that moves nothing
        # will move nothing on the next one either (measured: a second pass over a refused node makes
        # no progress), so stop rather than spin.
        if progressed == 0:
            break
    if skipped:
        print(f"[dace_fortran.offload] force_inline: {len(skipped)} node(s) left alone -- they are "
              f"multi-state, which this transformation cannot lower (see _force_inline)")
    return moved


def _collapsible_pairs(sdfg):
    """(outer, inner) Map pairs `MapCollapse` WOULD fuse and has not -- the Class-1 loss, asked directly.

    Why not `uncoalesced_device_maps`: that asks whether a Map holds SIBLING Maps, and after a split
    every Map holds a single child -- so it reports 0 about a Map that is still starved.  A check that
    says "nothing wrong" where something is wrong is worse than no check, and this is the question
    that has an answer.

    Why not `apply_transformations_repeated(MapCollapse)`: its pattern graph runs `can_be_applied`
    with `permissive=False`, which refuses on a SCHEDULE MISMATCH -- and that mismatch is the state
    the offload itself creates (outer GPU_Device, inner Sequential).  Measured on the real sibling TU:
    repeated -> 0, permissive manual -> 2, device-map dims [1, 1] -> [3, 3].
    """
    from dace.transformation.dataflow import MapCollapse
    out = []
    for state in sdfg.all_states():
        for outer in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
            for inner in [n for n in state.nodes()
                          if isinstance(n, nodes.MapEntry) and state.entry_node(n) is outer]:
                t = MapCollapse()
                t.setup_match(sdfg=sdfg, cfg_id=state.parent_graph.cfg_id, state_id=state.block_id,
                              subgraph={MapCollapse.outer_map_entry: outer,
                                        MapCollapse.inner_map_entry: inner},
                              expr_index=0, override=True)
                if t.can_be_applied(state, 0, sdfg, permissive=True):
                    out.append((outer, inner))
    return out


def _split_siblings(sdfg) -> int:
    """Clone the enclosing Map per child so each stencil arm's chain can collapse on its own.

    THE LOSS THIS RECOVERS, measured on a real kernel: the sibling form (one loop per arm, the way
    the source usually writes it) leaves the enclosing Map covering its own dimension ALONE -- 32
    threads with every access strided -- at **2.0 ms/launch against 5.40 us = 372x**, 81.6% of a run's
    GPU time against 1.2%.  The arithmetic is right and nothing warns.

    Note what this does NOT need: any relation between the arms' bounds.  The union route is
    underdetermined (`sympy` cannot decide `ulb >= c_lo`; there is no `Range.union`), which is why
    this is a split and not a fusion.

    THE COLLAPSE IS RUN HERE, MANUALLY, AND THEN ASSERTED -- because the split is only half a fix and
    the other half fails SILENTLY.  `apply_transformations_repeated(MapCollapse)` finds none of these
    (see `_collapsible_pairs`), so wiring the split in beside it produces TWO starved Maps instead of
    one, while `uncoalesced_device_maps` -- the check T6.3's gate names -- goes quiet about both,
    since each now has a single child.  A split that does not collapse its chains is worse than no
    split, and it passes the gate as written; hence the assertion.
    """
    from dace.transformation.dataflow.map_collapse import split_sibling_maps

    n = split_sibling_maps(sdfg)
    if not n:
        return 0
    # To a fixpoint: merging a pair can expose another (the merged Map now sits directly under a
    # grandparent it could join), so one sweep is not enough.  Bounded, because each collapse
    # strictly reduces the Map count.
    for _ in range(32):
        pairs = _collapsible_pairs(sdfg)
        if not pairs:
            break
        for outer, inner in pairs:
            state = next(st for st in sdfg.all_states() if outer in st.nodes())
            _collapse_match(sdfg, outer, inner).apply(state, sdfg)
    left = _collapsible_pairs(sdfg)
    if left:
        raise RuntimeError(
            f"offload: split {n} sibling map(s) but {len(left)} collapsible chain(s) survived -- the "
            f"split has produced starved Maps that `uncoalesced_device_maps` will NOT report (each "
            f"now has a single child).  Refusing rather than emitting an uncoalesced kernel that every "
            f"gate would pass.")
    return n


def _collapse_match(sdfg, outer, inner):
    """A `MapCollapse` bound to one (outer, inner) pair on the permissive path."""
    from dace.transformation.dataflow import MapCollapse
    state = next(st for st in sdfg.all_states() if outer in st.nodes())
    t = MapCollapse()
    t.setup_match(sdfg=sdfg, cfg_id=state.parent_graph.cfg_id, state_id=state.block_id,
                  subgraph={MapCollapse.outer_map_entry: outer,
                            MapCollapse.inner_map_entry: inner},
                  expr_index=0, override=True)
    return t


def count_gpu(sdfg):
    """(device maps, device arrays) -- the offload's own yardstick, for logging after the fact.

    `offload_device_resident` returns the pair, but `pipelines.optimize(gpu=True)` only returns the
    SDFG (it has to, it is the entry point for the whole pipeline), so a caller that wants to report
    what the offload did asks this instead.
    """
    from dace import data as _data
    n_gpu = sum(1 for st in sdfg.all_states() for nd in st.nodes()
                if isinstance(nd, nodes.MapEntry)
                and nd.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    n_dev = sum(1 for a in sdfg.arrays.values()
                if not a.transient and not isinstance(a, _data.Scalar)
                and a.storage == dace.dtypes.StorageType.GPU_Global)
    return n_gpu, n_dev


def offload_device_resident(sdfg, block_size=None, force_inline=False, split_siblings=False,
                            permissive_ltm=False):
    """Schedule + storage assignment for device-resident offload. In place.

    `force_inline` is OFF by default and must be turned on PER KERNEL, because it is not generally
    sound: applied to the whole library set it made the default dispatch path produce NaN at step 1
    (the flux-difference family), which is exactly the kind of thing `InlineSDFG.can_be_applied`'s
    refusal was protecting against.  It is enabled for `mfc_dace_fdiff_src_*`, where the resulting
    kernel is verified bit-identical and 1.57x faster, and for nothing else.

    `split_siblings` is OFF by default for the same reason: it is a STRUCTURAL rewrite whose failure
    mode is a silently wrong or silently slow kernel.  It is a no-op on any SDFG whose device Maps do
    not hold sibling Maps -- which is every kernel in the shipped MFC set, since those TUs are
    hand-written in the union form -- so turning it on cannot move them; it exists for the TUs the
    extractor emits from the source's own sibling form.  Turn it on per kernel, with that kernel's
    differential.

    `permissive_ltm` is OFF by default and turned on PER KERNEL, for the same reason as the other two:
    the permissive `LoopToMap` escalation is a STRUCTURAL rewrite with no conservative soundness
    check behind it (see `_mapify`).  Measured (W9-localise): on the fused acoustic-substep window it
    silently restructured ~24 nests and produced a kernel that diverges wildly from stock, while
    conservative-only is byte-identical.  With it off, a kernel whose loops the conservative check
    refuses stays correct (loops left as control flow) and, if it has no map left to schedule, fails
    loudly at the `n_gpu == 0` guard below.  A caller with a per-kernel bit-exact differential may
    opt in.
    """
    # 0a. Mapify every level, then inline the loop-body NestedSDFGs the
    #     frontend introduces: MapCollapse cannot fuse across the nested-SDFG
    #     boundary, and without fusion the GPU map would cover only the outer
    #     loop (sys_size/nvar threads) with the N^3 loops serial per thread.
    _mapify(sdfg, permissive_ltm=permissive_ltm)
    from dace.sdfg import utils as sutils
    sutils.inline_sdfgs(sdfg, permissive=True)
    if force_inline:
        _force_inline(sdfg)
    # 0b. Collapse perfectly-nested map chains into single maps: scheduling the
    #     outermost of i(l(k(j))) as a GPU map would put sys_size (=5) threads on
    #     the device and run the N^3 spatial loops serially per thread.
    from dace.transformation.dataflow import MapCollapse
    sdfg.apply_transformations_repeated(MapCollapse, validate=False)

    # 0b'. Split sibling Maps -- AFTER the collapse above, and that order is load-bearing.
    #      The split decides whether two children are INDEPENDENT by resolving what each one writes,
    #      and that walk only finds writes directly in a child's own scope.  Run before the collapse,
    #      each arm is still an uncollapsed chain (`_loop_it_4` whose body is `_loop_it_5`), so
    #      neither child appears to write anything, the conservative independence check refuses, and
    #      the split silently does nothing -- measured, and it is why the first wiring of this
    #      pass changed no kernel.  After the collapse each arm IS one Map holding its tasklets.
    if split_siblings:
        _split_siblings(sdfg)

    # 0c. Reorder the collapsed map's dims: PIN the innermost (memory-fastest)
    #     dim last -- the block's threads span it, so it must stay the coalesced
    #     access dim -- and sort the rest ASCENDING (symbolic = large last).
    #     cuda.py's grid mapping: grid.x = ceil(range_last/block), grid.y =
    #     range[-2], grid.z = flatten(range[:-2]) with CUDA z-limit 65535.
    #     RK nest (i,l,k,j) stays (z=i*l=1280 OK, block spans j).  Pack nest
    #     (l,k,j,i) would put z=l*k=65536; (j,l,k,i) yields z=j*l=512.
    # NOTE (S48, measured): this pins the LAST MAP DIM on the thread axis, on the assumption that
    # the innermost FORTRAN loop is the memory-fastest one.  That is true only when the TU's loops
    # follow storage order.  A stride-aware version was written and TRIED -- choose the dim that
    # indexes contiguous memory -- and it was a NO-OP on every kernel in the set: the kernels that
    # are coalesced already had the unit-stride dim last.  `mfc_dace_periodic_x` is the one kernel
    # the audit flags, and no permutation can fix it, because there the unit-stride dim is the SWEPT
    # axis `gg` and it is not in the map at all (its bound is a literal, which the bake requires).
    # So the reorder is not the lever; the lever is getting the right dims INTO the map.  Reverted
    # rather than kept on principle -- an unmeasured pass change does not stay.

    def _reorder_map_dims(state, entry):
        sizes = entry.map.range.size()
        if len(sizes) <= 1:
            return

        def key(d):
            try:
                return (0, int(sizes[d]))    # ascending
            except (TypeError, ValueError):
                return (1, 0)                # symbolic = large: sort last
        n = len(sizes)
        front = sorted(range(n - 1), key=key)   # smallest first -> flattened z
        order = front + [n - 1]                 # innermost dim pinned last
        if order == list(range(n)):
            return
        entry.map.params = [entry.map.params[d] for d in order]
        entry.map.range = dace.subsets.Range(
            [entry.map.range.ranges[d] for d in order])

    # 1. GPU_Device on the outermost map of every top-level state; everything
    #    else Sequential = per-thread loops inside the kernel.  The recursion
    #    matters: mapify+inline can leave a NestedSDFG holding the inner
    #    (reduction/elementwise) loops, and a Default-scheduled map there
    #    becomes a 1-thread thread-block map in CUDA codegen — the loop then
    #    runs ONE iteration per thread (the conv kernel's dyn_pres_K lost all
    #    but the first momentum term this way).
    all_states = list(sdfg.all_states())
    for state in all_states:
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and sdict.get(node) is None:
                _reorder_map_dims(state, node)
                node.schedule = dace.dtypes.ScheduleType.GPU_Device
                if block_size:
                    node.gpu_block_size = tuple(block_size)
    for state in all_states:
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and sdict.get(node) is not None:
                node.schedule = dace.dtypes.ScheduleType.Sequential
    from dace.sdfg import nodes as _nodes

    def _seq_nested(sdfg):
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, _nodes.NestedSDFG):
                    for nstate in node.sdfg.all_states():
                        for nnode in nstate.nodes():
                            if isinstance(nnode, nodes.MapEntry):
                                nnode.schedule = dace.dtypes.ScheduleType.Sequential
                    _seq_nested(node.sdfg)
    _seq_nested(sdfg)

    # 2. Device-resident storage for real arrays; scalars stay host.
    #
    # Then DEMOTE whatever the validator refuses, using it as the oracle.  Promoting every array to
    # GPU_Global is wrong whenever an array is also touched by HOST code, and there is more than one
    # way to be host-touched:
    #
    #   InvalidSDFGInterstateEdgeError: ... "kpbl" (StorageType.GPU_Global) in host code interstate
    #       edge          -- a loop bound / branch condition (`do k = kpbl(i,j), kte`)
    #   InvalidSDFGEdgeError: ... "brcr" is stored as StorageType.GPU_Global but accessed on host
    #       (at state if_3) -- a scalar read in a host tasklet or a dynamic map range
    #
    # Hand-copying the validator's notion of "host" would be a second implementation of a rule that
    # already exists, and the two would drift.  So ask it: promote everything, validate, and demote
    # the container it names, until it is satisfied.  Correct by construction, and it terminates --
    # each round removes one container from GPU_Global, and the set is finite.
    #
    # The cost is real and deliberate: a demoted array is host-resident, so DaCe inserts the
    # host<->device movement at each use.  MEASURED on ysu_prod, because the obvious guess is wrong:
    # all 83 demotions are the "accessed on host" kind, on 2-D (i,j) arrays (kpbl, brcr, hpbl, ...)
    # read as scalars -- host-scheduled tasklets the lowering did not mapify -- and NOT the
    # data-dependent loop bounds the interstate-edge arm exists for.  Rewriting the slice's
    # `do k = kpbl(i,j), kte` as a masked static range moves the count by zero (83 -> 83).  So this
    # demotion is the price of partial mapification in the lowering; the fix lives there, not in
    # the slice.
    n_dev = 0
    promoted = []
    for name, arr in sdfg.arrays.items():
        if arr.transient:
            continue
        if isinstance(arr, dace.data.Scalar):
            continue
        try:
            is_len1 = all(int(s) == 1 for s in arr.shape)
        except (TypeError, ValueError):  # symbolic extents -> real array
            is_len1 = False
        if not is_len1:
            arr.storage = dace.dtypes.StorageType.GPU_Global
            promoted.append(name)
            n_dev += 1

    from dace.sdfg.validation import InvalidSDFGEdgeError, InvalidSDFGInterstateEdgeError
    demoted = []
    for _ in range(len(promoted) + 1):
        try:
            sdfg.validate()
            break
        except (InvalidSDFGEdgeError, InvalidSDFGInterstateEdgeError) as err:
            # `.message`, never `str(err)`: the exception's __str__ can itself raise (it indexes
            # a state's edges by a region id), so formatting is not a safe way to read a message.
            msg = getattr(err, "message", "") or ""
            m = re.search(r'data container "([^"]+)"|Data container "([^"]+)"', msg)
            culprit = (m.group(1) or m.group(2)) if m else None
            if culprit is None or culprit not in sdfg.arrays or culprit in demoted:
                raise
            sdfg.arrays[culprit].storage = dace.dtypes.StorageType.Default
            demoted.append(culprit)
            n_dev -= 1
    else:
        raise RuntimeError("offload: could not reach a valid graph by demoting containers")
    if demoted:
        print(f"[dace_fortran.offload] demoted {len(demoted)} array(s) to host storage: {demoted}")

    # 3. Scheduling validation: at least one GPU map, all GPU maps have device arrays.
    n_gpu = sum(1 for st in sdfg.all_states() for nd in st.nodes()
                if isinstance(nd, nodes.MapEntry) and nd.schedule == dace.dtypes.ScheduleType.GPU_Device)
    if n_gpu == 0:
        # Loud failure, deliberately: the alternative is the silent restructure the permissive
        # LoopToMap escalation used to do (see `_mapify`).  When the conservative check refused every
        # loop, the body may actually be mappable -- but proving that needs the permissive path and
        # a per-kernel bit-exact differential, so it is the caller's explicit choice, not ours.
        hint = ("" if permissive_ltm else
                "; the conservative LoopToMap mapped nothing and the permissive escalation is OFF "
                "-- if this kernel's loop bodies are refused by the conservative affine-write check, "
                "opt in per kernel with permissive_ltm=True AND pay for it with that kernel's "
                "bit-exact differential")
        raise RuntimeError("offload_device_resident: no top-level map found to schedule" + hint)
    return sdfg, n_gpu, n_dev
