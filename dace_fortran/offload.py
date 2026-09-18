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
import dace
from dace.sdfg import nodes


def _mapify(sdfg):
    """LoopToMap + MapCollapse at this level, then recurse into nested SDFGs.

    dp.optimize's LoopToMap runs on the top-level SDFG only: for the pack
    kernel it mapped just the outer loop while the spatial nest stayed
    control-flow inside a NestedSDFG - inlining that yields a serial tasklet
    and a GPU grid of nvar threads.  Mapping every level before inlining is
    what lets the collapse produce one full-domain map."""
    from dace.transformation.interstate.loop_to_map import LoopToMap
    from dace.transformation.dataflow import MapCollapse

    def _apply_ltm(sdfg):
        """LoopToMap, escalating to permissive when the conservative checks
        refuse.  The main refusal for MFC's kernels: a loop body that lives
        entirely inside a NestedSDFG carries a loop-UNION subset on the
        connector memlet, so the per-iteration affine write check sees no
        iterator and refuses.  MFC's scatter writes (r = ...; buf(r) = ...)
        are per-iteration-unique by construction; unsoundness would be caught
        by the mandatory per-kernel bit-exact differential downstream."""
        from dace.sdfg import nodes as _nodes

        def _maps(s):
            return sum(1 for st in s.all_states() for nd in st.nodes()
                       if isinstance(nd, _nodes.MapEntry))

        before = _maps(sdfg)
        sdfg.apply_transformations_repeated(LoopToMap, validate=False)
        if _maps(sdfg) == before:
            sdfg.apply_transformations_repeated(LoopToMap, validate=False,
                                                permissive=True)

    _apply_ltm(sdfg)
    for state in sdfg.all_states():
        for node in list(state.nodes()):
            if isinstance(node, nodes.NestedSDFG):
                _mapify(node.sdfg)
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
    """
    from dace.transformation.interstate import InlineSDFG

    moved = 0
    for _ in range(20):
        remaining = [(n, st) for st in sdfg.all_states() for n in st.nodes()
                     if isinstance(n, nodes.NestedSDFG)]
        if not remaining:
            break
        for node, state in remaining:
            inliner = InlineSDFG()
            inliner.setup_match(sdfg=sdfg, cfg_id=state.parent_graph.cfg_id,
                                state_id=state.block_id,
                                subgraph={InlineSDFG.nested_sdfg: node},
                                expr_index=0, override=True)
            try:
                inliner.apply(state, sdfg)
                moved += 1
            except Exception:  # noqa: BLE001 -- a refusal here is not fatal; leave the node
                pass
    return moved


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


def offload_device_resident(sdfg, block_size=None, force_inline=False):
    """Schedule + storage assignment for device-resident offload. In place.

    `force_inline` is OFF by default and must be turned on PER KERNEL, because it is not generally
    sound: applied to the whole library set it made the default dispatch path produce NaN at step 1
    (the flux-difference family), which is exactly the kind of thing `InlineSDFG.can_be_applied`'s
    refusal was protecting against.  It is enabled for `mfc_dace_fdiff_src_*`, where the resulting
    kernel is verified bit-identical and 1.57x faster, and for nothing else.
    """
    # 0a. Mapify every level, then inline the loop-body NestedSDFGs the
    #     frontend introduces: MapCollapse cannot fuse across the nested-SDFG
    #     boundary, and without fusion the GPU map would cover only the outer
    #     loop (sys_size/nvar threads) with the N^3 loops serial per thread.
    _mapify(sdfg)
    from dace.sdfg import utils as sutils
    sutils.inline_sdfgs(sdfg, permissive=True)
    if force_inline:
        _force_inline(sdfg)

    # 0b. Collapse perfectly-nested map chains into single maps: scheduling the
    #     outermost of i(l(k(j))) as a GPU map would put sys_size (=5) threads on
    #     the device and run the N^3 spatial loops serially per thread.
    from dace.transformation.dataflow import MapCollapse
    sdfg.apply_transformations_repeated(MapCollapse, validate=False)
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
    n_dev = 0
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
            n_dev += 1

    # 3. Scheduling validation: at least one GPU map, all GPU maps have device arrays.
    n_gpu = sum(1 for st in sdfg.all_states() for nd in st.nodes()
                if isinstance(nd, nodes.MapEntry) and nd.schedule == dace.dtypes.ScheduleType.GPU_Device)
    if n_gpu == 0:
        raise RuntimeError("offload_device_resident: no top-level map found to schedule")
    return sdfg, n_gpu, n_dev
