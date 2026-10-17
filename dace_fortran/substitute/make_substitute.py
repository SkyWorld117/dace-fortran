"""Build the DaCe substitute of one routine: compile its SDFG and emit the Fortran binding (production contract).

Usage: python -m dace_fortran.substitute.make_substitute [--config <file.toml>] [--gpu]
           [--record <file> | --specialize <file>] <routine.hlfir> <module::routine> <out dir>

Writes <out dir>/<routine>_bindings.f90 and <out dir>/substitute.json: the SDFG library, the binding, and what the
link step needs to put the substitute in place of the routine -- the routine's symbol (nvfortran mangles
``module::routine`` as ``module_routine_``) and the substitute entry, an external subroutine ``dace_sub_<routine>``
with the routine's own interface that calls the binding, and runs the original routine (``dace_original_<routine>``)
when the binding declines the call (generated, or hand-written with --shim).

--config <file.toml>: defaults of these options from the file's [make_substitute] table (substitute.config).

--record <file>: the binding appends the integer symbols and scalar inputs of every call to <file>.  --specialize <file>:
the SDFG is specialised to the values that stayed the same in every call recorded in <file> (and that the binding can
check before it does any work); the binding declines calls with other values.  The specialised element loops unroll,
and an array of records they then index by constants is handed over member by member instead of packed.

--gpu: the caller keeps its arrays on the GPU (an OpenACC build).  The SDFG is parallelised (dace-fortran's
optimize, then the nested maps of a loop nest are collapsed and their parameters ordered for coalesced access), every
argument array a map reads or writes is moved
to the GPU, and DaCe's GPU transformation turns the maps into CUDA kernels; the binding hands the SDFG the caller's
device copies (dace_fortran.bindings.acc_transfers.plan_device_resident_caller).
"""

import argparse
import functools
import json
from dataclasses import replace
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import dace
from dace.sdfg import infer_types, nodes
from dace.transformation.dataflow import MapCollapse, MapDimShuffle
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll
from dace.transformation.passes.persistent_transients import MakeTransientsPersistent

from dace_fortran.bindings import FlattenPlan, emit_bindings
from dace_fortran.bindings.acc_transfers import plan_device_resident_caller
from dace_fortran.bindings.block_builders import unsourced_array_args
from dace_fortran.bindings.build_fortran_library import compiled_init_symbols
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.frozen_signature import attach_to_sdfg, drop_unread_module_inputs, get_frozen_signature, refreeze
from dace_fortran.bindings.record_split import split_record_companions
from dace_fortran.bindings.specialization import Specialization, recorded_aliases, recorded_invariants, specializable
from dace_fortran.builder import SDFGBuilder
from dace_fortran.external import keep_external
from dace_fortran import pipelines as P
from dace_fortran.pipelines import num_maps, specialize
from dace_fortran.substitute.config import parse_with_config
from dace_fortran.substitute.module_copies import access_entries, module_files, write_access_copies
from dace_fortran.substitute.wcr_accumulations import accumulations_to_wcr

#: Threads per block of the GPU kernels (DaCe's default is 32).
GPU_BLOCK_SIZE = [128, 1, 1]
FUSED_BLOCK_SIZE = [64, 1, 1]  # a fused stencil kernel: long per-thread work, 116 registers
#: --heavy-launch: a kernel with a long straight-line body is latency-bound with few warps resident; 32 threads along
#: the contiguous dimension (coalesced, little tail waste on extents that are not multiples of 128) by 4 along the
#: next, and room for 4 such blocks per multiprocessor (at most 128 registers per thread). On a fused fifth-order
#: reconstruction + Riemann solver kernel: 5.3 -> 4.1 ms (166 registers before, some spills after); on a third-order
#: one (120 registers) 2.3-2.4 -> 2.1-2.2 ms, from the block shape alone.
HEAVY_BLOCK_SIZE = [32, 4, 1]
#: Their __launch_bounds__, "<threads>, <blocks per multiprocessor>" (--heavy-launch-bounds).
HEAVY_LAUNCH_BOUNDS = "128, 4"
#: The CUDA architecture the kernels are compiled for (--cuda-arch).
CUDA_ARCH = "90"
HEAVY_LAUNCH_TASKLETS = 200


def set_heavy_launch(gpu_map) -> None:
    """Launch settings of a heavy kernel (see HEAVY_BLOCK_SIZE); a one-dimensional map keeps a flat block."""
    threads = HEAVY_BLOCK_SIZE[0] * HEAVY_BLOCK_SIZE[1] * HEAVY_BLOCK_SIZE[2]
    gpu_map.gpu_block_size = list(HEAVY_BLOCK_SIZE) if len(gpu_map.params) >= 2 else [threads, 1, 1]
    gpu_map.gpu_launch_bounds = HEAVY_LAUNCH_BOUNDS


def explicit_block_threads(scope_nodes) -> int:
    """The threads of a kernel's block when its thread-block maps define it: per dimension the largest of their
    extents (over-approximated like the CUDA code generator does), multiplied."""
    dims: list[int] = []
    for n in scope_nodes:
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_ThreadBlock:
            sizes = [int(dace.symbolic.overapproximate(s)) for s in n.map.range.size()][::-1]  # last param: x
            dims = [max(a, b) for a, b in zip(dims, sizes)] + (dims[len(sizes):] or sizes[len(dims):])
    threads = 1
    for d in dims:
        threads *= d
    return threads


def arrays_used_by_maps(sdfg: dace.SDFG) -> set[str]:
    """Top-level data containers some map reads or writes: the data of every memlet entering or leaving a top-level
    map scope, in any state of the outermost SDFG."""
    used = set()
    for state in sdfg.states():
        sd = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and sd[node] is None:
                used.update(e.data.data for e in state.in_edges(node) if e.data.data)
                used.update(e.data.data for e in state.out_edges(state.exit_node(node)) if e.data.data)
    return used


def arrays_used_by_free_tasklets(sdfg: dace.SDFG) -> set[str]:
    """Top-level data containers a tasklet outside every map reads or writes: the GPU transformation wraps such a
    tasklet in a kernel of its own, so its arrays must live on the device like a map's."""
    used = set()
    for state in sdfg.states():
        sd = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet) and sd[node] is None:
                used.update(e.data.data for e in state.in_edges(node) if e.data.data)
                used.update(e.data.data for e in state.out_edges(node) if e.data.data)
    return used


def _stride_one_subscripts(state: dace.SDFGState, entry: nodes.MapEntry) -> Counter:
    """Per map parameter, the memlets in the scope of ``entry`` that subscript an array's stride-1 dimension with it,
    weighted by the array's size; nested SDFGs included (their symbols are traced back to the parameters through
    the symbol mappings)."""
    counts = Counter()

    def weight(desc) -> int:
        # A subscript into a big field array matters; one into a small table of coefficients (cached) does not, and
        # a handful of tables read many times must not outvote the fields.
        try:
            return max(int(desc.total_size), 1)
        except (TypeError, ValueError):
            return 1 << 30  # a symbolic size: a field

    def count(sdfg: dace.SDFG, edges, origin: dict[str, set[str]]) -> None:
        for e in edges:
            m = e.data
            if m.data is None or m.subset is None or m.data not in sdfg.arrays:
                continue
            desc = sdfg.arrays[m.data]
            unit = [d for d, stride in enumerate(desc.strides) if stride == 1]
            if not unit:
                continue
            begin = m.subset[unit[0]][0]
            if isinstance(begin, dace.symbolic.SymExpr):  # a strip-mined range: its exact expression
                begin = begin.expr
            for sym in getattr(begin, "free_symbols", ()):
                for p in origin.get(str(sym), ()):
                    counts[p] += weight(desc)

    def nested(node: nodes.NestedSDFG, origin: dict[str, set[str]]) -> None:
        inner = {}
        for sym, value in node.symbol_mapping.items():
            free = getattr(value, "free_symbols", ())
            inner[sym] = set().union(*(origin.get(str(f), ()) for f in free)) if free else set()
        for st in node.sdfg.states():
            count(node.sdfg, st.edges(), inner)
            for n in st.nodes():
                if isinstance(n, nodes.NestedSDFG):
                    nested(n, inner)

    params = {p: {p} for p in entry.map.params}
    scope = state.scope_subgraph(entry)
    count(state.sdfg, scope.edges(), params)
    for node in scope.nodes():
        if isinstance(node, nodes.NestedSDFG):
            nested(node, params)
    return counts


def order_for_coalescing(sdfg: dace.SDFG) -> int:
    """Reorder each top-level map's parameters so the one that subscripts the most stride-1 dimensions comes last:
    DaCe's GPU code generation puts the last parameter on the thread x dimension, and a loop nest whose innermost loop
    runs over a few variables would otherwise leave most threads of a block idle and read memory with a stride.
    Returns the number of maps reordered."""
    reordered = 0
    for state in sdfg.states():
        scopes = state.scope_dict()
        for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scopes[n] is None]:
            params = list(entry.map.params)
            counts = _stride_one_subscripts(state, entry)
            # Stable: parameters with equally many stride-1 subscripts keep their order.
            order = sorted(params, key=lambda p: counts.get(p, 0))
            if order != params:
                MapDimShuffle.apply_to(sdfg, map_entry=entry, options={"parameters": order}, save=False)
                reordered += 1
    return reordered


#: Loops of at most this many iterations are unrolled once the SDFG is specialised: the loops over the elements of an
#: array of records, whose members are then handed over one by one.  It must cover the largest system (MHD has 9
#: equations: with the limit at 8 its element loops stayed loops, q_cons_vf/rhs_vf were packed on every call and the
#: reconstruction bodies kept a sequential loop over the equations); heavy loops become maps before the unrolling
#: (``map_heavy_short_loops``), so only light bodies are multiplied.
UNROLL_LIMIT = 16


#: --startup-constants: where code that may run after the routine's first call changes module variables
#: (startup_constants.StartupAnalysis); None: every recorded value counts.
STARTUP = None


def _report_startup(kept: dict, dropped: dict, what: str) -> None:
    print(f"start-up constants: {len(kept)} of {len(kept) + len(dropped)} {what} kept", flush=True)
    for name, why in sorted(dropped.items()):
        more = f" (+{len(why) - 1} more)" if len(why) > 1 else ""
        print(f"  may change after the first call: {name}: {why[0]}{more}", flush=True)


def startup_constant(name: str) -> bool:
    """Whether the recorded module value ``module::variable...`` keeps its value after the routine's first call."""
    if STARTUP is None:
        return True
    module, var = name.split("::", 1)
    return not STARTUP.changes((module, re.split(r"[(%]", var)[0]))


def keep_startup_values(values: dict, what: str) -> dict:
    """The entries of ``values`` (``module::variable...`` -> value) whose variable keeps its value after the routine's
    first call (all of them without --startup-constants)."""
    if STARTUP is None:
        return values
    kept, dropped = {}, {}
    for name, value in values.items():
        module, var = name.split("::", 1)
        why = STARTUP.changes((module, re.split(r"[(%]", var)[0]))
        if why:
            dropped[name] = why
        else:
            kept[name] = value
    _report_startup(kept, dropped, what)
    return kept


def keep_startup_symbols(recorded: dict, frozen, iface, plan: FlattenPlan) -> dict:
    """The recorded symbols and scalar inputs (``--specialize``) that keep their values after the routine's first call:
    one the binding computes from a module variable (read through its import alias ``<name>__mod``) that code may
    change later is dropped; one computed from the routine's own arguments stays (the recorded calls decide it)."""
    from dace_fortran.bindings.block_builders import effective_module_sources  # noqa: PLC0415
    from dace_fortran.bindings.specialization import _guard_assigns, _scalar_sources  # noqa: PLC0415

    sources = effective_module_sources(frozen, iface, plan)
    scalars = _scalar_sources(frozen, iface, plan)
    assigns = _guard_assigns(frozen, iface, plan, [n for n in recorded if n not in scalars])
    kept, dropped = {}, {}
    for name, value in recorded.items():
        text = scalars[name][0] if name in scalars else "\n".join(assigns.get(name, []))
        origins = {sources[name]} if name in sources else set()
        for tok in re.findall(r"[a-z_]\w*", text.lower()):
            base = tok[: -len("__mod")] if tok.endswith("__mod") else tok
            if base in sources:
                origins.add(sources[base])
        why = [w for o in sorted(origins) for w in STARTUP.changes((o.module, o.entity))]
        if why:
            dropped[name] = why
        else:
            kept[name] = value
    _report_startup(kept, dropped, "recorded symbols and scalars")
    return kept


def merged_invariants(records: list[Path]) -> dict:
    """The values that stayed the same in every call of every record of ``records`` (successive recording rounds of
    one run: a later round's recorder was pruned by the earlier values and records the rest); a name the records
    disagree on is no invariant."""
    merged: dict = {}
    conflicting = set()
    for record in records:
        for name, value in recorded_invariants(record).items():
            if name in merged and merged[name] != value:
                conflicting.add(name)
            merged.setdefault(name, value)
    return {name: value for name, value in merged.items() if name not in conflicting}


def recorded_elements(records: list[Path], sdfg: dace.SDFG) -> dict[str, list]:
    """The recorded elements of module arrays of records (``module::name(k)%member value``) as the contents of their
    SDFG images (``name_member``), and those of module arrays of scalars (``module::name(k) value``) as the contents of
    the arrays themselves, for every one the SDFG takes whose every element the records have. Their reads in control
    flow (bounds symbols assigned from them, a branch on ``flags[d]`` once ``d`` is known) can then be folded."""
    found: dict[str, dict[int, object]] = {}
    plain: dict[str, dict[int, object]] = {}  # arrays of scalars, by their own (Fortran) subscripts
    lines = [line for record in records for line in Path(record).read_text().splitlines()]

    def number(value: str):
        return int(value) if re.fullmatch(r"-?\d+", value) else float(value)

    for line in lines:
        if not line.strip() or not startup_constant(line.split()[0]):
            continue
        m = re.match(r"\S+::(\w+)\((-?\d+)\)%(\w+)\s+(\S+)$", line)
        if m and f"{m.group(1)}_{m.group(3)}" in sdfg.arrays:
            found.setdefault(f"{m.group(1)}_{m.group(3)}", {})[int(m.group(2)) - 1] = number(m.group(4))
            continue
        m = re.match(r"\S+::(\w+)\((-?\d+)\)\s+(\S+)$", line)
        if m and m.group(1) in sdfg.arrays and not sdfg.arrays[m.group(1)].transient:
            plain.setdefault(m.group(1), {})[int(m.group(2))] = number(m.group(3))
    # The SDFG numbers an array of scalars from 0 at its lower bound: the lowest subscript recorded, once every element
    # is (a recorder records all elements of an array read at a runtime subscript).
    for name, by_subscript in plain.items():
        lower = min(by_subscript)
        found[name] = {k - lower: v for k, v in by_subscript.items()}
    elements = {}
    for name, by_index in found.items():
        desc = sdfg.arrays[name]
        try:
            size = int(desc.shape[0])
        except TypeError:
            continue
        if len(desc.shape) == 1 and all(i in by_index for i in range(size)):
            elements[name] = [by_index[i] for i in range(size)]
    return elements


def make_scratch_transient(sdfg: dace.SDFG, names: list[str], values: dict | None = None) -> tuple[list[str], dict[str, dict]]:
    """Turn the module arrays ``names`` (``module::variable``) into transients of ``sdfg``: state the routine only
    uses as scratch within one call need not come from, or go back to, the caller. A variable may be a module array
    (matched by its origin) or a module record whose packed image the SDFG takes (matched by the record the image
    was built from); an image's shape and offset symbols, which the binding would otherwise measure on the record's
    members, take their recorded ``values``: returned as the second item, to be folded once the transformations that
    propagate memlets are done (a literal shape makes DaCe check the nested connectors it tolerates while symbolic).
    An image leaves the frozen signature here: it is no longer an argument the binding packs. Returns the SDFG names
    and the symbol values to fold."""
    frozen = get_frozen_signature(sdfg)
    wanted = {n.lower() for n in names}
    found, done, images = set(), [], {}
    for sdfg_name, origin in frozen.module_symbol_origins.items():
        key = f"{origin.module}::{origin.entity}".lower()
        if key in wanted and sdfg_name in sdfg.arrays:
            found.add(key)
            sdfg.arrays[sdfg_name].transient = True
            done.append(sdfg_name)
    for a in frozen.args:
        struct = getattr(a, "aos_origin_struct", None)
        if not struct or a.sdfg_name not in sdfg.arrays:
            continue
        entity = struct.split("%")[0].strip().lower()
        module = (getattr(a, "aos_origin_mod", "") or "").lower()
        key = next((n for n in wanted if n == f"{module}::{entity}" or (not module and n.split("::")[-1] == entity)),
                   None)
        if key is None:
            continue
        found.add(key)
        desc = sdfg.arrays[a.sdfg_name]
        # Deferred: an image stays an ordinary array while the stencils fuse and the copies forward (its symbolic
        # shape is what those tolerate), see :func:`finish_scratch_images`.
        images[a.sdfg_name] = image_symbols(a.sdfg_name, desc, values or {})
    missing = sorted(wanted - found)
    if missing:
        # One argument list serves every configuration: an array a configuration does not use is not an input here.
        print(f"--scratch: not module inputs of this SDFG (skipped): {missing}", flush=True)
    return done, images


def make_unsourced_transient(sdfg: dace.SDFG, frozen, iface, plan: FlattenPlan) -> list[str]:
    """Turn the arrays ``sdfg`` takes that no caller supplies (the data of an inlined callee's optional argument that
    a call leaves absent: the binding would hand over a zeroed local and drop it again) into transients. As
    arguments, the copies the bridge stages into and out of them make a loop's private scratch look shared
    (2D_lagrange_bubblescreen: the pressure corrector's Romega kept its bubble loop sequential). They leave the frozen
    signature here: the binding no longer hands them over. Returns their names."""
    names = [a.sdfg_name for a in unsourced_array_args(frozen, iface, plan) if a.sdfg_name in sdfg.arrays]
    for name in names:
        sdfg.arrays[name].transient = True
    attach_to_sdfg(sdfg, replace(frozen, args=tuple(a for a in frozen.args if a.sdfg_name not in names)))
    return names


def image_symbols(name: str, desc, values: dict) -> dict:
    """The recorded ``values`` of a record image's extents and lower bounds: the descriptor's symbols, and the offset
    symbols the memlets subtract (the lower bounds live in the memlets, not in the descriptor). An image made a
    transient takes them (:func:`finish_scratch_images`)."""
    prefixes = (f"{name}_d", f"offset_{name}_d")
    out = {k: v for k, v in values.items() if k.startswith(prefixes)}
    out.update({str(sym): values[str(sym)] for sym in desc.free_symbols if str(sym) in values})
    return out


def finish_scratch_images(sdfg: dace.SDFG, images: dict[str, dict]) -> None:
    """Make the packed record images ``images`` (name -> recorded shape and offset symbol values) transients: they
    leave the frozen signature (no longer arguments the binding packs), their symbols take the recorded values, and
    they are laid out with the field dimensions contiguous."""
    if not images:
        return
    frozen = get_frozen_signature(sdfg)
    attach_to_sdfg(sdfg, replace(frozen, args=tuple(a for a in frozen.args if a.sdfg_name not in images)))
    fold = {}
    for name, values in images.items():
        sdfg.arrays[name].transient = True
        fold.update(values)
    if fold:
        specialize(sdfg, symbols=fold)
    print(f"scratch images made transient: {sorted(images)}; symbols fixed: {fold}", flush=True)
    print(f"scratch arrays laid out with the field dimensions contiguous: {relayout_field_last(sdfg, sorted(images))}", flush=True)



def relayout_field_last(sdfg: dace.SDFG, names: list[str], small: int = 16) -> dict[str, tuple]:
    """Lay the transient arrays ``names`` out with their small dimensions (a record's index, a component) slowest:
    a thread then steps through a field dimension contiguously instead of jumping over every component. Applied
    to the descriptor and to every nested descriptor connected to it. Returns the new strides per array."""
    done = {}
    for name in names:
        desc = sdfg.arrays[name]
        try:
            extents = [int(x) for x in desc.shape]
        except (TypeError, ValueError):
            continue  # a symbolic shape: left alone
        order = [d for d, x in enumerate(extents) if x > small] + [d for d, x in enumerate(extents) if x <= small]
        if order == list(range(len(extents))):
            continue
        strides = [0] * len(extents)
        stride = 1
        for d in order:
            strides[d] = stride
            stride *= extents[d]
        new = tuple(strides)

        def apply(container: dace.SDFG, array: str) -> None:
            container.arrays[array].strides = new
            container.arrays[array].total_size = stride
            for st in container.all_states():
                for n in st.nodes():
                    if isinstance(n, nodes.NestedSDFG):
                        for e in list(st.in_edges(n)) + list(st.out_edges(n)):
                            conn = e.dst_conn if e.dst is n else e.src_conn
                            if e.data.data == array and conn and conn in n.sdfg.arrays \
                                    and len(n.sdfg.arrays[conn].shape) == len(extents):
                                apply(n.sdfg, conn)

        apply(sdfg, name)
        done[name] = new
    return done

def body_tasklets(region) -> int:
    """Tasklets in a control-flow region, nested SDFGs included."""
    n = 0
    for st in region.all_states():
        for node in st.nodes():
            if isinstance(node, nodes.Tasklet):
                n += 1
            elif isinstance(node, nodes.NestedSDFG):
                n += sum(1 for x, _ in node.sdfg.all_nodes_recursive() if isinstance(x, nodes.Tasklet))
    return n


def heavy_innermost_loop(min_tasklets: int, limit: int = UNROLL_LIMIT):
    """ShortLoopUnroll ``keep`` predicate: a loop whose body holds at least ``min_tasklets`` tasklets and whose own
    inner loops are all short constant-trip ones (unrolled with the rest: a reconstruction's loop over the variables
    holds the stencil loops), for :func:`map_heavy_short_loops` to make a map."""
    from dace.sdfg.state import LoopRegion  # noqa: PLC0415
    from dace.transformation.passes.parallelization_prep import _constant_trip_count  # noqa: PLC0415

    def keep(loop, sdfg) -> bool:
        if body_tasklets(loop) < min_tasklets:
            return False
        for r in loop.all_control_flow_regions():
            if isinstance(r, LoopRegion) and r is not loop:
                trip = _constant_trip_count(r, sdfg)
                if trip is None or trip > limit:
                    return False
        return True

    return keep


def specialize_and_split(sdfg: dace.SDFG, plan: FlattenPlan, symbols: dict, scalars: dict,
                         keep_heavy_loops: int = 0) -> dict[str, dict[int, str]]:
    """Specialise ``sdfg`` and split the arrays of records its element loops then index by constants.

    Constant propagation and dead-branch elimination come first: a loop bound computed from a specialised value is a
    literal only after them, and indices into parts the case does not have sit in branches it never takes.
    ``keep_heavy_loops``: innermost short loops with bodies of at least this many tasklets stay loops (0: none).
    """
    specialize(sdfg, symbols=symbols, scalars=scalars)
    # Until the dead branches are gone, accesses in them may fall outside arrays the case left degenerate (a
    # boundary condition's coefficients for a direction without it): validate once they are.
    sdfg.simplify(validate=False, validate_all=False)
    if DUMP_DIR != Path("."):  # the tool's output directory (not a test's working directory)
        # The specialised SDFG with its short loops still loops: the input for designing which of them become maps.
        sdfg.save(str(DUMP_DIR / "pre_unroll.sdfgz"), compress=True)
    keep = heavy_innermost_loop(keep_heavy_loops) if keep_heavy_loops else None
    unrolled = ShortLoopUnroll(UNROLL_LIMIT, keep=keep).apply_pass(sdfg, {})
    if keep is not None:
        from dace.sdfg.state import LoopRegion  # noqa: PLC0415
        from dace.transformation.passes.parallelization_prep import _constant_trip_count  # noqa: PLC0415

        kept = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)
                and (_constant_trip_count(r, r.sdfg) or UNROLL_LIMIT + 1) <= UNROLL_LIMIT and keep(r, sdfg)]
        print(f"short loops unrolled before splitting: {unrolled}; heavy innermost loops kept: "
              f"{[(r.label[:40], r.loop_variable) for r in kept][:12]}", flush=True)
    sdfg.simplify(validate=False, validate_all=False)
    sdfg.validate()
    return split_record_companions(sdfg, plan)


def map_heavy_short_loops(sdfg: dace.SDFG, min_tasklets: int, limit: int = UNROLL_LIMIT) -> int:
    """Turn every parallel constant-trip loop of at most ``limit`` iterations whose body holds at least
    ``min_tasklets`` tasklets into a map, before short loops are unrolled: unrolled, such a loop (a loop over the
    variables of a system around a whole reconstruction, say) multiplies the work and the registers of one thread
    by its trip count; as a map dimension it gives each iteration its own thread. Returns the number converted."""
    from dace.sdfg.state import LoopRegion
    from dace.transformation.passes.parallelization_prep import _constant_trip_count

    tasklets = body_tasklets

    # Report the heavy loops and why each stays a loop, once.
    for sd in sdfg.all_sdfgs_recursive():
        for loop in [r for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)]:
            n = tasklets(loop)
            if n < min_tasklets:
                continue
            trip = _constant_trip_count(loop, sd)
            ok = P.LoopToMap.can_be_applied_to(loop.sdfg, loop=loop) if trip is not None and trip <= limit else None
            print(f"  heavy loop {loop.label[:50]} var {loop.loop_variable}: trip {trip}, {n} tasklets, "
                  f"loop-to-map {ok}", flush=True)
    converted = 0
    changed = True
    while changed:
        changed = False
        for sd in sdfg.all_sdfgs_recursive():
            for loop in [r for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                trip = _constant_trip_count(loop, sd)
                if trip is None or trip > limit or tasklets(loop) < min_tasklets:
                    continue
                if P.LoopToMap.can_be_applied_to(loop.sdfg, loop=loop):
                    P.LoopToMap.apply_to(loop.sdfg, loop=loop, verify=False)
                    converted += 1
                    changed = True
                    break
            if changed:
                break
    return converted


def collapse_then_fuse(sdfg: dace.SDFG, map_heavy_loops: int = 0) -> int:
    """dace_fortran.pipelines.optimize with MapCollapse before map fusion: fusing first can fuse two loop nests over
    their shared outer dimension only (two collapsed loop nests of a Riemann solver over l), after which neither
    collapses and the kernel runs one thread per l.  ``map_heavy_loops``: see :func:`map_heavy_short_loops` (0 off)."""
    P.specialize(sdfg)
    P.ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})
    if map_heavy_loops:
        P.UniqueLoopIterators().apply_pass(sdfg, {})
        P.fission_scalars(sdfg)
        sdfg.simplify()
        P.privatize_loop_local_arrays(sdfg)
        print(f"heavy short loops made maps: {map_heavy_short_loops(sdfg, map_heavy_loops)}", flush=True)
    P.ShortLoopUnroll(UNROLL_LIMIT).apply_pass(sdfg, {})
    P.UniqueLoopIterators().apply_pass(sdfg, {})
    P.fission_scalars(sdfg)
    sdfg.simplify()
    sdfg.apply_transformations_repeated(P.StateFusionExtended)
    P.privatize_loop_local_arrays(sdfg)
    # Accumulations into the cells on both sides of a face (IGR's Riemann solvers, atomic updates in the source)
    # become atomic sums, so their loops can map.
    print(f"accumulations made atomic sums: {len(accumulations_to_wcr(sdfg))}", flush=True)
    sdfg.apply_transformations_repeated(P.LoopToMap)
    sdfg.apply_transformations_repeated(P.StateFusionExtended)
    sdfg.simplify()
    collapses = sdfg.apply_transformations_repeated(MapCollapse)
    P.FullMapFusion().apply_pass(sdfg, {})
    collapses += sdfg.apply_transformations_repeated(MapCollapse)
    MakeTransientsPersistent().apply_pass(sdfg, {})
    return collapses


def host_read_arrays(sdfg: dace.SDFG) -> set[str]:
    """Non-transient arrays whose elements interstate edges read outside any map (host-side control flow), in the SDFG
    and in the nested SDFGs that sit outside maps."""
    found: set[str] = set()

    def walk(sd: dace.SDFG) -> None:
        # recursive=False: the regions of this SDFG only; DaCe's recursion would descend into every nested SDFG,
        # the bodies of kernels included, whose reads are device-side.
        for e in sd.all_interstate_edges(recursive=False):
            for m in e.data.get_read_memlets(sd.arrays):
                if m.data in sd.arrays and not sd.arrays[m.data].transient:
                    found.add(m.data)
        for st in sd.all_states():
            scopes = st.scope_dict()
            for n in st.nodes():
                if isinstance(n, nodes.NestedSDFG) and scopes[n] is None:
                    walk(n.sdfg)

    walk(sdfg)
    return found


def report_sequential_loops(sdfg: dace.SDFG) -> list[str]:
    """The loops the parallelisation left sequential, largest body first: on the GPU every tasklet of such a body
    becomes a kernel launch per iteration, so a loop over the domain here means the substitute is unusable."""
    from dace.sdfg.state import LoopRegion

    rows = []
    for region in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(region, LoopRegion):
            continue
        tasklets = sum(1 for st in region.all_states() for n in st.nodes() if isinstance(n, nodes.Tasklet))
        cond = region.loop_condition.as_string if region.loop_condition is not None else ""
        rows.append((tasklets, f"{region.label} ({region.loop_variable}: {cond}): {tasklets} tasklets"))
    rows.sort(reverse=True)
    if rows:
        print(f"WARNING: {len(rows)} loops stayed sequential; the largest:", flush=True)
        for _, row in rows[:6]:
            print(f"  {row}", flush=True)
    return [row for _, row in rows]



def order_components_sharing_data(sdfg: dace.SDFG) -> int:
    """Within each state, make the components that touch the same array, and have no path between them, run in the
    order they were created in (the program order the frontend emitted): a dependency edge (an empty memlet) from
    the earlier component's sink to the later one's source. Unconnected components are concurrent by the SDFG's
    semantics; a sequential backend hides that, one CUDA stream per component exposes it. Returns the edges added."""
    import networkx as nx

    added = 0
    for state in sdfg.all_states():
        g = nx.DiGraph()
        g.add_nodes_from(state.nodes())
        g.add_edges_from((e.src, e.dst) for e in state.edges())
        order = {n: i for i, n in enumerate(state.nodes())}
        components = sorted(
            (sorted(c, key=order.__getitem__) for c in nx.weakly_connected_components(g)), key=lambda c: order[c[0]]
        )
        if len(components) < 2:
            continue
        info = []
        for c in components:
            accessed = [n for n in c if isinstance(n, nodes.AccessNode)]
            reads = {n.data for n in accessed if state.out_degree(n) > 0}
            writes = {n.data for n in accessed if state.in_degree(n) > 0}
            sources = [n for n in c if state.in_degree(n) == 0]
            sinks = [n for n in c if state.out_degree(n) == 0]
            info.append((reads, writes, sources, sinks))
        for i, (ra, wa, _sa, sinks_a) in enumerate(info):
            for j in range(i + 1, len(info)):
                rb, wb, sources_b, _sb = info[j]
                if not ((wa & (rb | wb)) or (ra & wb)):
                    continue
                src = next((n for n in sinks_a if isinstance(n, nodes.AccessNode)), sinks_a[0])
                dst = next((n for n in sources_b if isinstance(n, nodes.AccessNode)), sources_b[0])
                if nx.has_path(g, src, dst):
                    continue
                state.add_nedge(src, dst, dace.Memlet())
                g.add_edge(src, dst)
                added += 1
    return added


def _scope_nodes_recursive(state: dace.SDFGState, entry: nodes.MapEntry):
    """The nodes in the scope of ``entry``, descending into the nested SDFGs it holds (a recomputed stencil sits in
    the consumer's body)."""
    for n in state.scope_subgraph(entry).nodes():
        yield n
        if isinstance(n, nodes.NestedSDFG):
            for inner, _parent in n.sdfg.all_nodes_recursive():
                yield inner

DUMP_DIR = Path(".")  # where the debugging dumps of the pipeline stages go (the output dir)


def kernel_body_states(sdfg: dace.SDFG) -> list[int]:
    """The number of states of every nested SDFG that is the body of a top-level map (a kernel body)."""
    counts = []
    for state in sdfg.states():
        scopes = state.scope_dict()
        for n in state.nodes():
            if isinstance(n, nodes.NestedSDFG) and scopes[n] is not None and scopes[scopes[n]] is None:
                counts.append(len(list(n.sdfg.states())))
    return counts


def straighten_bodies(sdfg: dace.SDFG) -> None:
    """Turn kernel bodies into straight-line code: unroll the small maps inside them (straight.py), split their small
    scratch arrays into scalars and fuse the states that reuse kept apart (scalarise.py)."""
    from dace_fortran.substitute.scalarise import scalarise_small_arrays
    from dace_fortran.substitute.straight import unroll_small_maps

    before = Counter(kernel_body_states(sdfg))
    print(f"small maps inside kernels unrolled: {unroll_small_maps(sdfg)}", flush=True)
    print(f"small scratch arrays split into scalars: {scalarise_small_arrays(sdfg)}", flush=True)
    after = Counter(kernel_body_states(sdfg))
    print(f"kernel bodies by number of states (states: bodies): before {dict(sorted(before.items()))}, "
          f"after {dict(sorted(after.items()))}", flush=True)


def to_gpu(sdfg: dace.SDFG, fuse: bool = False, forward: bool = False, elements: dict | None = None,
           scratch_images: dict[str, dict] | None = None, combine_divisions: bool = False,
           scalarise: bool = False, map_heavy_loops: int = 0, heavy_launch: int = 0,
           host_work_limit: int = 0, fuse_small: int = 0, tile_fuse: bool = False,
           members_first: bool = True, extents: dict | None = None, forward_skip: Iterable[str] = ()) -> list[str]:
    """Parallelise ``sdfg`` and turn it into CUDA kernels over device-resident argument arrays.

    ``members_first``: the buffers of module arrays of records the SDFG takes as arguments are laid out with the
    member (field) dimensions fastest instead of the record indices (``record_layout.lay_out_members_first``), so that
    the threads of a kernel read neighbouring points from neighbouring addresses.  Returns the buffers laid out that
    way, for ``emit_bindings(members_first=...)``.

    ``host_work_limit``: maps and free tasklets over arrays of at most this many elements that no kernel touches
    stay on the host (small_kernels.host_small_work); ``fuse_small``: fuse the states around maps of at most this
    many iterations and those maps with each other (small_kernels.fuse_small_maps); 0: off.

    ``fuse``: recompute every stencil producer inside its consumer (fuse_stencils.fuse_stencil_pairs); ``tile_fuse``:
    fuse through a shared-memory tile instead (the producer once per tile, explicit thread-block maps); ``forward``:
    drop the copies that only mirror an argument array in a scratch one (fuse_stencils.forward_copies);
    ``elements``: recorded constant elements of arrays (fuse_stencils.fold_control_flow_constants), folded again once
    the loops are unrolled: a loop body that subscripts them with the loop variable does so with constants only
    then, and the states of a producer and its consumer fuse only once the bounds between them are constants."""
    t0 = time.time()
    collapses = collapse_then_fuse(sdfg, map_heavy_loops)
    fused_maps = []
    if fuse or forward or scalarise:
        from dace_fortran.substitute.fuse_stencils import fold_control_flow_constants, forward_copies, fuse_stencil_pairs

        if elements:
            before = len(list(sdfg.states()))
            sdfg.save(str(DUMP_DIR / "pre_fold.sdfgz"), compress=True)  # debugging aid: the SDFG the second folding starts from
            sample = [v for e in sdfg.all_interstate_edges(recursive=True) for v in e.data.assignments.values()
                      if any(name in str(v) for name in elements)]
            print(f"assignments reading the recorded arrays before the second folding: {len(sample)}, e.g. {sample[:3]}", flush=True)
            fold_control_flow_constants(sdfg, {}, elements)
            sdfg.simplify(validate=False, validate_all=False)
            # A loop the folding gave a constant trip count (bounded by an element read inside a kernel body) is
            # unrolled like the others were.
            unrolled = P.ShortLoopUnroll(UNROLL_LIMIT).apply_pass(sdfg, {})
            if unrolled:
                sdfg.simplify(validate=False, validate_all=False)
            sample = [v for e in sdfg.all_interstate_edges(recursive=True) for v in e.data.assignments.values()
                      if any(name in str(v) for name in elements)]
            print(f"recorded elements folded after unrolling: {before} -> {len(list(sdfg.states()))} states, "
                  f"{unrolled or 0} short loops unrolled after it; {len(sample)} such assignments left, "
                  f"e.g. {sample[:3]}", flush=True)
        if scalarise:
            # After the folding's simplification, which would inline a body that became one state into its map: the
            # stencil fusion recomputes a producer whose body is one nested SDFG.
            straighten_bodies(sdfg)
        if fuse:
            fused_maps = fuse_stencil_pairs(sdfg, tile=tile_fuse)
            print(f"stencil fusion: {'tiled' if tile_fuse else 'recomputed'} {fused_maps}", flush=True)
        if forward:
            print(f"copy forwarding: {forward_copies(sdfg, known=extents, skip=forward_skip)}", flush=True)
        sdfg.validate()
        sdfg.save(str(DUMP_DIR / "post_forward.sdfgz"), compress=True)  # debugging aid
    finish_scratch_images(sdfg, scratch_images or {})
    laid_out = []
    if members_first:
        from dace_fortran.bindings.record_layout import lay_out_members_first, record_buffers

        laid_out = lay_out_members_first(sdfg, record_buffers(get_frozen_signature(sdfg)))
        print(f"record images laid out with the member dimensions fastest: {laid_out}", flush=True)
    if combine_divisions:
        # After the fusion: a recomputed stencil body is one more copy of the normalisations to rewrite.
        from dace_fortran.substitute.numerics import combine_normalised_quotients

        print(f"normalisations computed over a common denominator: {combine_normalised_quotients(sdfg)}", flush=True)
    from dace_fortran.substitute.collapse_bodies import collapse_through_bodies

    print(f"maps collapsed with the map their nested body holds: {collapse_through_bodies(sdfg)}", flush=True)
    sdfg.save(str(DUMP_DIR / "post_images.sdfgz"), compress=True)  # debugging aid
    reordered = order_for_coalescing(sdfg)
    report_sequential_loops(sdfg)
    # The module variables the optimised code no longer reads: the binding need not copy them in on every call.
    dropped = drop_unread_module_inputs(sdfg)
    print(
        f"optimized in {time.time() - t0:.0f} s: {num_maps(sdfg)} maps ({collapses} collapsed, {reordered} reordered), "
        f"{len(dropped)} unread module inputs dropped",
        flush=True,
    )
    from dace_fortran.substitute.small_kernels import (fuse_small_maps, host_maps_sequential, host_small_work, inline_small_reductions,
                               run_sequentially)

    if fuse_small:
        fused_states, fused_small = fuse_small_maps(sdfg, fuse_small)
        print(f"small maps (<= {fuse_small} iterations): {fused_states} states and {fused_small} maps fused, "
              f"{num_maps(sdfg)} maps left", flush=True)
    print(f"small ANY/ALL made tasklets: {inline_small_reductions(sdfg)}", flush=True)
    host_maps, host_arrays = host_small_work(sdfg, host_work_limit)
    run_sequentially(sdfg, host_maps)
    if host_work_limit:
        print(f"small work kept on the host: {len(host_maps)} maps over {sorted(host_arrays)}", flush=True)
    # Free tasklets too: the GPU transformation makes a one-thread kernel of each, which cannot write a host array
    # (a MUSCL build set a five-element module array from the grid spacing that way).
    on_gpu = (arrays_used_by_maps(sdfg) | arrays_used_by_free_tasklets(sdfg)) - host_arrays
    # An array some host-side control flow reads an element of (a bounds symbol, a direction index) stays where the
    # caller holds it: the kernels get a device copy of it; a device-resident one would be read through its device
    # pointer by host code.
    host_read = host_read_arrays(sdfg)
    moved = []
    for name, desc in sdfg.arrays.items():
        if name in sdfg.constants:
            continue  # a constant is a host constexpr array: device storage would make its copies device-to-device
        if not desc.transient and isinstance(desc, dace.data.Array) and name in on_gpu and name not in host_read:
            desc.storage = dace.StorageType.GPU_Global
            moved.append(name)
    print(f"{len(moved)} argument arrays on the GPU: {sorted(moved)}; read by host control flow, kept on the host: "
          f"{sorted(host_read & on_gpu)}", flush=True)
    # Arguments no map touches stay where the caller holds them: a copy between two of them (a record's array members
    # assigned whole, say) runs on the host instead of through device copies made and dropped on every call.
    host = [n for n, d in sdfg.arrays.items() if not d.transient and d.storage != dace.StorageType.GPU_Global]
    host += sorted(n for n in host_arrays if n not in host)  # the transients of the small work kept on the host
    sdfg.save(str(DUMP_DIR / "pre_gpu.sdfgz"), compress=True)  # debugging aid: the SDFG the GPU transformation starts from
    sdfg.apply_gpu_transformations(host_data=host, host_maps=host_maps)
    # A constant is a host constexpr array: with device storage on either of its descriptors its copies to device
    # buffers would be emitted device-to-device from a host pointer.
    for name in sdfg.constants:
        for desc in (sdfg.arrays.get(name), sdfg.constants_prop[name][0]):
            if isinstance(desc, dace.data.Array) and desc.storage == dace.StorageType.GPU_Global:
                desc.storage = dace.StorageType.Default
    if fuse_small:
        # The free tasklets the transformation wrapped in one-thread kernels (element copies between device arrays).
        _states, wrapped = fuse_small_maps(sdfg, fuse_small, fuse_states=False)
        print(f"one-thread kernels of free tasklets fused: {wrapped}", flush=True)
    # The device buffers the transformation added: allocated once, not on every call.
    MakeTransientsPersistent().apply_pass(sdfg, {})
    print(f"dependency edges added between components sharing an array: {order_components_sharing_data(sdfg)}", flush=True)
    heavy = []
    for node, _parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            scope_nodes = list(_scope_nodes_recursive(_parent, node))
            # A kernel tiled into shared memory has its block in its thread-block maps.
            explicit_block = any(isinstance(n, nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_ThreadBlock
                                 for n in scope_nodes)
            # A kernel that recomputes a stencil runs long per thread: smaller blocks schedule better.
            recomputes = fused_maps and any(
                isinstance(n, nodes.MapEntry) and n.map.label.endswith("_recompute") for n in scope_nodes
            )
            if not explicit_block:
                node.map.gpu_block_size = FUSED_BLOCK_SIZE if recomputes else GPU_BLOCK_SIZE
            if heavy_launch:
                tasklets = sum(1 for n in scope_nodes if isinstance(n, nodes.Tasklet))
                if tasklets >= heavy_launch:
                    if explicit_block:
                        # The block the code generator takes from the thread-block maps (their largest extents), with
                        # as many resident blocks as the register file allows at the heavy kernels' register budget.
                        threads = explicit_block_threads(scope_nodes)
                        node.map.gpu_launch_bounds = f"{threads}, {max(1, min(4, 65536 // (threads * 128)))}"
                    else:
                        set_heavy_launch(node.map)
                    heavy.append((node.map.label, tasklets, node.map.gpu_launch_bounds))
    if heavy_launch:
        print(f"heavy kernels (>= {heavy_launch} tasklets) launched {HEAVY_BLOCK_SIZE} with __launch_bounds__"
              f"({HEAVY_LAUNCH_BOUNDS}): {heavy}", flush=True)
    # The library nodes expand here, as the code generator would (after the same defaults), so that the host maps of
    # their expansions (a copy of a small constant array into a host buffer, say) are run sequentially too.
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.expand_library_nodes()
    print(f"host maps made sequential: {host_maps_sequential(sdfg)}", flush=True)
    refreeze(sdfg)
    sdfg.validate()
    dace.Config.set("compiler", "cuda", "cuda_arch", value=CUDA_ARCH)
    return laid_out


def instrument_kernels(sdfg: dace.SDFG) -> int:
    """Time every top-level GPU map with CUDA events (``InstrumentationType.GPU_Events``): the generated program
    records one duration per launch and writes a chrome-trace report into the build folder's ``perf`` directory when
    it exits -- a kernel table without a profiler on the node.  Returns the number of maps instrumented."""
    n = 0
    for state in sdfg.states():
        scopes = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and scopes[node] is None \
                    and node.map.schedule == dace.ScheduleType.GPU_Device:
                node.map.instrument = dace.InstrumentationType.GPU_Events
                n += 1
    return n


#: Linker options of the SDFG library.  -Bsymbolic: every DaCe program exports the same unmangled helpers
#: (__dace_init_cuda, __dace_gpu_set_stream, ...), and once several substitutes are linked into one executable a
#: program's call of its own helper would resolve to the first library's copy, which sets up the wrong state.
#: --as-needed: DaCe links every program against OpenMP; the host code of GPU kernels makes no OpenMP call, and
#: GCC's libgomp, once loaded into an NVHPC OpenACC process, interposes its own OpenACC entry points (acc_*) on the
#: application's runtime.
LIBRARY_LINK_FLAGS = "-Wl,-Bsymbolic -Wl,--as-needed"


def expose_private_imports(
    bindings: Path, dirs: list[Path], out: Path, extra: dict[str, list[str]] | None = None
) -> list[dict]:
    """For every module whose private variables ``bindings`` imports, or ``extra`` names (module -> variables): a copy
    of its source (found in ``dirs``) with accessors to them, the variables, and the ``use`` lines their declarations
    need. One copy per module, with the variables of both: only one can replace the module's object."""
    from dace_fortran.bindings.module_access import public_variables  # noqa: PLC0415

    text = bindings.read_text().replace("&\n", " ")
    files = module_files(dirs)
    sources = {mod: path.read_text(errors="replace") for mod, path in files.items()}
    wanted: dict[str, list[str]] = {mod: list(names) for mod, names in (extra or {}).items()}
    for m in re.finditer(r"^\s*use (\w+), only: (.*)$", text, re.MULTILINE):
        mod = m.group(1).lower()
        if mod not in sources:
            continue
        public = public_variables(sources[mod], mod)
        for item in m.group(2).split(","):
            name = item.split("=>")[-1].strip().lower()
            declared = re.search(rf"::[^!\n]*\b{name}\b", sources[mod], re.IGNORECASE)
            if name and name not in public and declared and name not in wanted.get(mod, []):
                wanted.setdefault(mod, []).append(name)
    return write_access_copies(wanted, files, out)


def private_baked_variables(baked: list, dirs: list[Path], records: list[Path]) -> dict[str, list[str]]:
    """Per module, the variables of the ``baked`` values (``RecordedValue``, value) their modules keep private, from
    the sources in ``dirs``: the substitute entry reads them through accessors. A module without a source there is
    taken as public. Without ``dirs``, a baked variable a recorder of ``records`` read through an accessor (its
    substitute.json "access") is refused: the entry would import a private variable and not compile."""
    from dace_fortran.bindings.module_access import public_variables  # noqa: PLC0415

    def variable(name: str) -> str:
        return re.split(r"[(%]", name)[0].lower()

    if not dirs:
        listed = set()
        for record in records:
            recorder = Path(record).resolve().parent / "substitute.json"
            if recorder.is_file():
                for a in json.loads(recorder.read_text()).get("access", []):
                    listed |= {(a["module"], name) for name in a.get("variables", [])}
        private = sorted(f"{v.module}::{v.name}" for v, _ in baked if (v.module.lower(), variable(v.name)) in listed)
        if private:
            raise SystemExit(f"baked values of private variables {private} need --fypp (the build's sources) to be "
                             "checked through accessors")
        return {}
    files = module_files(dirs)
    public: dict[str, set[str]] = {}
    out: dict[str, list[str]] = {}
    for v, _ in baked:
        mod, var = v.module.lower(), variable(v.name)
        if mod not in files:
            continue
        if mod not in public:
            public[mod] = public_variables(files[mod].read_text(errors="replace"), mod)
        if var not in public[mod] and var not in out.get(mod, []):
            out.setdefault(mod, []).append(var)
    return out



def infer_scratch(sdfg: dace.SDFG, args, plan: FlattenPlan, symbols: dict, scalars: dict, dropped: list[str],
                  recorded: dict) -> list[str]:
    """--auto-scratch: what make_scratch_transient is to match for the module variables auto_scratch.infer finds
    scratch, those given with --scratch left out. The in-call analysis runs on a copy specialised as this build
    specialises the SDFG (the branches the recorded configuration never takes are gone); ``dropped``: the procedures
    whose calls the build drops; ``recorded``: the recorded symbols the build may rely on."""
    from dace_fortran.substitute.auto_scratch import infer, scratch_keys  # noqa: PLC0415


    def specialise(copy: dace.SDFG) -> dict:
        split = specialize_and_split(copy, plan, symbols, scalars, args.map_heavy_loops) if symbols or scalars else {}
        # A record image made a transient takes the recorded extents and lower bounds (finish_scratch_images): the
        # analysis judges each image as it would then be built.
        fold = {}
        for a in get_frozen_signature(copy).args:
            if getattr(a, "aos_origin_struct", "") and a.sdfg_name in copy.arrays:
                fold.update(image_symbols(a.sdfg_name, copy.arrays[a.sdfg_name], recorded))
        if fold:
            specialize(copy, symbols=fold)
        elements = recorded_elements(args.globals, copy) if args.globals else {}
        if elements:
            from dace_fortran.substitute.fuse_stencils import fold_control_flow_constants  # noqa: PLC0415

            fold_control_flow_constants(copy, {}, elements)
            copy.simplify(validate=False, validate_all=False)
        return split

    result = infer(sdfg, args.entry, specialise, args.fypp, dropped, asserted=args.scratch,
                   synthetic=plan.synthetic_globals)
    print(result.report(), flush=True)
    given = {s.lower() for s in args.scratch}
    used = [c for v in result.scratch for c in result.containers[v] if c in result.used]
    return [k for k in scratch_keys(sdfg, used) if k not in given]


def kept_buffers(frozen, keys: list[str]) -> list[str]:
    """The buffers of the module arrays of records ``keys`` (``module::record``) the SDFG takes as arguments, for the
    binding to keep between calls (--keep, --auto-keep)."""
    if not keys:
        return []
    from dace_fortran.bindings.block_builders import keepable  # noqa: PLC0415

    found, names = set(), []
    for a in frozen.args:
        if keepable(a):
            key = f"{a.aos_origin_mod}::{a.aos_origin_struct.split('%')[0].strip()}".lower()
            if key in keys:
                found.add(key)
                names.append(a.sdfg_name)
    if set(keys) - found:
        print(f"--keep: no buffer of these the binding can keep among the SDFG's arguments: "
              f"{sorted(set(keys) - found)}", flush=True)
    print(f"buffers kept between calls: {names}", flush=True)
    return names


def baked_values(records: list[Path], values: dict | None) -> list:
    """The module variables baked into the SDFG (``--globals``), with their types from the recorders'
    ``recorded_types.json``, for the substitute entry to check on every call."""
    if not records or not values:
        return []
    from dace_fortran.bindings.standin import RecordedValue  # noqa: PLC0415

    types = {}
    for record in records:
        types_file = Path(record).resolve().parent / "recorded_types.json"
        if not types_file.is_file():
            raise SystemExit(f"{types_file} is missing: rerun make_recorder.py for {record}; without the types the "
                             "substitute cannot check the values it was specialised to")
        types.update(json.loads(types_file.read_text()))
    missing = sorted(set(values) - set(types))
    if missing:
        raise SystemExit(f"no recorded type for {missing}")
    baked = [(RecordedValue(*name.split("::", 1), types[name]), value) for name, value in sorted(values.items())]
    print(f"substitute entry checks {len(baked)} baked values", flush=True)
    return baked

def main() -> int:
    global DUMP_DIR, HEAVY_LAUNCH_BOUNDS, CUDA_ARCH, STARTUP
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--combine-divisions", action="store_true",
                    help="opt-in numerics: compute each normalisation of quotients (w_k = a_k / sum_j a_j with "
                         "a_k = n_k / d_k) over a common denominator, one division instead of 2m; changes rounding")
    ap.add_argument("--fma", action=argparse.BooleanOptionalAction, default=False,
                    help="contract multiplies and adds into fused multiply-adds (nvcc's default). Off by default "
                         "(nvcc --fmad=false, host -ffp-contract=off): the IEEE operations of the source in its order, "
                         "which a reference built likewise reproduces bit for bit; --fma changes results at ~1e-12; "
                         "--no-fma, the default, is accepted for scripts that pass it")
    ap.add_argument("--cuda-arch", default=CUDA_ARCH, help=f"CUDA architecture of the kernels (default {CUDA_ARCH})")
    ap.add_argument("--heavy-launch-bounds", default=HEAVY_LAUNCH_BOUNDS, metavar="THREADS, BLOCKS",
                    help=f"__launch_bounds__ of the heavy kernels (default {HEAVY_LAUNCH_BOUNDS!r})")
    ap.add_argument("--heavy-launch", type=int, default=HEAVY_LAUNCH_TASKLETS, metavar="TASKLETS",
                    help="kernels holding at least this many tasklets get the heavy-kernel launch settings "
                         f"(HEAVY_BLOCK_SIZE, HEAVY_LAUNCH_BOUNDS; default {HEAVY_LAUNCH_TASKLETS}, 0: off)")
    ap.add_argument("--host-work-limit", type=int, default=64, metavar="ELEMENTS",
                    help="maps and free tasklets whose arrays all hold at most this many elements and are touched by "
                         "no kernel stay on the host instead of becoming one-block kernels with device copies "
                         "(small_kernels.host_small_work; 0: off)")
    ap.add_argument("--fuse-small-maps", type=int, default=1 << 18, metavar="ITERATIONS",
                    help="fuse the states around maps of at most this many iterations (a face of the domain), then "
                         "such maps with the same range into one kernel (small_kernels.fuse_small_maps; 0: off)")
    ap.add_argument("--map-heavy-loops", type=int, default=0, metavar="TASKLETS",
                    help="experimental: make parallel short loops whose bodies hold at least this many tasklets maps "
                         "before short loops are unrolled (0: off)")
    ap.add_argument("--scalarise", action=argparse.BooleanOptionalAction, default=True,
                    help="straight-line kernel bodies: unroll small maps inside kernels, split small scratch arrays "
                    "into scalars and fuse the bodies' states, before the stencil fusion (straight.py, scalarise.py); "
                    "on by default (same results; fused fifth-order kernels 1.7x faster), --no-scalarise turns it off")
    ap.add_argument("--gpu-streams", type=int, default=1,
                    help="CUDA streams the generated program may use (default 1: every kernel and copy in program "
                         "order on one stream, which the program synchronises before it returns; with more streams, "
                         "work left on the other streams races with what the caller does on the device next)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--record", type=Path, help="file the binding appends the integer symbols of every call to")
    mode.add_argument("--specialize", type=Path, help="symbols recorded with --record to specialise the SDFG to")
    ap.add_argument("--globals", type=Path, action="append", default=[], help="record of a recording stand-in "
                    "(make_recorder.py): module variables to specialise before inlining (repeatable: the records of "
                    "successive recording rounds, each pruned by the ones before, together hold every value)")
    ap.add_argument("--record-only", action="store_true", help="with --record: decline every call once recorded, so "
                    "the original runs (a sizing run that never executes the general SDFG)")
    ap.add_argument("--stub", action="append", default=[], help="procedure whose calls are dropped (repeatable)")
    ap.add_argument("--scratch", action="append", default=[], help="module array the routine only uses as scratch "
                    "(module::variable): it becomes a transient of the SDFG and leaves the binding (repeatable)")
    ap.add_argument("--auto-scratch", action="store_true", help="infer the module arrays the routine only uses as "
                    "scratch (auto_scratch.py: every element read written earlier in the call, nothing outside the "
                    "routine reading what it leaves; needs --fypp) and treat them as --scratch, with any given")
    ap.add_argument("--keep", action="append", default=[], metavar="MODULE::RECORD",
                    help="a module array of records only the routine uses (nothing outside its calls reads or writes "
                    "it): the binding packs its buffer once and keeps it between calls instead of copying it in and "
                    "out on every call (repeatable)")
    ap.add_argument("--auto-keep", action="store_true", help="infer the module arrays of records to keep "
                    "(auto_scratch.infer_kept: nothing outside the routine's calls reads, writes, aliases or takes the "
                    "address of them; needs --fypp) and keep them, with any given")
    ap.add_argument("--startup-constants", action="store_true",
                    help="specialise only to recorded values that no code after the routine's first call can change "
                    "(startup_constants.py, on the build's sources: needs --fypp); the recorded calls still decide "
                    "which values stay the same")
    ap.add_argument("--fuse", action="store_true", help="recompute stencil producers inside their consumers (GPU)")
    ap.add_argument("--tile-fuse", action="store_true",
                    help="with --fuse: fuse through a shared-memory tile (the producer once per tile) instead")
    ap.add_argument("--instrument", action="store_true",
                    help="time every GPU kernel with CUDA events (DaCe GPU_Events instrumentation): the report lands in "
                         "<out>/.dacecache/<sdfg>/perf/ when the program exits; summarise with rhs/perf_report.py")
    ap.add_argument("--records-first", action="store_true",
                    help="keep the buffers of module arrays of records laid out with the record indices fastest (GPU; "
                         "by default the field dimensions are fastest)")
    ap.add_argument("--forward", action="store_true", help="drop copies that only mirror an argument in scratch (GPU)")
    ap.add_argument("--forward-skip", action="append", default=[], metavar="ARRAY",
                    help="with --forward: leave this scratch array's copies as they are (repeatable; an A/B)")
    ap.add_argument("--fypp", type=Path, action="append", default=[], help="a directory of the build's (preprocessed) "
                    "sources, repeatable: module variables the binding needs but their modules keep private are reached "
                    "through accessors of copies of these modules")
    ap.add_argument("--shim", type=Path, help="a hand-written substitute entry (external subroutine dace_sub_<routine> "
                    "calling the binding) instead of the generated one; not with --globals, whose baked values only "
                    "the generated entry checks")
    ap.add_argument("hlfir")
    ap.add_argument("entry")
    ap.add_argument("out")
    args = parse_with_config(ap, "make_substitute")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    DUMP_DIR = out
    HEAVY_LAUNCH_BOUNDS = args.heavy_launch_bounds
    CUDA_ARCH = args.cuda_arch
    if args.startup_constants:
        if not args.fypp:
            raise SystemExit("--startup-constants needs the build's sources (--fypp)")
        from dace_fortran.substitute.outside_refs import parse_sources, source_files  # noqa: PLC0415
        from dace_fortran.substitute.startup_constants import StartupAnalysis  # noqa: PLC0415

        STARTUP = StartupAnalysis(parse_sources(source_files(args.fypp)), tuple(args.entry.lower().split("::")))
        start = f"main program {STARTUP.main.name}" if STARTUP.main else "no single main program"
        print(f"start-up: {start}; {len(STARTUP.after_procedures)} procedures may run after the first call",
              flush=True)
    globals_ = keep_startup_values(merged_invariants(args.globals), "recorded module values") if args.globals else None
    for name in args.stub:
        keep_external(name, stub=True)
    builder = SDFGBuilder(args.hlfir, entry=args.entry, specialized_globals=globals_)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.validate()
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)
    general = get_frozen_signature(sdfg)
    unsourced = make_unsourced_transient(sdfg, general, iface, plan)
    if unsourced:
        print(f"arguments no caller supplies made transient: {unsourced}", flush=True)
    recorded = recorded_invariants(args.specialize) if args.specialize else {}
    if STARTUP is not None and recorded:
        recorded = keep_startup_symbols(recorded, general, iface, plan)
    values = {}
    if args.specialize:
        guardable = specializable(general, iface, plan)
        values = {name: value for name, value in sorted(recorded.items()) if name in guardable}
    symbols = {name: value for name, value in values.items() if name in general.free_symbols}
    scalars = {name: value for name, value in values.items() if name not in symbols}
    scratch_names = list(args.scratch)
    if args.auto_scratch:
        scratch_names += infer_scratch(sdfg, args, plan, symbols, scalars, args.stub, recorded)
    scratch, scratch_images = [], {}
    if scratch_names:
        scratch, scratch_images = make_scratch_transient(sdfg, scratch_names, recorded)
        print(f"scratch module arrays made transient: {scratch}; images deferred: {sorted(scratch_images)}", flush=True)
    keep_keys = [k.lower() for k in args.keep]
    if args.auto_keep:
        from dace_fortran.substitute.auto_scratch import infer_kept  # noqa: PLC0415

        kept, rejected = infer_kept(sdfg, args.entry, args.fypp,
                                    args.stub)
        print(f"auto-keep: {len(kept)} of {len(kept) + len(rejected)} module arrays of records the SDFG takes are "
              f"kept between calls: {sorted(kept)}", flush=True)
        for key, why in sorted(rejected.items()):
            print(f"  not kept: {key}: {why[0]}" + (f" (+{len(why) - 1} more)" if len(why) > 1 else ""), flush=True)
        keep_keys += [k for k in kept if k not in keep_keys]
    if args.specialize:
        print(f"specialised to {len(values)} of {len(recorded)} invariant values: "
              f"{values}", flush=True)
    if values:
        split = specialize_and_split(sdfg, plan, symbols, scalars, args.map_heavy_loops)
        print(f"arrays of records handed over per element: {split}", flush=True)
    if args.specialize:
        # A member of an element of a module array of records that IS another array of the SDFG (the sizing run's
        # alias record): one array, or the storage guard would decline every call.
        from dace_fortran.substitute.fold_aliases import fold_aliases  # noqa: PLC0415

        aliases = recorded_aliases(args.specialize)
        if aliases:
            print(f"aliases folded: {fold_aliases(sdfg, aliases, recorded)}", flush=True)
    elements = {}
    if args.globals and (args.fuse or args.forward):
        elements = recorded_elements(args.globals, sdfg)
        if elements:
            from dace_fortran.substitute.fuse_stencils import fold_control_flow_constants

            fold_control_flow_constants(sdfg, {}, elements)
            sdfg.simplify(validate=False, validate_all=False)
            print(f"recorded elements of module arrays of records folded: {elements}", flush=True)
    members_first = []
    if args.gpu:
        members_first = to_gpu(sdfg, fuse=args.fuse, forward=args.forward, elements=elements,
                               scratch_images=scratch_images, combine_divisions=args.combine_divisions,
                               scalarise=args.scalarise, map_heavy_loops=args.map_heavy_loops,
                               heavy_launch=args.heavy_launch, host_work_limit=args.host_work_limit,
                               fuse_small=args.fuse_small_maps, tile_fuse=args.tile_fuse,
                               members_first=not args.records_first,
                               extents=recorded, forward_skip=args.forward_skip)
    elif values:
        refreeze(sdfg)
    if args.record_only and not args.gpu:
        # A sizing run never executes the SDFG: sequential maps keep OpenMP (GCC's runtime, which clashes with the
        # NVHPC runtime of a GPU build of MFC) out of the library. Library nodes expand into maps of their own, so
        # expand them first.
        sdfg.expand_library_nodes()
        for node, _parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry):
                node.map.schedule = dace.ScheduleType.Sequential
    dace.Config.set("compiler", "linker", "args", value=LIBRARY_LINK_FLAGS)
    if args.gpu:
        dace.Config.set("compiler", "cuda", "max_concurrent_streams", value=args.gpu_streams)
    if not args.fma:
        cuda_args = dace.Config.get("compiler", "cuda", "args")
        dace.Config.set("compiler", "cuda", "args", value=f"{cuda_args} --fmad=false")
        cpu_args = dace.Config.get("compiler", "cpu", "args")
        dace.Config.set("compiler", "cpu", "args", value=f"{cpu_args} -ffp-contract=off")
        print(f"no FMA contraction: cuda args '{dace.Config.get('compiler', 'cuda', 'args')}'", flush=True)
    if args.instrument:
        print(f"kernels instrumented with GPU events: {instrument_kernels(sdfg)}", flush=True)
    t0 = time.time()
    compiled = sdfg.compile()
    print(f"compiled in {time.time() - t0:.0f} s", flush=True)
    sdfg_so = Path(compiled._lib._library_filename)
    frozen = get_frozen_signature(sdfg)
    frozen.verify_against(sdfg)
    name = args.entry.split("::")[-1]
    bindings = out / f"{name}_bindings.f90"
    acc = plan_device_resident_caller(sdfg, frozen, iface, plan) if args.gpu else None
    kept_images = kept_buffers(frozen, keep_keys)
    # The caller's modules are used as they are (no TARGET patched in): copy allocatable module arrays, do not alias
    # them.
    emit = functools.partial(
        emit_bindings,
        frozen,
        iface,
        plan,
        str(bindings),
        tuple(compiled._sig or ()),
        init_symbols=compiled_init_symbols(sdfg_so, sdfg.name),
        module_allocatables_are_targets=False,
        acc_residency=acc,
        specialization=Specialization(values, general),
        record_symbols_to=args.record.resolve() if args.record else None,
        record_only=args.record_only,
        kept_images=kept_images,
        members_first=members_first,
    )
    try:
        emit()
    except Exception:
        # The SDFG the binding could not be emitted for, for a look at where the offending symbols are used.
        try:
            sdfg.save(str(out / "at_emit_failure.sdfg"), compress=False)
            print(f"binding emission failed: SDFG saved to {out / 'at_emit_failure.sdfg'}", flush=True)
        except Exception as save_error:  # noqa: BLE001
            print(f"binding emission failed (and the SDFG could not be saved: {save_error})", flush=True)
        raise
    module, routine = args.entry.split("::")
    shim = args.shim
    if args.globals and shim is not None:
        raise SystemExit(f"--globals needs the generated substitute entry, which checks the baked values; {shim} "
                         "is hand-written")
    dirs = [d.resolve() for d in args.fypp]
    baked = [] if shim is not None else baked_values(args.globals, globals_)
    # The baked variables their modules keep private: the entry checks them through the accessors of the same module
    # copies the binding uses.
    private_baked = private_baked_variables(baked, dirs, args.globals)
    access = []
    if dirs:
        access = expose_private_imports(bindings, dirs, out / "access", extra=private_baked)
        if access:
            emit(
                private_access={a["module"]: a["variables"] for a in access},
                private_imports=list(dict.fromkeys(line for a in access for line in a["imports"])),
            )
    if shim is None:
        # No hand-written shim: generate the substitute entry from the interface snapshot.
        from dace_fortran.bindings.standin import emit_substitute_entry  # noqa: PLC0415

        shim = emit_substitute_entry(
            iface,
            f"{sdfg.name}_dace_bindings",
            f"{sdfg.name}_dace",
            out / f"sub_{routine}.f90",
            baked=baked,
            private_access={a["module"]: a["variables"] for a in access},
            private_imports=list(dict.fromkeys(line for a in access for line in a["imports"])),
        )
        if private_baked:
            print(f"substitute entry checks baked private variables through accessors: {private_baked}", flush=True)
    info = {
        "sdfg_so": str(sdfg_so),
        "bindings": str(bindings),
        "sdfg_name": sdfg.name,
        "entry": iface.entry,
        "gpu": args.gpu,
        "wrap_symbol": f"{module}_{routine}_",
        "shim": str(shim),
        "shim_symbol": f"dace_sub_{routine}_",
        "original_symbol": f"dace_original_{routine}_",
        "record": str(args.record.resolve()) if args.record else None,
        "specialized": values,
        # Module copies with accessors, to compile with the build's own command and link in place of the originals.
        "access": access_entries(access),
    }
    (out / "substitute.json").write_text(json.dumps(info, indent=2))
    sdfg.save(str(out / f"{name}.sdfg"))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
