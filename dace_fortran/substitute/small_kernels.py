"""Fewer, larger launches: the small work a whole-routine GPU substitute would otherwise launch by the hundred.

Two generic rules, for the outermost SDFG of a routine about to be turned into GPU code.

:func:`host_small_work` finds the work that should not be a kernel at all: a top-level map whose trip count and data
are tiny (the element-wise copies between the array members of a small record assigned whole, say), and the free
tasklets around it, as long as none of their data is touched by device work. On the device, each of them is a
one-block kernel, and each host read of what it wrote a device-to-host copy plus a stream synchronisation; on the host
it is a few loads and stores. The caller keeps the returned maps and arrays on the host (``host_maps`` and
``host_data`` of the GPU transformation).

:func:`fuse_small_maps` merges the kernels that are too small to fill the device: maps over a face of the domain, not
its volume (the per-field copies of a boundary condition into and out of its face arrays, say). It fuses the states
around them (StateFusion, StateFusionExtended), then the independent maps of one state that iterate over the same range
(MapFusionHorizontal), to a fixed point. Only maps of at most ``max_iterations`` iterations whose bodies, together,
hold at most ``max_tasklets`` tasklets take part: a long body (a fused stencil kernel) keeps its own launch settings.
Fusion moves no arithmetic, so results stay bit-identical.
"""
from __future__ import annotations

import dace
from dace.sdfg import nodes
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.dataflow import MapFusionHorizontal
from dace.transformation.interstate import StateFusion
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended


def _iterations(entry: nodes.MapEntry) -> int | None:
    """The constant trip count of a map, or None."""
    try:
        return int(entry.map.range.num_elements())
    except (TypeError, ValueError):
        return None


def _elements(desc) -> int | None:
    try:
        return int(desc.total_size)
    except (TypeError, ValueError):
        return None


def _scope_data(state: dace.SDFGState, node: nodes.Node) -> set[str]:
    """The containers a top-level map (its whole scope) or code node reads or writes."""
    exit_ = state.exit_node(node) if isinstance(node, nodes.EntryNode) else node
    ends = [state.memlet_tree(e).root().edge.src for e in state.in_edges(node)]
    ends += [state.memlet_tree(e).root().edge.dst for e in state.out_edges(exit_)]
    return {n.data for n in ends if isinstance(n, nodes.AccessNode)}


def inline_small_reductions(sdfg: dace.SDFG, max_elements: int = 64) -> int:
    """Replace every whole-array ANY/ALL (an AnyNode or AllNode reducing to a scalar) over a rank-1 mask of at most
    ``max_elements`` elements by a tasklet that ORs (ANDs) them. The GPU transformation treats a tasklet like any other
    small work (on the host, or a one-thread kernel); the library node's reduction expansion put a host-side reduction
    into device code (a hypoelastic build failed validation that way: ANY over the three shear indices). Returns the
    number replaced."""
    import copy

    from dace.libraries.standard.nodes.allany import AllNode, AnyNode

    replaced = 0
    for node, state in list(sdfg.all_nodes_recursive()):
        if not isinstance(node, (AnyNode, AllNode)) or node.dim != -1:
            continue
        mask = next(e for e in state.in_edges(node) if e.dst_conn == AnyNode.INPUT_CONNECTOR_NAME)
        out = next(e for e in state.out_edges(node) if e.src_conn == AnyNode.OUTPUT_CONNECTOR_NAME)
        size = mask.data.subset.num_elements()
        if len(mask.data.subset.size()) != 1 or not str(size).isdigit() or int(size) > max_elements:
            continue
        join = " or " if isinstance(node, AnyNode) else " and "
        code = "_out = " + join.join(f"(_mask[{i}] != 0)" for i in range(int(size)))
        tasklet = state.add_tasklet(node.label, {"_mask"}, {"_out"}, code)
        state.add_edge(mask.src, mask.src_conn, tasklet, "_mask", copy.deepcopy(mask.data))
        state.add_edge(tasklet, "_out", out.dst, out.dst_conn, copy.deepcopy(out.data))
        state.remove_node(node)
        replaced += 1
    return replaced


def host_small_work(sdfg: dace.SDFG, max_elements: int) -> tuple[list[str], set[str]]:
    """The top-level maps of ``sdfg`` that should run on the host, and the containers that must stay there with them.

    A candidate map has a constant trip count of at most ``max_elements``, reads its dynamic range from nothing, and
    touches only arrays (or scalars) of at most ``max_elements`` elements that are not device-resident already; a
    candidate library node (a copy) touches only such arrays. Device work is every other top-level map, library node
    and nested SDFG; a free tasklet is device work once one of its
    containers is touched by device work (the GPU transformation then runs it in a one-thread kernel). A candidate
    whose data meets device work is device work too, to a fixed point. Returns the GUIDs of the remaining candidates
    and the containers of them and of the free tasklets left on the host (transients among them must be host data,
    or the GPU transformation would move them to the device).
    """
    if max_elements <= 0:
        return [], set()
    candidates: list[tuple[nodes.MapEntry, set[str]]] = []
    tasklets: list[set[str]] = []
    device: set[str] = {name for name, desc in sdfg.arrays.items() if desc.storage in dace.dtypes.GPU_STORAGES}

    def small(name: str) -> bool:
        desc = sdfg.arrays[name]
        size = _elements(desc)
        return (isinstance(desc, (dace.data.Array, dace.data.Scalar)) and not isinstance(desc, dace.data.View)
                and size is not None and size <= max_elements and desc.storage not in dace.dtypes.GPU_STORAGES)

    for state in sdfg.states():
        scope = state.scope_dict()
        for node in state.nodes():
            if scope[node] is not None:
                continue
            if isinstance(node, nodes.MapEntry):
                data = _scope_data(state, node)
                trips = _iterations(node)
                dynamic = any(e.dst_conn and not e.dst_conn.startswith("IN_") for e in state.in_edges(node))
                if trips is not None and trips <= max_elements and not dynamic and data and all(map(small, data)):
                    candidates.append((node, data))
                else:
                    device |= data
            elif isinstance(node, nodes.LibraryNode):
                # A copy between record members, say: on the host as long as its data is (see the GPU
                # transformation's host data).
                data = _scope_data(state, node)
                if data and all(map(small, data)):
                    candidates.append((node, data))
                else:
                    device |= data
            elif isinstance(node, nodes.NestedSDFG):
                device |= _scope_data(state, node)
            elif isinstance(node, nodes.Tasklet):
                tasklets.append(_scope_data(state, node))
    changed = True
    while changed:
        changed = False
        for data in tasklets:
            if data & device and not data <= device:
                device |= data
                changed = True
        for _entry, data in candidates:
            if data & device and not data <= device:
                device |= data
                changed = True
    host_nodes = [node for node, data in candidates if not data & device]
    host_data = set().union(*(data for node, data in candidates if node in host_nodes), set())
    host_data |= set().union(*(data for data in tasklets if not data & device), set())
    return [node.guid for node in host_nodes if isinstance(node, nodes.MapEntry)], host_data


def run_sequentially(sdfg: dace.SDFG, guids: list[str]) -> int:
    """Give the maps of ``guids`` (the host maps of :func:`host_small_work`) a sequential schedule: a few iterations
    gain nothing from threads, and an OpenMP loop would link the host code against an OpenMP runtime (GCC's
    interposes its own OpenACC entry points on an OpenACC application). Returns the number of maps changed."""
    wanted = set(guids)
    changed = 0
    for node, _parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.guid in wanted:
            node.map.schedule = dace.ScheduleType.Sequential
            changed += 1
    return changed



def host_maps_sequential(sdfg: dace.SDFG) -> int:
    """After the GPU transformation: every map left outside device code gets a sequential schedule, for the reason of
    :func:`run_sequentially` (a map of a nested SDFG the frontend made, say: 2D_hypo_hlld's copy of the six-element
    stress permutation became an OpenMP loop, and GCC's runtime aborted next to NVHPC's with "libgomp: TODO").
    Inside device code a default map is a thread-block map and stays as it is. Returns the number of maps changed."""
    changed = 0
    for node, state in sdfg.all_nodes_recursive():
        if (isinstance(node, nodes.MapEntry)
                and node.map.schedule in (dace.ScheduleType.Default, dace.ScheduleType.CPU_Multicore)
                and not is_devicelevel_gpu(state.sdfg, state, node)):
            node.map.schedule = dace.ScheduleType.Sequential
            changed += 1
    return changed

def _body_tasklets(state: dace.SDFGState, entry: nodes.MapEntry) -> int:
    count = 0
    for node in state.scope_subgraph(entry).nodes():
        if isinstance(node, nodes.Tasklet):
            count += 1
        elif isinstance(node, nodes.NestedSDFG):
            count += sum(1 for n, _ in node.sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    return count


class OuterStateFusion(StateFusion):
    """StateFusion of the outermost SDFG only: the states of kernel bodies stay as they are."""

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return sdfg.parent_nsdfg_node is None and super().can_be_applied(graph, expr_index, sdfg, permissive)


class OuterStateFusionExtended(StateFusionExtended):
    """StateFusionExtended of the outermost SDFG only."""

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return sdfg.parent_nsdfg_node is None and super().can_be_applied(graph, expr_index, sdfg, permissive)


@dace.properties.make_properties
class SmallMapFusionHorizontal(MapFusionHorizontal):
    """MapFusionHorizontal restricted to top-level maps of the outermost SDFG of at most ``max_iterations``
    iterations each whose bodies hold at most ``max_tasklets`` tasklets together."""

    max_iterations = dace.properties.Property(dtype=int, default=1 << 18, desc="Largest trip count of a fused map")
    max_tasklets = dace.properties.Property(dtype=int, default=128, desc="Most tasklets of the fused body")

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if sdfg.parent_nsdfg_node is not None:
            return False
        first, second = self.first_parallel_map_entry, self.second_parallel_map_entry
        for entry in (first, second):
            trips = _iterations(entry)
            if trips is None or trips > self.max_iterations:
                return False
        if first.map.schedule != second.map.schedule:
            return False
        if _body_tasklets(graph, first) + _body_tasklets(graph, second) > self.max_tasklets:
            return False
        return super().can_be_applied(graph, expr_index, sdfg, permissive)


def fuse_small_maps(sdfg: dace.SDFG, max_iterations: int, max_tasklets: int = 128,
                    fuse_states: bool = True) -> tuple[int, int]:
    """Fuse the states around small maps and the small independent maps of a state with the same range, to a fixed
    point (see the module docstring). ``fuse_states``: also fuse states (off once the GPU transformation added its
    copy states). Returns (states fused, maps fused)."""
    if max_iterations <= 0:
        return 0, 0
    # ordering_edges_into_sources: maps of consecutive states that StateFusionExtended ordered (writes to other
    # fields of one array, say) fuse too.
    options = dict(only_toplevel_maps=True, max_iterations=max_iterations, max_tasklets=max_tasklets,
                   ordering_edges_into_sources=True)
    states = maps = 0
    while True:
        s = sdfg.apply_transformations_repeated([OuterStateFusion, OuterStateFusionExtended], validate=False,
                                                print_report=False) if fuse_states else 0
        m = sdfg.apply_transformations_repeated(SmallMapFusionHorizontal, options=options, validate=False,
                                                print_report=False)
        states += s
        maps += m
        if not s and not m:
            return states, maps
