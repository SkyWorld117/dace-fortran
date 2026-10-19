# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Device copies of the host scalars kernels read.

A map that becomes a kernel may read a scalar the host holds: an argument of the SDFG (a wall velocity, a
tolerance) or a transient a host tasklet writes (a specialised value). A tasklet gets such a value as a kernel
argument, but a nested SDFG in the kernel (a loop body) takes it through a memlet, and the CUDA code generator
refuses the host-to-device copy that would mean inside the kernel. Every such scalar gets a transient device copy,
made in each state where a map reads it, after whatever the host writes to it there; the map reads the copy.
"""
import dace
from dace.sdfg import nodes


def device_copies_of_kernel_read_scalars(sdfg: dace.SDFG) -> list[str]:
    """Give every scalar a top-level map reads (through an edge into the map, or an access node inside it), and
    nothing inside a map writes, a transient device copy made where it is read; in ``sdfg`` and in the nested SDFGs
    that sit outside maps (host-side nested SDFGs, which the GPU transformation treats as programs of their own).
    Returns the scalars copied, qualified by the SDFG's label when nested."""
    copied = _device_copies(sdfg)
    for state in sdfg.states():
        scope = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG) and scope[node] is None:
                copied += [f"{node.sdfg.label}.{name}" for name in device_copies_of_kernel_read_scalars(node.sdfg)]
    return copied


def _device_copies(sdfg: dace.SDFG) -> list[str]:
    written_in_maps = set()
    uses: list[tuple[dace.SDFGState, nodes.AccessNode, bool]] = []  # (state, access node, inside a map)
    for state in sdfg.states():
        scope = state.scope_dict()
        for node in state.data_nodes():
            if not isinstance(sdfg.arrays[node.data], dace.data.Scalar):
                continue
            if scope[node] is not None:
                if any(not e.data.is_empty() for e in state.in_edges(node)):
                    written_in_maps.add(node.data)
                else:
                    uses.append((state, node, True))
            elif any(isinstance(e.dst, nodes.MapEntry) for e in state.out_edges(node)):
                uses.append((state, node, False))
    uses = [u for u in uses if u[1].data not in written_in_maps]
    names = sorted({node.data for _state, node, _inside in uses})
    copies = {}
    for name in names:
        copies[name], _ = sdfg.add_scalar(f"{name}_dev", sdfg.arrays[name].dtype, storage=dace.StorageType.GPU_Global,
                                          transient=True, find_new_name=True)
    for state, node, inside in uses:
        name, dev = node.data, copies[node.data]
        entries = [state.scope_dict()[node]] if inside else [e.dst for e in state.out_edges(node)
                                                             if isinstance(e.dst, nodes.MapEntry)]
        while state.scope_dict().get(entries[0]) is not None:  # the outermost map around an inside use
            entries = [state.scope_dict()[entries[0]]]
        for entry in entries:
            for inner in state.scope_subgraph(entry).edges():
                if inner.data.data == name:
                    inner.data.data = dev
        host = state.add_read(name) if inside else node
        dev_node = state.add_access(dev)
        state.add_nedge(host, dev_node, dace.Memlet.from_array(name, sdfg.arrays[name]))  # after any host write
        if inside:
            node.data = dev
            state.add_nedge(dev_node, entries[0], dace.Memlet())  # the copy before the kernel
            continue
        for e in list(state.out_edges(node)):
            if isinstance(e.dst, nodes.MapEntry):
                state.add_edge(dev_node, None, e.dst, e.dst_conn, dace.Memlet.from_array(dev, sdfg.arrays[dev]))
                state.remove_edge(e)
    return names
