# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Device copies of the host scalars kernels read.

A map that becomes a kernel may read a scalar argument of the SDFG (a wall velocity, a tolerance). A tasklet gets
such a value as a kernel argument, but a nested SDFG in the kernel (a loop body) takes it through a memlet, and the
CUDA code generator refuses a host-to-device copy inside a kernel. Each such scalar gets a transient device copy,
made once at the start of the SDFG; the maps read the copy. A scalar the SDFG writes is left alone: its device copy
would go stale.
"""
import dace
from dace.sdfg import nodes


def device_copies_of_kernel_read_scalars(sdfg: dace.SDFG) -> list[str]:
    """Give every non-transient scalar a top-level map reads (through an edge into the map, or an access node inside
    it), and nothing in the SDFG writes, a device copy made in a new first state, and let the maps read the copy; in
    ``sdfg`` and in the nested SDFGs that sit outside maps (host-side nested SDFGs, which the GPU transformation
    treats as programs of their own). Returns the scalars copied, qualified by the SDFG's label when nested."""
    copied = _device_copies(sdfg)
    for state in sdfg.states():
        scope = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG) and scope[node] is None:
                copied += [f"{node.sdfg.label}.{name}" for name in device_copies_of_kernel_read_scalars(node.sdfg)]
    return copied


def _device_copies(sdfg: dace.SDFG) -> list[str]:
    written = {n.data for st in sdfg.states() for n in st.data_nodes()
               if any(not e.data.is_empty() for e in st.in_edges(n))}
    uses: list[tuple[dace.SDFGState, nodes.AccessNode, bool]] = []  # (state, access node, inside a map)
    for state in sdfg.states():
        scope = state.scope_dict()
        for node in state.data_nodes():
            desc = sdfg.arrays[node.data]
            if not isinstance(desc, dace.data.Scalar) or desc.transient or node.data in written:
                continue
            if scope[node] is not None:
                uses.append((state, node, True))
            elif any(isinstance(e.dst, nodes.MapEntry) for e in state.out_edges(node)):
                uses.append((state, node, False))
    names = sorted({node.data for _state, node, _inside in uses})
    if not names:
        return []
    copy_state = sdfg.add_state_before(sdfg.start_block, "scalars_to_device")
    copies = {}
    for name in names:
        dev, _ = sdfg.add_scalar(f"{name}_dev", sdfg.arrays[name].dtype, storage=dace.StorageType.GPU_Global,
                                 transient=True, find_new_name=True)
        copies[name] = dev
        copy_state.add_nedge(copy_state.add_read(name), copy_state.add_write(dev),
                             dace.Memlet.from_array(name, sdfg.arrays[name]))
    for state, node, inside in uses:
        name, dev = node.data, copies[node.data]
        scopes = [state.scope_dict()[node]] if inside else [e.dst for e in state.out_edges(node)
                                                            if isinstance(e.dst, nodes.MapEntry)]
        for entry in scopes:
            for inner in state.scope_subgraph(entry).edges():
                if inner.data.data == name:
                    inner.data.data = dev
        if inside:
            node.data = dev
            continue
        for e in list(state.out_edges(node)):
            if isinstance(e.dst, nodes.MapEntry):
                state.add_edge(state.add_read(dev), None, e.dst, e.dst_conn, dace.Memlet.from_array(dev, sdfg.arrays[dev]))
                state.remove_edge(e)
        if state.degree(node) == 0:
            state.remove_node(node)
    return names
