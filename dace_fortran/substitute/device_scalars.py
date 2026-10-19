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
    """Give every non-transient scalar a top-level map reads, and nothing in ``sdfg`` writes, a device copy made in a
    new first state, and let the maps read the copy. Returns the scalars copied."""
    written = {n.data for st in sdfg.states() for n in st.data_nodes() if st.in_degree(n)}
    read_by_maps = set()
    for state in sdfg.states():
        scope = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and scope[node] is None:
                for e in state.in_edges(node):
                    desc = sdfg.arrays.get(e.data.data)
                    if isinstance(desc, dace.data.Scalar) and not desc.transient and e.data.data not in written:
                        read_by_maps.add(e.data.data)
    if not read_by_maps:
        return []
    copy_state = sdfg.add_state_before(sdfg.start_block, "scalars_to_device")
    for name in sorted(read_by_maps):
        dev, _ = sdfg.add_scalar(f"{name}_dev", sdfg.arrays[name].dtype, storage=dace.StorageType.GPU_Global,
                                 transient=True, find_new_name=True)
        copy_state.add_nedge(copy_state.add_read(name), copy_state.add_write(dev), dace.Memlet(name))
        for state in sdfg.states():
            scope = state.scope_dict()
            for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scope[n] is None]:
                for e in list(state.in_edges(entry)):
                    if e.data.data != name:
                        continue
                    for inner in state.scope_subgraph(entry).edges():
                        if inner.data.data == name:
                            inner.data.data = dev
                    state.add_edge(state.add_read(dev), None, entry, e.dst_conn, dace.Memlet(dev))
                    src = e.src
                    state.remove_edge(e)
                    if state.degree(src) == 0:
                        state.remove_node(src)
    return sorted(read_by_maps)
