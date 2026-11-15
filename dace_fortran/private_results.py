# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep bridge-owned conditional results local after scalar promotion."""
import re
import dace


def restore_private_result_storage(sdfg, private_names=()):
    """Restore storage for private SCF results exposed by path-insensitive liveness.

    A branch can define a result that a later matching guard alone reads. Scalar
    promotion turns it into a symbol; conservative branch analysis then treats
    it as a caller input. Restore its original transient-scalar storage without
    initializing it or changing any executed assignment or guarded read. Only
    bridge-reserved result names or explicitly contracted private locals with
    actual definitions qualify. Unused forwarding declarations are removed.
    """
    private_names = set(private_names)
    restored = set()
    # Restore callees first, then remove forwarding-only parameters at their
    # callers. Additional names come from a frontend's private-local contract.
    for graph in reversed(list(sdfg.all_sdfgs_recursive())):
        names = {name for name in graph.free_symbols
                 if re.fullmatch(r'__sc_[0-9]+', name) or name in private_names}
        if not names:
            continue
        loop_names = {getattr(block, 'loop_variable', None)
                      for block in graph.all_control_flow_blocks()}
        if names & loop_names:
            raise ValueError('private results cannot replace loop variables: ' +
                             ', '.join(sorted(names & loop_names)))
        assigned = {name for edge in graph.all_interstate_edges() for name in edge.data.assignments}
        missing = names - assigned
        if missing:
            used = graph.used_symbols(all_symbols=False)
            shape_names = {str(symbol) for desc in graph.arrays.values() for symbol in desc.free_symbols}
            unused = missing - used - shape_names
            for name in unused:
                graph.remove_symbol(name)
                restored.add(name)
            names -= unused
            missing -= unused
        if missing:
            raise ValueError('undefined private SCF results: ' + ', '.join(sorted(missing)))
        for name in names:
            dtype = graph.symbols[name]
            graph.remove_symbol(name)
            graph.add_scalar(name, dtype, transient=True)
            if graph.parent_nsdfg_node is not None:
                graph.parent_nsdfg_node.symbol_mapping.pop(name, None)
            restored.add(name)
    return restored
