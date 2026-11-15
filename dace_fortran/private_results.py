# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep bridge-owned conditional results local after scalar promotion."""
import re
import dace


def restore_private_result_storage(sdfg):
    """Restore storage for private SCF results exposed by path-insensitive liveness.

    A branch can define a result that a later matching guard alone reads. Scalar
    promotion turns it into a symbol; conservative branch analysis then treats
    it as a caller input. Restore its original transient-scalar storage without
    initializing it or changing any executed assignment or guarded read. Only
    bridge-reserved result names with an actual definition qualify.
    """
    restored = set()
    for graph in sdfg.all_sdfgs_recursive():
        names = {name for name in graph.free_symbols if re.fullmatch(r'__sc_[0-9]+', name)}
        if not names:
            continue
        assigned = {name for edge in graph.all_interstate_edges() for name in edge.data.assignments}
        missing = names - assigned
        if missing:
            raise ValueError('undefined private SCF results: ' + ', '.join(sorted(missing)))
        for name in names:
            dtype = graph.symbols[name]
            graph.remove_symbol(name)
            graph.add_scalar(name, dtype, transient=True)
            restored.add(name)
    return restored
