# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional loop-label selection for conservative parallelization."""

import re

from dace.transformation import explicit_cf_compatible
from dace.transformation.interstate.loop_to_map import LoopToMap


def loop_to_map_type(patterns=None):
    """Restrict mapping candidates by full label matches, retaining dependence checks.

    None preserves the default transformation. An empty list selects no loops.
    Restricting candidates lets a caller retain sequential recurrence/search
    loops inside a parallel column without repeatedly analyzing those loops.
    """
    if patterns is None:
        return LoopToMap
    if not isinstance(patterns, (list, tuple)) or not all(isinstance(p, str) for p in patterns):
        raise ValueError("loop_map_patterns must be a sequence of regular-expression strings")
    compiled = [re.compile(pattern) for pattern in patterns]

    @explicit_cf_compatible
    class SelectedLoopToMap(LoopToMap):
        def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
            if not any(pattern.fullmatch(self.loop.label) for pattern in compiled):
                return False
            return super().can_be_applied(graph, expr_index, sdfg, permissive=permissive)

    return SelectedLoopToMap
