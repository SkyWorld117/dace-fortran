# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Arrays of records handed over member by member instead of packed.

The binding of a kernel reading ``q(i)%sf`` packs the members of all elements into one buffer, on every call, and
unpacks what the kernel wrote.  Once every access of the SDFG picks its element by a constant -- as after
specialising the element loops away -- the buffer splits into one array per element
(:func:`dace_fortran.array_split.split_array`), and each of them is a member as the caller holds it: the binding
aliases ``q(k)%sf`` in place (:func:`dace_fortran.bindings.loop_copy.render_aos_alloc_members`).
"""

from __future__ import annotations

from dataclasses import replace

from dace import SDFG

from dace_fortran.array_split import split_array
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.frozen_signature import attach_to_sdfg, get_frozen_signature


def split_record_companions(sdfg: SDFG, plan: FlattenPlan) -> dict[str, dict[int, str]]:
    """Split the packed companion of every array of records whose accesses all pick their element by a constant, and
    return the arrays each split into, by element index.

    The frozen signature attached to ``sdfg`` takes one argument per element in place of the companion
    (``FrozenArg.element_of``), so the binding passes the members themselves.
    """
    frozen = get_frozen_signature(sdfg)
    if frozen is None:
        return {}
    args = list(frozen.args)
    split: dict[str, dict[int, str]] = {}
    for entry in plan.entries:
        recipe = entry.recipe
        if not recipe.aos_alloc:
            continue
        flat = recipe.flat_names[0]
        position = next((i for i, a in enumerate(args) if a.sdfg_name == flat), None)
        if position is None:
            continue
        names = split_array(sdfg, flat)
        if names is None:
            continue
        companion = args[position]
        args[position : position + 1] = [
            replace(
                companion,
                sdfg_name=name,
                rank=companion.rank - 1,
                shape=tuple(companion.shape[1:]),
                element_of=flat,
                element_index=index,
            )
            for index, name in names.items()
        ]
        split[flat] = names
    if split:
        attach_to_sdfg(sdfg, replace(frozen, args=tuple(args)))
    return split
