# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Locate where an SDFG references a symbol name, for diagnostics.

Kept apart from :mod:`dace_fortran.builder` (which loads the compiled bridge on import) so it can be used and tested on
a plain ``dace.SDFG``.
"""

from __future__ import annotations

import re

from dace import SDFG, SDFGState
from dace.sdfg.nodes import NestedSDFG, Tasklet


def _is_wired_connector(state: SDFGState, tasklet: Tasklet, name: str) -> bool:
    """Whether ``name`` is an input or output connector of ``tasklet`` with an edge attached.

    Such a name in the tasklet's code is the connector's value, not a free symbol.
    """
    if name in tasklet.in_connectors and any(e.dst_conn == name for e in state.in_edges(tasklet)):
        return True
    return name in tasklet.out_connectors and any(e.src_conn == name for e in state.out_edges(tasklet))


def find_symbol_use_sites(sdfg: SDFG, symbol: str, limit: int = 3) -> list[str]:
    """Up to ``limit`` descriptions of the places in ``sdfg`` that reference the bare name ``symbol``.

    Looks at memlet subsets, tasklet code and nested-SDFG symbol mappings.  A tasklet whose code uses ``symbol`` as a
    wired connector (an edge attached) is skipped: the name is bound there, so it is not what makes the symbol free.
    A tasklet that uses the name with no connector, or with a connector nothing is wired to, is reported.

    The match is on word boundaries, so a symbol ``p_patch`` does not match ``p_patch_nlev``.

    :param sdfg: The SDFG to search.
    :param symbol: The name to look for.
    :param limit: Stop after this many sites.
    :return: One line per site, in state order, memlets before nodes within a state.
    """
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")
    examples: list[str] = []
    for state in sdfg.all_states():
        for edge in state.edges():
            m = edge.data
            if m is None or not m.data:
                continue
            subset_str = f"{m.subset}"
            if pat.search(subset_str):
                examples.append(f"memlet[{state.label}] data={m.data} subset={subset_str[:160]}")
                if len(examples) >= limit:
                    break
        if len(examples) >= limit:
            break
        for n in state.nodes():
            if isinstance(n, Tasklet) and pat.search(n.code.as_string) and not _is_wired_connector(state, n, symbol):
                examples.append(f"tasklet[{state.label}::{n.label}] code={n.code.as_string[:160]}")
                if len(examples) >= limit:
                    break
            if isinstance(n, NestedSDFG):
                for k, v in n.symbol_mapping.items():
                    if pat.search(str(v)):
                        examples.append(f"nsdfg[{state.label}::{n.label}] symbol_mapping[{k}]={v}")
                        if len(examples) >= limit:
                            break
        if len(examples) >= limit:
            break
    return examples
