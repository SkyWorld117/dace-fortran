"""Accumulations into neighbouring elements become write-conflict-resolved writes, so the loop can become a map.

MFC's IGR Riemann solvers add a face's flux to the cells on both sides, ``rhs(j) = rhs(j) + f`` and
``rhs(j + 1) = rhs(j + 1) - f``, under an OpenACC atomic that flang drops from the HLFIR: iteration ``j`` and
iteration ``j + 1`` write the same element, so LoopToMap keeps the loop sequential and the whole sweep runs serially
(3D_IGR_jet: 60x slower than the original). Such a tasklet, ``out = in +/- e`` reading and writing one element of an
array the loop writes at more than one offset, becomes ``out = +/-e`` with a sum write-conflict resolution on its
write (an atomic add on the GPU, as the original); the loop then maps. So does an accumulation into an element the
iteration picks through a value it loads: MFC's acoustic sources add to the cell ``coord(:, i)`` of each support
point (distinct cells, so the original needs no atomic; the loop cannot know that).
"""

from __future__ import annotations

import ast

import dace
from dace import nodes
from dace.sdfg.state import LoopRegion


def _contribution(code: str, c_in: str, c_out: str) -> str | None:
    """``out = in + e`` -> ``e``; ``out = in - e`` -> ``-(e)``; ``out = e + in`` -> ``e``; else None."""
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    assign = tree.body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name) or assign.targets[0].id != c_out:
        return None
    value = assign.value
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, (ast.Add, ast.Sub)):
        return None
    is_in = lambda n: isinstance(n, ast.Name) and n.id == c_in  # noqa: E731
    if is_in(value.left):
        rest = ast.unparse(value.right)
        return rest if isinstance(value.op, ast.Add) else f"-({rest})"
    if is_in(value.right) and isinstance(value.op, ast.Add):
        return ast.unparse(value.left)
    return None


def _loop_writes(loop: LoopRegion) -> dict[str, set[str]]:
    """Per array, the distinct subsets the loop's states write."""
    writes: dict[str, set[str]] = {}
    for st in loop.all_states():
        for e in st.edges():
            if e.data.data is not None and isinstance(e.dst, nodes.AccessNode):
                writes.setdefault(e.data.data, set()).add(str(e.data.subset))
    return writes


def _candidates(loop: LoopRegion, writes: dict) -> dict[str, list]:
    """Per array, the accumulations ``out = in +/- e`` of the loop into one element of it that may repeat across
    iterations -- picked by the loop variable in an array the loop writes at more than one offset, or by a symbol an
    iteration assigns: ``(state, write edge, tasklet, read edge, contribution)``."""
    found: dict[str, list] = {}
    var = loop.loop_variable
    assigned = {k for e in loop.all_interstate_edges() for k in e.data.assignments}
    for st in loop.all_states():
        for e in st.edges():
            t = e.src
            if not isinstance(t, nodes.Tasklet) or not isinstance(e.dst, nodes.AccessNode) or e.data.wcr:
                continue
            name, subset = e.data.data, e.data.subset
            if name is None or subset is None or subset.num_elements() != 1:
                continue
            used = {str(s) for s in subset.free_symbols}
            if not (used & assigned or (var in used and len(writes.get(name, ())) >= 2)):
                continue
            if st.out_degree(t) != 1:
                continue
            feeds = [x for x in st.in_edges(t) if isinstance(x.src, nodes.AccessNode) and x.data.data == name
                     and str(x.data.subset) == str(subset)]
            if len(feeds) != 1:
                continue
            contribution = _contribution(t.code.as_string, feeds[0].dst_conn, e.src_conn)
            if contribution is not None:
                found.setdefault(name, []).append((st, e, t, feeds[0], contribution))
    return found


def _reads(loop: LoopRegion, name: str) -> list:
    """Every edge in the loop's states that reads ``name``."""
    return [e for st in loop.all_states() for e in st.edges()
            if isinstance(e.src, nodes.AccessNode) and e.src.data == name and not e.data.is_empty()]


def accumulations_to_wcr(sdfg: dace.SDFG) -> list[str]:
    """Apply the rewrite to every loop of ``sdfg`` (nested SDFGs included); returns ``array[subset]`` per write.
    An array is converted only when its accumulations are the only reads of it in the loop: any other read would see
    the sums in an order the sequential loop does not have."""
    done = []
    for sd in sdfg.all_sdfgs_recursive():
        for loop in [r for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)]:
            for name, accs in _candidates(loop, _loop_writes(loop)).items():
                feeds = {id(feed) for _st, _e, _t, feed, _c in accs}
                if any(id(r) not in feeds for r in _reads(loop, name)):
                    continue
                for st, e, t, feed, contribution in accs:
                    t.code = dace.properties.CodeBlock(f"{e.src_conn} = {contribution}", t.language)
                    t.remove_in_connector(feed.dst_conn)
                    src = feed.src
                    st.remove_edge(feed)
                    if st.degree(src) == 0:
                        st.remove_node(src)
                    e.data.wcr = "lambda a, b: a + b"
                    done.append(f"{name}[{e.data.subset}]")
    return done
