# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Give each loop its own copy of a transient array that is local to every iteration of it.

``LoopToMap`` privatises a transient only when no state outside the loop touches it.  A Fortran work array
(``real(8) :: w(0:2)``) is typically shared by several loop nests of one routine -- one per direction, per scheme --
so each nest sees the others' accesses, treats the array as shared and stays sequential, although no value ever flows
from one nest to another.

:func:`privatize_loop_local_arrays` renames such an array per outermost loop when, in every loop using it, each read
is covered by a write earlier in the same iteration: then the loops cannot exchange values through it, and each copy
is local to its loop for ``LoopToMap`` to privatise.  The same holds for an array that one loop uses this way and
that code outside the loops only writes: nothing outside reads what the loop leaves in it (the copy-in of a callee's
``intent(out)`` argument the bridge stages before the loop, say).  Scalars are covered too: ``ScalarFission`` splits a scalar only
where one write dominates its reads and leaves it whole otherwise.  A write counts once some path through the iteration
makes it and names constant elements, a read (in dataflow or on an interstate edge) only if it names constant elements
already written, or any element once the whole array is; an array the check cannot decide is left as it is.  Some
path, not every one: the walk cannot see conditions that correlate (``flag = test`` before ``if (.not. flag)``), so a
read is taken to lie on the paths that prepare it, as ``LoopToMap`` takes it of a transient one loop uses.  A loop
that carries a value through a local only some paths write would be split wrongly; a read that no path prepares (what
an earlier loop or iteration left) keeps the array whole.  Loops that can never both run -- the branches of one
conditional, or blocks of one region
with no path between them -- cannot exchange values either, whatever the loops do with the array, as long as nothing
outside the loops touches it: each of those gets its own copy too.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
from math import prod

import sympy

import ast

from dace import SDFG, subsets, symbolic
from dace.frontend.python import astutils
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.utils import dfs_topological_sort

#: Element sets larger than this are not enumerated: such an access is treated as undecidable.
_MAX_ELEMENTS = 4096


def privatize_loop_local_arrays(sdfg: SDFG) -> dict[str, list[str]]:
    """Rename every transient array that is local to each iteration of the loops using it, one copy per outermost
    loop, in ``sdfg`` and its nested SDFGs.  Returns ``{original name: [names of the new copies]}``."""
    renamed: dict[str, list[str]] = {}
    for sd in sdfg.all_sdfgs_recursive():
        renamed.update(_privatize(sd))
    return renamed


def _rename_in_condition(condition: CodeBlock | None, name: str, new_name: str) -> CodeBlock | None:
    """``condition`` with every reference to ``name`` replaced.

    A structured control-flow condition (``ConditionalBlock.branches``) is an expression, not a dataflow access
    node: renaming the containers and the memlets does not reach it, and a stale name there becomes a free symbol
    at code generation (observed with a logical flag read by an ``if`` inside a mapped loop)."""
    if condition is None or name not in condition.get_free_symbols():
        return condition
    replacer = astutils.ASTFindReplace({name: new_name})
    try:
        tree = replacer.visit(ast.parse(condition.as_string))
    except SyntaxError:
        return condition
    if replacer.replace_count == 0:
        return condition
    return CodeBlock(astutils.unparse(tree), language=condition.language)


def _privatize(sdfg: SDFG) -> dict[str, list[str]]:
    # Per array and outermost loop: the blocks accessing it -- states through access nodes, and the source blocks of
    # interstate edges whose condition or assignments read an element of it.
    users: dict[str, dict[LoopRegion, list[ControlFlowBlock]]] = defaultdict(lambda: defaultdict(list))
    # Arrays read, and arrays only written, by code outside every loop.
    read_outside: set[str] = set()
    written_outside: set[str] = set()
    for state in sdfg.states():
        loop = _outermost_loop(state, sdfg)
        for node in state.data_nodes():
            if not sdfg.arrays[node.data].transient:
                continue
            if loop is not None:
                users[node.data][loop].append(state)
            elif state.out_degree(node):
                read_outside.add(node.data)
            elif state.in_degree(node):
                written_outside.add(node.data)
    for edge in sdfg.all_interstate_edges(recursive=True):
        loop = _outermost_loop(edge.src, sdfg)
        for read in edge.data.get_read_memlets(sdfg.arrays):
            if read.data not in sdfg.arrays or not sdfg.arrays[read.data].transient:
                continue
            if loop is None:
                read_outside.add(read.data)
            else:
                users[read.data][loop].append(edge.src)

    renamed: dict[str, list[str]] = {}
    for name, loops in users.items():
        # One loop and no other user: LoopToMap privatises it itself.
        if name in read_outside or (len(loops) < 2 and name not in written_outside):
            continue
        exclusive = name not in written_outside and _mutually_exclusive(list(loops), sdfg)
        if not exclusive and not all(_iteration_local(name, loop, blocks) for loop, blocks in loops.items()):
            continue
        copies = []
        # The writes outside keep the original name; every loop then needs a copy of its own.
        for loop in list(loops) if name in written_outside else list(loops)[1:]:
            new_name = sdfg.add_datadesc(name, sdfg.arrays[name].clone(), find_new_name=True)
            for state in loop.all_states():
                for node in state.data_nodes():
                    if node.data == name:
                        node.data = new_name
                for edge in state.edges():
                    if edge.data.data == name:
                        edge.data.data = new_name
            for edge in loop.all_interstate_edges(recursive=True):
                edge.data.replace(name, new_name, replace_keys=False)
            for block in loop.all_control_flow_blocks(recursive=True):
                if not isinstance(block, ConditionalBlock):
                    continue
                for i, (condition, branch) in enumerate(list(block.branches)):
                    renamed_condition = _rename_in_condition(condition, name, new_name)
                    if renamed_condition is not condition:
                        block.branches[i] = (renamed_condition, branch)
            copies.append(new_name)
        renamed[name] = copies
    return renamed


def _outermost_loop(block: ControlFlowBlock, top: SDFG) -> LoopRegion | None:
    found, region = None, block.parent_graph
    while region is not None and region is not top:
        if isinstance(region, LoopRegion):
            found = region
        region = region.parent_graph
    return found


def _mutually_exclusive(loops: list[LoopRegion], top: SDFG) -> bool:
    """Can no two of ``loops`` (outermost loops of ``top``) run in one execution of ``top``?"""
    return all(_exclusive(a, b, top) for a, b in combinations(loops, 2))


def _exclusive(a: ControlFlowBlock, b: ControlFlowBlock, top: SDFG) -> bool:
    """Do ``a`` and ``b`` sit in different branches of one conditional, or in blocks of one region with no path
    from either to the other?  Neither is inside a loop, so nothing runs the pair again."""
    chain_a, chain_b = _ancestry(a, top), _ancestry(b, top)
    common = next(x for x in chain_a if x in chain_b)
    ia, ib = chain_a.index(common), chain_b.index(common)
    if ia == 0 or ib == 0:
        return False  # one holds the other
    child_a, child_b = chain_a[ia - 1], chain_b[ib - 1]
    if isinstance(common, ConditionalBlock):
        return child_a is not child_b
    if isinstance(common, ControlFlowRegion):
        if not _reaches(common, child_a, child_b) and not _reaches(common, child_b, child_a):
            return True
        # Two conditionals in sequence whose branches (the ones holding the loops) can never both be taken:
        # ``if (idir == 1) ... end if; if (idir == 2) ... end if``.
        if isinstance(child_a, ConditionalBlock) and isinstance(child_b, ConditionalBlock) and ia >= 2 and ib >= 2:
            return _branches_contradict(child_a, chain_a[ia - 2], child_b, chain_b[ib - 2], common)
    return False


def _branches_contradict(block_a: ConditionalBlock, branch_a: ControlFlowRegion, block_b: ConditionalBlock,
                         branch_b: ControlFlowRegion, region: ControlFlowRegion) -> bool:
    """Can the conditions of ``branch_a`` of ``block_a`` and ``branch_b`` of ``block_b`` not both hold?  Only when
    sympy reduces their conjunction to false and no edge of ``region`` assigns a symbol they read (an ``else``
    branch has no condition of its own: never)."""
    conditions = []
    for block, branch in ((block_a, branch_a), (block_b, branch_b)):
        condition = next((c for c, b in block.branches if b is branch), None)
        if condition is None:
            return False
        conditions.append(condition.as_string)
    # A condition may be a symbol an edge assigns the test to (``if_cond_1 = idir == 1``): one assigned once in the
    # whole SDFG stands for its expression.
    assigned_once: dict[str, list[str]] = defaultdict(list)
    for edge in region.sdfg.all_interstate_edges(recursive=True):
        for sym, value in edge.data.assignments.items():
            assigned_once[sym].append(value)
    definitions = {sym: values[0] for sym, values in assigned_once.items() if len(values) == 1}
    try:
        terms = []
        for condition in conditions:
            term = symbolic.pystr_to_symbolic(condition)
            for sym in list(term.free_symbols):
                if str(sym) in definitions:
                    term = term.subs(sym, symbolic.pystr_to_symbolic(definitions[str(sym)]))
            terms.append(term)
        both = sympy.And(*terms)
        if sympy.simplify(both) is not sympy.false:
            return False
    except (TypeError, ValueError, AttributeError, sympy.SympifyError):
        return False
    read = set(map(str, both.free_symbols))
    return not any(read & edge.data.assignments.keys() for edge in region.all_interstate_edges(recursive=True))


def _ancestry(block: ControlFlowBlock, top: SDFG) -> list:
    """``block``, its parent, and so on up to ``top``."""
    chain = [block]
    while chain[-1] is not top:
        chain.append(chain[-1].parent_graph)
    return chain


def _reaches(region: ControlFlowRegion, src: ControlFlowBlock, dst: ControlFlowBlock) -> bool:
    seen, stack = {src}, [src]
    while stack:
        for nxt in region.successors(stack.pop()):
            if nxt is dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _loop_chain(block: ControlFlowBlock, stop: LoopRegion) -> list[LoopRegion]:
    """The loop regions enclosing ``block``, innermost first, up to and including ``stop``."""
    chain, region = [], block.parent_graph
    while region is not None:
        if isinstance(region, LoopRegion):
            chain.append(region)
        if region is stop:
            break
        region = region.parent_graph
    return chain


def _elements(subset: subsets.Subset | None) -> set[tuple[int, ...]] | None:
    """The elements ``subset`` names, when its bounds are constants; ``None`` otherwise."""
    if subset is None:
        return None
    ranges = []
    try:
        for begin, end, step in subset.ndrange():
            ranges.append(range(int(begin), int(end) + 1, int(step)))
    except TypeError:
        return None
    if prod(len(r) for r in ranges) > _MAX_ELEMENTS:
        return None
    return set(product(*ranges))


def _iteration_local(name: str, loop: LoopRegion, blocks: list[ControlFlowBlock]) -> bool:
    """Is every read of ``name`` in ``loop`` covered by a write earlier in the same iteration, on some path, of the
    innermost loop that holds all its accesses (``blocks``: the states and edge sources accessing it)?"""
    chains = [_loop_chain(block, loop) for block in blocks]
    common = set(chains[0])
    for chain in chains[1:]:
        common &= set(chain)
    body = next(region for region in chains[0] if region in common)  # innermost loop holding every access
    return _written_after(body, name, set()) is not None


def _written_after(region: ControlFlowRegion, name: str, written: set) -> set | None:
    """The elements of ``name`` written on some path once ``region`` has run, given those written before it; ``None``
    when some read in it is not covered.  A conditional's branches each start from ``written``, and what any of them
    writes counts after it; so do an inner loop's writes, whose reads must be covered within its iteration."""
    if any(not e.data.is_unconditional() for e in region.edges()):
        return None  # branches as conditional edges: which blocks run is not plain to see
    whole = _whole(region.sdfg, name)
    for block in cfg_analysis.blockorder_topological_sort(region, recursive=False, ignore_nonstate_blocks=False):
        if isinstance(block, SDFGState):
            if not _state_covered(block, name, written):
                return None
        elif isinstance(block, ConditionalBlock):
            # A branch condition is a read: it sees what the writes before the conditional left.
            for condition, _branch in block.branches:
                if condition is not None and name in condition.get_free_symbols():
                    if not _covered(whole, written, None):
                        return None
            after = set(written)
            for _condition, branch in block.branches:
                branch_written = _written_after(branch, name, set(written))
                if branch_written is None:
                    return None
                after |= branch_written
            written = after
        elif isinstance(block, LoopRegion):
            inner = _written_after(block, name, set(written))
            if inner is None:
                return None
            written = inner
        elif isinstance(block, ControlFlowRegion):
            inner = _written_after(block, name, written)
            if inner is None:
                return None
            written = inner
        else:
            return None  # a block kind the walk does not know
        # The reads of the edges leaving ``block`` come after its writes; ``written`` does not change along them.
        for edge in region.out_edges(block):
            for read in edge.data.get_read_memlets(region.sdfg.arrays):
                if read.data == name and not _covered(whole, written, _elements(read.subset)):
                    return None
    return written


def _whole(sdfg: SDFG, name: str) -> set[tuple[int, ...]] | None:
    """Every element of ``name``, when its shape is constant."""
    return _elements(subsets.Range.from_array(sdfg.arrays[name]))


def _covered(whole: set | None, written: set, elems: set | None) -> bool:
    """Is a read of ``elems`` covered by ``written``?  A read whose element cannot be named (``w(idx)`` with ``idx``
    a symbol) is covered once every element of the array has been written."""
    if elems is None:
        return whole is not None and whole <= written
    return elems <= written


def _state_covered(state: SDFGState, name: str, written: set) -> bool:
    """Walk ``state``'s accesses to ``name`` in dataflow order: adds the elements it certainly writes to ``written``,
    and says whether each read is covered."""
    whole = _whole(state.sdfg, name)
    for node in dfs_topological_sort(state):
        if not isinstance(node, nodes.AccessNode) or node.data != name:
            continue
        for edge in state.in_edges(node):
            if edge.data.is_empty():
                continue
            elems = _elements(edge.data.get_dst_subset(edge, state))
            if edge.data.wcr is not None and not _covered(whole, written, elems):
                return False  # an accumulation reads the element it updates
            # A nested SDFG's memlet may cover more than it writes: only a direct write is certain.
            if not edge.data.dynamic and not isinstance(edge.src, nodes.NestedSDFG) and elems is not None:
                written |= elems
        for edge in state.out_edges(node):
            if edge.data.is_empty():
                continue
            if not _covered(whole, written, _elements(edge.data.get_src_subset(edge, state))):
                return False
    return True
