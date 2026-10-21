"""Which arrays an SDFG never reads a value of from before its own execution started.

:func:`uncovered_reads` walks the SDFG's control flow forward and keeps, per array, the boxes of elements written on
every path so far (*must-write* sets).  A read of an array is *covered* when its subset lies in the union of those
boxes.  An array whose every read is covered takes no value in from the caller: the SDFG may treat it as scratch.

The analysis is conservative: anything it cannot prove leaves a read uncovered.

* Subsets are boxes of inclusive bounds, symbolic.  Comparisons are decided over the ranges of the enclosing loop
  variables and map parameters (a bound linear in them is evaluated at the extreme that decides it); a comparison
  that still depends on other symbols is decided over the combinations of values they can take (``cases``: one per
  rank of a recorded run, whose binding admits no other), and undecided otherwise, as is anything non-linear.  Whether a union of boxes holds a box
  is decided by subtracting one box after the other.
* A state: a read at an access node is covered by the boxes written before the state and by the writes into the access
  nodes of the same array it is reachable from (its own in-edges included).  The map scopes a read and a write share
  keep their parameters (the same iteration); the write's other scopes are expanded exactly.  Copies, tasklets and
  reductions write their whole output memlet; a dynamic or write-conflict-resolved memlet, or another library node,
  writes nothing certain (a write-conflict resolution, or a reduction without identity, also reads its output).
* A loop: its body sees the boxes from before the loop (minus those naming symbols the body reassigns), never those of
  earlier iterations.  After it, a box its body writes in every iteration is expanded over the iteration space when
  that is provably non-empty, the step is one and the box is affine in the loop variable with coefficient one in a
  single dimension; otherwise it is dropped.  A body with ``break``, ``continue`` or ``return`` writes nothing certain.
* Branches: the intersection over the branches (a conditional without ``else`` keeps what held before it).  Interstate
  assignments: a symbol assigned an expression of symbols takes that value; reassigning a symbol drops the boxes and
  values naming it.  Conditions, loop headers and assignments that read an array are reads.
* A nested SDFG: analysed in place, its connectors mapped to the outer arrays through the outer memlets (same rank, or
  the outer subset's unit dimensions squeezed); otherwise its inputs are reads of their outer subsets and its outputs
  writes of nothing certain.
* An array a view or reference is attached to is not tracked: it counts as read uncovered.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import sympy

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion, ReturnBlock,
                             SDFGState)
from dace.transformation.passes.analysis import loop_analysis

Box = tuple  # ((lo, hi), ...) of sympy expressions, inclusive bounds

_fresh = itertools.count()


def _norm(expr, versions: dict | None = None, contents: dict | None = None) -> sympy.Expr | None:
    """``expr`` over plain symbols (no assumptions), or None if it calls a function or reads an array.

    With ``versions`` (array -> how often it has been written so far), an array element at a constant index is a
    value of its own: its value if ``contents`` (array -> its elements, flat) knows it, else the symbol
    ``a[0]@<version>``, the same for every read until ``a`` is written again."""
    try:
        e = symbolic.pystr_to_symbolic(expr) if isinstance(expr, str) else sympy.sympify(expr)
    except Exception:  # noqa: BLE001 - anything unparsable is unknown
        return None
    if not isinstance(e, sympy.Basic):
        return None
    if versions is not None:
        elements = {}
        for f in e.atoms(sympy.Function):
            if (type(f).__name__ == "Subscript" and f.args and isinstance(f.args[0], sympy.Symbol)
                    and all(a.is_Integer for a in f.args[1:])):
                name = f.args[0].name
                index = ", ".join(str(a) for a in f.args[1:])
                known = (contents or {}).get(name)
                if known is not None and len(f.args) == 2 and 0 <= int(f.args[1]) < len(known):
                    elements[f] = sympy.sympify(known[int(f.args[1])])
                else:
                    elements[f] = sympy.Symbol(f"{name}[{index}]@{versions.get(name, 0)}")
        if elements:
            e = e.xreplace(elements)
    if e.atoms(sympy.Indexed) or any(not isinstance(f, (sympy.Min, sympy.Max)) for f in e.atoms(sympy.Function)):
        return None
    return e.xreplace({s: sympy.Symbol(s.name) for s in e.free_symbols if isinstance(s, sympy.Symbol)})


def _written_arrays(region) -> set[str]:
    """The containers some state of ``region`` (an SDFG or a control flow region) writes."""
    return {n.data for st in region.all_states() for n in st.data_nodes()
            if any(not e.data.is_empty() for e in st.in_edges(n))}


@dataclass
class Facts:
    """Ranges of the loop variables and map parameters in scope, outermost first (inclusive; None: unbounded)."""

    ranges: dict = field(default_factory=dict)  # sympy.Symbol -> (lo, hi)
    cases: tuple = ()  # the combinations of values the other symbols can take: ({name: value}, ...)

    def with_range(self, var, lo, hi) -> "Facts":
        r = dict(self.ranges)
        r[var] = (lo, hi)
        return Facts(r, self.cases)

    def nonneg(self, expr) -> bool:
        """``expr >= 0`` for every value of the variables in their ranges (False when undecided)."""
        e = sympy.expand(expr)
        for var in reversed(list(self.ranges)):
            if var not in e.free_symbols:
                continue
            coeff = e.coeff(var, 1)
            rest = sympy.expand(e - coeff * var)
            if not coeff.is_number or var in rest.free_symbols:
                return False
            lo, hi = self.ranges[var]
            bound = lo if coeff > 0 else hi
            if bound is None:
                return False
            e = sympy.expand(rest + coeff * bound)
        if e.is_number:
            return bool(e >= 0)
        names = {str(s) for s in e.free_symbols}
        if not self.cases or any(not names <= case.keys() for case in self.cases):
            return False
        for case in self.cases:
            value = e.xreplace({s: sympy.sympify(case[str(s)]) for s in e.free_symbols})
            if not value.is_number or not bool(value >= 0):
                return False
        return True

    def le(self, a, b) -> bool:
        return self.nonneg(b - a)

    def lt(self, a, b) -> bool:
        return self.nonneg(b - a - 1)


def _subtract(r: Box, w: Box, facts: Facts) -> list[Box] | None:
    """``r`` minus ``w`` as a list of boxes, or None when a comparison it needs is undecided."""
    if any(facts.lt(rh, wl) or facts.lt(wh, rl) for (rl, rh), (wl, wh) in zip(r, w)):
        return [r]  # disjoint
    pieces = []
    cur = list(r)
    for d, (wl, wh) in enumerate(w):
        rl, rh = cur[d]
        if not facts.le(wl, rl):
            if not facts.lt(rl, wl):
                return None
            pieces.append(tuple(cur[:d] + [(rl, wl - 1)] + cur[d + 1:]))
            rl = wl
        if not facts.le(rh, wh):
            if not facts.lt(wh, rh):
                return None
            pieces.append(tuple(cur[:d] + [(wh + 1, rh)] + cur[d + 1:]))
            rh = wh
        cur[d] = (rl, rh)
    return pieces


def _subtract_all(boxes: list[Box], written: Iterable[Box], facts: Facts) -> list[Box]:
    remaining = list(boxes)
    for w in written:
        nxt = []
        for b in remaining:
            diff = _subtract(b, w, facts) if len(w) == len(b) else None
            nxt.extend([b] if diff is None else diff)
        # A piece provably empty for every value in range is gone.
        remaining = [b for b in nxt if not any(facts.lt(hi, lo) for lo, hi in b)]
        if not remaining:
            break
    return remaining


def _hull(box: Box, var, lo, hi) -> Box | None:
    """The smallest box holding ``box`` for every ``var`` in [lo, hi]; None if a bound is not linear in ``var``."""
    out = []
    for a, b in box:
        na, nb = _extreme(a, var, lo, hi, low=True), _extreme(b, var, lo, hi, low=False)
        if na is None or nb is None:
            return None
        out.append((na, nb))
    return tuple(out)


def _extreme(e, var, lo, hi, low: bool):
    if var not in e.free_symbols:
        return e
    coeff = sympy.expand(e).coeff(var, 1)
    rest = sympy.expand(e - coeff * var)
    if not coeff.is_number or var in rest.free_symbols:
        return None
    return sympy.expand(rest + coeff * (lo if (coeff > 0) == low else hi))


def covered(r: Box, written: Iterable[Box], facts: Facts) -> bool:
    """Whether the union of the boxes ``written`` provably holds the box ``r``.

    Where the comparisons depend on a loop variable or map parameter, the part of ``r`` the boxes of the same
    iteration leave is widened over that variable's range and compared with the boxes that do not depend on it:
    a union of boxes written before the loop (an interior and its halos) then covers an element read in it."""
    written = list(written)
    remaining = _subtract_all([r], written, facts)
    for var in reversed(list(facts.ranges)):
        if not remaining:
            return True
        if not any(_mentions(b, {var}) for b in remaining):
            continue
        lo, hi = facts.ranges[var]
        if lo is None or hi is None:
            return False
        hulls = [_hull(b, var, lo, hi) for b in remaining]
        if any(h is None for h in hulls):
            return False
        written = [w for w in written if not _mentions(w, {var})]
        remaining = _subtract_all(hulls, written, facts)
    return not remaining


def _box_of(subset, subs: dict) -> Box | None:
    """A memlet subset as a box with ``subs`` applied; None if a bound is not an expression of symbols, or a range
    of more than one element has a step other than one (its hull would be read as more, written as more)."""
    if subset is None:
        return None
    out = []
    for rng in subset.ndrange():
        lo, hi, step = (_norm(x) for x in rng)
        if lo is None or hi is None or step is None:
            return None
        lo, hi = sympy.expand(lo.xreplace(subs)), sympy.expand(hi.xreplace(subs))
        if step != 1 and sympy.expand(hi - lo) != 0:
            return None
        out.append((lo, hi))
    return tuple(out)


def _write_union(box: Box, var, lo, hi, facts: Facts) -> Box | None:
    """The union of ``box`` over ``var`` in [lo, hi] (step one) when that is exactly a box, else None."""
    if not any(var in (a.free_symbols | b.free_symbols) for a, b in box):
        return box
    if lo is None or hi is None or not facts.le(lo, hi):
        return None
    out, used = [], False
    for a, b in box:
        if var not in (a.free_symbols | b.free_symbols):
            out.append((a, b))
            continue
        if used:
            return None  # the variable in two dimensions: a diagonal, not a box
        used = True
        ca, cb = sympy.expand(a).coeff(var, 1), sympy.expand(b).coeff(var, 1)
        ra, rb = sympy.expand(a - ca * var), sympy.expand(b - cb * var)
        if var in (ra.free_symbols | rb.free_symbols) or ca != cb or ca not in (1, -1):
            return None
        if not facts.le(ra, rb):
            return None  # an iteration might write nothing: the union might have holes
        if ca == 1:
            out.append((sympy.expand(ra + lo), sympy.expand(rb + hi)))
        else:
            out.append((sympy.expand(ra - hi), sympy.expand(rb - lo)))
    return tuple(out)


def _mentions(box: Box, syms: set) -> bool:
    return any((a.free_symbols | b.free_symbols) & syms for a, b in box)


def _show(box) -> str:
    if box is None:
        return "an unbounded subset"
    return "[" + ", ".join(f"{a}" if a == b else f"{a}:{b}" for a, b in box) + "]"


@dataclass
class Env:
    """The abstract state at a point of the control flow: must-written boxes per tracked array, symbol values."""

    written: dict = field(default_factory=dict)  # name -> list[Box]
    defs: dict = field(default_factory=dict)  # sympy.Symbol -> expression, or None when unknown
    versions: dict = field(default_factory=dict)  # array -> version of its contents (bumped by every write)
    contents: dict = field(default_factory=dict)  # small array -> its elements (flat), when a constant was copied in

    def copy(self) -> "Env":
        return Env({k: list(v) for k, v in self.written.items()}, dict(self.defs), dict(self.versions),
                   dict(self.contents))

    def bump(self, arrays: Iterable[str]) -> None:
        for a in arrays:
            self.versions[a] = next(_fresh)
            self.contents.pop(a, None)

    def add(self, name: str, box: Box, facts: Facts) -> None:
        boxes = self.written.setdefault(name, [])
        if not covered(box, boxes, facts):
            boxes.append(box)

    def forget(self, syms: set) -> None:
        """The symbols ``syms`` are reassigned: the boxes and values naming them no longer hold."""
        for name in self.written:
            self.written[name] = [b for b in self.written[name] if not _mentions(b, syms)]
        for s, v in list(self.defs.items()):
            if s in syms or (v is not None and v.free_symbols & syms):
                self.defs[s] = None

    def values(self) -> dict:
        return {s: v for s, v in self.defs.items() if v is not None}


def _meet(envs: list[Env], facts: Facts) -> Env:
    """What holds after each of ``envs``: the boxes every one of them covers, the symbol values they agree on."""
    out = Env()
    for name, boxes in envs[0].written.items():
        keep = [b for b in boxes if all(covered(b, e.written.get(name, []), facts) for e in envs[1:])]
        if keep:
            out.written[name] = keep
    for s in set().union(*(e.defs.keys() for e in envs)):
        vals = [e.defs.get(s) for e in envs]
        out.defs[s] = vals[0] if vals[0] is not None and all(v == vals[0] for v in vals) else None
    for a in set().union(*(e.versions.keys() for e in envs)):
        vals = [e.versions.get(a, 0) for e in envs]
        out.versions[a] = vals[0] if all(v == vals[0] for v in vals) else next(_fresh)
    for a, vals in envs[0].contents.items():
        if all(e.contents.get(a) == vals for e in envs[1:]):
            out.contents[a] = vals
    return out


@dataclass
class _Ctx:
    sdfg: dace.SDFG
    tracked: set  # array names of this SDFG followed
    subs: dict = field(default_factory=dict)  # symbol -> expression: renamed loop variables, nested symbol mappings
    constants: dict = field(default_factory=dict)  # constant array -> its elements (flat)

    def __post_init__(self):
        if not self.constants:
            self.constants = _constant_contents(self.sdfg)

    def with_sub(self, var, value) -> "_Ctx":
        return _Ctx(self.sdfg, self.tracked, {**self.subs, var: value}, self.constants)


def _constant_contents(sdfg: dace.SDFG) -> dict:
    """The one-dimensional integer constants of ``sdfg`` (literal arrays), as tuples."""
    out = {}
    for name, (desc, value) in sdfg.constants_prop.items():
        try:
            flat = [int(v) for v in list(getattr(value, "flat", value))]
        except (TypeError, ValueError):
            continue
        if isinstance(desc, dace.data.Array) and len(desc.shape) == 1 and len(flat) <= 64:
            out[name] = tuple(flat)
    return out


def _subset_of(edge, data: str):
    """The subset of ``data`` an edge accesses; None when the memlet names the other array without saying where in
    ``data`` (a copy's destination subset left implicit)."""
    m = edge.data
    if m.data == data:
        return m.subset
    return m.other_subset


class _Analysis:
    def __init__(self):
        self.reasons: dict[str, list[str]] = {}
        self._params: dict = {}  # (map entry, parameter) -> fresh symbol

    def uncovered(self, name: str, where: str, box) -> None:
        msgs = self.reasons.setdefault(name, [])
        if len(msgs) < 3:
            msgs.append(f"{where}: reads {_show(box)} not provably written before")

    def check(self, name: str, box, written: Iterable[Box], facts: Facts, where: str) -> None:
        written = list(written)
        if box is None or not covered(box, written, facts):
            # The symbols the comparisons met that are neither loop variables nor given values: the read box, the
            # boxes written, and the ranges of the loop variables in scope.
            known = {str(v) for v in facts.ranges} | {k for case in facts.cases for k in case}
            pairs = [*(box or ()), *(pair for w in written for pair in w), *facts.ranges.values()]
            others = sorted({str(s) for pair in pairs for e in pair if e is not None
                             for s in getattr(e, "free_symbols", ())} - known)
            self.uncovered(name, where + (f" (over the symbols {', '.join(others[:6])})" if others else ""), box)

    def values(self, env: Env, ctx: _Ctx) -> dict:
        return {**ctx.subs, **env.values()}

    # -- control flow -----------------------------------------------------------------------------------------------

    def region(self, region: ControlFlowRegion, env: Env, facts: Facts, ctx: _Ctx) -> Env:
        """Analyse ``region`` entered with ``env``; returns what holds when it is left normally."""
        graph = region.nx
        if not nx.is_directed_acyclic_graph(graph):
            # Unstructured cycles: no write inside counts, every read is checked against what holds on entry, less
            # what any pass through the cycle may have changed.
            inner = env.copy()
            inner.forget({sympy.Symbol(k) for e in region.all_interstate_edges() for k in e.data.assignments})
            inner.bump(_written_arrays(region))
            for blk in region.nodes():
                self.block(blk, inner.copy(), facts, ctx)
            for e in region.edges():
                self.edge(e, inner, facts, ctx)
            return inner
        after: dict = {}
        for blk in nx.topological_sort(graph):
            if blk is region.start_block:
                incoming = [env]
            else:
                incoming = [self.edge(e, after[e.src], facts, ctx) for e in region.in_edges(blk) if e.src in after]
                if not incoming:
                    continue  # unreachable
            here = incoming[0] if len(incoming) == 1 else _meet(incoming, facts)
            after[blk] = self.block(blk, here, facts, ctx)
        ends = [after[b] for b in region.nodes() if region.out_degree(b) == 0 and b in after
                and not isinstance(b, (ReturnBlock, BreakBlock, ContinueBlock))]
        if not ends:
            return env
        return ends[0] if len(ends) == 1 else _meet(ends, facts)

    def edge(self, e, env: Env, facts: Facts, ctx: _Ctx) -> Env:
        self.code_reads(e.data, env, facts, ctx, f"interstate edge {e.src.label} -> {e.dst.label}")
        if not e.data.assignments:
            return env
        out = env.copy()
        values = self.values(env, ctx)
        new = {}
        for k, v in e.data.assignments.items():
            val = _norm(v, env.versions, {**ctx.constants, **env.contents})
            new[sympy.Symbol(k)] = None if val is None else sympy.expand(val.xreplace(values))
        out.forget(set(new))
        out.defs.update(new)
        return out

    def code_reads(self, isedge, env: Env, facts: Facts, ctx: _Ctx, where: str) -> None:
        """The array elements a condition or assignment reads must have been written."""
        try:
            memlets = isedge.get_read_memlets(ctx.sdfg.arrays)
        except Exception:  # noqa: BLE001 - unparsable: reads whatever tracked array it names
            text = isedge.condition.as_string + " " + " ".join(isedge.assignments.values())
            for n in ctx.tracked:
                if n in text:
                    self.uncovered(n, where, None)
            return
        for m in memlets:
            if m.data in ctx.tracked:
                self.check(m.data, _box_of(m.subset, self.values(env, ctx)), env.written.get(m.data, []), facts,
                           where)

    def block(self, blk, env: Env, facts: Facts, ctx: _Ctx) -> Env:
        if isinstance(blk, SDFGState):
            return self.state(blk, env, facts, ctx)
        if isinstance(blk, LoopRegion):
            return self.loop(blk, env, facts, ctx)
        if isinstance(blk, ConditionalBlock):
            outs, has_else = [], False
            for cond, branch in blk.branches:
                if cond is None or cond.as_string.strip() in ("1", "True"):
                    has_else = True
                else:
                    self.code_reads(dace.InterstateEdge(condition=cond.as_string), env, facts, ctx,
                                    f"condition of {blk.label}")
                outs.append(self.region(branch, env.copy(), facts, ctx))
            if not has_else:
                outs.append(env)
            return _meet(outs, facts)
        if isinstance(blk, ControlFlowRegion):
            return self.region(blk, env, facts, ctx)
        return env  # break, continue, return

    def loop(self, loop: LoopRegion, env: Env, facts: Facts, ctx: _Ctx) -> Env:
        where = f"loop header of {loop.label}"
        if loop.loop_condition is not None:
            self.code_reads(dace.InterstateEdge(condition=loop.loop_condition.as_string), env, facts, ctx, where)
        for code in (loop.init_statement, loop.update_statement):
            if code is not None and "=" in code.as_string:
                rhs = code.as_string.split("=", 1)[1]
                self.code_reads(dace.InterstateEdge(assignments={"__rhs": rhs}), env, facts, ctx, where)
        assigned = {sympy.Symbol(k) for e in loop.all_interstate_edges() for k in e.data.assignments}
        var = sympy.Symbol(loop.loop_variable) if loop.loop_variable else None
        lo = hi = None
        exact = False
        if var is not None:
            assigned.add(var)
            start, end, stride = (loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop),
                                  loop_analysis.get_loop_stride(loop))
            start, end, stride = (None if x is None else _norm(x) for x in (start, end, stride))
            if start is not None and end is not None and stride is not None and stride.is_number and stride != 0:
                values = self.values(env, ctx)
                start, end = sympy.expand(start.xreplace(values)), sympy.expand(end.xreplace(values))
                lo, hi = (start, end) if stride > 0 else (end, start)
                exact = abs(stride) == 1
                if loop.inverted and not facts.le(lo, hi):
                    lo = hi = None  # a do-while running once past its bound
        body = env.copy()
        body.forget(assigned)
        changed = _written_arrays(loop)
        body.bump(changed)  # a later iteration sees what earlier ones wrote
        body_facts, body_ctx, inner = facts, ctx, None
        if var is not None:
            # A fresh name: the variable of a loop in a nested SDFG may repeat an outer name.
            inner = sympy.Symbol(f"{var.name}__{next(_fresh)}")
            body_ctx = ctx.with_sub(var, inner)
            body_facts = facts.with_range(inner, lo, hi)
        out = self.region(loop, body, body_facts, body_ctx)
        result = env.copy()
        result.forget(assigned)
        result.bump(changed)
        leaves = any(isinstance(b, (BreakBlock, ContinueBlock, ReturnBlock)) for b in loop.all_control_flow_blocks())
        if inner is None or leaves or not exact or lo is None or not facts.le(lo, hi):
            return result
        for name, boxes in out.written.items():
            before = body.written.get(name, [])
            for box in boxes:
                if any(box is b for b in before) or _mentions(box, assigned):
                    continue
                union = _write_union(box, inner, lo, hi, body_facts)
                if union is not None and not _mentions(union, {inner}):
                    result.add(name, union, facts)
        return result

    # -- dataflow ---------------------------------------------------------------------------------------------------

    def state(self, state: SDFGState, env: Env, facts: Facts, ctx: _Ctx) -> Env:
        anodes = [n for n in state.data_nodes() if n.data in ctx.tracked]
        nested = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]
        if not anodes and not nested:
            changed = {n.data for n in state.data_nodes() if any(not e.data.is_empty() for e in state.in_edges(n))}
            if not changed:
                return env
            out = env.copy()
            out.bump(changed)
            out.contents.update(_copied_constants(state, ctx.constants))
            return out
        values = self.values(env, ctx)
        scopes = state.scope_dict()
        chains: dict = {}

        def chain(node) -> tuple:
            if node not in chains:
                out, s = [], scopes[node]
                while s is not None:
                    out.append(s)
                    s = scopes[s]
                chains[node] = tuple(reversed(out))
            return chains[node]

        scope_facts: dict = {}

        def scope(ch: tuple):
            """Facts and parameter substitutions inside the scopes ``ch``; (None, None) for a non-map scope."""
            if ch not in scope_facts:
                f, subs = facts, dict(values)
                for entry in ch:
                    if not isinstance(entry, nodes.MapEntry):
                        scope_facts[ch] = (None, None)
                        return scope_facts[ch]
                    fresh = {}
                    for p in entry.map.params:
                        key = (entry, p)
                        if key not in self._params:
                            self._params[key] = sympy.Symbol(f"{p}__{next(_fresh)}")
                        fresh[sympy.Symbol(p)] = self._params[key]
                    for p, rng in zip(entry.map.params, entry.map.range.ndrange()):
                        a, b, s = (_norm(x) for x in rng)
                        sym = fresh[sympy.Symbol(p)]
                        if a is None or b is None or s != 1 or entry.map.range is None:
                            f = f.with_range(sym, None, None)
                        else:
                            f = f.with_range(sym, sympy.expand(a.xreplace(subs)), sympy.expand(b.xreplace(subs)))
                    subs.update(fresh)
                scope_facts[ch] = (f, subs)
            return scope_facts[ch]

        writes: list = []  # (array, box, scope chain, access node, certain)
        reads: list = []  # (array, box, scope chain, access node, where)
        where = f"state {state.label}"

        def write(edge, producer, node, ch) -> None:
            if edge.data.is_empty():
                return  # an ordering edge
            f, subs = scope(ch)
            if f is None:
                writes.append((node.data, None, ch, node, False))
                return
            m = edge.data
            box = _box_of(_subset_of(edge, node.data), subs)
            certain = not m.dynamic and m.wcr is None
            if isinstance(producer, nodes.LibraryNode):
                from dace.libraries.standard.nodes import Reduce
                from dace.libraries.standard.nodes.copy.node import CopyLibraryNode

                if isinstance(producer, Reduce):
                    if producer.identity is None:
                        reads.append((node.data, box, ch, node, where))
                elif not isinstance(producer, CopyLibraryNode):
                    certain = False
            if m.wcr is not None:
                reads.append((node.data, box, ch, node, where))
            writes.append((node.data, box, ch, node, certain))

        for n in anodes:
            for e in state.in_edges(n):
                if isinstance(e.src, (nodes.EntryNode, nodes.NestedSDFG)):
                    continue  # data passed into a scope; nested SDFGs come below
                if isinstance(e.src, nodes.ExitNode):
                    for leaf in state.memlet_tree(e).leaves():
                        if not isinstance(leaf.src, (nodes.AccessNode, nodes.NestedSDFG)):
                            write(leaf, leaf.src, n, chain(leaf.src))
                    continue
                write(e, e.src, n, chain(n))
            for e in state.out_edges(n):
                if isinstance(e.dst, nodes.ExitNode):
                    continue
                leaves = state.memlet_tree(e).leaves() if isinstance(e.dst, nodes.EntryNode) else [e]
                for leaf in leaves:
                    if leaf.data.is_empty():
                        continue  # an ordering edge
                    if isinstance(leaf.dst, nodes.NestedSDFG) or (isinstance(leaf.dst, nodes.AccessNode) and leaf is not e):
                        continue  # nested SDFGs below; an inner access node's reads are its own
                    ch = chain(n) if leaf is e else chain(leaf.dst)
                    f, subs = scope(ch)
                    box = None if f is None else _box_of(_subset_of(leaf, n.data), subs)
                    reads.append((n.data, box, ch, n, where))

        graph = state.nx
        ancestry: dict = {}

        def available(arr, node, ch) -> list:
            """Boxes of ``arr`` written before a read at ``node`` in the scopes ``ch``, in this state."""
            if node not in ancestry:
                ancestry[node] = nx.ancestors(graph, node) | {node}
            anc = ancestry[node]
            out = []
            for warr, wbox, wch, wnode, certain in writes:
                if warr != arr or not certain or wbox is None or wnode not in anc:
                    continue
                common = 0
                while common < min(len(ch), len(wch)) and ch[common] is wch[common]:
                    common += 1
                ex = self.expand(wbox, wch, common, scope)
                if ex is not None:
                    out.append(ex)
            return out

        for nst in [n for n in nx.topological_sort(state.nx) if isinstance(n, nodes.NestedSDFG)]:
            self.nested(state, nst, env, ctx, chain, scope, available, writes, reads)

        for arr, box, ch, node, wh in reads:
            f, _ = scope(ch)
            self.check(arr, box, env.written.get(arr, []) + available(arr, node, ch), f or facts, wh)

        out = env.copy()
        out.bump({n.data for n in state.data_nodes() if any(not e.data.is_empty() for e in state.in_edges(n))})
        out.contents.update(_copied_constants(state, ctx.constants))
        for warr, wbox, wch, _wnode, certain in writes:
            if certain and wbox is not None:
                ex = self.expand(wbox, wch, 0, scope)
                if ex is not None:
                    out.add(warr, ex, facts)
        return out

    def expand(self, box: Box, ch: tuple, keep: int, scope) -> Box | None:
        """The exact union of the write ``box`` over the map scopes ``ch[keep:]`` (innermost first), or None."""
        cur = box
        for depth in range(len(ch), keep, -1):
            f, subs = scope(ch[:depth])
            if f is None:
                return None
            for p in reversed(ch[depth - 1].map.params):
                var = subs[sympy.Symbol(p)]
                lo, hi = f.ranges[var]
                cur = _write_union(cur, var, lo, hi, f)
                if cur is None:
                    return None
        return cur

    def nested(self, state, nst, env, ctx, chain, scope, available, writes, reads) -> None:
        """A nested SDFG: analysed with what holds at its inputs; its certain writes become writes of the state."""
        ch = chain(nst)
        f, subs = scope(ch)
        where = f"nested SDFG {nst.label} in state {state.label}"
        inputs, outputs = {}, {}
        for e in state.in_edges(nst) + state.out_edges(nst):
            incoming = e.dst is nst
            conn = e.dst_conn if incoming else e.src_conn
            path = state.memlet_path(e)
            outer = path[0].src if incoming else path[-1].dst
            if conn is None or e.data.is_empty() or not isinstance(outer, nodes.AccessNode) or outer.data not in ctx.tracked:
                continue
            box = None if f is None else _box_of(e.data.subset, subs)
            inner = nst.sdfg.arrays.get(conn)
            keep = None
            if box is not None and isinstance(inner, dace.data.Array) and not isinstance(
                    inner, (dace.data.View, dace.data.Reference)):
                keep = list(range(len(box)))
                if len(inner.shape) < len(box):
                    units = [d for d, (a, b) in enumerate(box) if f.le(a, b) and f.le(b, a)]
                    keep = [d for d in keep if d not in units] if len(box) - len(inner.shape) == len(units) else None
                if keep is not None and len(keep) != len(inner.shape):
                    keep = None
            if keep is None:
                if incoming:
                    reads.append((outer.data, box, ch, outer, where))
                else:
                    writes.append((outer.data, None, ch, outer, False))
                continue
            (inputs if incoming else outputs)[conn] = (outer, box, keep)
        if not inputs and not outputs:
            return
        sub_ctx = _Ctx(nst.sdfg, set(inputs) | set(outputs), {
            **ctx.subs, **{sympy.Symbol(k): sympy.expand(v.xreplace(subs)) for k, v in
                           ((k, _norm(v)) for k, v in nst.symbol_mapping.items()) if v is not None}})
        # What holds for each inner array on entry: the outer boxes, in the inner array's coordinates.
        sub_env = Env()
        for conn, (outer, box, keep) in list(inputs.items()) + list(outputs.items()):
            if conn in sub_env.written:
                continue
            if conn not in inputs:
                sub_env.written[conn] = []
                continue
            have = env.written.get(outer.data, []) + available(outer.data, outer, ch)
            sub_env.written[conn] = [
                tuple((sympy.expand(a - box[d][0]), sympy.expand(b - box[d][0])) for d, (a, b) in enumerate(h)
                      if d in keep)
                for h in have
                if len(h) == len(box) and all(f.le(h[d][0], box[d][0]) and f.le(box[d][1], h[d][1])
                                              for d in range(len(box)) if d not in keep)
            ]
        viewed = _viewed(nst.sdfg, sub_ctx.tracked)
        sub = _Analysis()
        sub._params = self._params
        sub_ctx.tracked -= set(viewed)
        result = sub.region(nst.sdfg, sub_env, f, sub_ctx)
        for conn, why in list(sub.reasons.items()) + [(c, [w]) for c, w in viewed.items()]:
            outer = (inputs.get(conn) or outputs.get(conn))[0]
            for msg in why:
                self.reasons.setdefault(outer.data, []).append(f"{where}: {msg}")
        for conn, (outer, box, keep) in outputs.items():
            if conn in viewed:
                writes.append((outer.data, None, ch, outer, False))
                continue
            for bx in result.written.get(conn, []):
                it = iter(bx)
                full = []
                for d in range(len(box)):
                    if d in keep:
                        a, b = next(it)
                        full.append((sympy.expand(a + box[d][0]), sympy.expand(b + box[d][0])))
                    else:
                        full.append(box[d])
                writes.append((outer.data, tuple(full), ch, outer, True))


def _copied_constants(state: SDFGState, constants: dict) -> dict:
    """Arrays ``state`` overwrites whole with a constant array of the same length (``a = (/1, 2, 3/)``), as their new
    contents; only arrays with no other write in the state."""
    out = {}
    for n in state.data_nodes():
        ins = [e for e in state.in_edges(n) if not e.data.is_empty()]
        if len(ins) != 1 or len(state.sdfg.arrays[n.data].shape) != 1:
            continue
        e = ins[0]
        src = e.src
        if isinstance(src, nodes.LibraryNode) and type(src).__name__ == "CopyLibraryNode":
            feeds = [f for f in state.in_edges(src) if not f.data.is_empty()]
            if len(feeds) != 1 or not isinstance(feeds[0].src, nodes.AccessNode):
                continue
            src, src_subset = feeds[0].src, feeds[0].data.subset
            dst_subset = e.data.subset
        elif isinstance(src, nodes.AccessNode):
            src_subset = e.data.src_subset if e.data.src_subset is not None else e.data.subset
            dst_subset = e.data.dst_subset if e.data.dst_subset is not None else e.data.subset
        else:
            continue
        vals = constants.get(src.data)
        size = state.sdfg.arrays[n.data].shape[0]
        if vals is None or src_subset is None or dst_subset is None or not (size == len(vals)):
            continue
        whole = [(0, len(vals) - 1, 1)]
        if [tuple(int(x) for x in r) for r in src_subset.ndrange()] == whole and \
                [tuple(int(x) for x in r) for r in dst_subset.ndrange()] == whole:
            out[n.data] = vals
    return out


def _viewed(sdfg: dace.SDFG, names: set[str]) -> dict[str, str]:
    """The arrays of ``names`` (of ``sdfg`` itself) a view or reference node is attached to."""
    out = {}
    for st in sdfg.all_states():
        if st.sdfg is not sdfg:
            continue
        for n in st.data_nodes():
            if isinstance(sdfg.arrays.get(n.data), (dace.data.View, dace.data.Reference)):
                for e in st.all_edges(n):
                    other = e.src if e.dst is n else e.dst
                    if isinstance(other, nodes.AccessNode) and other.data in names:
                        out[other.data] = f"accessed through the view or reference {n.data} in state {st.label}"
            elif n.data in names and isinstance(sdfg.arrays.get(n.data), dace.data.Reference):
                out[n.data] = f"is a reference ({st.label})"
    return out


def uncovered_reads(sdfg: dace.SDFG, names: Iterable[str], cases: Iterable[dict] = ()) -> dict[str, list[str]]:
    """For each array of ``names`` (containers of ``sdfg``): where it reads elements not provably written earlier in
    the same execution -- empty when there is no such read.  ``cases``: the combinations of values the SDFG's free
    symbols can take (each ``{name: value}``; see :class:`Facts`)."""
    names = [n for n in names if n in sdfg.arrays]
    viewed = _viewed(sdfg, set(names))
    analysis = _Analysis()
    analysis.region(sdfg, Env(), Facts(cases=tuple(cases)), _Ctx(sdfg, set(names) - set(viewed)))
    out = {n: list(analysis.reasons.get(n, [])) for n in names}
    for n, why in viewed.items():
        out[n] = [why]
    return out
