# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit a translation unit (a Fortran subroutine a frontend lowers into an SDFG) from a shape.

This is the third stage of the TU-generation path: discovery says which shapes exist, extract gets
their bodies out, and this puts a body back together as something a frontend can lower.

THE ONE TRANSFORMATION THAT MATTERS HERE is the union nest, and it is not cosmetic.

A stencil pair written the way the source writes it -- one loop per arm -- lowers to TWO SIBLING
maps, and a map-collapse pass fuses a CHAIN, not siblings.  The surrogate outcome is a device map
covering only the OUTER loop: with the arms' innermost, unit-stride dimension left serial, i.e. the
strided dimension on the thread axis and no coalescing.  Measured on one such kernel: 2.0 ms/launch
against 5.4 us once the arms were written as one nest -- 372x, and the difference between 81.6% of a
run's GPU time and 1.2% of it.

So the arms are laid out inside ONE nest over the UNION of their ranges, each arm behind a guard on
the difference axis:

    do c = lo, hi              ! the union, which is exactly what makes the (c) map fusable
      if (c >= l_hi_bound) ... ! the L arm, guarded by ITS OWN bound
      if (c <= r_lo_bound) ... ! the R arm, guarded by ITS OWN bound
    end do

The guards are derived from the arms' own loop bounds rather than passed in as literals, because that
is the thing a hand-written version gets wrong: the union is contiguous only if the arms overlap, and
the guards are what make a non-contiguous pair impossible to express silently.

What is NOT here: which array becomes which dummy, and the prelude a TU compiles against.  Those are
properties of the code being ported, not of Fortran -- they come in as arguments.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dace_fortran import discovery
from dace_fortran.discovery import Shape
# `from dace_fortran.extract import nest as _extract_nest` and not a plain `nest`: this module has its
# own `nest` (the union-nest emitter), and a caller reading `nest(...)` in `arms_of` would have to
# work out which one it got.
from dace_fortran.extract import nest as _extract_nest

Bounds = Tuple[str, str]


class ArmsNotFormable(ValueError):
    """The two arms cannot become one nest, and saying so beats emitting something plausible.

    Raised rather than guessed: a union nest whose guards do not cover both arms' ranges is a kernel
    that silently computes the wrong cells, which is exactly the failure mode this module exists to
    prevent.
    """


def _base_offset(expr: str) -> Tuple[str, int]:
    """``'is2_viscous%end - 1'`` -> ``('is2_viscous%end', -1)``.

    Splits a bound into the expression a rebase leaves alone and the integer offset a stencil moves.
    That split is what makes the symbolic case checkable at all: the two arms of one stencil pair do
    not have ORDERED bounds here, but they do have the SAME bases and offsets one apart.
    """
    m = re.match(r'^(.*?)\s*([+-])\s*(\d+)\s*$', expr.strip())
    if m:
        return m.group(1).strip(), int(m.group(3)) * (1 if m.group(2) == '+' else -1)
    return expr.strip(), 0


def union_range(left: Bounds, right: Bounds,
                aliases: Optional[Dict[str, str]] = None) -> Bounds:
    """The union of two arms' ranges, CHECKED -- for symbolic bounds as well as literal ones.

    The arms of a stencil pair share their difference axis and differ by one cell at each end
    (`[a+1, b]` and `[a, b-1]`, or the mirror).  A pair whose ranges do not meet cannot be one nest
    with two guards: pretending otherwise puts a guard in the middle of a gap, or -- worse, because it
    is silent -- pairs the wrong two arms and writes the wrong cells.

    THE SYMBOLIC CASE IS CHECKED TOO, and an earlier version of this function said it could not be.
    It is true that `ulb` cannot be ordered against `clo` -- the SDFG records no relation between the
    caller's bound names.  But ordering is not what is needed.  Two arms of one stencil differ by a
    REBASE: their bounds have the same base expressions and integer offsets exactly one apart, in the
    same direction.  That is a structural property of the text, so it is verified rather than trusted.

    What this catches, concretely: a caller that pairs a shape's arms by grouping key alone, when the
    key is not injective, hands over the dx-down arm and the dy-up arm.  Those bounds share no bases,
    so the call now raises instead of emitting a nest whose guards cover the wrong axis.
    """
    lo_l, hi_l, lo_r, hi_r = left[0], left[1], right[0], right[1]

    def _int(s: str) -> Optional[int]:
        try:
            return int(s)
        except ValueError:
            return None

    il, ir, jl, jr = _int(lo_l), _int(lo_r), _int(hi_l), _int(hi_r)
    if None not in (il, ir, jl, jr):
        # Adjacent counts as contiguous: [0, 4] and [5, 9] tile with no gap, so the union of the two
        # guards covers every cell exactly once.
        if jl + 1 < ir or jr + 1 < il:
            raise ArmsNotFormable(
                f"the arms {left} and {right} leave a gap; one nest with two guards cannot express "
                f"them, and emitting it anyway would silently skip the cells between")
        return (lo_l if il <= ir else lo_r, hi_l if il <= ir else hi_r)

    (bl_lo, of_lo), (bl_hi, of_hi) = _base_offset(lo_l), _base_offset(hi_l)
    (br_lo, or_lo), (br_hi, or_hi) = _base_offset(lo_r), _base_offset(hi_r)
    bl_lo, bl_hi = _canonical(bl_lo, aliases), _canonical(bl_hi, aliases)
    br_lo, br_hi = _canonical(br_lo, aliases), _canonical(br_hi, aliases)
    if (bl_lo, bl_hi) != (br_lo, br_hi):
        raise ArmsNotFormable(
            f"the arms {left} and {right} do not share their bounds' bases "
            f"({bl_lo!r}/{bl_hi!r} against {br_lo!r}/{br_hi!r}), so they are not the two arms of one "
            f"stencil -- they are two different pairs, and one nest over their union would guard the "
            f"wrong axis")
    d_lo, d_hi = of_lo - or_lo, of_hi - or_hi
    if abs(d_lo) != 1 or abs(d_hi) != 1 or d_lo != d_hi:
        raise ArmsNotFormable(
            f"the arms {left} and {right} are not a one-cell mirror: their bounds differ by "
            f"({d_lo}, {d_hi}) where a stencil pair differs by (+-1, +-1) in the SAME direction.  "
            f"One nest over the union of the two guards would not cover both ranges exactly once")
    # d > 0 means the LEFT arm's bounds sit one cell HIGHER: it is the upper arm, so the union takes
    # its lower bound from the right arm and its upper bound from the left.
    return (lo_r, hi_l) if d_lo > 0 else (lo_l, hi_r)


def alias_map(assignments: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    """``[('is2_viscous', 'iy')]`` -> ``{'is2_viscous': 'iy', 'iy': 'iy'}``, transitively.

    WHY A CALLER-SUPPLIED RELATION IS NEEDED AND IS NOT A HOLE.  `is1_viscous = ix; is2_viscous = iy`
    sits at the top of the routine, so two arms whose transverse bounds read `is2_viscous%beg` and
    `iy%beg` run over exactly the same cells -- and no reading of the bounds alone can know that.
    Refusing is the safe direction (and this module refuses by default), but it is a FALSE NEGATIVE,
    and a caller that has the assignment in front of it should be able to say so.

    The relation is still checked rather than trusted: it is consulted only for the two comparisons
    below, so an alias map that is wrong still has to reconcile the bounds it claims are equal.
    """
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x] = parent.get(parent[x], parent[x])
        return x

    for a, b in assignments:
        if a != b:
            parent[find(a)] = find(b)
    names = {n for pair in assignments for n in pair}
    return {n: find(n) for n in names}


def _canonical(expr: str, canon: Optional[Dict[str, str]]) -> str:
    """Rewrite the leading name of every ``x%comp`` through ``canon``."""
    if not canon:
        return expr
    return re.sub(r'\b([A-Za-z_]\w*)(?=%)',
                  lambda m: canon.get(m.group(1), m.group(1)), expr)


def check_nest_pair(axis_pos: int,
                    left: Sequence[Bounds], right: Sequence[Bounds],
                    aliases: Optional[Dict[str, str]] = None) -> Bounds:
    """Validate that two FULL nests are the two arms of ONE stencil; return the difference axis' union.

    WHY THIS EXISTS, and why `union_range` alone is not enough.  A *shape* key groups loops by their
    arithmetic form, and that grouping is deliberately coarser than the families in the source: the
    x-gradient and the z-gradient averaged over the y-face have the SAME form (one kernel serves
    both, which is the whole point of a shape), so one key legitimately holds FOUR nests -- two
    pairs.  Pairing by key alone then has to pick one L and one R out of that group, and if it picks
    across the two pairs it emits a nest over the wrong arrays.

    The difference axis cannot catch that, and this is the part that is easy to get wrong when
    reasoning about it: for the dx pair and the dz pair, the difference-axis bounds ARE a perfect
    one-cell mirror of each other.  What differs is the TRANSVERSE axes -- the dx arm tightens `k`
    where the dz arm tightens `l`.  So the check is:

      * every axis EXCEPT the difference one has IDENTICAL bounds in both arms, and
      * the difference axis is a one-cell mirror (`union_range`).

    Both halves are needed: without the first, a cross-pair is invisible; without the second, an
    unrelated pair is accepted.
    """
    if len(left) != len(right):
        raise ArmsNotFormable(
            f"the arms expose {len(left)} and {len(right)} loop bounds; two arms of one stencil are "
            f"the same nest depth")
    for i, (lb, rb) in enumerate(zip(left, right)):
        if i == axis_pos:
            continue
        clb = tuple(_canonical(b, aliases) for b in lb)
        crb = tuple(_canonical(b, aliases) for b in rb)
        if clb != crb:
            raise ArmsNotFormable(
                f"the arms are not the same nest: axis {i} runs {lb} in one and {rb} in the other "
                f"({clb} against {crb} after aliasing).  They are two different stencils whose "
                f"difference axis happens to mirror -- pairing them would emit a nest whose guards "
                f"cover the wrong arrays")
    return union_range(left[axis_pos], right[axis_pos],
                       aliases=aliases)


def union_arms(axis: str,
               left: Bounds, right: Bounds,
               left_stmts: Sequence[str], right_stmts: Sequence[str],
               *,
               left_is_upper: bool = True,
               indent: str = "  ",
               aliases: Optional[Dict[str, str]] = None) -> str:
    """One nest body: the two arms inside guard blocks, over the union of their ranges.

    ``left_is_upper`` says which END of the axis the L arm occupies -- the L arm of a face average
    runs ``[ulb, hi]`` and the R arm ``[lo, ure]``, so the L arm is the UPPER one.  Getting this
    backwards is silent: both guards still compile and both arms still run, they just write the wrong
    cells, so the flag is explicit and the guards are derived from it rather than from the order the
    arms happen to be passed in.
    """
    union_range(left, right, aliases=aliases)   # validated for overlap, and the reason it can fail
    # The guard on the L arm is its own lower bound; on the R arm, its own upper bound.
    if left_is_upper:
        l_guard, r_guard = f"{axis} >= {left[0]}", f"{axis} <= {right[1]}"
    else:
        l_guard, r_guard = f"{axis} <= {left[1]}", f"{axis} >= {right[0]}"

    out: List[str] = []
    for guard, stmts in ((l_guard, left_stmts), (r_guard, right_stmts)):
        out.append(f"{indent}if ({guard}) then")
        out.extend(f"{indent}  {s}" for s in stmts)
        out.append(f"{indent}end if")
    return "\n".join(out)


def nest(vars_: Sequence[str], bounds: Sequence[Bounds], body: str) -> str:
    """A ``do``-nest over ``vars_`` (outermost first) with ``body`` indented inside it."""
    out: List[str] = []
    for i, v in enumerate(vars_):
        out.append(f"{'  ' * (i + 1)}do {v} = {bounds[i][0]}, {bounds[i][1]}")
    out.append(_indent(body, 2 * len(vars_)))
    for i in reversed(range(len(vars_))):
        out.append(f"{'  ' * (i + 1)}end do")
    return "\n".join(out)


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + l if l.strip() else l for l in text.split("\n"))


def module(name: str,
           kernel: str,
           *,
           dummies: Sequence[Tuple[str, int]],
           scalars: Sequence[str],
           body: str,
           nest_vars: Sequence[str],
           nest_bounds: Sequence[Bounds],
           doc: str = "",
           wp: int = 8,
           locals: Sequence[str] = ()) -> str:
    """... ``locals`` declares INTERMEDIATE scalars the body uses.

    The source hoists reads into temporaries -- MFC's flux difference is
    ``flux_face1 = f(..j-1..); flux_face2 = f(..j..); rh = inv_ds*(flux_face1 -
    flux_face2)`` -- and while the declarations above cover the dummies, the
    scalars and the loop variables, nothing declared THOSE.  MEASURED: the emitted
    unit referenced `inv_ds`, `flux_face1` and `flux_face2` with no declaration,
    i.e. it was not merely a different form from the committed unit, it was
    INVALID FORTRAN.  A generated unit that cannot compile is worse than one that
    compiles and is wrong, because no gate here would have said so.
    """
    """A complete TU: a module holding one kernel subroutine.

    ``dummies`` is ``(name, rank)`` or ``(name, rank, intent)`` per array argument -- the rank comes
    from the caller because it is a property of the data being ported, and a hand-written emitter that
    assumed rank 3 declared the one 1-D array as 3-D until someone special-cased its NAME.

    The optional third element is the INTENT, defaulting to ``inout`` so an existing 2-tuple caller is
    unaffected.  It is worth carrying rather than forcing ``inout`` everywhere: an input declared
    ``inout`` is not a compile error and dace infers read/write from the memlets, so nothing catches
    it -- but the Fortran-level contract then stops matching the kernel's, which is the drift this
    file exists to keep out.
    """
    norm = [(d[0], d[1], d[2] if len(d) > 2 else "inout") for d in dummies]
    decl = "\n".join(
        f"    real({wp}), intent({i}) :: {n}({', '.join('0:' for _ in range(r))})"
        for n, r, i in norm)
    sig = [n for n, _, _ in norm] + list(scalars)
    scal = ", ".join(scalars)
    # WRAP THE SIGNATURE, and it is not cosmetic: gfortran's free-form limit is 132 columns and
    # nvfortran's is longer, so a unit that bakes fine can fail to COMPILE in the gate --
    # MEASURED, verbatim: `Error: Line truncated at (1) [-Werror=line-truncation]` on a signature the
    # bake had accepted.  Two more scalars were enough to cross it.  Free-form continuations (`&` at
    # the end of the line being left, and at the start of the next) keep it inside the limit wherever
    # the unit is compiled.
    _sig = []
    _cur = f"  subroutine {kernel}("
    for _i, _n in enumerate(sig):
        _piece = _n if _i == len(sig) - 1 else _n + ","
        if len(_cur) + 1 + len(_piece) + (1 if _i == len(sig) - 1 else 0) <= 100:
            _cur = _cur + " " + _piece
        else:
            _sig.append(_cur + " &")
            _cur = "      & " + _piece
    _sig.append(_cur + ")")
    _sigline = "\n".join(_sig)
    locals_ = f"    real({wp}) :: {', '.join(locals)}" if locals else ""
    return f"""module {name}
  implicit none
  integer, parameter :: wp = {wp}
contains
  !> {doc}
{_sigline}
    implicit none
{decl}
    integer, intent(in) :: {scal}
    integer :: {', '.join(nest_vars)}
{locals_}

{nest(nest_vars, nest_bounds, body)}
  end subroutine {kernel}
end module {name}
"""


# ---------------------------------------------------------------------------------------------------
# Analysing a shape's nests: which nest loop carries the difference, what its bounds are, whether two
# emitted units are the same arithmetic, and which source names are aliases of each other.
#
# These are properties of FORTRAN and of `Shape`, not of any one code being ported, which is why they
# live here.  What is deliberately absent is anything that knows what an ACCESS MEANS: the helpers
# below that read `%sf(...)` take the access idiom as an argument instead of assuming it, and the
# substitution that turns an access into a DUMMY stays with the caller -- see the module docstring.
# ---------------------------------------------------------------------------------------------------


def axis_index(shape: Shape) -> int:
    """The NEST position (0 = outermost) whose loop indexes storage position ``shape.dim``.

    The house convention, which :mod:`dace_fortran.discovery` documents where it builds ``loops``, is
    outermost = LAST storage dimension, innermost = first, so position ``dim`` in a subscript tuple is
    ``loops[-1 - dim]``.

    THIS IS THE BUG THIS FUNCTION EXISTS TO NOT HAVE.  An emitter that used ``shape.loops[-1]`` and a
    hardcoded axis -- the INNERMOST loop -- guards the difference on the wrong loop for four of six
    shapes, and nothing raises: the bounds are read off the wrong axis, so the union is taken over the
    wrong axis and the emitted nest writes the wrong cells.  It reported "6 shape(s) emitted, 0
    blocked" the whole time, which is why a success counter is not evidence.

    A SHAPE WITH NO LOOPS RAISES, and it used to return ``-1``.  ``-1`` is a VALID Python index, so a
    caller doing ``bounds[axis_index(shape)]`` silently read the LAST bound instead of failing -- the
    same silent-wrong-answer shape as the hardcoded innermost loop above, one step further out.  There
    is no axis to name, so it raises.
    """
    if not shape.loops:
        raise ValueError(f"shape {shape.op}:{shape.dim} has no loops, so it has no difference axis")
    d = shape.dim if shape.dim is not None else 0
    return len(shape.loops) - 1 - d


def axis_letter(shape: Shape) -> Optional[str]:
    """The loop variable of the difference axis -- :func:`axis_index`'s letter, or ``None``.

    ``None`` is a real answer and not an error: ``Shape.loops`` may be shorter than the nest, so a
    shape whose difference rides a loop the canonicaliser dropped has no letter here.  Callers must
    handle it rather than index blindly -- an emitter that assumed a letter classified every arm of
    such a shape as neither up nor down and refused the whole group.
    """
    return shape.loops[axis_index(shape)] if shape.loops else None


def nest_bounds(body: str) -> List[Tuple[str, str, str]]:
    """``(var, lo, hi)`` per ``do`` header in ``body``, outermost first -- the union's inputs.

    ONE LETTER CLASS IS A BUG, and it was found four separate times in this path.  The loop variable
    is ``[a-z_]\\w*`` and not ``[a-z]``: a nest that spells its loops ``k_loop``/``l_loop`` yields NO
    loops under the narrow class, and the caller then emits a unit with no ``do`` statements at all
    whose body references ``j``/``k``/``l`` undeclared.

    WIDENING IT IS NOT FREE, and that is the other half.  This decides how a shape is RENDERED, and it
    is shared by every family already validated: MEASURED, widening it changed two pairs' emitted text
    so they no longer agreed and the same-shape guard refused the emission -- for a family verified
    bit-identical an hour earlier.  A change in shared rendering code has to arrive with the family it
    is for.
    """
    out: List[Tuple[str, str, str]] = []
    for ln in body.split("\n"):
        m = re.match(r"\s*do\s+([a-z_]\w*)\s*=\s*(.+?)\s*,\s*(.+?)\s*$", ln)
        if m:
            out.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    return out


def shape_signature(unit_src: str) -> str:
    """The unit's ARITHMETIC identity: everything a caller does NOT supply.

    One library per SHAPE serves every loop of that shape, each with its own ranges -- that is how one
    ``AVG dim0`` unit serves both stencils of that shape.  So the loop RANGES are arguments and are
    not part of the kernel, and comparing raw text would call a shape a mismatch for being used the
    way it is designed to be used.

    THREE THINGS HAVE TO BE NORMALISED AWAY and the third is not obvious: the doc line carries a
    source LINE NUMBER; the unit's own NAME differs between a per-pair and a per-shape naming; and the
    SIGNATURE's wrap point moves with that name's LENGTH, because :func:`module` breaks the argument
    list at 100 columns.  MEASURED: without the last one, two identical units reported a mismatch at
    ``& b_lo, b_hi, c_lo, ...`` against ``& b_hi, c_lo, c_hi, ...`` -- a pure formatting artefact of
    the rename being checked for.
    """
    src = "\n".join(l for l in unit_src.split("\n") if not l.strip().startswith("!>"))
    src = re.sub(r"&\s*\n\s*&?", " ", src)            # fold free-form continuations before comparing
    src = re.sub(r"\b\w+_mod_kernel\b", "KERNEL", src)
    src = re.sub(r"\b\w+_mod\b", "UNIT", src)
    keep: List[str] = []
    for ln in src.split("\n"):
        s = " ".join(ln.split())
        m = re.match(r"do\s+(\w+)\s*=", s)
        keep.append(f"do {m.group(1)}" if m else s)
    return "\n".join(keep)


def aliases_of(scoped: str) -> Dict[str, str]:
    """Simple ``a = b`` equalities in a routine, as the map :func:`check_nest_pair` takes.

    WHY THIS IS EXTRACTED RATHER THAN ASSUMED.  ``is1_viscous = ix; is2_viscous = iy`` at the top of a
    routine means two arms whose transverse bounds read ``is2_viscous%beg`` and ``iy%beg`` run over
    exactly the same cells -- and no reading of the bounds alone can know that.  Without the map such a
    pair is refused as "not the same nest", which is safe but wrong.

    ``;``-SEPARATED, and that is not a detail: sources write all three assignments on ONE line, so a
    line-anchored ``^\\s*(\\w+)\\s*=\\s*(\\w+)\\s*$`` matches none of them and the map comes out empty
    for a reason that is no longer true.  Only assignments of one bare name to another count -- an
    alias that is not a pure rename is not an alias.
    """
    pairs = []
    for ln in scoped.split("\n"):
        for part in ln.split(";"):
            m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*$", part)
            if m and m.group(1) != m.group(2):
                pairs.append((m.group(1), m.group(2)))
    return alias_map(pairs)


#: The component an array's storage is read through -- `%sf` in the code these were written against.
#: IT IS A PARAMETER AND NOT A CONSTANT because it is a property of the CODE BEING PORTED: a project
#: whose derived types spell the field differently gets the same machinery, and the one thing this
#: module must not do is assume one project's spelling.
ACCESS_DEFAULT = "%sf"


class NotAnAccess(ValueError):
    """The text is not an assignment through the access idiom, or its subscripts do not balance."""


def subscripts(stmt: str, pos: int, access: str = ACCESS_DEFAULT) -> Tuple[str, str]:
    """``(dst, src)`` -- the ``pos``-th subscript (1-based) of each side of a copy statement.

    Paren-aware splitting, because the expressions carry parentheses of their own (`m - (j - 1)`) and
    a naive `split(',')` would cut one in half -- and the cut expression would parse as a DIFFERENT
    affine function rather than failing.
    """
    if "=" not in stmt:
        raise NotAnAccess(f"not an assignment: {stmt!r}")
    open_at = access + "("
    lhs, rhs = stmt.split("=", 1)

    def subs_of(side: str) -> str:
        i = side.find(open_at)
        if i < 0:
            raise NotAnAccess(f"no `{open_at}` access in {side!r}")
        depth, start = 1, i + len(open_at)
        for j in range(start, len(side)):
            if side[j] == "(":
                depth += 1
            elif side[j] == ")":
                depth -= 1
                if depth == 0:
                    body = side[start:j]
                    break
        else:
            raise NotAnAccess(f"unbalanced parentheses in {side!r}")
        parts, depth, cur = [], 0, ""
        for ch in body:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        parts.append(cur.strip())
        if len(parts) < pos:
            raise NotAnAccess(f"access has {len(parts)} subscript(s), wanted {pos}: {side!r}")
        return parts[pos - 1]

    return subs_of(lhs), subs_of(rhs)


# Names that look like a subscripted access and are not data.  Small on purpose: an emitter that
# guessed would silently declare a function as an array, which fails at link time in a way that is
# harder to read than a compile error.
_NOT_DATA = {
    "real", "mod", "min", "max", "abs", "sqrt", "exp", "log", "sin", "cos", "tan", "int", "nint",
    "if", "do", "end", "then", "write", "read", "open", "close", "size", "sum", "allocated",
    "present", "trim", "reshape", "matmul", "sign", "dim", "epsilon", "huge", "tiny", "null",
}


def dummies(stmts: Sequence[str], access: str = ACCESS_DEFAULT):
    """``([(dummy, rank, base)], {base: dummy})`` per distinct array ACCESS in ``stmts``.

    Deliberately the weakest thing that can work, and it is worth saying what it is NOT: it does not
    know that one gradient array is the level-2 x-gradient of the LEFT state, nor that an index is
    momentum row 0.  Those names are what makes an emitted unit CALLABLE, and they are a binding
    record -- a separate concern, and `dace_fortran.tu_bindings` is the module for it.

    What this does establish is the property a compile-and-compare gate is about: the emitted nest is
    FORTRAN THAT COMPILES, built from the source's own statements with no arithmetic composed
    anywhere.  A substitution that guessed at meaning would be worse than none.
    """
    seen: Dict[str, str] = {}
    out: List[Tuple[str, int, str]] = []

    def add(base: str, rank: int) -> None:
        if base not in seen:
            seen[base] = f"a{len(seen) + 1}"
            out.append((seen[base], rank, base))

    comp = re.compile(r"([A-Za-z_]\w*(?:\([^)]*\))?(?:%\w+(?:\([^)]*\))?)*)"
                      + re.escape(access) + r"\s*\(([^)]*)\)")
    for st in stmts:
        for m in comp.finditer(st):
            add(m.group(1), len(m.group(2).split(",")))
    # ... AND THE PLAIN ARRAYS.  The component idiom is not the only way Fortran reads memory: a
    # shape may divide by a coordinate spacing that is a MODULE ARRAY, which the pattern above never
    # sees and which therefore reaches the compiler undeclared -- "Function 'y_cc' at (1) has no
    # IMPLICIT type", i.e. a unit that does not compile, which is exactly what this is for.
    # Intrinsics and keywords are excluded by name; anything else with a subscript is data.
    for st in stmts:
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", st):
            name = m.group(1)
            if name.lower() in _NOT_DATA:
                continue
            # ... but NOT the inner part of a `%`-chain: the component pass above already owns it,
            # and adding it gives the SAME array two dummies of different ranks, so the two units then
            # disagree about which is which (`Rank mismatch in argument 'a5'`).  A `%` before the name
            # or after its closing paren means the component pass owns it.
            if st[m.start() - 1:m.start()] == "%":
                continue
            depth, j = 0, m.end() - 1
            while j < len(st):
                if st[j] == "(":
                    depth += 1
                elif st[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if st[j + 1:j + 2] == "%":
                continue
            add(name, st[m.end():j].count(",") + 1)
    return out, seen


def arms_of(text: str, lines: Sequence[int], shape: Shape,
            access: str = ACCESS_DEFAULT) -> List[dict]:
    """Split a shape's nests into arms by the direction their difference points.

    The arms of a stencil pair differ in ONE way: which way the stencil reads.  The L arm writes where
    its reads reach DOWN (`c - 1`), the R arm where they reach UP (`c + 1`).  Reading that off the body
    is the only sound way to pair them -- pairing by adjacency would mis-pair a family whose nests are
    not stored L,R,L,R.

    AXIS-FREE, AND THAT IS THE FIX RATHER THAN A SIMPLIFICATION.  This used to derive the difference
    axis from the shape and look ONLY at that subscript position -- and `Shape.loops` can be SHORTER
    than the nest, so a shape whose swept axis is the fourth loop has no letter, both flags stay
    False, and every arm of it is classified `both/neither`: the group is refused.
    MEASURED: `FLUX dim0: 2 nest(s) -> 0 down-arm, 0 up-arm` while its neighbouring nest of the same
    shape and direction classified `1 down`.  A flux difference reads `f(k-1)` and `f(k)` -- one DOWN
    and no UP -- so `0/0` could only mean the lookup found nothing.

    So it asks the question directly: for each access, is the variable being shifted a LOOP OF THIS
    NEST?  No shape mapping, no subscript position, and it reads `k_loop` and `k` alike.  ANCHORED to
    the loop variables rather than a bare `[a-z]\\s*-\\s*1`, which matches identifiers and unrelated
    expressions -- it found `d - 1` and `g + 1` inside array NAMES and reported every arm as "both".
    """
    out = []
    for ln in lines:
        body = _extract_nest(text, ln)
        stmts = discovery.statements(body.split("\n"))
        shift = discovery._difference_position(" ".join(stmts))
        # WIDE on purpose, and NOT via `nest_bounds`: that one uses a single-letter class because it
        # decides how a shape is RENDERED and it is shared by every family already validated.  This
        # use is a CLASSIFICATION, so it can be as permissive as the source needs.
        loopvars = {m.group(1) for ln in body.split("\n")
                    for m in [re.match(r"\s*do\s+([a-z_]\w*)\s*=", ln)] if m}
        minus = plus = False
        for m in re.finditer(re.escape(access) + r"\s*\(([^)]*)\)", " ".join(stmts)):
            for sub in (x.strip() for x in m.group(1).split(",")):
                for v in loopvars:
                    if re.search(rf"\b{re.escape(v)}\s*-\s*1\b", sub):
                        minus = True
                    if re.search(rf"\b{re.escape(v)}\s*\+\s*1\b", sub):
                        plus = True
        out.append({"line": ln, "body": body, "stmts": stmts, "dim": shift,
                    "reaches": "down" if minus and not plus else ("up" if plus and not minus
                                                                   else "both/neither")})
    return out
