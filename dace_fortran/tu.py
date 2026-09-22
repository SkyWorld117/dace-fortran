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

from dace_fortran.discovery import Shape

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
