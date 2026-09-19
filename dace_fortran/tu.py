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


def union_range(left: Bounds, right: Bounds) -> Bounds:
    """The union of two arms' ranges, checked for the overlap that makes a union expressible.

    The arms of a stencil pair share their difference axis and differ by one cell at one end
    (`[a, b]` and `[a-1, b-1]`, or the mirror).  A pair whose ranges do NOT overlap cannot be one
    nest with two guards -- and pretending otherwise would put a guard in the middle of a gap.

    THE CHECK IS ONLY POSSIBLE FOR LITERAL BOUNDS.  Real callers pass the caller's own bound names
    (`ulb`/`chi` against `clo`/`ure`), which no amount of arithmetic here can order, so for symbolic
    bounds the mirrored-pair convention is the CALLER's contract and this function cannot verify it.
    Saying so is the point: a check that appears to cover the symbolic case would be worse than no
    check, because a reader would trust it.
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
    # Symbolic bounds: the caller's names cannot be ordered here (see the docstring).
    return (lo_l, hi_r)


def union_arms(axis: str,
               left: Bounds, right: Bounds,
               left_stmts: Sequence[str], right_stmts: Sequence[str],
               *,
               left_is_upper: bool = True,
               indent: str = "  ") -> str:
    """One nest body: the two arms inside guard blocks, over the union of their ranges.

    ``left_is_upper`` says which END of the axis the L arm occupies -- the L arm of a face average
    runs ``[ulb, hi]`` and the R arm ``[lo, ure]``, so the L arm is the UPPER one.  Getting this
    backwards is silent: both guards still compile and both arms still run, they just write the wrong
    cells, so the flag is explicit and the guards are derived from it rather than from the order the
    arms happen to be passed in.
    """
    union_range(left, right)   # validated for overlap, and the reason it can fail
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
           wp: int = 8) -> str:
    """A complete TU: a module holding one kernel subroutine.

    ``dummies`` is ``(name, rank)`` per array argument -- the rank comes from the caller because it is
    a property of the data being ported, and a hand-written emitter that assumed rank 3 declared the
    one 1-D array as 3-D until someone special-cased its NAME.
    """
    decl = "\n".join(f"    real({wp}), intent(inout) :: {n}({', '.join('0:' for _ in range(r))})"
                     for n, r in dummies)
    sig = [n for n, _ in dummies] + list(scalars)
    scal = ", ".join(scalars)
    return f"""module {name}
  implicit none
  integer, parameter :: wp = {wp}
contains
  !> {doc}
  subroutine {kernel}({', '.join(sig)})
    implicit none
{decl}
    integer, intent(in) :: {scal}
    integer :: {', '.join(nest_vars)}

{nest(nest_vars, nest_bounds, body)}
  end subroutine {kernel}
end module {name}
"""
