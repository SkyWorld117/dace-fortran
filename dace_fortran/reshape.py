# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a nest into the OFFSET-COPY shape, deriving its constants from the source.

WHY THIS IS A SEPARATE MODE AND NOT A PARAMETER OF THE UNION NEST.  A stencil pair's two arms differ
on ONE axis and share the others, so `tu.union_arms` can put them in one nest behind guards on the
difference axis.  A *boundary copy* is not that: its two "arms" are two FACES of the same array, they
write DISJOINT planes, and their subscripts are related by nothing on any axis -- `-j` against `m + j`
is not a mirror, it is a different plane.  What they do share is a SHAPE:

    dst = doff + gg,      src = soff + gg          (gg = 0 .. bb-1, ascending)

with `doff`/`soff` supplied by the caller.  So one kernel serves both faces, and the face selection
moves out of the kernel entirely -- which is not a convenience: a face branch INSIDE the nest gives one
array two write subsets (`q(gg)` and `q(mw1+gg)`), `_writes_may_overlap` cannot prove them disjoint for
a symbolic offset, and the map is refused.  MEASURED, and it is why the shipped `s_periodic` family
looks the way it does.

WHY THE OFFSETS ARE DERIVED RATHER THAN DECLARED.  They are fixed host-side today
(`m_dace_kernels_periodic.fpp`: `loc == -1` -> `doff = 0, soff = nsweep + 1`, else
`doff = nsweep + bb + 1, soff = bb`) and nothing links those integers to the Fortran they implement.
Deriving them from the source's own subscripts turns that into something checkable: the derivation is
compared against what the call sites pass, and a source edit that moves a plane stops the build
instead of copying the wrong cells.

THE DERIVATION, and it is three lines:

    the swept subscript is affine in the loop variable, `a*j + b` plus `n` copies of the extent, with
    |a| = 1, and dst and src must share `a` -- that is what makes one ascending `gg` serve both;

    c    = bb if a == -1 else -1        normalises {a*j + c : j = 1..bb} onto {0..bb-1}
    doff = b_dst + bb - c   (+ n*EXT)   the +bb is the OFFSET FRAME: the dummies' base element is the
    soff = b_src + bb - c   (+ n*EXT)   raw (-bb,-bb,-bb) corner, so every subscript carries it

Verified against all six faces of MFC's `s_periodic`: it reproduces `(0, EXT + 1)` for `%beg` and
`(EXT + bb + 1, bb)` for `%end`, exactly what the shipped shim passes.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple


class NotAnOffsetCopy(ValueError):
    """The nest is not a pair of offset-copies, and saying so beats emitting something plausible."""


def parse_affine(expr: str, ext: str, jvar: str = "j") -> Tuple[int, int, int]:
    """``'-j'`` -> ``(-1, 0, 0)``; ``'m - (j - 1)'`` -> ``(-1, 1, 1)``.

    Returns ``(a, b, n_ext)`` for ``a*jvar + b + n_ext*ext``.

    THE SIGN IS DISTRIBUTED, NOT FAKED.  The first version of this replaced ``(j-1)`` textually and so
    turned ``m - (j - 1)`` into ``m - j - 1`` -- off by two in ``b``, and invisible in ``a``.  That
    probe then reported three mismatches against an implementation that was correct.  The arithmetic is
    done here, once, with the distribution spelled out.
    """
    e = expr.replace(" ", "")
    e = re.sub(r"-\s*\(%s-1\)" % jvar, f"-{jvar}+1", e)
    e = re.sub(r"\+\s*\(%s-1\)" % jvar, f"+{jvar}-1", e)
    e = re.sub(r"^\(%s-1\)" % jvar, f"{jvar}-1", e)
    if "(" in e or ")" in e:
        raise NotAnOffsetCopy(f"cannot parse {expr!r}: an unexpected parenthesis survived")
    n_ext = e.count(ext)
    e = e.replace(ext, "")
    a = 0
    if jvar in e:
        # An explicit coefficient is parsed rather than rejected, so the SLOPE is what the caller gets
        # to complain about: `2*j` should say "not +-1", not "cannot parse".  A parser that fails one
        # layer early reports a syntax error for what is a semantic one.
        m = re.search(rf"(\d*)\*?([+-]?){jvar}", e)
        coeff = int(m.group(1)) if m.group(1) else 1
        a = -coeff if m.group(2) == "-" else coeff
        e = e.replace(m.group(0), "", 1)
    if jvar in e:
        raise NotAnOffsetCopy(f"cannot parse {expr!r}: {jvar} appears more than once")
    e = e.strip("+") or "0"
    if e in ("", "-"):
        return a, 0, n_ext
    # A MALFORMED EXPRESSION MUST RAISE THIS MODULE'S OWN ERROR.  `2*j` used to reach `int("2*")` and
    # escape as a bare `ValueError` from the standard library, which a caller catching
    # `NotAnOffsetCopy` -- i.e. every caller -- would not catch.  MEASURED: that is how the test for a
    # non-unit slope failed, and the failure was in the error path rather than in the check.
    if not re.fullmatch(r"[+-]?\d+", e):
        raise NotAnOffsetCopy(f"cannot parse {expr!r}: {e!r} is not an integer term")
    return a, int(e), n_ext


def _sum(terms: Sequence[str]) -> str:
    """Fold the integer terms of a sum, keep the symbols in order: `['m','6','6'] -> 'm + 12'`."""
    total, syms = 0, []
    for t in terms:
        t = t.strip()
        if not t:
            continue
        if re.fullmatch(r"-?\d+", t):
            total += int(t)
        else:
            syms.append(t)
    if total:
        syms.append(str(total))
    return " + ".join(syms) if syms else "0"


def derive_offsets(dst_expr: str,
                   src_expr: str,
                   ext: str,
                   *,
                   ghost: str = "bb",
                   jvar: str = "j") -> Tuple[str, str]:
    """The ``(doff, soff)`` the CALLER must pass, as source TEXT, from the two swept subscripts.

    ``ext`` is the swept extent's symbol (``m``/``n``/``p`` in MFC) and ``ghost`` the ghost width's,
    both as text, because the answer is symbolic: ``doff = EXT + bb + 1`` is a formula the call site
    evaluates, not a number this function can know.

    `ghost` is kept SYMBOLIC rather than folded into a constant so that it cancels where it cancels:
    for a slope of -1 the frame's `+bb` IS the `c` that normalises the iterate, so the ghost width
    never reaches the caller's expression -- which is why `%beg` is `(0, EXT + 1)` and not something
    with a `bb` in it.  Substituting 6 here instead would hide that.
    """
    ad, bd, nd = parse_affine(dst_expr, ext, jvar)
    as_, bs, ns = parse_affine(src_expr, ext, jvar)
    if ad != as_:
        raise NotAnOffsetCopy(
            f"the swept subscripts {dst_expr!r} and {src_expr!r} have different slopes ({ad} and "
            f"{as_}); one ascending iterate cannot serve both, so this is not an offset copy")
    if abs(ad) != 1:
        raise NotAnOffsetCopy(f"the swept slope is {ad}, not +-1")
    # c normalises {a*j + c : j = 1..bb} onto {0..bb-1}.  Both faces iterate j = 1..buff_size, so for a
    # slope of -1 the iterate runs BACKWARDS -- which is fine: the kernel only needs the set.
    #   a = -1:  {c-1, ..., c-bb} = {0..bb-1}  <=>  c = bb   (the GHOST WIDTH ITSELF, not 1)
    #   a = +1:  {c+1, ..., c+bb} = {0..bb-1}  <=>  c = -1
    # MEASURED, after getting it wrong: writing `1` for the slope--1 case yields `bb + -1` against the
    # shim's `0`, because the whole point of that branch is that the frame's `+bb` cancels exactly.
    c = ghost if ad == -1 else -1

    def frame(b: int, n_ext: int) -> str:
        terms = ["" if not n_ext else (ext if n_ext == 1 else f"{n_ext}*{ext}"), str(b)]
        # `ghost - c`, folded here rather than as two terms: for a -1 slope they are the SAME symbol
        # and a term list would emit `bb + -bb`, which is arithmetically right and unreadable.
        if c != ghost:
            terms += [ghost, str(-c)]
        return _sum(terms)

    return frame(bd, nd), frame(bs, ns)


def nest_vars_and_bounds(transverse: Sequence[Tuple[str, str, str]],
                         ghost: str,
                         iter_var: str = "gg") -> Tuple[List[str], List[Tuple[str, str]]]:
    """``[('kb','ke','kk'), ('lb','le','ll')]`` -> the nest `(ll, kk, gg)` and its bounds.

    ``transverse`` is ``(begin, end, iter)`` per axis IN STORAGE ORDER, and the three names are taken
    rather than derived from each other: inferring `ke` from `kb` by chopping the last letter is a
    convention that holds until someone names a scalar differently, and it would hold silently.

    THE ORDER IS NOT COSMETIC.  `gg` must be INNERMOST: it is the unit-stride storage subscript, and
    the device map's innermost loop is what lands on the thread axis.  MEASURED for this family -- with
    `gg` innermost the map is 3-D and coalesced; with it serial every warp strides across the row.

    The transverse loops run over ``end - begin`` because the caller passes both ends of each range
    ALREADY SHIFTED into the offset frame, so a 0-based iterate over the difference covers exactly the
    original cells.
    """
    if len(transverse) != 2:
        raise NotAnOffsetCopy(f"two transverse axes are expected, got {len(transverse)}")
    (low_b, low_e, low_i), (high_b, high_e, high_i) = transverse
    return ([high_i, low_i, iter_var],
            [("0", f"{high_e} - {high_b}"), ("0", f"{low_e} - {low_b}"), ("0", f"{ghost} - 1")])


def render_offset_copy(*,
                       module: str,
                       kernel: str,
                       swept: int,
                       transverse: Sequence[Tuple[str, str, str]],
                       nrows: int,
                       scalars: Sequence[str],
                       ghost: str = "bb",
                       iter_var: str = "gg",
                       doc: str = "",
                       wp: int = 8) -> str:
    """One kernel per direction: ``q{r}d(doff + gg, ...) = q{r}s(soff + gg, ...)``.

    ``swept`` is the 1-BASED STORAGE subscript the swept axis sits on (x=1, y=2, z=3).  The two
    transverse subscripts are placed by ITERATING STORAGE POSITION and consuming the transverse list in
    order, rather than by building a permutation: the y and z cases differ only in where the swept axis
    falls, and a hand-built permutation is where that gets crossed.

    The destination and the source are SEPARATE dummies although they are one array at the call.  That
    is not decoration: with one dummy the innermost loop cannot become a Map (`q(i + mm1)` and `q(i)`
    are not provably disjoint for a symbolic offset, and a Map carries no cross-iteration ordering), so
    `gg` -- the unit-stride subscript -- stays serial per thread.  MEASURED: two dummies take the
    device map from 2 dims to 3 with `gg` innermost; with one, every warp strides across the row.
    """
    from dace_fortran import tu

    if not (1 <= swept <= 3):
        raise NotAnOffsetCopy(f"swept subscript {swept} is not a storage position (1..3)")
    if nrows < 1:
        raise NotAnOffsetCopy("a positive row count is required")
    nest_vars, nest_bounds = nest_vars_and_bounds(transverse, ghost, iter_var)

    def subs(plane: str) -> str:
        out, ti = [], 0
        for pos in (1, 2, 3):
            if pos == swept:
                out.append(f"{plane} + {iter_var}")
            else:
                base, _, iv = transverse[ti]
                out.append(f"{base} + {iv}")
                ti += 1
        return ", ".join(out)

    dst, src = subs("doff"), subs("soff")
    body = "\n".join(f"        q{r}d({dst}) = q{r}s({src})" for r in range(1, nrows + 1))
    dummies = ([(f"q{r}d", 3, "inout") for r in range(1, nrows + 1)] +
               [(f"q{r}s", 3, "in") for r in range(1, nrows + 1)])
    return tu.module(module, kernel, dummies=dummies, scalars=list(scalars), body=body,
                     nest_vars=nest_vars, nest_bounds=nest_bounds, doc=doc, wp=wp)
