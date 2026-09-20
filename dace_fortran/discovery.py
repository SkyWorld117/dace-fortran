# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Discover the loop FAMILIES in a Fortran routine, and report their shapes.

A *shape* is what makes one generated kernel cover many loops.  In the routines this was built for
every ``collapse(3)`` nest has the same structure -- a three-deep do-nest over the storage dimensions
with the accesses written in the same subscript order -- so the subscript ORDER carries no
information at all, and the only things that vary are

  * which subscript position carries the difference (the ``-1`` or ``+1``), and
  * the operation being performed.

The signature is therefore ``(op, difference subscript position)``, and a *family* is a set of loops
that are the two ARMS (L and R) of one such shape.  Groups found this way are exactly the groups a
human arrives at by reading, and -- more usefully -- they are checkable: a shape present in the
source and absent from a generator's coverage table is a hard failure rather than a silent omission.

THREE PROPERTIES OF THE SOURCE, each of which broke a first version of this parser.  They are
properties of a common Fortran house style rather than of any one project, so they are worth knowing
before trusting a rewrite of this file:

  * **the subscript order carries no information.**  Loops run outermost = last storage dimension,
    innermost = first, and the array is declared ``(dim0, dim1, dim2)`` -- so every access reads the
    nest's letters in reverse order.  What varies is only *which position* carries the difference.
  * **the difference lives on the READS.**  These loops write at the base index and shift the stencil
    on the right-hand side, so scanning the write's subscripts finds nothing.
  * **the operation is a property of the WHOLE body.**  A face average is often TWO statements (the
    multi-term sum, then a scaling of the same lvalue), so classifying from the first assignment
    misses the factor.  And lines are wrapped, so continuations must be folded first -- reading them
    one at a time truncates every right-hand side.

INPUT is preprocessed source: the raw source carries preprocessor directives and the loop bodies are
what we want, which is what a preprocessor has already produced.

The result is a proposal, never an approximation: a nest that does not fit the signature is reported
as unrecognised rather than forced into a shape.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


class Shape(NamedTuple):
    """The signature of one loop family.  ``op`` is the operation the body performs and ``dim`` the
    storage subscript position carrying the difference (``None`` when it could not be determined)."""
    op: str
    dim: Optional[int]
    loops: Tuple[str, ...]


def scope_to_routine(text: str, routine: str) -> str:
    """Just the named subroutine, so a caller can discover one routine's shapes.

    Separate from :func:`discover` because a gate should ask the same question the human-readable
    report answers -- a report and a gate that can disagree about what the source contains is exactly
    what a hand-maintained table hides.
    """
    lines = text.split('\n')
    b = next(i for i, l in enumerate(lines) if re.search(rf'\bsubroutine\s+{routine}\b', l, re.I))
    e = next(i for i, l in enumerate(lines) if i > b and re.match(r'\s*end subroutine', l, re.I))
    return '\n'.join(lines[b:e + 1])


def blocks(text: str, collapse: int = 3) -> List[Tuple[int, List[str]]]:
    """Every ``parallel loop collapse(n)`` nest, as ``(start_line, body_lines)``.

    The nest is delimited by its OUTERMOST ``do`` and the matching ``end do``: the body is collected
    with a depth counter so a nest whose body itself contains a loop is not mis-split.
    """
    lines = text.split('\n')
    out, i = [], 0
    pat = re.compile(r'!?\$acc parallel loop.*collapse\(%d\)' % collapse)
    while i < len(lines):
        if pat.search(lines[i]):
            out.append((i + 1, nest_body(lines, i + 1)))
            i += 1
        i += 1
    return out


def nest_body(lines: List[str], start: int) -> List[str]:
    """The lines of the nest beginning at ``start``, INCLUDING its closing ``end do``s.

    Counting ``end do`` per OCCURRENCE rather than per line, and dropping trailing comments before
    counting: house styles routinely pack several loop ends onto one line (``end do; end do; end do``)
    and a counter that only recognises a lone ``end do`` never returns to zero, so the scan runs to
    end-of-file and every nest collapses into one.
    """
    j, depth, started, body = start, 0, False, []
    while j < len(lines):
        code = lines[j].split('!')[0]
        n_end = len(re.findall(r'\bend\s*do\b', code, re.I))
        if re.match(r'\s*do\b', code, re.I):
            depth += 1
            started = True
        if started:
            depth -= n_end
            body.append(lines[j])
            if depth <= 0:
                break
        else:
            body.append(lines[j])
        j += 1
    return body


def _join(body: List[str]) -> List[str]:
    """The body's statements, with Fortran continuations folded.

    Necessary, not cosmetic: these lines are wrapped at ~120 columns and the right-hand side of every
    assignment continues onto the next line -- reading them line by line truncates the expression,
    which is how a first version of this classified nineteen averaging loops as unknown.

    THE `&` IS AT BOTH ENDS, and stripping only the trailing one is not a near miss -- it is INVALID
    FORTRAN.  The house style this was built against wraps as

        dqL_prim_dx_n(2)%vf(i)%sf(k, j, &
                      & l) + dqR_prim_dx_n(1)%vf(i)%sf(k, j, l)

    so a fold that keeps the leading `&` emits `%sf(k, j, & l)` in the middle of a statement.  That
    compiles nowhere, and it is the kind of defect that surfaces as "the generator produced
    something" rather than as a shape that failed to classify.  Both ends are stripped.
    """
    out, cur = [], ''
    for ln in body:
        t = ln.strip()
        if not t or t.startswith('!'):
            continue
        if t.startswith('&'):
            t = t[1:].lstrip()
        if t.endswith('&'):
            cur += t[:-1]
            continue
        cur += t
        out.append(cur)
        cur = ''
    if cur:
        out.append(cur)
    return out


# The array-access idiom whose subscripts carry the stencil.  `%sf(...)` is a derived-type component
# access -- generic Fortran, and the one this was built against -- but it is an IDIOM, not a language
# feature, so it is a parameter rather than a literal.  Everything that reads subscripts goes through
# it, because a parser that hard-codes one house style is a parser that silently finds nothing in the
# next codebase (it returns zero shapes, not an error).
ACCESS = r'%sf'


def _access_re(access: str) -> "re.Pattern":
    return re.compile(re.escape(access) + r'\s*\(([^)]*)\)')


# Public alias: the statement folding is useful on its own (extraction hands these to a caller), and
# the leading underscore was only ever an accident of it being written for this module first.
statements = _join


def _difference_position(stmt: str, access: str = ACCESS) -> Optional[int]:
    """The subscript position carrying a ``+1``/``-1`` shift, from the READS on the right-hand side.

    Scans the whole right-hand side rather than the first access, and prefers a unit shift if one is
    present, because a stencil may legitimately reference other offsets too.
    """
    d = None
    for m in _access_re(access).finditer(stmt):
        for pos, sub in enumerate(x.strip() for x in m.group(1).split(',')):
            shift = re.search(r'[+-]\s*(\d)', sub)
            if shift and (d is None or abs(int(shift.group(1))) == 1):
                d = pos
        if d is not None:
            break
    return d


def canonical(body: List[str], ops=("AVG", "GRAD"), access: str = ACCESS) -> Optional[Shape]:
    """The shape of one loop body, or ``None`` when it does not fit the signature.

    ``None`` is a REPORTED outcome, not a fallback: a caller must be able to say "this nest is not
    one I recognise" rather than silently classifying it as something approximating.
    """
    loops = []
    for ln in body:
        m = re.match(r'\s*do\s+([a-z])\s*=\s*(.+?)\s*,\s*(.+?)\s*$', ln)
        if m:
            loops.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    # The THREE spatial loops.  A fourth loop (the equation rows) often sits inside every one of
    # them and is not part of the shape -- the rows are unrolled in the generated kernel.
    if len(loops) < 3:
        return None
    loops = loops[:3]

    stmts = [st for st in _join(body) if access + '(' in st and '=' in st]
    if not stmts:
        return None
    st = stmts[0]
    # The write's subscript tuple.  Match AFTER `%sf(`: an array NAME carries its own parentheses
    # (`foo(2)%vf(i)%sf`), so a `name(...)` pattern captures the `(2)` and sees a single subscript.
    write = re.search(re.escape(access) + r'\s*\(([^)]*)\)\s*=', st)
    rhs = st[write.end():] if write else st
    dim = _difference_position(rhs, access)

    # The operation is a property of the WHOLE body (see the module docstring).
    whole = ' '.join(stmts)
    is_avg = bool(re.search(r'25\.e-2|25\.d-2|0\.25', whole))
    is_grad = bool(re.search(r'/\s*\(', whole)) and not is_avg
    op = 'AVG' if is_avg else ('GRAD' if is_grad else '?')
    if op not in ops and op != '?':
        op = '?'
    return Shape(op, dim, tuple(l[0] for l in loops))


def discover(text: str, routine: Optional[str] = None,
             access: str = ACCESS) -> Dict[Shape, List[int]]:
    """``{shape: [source line numbers]}`` for the nests in ``text`` (or one routine of it)."""
    if routine:
        text = scope_to_routine(text, routine)
    found: Dict[Shape, List[int]] = {}
    for ln, body in blocks(text):
        shape = canonical(body, access=access)
        if shape is None:
            continue
        found.setdefault(shape, []).append(ln)
    return found


def report(text: str, routine: Optional[str] = None, source: str = "<text>") -> str:
    """A human-readable summary, in the same terms :func:`discover` reports."""
    if routine:
        text = scope_to_routine(text, routine)
    nests = blocks(text)
    found: Dict[Shape, List[int]] = {}
    unrecognised = []
    for ln, body in nests:
        shape = canonical(body)
        if shape is None:
            unrecognised.append(ln)
            continue
        found.setdefault(shape, []).append(ln)

    out = [f"{len(nests)} collapse(3) nest(s) in {source}"]
    if routine:
        out.insert(0, f"scoped to {routine}")
    for ln in unrecognised:
        out.append(f"  line {ln:5d}  (not a simple 3-deep nest -- UNRECOGNISED, not classified)")
    out.append("SHAPES DISCOVERED (op, difference subscript position):")
    for shape in sorted(found, key=lambda k: (k.op, -1 if k.dim is None else k.dim)):
        lns = found[shape]
        pairs = f"{len(lns)} loop(s) = {len(lns) // 2} pair(s)"
        out.append(f"  {shape.op:<5} dim{shape.dim}   {pairs:<22} lines {lns}")
    out.append(f"{len(found)} distinct shape(s), {len(unrecognised)} unrecognised nest(s)")
    return '\n'.join(out)


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    ap.add_argument("source", nargs="?", help="a preprocessed .f90 (or omit with --fixture)")
    ap.add_argument("--routine", default=None, help="scope to one subroutine")
    ap.add_argument("--fixture", action="store_true", help="run the packaged self-test fixture")
    a = ap.parse_args(argv)
    if a.fixture or not a.source:
        from pathlib import Path as _P
        f = _P(__file__).resolve().parents[1] / "tests" / "fixtures" / "nest_shapes.f90"
        text = f.read_text()
        print(report(text, "shapes", source=f.name))
        return 0
    p = Path(a.source)
    print(report(p.read_text(), a.routine, source=p.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
