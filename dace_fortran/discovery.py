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
    # A ROUTINE THAT IS NOT HERE IS AN ANSWER, NOT A TRACEBACK.  Both of these were bare `next(...)`
    # calls, so a caller who named the wrong file -- easy, since a routine's home is not derivable
    # from its name -- got `StopIteration` with no mention of the routine it was looking for.
    # MEASURED: that is how `s_periodic` (which lives in `m_boundary_primitives.fpp`, not
    # `m_boundary_common.fpp`) failed the first time it was asked for by name.
    b = next((i for i, l in enumerate(lines)
              if re.search(rf'\bsubroutine\s+{routine}\b', l, re.I)), None)
    if b is None:
        raise ValueError(f"no `subroutine {routine}` in the text ({len(lines)} lines); "
                         f"a routine's source file is not derivable from its name")
    e = next((i for i, l in enumerate(lines)
              if i > b and re.match(r'\s*end subroutine', l, re.I)), None)
    if e is None:
        raise ValueError(f"`subroutine {routine}` at line {b + 1} has no `end subroutine`")
    return '\n'.join(lines[b:e + 1])


# The two spellings of the same anchor.  A project's macro is the one to prefer where it exists --
# it is SEMANTIC (a reformat or a line insertion cannot break it), it BRACKETS the nest, and it
# CARRIES the structure: `collapse=N` is the depth the author intended and `private='[...]'` IS the
# set of loop variables.  Upstream cannot delete it without deleting their own GPU backend.
#
# The OpenACC directive is what the macro EXPANDS to, so it is the fallback -- and it is the reason
# a project's macro must be matched FIRST: the two agree post-expansion, but only the macro survives
# the case where the accelerator directives were not defined at all.
_ANCHORS = (
    # `$:GPU_PARALLEL_LOOP(collapse=3, private='[i, j, k, l]', ...)` -- MFC's, and a common shape.
    # The negation is load-bearing: the CLOSER is `$:END_GPU_PARALLEL_LOOP()`, which `\w*` happily
    # matches, so without it every nest is counted twice -- once at its opener and once at its closer,
    # where `nest_body` finds no `do` and runs to end-of-file.
    re.compile(r'\$:\s*(?!END_)\w*PARALLEL_LOOP\s*\(([^)]*)'),
    # `!$acc parallel loop collapse(3) ...`
    re.compile(r'!?\$acc\s+parallel\s+loop\b([^\n]*)'),
)


def _anchor_depth(clause: str, default: Optional[int]) -> Optional[int]:
    """The ``collapse=N`` a directive asks for, or ``default`` when it asks for none.

    READ, NOT ASSUMED.  The previous version keyed on the literal ``collapse(3)``: a fixed depth, so
    a nest the source collapses to 4 was invisible -- and a SILENT zero shapes is the failure mode
    here, not an error, because the anchor lines simply stop matching.
    """
    m = re.search(r'collapse\s*[=(]\s*(\d+)', clause)
    if m:
        return int(m.group(1))
    return default


def blocks(text: str, collapse: Optional[int] = None) -> List[Tuple[int, List[str]]]:
    """Every ``parallel loop collapse(n)`` nest, as ``(start_line, body_lines)``.

    The nest is delimited by its OUTERMOST ``do`` and the matching ``end do``: the body is collected
    with a depth counter so a nest whose body itself contains a loop is not mis-split.

    ``collapse`` filters by depth when given and accepts any depth when ``None``.  The default is
    ``None`` on purpose: the depth is IN the directive, so requiring the caller to already know it
    is requiring them to know the answer.
    """
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        for pat in _ANCHORS:
            m = pat.search(lines[i])
            if not m:
                continue
            d = _anchor_depth(m.group(1), collapse)
            if collapse is None or d == collapse:
                out.append((i + 1, nest_body(lines, i + 1)))
            break
        i += 1
    return out


def anchorless_blocks(text: str) -> List[Tuple[int, List[str]]]:
    """Outermost ``do``-nests, for a routine that carries no parallel-loop anchor.

    A FALLBACK, not the default, and that distinction is deliberate: for every family that HAS the
    macro, the macro is the better anchor -- it is semantic, it brackets the nest, and it carries
    ``collapse``.  Using the weaker signal by default would silently re-found every family by it.

    Why it is needed at all: MFC's ``s_periodic`` is CALLED from inside ``m_boundary_common``'s
    per-cell loop rather than containing a ``$:GPU_PARALLEL_LOOP`` of its own, so ``blocks()`` returns
    nothing, ``discover()`` finds no nests, and the routine reports ZERO SHAPES however good the shape
    vocabulary is.  MEASURED: adding a ``COPY`` op to ``canonical`` did not change
    ``--routine s_periodic`` from 0 shapes for exactly this reason.
    """
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        if re.match(r'\s*do\b', lines[i]):
            body = nest_body(lines, i)
            # REPORT `i`, NOT `i + 1`, so this COMPOSES with `extract.nest`.  `blocks()` reports the
            # 1-BASED line of the ANCHOR and collects from the line AFTER it -- so `nest(text, line)`
            # resumes at 0-based `line`.  Here the `do` IS the anchor, so the number that makes
            # `nest()` start at it is `i`.  MEASURED: reporting `i + 1` gave a body starting at the
            # INNER `do`, which cost each nest its outermost loop (`lb_lens=[1, 1]` for a 2-loop
            # copy) and made the emit route skip all three shapes silently.
            out.append((i, body))
            i += max(1, len(body))          # past this nest, so an inner `do` is not re-reported
            continue
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


def canonical(body: List[str], ops=("AVG", "GRAD", "FLUX", "COPY"), access: str = ACCESS) -> Optional[Shape]:
    """The shape of one loop body, or ``None`` when it does not fit the signature.

    ``None`` is a REPORTED outcome, not a fallback: a caller must be able to say "this nest is not
    one I recognise" rather than silently classifying it as something approximating.
    """
    loops = []
    for ln in body:
        # `[a-z_]\w*` AND NOT `[a-z]`: MFC spells the x-direction flux loop `k_loop` while the SAME
        # construct in y/z uses `k`.  A single-letter class matched ONE of the four loops, the
        # `len(loops) < 3` guard fired, and `canonical` returned None -- so the x nest was never a
        # FLUX shape, `arms_of` had no axis letter, and the whole direction was refused.
        # MEASURED: same construct, same routine, `canonical(x nest) = None` against
        # `canonical(y nest) = Shape(op='FLUX', dim=1, ...)` with IDENTICAL loop and statement
        # counts -- the only difference was the lettering.
        m = re.match(r'\s*do\s+([a-z_]\w*)\s*=\s*(.+?)\s*,\s*(.+?)\s*$', ln)
        if m:
            loops.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    stmts = [st for st in _join(body) if access + '(' in st and '=' in st]
    if not stmts:
        return None

    # A PURE COPY has no arithmetic, and that changes the loop rule below: MFC's `s_periodic` is
    #     do i = 1, sys_size                  <- the equation rows
    #       do j = 1, buff_size
    #         q_prim_vf(i)%sf(-j, k, l) = q_prim_vf(i)%sf(m - (j - 1), k, l)
    # -- TWO loops, not three, because a copy has no stencil to sweep and no transverse window.
    # MEASURED: s_periodic reported 0 shapes until this, and every reason was structural.
    # A copy's right-hand side is ONE ACCESS and nothing else.  Testing "no operators anywhere" is
    # WRONG, and measured wrong: MFC's `s_periodic` is
    #     q_prim_vf(i)%sf(-j, k, l) = q_prim_vf(i)%sf(m - (j - 1), k, l)
    # -- whose right-hand side contains `-`, in its INDEX.  The operator is in the subscript, not in
    # the arithmetic, so the test has to look at the whole shape of the RHS: an access with its
    # subscript list, optionally wrapped, and nothing else.
    # The WHOLE access, BASE INCLUDED -- `q_prim_vf(i)%sf(...)`, not just the `%sf(...)` tail.  An
    # anchored pattern over the tail alone never matches, because the string STARTS with the base.
    # (Measured: the first version of this returned False for the very copy it was written for.)
    _chain = (r'[A-Za-z_]\w*(?:\([^()]*\))?'
              r'(?:%\w+(?:\([^()]*\))?)*' + re.escape(access) +
              r'\s*\([^()]*(?:\([^()]*\)[^()]*)*\)')
    _bare = re.compile(r'^[\s&]*' + _chain + r'[\s&]*$')
    is_copy = bool(stmts) and all(
        _bare.match(st.split('=', 1)[1]) for st in stmts if '=' in st)

    # The THREE spatial loops.  A fourth loop (the equation rows) often sits inside every one of
    # them and is not part of the shape -- the rows are unrolled in the generated kernel.
    if len(loops) < (2 if is_copy else 3):
        return None
    loops = loops[:2 if is_copy else 3]
    st = stmts[0]
    # The write's subscript tuple.  Match AFTER `%sf(`: an array NAME carries its own parentheses
    # (`foo(2)%vf(i)%sf`), so a `name(...)` pattern captures the `(2)` and sees a single subscript.
    write = re.search(re.escape(access) + r'\s*\(([^)]*)\)\s*=', st)
    rhs = st[write.end():] if write else st
    # The operation is a property of the WHOLE body (see the module docstring).
    whole = ' '.join(stmts)
    dim = _difference_position(rhs, access)
    if dim is None:
        # THE READS MAY BE HOISTED INTO TEMPORARIES.  Measured on MFC's
        # `s_compute_advection_source_term`: `flux_face1 = flux_n(1)%vf(j)%sf(k_loop - 1, ...)` sits
        # two statements BEFORE the write, whose right-hand side is then
        # `inv_ds*(flux_face1 - flux_face2)` and contains no access at all.  Scanning only the RHS
        # finds nothing, the shape comes out `dim = None`, and with no axis letter `arms_of`
        # classifies every arm `both/neither` -- so nothing is ever paired and the group is refused.
        # A hoisted read is no less a read, so the whole body is the right scope.
        dim = _difference_position(whole, access)
    is_avg = bool(re.search(r'25\.e-2|25\.d-2|0\.25', whole))
    is_grad = bool(re.search(r'/\s*\(', whole)) and not is_avg
    # A ONE-SIDED FLUX DIFFERENCE: `inv_ds*(f(k-1) - f(k))`, both reads on the same access.  It is
    # neither AVG (a four-term face average) nor GRAD (a difference divided by a SPACING difference):
    # it scales a face difference by a reciprocal.  Added for MFC's `s_compute_advection_source_term`,
    # whose nests previously came back `?` -- and a `?` shape has NO axis letter, so `arms_of`
    # classified every arm `both/neither`, nothing was ever paired, and the whole group was blocked
    # with "could not pair the arms by the direction their stencil reaches".  MEASURED before this:
    # 6 shapes discovered, 6 blocked, 0 emitted.
    #
    # `inv_ds` alone does not name it -- the advection term uses it too -- so the body must also READ
    # one access at an index and at its neighbour.  That is what makes it a DIFFERENCE and not a
    # scaling.
    is_flux = (bool(re.search(r'\binv_ds\b', whole)) and not is_avg and not is_grad
               and bool(re.search(re.escape(access) + r'\s*\([^)]*-\s*1\b', whole)))
    op = 'COPY' if is_copy else ('AVG' if is_avg else ('GRAD' if is_grad else ('FLUX' if is_flux else '?')))
    if op not in ops and op != '?':
        op = '?'
    return Shape(op, dim, tuple(l[0] for l in loops))


def discover(text: str, routine: Optional[str] = None,
             access: str = ACCESS) -> Dict[Shape, List[int]]:
    """``{shape: [source line numbers]}`` for the nests in ``text`` (or one routine of it)."""
    if routine:
        text = scope_to_routine(text, routine)
    found: Dict[Shape, List[int]] = {}
    blks = blocks(text)
    if not blks:
        # See `anchorless_blocks`: a routine with no anchor yields no blocks and therefore no
        # shapes, whatever the vocabulary can name.  The fallback is used ONLY when the anchor
        # found nothing, so no family that has the macro changes behaviour.
        blks = anchorless_blocks(text)
    for ln, body in blks:
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
