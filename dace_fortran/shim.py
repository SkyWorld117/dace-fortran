# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Normalise a Fortran shim that calls a compiled SDFG.

A hand-written (or generated) Fortran binding to a DaCe library has to solve a problem DaCe creates:
**the kernel launches on a stream DaCe created itself, with ``cudaStreamNonBlocking``** -- i.e.
explicitly opted out of the legacy default stream's implicit synchronisation -- while the rest of an
existing accelerator codebase is very likely on that default stream.  So a DaCe kernel and the loops
that fill its inputs, or read its outputs, are unordered in BOTH directions.  That is a
read-after-write hazard, and it is invisible: the code is correct-looking and mostly passes.

Two things go wrong in practice, and this module exists for both:

  * the ordering is written as a HOST-BLOCKING synchronise (``cudaDeviceSynchronize``), which costs
    wall-clock for no reason -- the ordering needed is a stream assignment, not a wait;
  * it is applied INCONSISTENTLY.  In the codebase this was written for, eleven shims ended each
    entry with a synchronise and three had none at all -- so those three ran unordered.

So :func:`route_launches` rewrites every kernel launch in a shim to be followed by one call to an
ordering helper, inserting it where absent and replacing a synchronise where present, and places the
helper's import in the module HEADER (a `use` inside a contained subroutine lands at the wrong scope,
which the compiler reports only as an undefined external at link time).

WHAT IS NOT HERE.  The helper itself -- what stream to point DaCe at, and how to report a launch
error -- is the caller's module, because it is a policy choice (this tree uses
``__dace_gpu_set_all_streams(state, 0)``, see dace's exported ABI).  The names come in as arguments.

TWO PASSES, because the ordering is written in two places: after each launch, and at the END of an
entry, where a multi-case shim puts one synchronise after a ``select``/``if`` chain instead of after
each launch.  A sync that is NEITHER is left alone -- those sit between staging steps (a device->host
copy after a kernel), where the sync may be load bearing for the copy rather than for the launch.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# `call <name>_run(_x) (<state>, ...)`.  The state variable has no single convention: `<tag>_state`
# and `state_<tag>` both appear in one codebase, and a state may be an ARRAY (`state_sweeps(3)`)
# passed with a subscript -- so the tag is taken from the CALL, which is uniform, and any subscript
# is captured with the state rather than dropped.
LAUNCH = re.compile(r'\s*call\s+(\w+_run(?:_[xyz])?)\s*\(\s*((?:\w+_state|state_\w+)(?:\([^)]*\))?)')
SYNC = re.compile(r'\s*ierr\s*=\s*cudaDeviceSynchronize_\(\)')
ONELINE = re.compile(r'\s*ierr\s*=\s*cudaDeviceSynchronize_\(\)\s*;\s*call\s+chk\(')


def statement_end(lines: Sequence[str], i: int) -> int:
    """Last line index of the statement starting at `i` (Fortran `&` continuation)."""
    while lines[i].rstrip().endswith('&'):
        i += 1
    return i


def skip_blanks(lines: Sequence[str], i: int) -> int:
    while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith('!')):
        i += 1
    return i


def tag_of(call_name: str) -> str:
    """`sweeps_run_x` -> `sweeps_x`, `rk_run` -> `rk`.

    `_run` may be INFIXED (the directional families) or suffixed, so it cannot simply be stripped
    from the end -- doing that leaves `sweeps_run_x` unchanged and produces a silently confusing tag.
    """
    return re.sub(r'_run(?=_|$)', '', call_name)


def sync_span(lines: Sequence[str], i: int) -> Tuple[int, int] | None:
    """If a synchronise statement starts at `i`, the (first, last) indices of the whole thing.

    Three spellings, all a scheduling barrier followed by an error report: an
    ``if (ierr /= 0) then ... end if`` block; the same thing on one line as
    ``ierr = cudaDeviceSynchronize_(); call chk(ierr, '...')``; and neither.  MISSING ONE DOES NOT
    FAIL LOUDLY -- it silently leaves that launch unordered, which is the whole bug -- so all are
    matched here.
    """
    if ONELINE.match(lines[i]):
        return i, i
    if not SYNC.match(lines[i]):
        return None
    j = skip_blanks(lines, i + 1)
    if j >= len(lines) or lines[j].strip() != 'if (ierr /= 0) then':
        return None
    while j < len(lines) and lines[j].strip() != 'end if':
        j += 1
    return i, j


def route_launches(text: str, *, call: str, use_line: str, comment: Sequence[str] = (),
                   ) -> Tuple[str, List[str]]:
    """Ensure every kernel launch is PRECEDED by exactly one ordering call.  Idempotent.

    NOT "followed by": the launch reads ``state->gpu_context->streams[0]`` as its own trailing
    argument at the call, so an ordering call placed after it misses that launch and only takes
    effect on the next one.  See ``_pass_launches`` for the generated-code evidence.

    :param call: the ordering routine, e.g. ``"dace_order"``.
    :param use_line: the whole import line, e.g. ``"  use m_dace_order, only: dace_order"``.
    :param comment: lines explaining the ordering, emitted above each inserted call.
    :returns: ``(new_text, rewrites)`` -- the rewrites are for a caller that wants to report them.
    """
    rewrites: List[str] = []
    lines = _pass_launches(text.split('\n'), call, comment, rewrites)
    lines = _pass_entry_tail(lines, rewrites)
    text = '\n'.join(lines)

    # Import it in the module HEADER.  A big shim has `use` statements inside its contained
    # subroutines, and inserting after one of THOSE puts the import at the wrong scope: the routine
    # is then not visible at module level, the compiler treats it as a plain external, and the link
    # fails with `undefined reference` -- the compiler says nothing.
    if f'call {call}(' in text:
        ls = text.split('\n')
        head = next((i for i, l in enumerate(ls) if re.match(r'\s*contains\b', l)), len(ls))
        strays = [i for i, l in enumerate(ls) if i >= head and use_line.strip() in l]
        for i in reversed(strays):
            del ls[i]
        if strays:
            rewrites.append(f"moved a misplaced `{use_line.strip()}` out of a subroutine")
        if not any(use_line.strip() in l for l in ls):
            uses = [i for i, l in enumerate(ls) if i < head and re.match(r'\s*use\s', l)]
            ls.insert(max(uses) + 1 if uses else head, use_line)
            rewrites.append(f"added `{use_line.strip()}` to the module header")
        text = '\n'.join(ls)

    # Drop the synchronise interface only when nothing calls it any more.
    if not SYNC.search(text):
        text = re.sub(r'\n\s*function cudaDeviceSynchronize_\(\).*?\n\s*end function\n',
                      '\n', text, flags=re.S)
    return text, rewrites


def _pass_launches(lines: List[str], call: str, comment: Sequence[str],
                   rewrites: List[str]) -> List[str]:
    """Put the ordering call BEFORE each launch, not after it.

    THE SIDE MATTERS, AND IT IS DECIDABLE FROM THE GENERATED CODE.  DaCe's launch reads the stream
    as its own trailing argument, evaluated at the call::

        cudaLaunchKernel((void*)…_map_0_1_17, dim3(…), dim3(32,1,1), …_args, 0,
                         __state->gpu_context->streams[0]);

    and ``__dace_gpu_set_all_streams`` writes exactly that field.  A setter invoked AFTER the launch
    has therefore already missed it: the call handed the previous ``streams[0]`` to
    ``cudaLaunchKernel``, which is the stream DaCe created with ``cudaStreamNonBlocking`` -- i.e.
    explicitly NOT ordered against the legacy default stream the surrounding OpenACC loops use.  The
    setter then takes effect on the NEXT launch, so the first launch of each direction stays
    unordered.  This pass used to emit the call after the launch, which is that bug.

    Idempotent, and idempotent specifically FOR THIS PLACEMENT: an existing call in front of the
    launch is left exactly as it is, but only when its TAG matches -- a call belonging to a
    different launch is not this launch's ordering, and treating it as one would silently leave this
    one unordered.  A call the OLD placement left downstream, or a sync sitting there, is consumed
    so the rewrite converges rather than duplicating.
    """
    tag_re = rf"\s*call\s+{re.escape(call)}\s*\([^,]+,\s*'([^']+)'\s*\)"
    out, i = [], 0
    while i < len(lines):
        m = LAUNCH.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        last = statement_end(lines, i)
        state, tag = m.group(2), tag_of(m.group(1))
        want = f"    call {call}({state}, '{tag}')"
        # Already ordered FOR THIS TAG: an existing call may sit after blank lines and a comment
        # block, so look backward past both rather than at the immediately preceding line.
        b = i - 1
        while b >= 0 and (not lines[b].strip() or lines[b].strip().startswith('!')):
            b -= 1
        if b >= 0:
            prev = re.match(tag_re, lines[b])
            if prev and prev.group(1) == tag:
                # Already routed FOR THIS TAG.  CORRECT it if the state expression is wrong: an
                # array-valued state is one library per direction and must carry its subscript
                # (`state_sweeps(1)`, not `state_sweeps`), so a transformer that merely skipped would
                # leave that mistake in place forever.  The line is already in `out`, possibly
                # shifted by earlier insertions, so find it by scanning back rather than by index.
                if lines[b].strip() != want.strip():
                    for idx in range(len(out) - 1, -1, -1):
                        if re.match(tag_re, out[idx]):
                            out[idx] = want
                            rewrites.append(f"corrected {call}({tag}) -> {state}")
                            break
                out.append(lines[i])
                i += 1
                continue
        # Consume what the old placement left DOWNSTREAM of this launch, so a shim normalised by
        # the previous version converges here instead of gaining a second call.
        tail = last
        j = skip_blanks(lines, last + 1)
        if j < len(lines):
            if re.match(rf'\s*call\s+{re.escape(call)}\s*\(', lines[j]):
                tail = j
                # It is being moved, but say so when its state expression was wrong -- the same
                # mistake the already-routed branch corrects, seen from the other side.
                if lines[j].strip() != want.strip():
                    rewrites.append(f"corrected {call}({tag}) -> {state}")
            else:
                span = sync_span(lines, j)
                if span is not None:
                    tail = span[1]
        out.extend(comment)
        out.append(want)
        out.extend(lines[i:last + 1])
        rewrites.append(f"inserted {call}({tag}) before {m.group(1)}")
        i = tail + 1
    return out


def _pass_entry_tail(lines: List[str], rewrites: List[str]) -> List[str]:
    out, i, n = [], 0, len(lines)
    while i < n:
        if re.match(r'\s*subroutine\s', lines[i]) or re.match(r'\s*function\s', lines[i]):
            end = i
            while end < n and not re.match(r'\s*end (subroutine|function)', lines[end]):
                end += 1
            body = lines[i:end]
            k = len(body) - 1
            while k >= 0 and (not body[k].strip() or body[k].strip().startswith('!')):
                k -= 1
            span = sync_span(body, k) if k >= 0 else None
            if span is not None and span[0] == k:
                rewrites.append(f"dropped the entry-tail sync in {body[0].strip()[:44]}")
                body = body[:k] + body[span[1] + 1:]
            out.extend(body)
            i = end
            continue
        out.append(lines[i])
        i += 1
    out.extend(lines[i:])
    return out
