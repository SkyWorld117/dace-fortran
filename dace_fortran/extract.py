# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pull a loop nest, and the statements of its body, out of preprocessed Fortran.

This is the second stage of the TU-generation path: :mod:`dace_fortran.discovery` says WHICH nests
exist and what shape they are; this says how to get one's text out, and how to get from a `.fpp` to
something parseable at all.

What is here is the part that generalises -- the C-preprocessor pass, the nest slice, and the
statement folding.  What is deliberately NOT here is fypp: templating is a property of the *project*
being ported (its directives, its `#:set` variables, the `.venv` that has the tool), not of Fortran,
and a general frontend that shipped one project's template engine would be a general frontend with a
project's build system inside it.  A caller that needs it runs it and hands the result here, which is
exactly the interface these functions take: TEXT in, TEXT out.

THE ORDER MATTERS, AND GETTING IT WRONG IS SILENT.  A templating pass is not the last step: its
output still carries `#if` directives and `# <line> "<file>"` markers, because the C preprocessor runs
later, as part of compilation.  Parsing that raw output works for SHAPE discovery (the directives
contain no `do` and no access subscripts) and fails for anything that reads statements.  So the order
is: template -> :func:`cpp` -> discovery -> extraction.

And the define list must be the BUILD's, not a guess.  A codebase that guards its accelerator
directives with `#if defined(SOMETHING)` will, when that macro is omitted, have every anchor line
stripped -- and the anchors are how nests are located, so the failure mode is a silent zero shapes
rather than an error.  Take the defines from the build's own flags
(`CMakeFiles/<target>_lib.dir/flags.make` or equivalent), not from a plausible-looking list.

The three functions were each a hand-rolled step in the porting scripts, reimplemented per generator:
`subprocess.check_call([... "cpp", "-P", "-C"] + [["-D", d] for d in defines])` in one, a line-slice
in another, a continuation folder in a third.  Consolidating them is what makes the extraction step
checkable -- a difference that used to be invisible is now one function's contract.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from dace_fortran.discovery import blocks, nest_body, statements  # public re-export


class PreprocessorMissing(RuntimeError):
    """The C preprocessor is not on PATH.  Named, because the failure is otherwise an
    opaque ``FileNotFoundError`` from deep inside a subprocess call."""


def cpp(text: str, defines: Sequence[str] = (), keep_comments: bool = True,
        executable: Optional[str] = None) -> str:
    """Run the C preprocessor over ``text`` and return the result.

    ``-P`` (no line markers) and ``-C`` (keep comments) are not incidental: the porting scripts'
    markers ARE comments -- a nest is located by the ``!$acc parallel loop`` above it -- so dropping
    comments would drop the anchors that make extraction possible at all.
    """
    exe = executable or shutil.which("cpp")
    if not exe:
        raise PreprocessorMissing(
            "no C preprocessor on PATH (looked for 'cpp'); pass executable=... or install one. "
            "It is needed because the source carries #if/#define that change which loops exist.")
    cmd = [exe, "-P"] + (["-C"] if keep_comments else []) + sum([["-D", d] for d in defines], [])
    cp = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"cpp failed ({cp.returncode}):\n{cp.stderr}")
    return cp.stdout


def nest(text: str, line: int) -> str:
    """The loop nest beginning at ``line`` (1-based), INCLUDING its closing ``end do``s.

    ``line`` is a line number as :func:`dace_fortran.discovery.blocks` reports them, so the two
    compose: discover says ``line 212``, this returns that nest.
    """
    lines = text.split('\n')
    if not 1 <= line <= len(lines):
        raise IndexError(f"line {line} is outside the text ({len(lines)} lines)")
    # NOT `line - 1`: `blocks` reports the 1-BASED line of the `!$acc` directive but collects from the
    # line AFTER it, so the 0-based index to resume at is exactly `line`.
    return '\n'.join(nest_body(lines, line))


def nest_at_nest_index(text: str, index: int, collapse: int = 3) -> str:
    """The ``index``-th nest (0-based) in ``text``, for callers that iterate rather than locate."""
    found: List[Tuple[int, List[str]]] = blocks(text, collapse=collapse)
    if not 0 <= index < len(found):
        raise IndexError(f"nest {index} does not exist: {len(found)} nest(s) in the text")
    return '\n'.join(found[index][1])


def routine(text: str, name: str) -> str:
    """The text of one subroutine or function, by name.  Raises rather than guessing."""
    lines = text.split('\n')
    pat = rf'^\s*(subroutine|function)\s+{name}\b'
    import re
    b = next((i for i, l in enumerate(lines) if re.match(pat, l, re.I)), None)
    if b is None:
        raise KeyError(f"no subroutine/function named {name!r}")
    kind = 'function' if re.match(r'\s*function', lines[b], re.I) else 'subroutine'
    e = next((i for i, l in enumerate(lines) if i > b and re.match(rf'\s*end\s*{kind}', l, re.I)),
             None)
    if e is None:
        raise KeyError(f"{name!r} has no matching end {kind}")
    return '\n'.join(lines[b:e + 1])


def from_file(path, defines: Sequence[str] = (), **kw) -> str:
    """Read a source file and preprocess it.  Convenience for the common two-step."""
    return cpp(Path(path).read_text(), defines=defines, **kw)


def branch_of(scoped: str, line: int) -> str:
    """The branch a 0-based ``line`` sits in: the nearest PRECEDING ``if (...) then``, as text.

    Returns ``""`` when the line is in no ``if`` at all.

    AN ``else`` ARM IS THE NEGATION OF ITS ``if``, AND RETURNING THE SAME TEXT FOR BOTH IS THE BUG
    THIS GUARDS.  Scanning back for the nearest ``if (...) then`` alone gives an ``if`` arm and its
    ``else`` arm the SAME string -- MEASURED on a source whose x-direction nest appears once in
    ``if (hypo_nc_mode /= hypo_nc_mode_dual_pass)`` and once in the ``else``.  A caller that selects
    "the nest inside guard G" then keeps BOTH, emits two DIFFERENT arithmetic forms under one library
    name, and the second silently overwrites the first on the way to the bake.

    So an ``else`` arm is returned negated, and a caller naming the positive guard selects the positive
    arm and only that one.

    AN ``else if`` ARM CARRIES ITS OWN CONDITION TOO, and returning only the negated outer guard makes
    it INDISTINGUISHABLE FROM THE PLAIN ``else`` -- both would answer ``!.(a)`` in
    ``if (a) ... else if (b) ... else ...``, which is the same collision one level in.  So an ``else
    if`` arm answers ``!.(a) && (b)``: the negation of the guard it is the alternative to, conjoined
    with its own.
    """
    lines = scoped.split("\n")
    for i in range(min(line, len(lines)) - 1, -1, -1):
        if re.match(r"\s*else\b", lines[i], re.I):
            m_ei = re.match(r"\s*else\s+if\s*\((.*)\)\s*then", lines[i], re.I)
            for j in range(i - 1, -1, -1):
                m = re.match(r"\s*(?:else\s+)?if\s*\((.*)\)\s*then", lines[j], re.I)
                if m:
                    outer = "!.(" + m.group(1).strip() + ")"
                    return f"{outer} && ({m_ei.group(1).strip()})" if m_ei else outer
            return "!(else)"
        m = re.match(r"\s*if\s*\((.*)\)\s*then", lines[i], re.I)
        if m:
            return m.group(1).strip()
    return ""
