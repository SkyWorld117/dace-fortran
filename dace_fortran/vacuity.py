# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Anti-vacuous extraction guards: refuse a built SDFG whose work the build silently dropped.

WHY THIS EXISTS.  A build that succeeds is not evidence that the build preserved the source.
Two measured holes motivate this module -- both produce an SDFG that compiles, runs, and is WRONG
in a way no counter can see:

  1. ZERO COMPUTE NODES.  A routine whose body is a single unresolved ``CALL
     some_unknown_external(a, n, n)`` builds to an SDFG with one empty state, one array, no error,
     and the used dummy ``n`` gone from the argument list.  The generic ``kind="call"`` AST node
     reaches :func:`dace_fortran.builder.emit_library.emit_call`, which for an unregistered callee
     is a documented NO-OP (``if sig is None: return``).  The call -- the whole body -- is dropped.
     The verify pass that would have caught it (``hlfir-verify-no-unresolved-calls``) exists in the
     C++ bridge but is only wired into ``MULTI_FILE_PIPELINE``, which no ``build_sdfg*`` entry point
     selects (they all default to ``DEFAULT_PIPELINE``).  <- the root cause, on the *build* side.

  2. EXTERNALISED CALL CARRYING ARITHMETIC.  ``keep_external`` / ``apply_external_functions`` is the
     seam for halos, Registry, I/O and error helpers: it stubs the callee's body before inlining and
     leaves an opaque call.  That is correct for a helper with no arithmetic and WRONG for a math
     helper -- the arithmetic never enters the SDFG, yet the graph looks plausible.

This module is deliberately PYTHON-ONLY: it inspects an already-built :class:`dace.SDFG` and the
source it came from, so it needs no C++ bridge rebuild and can gate any producer.  It is a LIVENESS
check, not a semantic-equivalence check -- see :func:`assert_nonvacuous` for what it cannot see.

Call it from the probe harness and from the bake/producer, on every extraction:

    from dace_fortran.vacuity import assert_nonvacuous
    assert_nonvacuous(sdfg, source=merged_tu, externalized=registered_names(),
                      callee_sources=original_bodies, allow=["mpi_send"])
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

__all__ = [
    "VacuityError",
    "compute_nodes",
    "external_call_names",
    "defined_procedures",
    "source_call_sites",
    "arithmetic_lines",
    "assert_nonvacuous",
]


class VacuityError(RuntimeError):
    """A built SDFG is vacuous -- the build dropped work the source performs.

    Raised (never caught by this module) with a message naming the callee / the state, so the fix
    is visible: add the missing dependency, register the external, or stop externalising a body
    that carries arithmetic.
    """


# ---------------------------------------------------------------------------------------------
# SDFG inspection
# ---------------------------------------------------------------------------------------------
def _compute_classes():
    from dace.sdfg import nodes as _n

    # A MapEntry stands in for its whole map (entry + exit); MapExit alone is not compute.
    # LibraryNode covers ExternalCall / BLAS / FFT / etc.  NestedSDFG is compute because it wraps
    # a body.  ConsumeEntry covers stream/consume regions.
    return (_n.Tasklet, _n.MapEntry, _n.LibraryNode, _n.NestedSDFG, _n.ConsumeEntry)


def compute_nodes(sdfg) -> list:
    """Every node that represents WORK (a tasklet, map, library/external node, or nested SDFG).

    Returned as ``[(state, node), ...]`` in state/node order so a caller can print the first few.
    Access nodes, empty states and interstate edges are not work.
    """
    classes = _compute_classes()
    out = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, classes):
                out.append((state, node))
    return out


def external_call_names(sdfg) -> Set[str]:
    """The ``c_name`` of every :class:`dace_fortran.external.ExternalCall` library node in ``sdfg``.

    These are calls that were *emitted* by ``emit_call`` for a registered external -- i.e. the
    ``keep_external`` / ``apply_external_functions`` policy actually fired.  An unregistered call
    leaves NO node (it is dropped), which is why the source-side accounting legs below exist.
    """
    try:
        from dace_fortran.external import ExternalCall
    except Exception:  # pragma: no cover - dace_fortran.external always imports
        return set()
    names: Set[str] = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, ExternalCall):
                names.add(node.c_name or node.name)
    return names


# ---------------------------------------------------------------------------------------------
# Source inspection (a heuristic scanner -- see limits in assert_nonvacuous)
# ---------------------------------------------------------------------------------------------
# The ``(?!\s*end\b)`` guard is load-bearing: the loose return-type prefix ``(?:[\w()*,:\s]+\s+)?``
# would otherwise swallow the leading ``end`` of ``end subroutine foo`` and read the END line as a
# *definition* of ``foo`` -- incrementing depth instead of closing it, so a named-END source nests
# forever.  (``endif``/``enddo`` are already rejected: ``end`` is not a word boundary inside them.)
# The lookahead is anchored at ``^`` (not after ``\s*``) on purpose: a trailing ``\s*`` would
# backtrack and let the guard slide past the ``end`` token, defeating it.
_PROC_DEF = re.compile(
    r"^(?!\s*end\b)\s*(?:(?:pure|impure|elemental|recursive|module|non_recursive)\s+)*"
    r"(?:[\w()*,:\s]+\s+)?(subroutine|function)\s+([A-Za-z_]\w*)",
    re.IGNORECASE,
)
# An END statement closes a program unit.  The name after the keyword is OPTIONAL Fortran
# (``END SUBROUTINE foo`` == ``END SUBROUTINE``); before this fix the regex required the
# keyword and name to be ABSENT (``^\s*end\s*(subroutine|function)?\s*$``), so a source that
# writes every procedure's name on its END -- as the WRF/WPS slices do -- never closed a level
# and ``defined_procedures`` returned only the first (outermost) name.  A bare ``END`` still
# matches.  ``END DO`` / ``END IF`` / ``END WHERE`` do NOT: the alternation requires one of the
# program-unit keywords, and those loop/construct ends are not in it.
_PROC_END = re.compile(
    r"^\s*end\s*(?:(?:subroutine|function|module|program|block(?:\s+data)?)\b\s*(?:\w+)?)?\s*$",
    re.IGNORECASE,
)
_CALL_SITE = re.compile(r"\bcall\s+([A-Za-z_]\w*)", re.IGNORECASE)
_COMMENT = re.compile(r"!.*$")
_STRING = re.compile(r"'[^']*'|\"[^\"]*\"")
# Binary arithmetic operators, after comparisons (== /= <= >=) and pointer-assoc (=>) are removed.
_COMPARISONS = re.compile(r"==|/=|<=|>=|=>")
_MATH_INTRINSIC = re.compile(
    r"\b(sqrt|exp|log|log10|log2|sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|"
    r"asinh|acosh|atanh|pow|modulo|mod)\s*\(",
    re.IGNORECASE,
)


def _strip_code(line: str) -> str:
    """Drop the comment and any string literals so operators inside them never count."""
    line = _COMMENT.sub("", line)
    line = _STRING.sub("''", line)
    return line.strip()


def defined_procedures(text: str) -> Set[str]:
    """Lower-cased names of every ``SUBROUTINE`` / ``FUNCTION`` DEFINITION in ``text``.

    A definition is what the merge / inliner can absorb into the SDFG; a name that appears only in
    a ``CALL`` and never here is an unresolved external.
    """
    out: Set[str] = set()
    depth = 0
    for raw in text.splitlines():
        line = _strip_code(raw)
        m_def = _PROC_DEF.match(raw)
        if m_def:
            if depth == 0:
                out.add(m_def.group(2).lower())
            depth += 1
        elif _PROC_END.match(line) and depth > 0:
            depth -= 1
    return out


def source_call_sites(text: str) -> Set[str]:
    """Lower-cased callees of every ``CALL <name>`` in ``text`` (comments/strings stripped)."""
    out: Set[str] = set()
    for raw in text.splitlines():
        line = _strip_code(raw)
        if not line:
            continue
        for m in _CALL_SITE.finditer(line):
            out.add(m.group(1).lower())
    return out


def _procedure_body(text: str, name: str) -> Optional[str]:
    """The body of procedure ``name`` (case-insensitive), or ``None`` if not defined here.

    Heuristic nesting: a definition opens a level, an ``END SUBROUTINE`` / ``END FUNCTION`` /
    bare ``END`` closes the innermost open level.  Good enough to isolate a callee's executable
    statements; interface blocks are not traversed specially (they carry no CALL/arithmetic).
    """
    want = name.lower()
    lines = text.splitlines()
    depth = 0
    start: Optional[int] = None
    for i, raw in enumerate(lines):
        line = _strip_code(raw)
        m_def = _PROC_DEF.match(raw)
        if m_def:
            if depth == 0 and start is None and m_def.group(2).lower() == want:
                start = i + 1
                depth = 1
                continue
            if start is not None:
                depth += 1
            elif depth == 0:
                pass
            continue
        if start is not None and _PROC_END.match(line):
            depth -= 1
            if depth <= 0:
                return "\n".join(lines[start:i])
    return None if start is None else "\n".join(lines[start:])


def arithmetic_lines(body: str) -> list:
    """Executable-looking lines in ``body`` that compute arithmetic.

    A line counts when, after comments/strings are stripped and comparisons / ``=>`` are removed,
    it either (a) contains a binary ``+ - * /`` or ``**``, or (b) calls a math intrinsic.  A bare
    ``a(i) = b(i)`` copy, an ``errflg = 1`` flag set and a ``CALL`` are NOT arithmetic -- which is
    exactly what keeps halo / MPI / I-O / error helpers passing.
    """
    hits = []
    for raw in body.splitlines():
        line = _strip_code(raw)
        if not line:
            continue
        # Skip obvious declarations (a type keyword before the name, no assignment).
        if "=" not in line and "(" not in line:
            continue
        probe = _COMPARISONS.sub(" ", line)
        if "**" in probe:
            hits.append(line)
            continue
        # Remove a leading unary sign on a constant (`x = -1.0`) so it is not read as arithmetic.
        probe = re.sub(r"=\s*[-+]\s*", "= ", probe, count=1)
        if re.search(r"[A-Za-z0-9_)\]]\s*[-+*/]\s*[A-Za-z0-9_(\[]", probe):
            hits.append(line)
            continue
        if _MATH_INTRINSIC.search(line):
            hits.append(line)
    return hits


#: Callees that are legitimately resolved outside the TU.  Prefixes cover Flang runtime, C stdlib,
#: the MPI binding, and the common WRF/WRF-adjacent halo + I/O helper spellings.
_RUNTIME_PREFIXES = ("_fortran", "mpi_", "pmpi_")
_RUNTIME_EXACT = {"malloc", "free", "abort", "exit"}


def _is_runtime_callee(name: str) -> bool:
    n = name.lower()
    return n.startswith(_RUNTIME_PREFIXES) or n in _RUNTIME_EXACT


# ---------------------------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------------------------
def assert_nonvacuous(sdfg,
                      *,
                      source: Optional[str] = None,
                      sources: Sequence[str] = (),
                      externalized: Iterable[str] = (),
                      stubbed: Iterable[str] = (),
                      callee_sources: Optional[Mapping[str, str]] = None,
                      allow: Iterable[str] = (),
                      allow_empty: bool = False,
                      allow_arithmetic: Iterable[str] = ()) -> None:
    """Refuse ``sdfg`` if the build dropped work the source performs.  Loud, never silent.

    Legs (all raise :class:`VacuityError`):

    (A) **ZERO COMPUTE NODES.**  An SDFG with no tasklet / map / library / nested node did no work;
        the empty-but-passing failure mode.  Pass ``allow_empty=True`` for a kernel that genuinely
        only moves data.

    (B) **UNRESOLVED EXTERNAL CALL.**  Every ``CALL <name>`` in ``source`` / ``sources`` must be
        accounted for: defined in the supplied text (so the merge/inliner absorbed it), declared an
        emitted external in ``externalized``, declared a deliberate stub in ``stubbed``, a runtime/
        libm/MPI entry, or in ``allow``.  An unaccounted callee is the silenced ``emit_call`` no-op:
        the call is gone from the SDFG with no error.

    (C) **EXTERNALISED CALL CARRYING ARITHMETIC.**  For every emitted external in ``externalized``,
        if a body for it is available in ``callee_sources`` (the ORIGINAL, pre-stub source -- the
        merge already emptied the copy in ``source``), and that body computes arithmetic, refuse:
        ``keep_external`` stubbed a math helper, so the SDFG is plausible but missing the math.
        ``allow_arithmetic`` names documented exceptions (e.g. a pure iteration counter).

    WHAT THIS CANNOT SEE (stated honestly):

    * It cannot see arithmetic inside a callee it has no source for -- MPI internals, the Flang
      runtime, a separately compiled ``.so``.  Those are out of scope by construction; supply
      ``callee_sources`` for any helper whose arithmetic you must rule out.
    * It cannot see arithmetic reached INDIRECTLY: if an externalised helper calls a second helper
      that does the math, only the first body is scanned.  Recurse yourself, or pass every body.
    * It cannot prove the SDFG's arithmetic EQUALS the source's -- only that work was not dropped
      wholesale.  Numeric equivalence still needs the differential/oracle gate.
    * ``source``/``sources`` is scanned with a regex heuristic: a ``CALL`` produced by cpp macros
      (``#define FOO call bar``), a call in an ``interface``/``contains``-split line, or a
      ``procedure(...)`` pointer invocation using function-call syntax without ``CALL`` can be
      missed.  The body/arithmetic scanner likewise reads executable-looking lines, not a parse
      tree.
    * Leg (A) is a proxy: an SDFG with one trivial tasklet and the rest dropped passes it.  Leg (B)
      is the one that catches a drop that leaves unrelated arithmetic behind.
    """
    ext = {str(n).lower() for n in externalized}
    stub = {str(n).lower() for n in stubbed}
    allow_set = {str(n).lower() for n in allow}
    allow_arith = {str(n).lower() for n in allow_arithmetic}

    # ---- Leg A: zero compute nodes ----------------------------------------------------------
    comp = compute_nodes(sdfg)
    if not comp and not allow_empty:
        raise VacuityError(
            "vacuity: the built SDFG has ZERO compute nodes (no tasklet, map, library or nested "
            "node) -- the body was dropped and the graph is empty-but-passing.  Most common cause: "
            "an unresolved external CALL, which the build silences at builder/emit_library.py "
            "emit_call (`if sig is None: return`).  Evidence: states="
            f"{len(list(sdfg.all_states()))}, arrays={sorted(sdfg.arrays)}, "
            f"args={sorted(sdfg.arglist())}.  Fix: add the callee's HLFIR/source to the build, or "
            "register it with dace_fortran.external.keep_external, or pass allow_empty=True only if "
            "this kernel truly does no work.")

    # ---- Leg B: every source CALL must be accounted for -------------------------------------
    texts = ([source] if source else []) + list(sources)
    if texts:
        merged = "\n".join(texts)
        defined = defined_procedures(merged)
        calls = source_call_sites(merged)
        unaccounted = sorted(
            c for c in calls
            if c not in defined and c not in ext and c not in stub
            and c not in allow_set and not _is_runtime_callee(c)
        )
        if unaccounted:
            raise VacuityError(
                "vacuity: the source CALLs callee(s) the build cannot account for: "
                f"{unaccounted}.  Each is an external with no body in the supplied source and no "
                "keep_external / apply_external_functions registration, so the call is silently "
                "DROPPED at emit_call (builder/emit_library.py: `if sig is None: return`).  If the "
                "callee is arithmetic-bearing the kernel is wrong-but-plausible.  Fix: add its "
                "source to the build, register it as an emitted external, list it in do_not_emit if "
                "the drop is intentional, or name it in allow=.")

    # ---- Leg C: no externalised call may carry arithmetic -----------------------------------
    if callee_sources:
        for name in sorted(ext):
            body = None
            for key, text in callee_sources.items():
                if str(key).lower() == name:
                    body = text
                    break
            if body is None:
                continue  # no source -> cannot see; documented limit
            hits = arithmetic_lines(body)
            if hits and name not in allow_arith:
                head = "\n      ".join(hits[:3])
                raise VacuityError(
                    f"vacuity: externalised callee {name!r} carries arithmetic -- keep_external "
                    "stubbed its body, so this math is NOT in the SDFG (wrong-but-plausible).  "
                    f"Offending line(s):\n      {head}\n"
                    "  Fix: remove it from the external policy so its body inlines, or, if the "
                    "arithmetic is genuinely inert (e.g. an iteration counter), pass "
                    f"allow_arithmetic=[{name!r}] with a recorded justification.")
