"""Which module variables code outside one routine's calls can observe, from a build's (preprocessed) Fortran sources.

A module array the routine writes is scratch for it only if nothing that runs outside its calls reads what a call
leaves there.  :func:`outside_readers` decides that from the sources of the whole program, conservatively:

* The procedures that run only inside the routine's calls (its *exclusive* call tree): every procedure the routine
  reaches by name, then repeatedly dropping any that something outside the set references, that a C caller, a type
  binding or an operator can reach, or that is handed around as a procedure argument.  A procedure the routine calls
  that the time stepper calls too is outside.  References are matched by name over-approximately (every procedure of
  that name, every specific of a generic of that name), which only ever moves procedures outside.
* In every procedure outside that set, and in every module's specification part, each name that resolves to the
  module variable (by host or use association, renames and re-exports followed; a local declaration shadows it only
  where no preprocessor condition guards it) is classified.  Harmless: ``allocate``/``deallocate``/``nullify`` (a
  ``source=`` is read), inquiries (``allocated``, ``associated``, ``lbound``, ``ubound``, ``size``, ``shape``,
  ``present``, ...), declarations and accessibility statements, and an assignment that only writes it (a value written
  outside is never read inside, by the in-call condition).  Everything else -- an expression, an argument, a condition,
  I/O, a namelist -- reads it.
* Everywhere, the inside included: a pointer assignment with the variable on either side aliases it with the other
  side (``p%vf(l)%sf => q%vf(l)%sf``); ``c_loc``/``loc`` takes its address; ``equivalence`` and ``common`` share its
  storage.  Aliasing with another module variable puts both in one group, scratch only together; with anything else
  (a dummy, a local, an address) the variable is not scratch.

Not tracked: aliases made through a dummy argument with the TARGET or POINTER attribute (a callee associating a
pointer that outlives the call with its dummy).  All preprocessor branches are scanned (more references, never fewer).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_ID = re.compile(r"[a-z_][a-z0-9_]*")
_PREFIX = r"(?:(?:pure|impure|elemental|recursive|non_recursive|module|simple)\s+|" \
    r"(?:integer|real|logical|complex|character|double\s+precision|type|class)\s*(?:\((?:[^()]|\([^()]*\))*\))?\s*(?:\*\s*\d+\s*)?)*"
_PROC_START = re.compile(rf"^{_PREFIX}(subroutine|function)\s+([a-z_]\w*)\s*(?:\(([^)]*)\))?(.*)$")
_END_UNIT = re.compile(r"^end\s*(subroutine|function|module|program|submodule)?\b(\s+\w+)?$")
_TYPE_DECL = re.compile(
    r"^(integer|real|logical|complex|character|double\s+precision|doubleprecision|type\s*\(|class\s*\(|procedure\s*\()"
)
_ATTR_STMT = re.compile(r"^(dimension|allocatable|pointer|target|save|parameter|external|intrinsic|optional|"
                        r"intent|value|volatile|asynchronous|contiguous|protected|data)\b")
#: Intrinsics that inquire about an object's allocation, association or shape, not its values.
INQUIRIES = ("allocated", "associated", "lbound", "ubound", "size", "shape", "present", "kind", "rank",
             "is_contiguous", "storage_size", "len", "lcobound", "ucobound")
_ADDRESS = ("c_loc", "loc", "c_funloc")
_NAMES_ONLY = re.compile(r"^(public|private|save|target|allocatable|pointer|protected|volatile|asynchronous|"
                         r"contiguous|dimension|import|implicit|use)\b")


@dataclass
class Statement:
    text: str  # lowercase, string literals blanked, no comments, continuations joined
    line: int  # first line in its file
    guarded: bool  # inside a preprocessor conditional


@dataclass
class Use:
    module: str
    only: dict[str, str] | None  # local name -> name in the module; None: every public entity
    renames: dict[str, str] = field(default_factory=dict)

    def local_names(self, exported: set[str]) -> set[str]:
        """The names this use statement makes the entity known in the module by ``exported`` available under."""
        if self.only is not None:
            return {local for local, remote in self.only.items() if remote in exported}
        # Conservative: the original names stay too, as if no rename hid them.
        return set(exported) | {local for local, remote in self.renames.items() if remote in exported}


@dataclass(eq=False)
class Scope:
    """A module or a procedure (a program unit counts as a procedure)."""

    name: str
    kind: str  # module, subroutine, function, program
    file: Path
    line: int
    module: "Scope | None" = None  # the module a procedure is in
    host: "Scope | None" = None  # the enclosing scope (module or host procedure); None for a module
    dummies: set[str] = field(default_factory=set)
    declared: set[str] = field(default_factory=set)  # unguarded local declarations (and dummies, result)
    guarded_declared: set[str] = field(default_factory=set)
    uses: list[Use] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)
    bind_c: bool = False
    procedures: list["Scope"] = field(default_factory=list)  # contained procedures

    @property
    def qualified(self) -> str:
        chain, s = [], self
        while s is not None:
            chain.append(s.name)
            s = s.host
        return "::".join(reversed(chain))


def _logical_lines(text: str) -> list[Statement]:
    """The statements of a free-form source: comments dropped, string literals blanked, continuations joined, split at
    semicolons, lowercased; preprocessor lines dropped (each statement knows whether a conditional guards it)."""
    out: list[Statement] = []
    depth = 0
    pending, start = "", 0
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            word = stripped[1:].strip().split(maxsplit=1)[0] if len(stripped) > 1 else ""
            if word in ("if", "ifdef", "ifndef"):
                depth += 1
            elif word == "endif":
                depth = max(depth - 1, 0)
            continue
        # Blank strings and drop the comment, character by character (quotes double inside a literal).
        chars, quote, i = [], "", 0
        line = raw
        while i < len(line):
            c = line[i]
            if quote:
                if c == quote:
                    if i + 1 < len(line) and line[i + 1] == quote:
                        i += 2
                        continue
                    quote = ""
                    chars.append(c)
                i += 1
                continue
            if c in "'\"":
                quote = c
                chars.append(c)
            elif c == "!":
                break
            else:
                chars.append(c)
            i += 1
        code = "".join(chars).rstrip()
        if quote:
            # A literal continued on the next line: close it here (its contents are blanked anyway).
            code += quote
        if not pending:
            start = number
            guarded_at_start = depth > 0
        body = code.strip()
        if pending and body.startswith("&"):
            body = body[1:]
        if body.endswith("&"):
            pending += body[:-1] + " "
            continue
        full = (pending + body).strip()
        pending = ""
        if not full:
            continue
        for part in full.split(";"):
            part = part.strip()
            if part:
                out.append(Statement(re.sub(r"\s+", " ", part.lower()), start, guarded_at_start))
    return out


def _split_top(text: str, sep: str = ",") -> list[str]:
    parts, depth, cur = [], 0, []
    for c in text:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def _matching_paren(text: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def _parse_use(text: str) -> Use | None:
    m = re.match(r"^use\b\s*(?:,\s*(?:intrinsic|non_intrinsic)\s*)?(?:::)?\s*([a-z_]\w*)\s*(?:,\s*(.*))?$", text)
    if not m:
        return None
    module, rest = m.group(1), m.group(2)
    if rest is None:
        return Use(module, None)
    only = re.match(r"^only\s*:\s*(.*)$", rest)
    pairs = {}
    for item in _split_top(only.group(1) if only else rest):
        if not item:
            continue
        if "=>" in item:
            local, remote = (s.strip() for s in item.split("=>", 1))
        else:
            local = remote = item.strip()
        if re.fullmatch(r"[a-z_]\w*", local) and re.fullmatch(r"[a-z_]\w*", remote):
            pairs[local] = remote
    return Use(module, pairs) if only else Use(module, None, pairs)


def _declared_entities(text: str) -> set[str] | None:
    """The names a type declaration statement declares, or None if ``text`` is not one."""
    if not _TYPE_DECL.match(text):
        return None
    if re.match(r"^(type|class)\s*\(", text) is None and re.match(r"^\w+(\s+precision)?\s*(\(|\*|::|\s|,)", text) is None:
        return None
    if "::" in text:
        entities = text.split("::", 1)[1]
    else:
        m = re.match(r"^(?:double\s+precision|\w+)\s*(?:\((?:[^()]|\([^()]*\))*\))?\s*(?:\*\s*\d+)?\s*(.*)$", text)
        entities = m.group(1) if m else ""
    names = set()
    for item in _split_top(entities):
        m = re.match(r"([a-z_]\w*)", item)
        if m:
            names.add(m.group(1))
    return names


def _attr_entities(text: str) -> set[str]:
    """Names an attribute statement (``allocatable :: a, b``, ``dimension a(3)``) gives an attribute: declarations."""
    rest = text.split("::", 1)[1] if "::" in text else re.sub(r"^\w+\s*(\([^)]*\))?", "", text, count=1)
    return {m.group(1) for item in _split_top(rest) if (m := re.match(r"([a-z_]\w*)", item))}


@dataclass
class Program:
    """A whole program's procedures and modules."""

    modules: dict[str, Scope] = field(default_factory=dict)
    procedures: list[Scope] = field(default_factory=list)  # every procedure, internal ones included
    generics: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # generic -> specific names
    bound: set[str] = field(default_factory=set)  # specifics reachable by type binding or operator: unknown callers
    bindings: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # binding name -> specifics


def parse_sources(files: Iterable[Path]) -> Program:
    prog = Program()
    for path in files:
        _parse_file(Path(path), prog)
    return prog


def source_files(dirs: Iterable[Path]) -> list[Path]:
    """The ``*.f90`` files of ``dirs``; a module defined in several of them is taken from the first directory, as
    module_copies.module_files does, and a file without modules (a program) by its name."""
    seen_modules: set[str] = set()
    seen_names: set[str] = set()
    out = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.f90")):
            text = p.read_text(errors="replace")
            mods = {m.lower() for m in re.findall(r"^\s*module\s+(?!procedure\b)([a-z_]\w*)\s*$", text, re.I | re.M)}
            if mods:
                if mods <= seen_modules:
                    continue
                seen_modules |= mods
            elif p.name in seen_names:
                continue
            seen_names.add(p.name)
            out.append(p)
    return out


def _parse_file(path: Path, prog: Program) -> None:
    stack: list[Scope] = []  # open modules / procedures
    in_contains: dict[int, bool] = {}
    interface_depth = 0  # inside interface blocks (bodies there declare, they define nothing)
    interface_name: list[str | None] = []
    type_depth = 0
    blocks: dict[int, int] = {}  # open BLOCK constructs per scope
    for st in _logical_lines(path.read_text(errors="replace")):
        s = st.text
        if type_depth:
            if re.match(r"^end\s*type\b", s):
                type_depth -= 1
                continue
            m = re.match(r"^(procedure|generic|final)\b[^:]*::\s*(.*)$", s)
            if m:
                for item in _split_top(m.group(2)):
                    if "=>" in item:
                        binding, impls = (x.strip() for x in item.split("=>", 1))
                    else:
                        binding = impls = item.strip()
                    for impl in _split_top(impls):
                        prog.bindings[binding].add(impl)
                        prog.bound.add(impl)
            continue
        if interface_depth:
            if re.match(r"^end\s*interface\b", s):
                interface_depth -= 1
                interface_name.pop()
                continue
            if re.match(r"^(abstract\s+)?interface\b", s):
                interface_depth += 1
                interface_name.append(None)
                continue
            m = re.match(r"^(?:module\s+)?procedure\s*(?:::)?\s*(.*)$", s)
            if m:
                names = [n for n in _split_top(m.group(1)) if re.fullmatch(r"[a-z_]\w*", n)]
                generic = interface_name[-1]
                for n in names:
                    if generic is None or generic.startswith(("operator", "assignment")):
                        prog.bound.add(n)
                    else:
                        prog.generics[generic].add(n)
            continue
        m = re.match(r"^(abstract\s+)?interface\b\s*(.*)$", s)
        if m:
            interface_depth = 1
            spec = m.group(2).strip().replace(" ", "")
            interface_name = [spec or None]
            if stack:
                stack[-1].statements.append(st)
            continue
        if (re.match(r"^type\s*(,[^:]*)?(::)?\s*[a-z_]\w*\s*(\(.*\))?$", s)
                and not re.match(r"^type\s*\(|^type\s+is\b", s)):
            type_depth = 1
            continue
        m = re.match(r"^module\s+(?!procedure\b|subroutine\b|function\b)([a-z_]\w*)$", s)
        if m:
            mod = Scope(m.group(1), "module", path, st.line)
            prog.modules[mod.name] = mod
            stack = [mod]
            continue
        m = re.match(r"^program\s+([a-z_]\w*)$", s)
        if m:
            unit = Scope(m.group(1), "program", path, st.line)
            prog.procedures.append(unit)
            stack = [unit]
            continue
        m = _PROC_START.match(s)
        if m and not s.startswith(("end", "call")):
            kind, name, args, rest = m.group(1), m.group(2), m.group(3) or "", m.group(4) or ""
            host = stack[-1] if stack else None
            proc = Scope(name, kind, path, st.line, host=host)
            proc.module = host if host is not None and host.kind == "module" else (host.module if host else None)
            proc.dummies = {a.strip() for a in args.split(",") if a.strip() and a.strip() != "*"}
            proc.declared |= proc.dummies
            if kind == "function":
                result = re.search(r"\bresult\s*\(\s*([a-z_]\w*)\s*\)", rest)
                proc.declared.add(result.group(1) if result else name)
            proc.bind_c = bool(re.search(r"\bbind\s*\(\s*c\b", rest))
            if host is not None:
                host.procedures.append(proc)
            prog.procedures.append(proc)
            stack.append(proc)
            continue
        if _END_UNIT.match(s) and not re.match(r"^end\s*(if|do|select|block|where|forall|associate|type|interface|"
                                               r"critical|enum|team)\b", s):
            if stack:
                stack.pop()
            continue
        if not stack:
            continue
        scope = stack[-1]
        if s == "contains":
            continue
        if re.match(r"^([a-z_]\w*\s*:\s*)?block$", s):
            blocks[id(scope)] = blocks.get(id(scope), 0) + 1
        elif re.match(r"^end\s*block\b", s):
            blocks[id(scope)] = max(blocks.get(id(scope), 0) - 1, 0)
        use = _parse_use(s)
        if use is not None:
            scope.uses.append(use)
            scope.statements.append(st)
            continue
        names = _declared_entities(s)
        if names is None and _ATTR_STMT.match(s) and not re.match(r"^data\b", s):
            names = _attr_entities(s)
        if names:
            # Under a preprocessor condition or in a BLOCK construct a declaration may not hold where a reference is:
            # it shadows nothing (more references found, never fewer).
            local = st.guarded or blocks.get(id(scope), 0) > 0
            (scope.guarded_declared if local else scope.declared).update(names)
        scope.statements.append(st)


# --- name resolution ---------------------------------------------------------------------------------------------------


class Resolver:
    """Which local names of a scope denote a module variable."""

    def __init__(self, prog: Program):
        self.prog = prog
        self._exported: dict[tuple[str, str, str], set[str]] = {}

    def exported(self, module: str, key: tuple[str, str], seen: frozenset = frozenset()) -> set[str]:
        """Names under which ``module`` makes the variable ``key`` (module, name) known to a user of it: its own
        variable, or one it uses (re-exports, private or not: conservative)."""
        memo = (module, *key)
        if memo in self._exported:
            return self._exported[memo]
        if module in seen:
            return set()
        names: set[str] = set()
        if module == key[0]:
            names.add(key[1])
        mod = self.prog.modules.get(module)
        if mod is not None:
            for use in mod.uses:
                names |= use.local_names(self.exported(use.module, key, seen | {module}))
        self._exported[memo] = names
        return names

    def names_in(self, scope: Scope, key: tuple[str, str]) -> set[str]:
        """The names that denote ``key`` in ``scope`` (its own statements)."""
        candidates: set[str] = set()
        s: Scope | None = scope
        while s is not None:
            if s.kind == "module" and s.name == key[0]:
                candidates.add(key[1])
            for use in s.uses:
                candidates |= use.local_names(self.exported(use.module, key))
            s = s.host
        found = set()
        for name in candidates:
            if self._denotes(scope, name, key):
                found.add(name)
        return found

    def _denotes(self, scope: Scope, name: str, key: tuple[str, str]) -> bool:
        s: Scope | None = scope
        while s is not None:
            if s.kind == "module" and s.name == key[0] and name == key[1]:
                return True
            for use in s.uses:
                if name in use.local_names(self.exported(use.module, key)):
                    return True
            if name in s.declared:
                return False  # a local (unguarded) declaration: another entity
            s = s.host
        return False


# --- the exclusive call tree -------------------------------------------------------------------------------------------


def _tokens(text: str) -> list[tuple[int, str]]:
    """Identifier tokens of a statement with their positions, components (after ``%``) excluded."""
    out = []
    for m in _ID.finditer(text):
        before = text[: m.start()].rstrip()
        if before.endswith("%"):
            continue
        if m.start() > 0 and (text[m.start() - 1].isalnum() or text[m.start() - 1] in "_."):
            continue  # inside a literal like 1.0_wp or .and.
        out.append((m.start(), m.group(0)))
    return out


def exclusive_procedures(prog: Program, routine: Scope, outside: Iterable[str] = ()) -> set[int]:
    """ids of the procedures that only ever run inside calls of ``routine`` (``routine`` included); the procedures
    named in ``outside`` (calls a build drops or keeps external) never count as inside."""
    by_name: dict[str, list[Scope]] = defaultdict(list)
    for p in prog.procedures:
        by_name[p.name].append(p)
    forced_out = {n.lower() for n in outside}
    callees: dict[int, set[int]] = {}
    address_taken: set[str] = set()
    for p in prog.procedures:
        targets: set[int] = set()
        for st in p.statements:
            s = st.text
            plain = not _NAMES_ONLY.match(s) and not re.match(r"^(module\s+)?procedure\b", s)
            for pos, tok in _tokens(s):
                names = {tok} | prog.generics.get(tok, set())
                for n in names:
                    for q in by_name.get(n, ()):
                        targets.add(id(q))
                if tok in by_name and plain:
                    after = s[pos + len(tok):].lstrip()
                    before = s[:pos].rstrip()
                    if not after.startswith("(") and not before.endswith("call"):
                        address_taken.add(tok)
            for pos in [m.end() for m in re.finditer(r"%\s*", s)]:
                m = _ID.match(s, pos)
                if m and m.group(0) in prog.bindings:
                    for n in prog.bindings[m.group(0)]:
                        for q in by_name.get(n, ()):
                            targets.add(id(q))
        callees[id(p)] = targets
    callers: dict[int, set[int]] = defaultdict(set)
    for p, ts in callees.items():
        for t in ts:
            callers[t].add(p)
    procs = {id(p): p for p in prog.procedures}
    inside = {id(routine)}
    frontier = [id(routine)]
    while frontier:
        for t in callees[frontier.pop()]:
            if t not in inside:
                inside.add(t)
                frontier.append(t)
    changed = True
    while changed:
        changed = False
        for pid in list(inside):
            if pid == id(routine):
                continue
            p = procs[pid]
            if (p.kind == "program" or p.bind_c or p.name in prog.bound or p.name in address_taken
                    or p.name in forced_out or not callers[pid] <= inside):
                inside.discard(pid)
                changed = True
    return inside


# --- classifying references --------------------------------------------------------------------------------------------


def _masked(text: str, functions: Iterable[str]) -> str:
    """``text`` with the argument lists of calls of ``functions`` blanked."""
    out = list(text)
    for m in re.finditer(r"\b(" + "|".join(functions) + r")\s*\(", text):
        if m.start() > 0 and text[m.start() - 1] == "%":
            continue
        close = _matching_paren(text, m.end() - 1)
        for i in range(m.end(), close):
            out[i] = " "
    return "".join(out)


def _top_level(text: str, op: str) -> int:
    """Index of the first ``op`` outside parentheses, or -1."""
    depth = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif depth == 0 and text.startswith(op, i):
            if op == "=" and (text[i + 1 : i + 2] in ("=", ">") or text[i - 1 : i] in ("=", "<", ">", "/")):
                i += 1
                continue
            return i
        i += 1
    return -1


def _base(expr: str) -> str | None:
    m = re.match(r"\s*([a-z_]\w*)", expr)
    return m.group(1) if m else None


@dataclass
class Finding:
    kind: str  # read, alias, address
    scope: Scope
    statement: Statement
    other: str = ""  # alias: the other side's name

    def where(self) -> str:
        return f"{self.scope.qualified} ({self.scope.file.name}:{self.statement.line})"


def classify(statement: str, names: set[str]) -> list[tuple[str, str]]:
    """How ``statement`` uses the names ``names`` (one variable): a list of (kind, other) with kind ``read``, ``alias``
    (a pointer assignment; other = the base name on the other side), ``address``, ``shared`` or ``write`` (an
    assignment to it); empty when every use is harmless (an allocation, an inquiry, a declaration)."""
    s = statement
    found = [t for _, t in _tokens(s) if t in names]
    if not found:
        return []
    if _NAMES_ONLY.match(s) or _declared_entities(s) is not None and not _masked_has(s, names, decl=True):
        return []
    if re.match(r"^(equivalence|common)\b", s):
        return [("shared", "")]
    out = []
    for m in re.finditer(r"\b(" + "|".join(_ADDRESS) + r")\s*\(", s):
        close = _matching_paren(s, m.end() - 1)
        if any(t in names for _, t in _tokens(s[m.end():close])):
            out.append(("address", ""))
    s_q = _masked(s, INQUIRIES)
    if re.match(r"^(allocate|deallocate|nullify)\s*\(", s_q):
        close = _matching_paren(s_q, s_q.index("("))
        args = _split_top(s_q[s_q.index("(") + 1 : close])
        for a in args:
            if re.match(r"^source\s*=", a) and any(t in names for _, t in _tokens(a)):
                out.append(("read", ""))
        return out
    # A one-line IF: its condition reads, the statement it guards is classified on its own.
    m = re.match(r"^if\s*\(", s_q)
    if m:
        close = _matching_paren(s_q, m.end() - 1)
        rest = s_q[close + 1 :].strip()
        if not rest.startswith("then") and rest:
            if any(t in names for _, t in _tokens(s_q[: close + 1])):
                out.append(("read", ""))
            return out + classify(rest, names)
    arrow = _top_level(s_q, "=>")
    if arrow > 0 and re.match(r"^[a-z_]\w*\s*(%|\(|=>)", s_q) and not re.match(r"^(associate|use|procedure)\b", s_q):
        lhs, rhs = s_q[:arrow], s_q[arrow + 2 :]
        lb, rb = _base(lhs), _base(rhs)
        if lb in names:
            out.append(("alias", rb or ""))
        if rb in names:
            out.append(("alias", lb or ""))
        inner = [t for pos, t in _tokens(lhs) if t in names and pos != re.search(r"[a-z_]", lhs).start()]
        inner += [t for pos, t in _tokens(rhs) if t in names and pos != re.search(r"[a-z_]", rhs).start()]
        if inner:
            out.append(("read", ""))
        return out
    eq = _top_level(s_q, "=")
    if eq > 0 and re.match(r"^[a-z_]\w*\s*(%|\(|=)", s_q) and not re.match(
            r"^(do|if|else|select|case|where|forall|call|print|write|read|return|stop|associate|block)\b", s_q):
        lhs, rhs = s_q[:eq], s_q[eq + 1 :]
        lhs_base = re.search(r"[a-z_]", lhs).start()
        if any(t in names for _, t in _tokens(rhs)) or any(
                t in names for pos, t in _tokens(lhs) if pos != lhs_base):
            out.append(("read", ""))
        if _base(lhs) in names:
            out.append(("write", ""))
        return out
    if any(t in names for _, t in _tokens(s_q)):
        out.append(("read", ""))
    return out


def _masked_has(s: str, names: set[str], decl: bool) -> bool:
    """A declaration that names the variable in a specification expression (outside inquiries) reads it."""
    entities = _declared_entities(s) or set()
    head = s.split("::", 1)[0] if "::" in s else s
    head = _masked(head, INQUIRIES)
    if any(t in names for _, t in _tokens(head)):
        return True
    tail = s.split("::", 1)[1] if "::" in s else ""
    for item in _split_top(_masked(tail, INQUIRIES)):
        toks = _tokens(item)
        if any(t in names for t in [t for _, t in toks[1:]]):
            return True
    return False


@dataclass
class Verdict:
    key: tuple[str, str]
    reasons: list[str] = field(default_factory=list)
    aliases: set[tuple[str, str]] = field(default_factory=set)
    #: where code outside the routine's calls assigns to it (harmless for scratch: a value written outside is never
    #: read inside, by the in-call condition; not for state the routine keeps between its calls)
    writes: list[str] = field(default_factory=list)

    @property
    def scratch(self) -> bool:
        return not self.reasons


def escaping_procedures(prog: Program) -> set[str]:
    """Names of the procedures that may leave a pointer to an argument behind: one associating a pointer with a
    dummy (``p => dummy``) or taking its address, or passing a dummy on to such a procedure (by name, over-approx)."""
    escaping: set[str] = set()
    for p in prog.procedures:
        for st in p.statements:
            s = st.text
            arrow = _top_level(s, "=>")
            if arrow > 0 and _base(s[arrow + 2:]) in p.dummies and not re.match(r"^(associate|use|procedure|select)\b", s):
                escaping.add(p.name)
            for m in re.finditer(r"\b(" + "|".join(_ADDRESS) + r")\s*\(", s):
                close = _matching_paren(s, m.end() - 1)
                if any(t in p.dummies for _, t in _tokens(s[m.end():close])):
                    escaping.add(p.name)
    changed = True
    while changed:
        changed = False
        for p in prog.procedures:
            if p.name in escaping or not p.dummies:
                continue
            for st in p.statements:
                toks = {t for _, t in _tokens(st.text)}
                if toks & p.dummies and toks & (escaping - {p.name}):
                    escaping.add(p.name)
                    changed = True
                    break
    return escaping


def outside_readers(prog: Program, routine: tuple[str, str], keys: Iterable[tuple[str, str]],
                    outside: Iterable[str] = ()) -> dict[tuple[str, str], Verdict]:
    """For each module variable ``key`` (module, name): whether code outside the calls of ``routine`` (module,
    procedure) can observe it (:class:`Verdict` with the reasons it can; aliases with other variables in ``aliases``,
    see :func:`close_alias_groups`). ``outside``: procedures counted outside whatever calls them (stubs, externals)."""
    target = [p for p in prog.procedures if p.name == routine[1] and p.module is not None and p.module.name == routine[0]]
    if not target:
        raise ValueError(f"no procedure {routine[0]}::{routine[1]} in the sources")
    inside = exclusive_procedures(prog, target[0], outside)
    escaping = escaping_procedures(prog)
    res = Resolver(prog)
    verdicts = {}
    scopes: list[Scope] = list(prog.modules.values()) + prog.procedures
    for key in keys:
        key = (key[0].lower(), key[1].lower())
        v = Verdict(key)
        if key[0] not in prog.modules:
            v.reasons.append(f"module {key[0]} not in the sources")
            verdicts[key] = v
            continue
        for scope in scopes:
            names = res.names_in(scope, key)
            if not names:
                continue
            is_inside = id(scope) in inside
            for st in scope.statements:
                toks = [t for _, t in _tokens(st.text)]
                passed = sorted(set(toks) & escaping) if set(toks) & names else []
                if passed:
                    v.reasons.append(f"passed to {passed[0]}, which may keep a pointer to its argument, in "
                                     f"{scope.qualified} ({scope.file.name}:{st.line})")
                for kind, other in classify(st.text, names):
                    if kind == "write":
                        if not is_inside and scope.kind != "module":
                            v.writes.append(f"{scope.qualified} ({scope.file.name}:{st.line})")
                        continue
                    if kind == "read" and (is_inside or scope.kind == "module"):
                        # Inside: the routine's own reads, which the in-call analysis covers. A module's
                        # specification part reads nothing at run time but a namelist or an initializer.
                        if scope.kind == "module" and re.match(r"^namelist\b", st.text):
                            v.reasons.append(f"in namelist I/O of {scope.name} ({scope.file.name}:{st.line})")
                        continue
                    if kind == "read":
                        v.reasons.append(f"read outside the routine's calls in {scope.qualified} "
                                         f"({scope.file.name}:{st.line})")
                    elif kind == "alias":
                        other_key = _module_key(res, scope, other)
                        if other_key is None:
                            v.reasons.append(f"pointer-associated with `{other}` in {scope.qualified} "
                                             f"({scope.file.name}:{st.line})")
                        elif other_key != key:
                            v.aliases.add(other_key)
                    elif kind == "address":
                        v.reasons.append(f"address taken in {scope.qualified} ({scope.file.name}:{st.line})")
                    else:
                        v.reasons.append(f"storage shared (equivalence/common) in {scope.qualified}")
        verdicts[key] = v
    return verdicts


def _module_key(res: Resolver, scope: Scope, name: str) -> tuple[str, str] | None:
    """The module variable ``name`` denotes in ``scope``, if any."""
    if not name:
        return None
    s: Scope | None = scope
    while s is not None:
        if name in s.declared and s.kind != "module":
            return None
        if s.kind == "module" and (name in s.declared or name in s.guarded_declared):
            return (s.name, name)
        for use in s.uses:
            if use.only is not None and name in use.only:
                remote = use.only[name]
            elif use.only is None:
                remote = use.renames.get(name, name)
            else:
                continue
            found = _defining_module(res.prog, use.module, remote)
            if found is not None:
                return found
        s = s.host
    return None


def _defining_module(prog: Program, module: str, name: str, seen: frozenset = frozenset()) -> tuple[str, str] | None:
    mod = prog.modules.get(module)
    if mod is None or module in seen:
        return None
    if name in mod.declared or name in mod.guarded_declared:
        return (module, name)
    for use in mod.uses:
        if use.only is not None:
            if name in use.only:
                return _defining_module(prog, use.module, use.only[name], seen | {module})
            continue
        found = _defining_module(prog, use.module, use.renames.get(name, name), seen | {module})
        if found is not None:
            return found
    return None


def close_alias_groups(verdicts: dict[tuple[str, str], Verdict]) -> None:
    """A variable pointer-associated with others is scratch only if all of them are (a write through one is a write
    to the others): reject every member of a group with a rejected member, or a member that is no candidate."""
    changed = True
    while changed:
        changed = False
        for key, v in verdicts.items():
            if not v.scratch:
                continue
            for other in sorted(v.aliases):
                o = verdicts.get(other)
                if o is None:
                    v.reasons.append(f"pointer-associated with {other[0]}::{other[1]}, which the routine does not use "
                                     "as an array of its own")
                    changed = True
                    break
                if not o.scratch:
                    v.reasons.append(f"pointer-associated with {other[0]}::{other[1]}, which is not scratch")
                    changed = True
                    break
