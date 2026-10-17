"""Which module variables keep, for the rest of a run, the value they have when a routine is first called.

A substitute specialised to values recorded over a routine's first calls is right for a later call only if nothing
changed those values in between.  :class:`StartupAnalysis` tells, from the sources of the whole program and
conservatively, where code may change a module variable once the routine has been called:

* Start-up is the main program up to the first statement that can lead to a call of the routine (up to the outermost
  loop around that statement, if there is one), with the procedures only it calls.  Everything else may run after the
  first call: the rest of the main program and everything it calls (a procedure called in a loop runs again, its
  statements before the call included), the routine's own calls, and every procedure a type binding, C or a procedure
  pointer can reach.  Without exactly one main program in the sources, everything may run after the first call.
* There, a variable changes where it is assigned, allocated, deallocated or nullified, the variable of a DO loop, read
  into (an input item, a specifier such as ``iostat=``, a namelist group it belongs to), or passed as an actual
  argument to a procedure whose dummy is neither INTENT(IN) nor VALUE (every procedure of the name or generic; a
  procedure the sources do not define, such as an MPI routine, changes it).
* Anywhere in the program, a variable may change under another name when it is pointer-associated, its address taken,
  it shares storage (EQUIVALENCE, COMMON), or it is passed to a procedure that may keep a pointer to its argument.

Not followed: arguments of function references (taken as read), ASSOCIATE names, and calls the sources do not show.
The substitute checks every specialised value on every call regardless, and leaves a call with other values to the
original routine; this analysis keeps out of the specialisation the values a later call would find changed.
"""

from __future__ import annotations

import re
from collections import defaultdict

from dace_fortran.substitute.outside_refs import (
    Program,
    Resolver,
    Scope,
    _declared_entities,
    _ID,
    _matching_paren,
    _split_top,
    _tokens,
    classify,
    escaping_procedures,
)

Key = tuple[str, str]

_DO = re.compile(r"^(?:[a-z_]\w*\s*:\s*)?do\b")
_DO_VARIABLE = re.compile(r"^(?:[a-z_]\w*\s*:\s*)?do\s+(?:\d+\s+)?([a-z_]\w*)\s*=")
_END_DO = re.compile(r"^(?:end\s*do|enddo)\b")
_IO = re.compile(r"^(read|write|open|close|inquire|backspace|rewind|endfile|flush|wait)\s*\(")
_CALL = re.compile(r"^call\s+([a-z_]\w*)\s*(?:\((.*)\))?\s*$")
#: Specifiers of an I/O statement other than INQUIRE that only pass values in (the others return one).
_IO_INPUTS = {"unit", "fmt", "file", "nml", "rec", "status", "access", "form", "action", "position", "advance",
              "decimal", "round", "sign", "delim", "pad", "blank", "encoding", "asynchronous", "id", "pos", "recl",
              "err", "end", "eor", "convert"}
#: A variable or a part of one (an element, a section, a component): what an actual argument must be to be changed.
_DESIGNATOR = re.compile(r"^[a-z_]\w*(\s*\(.*\))?(\s*%\s*[a-z_]\w*(\s*\(.*\))?)*$")


def _statement_callees(prog: Program, by_name: dict[str, list[Scope]], text: str) -> set[int]:
    """ids of the procedures a statement may call: by name (generics expanded) and by type binding."""
    out: set[int] = set()
    for _, tok in _tokens(text):
        for name in {tok} | prog.generics.get(tok, set()):
            out.update(id(q) for q in by_name.get(name, ()))
    for m in re.finditer(r"%\s*([a-z_]\w*)", text):
        for name in prog.bindings.get(m.group(1), ()):
            out.update(id(q) for q in by_name.get(name, ()))
    return out


def _address_taken(prog: Program, by_name: dict[str, list[Scope]]) -> set[int]:
    """ids of the procedures named somewhere other than in a call (an actual argument, a procedure pointer target)."""
    out: set[int] = set()
    for p in prog.procedures:
        for st in p.statements:
            s = st.text
            if re.match(r"^(public|private|use|import|external|(module\s+)?procedure|interface|end)\b", s):
                continue
            for pos, tok in _tokens(s):
                if tok in by_name and not s[pos + len(tok):].lstrip().startswith("(") \
                        and not s[:pos].rstrip().endswith("call"):
                    out.update(id(q) for q in by_name[tok])
    return out


def _read_only_dummies(proc: Scope) -> set[str]:
    """The dummies of ``proc`` declared INTENT(IN) or VALUE."""
    out: set[str] = set()
    for st in proc.statements:
        s = st.text
        if not (re.search(r"\bintent\s*\(\s*in\s*\)", s) or re.search(r"(^|,)\s*value\b", s)):
            continue
        entities = _declared_entities(s)
        if entities is None and "::" in s:
            entities = {m.group(0) for item in _split_top(s.split("::", 1)[1]) if (m := _ID.match(item.strip()))}
        out |= (entities or set()) & proc.dummies
    return out


def _namelist_groups(prog: Program) -> dict[str, set[str]]:
    """Every namelist group's members, by group name (over all scopes)."""
    groups: dict[str, set[str]] = defaultdict(set)
    for scope in [*prog.modules.values(), *prog.procedures]:
        for st in scope.statements:
            for m in re.finditer(r"/\s*([a-z_]\w*)\s*/([^/]*)", st.text if st.text.startswith("namelist") else ""):
                groups[m.group(1)] |= {item.strip() for item in m.group(2).split(",") if item.strip()}
    return groups


class StartupAnalysis:
    """Where code that may run after the first call of ``routine`` (module, procedure) may change module variables."""

    def __init__(self, prog: Program, routine: Key):
        self.prog = prog
        self.by_name: dict[str, list[Scope]] = defaultdict(list)
        for p in prog.procedures:
            self.by_name[p.name].append(p)
        target = [p for p in prog.procedures
                  if p.name == routine[1] and p.module is not None and p.module.name == routine[0]]
        if not target:
            raise ValueError(f"no procedure {routine[0]}::{routine[1]} in the sources")
        procs = {id(p): p for p in prog.procedures}
        callees = {id(p): set().union(*(self._callees(st.text) for st in p.statements)) for p in prog.procedures}
        # The procedures that can lead to a call of the routine.
        leads = {id(target[0])}
        grown = True
        while grown:
            grown = False
            for pid, ts in callees.items():
                if pid not in leads and ts & leads:
                    leads.add(pid)
                    grown = True
        roots = {id(target[0])} | _address_taken(prog, self.by_name)
        roots |= {id(p) for p in prog.procedures if p.bind_c or p.name in prog.bound}
        #: per scope (id), the index of its first statement that may run after the routine's first call
        self.first_after: dict[int, int] = {}
        mains = [p for p in prog.procedures if p.kind == "program"]
        self.main = mains[0] if len(mains) == 1 else None
        if self.main is None:
            roots |= set(procs)
        else:
            start = self._startup_end(self.main, leads)
            self.first_after[id(self.main)] = start
            for st in self.main.statements[start:]:
                roots |= self._callees(st.text)
        after = set(roots)
        frontier = list(roots)
        while frontier:
            for t in callees.get(frontier.pop(), ()):
                if t not in after:
                    after.add(t)
                    frontier.append(t)
        for pid in after:
            self.first_after[pid] = 0
        self.after_procedures = {procs[pid].qualified for pid in after if pid in procs}
        self.resolver = Resolver(prog)
        self.escaping = escaping_procedures(prog)
        self.namelists = _namelist_groups(prog)
        self.read_only = {id(p): _read_only_dummies(p) for p in prog.procedures}
        self._cache: dict[Key, list[str]] = {}

    def _callees(self, text: str) -> set[int]:
        return _statement_callees(self.prog, self.by_name, text)

    def _startup_end(self, main: Scope, leads: set[int]) -> int:
        """Index of the first statement of ``main`` that may run after the routine's first call."""
        open_loops: list[int] = []
        for i, st in enumerate(main.statements):
            if _END_DO.match(st.text):
                if open_loops:
                    open_loops.pop()
                continue
            if self._callees(st.text) & leads:
                return open_loops[0] if open_loops else i
            if _DO.match(st.text):
                open_loops.append(i)
        return len(main.statements)

    def changes(self, key: Key) -> list[str]:
        """Where the module variable ``key`` (module, name) may change after the routine's first call; empty when its
        value at that call holds for the rest of the run."""
        key = (key[0].lower(), key[1].lower())
        if key not in self._cache:
            self._cache[key] = self._find_changes(key)
        return self._cache[key]

    def _find_changes(self, key: Key) -> list[str]:
        if key[0] not in self.prog.modules:
            return [f"module {key[0]} not in the sources"]
        out: list[str] = []
        for scope in [*self.prog.modules.values(), *self.prog.procedures]:
            names = self.resolver.names_in(scope, key)
            if not names:
                continue
            first = self.first_after.get(id(scope))
            for i, st in enumerate(scope.statements):
                toks = {t for _, t in _tokens(st.text)}
                if not toks & names and not st.text.startswith("read"):
                    continue
                where = f"{scope.qualified} ({scope.file.name}:{st.line})"
                kinds = [kind for kind, _ in classify(st.text, names)]
                for kind in ("alias", "address", "shared"):
                    if kind in kinds:
                        out.append(f"{kind} in {where}")
                passed = sorted((toks & self.escaping) - {scope.name}) if toks & names else []
                if passed:
                    out.append(f"passed to {passed[0]}, which may keep a pointer to its argument, in {where}")
                if first is None or i < first:
                    continue
                if "write" in kinds:
                    out.append(f"assigned in {where}")
                out += [f"{why} in {where}" for why in self._changed_by(st.text, names, toks)]
        return out

    def _changed_by(self, s: str, names: set[str], toks: set[str]) -> list[str]:
        """How statement ``s`` may change the variable known as ``names``, beyond an assignment to it."""
        out = []
        m = re.match(r"^(allocate|deallocate|nullify)\s*\(", s)
        if m:
            close = _matching_paren(s, m.end() - 1)
            for arg in _split_top(s[m.end():close]):
                head = arg.split("=", 1)
                target = head[1] if len(head) == 2 and re.fullmatch(r"\s*(stat|errmsg)\s*", head[0]) else head[0]
                if _base_name(target) in names:
                    out.append(m.group(1) + "d" if m.group(1) != "nullify" else "nullified")
            return out
        m = _DO_VARIABLE.match(s)
        if m and m.group(1) in names:
            return ["the variable of a DO loop"]
        m = _IO.match(s)
        if m:
            close = _matching_paren(s, m.end() - 1)
            control, items = s[m.end():close], s[close + 1:]
            specs = _split_top(control)
            for spec in specs:
                k, sep, v = (part.strip() for part in spec.partition("="))
                returned = k not in ("unit", "file", "id") if m.group(1) == "inquire" else k not in _IO_INPUTS
                if sep and returned and _DESIGNATOR.match(v) and _base_name(v) in names:
                    out.append(f"set by {k}=")
            if m.group(1) == "write" and specs and "=" not in specs[0] and _DESIGNATOR.match(specs[0].strip()) \
                    and _base_name(specs[0]) in names:
                out.append("written by an internal WRITE")
            if m.group(1) == "read":
                if any(_base_name(item) in names for item in _split_top(items)):
                    out.append("read into")
                groups = [v.strip() for k, _, v in (spec.partition("=") for spec in _split_top(control))
                          if k.strip() == "nml"]
                groups += [spec.strip() for spec in _split_top(control)[1:2] if spec.strip() in self.namelists]
                if any(names & self.namelists.get(g, set()) for g in groups):
                    out.append("read into (namelist)")
            return out
        m = _CALL.match(s)
        if m and toks & names:
            callee, args = m.group(1), _split_top(m.group(2) or "")
            candidates = [q for n in {callee} | self.prog.generics.get(callee, set()) for q in self.by_name.get(n, ())]
            for pos, arg in enumerate(args):
                keyword, _, value = arg.partition("=") if re.match(r"^\s*[a-z_]\w*\s*=(?!=)", arg) else ("", "", arg)
                if not _DESIGNATOR.match(value.strip()) or _base_name(value) not in names:
                    continue
                if not candidates:
                    out.append(f"passed to {callee}, which the sources do not define")
                    continue
                for q in candidates:
                    dummy = keyword.strip() if keyword else (q.dummy_order[pos] if pos < len(q.dummy_order) else None)
                    if dummy is None or dummy not in self.read_only[id(q)]:
                        out.append(f"passed to {q.qualified}, whose dummy {dummy} is not intent(in)")
                        break
        return out


def _base_name(expr: str) -> str | None:
    m = _ID.match(expr.strip())
    return m.group(0) if m else None
