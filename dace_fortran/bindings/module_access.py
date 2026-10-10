# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Access to a module's private variables from a binding, without editing the module's source.

A binding that substitutes a procedure has to hand the SDFG the module state the procedure works on, and that state is
often private. Fortran offers no way in from outside: a submodule needs the parent to declare a separate module
procedure, and compiling against a ``.mod`` whose visibility was widened does not match the original object (compilers
number a module's data blocks by its symbol table). What does work is a copy of the module's own source with an
accessor per variable appended, compiled with the original flags and linked in place of the original object:

* :func:`expose_module_variables` writes that copy: for each named variable a public function
  ``dace_access_<name>()`` returning a pointer to it (null for an unallocated allocatable), and a ``target`` attribute
  statement for those that are neither pointers nor targets yet. The declarations themselves are left as they are, so
  the module's data keeps its layout.
* :func:`changed_global_data_symbols` checks that: every global data symbol of the original object must be in the copy
  with the same size. Other objects of the program, compiled against the original module, reference only those symbols
  and the module's public procedures, whose names do not change; an empty result means the copy can replace the
  original at link time.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Name of the accessor of variable ``v``.
ACCESSOR_PREFIX = "dace_access_"

_TYPE_KEYWORD = re.compile(r"\s*(type|class|real|integer|logical|complex|character|double\s+precision)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ExposedVariable:
    """One module variable made accessible: the accessor's name and what the declaration says about the variable."""

    name: str
    accessor: str
    type_spec: str  # as written in the declaration: ``type(vector_field)``, ``real(wp)``
    rank: int
    allocatable: bool
    pointer: bool


class ModuleAccessError(ValueError):
    """The module or a requested variable cannot be made accessible."""


def _strip_comment(line: str) -> str:
    """``line`` without a trailing ``!`` comment (one outside a character literal)."""
    quote = None
    for i, c in enumerate(line):
        if quote:
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "!":
            return line[:i]
    return line


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """``text`` split at ``sep`` outside parentheses."""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def _statements(lines: Sequence[str]) -> list[str]:
    """Free-form statements of ``lines``: comments stripped, ``&`` continuations joined, ``;`` separated."""
    out, pending = [], ""
    for raw in lines:
        if raw.lstrip().startswith("#"):
            continue  # preprocessor line markers
        line = _strip_comment(raw).rstrip()
        if pending:
            line = line.lstrip()
            line = line.removeprefix("&")
        if line.endswith("&"):
            pending += line[:-1]
            continue
        statement = pending + line
        pending = ""
        out.extend(s for s in _split_top_level(statement, ";") if s)
    return out


def _parse_declaration(statement: str) -> tuple[str, list[str], list[str]] | None:
    """(type spec, attributes, entities) of a type declaration statement written with ``::``, else None."""
    match = _TYPE_KEYWORD.match(statement)
    if not match:
        return None
    end = match.end()
    rest = statement[end:]
    if rest.lstrip().startswith("("):  # kind or type selector
        depth = 0
        for i, c in enumerate(rest):
            depth += c == "("
            depth -= c == ")"
            if depth == 0 and c == ")":
                end += i + 1
                break
    type_spec, rest = statement[:end].strip(), statement[end:]
    if "::" not in rest:
        return None
    attrs, entities = rest.split("::", 1)
    attributes = [a.lower() for a in _split_top_level(attrs) if a]
    return type_spec, attributes, _split_top_level(entities)


def _rank(spec: str) -> int:
    """Rank of an array spec ``(:, :)`` / ``(1:3)``; 0 for none."""
    spec = spec.strip()
    if not spec.startswith("("):
        return 0
    return len(_split_top_level(spec[1 : spec.rindex(")")]))


def _module_span(lines: Sequence[str], module: str) -> tuple[int, int, int | None]:
    """Line indices of ``module <name>``, its ``end module`` and its ``contains`` (None without one)."""
    head = re.compile(rf"^\s*module\s+{re.escape(module)}\s*(!.*)?$", re.IGNORECASE)
    tail = re.compile(rf"^\s*end\s*module(\s+{re.escape(module)})?\s*(!.*)?$", re.IGNORECASE)
    contains = re.compile(r"^\s*contains\s*(!.*)?$", re.IGNORECASE)
    start = next((i for i, line in enumerate(lines) if head.match(line)), None)
    if start is None:
        raise ModuleAccessError(f"no module {module!r} in the source")
    end = next((i for i in range(start + 1, len(lines)) if tail.match(lines[i])), None)
    if end is None:
        raise ModuleAccessError(f"module {module!r} has no end module statement")
    first_contains = next((i for i in range(start + 1, end) if contains.match(lines[i])), None)
    return start, end, first_contains


def _accessor(var: ExposedVariable) -> list[str]:
    shape = "" if var.rank == 0 else "(" + ", ".join([":"] * var.rank) + ")"
    lines = [f"function {var.accessor}() result(p)", f"    {var.type_spec}, pointer :: p{shape}"]
    if var.allocatable:
        lines += [f"    if (allocated({var.name})) then", f"        p => {var.name}", "    else", "        nullify(p)",
                  "    end if"]
    else:
        lines.append(f"    p => {var.name}")
    lines.append(f"end function {var.accessor}")
    return lines


def expose_module_variables(source: str, module: str, names: Sequence[str]) -> tuple[str, dict[str, ExposedVariable]]:
    """``source`` with accessors to the variables ``names`` of ``module`` appended, and what each accessor returns.

    :raises ModuleAccessError: the module is not in ``source``, a name is no variable declared in its specification
        part, is a named constant, or an accessor name is already taken.
    """
    lines = source.splitlines()
    start, end, contains = _module_span(lines, module)
    wanted = {n.lower() for n in names}
    found: dict[str, ExposedVariable] = {}
    targets: set[str] = set()  # declared with the target attribute already
    for statement in _statements(lines[start + 1 : contains if contains is not None else end]):
        parsed = _parse_declaration(statement)
        if parsed is None:
            continue
        type_spec, attributes, entities = parsed
        dimension = next((a[len("dimension") :] for a in attributes if a.startswith("dimension")), "")
        for entity in entities:
            name = re.match(r"\s*([A-Za-z_]\w*)", entity)
            key = name.group(1).lower() if name else ""
            if key not in wanted:
                continue
            if "parameter" in attributes:
                raise ModuleAccessError(f"{module}::{key} is a named constant; it has no storage to point at")
            found[key] = ExposedVariable(
                name=key,
                accessor=ACCESSOR_PREFIX + key,
                type_spec=type_spec,
                rank=_rank(entity[name.end() :].split("=", 1)[0]) or _rank(dimension),
                allocatable="allocatable" in attributes,
                pointer="pointer" in attributes,
            )
            if "target" in attributes:
                targets.add(key)
    missing = sorted(wanted - set(found))
    if missing:
        raise ModuleAccessError(f"no variable {missing} declared in the specification part of module {module!r}")
    variables = [found[n.lower()] for n in names]
    taken = [v.accessor for v in variables if re.search(rf"\b{v.accessor}\b", source, re.IGNORECASE)]
    if taken:
        raise ModuleAccessError(f"module {module!r} already declares {taken}")

    needs_target = [v.name for v in variables if not v.pointer and v.name not in targets]
    spec_lines = ["    ! ----- dace_fortran.bindings.module_access: accessors to otherwise private variables -----"]
    if needs_target:
        spec_lines.append(f"    target :: {', '.join(needs_target)}")
    spec_lines.append(f"    public :: {', '.join(v.accessor for v in variables)}")
    procedures = [f"    {line}" for v in variables for line in _accessor(v)]
    if contains is None:
        out = [*lines[:end], *spec_lines, "contains", *procedures, *lines[end:]]
    else:
        out = [*lines[:contains], *spec_lines, *lines[contains:end], *procedures, *lines[end:]]
    return "\n".join(out) + "\n", {v.name: v for v in variables}


def _global_data_symbols(obj: Path) -> dict[str, int]:
    """Global data symbols an object defines (``nm`` types B, C, D, G, R, S), by name, with their sizes."""
    listing = subprocess.run(["nm", "-S", "--defined-only", str(obj)], capture_output=True, text=True, check=True)
    out = {}
    for line in listing.stdout.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[2] in "BCDGRS":
            out[fields[3]] = int(fields[1], 16)
    return out


def changed_global_data_symbols(original: Path, copy: Path) -> list[str]:
    """The global data symbols of ``original`` that ``copy`` lacks or sizes differently -- empty when the copy can
    replace the original object at link time."""
    before, after = _global_data_symbols(original), _global_data_symbols(copy)
    return sorted(name for name, size in before.items() if after.get(name) != size)
