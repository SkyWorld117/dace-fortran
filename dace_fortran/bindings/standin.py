# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stand-ins for a procedure the program calls: external subroutines with the procedure's own interface that the
linker puts in its place (``-Wl,--wrap=<symbol>`` with ``--defsym=__wrap_<symbol>=<stand-in>``), and that reach the
original procedure as ``dace_original_<entry>`` (``--defsym=dace_original_<entry>_=__real_<symbol>``).

* :func:`emit_recording_standin` appends the values of module variables to a file on every call, then calls the
  original -- the recording run a specialisation takes its values from
  (:func:`dace_fortran.bindings.specialization.recorded_invariants` reads the file).
* :func:`emit_substitute_entry` calls the binding of a built SDFG and runs the original when the binding declines.

The dummies are declared so that the caller's arguments arrive and leave unchanged (:func:`render_dummies`): a dummy
the caller passes by address (a scalar, an explicit-shape or assumed-size array) as a scalar or an assumed-size
array, one passed by descriptor as an assumed-shape, allocatable or pointer array of the same rank, a VALUE dummy by
value. That needs how each dummy is passed (``OriginalArg.passing``, from the bridge's interface snapshot).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dace_fortran.bindings.fortran_interface import OriginalArg, OriginalInterface


class UnsupportedStandinError(ValueError):
    """An entry dummy a stand-in cannot pass through unchanged."""


@dataclass(frozen=True, slots=True)
class RecordedValue:
    """A module variable a recording stand-in writes: ``module::name`` (``name%component`` for a scalar component of a
    record variable, ``name(k)%component`` for one of an element of an array of records) and its Fortran type (``logical(4)``, ``integer(4)``, ``real(8)``), as
    ``SDFGBuilder.configuration_candidates`` lists them."""

    module: str
    name: str
    fortran_type: str


def _variable(name: str) -> str:
    """The module variable of a recorded scalar: ``domain`` of ``domain%lo``, ``bounds`` of ``bounds(1)%lo``."""
    return re.split(r"[(%]", name, maxsplit=1)[0]


def standin_name(kind: str, entry: str) -> str:
    """``dace_<kind>_<entry>``: the stand-in's Fortran name (``kind`` is ``record``, ``sub`` or ``original``)."""
    return f"dace_{kind}_{entry}"


def _explicit_shape(a: OriginalArg) -> bool:
    """True when every extent of an array passed by address is known (``dimension(sys_size)``): the stand-in then
    declares it explicit-shape, so that it can hand the array on to an assumed-shape dummy (a binding's wrapper),
    which an assumed-size array cannot be."""
    return bool(a.rank) and len(a.shape) == a.rank and all(e.strip() not in ("", "*", ":", "?") for e in a.shape)


def _declaration(a: OriginalArg) -> str:
    attributes = [a.fortran_type]
    shape = ""
    if a.passing == "address":
        if _explicit_shape(a):
            shape = "(" + ", ".join(e.strip() for e in a.shape) + ")"
        else:
            shape = "(*)" if a.rank else ""
    elif a.passing in ("descriptor", "allocatable", "pointer"):
        if a.rank == 0 and a.passing == "descriptor":
            raise UnsupportedStandinError(f"dummy {a.name!r} is a scalar passed by descriptor (polymorphic)")
        if a.passing != "descriptor":
            attributes.append(a.passing)
        shape = "(" + ", ".join([":"] * a.rank) + ")" if a.rank else ""
    elif a.passing == "value":
        attributes.append("value")
    elif a.passing == "character":
        raise UnsupportedStandinError(f"dummy {a.name!r} is a CHARACTER; its length is passed separately")
    else:
        raise UnsupportedStandinError(f"dummy {a.name!r}: how the caller passes it is not known ({a.passing!r})")
    if a.intent and a.passing != "value":
        attributes.append(f"intent({a.intent})")
    if a.optional:
        attributes.append("optional")
    return f"{', '.join(attributes)} :: {a.name}{shape}"


def render_dummies(iface: OriginalInterface) -> tuple[list[str], list[str]]:
    """(``use`` statements, declarations) of the entry's dummies, declared so that the caller's arguments pass through
    a stand-in unchanged."""
    uses = ["use iso_c_binding"]
    for a in iface.args:
        if a.struct_type:
            layout = iface.struct_types.get(a.struct_type)
            module = layout.module if layout is not None else None
            module = module or next((m for m, syms in iface.used_modules.items() if a.struct_type in syms), None)
            if not module:
                raise UnsupportedStandinError(f"dummy {a.name!r}: no module defines type {a.struct_type!r}")
            uses.append(f"use {module}, only: {a.struct_type}")
    # The names explicit extents use (module variables such as ``sys_size``).
    extents = {w.lower() for a in iface.args if a.passing == "address" and _explicit_shape(a) for e in a.shape
               for w in re.findall(r"[A-Za-z_]\w*", e)}
    for module, syms in sorted(iface.used_modules.items()):
        names = sorted(s for s in syms if s.lower() in extents)
        if names:
            uses.append(f"use {module}, only: {', '.join(names)}")
    return list(dict.fromkeys(uses)), [_declaration(a) for a in iface.args]


def _with_arguments(head: str, names: Sequence[str]) -> list[str]:
    """``head(names)`` with one argument per continuation line, so that no line exceeds free-form Fortran's limit of
    132 characters however many arguments there are."""
    if not names:
        return [f"{head}()"]
    return [f"{head}( &", *(f"    {n}, &" for n in names[:-1]), f"    {names[-1]})"]


def _subroutine(
    name: str, iface: OriginalInterface, body: list[str], extra_uses: Sequence[str] = (), local: Sequence[str] = ()
) -> list[str]:
    uses, decls = render_dummies(iface)
    args = [a.name for a in iface.args]
    original = standin_name("original", iface.entry)
    return [
        *_with_arguments(f"subroutine {name}", args),
        *(f"    {u}" for u in [*uses, *extra_uses]),
        "    implicit none",
        *(f"    {d}" for d in decls),
        *(f"    {d}" for d in local),
        "    interface",
        *(f"        {line}" for line in _with_arguments(f"subroutine {original}", args)),
        *(f"            {u}" for u in uses),
        "            implicit none",
        *(f"            {d}" for d in decls),
        f"        end subroutine {original}",
        "    end interface",
        *(f"    {line}" for line in body),
        f"end subroutine {name}",
    ]


def _character_constant(name: str, text: str, chunk: int = 50) -> list[str]:
    """Declaration lines of a named character constant holding ``text``, concatenated from literals short enough
    that no line exceeds free-form Fortran's limit of 132 characters, however long ``text`` is."""
    pieces = [text[i : i + chunk] for i in range(0, len(text), chunk)] or [""]
    literals = ["'" + piece.replace("'", "''") + "'" for piece in pieces]
    lines = [f"character(len=*), parameter :: {name} = {literals[0]}"]
    for literal in literals[1:]:
        lines[-1] += " // &"
        lines.append(f"    {literal}")
    return lines


def _write(value: str, ftype: str, label: str) -> str:
    """One record line, in the format of :func:`dace_fortran.bindings.specialization.recorded_invariants`; ``label``
    names the character constant holding ``module::name``."""
    if ftype.startswith("logical"):
        return f"write(dace_record_unit, '(a, 1x, i0)') {label}, merge(1, 0, {value})"
    if ftype.startswith("real"):
        return f"write(dace_record_unit, '(a, 1x, es25.17e3)') {label}, {value}"
    return f"write(dace_record_unit, '(a, 1x, i0)') {label}, {value}"


def emit_recording_standin(
    iface: OriginalInterface, values: Sequence[RecordedValue], record_path: str | Path, out_path: str | Path
) -> Path:
    """Write ``dace_record_<entry>``: on every call it appends ``module::name value`` for each of ``values`` and an
    empty line to ``record_path``, then calls the original with the arguments unchanged. The variables must be public
    in their modules."""
    uses = [f"use {v.module}, only: dace_value_{i} => {_variable(v.name)}" for i, v in enumerate(values)]
    local = ["integer :: dace_record_unit", *_character_constant("dace_record_path", str(record_path))]
    for i, v in enumerate(values):
        local += _character_constant(f"dace_label_{i}", f"{v.module}::{v.name}")
    body = [
        "open(newunit=dace_record_unit, file=dace_record_path, position='append', action='write')",
        *(_write(f"dace_value_{i}{v.name[len(_variable(v.name)):]}", v.fortran_type, f"dace_label_{i}")
          for i, v in enumerate(values)),
        "write(dace_record_unit, '(a)') ''",
        "close(dace_record_unit)",
        *_with_arguments(f"call {standin_name('original', iface.entry)}", [a.name for a in iface.args]),
    ]
    lines = _subroutine(standin_name("record", iface.entry), iface, body, uses, local)
    out = Path(out_path)
    out.write_text("\n".join(lines) + "\n")
    return out


def _literal(value: int | float, ftype: str) -> str:
    """``value`` as a Fortran literal to compare a variable of type ``ftype`` with, exactly."""
    if ftype.startswith("logical"):
        return ".true." if value else ".false."
    if ftype.startswith("real"):
        if not math.isfinite(value):
            raise ValueError(f"cannot check a baked value against a non-finite one: {value}")
        text = repr(float(value))  # the shortest decimal that reads back as the same double
        return text.replace("e", "d") if "e" in text else f"{text}d0"
    return str(int(value))


def _differs(expr: str, value: int | float, ftype: str) -> str:
    """A Fortran condition true unless ``expr`` (of type ``ftype``) holds ``value``."""
    op = ".neqv." if ftype.startswith("logical") else "/="
    return f"{expr} {op} {_literal(value, ftype)}"


def emit_substitute_entry(
    iface: OriginalInterface,
    binding_module: str,
    wrapper: str,
    out_path: str | Path,
    baked: Sequence[tuple[RecordedValue, int | float]] = (),
) -> Path:
    """Write ``dace_sub_<entry>``: calls the binding's ``wrapper`` and, when it declines the call (``dace_declined``),
    the original instead.

    ``baked``: module variables the SDFG was specialised to before it was built (``SDFGBuilder``'s
    ``specialized_globals``), with their values. Their reads are gone from the SDFG, so the binding cannot check them;
    the entry does, before anything else, and calls the original when any of them differs."""
    args = [a.name for a in iface.args]
    original = standin_name("original", iface.entry)
    uses = [f"use {binding_module}, only: {wrapper}"]
    body: list[str] = []
    if baked:
        uses += [f"use {v.module}, only: dace_baked_{i} => {_variable(v.name)}" for i, (v, _) in enumerate(baked)]
        checks = [
            _differs(f"dace_baked_{i}{v.name[len(_variable(v.name)):]}", value, v.fortran_type)
            for i, (v, value) in enumerate(baked)
        ]
        body += ["declined = .false."]
        body += [f"if ({c}) declined = .true." for c in checks]
        body += ["if (declined) then", *(f"    {ln}" for ln in _with_arguments(f"call {original}", args)), "    return",
                 "end if"]
    body += [
        *_with_arguments(f"call {wrapper}", [*args, "dace_declined=declined"]),
        *_with_arguments(f"if (declined) call {original}", args),
    ]
    lines = _subroutine(standin_name("sub", iface.entry), iface, body, uses, ["logical :: declined"])
    out = Path(out_path)
    out.write_text("\n".join(lines) + "\n")
    return out
