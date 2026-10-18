# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Specialising a binding's SDFG to the values its caller runs it with.

An SDFG whose integer symbols and scalar inputs are replaced by constants (``symbols`` and ``scalars`` of
:func:`dace_fortran.pipelines.optimize`) unrolls the loops and folds the branches they bound, but it is only correct
for calls that pass exactly those values.  The pieces here connect the two ends of that bargain:

* :func:`render_symbol_record`: the binding of the general SDFG appends, once per call, every integer symbol it passes
  and every scalar it reads from a module variable or a scalar dummy to a file, so a representative run shows which
  values the caller holds fixed;
* :func:`recorded_invariants`: the values that stayed the same across all recorded calls;
* :class:`Specialization` and :func:`build_specialization_guard`: the binding of the specialised SDFG computes those
  values as the general binding would, and compares them with the baked ones before it does any work.  A mismatch
  declines the call through the optional ``dace_declined`` flag, so the caller can run its own code instead, or
  stops with the name of the value when the caller passes no flag.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import FORTRAN_C_TYPE, LOGICAL_DTYPES, OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenSignature, ModuleOrigin

_INDENT = "    "

Value = int | float


@dataclass(frozen=True)
class Specialization:
    """The integer symbols and scalar inputs an SDFG was specialised to, by name (logicals as 0 and 1).

    ``frozen`` is the signature of the SDFG before the specialisation, which still takes them: the binding computes
    them from it.
    """

    values: Mapping[str, Value]
    frozen: FrozenSignature


def _symbols(frozen: FrozenSignature) -> tuple[str, ...]:
    """The integer symbols a binding of ``frozen`` passes to the SDFG."""
    from dace_fortran.bindings.block_builders import _USER_COMM_SIZE_SYMBOL_NAME, _USER_COMM_SYMBOL_NAME

    return tuple(s for s in frozen.free_symbols if s not in (_USER_COMM_SYMBOL_NAME, _USER_COMM_SIZE_SYMBOL_NAME))


def _scalar_sources(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> dict[str, tuple[str, str]]:
    """The scalar inputs the SDFG takes from a module variable or a scalar dummy of the caller, by name: the Fortran
    expression of the value and its dtype.  A scalar the kernel writes is its output, never a value to bake."""
    from dace_fortran.bindings.block_builders import _module_value_expr, _synthetic_members, effective_module_sources

    module_sources = effective_module_sources(frozen, iface, plan)
    members = _synthetic_members(plan)
    dummies = {a.name.lower(): a for a in iface.args}
    out: dict[str, tuple[str, str]] = {}
    for a in frozen.args:
        one_element = a.rank == 0 or (a.rank == 1 and tuple(a.shape) == ("1",))
        if a.is_written or not one_element or a.dtype not in FORTRAN_C_TYPE or a.dtype.startswith("complex"):
            continue
        if a.sdfg_name in module_sources:
            out[a.sdfg_name] = (_module_value_expr(a.sdfg_name, members), a.dtype)
            continue
        dummy = dummies.get((a.fortran_name or a.sdfg_name).lower())
        if dummy is not None and dummy.rank == 0 and not dummy.optional:
            out[a.sdfg_name] = (dummy.name, a.dtype)
    return out


def _record_write(name: str, expr: str, dtype: str) -> str:
    """One record line: an integer as itself, a logical as 0 or 1, a real with the 17 decimals that round-trip it."""
    if dtype in LOGICAL_DTYPES:
        return f"write(dace_record_unit, '(a, 1x, i0)') '{name}', merge(1, 0, {expr})"
    if dtype.startswith("float"):
        return f"write(dace_record_unit, '(a, 1x, es25.17e3)') '{name}', {expr}"
    return f"write(dace_record_unit, '(a, 1x, i0)') '{name}', {expr}"


def render_symbol_record(
    frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan, path: str | Path
) -> list[str]:
    """Wrapper lines appending ``<name> <value>`` for every integer symbol and every scalar input from the caller to
    ``path``, and an empty line that ends the call.  They go after the wrapper has assigned all of them, right before
    the SDFG call."""
    lines = [
        f"{_INDENT}! ----- Symbol record: the integer symbols and scalar inputs of this call -----",
        f"{_INDENT}block",
        f"{_INDENT}  integer :: dace_record_unit",
        f"{_INDENT}  open(newunit=dace_record_unit, file='{Path(path)}', position='append', action='write')",
    ]
    lines += [f"{_INDENT}  " + _record_write(s, s, "int64") for s in _symbols(frozen)]
    lines += [f"{_INDENT}  " + _record_write(s, e, t) for s, (e, t) in _scalar_sources(frozen, iface, plan).items()]
    lines += [
        f"{_INDENT}  write(dace_record_unit, '(a)') ''",
        f"{_INDENT}  close(dace_record_unit)",
        f"{_INDENT}end block",
    ]
    return lines


def decline_after_record() -> list[str]:
    """Wrapper lines that follow :func:`render_symbol_record` in a sizing run: decline the call before the SDFG runs."""
    from dace_fortran.bindings.block_builders import DECLINED_FLAG

    return [
        f"{_INDENT}! Sizing run: the caller runs its own code.",
        f"{_INDENT}if (present({DECLINED_FLAG})) {DECLINED_FLAG} = .true.",
        f"{_INDENT}return",
    ]


def _parse(text: str) -> Value:
    try:
        return int(text)
    except ValueError:
        return float(text.replace("D", "E"))


def recorded_aliases(path: str | Path) -> list[tuple[str, int, str]]:
    """The ``(image, element, array)`` aliases every call recorded at ``path`` reported (``<path>.aliases``, written
    by the alias record of a sizing run: the member of ``element`` of the array of records ``image`` is the SDFG image
    of IS the array ``array``); empty without the file."""
    file = Path(f"{path}.aliases")
    if not file.is_file():
        return []
    calls: list[set[tuple[str, int, str]]] = [set()]
    for line in file.read_text().splitlines():
        fields = line.split()
        if not fields:
            if calls[-1]:
                calls.append(set())
            continue
        if len(fields) == 4 and fields[0] == "alias":
            calls[-1].add((fields[1], int(fields[2]), fields[3]))
    calls = [c for c in calls if c]
    if not calls:
        return []
    return sorted(set.intersection(*calls))


def recorded_member_aliases(path: str | Path) -> list[tuple[str, int, str, int]]:
    """The ``(image, element, other image, other element)`` member aliases the calls recorded at ``path`` reported
    (``same`` lines of ``<path>.aliases``: the member of ``element`` of the records ``image`` is the SDFG image of
    is that of ``other element`` of ``other image``), when every call reported the same ones; empty otherwise, and
    without the file. (Folding an alias some call did not have would be as wrong as missing one another call had.)"""
    file = Path(f"{path}.aliases")
    if not file.is_file():
        return []
    calls: list[set[tuple[str, int, str, int]]] = [set()]
    for line in file.read_text().splitlines():
        fields = line.split()
        if not fields:
            calls.append(set())
            continue
        if len(fields) == 5 and fields[0] == "same":
            calls[-1].add((fields[1], int(fields[2]), fields[3], int(fields[4])))
    calls = calls[:-1] if not calls[-1] else calls  # the empty line that ends the last call
    if not calls or any(c != calls[0] for c in calls):
        return []
    return sorted(calls[0])


def recorded_invariants(path: str | Path) -> dict[str, Value]:
    """The values that stayed the same in every call recorded at ``path`` (see :func:`render_symbol_record`)."""
    seen: dict[str, set[Value]] = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            seen.setdefault(fields[0], set()).add(_parse(fields[1]))
    return {name: values.pop() for name, values in seen.items() if len(values) == 1}


def _guard_assigns(
    frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan, names: Iterable[str]
) -> dict[str, list[str]]:
    """Per symbol of ``names`` the wrapper can compute from its inputs alone, before it allocates, copies or aliases
    anything: the lines assigning it, empty for a non-optional dummy that is the value itself."""
    from dace_fortran.bindings.block_builders import _build_symbol_assigns, buffer_local_names

    dummies = {a.name: a for a in iface.args}
    buffers = buffer_local_names(frozen, iface, plan)
    out: dict[str, list[str]] = {}
    for sym in names:
        if sym in dummies:
            if not dummies[sym].optional and dummies[sym].rank == 0:
                out[sym] = []
            continue
        lines = _build_symbol_assigns(replace(frozen, free_symbols=(sym,)), plan, set(dummies), iface)
        text = "\n".join(lines)
        if not lines or "TODO" in text or any(re.search(rf"\b{re.escape(b)}\b", text) for b in buffers):
            continue
        out[sym] = lines
    return out


def specializable(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> set[str]:
    """The names a specialisation guard can check: the free symbols of ``frozen`` the wrapper computes from its inputs
    alone (the extents and lower bounds of the record arrays it packs are only known once it has packed them), and
    the scalar inputs it reads from a module variable or a scalar dummy."""
    return set(_guard_assigns(frozen, iface, plan, _symbols(frozen))) | set(_scalar_sources(frozen, iface, plan))


def guard_module_sources(spec: Specialization, iface: OriginalInterface, plan: FlattenPlan) -> dict[str, ModuleOrigin]:
    """The module variables the guard of ``spec`` reads: those of its values that come from a module."""
    from dace_fortran.bindings.block_builders import effective_module_sources

    return {s: o for s, o in effective_module_sources(spec.frozen, iface, plan).items() if s in spec.values}


def _differs(expr: str, value: Value, dtype: str) -> str:
    """A Fortran condition true unless ``expr`` holds ``value``."""
    if dtype in LOGICAL_DTYPES:
        return f"{expr} .neqv. {'.true.' if value else '.false.'}"
    if dtype.startswith("float"):
        kind = "c_float" if dtype == "float32" else "c_double"
        return f"{expr} /= {float(value)!r}_{kind}"
    return f"{expr} /= {int(value)}"


def build_specialization_guard(spec: Specialization, iface: OriginalInterface, plan: FlattenPlan) -> list[str]:
    """Wrapper lines that compute every value of ``spec`` and decline the call (or stop) unless all of them hold what
    the SDFG was specialised to.  They go first in the wrapper body.  Without values, the wrapper of the general SDFG
    still takes the flag, and never sets it: one caller serves both."""
    from dace_fortran.bindings.block_builders import DECLINED_FLAG

    scalars = _scalar_sources(spec.frozen, iface, plan)
    assigns = _guard_assigns(spec.frozen, iface, plan, [s for s in spec.values if s not in scalars])
    unguardable = sorted(set(spec.values) - set(assigns) - set(scalars))
    if unguardable:
        raise ValueError(
            f"cannot guard the specialisation of {', '.join(unguardable)}: the wrapper computes "
            "them only after it has allocated or copied its buffers (see specializable)"
        )
    non_finite = sorted(s for s, v in spec.values.items() if isinstance(v, float) and not math.isfinite(v))
    if non_finite:
        raise ValueError(f"cannot specialise to a non-finite value: {', '.join(non_finite)}")
    lines = [
        f"{_INDENT}! ----- Specialisation guard: the SDFG holds these values -----",
        f"{_INDENT}if (present({DECLINED_FLAG})) {DECLINED_FLAG} = .false.",
    ]
    if not spec.values:
        return lines
    width = max(len(s) for s in spec.values)
    lines.append(f"{_INDENT}block")
    lines += [f"{_INDENT}  integer(c_int64_t) :: {s}" for s in spec.values if assigns.get(s)]
    lines.append(f"{_INDENT}  character(len={width}) :: dace_mismatch")
    lines.append(f"{_INDENT}  logical, save :: dace_said = .false.")
    lines.append(f"{_INDENT}  dace_mismatch = ''")
    for s in spec.values:
        lines += ["  " + ln for ln in assigns.get(s, ())]
    for s, v in spec.values.items():
        expr, dtype = scalars.get(s, (s, "int64"))
        lines.append(f"{_INDENT}  if ({_differs(expr, v, dtype)}) dace_mismatch = '{s}'")
    lines += [
        f"{_INDENT}  if (dace_mismatch /= '') then",
        f"{_INDENT}    if (present({DECLINED_FLAG})) then",
        f"{_INDENT}      if (.not. dace_said) write(0, '(3a)') '{iface.entry}_dace declines: ', trim(dace_mismatch), &",
        f"{_INDENT}        ' does not hold the value the SDFG was specialised to'",
        f"{_INDENT}      dace_said = .true.",
        f"{_INDENT}      {DECLINED_FLAG} = .true.",
        f"{_INDENT}      return",
        f"{_INDENT}    end if",
        f"{_INDENT}    write(*, '(3a)') '{iface.entry}_dace: ', trim(dace_mismatch), &",
        f"{_INDENT}      ' does not hold the value the SDFG was specialised to'",
        f"{_INDENT}    error stop",
        f"{_INDENT}  end if",
        f"{_INDENT}end block",
    ]
    return lines
