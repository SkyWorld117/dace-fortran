# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Specialising a binding's SDFG to the integer values its caller runs it with.

An SDFG whose integer symbols are replaced by constants (``symbols`` of :func:`dace_fortran.pipelines.optimize`)
unrolls the loops and folds the branches they bound, but it is only correct for calls that pass exactly those
values.  The pieces here connect the two ends of that bargain:

* :func:`render_symbol_record`: the binding of the general SDFG appends every integer symbol it passes to a file,
  once per call, so a representative run shows which values the caller holds fixed;
* :func:`recorded_invariants`: the symbols that kept one value across all recorded calls;
* :class:`Specialization` and :func:`build_specialization_guard`: the binding of the specialised SDFG computes those
  symbols as the general binding would, and compares them with the baked values before it does any work.  A
  mismatch declines the call through the optional ``dace_declined`` flag, so the caller can run its own code
  instead, or stops with the name of the symbol when the caller passes no flag.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenSignature

_INDENT = "    "


@dataclass(frozen=True)
class Specialization:
    """The integer symbols an SDFG was specialised to, by name.

    ``frozen`` is the signature of the SDFG before the specialisation, which still takes the symbols: the binding
    computes them from it.
    """

    values: Mapping[str, int]
    frozen: FrozenSignature


def recorded_symbols(frozen: FrozenSignature) -> tuple[str, ...]:
    """The integer symbols a binding of ``frozen`` passes to the SDFG, which :func:`render_symbol_record` writes."""
    from dace_fortran.bindings.block_builders import _USER_COMM_SIZE_SYMBOL_NAME, _USER_COMM_SYMBOL_NAME

    return tuple(s for s in frozen.free_symbols if s not in (_USER_COMM_SYMBOL_NAME, _USER_COMM_SIZE_SYMBOL_NAME))


def render_symbol_record(frozen: FrozenSignature, path: str | Path) -> list[str]:
    """Wrapper lines appending ``<symbol> <value>`` for every recorded symbol to ``path``, and an empty line that
    ends the call.  They go after the wrapper has assigned all of them, right before the SDFG call."""
    lines = [
        f"{_INDENT}! ----- Symbol record: the integer values of this call -----",
        f"{_INDENT}block",
        f"{_INDENT}  integer :: dace_record_unit",
        f"{_INDENT}  open(newunit=dace_record_unit, file='{Path(path)}', position='append', action='write')",
    ]
    lines += [f"{_INDENT}  write(dace_record_unit, '(a, 1x, i0)') '{s}', {s}" for s in recorded_symbols(frozen)]
    lines += [
        f"{_INDENT}  write(dace_record_unit, '(a)') ''",
        f"{_INDENT}  close(dace_record_unit)",
        f"{_INDENT}end block",
    ]
    return lines


def recorded_invariants(path: str | Path) -> dict[str, int]:
    """The symbols that held one value in every call recorded at ``path`` (see :func:`render_symbol_record`)."""
    seen: dict[str, set[int]] = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            seen.setdefault(fields[0], set()).add(int(fields[1]))
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


def specializable_symbols(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> set[str]:
    """The free symbols of ``frozen`` a specialisation guard can check: the ones the wrapper computes from its inputs
    alone.  The extents and lower bounds of the record arrays it packs are only known once it has packed them."""
    return set(_guard_assigns(frozen, iface, plan, frozen.free_symbols))


def build_specialization_guard(spec: Specialization, iface: OriginalInterface, plan: FlattenPlan) -> list[str]:
    """Wrapper lines that compute every symbol of ``spec`` and decline the call (or stop) unless all of them hold the
    values the SDFG was specialised to.  They go first in the wrapper body.  Without values, the wrapper of the
    general SDFG still takes the flag, and never sets it: one caller serves both."""
    from dace_fortran.bindings.block_builders import DECLINED_FLAG

    assigns = _guard_assigns(spec.frozen, iface, plan, spec.values)
    unguardable = sorted(set(spec.values) - set(assigns))
    if unguardable:
        raise ValueError(
            f"cannot guard the specialisation of {', '.join(unguardable)}: the wrapper computes "
            "them only after it has allocated or copied its buffers (see specializable_symbols)"
        )
    lines = [
        f"{_INDENT}! ----- Specialisation guard: the SDFG holds these symbols at the values below -----",
        f"{_INDENT}if (present({DECLINED_FLAG})) {DECLINED_FLAG} = .false.",
    ]
    if not spec.values:
        return lines
    width = max(len(s) for s in spec.values)
    lines.append(f"{_INDENT}block")
    lines += [f"{_INDENT}  integer(c_int64_t) :: {s}" for s in spec.values if assigns[s]]
    lines.append(f"{_INDENT}  character(len={width}) :: dace_mismatch")
    lines.append(f"{_INDENT}  dace_mismatch = ''")
    for s in spec.values:
        lines += ["  " + ln for ln in assigns[s]]
    lines += [f"{_INDENT}  if ({s} /= {v}) dace_mismatch = '{s}'" for s, v in spec.values.items()]
    lines += [
        f"{_INDENT}  if (dace_mismatch /= '') then",
        f"{_INDENT}    if (present({DECLINED_FLAG})) then",
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
