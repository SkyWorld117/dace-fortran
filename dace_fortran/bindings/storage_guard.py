# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Declining a call in which a member handed over apart from its record shares storage with another array.

The binding hands the SDFG some arrays apart from where the caller keeps them: the pointer and allocatable members
of records, packed into an image (``q_sf`` of ``q(:)%sf``), deep-copied, or split out one array per element.  A
pointer member may point into another array the SDFG also takes -- a module array, a dummy, the member of another
element -- and Fortran allows that, for an association made anywhere before the call.  The SDFG takes the two as
different containers: it reads a stale copy, the copy-back of the image overwrites what it wrote through the other
array, or its code assumes the two do not alias.  None of these is safe, and supporting the association would need
the SDFG built for it.

So the wrapper compares, before it does any work, the address ranges of those members (per element, as the caller
holds them) with those of every other array the SDFG takes, and declines the call through ``dace_declined`` when
two overlap and at least one of them is written -- the caller then runs its own code -- or stops when the caller
passes no flag.  Two arrays the SDFG takes in place (module arrays, dummies) are not compared with each other: their
aliasing is the caller's, as in the original.

The range of an array spans its first element to its last whatever the strides (the address of the first element
plus, per dimension, the extent times the step to the next element), so a strided section counts as everything
between its ends.  The ranges are sorted by their start and swept once: O(n log n) in the number of arrays and
elements of records, independent of their sizes.
"""

from __future__ import annotations

from dace_fortran.bindings.flatten_plan import FlattenEntry, FlattenPlan, strip_index_args
from dace_fortran.bindings.fortran_interface import OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenArgKind, FrozenSignature

_INDENT = "    "

#: Range kinds: a member handed over apart, read or written; an array the SDFG takes in place, read or written.
MEMBER_READ, MEMBER_WRITTEN, IN_PLACE_READ, IN_PLACE_WRITTEN = 1, 2, 3, 4

#: The wrapper locals the guard collects the ranges in.
_SPANS, _COUNT = "dace_spans", "dace_nspans"

#: The arrays whose ranges the guard being built appends, in order: a range's tag is its index here (1-based), and
#: the decline message names the two arrays that overlap.
_NAMES: list[str] = []
_NAME_LEN = 96


def _span_add(expr: str, rank: int, kind: int) -> str:
    """The statement appending the address range of the array ``expr`` (of rank ``rank``, defined) as ``kind``:
    nothing when it has no elements.  ``rank`` 0 takes ``expr`` as one element, scalar or not."""
    _NAMES.append(expr)
    tag = len(_NAMES)
    bytes_ = f"storage_size({expr}) / 8"
    if rank == 0:
        return (
            f"call dace_span_add({_SPANS}, {_COUNT}, {kind}, {bytes_}, [int(loc({expr}), c_intptr_t)], "
            f"[integer(c_intptr_t) ::], {tag})"
        )
    first = [f"lbound({expr}, {d})" for d in range(1, rank + 1)]
    corners = [f"loc({expr}({', '.join(first)}))"]
    for d in range(rank):
        step = list(first)
        step[d] = f"min(lbound({expr}, {d + 1}) + 1, ubound({expr}, {d + 1}))"
        corners.append(f"loc({expr}({', '.join(step)}))")
    extents = ", ".join(f"size({expr}, {d}, kind=c_intptr_t)" for d in range(1, rank + 1))
    return (
        f"if (size({expr}) > 0) call dace_span_add({_SPANS}, {_COUNT}, {kind}, {bytes_}, "
        f"int([{', '.join(corners)}], c_intptr_t), [{extents}], {tag})"
    )


def _guarded(conditions: list[str], lines: list[str]) -> list[str]:
    """``lines`` inside one ``if`` per condition, outermost first: Fortran's ``.and.`` does not short-circuit, and a
    member of an absent record has no definition to query."""
    out: list[str] = []
    for depth, cond in enumerate(conditions):
        out.append("  " * depth + f"if ({cond}) then")
    out += ["  " * len(conditions) + ln for ln in lines]
    out += ["  " * depth + "end if" for depth in reversed(range(len(conditions)))]
    return out


def _module_record_members(frozen: FrozenSignature) -> list[list[str]]:
    """Per image of a pointer or allocatable member of a module array of records (or of a record a dummy holds), the
    lines adding the range of every present element's member."""
    from dace_fortran.bindings.block_builders import (
        _aos_lead,
        _aos_loop_pieces,
        _aos_loops,
        _aos_member_is_static,
        _aos_module_args,
        _present,
    )

    out = []
    for a in _aos_module_args(frozen):
        member_rank = a.rank - _aos_lead(a)
        # A value member lies in its record, a member the kernel allocates holds no data yet.
        if member_rank == 0 or _aos_member_is_static(a) or a.global_alloc_inside:
            continue
        kind = MEMBER_WRITTEN if a.is_written else MEMBER_READ
        if _aos_lead(a) == 0:
            member = f"{a.aos_origin_struct}%{a.aos_member_path}"
            out.append(_guarded([_present(member, a.aos_member_pointer)], [_span_add(member, member_rank, kind)]))
            continue
        pieces = _aos_loop_pieces(a)
        loops = _aos_loops(a, pieces)
        member = pieces.accessor
        inner = _guarded([_present(member, a.aos_member_pointer)], [_span_add(member, member_rank, kind)])
        # The element loops of the copy-in, without its prelude (that sizes the buffer).
        nest = [ln.strip() for ln in loops.do_open]
        depth = len(nest)
        body = ["  " * d + ln for d, ln in enumerate(nest)] + ["  " * depth + ln for ln in inner]
        body += ["  " * d + ln.strip() for d, ln in zip(reversed(range(depth)), loops.do_close, strict=True)]
        out.append(_guarded([loops.present], body))
    return out


def _over_records(entry: FlattenEntry, body: list[str]) -> list[str]:
    """``body`` inside a loop nest over every element of the array of records of an aos_alloc entry, its last
    dimension outermost, bounded by the array itself (the extent symbols are not assigned yet)."""
    from dace_fortran.bindings.loop_copy import aos_outer_rank

    rank = aos_outer_rank(entry.recipe)
    lines = [f"{'  ' * depth}do i{d} = 1, size({entry.outer_expr}, {d})" for depth, d in enumerate(range(rank, 0, -1))]
    lines += ["  " * rank + ln for ln in body]
    lines += [f"{'  ' * depth}end do" for depth in reversed(range(rank))]
    return lines


def _dummy_record_members(
    frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan
) -> tuple[list[list[str]], list[list[str]]]:
    """Per flatten entry of a dummy's records, the lines adding the ranges of its storage: those of a pointer or
    allocatable member as members handed over apart (packed, split or copied, first list), those of a value member,
    which lies in its record, as taken in place (second list)."""
    from dace_fortran.bindings.block_builders import _recipe_presence_guard, live_entries, split_entries
    from dace_fortran.bindings.loop_copy import _aos_alloc_member_at_i, _aos_alloc_presence

    out: list[list[str]] = []
    in_records: list[list[str]] = []
    entries = [*live_entries(frozen, plan), *(entry for entry, _elements in split_entries(frozen, plan))]
    for entry in entries:
        r = entry.recipe
        written = entry.writeback_intent in ("out", "inout")
        if r.aos_alloc:
            member = _aos_alloc_member_at_i(r)
            kind = MEMBER_WRITTEN if written else MEMBER_READ
            span = _span_add(member, len(r.cap_symbols), kind)
            out.append(_over_records(entry, _guarded([_aos_alloc_presence(r, member)], [span])))
            continue
        base = strip_index_args(r.read_exprs[0]) if r.read_exprs else ""
        if not base or "$i" in base:
            continue
        guard = _recipe_presence_guard(iface, r)
        if guard:
            kind = MEMBER_WRITTEN if written else MEMBER_READ
            out.append(_guarded([guard], [_span_add(base, r.rank, kind)]))
        else:
            in_records.append([_span_add(base, r.rank, IN_PLACE_WRITTEN if written else IN_PLACE_READ)])
    return out, in_records


def _in_place_arrays(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> list[list[str]]:
    """The lines adding the ranges of the module arrays and the array dummies the SDFG takes."""
    from dace_fortran.bindings.block_builders import (
        _module_arg_may_be_unallocated,
        _module_value_expr,
        _orphan_module_args,
        _present,
        _synthetic_members,
    )

    out = []
    members = _synthetic_members(plan)
    for oa, _mod, _member in _orphan_module_args(frozen, iface, plan):
        if oa.kind is not FrozenArgKind.ARRAY or oa.rank == 0 or oa.global_alloc_inside or oa.sdfg_name in members:
            continue
        host = _module_value_expr(oa.sdfg_name, members)
        kind = IN_PLACE_WRITTEN if oa.is_written else IN_PLACE_READ
        # A length-1 argument may be a scalar module variable lifted to an array: take it as one element.
        span = _span_add(host, 0 if tuple(oa.shape) == ("1",) else oa.rank, kind)
        conditions = [_present(host, oa.module_origin_pointer)] if _module_arg_may_be_unallocated(oa) else []
        out.append(_guarded(conditions, [span]))
    dummies = {a.name.lower(): a for a in iface.args}
    pointers = {a.name for a in iface.args if f"{a.name}_allocated" in set(frozen.free_symbols)}
    for fa in frozen.args:
        dummy = dummies.get((fa.fortran_name or fa.sdfg_name).lower())
        if dummy is None or dummy.struct_type or fa.rank == 0 or dummy.rank == 0:
            continue
        kind = IN_PLACE_READ if dummy.intent.lower() == "in" else IN_PLACE_WRITTEN
        conditions = [f"present({dummy.name})"] if dummy.optional else []
        if dummy.name in pointers:
            conditions.append(f"associated({dummy.name})")
        out.append(_guarded(conditions, [_span_add(dummy.name, dummy.rank, kind)]))
    return out


def build_storage_guard(
    frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan, declinable: bool
) -> list[str]:
    """Wrapper lines declining the call (``declinable``) or stopping when a member handed over apart from its record
    shares storage with another array of the SDFG; empty when the SDFG takes no such member.  They go before any
    other work of the wrapper body, after a specialisation guard (which clears the flag)."""
    from dace_fortran.bindings.block_builders import DECLINED_FLAG

    _NAMES.clear()
    module_members = _module_record_members(frozen)
    dummy_members, in_records = _dummy_record_members(frozen, iface, plan)
    if not module_members and not dummy_members:
        return []
    groups = module_members + dummy_members + in_records + _in_place_arrays(frozen, iface, plan)
    names = [n.replace("'", "''")[:_NAME_LEN] for n in _NAMES]
    lines = [
        f"{_INDENT}! ----- Storage guard: members handed over apart share no storage with another array -----",
        f"{_INDENT}block",
        f"{_INDENT}  integer(c_intptr_t), allocatable :: {_SPANS}(:, :)",
        f"{_INDENT}  integer :: {_COUNT}, dace_tag_a, dace_tag_b",
        f"{_INDENT}  character(len={_NAME_LEN}), parameter :: dace_span_names({len(names)}) = [character(len={_NAME_LEN}) :: &",
        *(f"{_INDENT}    '{n}'{',' if i + 1 < len(names) else ''} &" for i, n in enumerate(names)),
        f"{_INDENT}    ]",
        f"{_INDENT}  {_COUNT} = 0",
    ]
    lines += [f"{_INDENT}  {ln}" for group in groups for ln in group]
    refuse = [f"if (present({DECLINED_FLAG})) then",
              f"  write(0, '(5a)') '{iface.entry}_dace declines: ', trim(dace_span_names(dace_tag_a)), &",
              "    ' shares storage with ', trim(dace_span_names(dace_tag_b)), &",
              "    ' (a member handed over apart from its record)'",
              f"  {DECLINED_FLAG} = .true.", "  return", "end if"]
    lines += [
        f"{_INDENT}  if (dace_spans_overlap({_SPANS}, {_COUNT}, dace_tag_a, dace_tag_b)) then",
        *(f"{_INDENT}    {ln}" for ln in (refuse if declinable else [])),
        f"{_INDENT}    write(0, '(5a)') '{iface.entry}_dace: ', trim(dace_span_names(dace_tag_a)), &",
        f"{_INDENT}      ' shares storage with ', trim(dace_span_names(dace_tag_b)), &",
        f"{_INDENT}      ' (a member handed over apart from its record)'",
        f"{_INDENT}    error stop '{iface.entry}_dace: a member handed over apart from its record shares storage "
        "with another array'",
        f"{_INDENT}  end if",
        f"{_INDENT}end block",
    ]
    return lines


#: The module procedures the guard calls.
STORAGE_GUARD_HELPERS = """\
  ! Append the address range of an array with elements to spans(:, :n) as (start, end, kind): corners(1) is the
  ! address of its first element, corners(1 + d) that of the next element along dim d (the first again when that
  ! dim has one element), extents its extents.  Whatever the strides, the range spans every element.
  subroutine dace_span_add(spans, n, kind, bytes, corners, extents, tag)
    integer(c_intptr_t), allocatable, intent(inout) :: spans(:, :)
    integer, intent(inout) :: n
    integer, intent(in) :: kind, bytes, tag
    integer(c_intptr_t), intent(in) :: corners(:), extents(:)
    integer(c_intptr_t), allocatable :: grown(:, :)
    integer(c_intptr_t) :: lo, hi, reach
    integer :: d
    lo = corners(1)
    hi = corners(1)
    do d = 1, size(extents)
      reach = (corners(1 + d) - corners(1)) * (extents(d) - 1)
      lo = lo + min(reach, 0_c_intptr_t)
      hi = hi + max(reach, 0_c_intptr_t)
    end do
    if (.not. allocated(spans)) allocate(spans(4, 16))
    if (n == size(spans, 2)) then
      allocate(grown(4, 2 * n))
      grown(:, :n) = spans(:, :n)
      call move_alloc(grown, spans)
    end if
    n = n + 1
    spans(:, n) = [lo, hi + bytes - 1, int(kind, c_intptr_t), int(tag, c_intptr_t)]
  end subroutine dace_span_add

  ! Whether two of the ranges spans(:, :n) overlap where the SDFG would see stale or clobbered data: a member handed
  ! over apart from its record (kind 1, or 2 when written) and any other array, unless neither is written.  Two
  ! arrays the SDFG takes in place (3, or 4 when written) are the caller's.  Sorts the ranges by their start
  ! (heapsort), then sweeps them once, keeping per kind the farthest end so far.
  logical function dace_spans_overlap(spans, n, tag_a, tag_b) result(overlap)
    integer(c_intptr_t), allocatable, intent(inout) :: spans(:, :)
    integer, intent(in) :: n
    integer, intent(out) :: tag_a, tag_b
    ! clash(j, k): a range of kind j must not overlap one of kind k.
    logical, parameter :: clash(4, 4) = reshape([.false., .true., .false., .true., .true., .true., .true., .true., &
                                                 .false., .true., .false., .false., .true., .true., .false., .false.], &
                                                [4, 4])
    integer(c_intptr_t) :: reach(4), held(4)
    integer :: i, k, j, root, child, last, reach_tag(4)
    overlap = .false.
    tag_a = 0
    tag_b = 0
    reach_tag = 0
    if (n < 2) return
    do i = n / 2, 1, -1
      call sift(i, n)
    end do
    do last = n, 2, -1
      held = spans(:, 1)
      spans(:, 1) = spans(:, last)
      spans(:, last) = held
      call sift(1, last - 1)
    end do
    reach = -huge(reach)
    do i = 1, n
      k = int(spans(3, i))
      do j = 1, 4
        if (clash(j, k) .and. spans(1, i) <= reach(j)) then
          overlap = .true.
          tag_a = int(spans(4, i))
          tag_b = reach_tag(j)
          return
        end if
      end do
      if (spans(2, i) > reach(k)) then
        reach(k) = spans(2, i)
        reach_tag(k) = int(spans(4, i))
      end if
    end do
  contains
    subroutine sift(first, bound)
      integer, intent(in) :: first, bound
      root = first
      held = spans(:, root)
      do
        child = 2 * root
        if (child > bound) exit
        if (child < bound) then
          if (spans(1, child + 1) > spans(1, child)) child = child + 1
        end if
        if (spans(1, child) <= held(1)) exit
        spans(:, root) = spans(:, child)
        root = child
      end do
      spans(:, root) = held
    end subroutine sift
  end function dace_spans_overlap"""
