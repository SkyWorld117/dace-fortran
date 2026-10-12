# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Recipe renderers -- the ONLY code that knows how to turn a
``FlattenRecipe`` into Fortran.

Three renderers: ``render_alias_calls`` (zero-copy ``c_f_pointer``
alias), ``render_copy_in_loop`` (allocate + forward do-loop copy),
``render_copy_out_loop`` (reverse do-loop copy + deallocate).  All
return ``List[str]`` of lines pre-indented to wrapper-body level.
"""

from collections.abc import Mapping

from dace_fortran.bindings.flatten_plan import (
    FlattenRecipe,
    strip_index_args,
    substitute_indices,
)
from dace_fortran.bindings.fortran_interface import FORTRAN_C_TYPE


def fortran_scalar_type(dtype: str) -> str:
    """Map a DaCe dtype string to its Fortran iso_c_binding form."""
    return FORTRAN_C_TYPE.get(dtype, "real(c_double)")


def _loop_index_names(rank: int) -> tuple[str, ...]:
    """Loop-index names the wrapper head declares: ``('i1', 'i2', ..., 'iN')``."""
    return tuple(f"i{d + 1}" for d in range(rank))


# ----------------------------------------------------------------------------
# Alias path
# ----------------------------------------------------------------------------


def render_alias_calls(recipe: FlattenRecipe) -> list[str]:
    """Zero-copy alias emission for an ``aliasable=True`` recipe -- one
    ``call c_f_pointer(c_loc(<outer>), <flat>, [<shape>])`` per flat.

    :raises ValueError: recipe is not ``aliasable``.
    """
    if not recipe.aliasable:
        raise ValueError("render_alias_calls called on non-aliasable recipe")
    shape_list = ", ".join(recipe.shape_exprs)
    out: list[str] = []
    for flat, read_expr in zip(recipe.flat_names, recipe.read_exprs, strict=False):
        base = strip_index_args(read_expr)
        # Scalar member: c_f_pointer must NOT get a shape arg -- `[]` is
        # an invalid empty array constructor in Fortran.
        if recipe.rank == 0 or not recipe.shape_exprs:
            out.append(f"    call c_f_pointer(c_loc({base}), {flat})")
        else:
            out.append(f"    call c_f_pointer(c_loc({base}), {flat}, [{shape_list}])")
    return out


# ----------------------------------------------------------------------------
# Forward copy (outer -> flats)
# ----------------------------------------------------------------------------


def render_copy_in_loop(recipe: FlattenRecipe) -> list[str]:
    """Generic forward copy: allocate flats, nested do-loops assign each
    flat from its ``read_expr`` with loop-index placeholders substituted.
    Requires ``aliasable=False``, ``rank >= 1``.

    :raises ValueError: recipe is ``aliasable``.
    """
    if recipe.aliasable:
        raise ValueError("render_copy_in_loop called on aliasable recipe  --  use render_alias_calls")
    out: list[str] = []
    for flat in recipe.flat_names:
        out.append(f"    allocate({flat}({', '.join(recipe.shape_exprs)}))")

    # Loop nest: outermost rank first (column-major).
    idx_names = _loop_index_names(recipe.rank)
    # Substituted indices so placeholders don't leak into the generated comment.
    summary = substitute_indices(recipe.read_exprs[0], idx_names)
    out.append(f"    ! Copy-in: {', '.join(recipe.flat_names)} <- {summary}")
    for d in reversed(range(recipe.rank)):
        indent = " " * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}do {idx_names[d]} = 1, {recipe.shape_exprs[d]}")

    body_indent = " " * (recipe.rank * 2)
    idx_tuple = ", ".join(idx_names)
    for flat, read_expr in zip(recipe.flat_names, recipe.read_exprs, strict=False):
        rhs = substitute_indices(read_expr, idx_names)
        out.append(f"    {body_indent}{flat}({idx_tuple}) = {rhs}")

    # Closing markers, innermost-first.
    for d in range(recipe.rank):
        indent = " " * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}end do")
    return out


# ----------------------------------------------------------------------------
# Reverse copy (flats -> outer) + dealloc
# ----------------------------------------------------------------------------


def render_copy_out_loop(recipe: FlattenRecipe, outer_expr: str) -> list[str]:
    """Inverse of ``render_copy_in_loop``: pack flat buffers back into
    the outer storage at each position, then deallocate.

    ``outer_expr`` is passed separately (same as
    ``FlattenEntry.outer_expr``) so the renderer doesn't reach back up
    to the entry.
    """
    out: list[str] = [f"    ! Copy-out: {outer_expr} <- {', '.join(recipe.flat_names)}"]

    idx_names = _loop_index_names(recipe.rank)
    idx_tuple = ", ".join(idx_names)
    for d in reversed(range(recipe.rank)):
        indent = " " * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}do {idx_names[d]} = 1, {recipe.shape_exprs[d]}")

    body_indent = " " * (recipe.rank * 2)
    # write_expr set: reconstruction recipe, outer_expr(idx) = write_expr.
    # write_expr empty: plain single-flat member, exact inverse of its
    # copy-in -- scatter into read_exprs[0] (encodes index placement,
    # e.g. pts(i)%x), giving pts(i)%x = pts_x(i).
    if recipe.write_expr:
        lhs = f"{outer_expr}({idx_tuple})"
        rhs = substitute_indices(recipe.write_expr, idx_names)
    else:
        if len(recipe.flat_names) != 1 or not recipe.read_exprs:
            raise ValueError("render_copy_out_loop: empty write_expr needs a single flat + read_expr")
        lhs = substitute_indices(recipe.read_exprs[0], idx_names)
        rhs = f"{recipe.flat_names[0]}({idx_tuple})"
    out.append(f"    {body_indent}{lhs} = {rhs}")

    for d in range(recipe.rank):
        indent = " " * ((recipe.rank - 1 - d) * 2)
        out.append(f"    {indent}end do")

    for flat in recipe.flat_names:
        out.append(f"    deallocate({flat})")
    return out


# ----------------------------------------------------------------------------
# AoS + allocatable pack/unpack (Phase 5c-B boundary)
# ----------------------------------------------------------------------------
#
# Padding-to-max contract for an AoS dummy whose elements own
# allocatable/pointer array members.  Full design lives on
# ``FlattenRecipe.aos_alloc`` in ``flatten_plan.py``; this module
# implements the two emitters that read those fields.
#
# Helpers below extract `A($i1, ..., $iR)%w` from recipe.read_exprs[0] by dropping the trailing member subscript
# list -- safe because the bridge always emits aos_alloc recipes as `<outer>($i1, ..., $iR)%<member>($iR+1, ...)`
# for an array of records of rank R.  i1 ... iR are the element indices; the wrapper head declares i1 ... i<rank> for
# any non-aliasable recipe of rank >= 1.


def aos_outer_rank(recipe: FlattenRecipe) -> int:
    """The rank of an aos_alloc recipe's array of records: its dimensions before the member's."""
    return recipe.rank - len(recipe.cap_symbols)


def _element_indices(recipe: FlattenRecipe) -> list[str]:
    """The element indices of an aos_alloc recipe, ``i1 ... iR``."""
    return [f"i{d}" for d in range(1, aos_outer_rank(recipe) + 1)]


def _aos_alloc_member_at_i(recipe: FlattenRecipe) -> str:
    """Extract ``<outer>(i1, ..., iR)%<member>`` (no member subscripts) from an aos_alloc recipe's
    ``read_exprs[0]``, for presence / size() / lbound() queries in the pack-in/pack-out emitters."""
    member = recipe.read_exprs[0].rsplit("(", 1)[0]  # "A($i1, ..., $iR)%w"
    for d in range(aos_outer_rank(recipe), 0, -1):  # $i12 before $i1
        member = member.replace(f"$i{d}", f"i{d}")
    return member


def _aos_alloc_extents(recipe: FlattenRecipe, outer_expr: str) -> list[str]:
    """The extents of an aos_alloc recipe's array of records, one per dimension."""
    rank = aos_outer_rank(recipe)
    if len(recipe.shape_exprs) >= rank:
        return list(recipe.shape_exprs[:rank])
    return [f"size({outer_expr}, {d})" for d in range(1, rank + 1)]


def _over_elements(recipe: FlattenRecipe, outer_expr: str, body: list[str]) -> list[str]:
    """``body`` -- lines indented for one loop level -- inside a loop nest over every element of the array of records,
    its last dimension outermost."""
    extents = _aos_alloc_extents(recipe, outer_expr)
    rank = len(extents)
    lines = [f"    {'  ' * depth}do i{d} = 1, {extents[d - 1]}" for depth, d in enumerate(range(rank, 0, -1))]
    lines += [("  " * (rank - 1)) + line for line in body]
    lines += [f"    {'  ' * depth}end do" for depth in reversed(range(rank))]
    return lines


def _aos_alloc_presence(recipe: FlattenRecipe, member: str) -> str:
    """Presence query for one element's member: ``associated`` for a pointer, else ``allocated``."""
    return f"associated({member})" if recipe.member_pointer else f"allocated({member})"


def aos_bounds_locals(recipe: FlattenRecipe) -> tuple[str, ...]:
    """Logical locals the pack-in of an aos_alloc recipe needs: whether the elements' common lower bounds were taken."""
    return (f"{recipe.flat_names[0]}_lb_taken",)


def aos_pack_extent_locals(recipe: FlattenRecipe) -> tuple[str, ...]:
    """Scalar locals holding one element's member extents for the device pack loops (``<flat>_ext1``, ...)."""
    return tuple(f"{recipe.flat_names[0]}_ext{k}" for k in range(1, len(recipe.cap_symbols) + 1))


def aos_pack_offset_locals(recipe: FlattenRecipe) -> tuple[str, ...]:
    """Scalar locals holding where one element's member starts in the buffer, per member dim: its lower bound minus
    the buffer's (``<flat>_off1``, ...)."""
    return tuple(f"{recipe.flat_names[0]}_off{k}" for k in range(1, len(recipe.cap_symbols) + 1))


def aos_kept_layout_locals(recipe: FlattenRecipe) -> tuple[str, str, str, str]:
    """Names around a device image kept across calls (see :func:`render_aos_alloc_pack_in`): the saved layout it was
    set up for, the current layout, a running index into it, and whether the image is set up anew in this call."""
    flat = recipe.flat_names[0]
    return f"{flat}_layout", f"{flat}_layout_now", f"{flat}_layout_k", f"{flat}_fresh"


def _kept_layout(recipe: FlattenRecipe, outer_expr: str, member_at_i: str, present: str) -> list[str]:
    """Write the current layout of the image into ``<flat>_layout_now``: the extents of the array of records, the caps,
    the common lower bounds, then per element whether its member is present and its bounds per dim. Two calls with the
    same layout fill the same elements of the image, and its padding stays as the setup left it."""
    saved, now, k, _fresh = aos_kept_layout_locals(recipe)
    extents = _aos_alloc_extents(recipe, outer_expr)
    rank = len(recipe.cap_symbols)
    head = [*extents, *recipe.cap_symbols, *recipe.lb_symbols]
    per_element = 1 + 2 * rank
    count = " * ".join(f"({e})" for e in extents) or "1"
    lines = [f"    allocate({now}({len(head)} + {count} * {per_element}))"]
    lines += [f"    {now}({j}) = {value}" for j, value in enumerate(head, start=1)]
    lines.append(f"    {k} = {len(head)}")
    body = [f"      {now}({k} + 1:{k} + {per_element}) = 0", f"      if ({present}) then", f"        {now}({k} + 1) = 1"]
    for d in range(1, rank + 1):
        body.append(f"        {now}({k} + {2 * d}) = lbound({member_at_i}, {d})")
        body.append(f"        {now}({k} + {2 * d + 1}) = ubound({member_at_i}, {d})")
    body += ["      end if", f"      {k} = {k} + {per_element}"]
    return lines + _over_elements(recipe, outer_expr, body)


def _device_pack_loop(recipe: FlattenRecipe, member_at_i: str, assign: str) -> list[str]:
    """An OpenACC loop nest over one present element's member, running ``assign`` per element of it: the buffer
    index of member dim ``k`` is ``i<R+k>`` (``i1 ... iR`` are the element indices), the member's own subscript its
    lower bound plus that.  Member and buffer are both present on the device.  The trip counts are taken on the host
    into scalars first: as loop bounds, ``size(<array>(i1)%<member>, k)`` gives nvfortran a garbage launch geometry."""
    rank = len(recipe.cap_symbols)
    outer = aos_outer_rank(recipe)
    dims = range(1, rank + 1)
    extents = aos_pack_extent_locals(recipe)
    offsets = aos_pack_offset_locals(recipe)
    lines = [f"        {ext} = size({member_at_i}, {k})" for k, ext in zip(dims, extents, strict=True)]
    lines += [
        f"        {off} = lbound({member_at_i}, {k}) - {lb}"
        for k, off, lb in zip(dims, offsets, recipe.lb_symbols, strict=True)
    ]
    lines.append(f"        !$acc parallel loop collapse({rank}) default(present)")
    for k in reversed(list(dims)):
        lines.append("  " * (rank - k) + f"        do i{outer + k} = 1, {extents[k - 1]}")
    lines.append("  " * rank + "        " + assign)
    lines.extend("  " * (rank - k) + "        end do" for k in dims)
    return lines


def aos_pack_subscripts(recipe: FlattenRecipe) -> tuple[str, str]:
    """``(buffer subscripts, member subscripts)`` of one element's value in a device pack loop: ``i1, off1 + i2,
    off2 + i3`` and ``lb1 + off1 + i2 - 1, lb2 + off2 + i3 - 1`` for a rank-2 member of a rank-1 array of records,
    with the recipe's lower-bound symbols (the buffer's) and the element's offsets in the buffer."""
    outer = aos_outer_rank(recipe)
    dims = range(1, len(recipe.cap_symbols) + 1)
    offsets = aos_pack_offset_locals(recipe)
    buffer = ", ".join([*_element_indices(recipe), *(f"{offsets[k - 1]} + i{outer + k}" for k in dims)])
    member = ", ".join(
        f"{lb} + {offsets[k - 1]} + i{outer + k} - 1" for k, lb in zip(dims, recipe.lb_symbols, strict=True)
    )
    return buffer, member


def _own_range(recipe: FlattenRecipe, member_at_i: str) -> str:
    """The buffer range one element's member fills, per member dim: its own bounds shifted by the buffer's."""
    return ", ".join(
        f"lbound({member_at_i}, {k}) - {lb} + 1:ubound({member_at_i}, {k}) - {lb} + 1"
        for k, lb in enumerate(recipe.lb_symbols, start=1)
    )


def render_aos_alloc_pack_in(recipe: FlattenRecipe, outer_expr: str, device: bool = False) -> list[str]:
    """Take the elements' common lower bounds, compute each member dim's
    cap, allocate the buffer, pack each present element's live region.
    ``recipe.aos_alloc`` must be True.

    ``device``: the elements' members are present on the device and the SDFG reads the buffer there, so the buffer
    gets a device copy (``enter data create``) and the packing runs as OpenACC loops; the host copy is only
    allocated.  The buffer is zeroed on the device only when some element is smaller than the cap (its padding
    would otherwise be uninitialised).  The buffer and its device copy are kept from one call to the next (a saved
    allocatable): they are set up again (allocated, created, zeroed) only when the layout changes, the array of
    records, the caps, the common lower bounds or an element's presence or bounds; every call packs the members.
    The padding is never written after the setup, so a kept image holds what a fresh one would.
    """
    if not recipe.aos_alloc:
        raise ValueError("render_aos_alloc_pack_in called on non-aos_alloc recipe")
    flat = recipe.flat_names[0]
    caps = recipe.cap_symbols
    shape = ", ".join([*_aos_alloc_extents(recipe, outer_expr), *caps])
    elements = ", ".join(_element_indices(recipe))
    member_at_i = _aos_alloc_member_at_i(recipe)
    present = _aos_alloc_presence(recipe, member_at_i)
    dims = range(1, len(caps) + 1)
    live = _own_range(recipe, member_at_i)
    if device:
        buffer_idx, member_idx = aos_pack_subscripts(recipe)
        padded = " .or. ".join(
            f"size({member_at_i}, {k}) < {cap} .or. lbound({member_at_i}, {k}) /= {lb}"
            for k, cap, lb in zip(dims, caps, recipe.lb_symbols, strict=True)
        )
        saved, now, _k, fresh = aos_kept_layout_locals(recipe)
        setup = [
            f"    if (allocated({flat})) then",
            f"      !$acc exit data delete({flat})",
            f"      deallocate({flat})",
            "    end if",
            f"    allocate({flat}({shape}))",
            f"    !$acc enter data create({flat})",
            f"    {flat}_padded = .false.",
            *_over_elements(
                recipe,
                outer_expr,
                [f"      if ({present}) then", f"        if ({padded}) {flat}_padded = .true.", "      end if"],
            ),
            f"    if ({flat}_padded) then",
            "      !$acc kernels default(present)",
            f"      {flat} = 0",
            "      !$acc end kernels",
            "    end if",
            f"    if (allocated({saved})) deallocate({saved})",
            f"    allocate({saved}(size({now})))",
            f"    {saved} = {now}",
        ]
        fill = [
            *_kept_layout(recipe, outer_expr, member_at_i, present),
            f"    {fresh} = .not. allocated({saved})",
            f"    if (.not. {fresh}) {fresh} = size({saved}) /= size({now})",
            f"    if (.not. {fresh}) {fresh} = any({saved} /= {now})",
            f"    if ({fresh}) then",
            *("  " + line for line in setup),
            "    end if",
            f"    deallocate({now})",
            *_over_elements(
                recipe,
                outer_expr,
                [
                    f"      if ({present}) then",
                    *_device_pack_loop(recipe, member_at_i, f"{flat}({buffer_idx}) = {member_at_i}({member_idx})"),
                    "      end if",
                ],
            ),
        ]
    else:
        fill = [
            f"    allocate({flat}({shape}))",
            f"    {flat} = 0",
            *_over_elements(
                recipe,
                outer_expr,
                [f"      if ({present}) then", f"        {flat}({elements}, {live}) = {member_at_i}", "      end if"],
            ),
        ]
    return [
        f"    ! ----- AoS pack-in: {outer_expr} -> {flat} (cap = {', '.join(caps)})"
        f"{' on the device, kept across calls' if device else ''} -----",
        *_aos_alloc_bounds(recipe, outer_expr, spread=True),
        *fill,
    ]


def _aos_alloc_bounds(recipe: FlattenRecipe, outer_expr: str, spread: bool = False) -> list[str]:
    """Assign the recipe's lower-bound symbols the members' common lower bounds and its cap symbols their largest
    extents, over the present elements.

    ``spread``: the members go into one buffer, each at its own bounds, so they need not share their lower bounds:
    the buffer starts at the lowest (per member dim) and reaches the highest upper bound (boundary planes of
    different directions, say)."""
    caps, lbs = recipe.cap_symbols, recipe.lb_symbols
    member_at_i = _aos_alloc_member_at_i(recipe)
    present = _aos_alloc_presence(recipe, member_at_i)
    dims = range(1, len(caps) + 1)
    (taken,) = aos_bounds_locals(recipe)
    if spread:
        return [
            *(f"    {cap} = 0" for cap in caps),
            *(f"    {lb} = 1" for lb in lbs),
            f"    {taken} = .false.",
            # Lowest lower bound over the present non-empty elements (an empty one has lbound 1 in a zero-extent
            # dim), then the extent reaching the highest upper bound.
            *_over_elements(
                recipe,
                outer_expr,
                [
                    f"      if ({present}) then",
                    f"        if (size({member_at_i}) > 0) then",
                    f"          if (.not. {taken}) then",
                    *(f"            {lb} = lbound({member_at_i}, {k})" for k, lb in zip(dims, lbs, strict=True)),
                    f"            {taken} = .true.",
                    "          end if",
                    *(
                        f"          {lb} = min({lb}, lbound({member_at_i}, {k}))"
                        for k, lb in zip(dims, lbs, strict=True)
                    ),
                    "        end if",
                    "      end if",
                ],
            ),
            *_over_elements(
                recipe,
                outer_expr,
                [
                    f"      if ({present}) then",
                    f"        if (size({member_at_i}) > 0) then",
                    *(
                        f"          {cap} = max({cap}, ubound({member_at_i}, {k}) - {lb} + 1)"
                        for k, cap, lb in zip(dims, caps, lbs, strict=True)
                    ),
                    "        end if",
                    "      end if",
                ],
            ),
            *(f"    if ({cap} == 0) {cap} = 1" for cap in caps),
        ]
    differs = "error stop 'elements of " + outer_expr + " differ in their lower bounds'"
    return [
        *(f"    {cap} = 0" for cap in caps),
        *(f"    {lb} = 1" for lb in lbs),
        # The buffer takes the lower bounds of the first present non-empty element (an empty one has lbound 1 in a
        # zero-extent dim); every other non-empty element must agree.
        f"    {taken} = .false.",
        *_over_elements(
            recipe,
            outer_expr,
            [
                f"      if ({present}) then",
                f"        if (size({member_at_i}) > 0) then",
                f"          if (.not. {taken}) then",
                *(f"            {lb} = lbound({member_at_i}, {k})" for k, lb in zip(dims, lbs, strict=True)),
                f"            {taken} = .true.",
                "          end if",
                *(
                    f"          if (lbound({member_at_i}, {k}) /= {lb}) {differs}"
                    for k, lb in zip(dims, lbs, strict=True)
                ),
                "        end if",
                *(
                    f"        if (size({member_at_i}, {k}) > {cap}) {cap} = size({member_at_i}, {k})"
                    for k, cap in zip(dims, caps, strict=True)
                ),
                "      end if",
            ],
        ),
        # Empty-batch sentinel: keep every cap >= 1 so the buffer is non-degenerate.
        *(f"    if ({cap} == 0) {cap} = 1" for cap in caps),
    ]


def render_aos_alloc_members(
    recipe: FlattenRecipe, outer_expr: str, elements: Mapping[int, str], declined_flag: str | None = None
) -> list[str]:
    """Hand the SDFG the members of an array of records one by one instead of packed: ``elements`` maps a 0-based
    element index to the SDFG array it split into (:func:`dace_fortran.array_split.split_array`), which becomes a
    pointer to that element's member.  The SDFG addresses every member with the recipe's caps and lower bounds, so
    each must be present, contiguous and exactly that shape; otherwise the wrapper declines the call through
    ``declined_flag`` (when the wrapper has one and the caller passes it) or stops."""
    if not recipe.aos_alloc:
        raise ValueError("render_aos_alloc_members called on non-aos_alloc recipe")
    if aos_outer_rank(recipe) != 1:
        raise ValueError("render_aos_alloc_members takes the members of a rank-1 array of records")
    member_at_i = _aos_alloc_member_at_i(recipe)
    present = _aos_alloc_presence(recipe, member_at_i)
    dims = range(1, len(recipe.cap_symbols) + 1)
    conforming = " .and. ".join(
        [f"is_contiguous({member_at_i})"]
        + [f"size({member_at_i}, {k}) == {cap}" for k, cap in zip(dims, recipe.cap_symbols, strict=True)]
        + [f"lbound({member_at_i}, {k}) == {lb}" for k, lb in zip(dims, recipe.lb_symbols, strict=True)]
    )
    refuse = []
    if declined_flag:
        refuse += [f"if (present({declined_flag})) then", f"  {declined_flag} = .true.", "  return", "end if"]
    refuse.append(f"error stop 'an element of {outer_expr} is absent or not of the shape its siblings share'")
    lines = [
        f"    ! ----- Members of {outer_expr}, one array per element, aliased in place -----",
        *_aos_alloc_bounds(recipe, outer_expr),
    ]
    for index, name in sorted(elements.items()):
        # Presence first: Fortran's .and. does not short-circuit, and an absent member has no shape to query.
        lines += [
            f"    i1 = {index + 1}",
            f"    if (.not. {present}) then",
            *(f"      {ln}" for ln in refuse),
            "    end if",
            f"    if (.not. ({conforming})) then",
            *(f"      {ln}" for ln in refuse),
            "    end if",
            f"    {name} => {member_at_i}",
        ]
    return lines


def render_aos_alloc_pack_out(recipe: FlattenRecipe, outer_expr: str, device: bool = False) -> list[str]:
    """Copy each present element's live region back from the buffer, free
    the scratch.  No reallocation -- 5c-B kernels don't change
    per-instance sizes (reserved for 5c-C).  ``device`` mirrors
    :func:`render_aos_alloc_pack_in`: OpenACC loops; the buffer and its device copy stay for the next call."""
    if not recipe.aos_alloc:
        raise ValueError("render_aos_alloc_pack_out called on non-aos_alloc recipe")
    flat = recipe.flat_names[0]
    elements = ", ".join(_element_indices(recipe))
    member_at_i = _aos_alloc_member_at_i(recipe)
    present = _aos_alloc_presence(recipe, member_at_i)
    live = _own_range(recipe, member_at_i)
    if device:
        buffer_idx, member_idx = aos_pack_subscripts(recipe)
        unpack = _device_pack_loop(recipe, member_at_i, f"{member_at_i}({member_idx}) = {flat}({buffer_idx})")
    else:
        unpack = [f"        {member_at_i} = {flat}({elements}, {live})"]
    return [
        f"    ! ----- AoS pack-out: {flat} -> {outer_expr}{' on the device' if device else ''} -----",
        *_over_elements(recipe, outer_expr, [f"      if ({present}) then", *unpack, "      end if"]),
        *render_aos_alloc_release(recipe, device),
    ]


def render_aos_alloc_release(recipe: FlattenRecipe, device: bool = False) -> list[str]:
    """Free the pack-in's buffer; a device image is kept for the next call (see :func:`render_aos_alloc_pack_in`)."""
    flat = recipe.flat_names[0]
    if device:
        return [f"    ! {flat} and its device copy stay allocated for the next call"]
    return [f"    deallocate({flat})"]
