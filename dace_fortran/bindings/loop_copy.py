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


def _device_pack_loop(recipe: FlattenRecipe, member_at_i: str, assign: str) -> list[str]:
    """An OpenACC loop nest over one present element's member, running ``assign`` per element of it: the buffer
    index of member dim ``k`` is ``i<R+k>`` (``i1 ... iR`` are the element indices), the member's own subscript its
    lower bound plus that.  Member and buffer are both present on the device.  The trip counts are taken on the host
    into scalars first: as loop bounds, ``size(<array>(i1)%<member>, k)`` gives nvfortran a garbage launch geometry."""
    rank = len(recipe.cap_symbols)
    outer = aos_outer_rank(recipe)
    dims = range(1, rank + 1)
    extents = aos_pack_extent_locals(recipe)
    lines = [f"        {ext} = size({member_at_i}, {k})" for k, ext in zip(dims, extents, strict=True)]
    lines.append(f"        !$acc parallel loop collapse({rank}) default(present)")
    for k in reversed(list(dims)):
        lines.append("  " * (rank - k) + f"        do i{outer + k} = 1, {extents[k - 1]}")
    lines.append("  " * rank + "        " + assign)
    lines.extend("  " * (rank - k) + "        end do" for k in dims)
    return lines


def aos_pack_subscripts(recipe: FlattenRecipe) -> tuple[str, str]:
    """``(buffer subscripts, member subscripts)`` of one element's value in a device pack loop: ``i1, i2, i3`` and
    ``lb1 + i2 - 1, lb2 + i3 - 1`` for a rank-2 member of a rank-1 array of records, with the recipe's common
    lower-bound symbols."""
    outer = aos_outer_rank(recipe)
    dims = range(1, len(recipe.cap_symbols) + 1)
    buffer = ", ".join([*_element_indices(recipe), *(f"i{outer + k}" for k in dims)])
    member = ", ".join(f"{lb} + i{outer + k} - 1" for k, lb in zip(dims, recipe.lb_symbols, strict=True))
    return buffer, member


def render_aos_alloc_pack_in(recipe: FlattenRecipe, outer_expr: str, device: bool = False) -> list[str]:
    """Take the elements' common lower bounds, compute each member dim's
    cap, allocate the buffer, pack each present element's live region.
    ``recipe.aos_alloc`` must be True.

    ``device``: the elements' members are present on the device and the SDFG reads the buffer there, so the buffer
    gets a device copy (``enter data create``) and the packing runs as OpenACC loops; the host copy is only
    allocated.  The buffer is zeroed on the device only when some element is smaller than the cap (its padding
    would otherwise be uninitialised).
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
    live = ", ".join(f"1:size({member_at_i}, {k})" for k in dims)
    if device:
        buffer_idx, member_idx = aos_pack_subscripts(recipe)
        padded = " .or. ".join(f"size({member_at_i}, {k}) < {cap}" for k, cap in zip(dims, caps, strict=True))
        fill = [
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
        f"    ! ----- AoS pack-in: {outer_expr} -> {flat} (cap = {', '.join(caps)}){' on the device' if device else ''} -----",
        *_aos_alloc_bounds(recipe, outer_expr),
        *fill,
    ]


def _aos_alloc_bounds(recipe: FlattenRecipe, outer_expr: str) -> list[str]:
    """Assign the recipe's lower-bound symbols the members' common lower bounds and its cap symbols their largest
    extents, over the present elements."""
    caps, lbs = recipe.cap_symbols, recipe.lb_symbols
    member_at_i = _aos_alloc_member_at_i(recipe)
    present = _aos_alloc_presence(recipe, member_at_i)
    dims = range(1, len(caps) + 1)
    differs = "error stop 'elements of " + outer_expr + " differ in their lower bounds'"
    (taken,) = aos_bounds_locals(recipe)
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
    :func:`render_aos_alloc_pack_in`: OpenACC loops, then the device copy is deleted."""
    if not recipe.aos_alloc:
        raise ValueError("render_aos_alloc_pack_out called on non-aos_alloc recipe")
    flat = recipe.flat_names[0]
    elements = ", ".join(_element_indices(recipe))
    member_at_i = _aos_alloc_member_at_i(recipe)
    present = _aos_alloc_presence(recipe, member_at_i)
    live = ", ".join(f"1:size({member_at_i}, {k})" for k in range(1, len(recipe.cap_symbols) + 1))
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
    """Free the pack-in's buffer, device copy first."""
    flat = recipe.flat_names[0]
    return [*([f"    !$acc exit data delete({flat})"] if device else []), f"    deallocate({flat})"]
