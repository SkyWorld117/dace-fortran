# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Recipe renderers -- the ONLY code that knows how to turn a
``FlattenRecipe`` into Fortran.

Three renderers: ``render_alias_calls`` (zero-copy ``c_f_pointer``
alias), ``render_copy_in_loop`` (allocate + forward do-loop copy),
``render_copy_out_loop`` (reverse do-loop copy + deallocate).  All
return ``List[str]`` of lines pre-indented to wrapper-body level.
"""

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
# Helpers below extract `A($i1)%w` from recipe.read_exprs[0] by dropping the
# trailing member subscript list -- safe because the bridge always emits
# aos_alloc recipes as `<outer>($i1)%<member>($i2, ...)`.  i1 is the iterator
# build_wrapper_head declares for any non-aliasable rank>=1 recipe.


def _aos_alloc_member_at_i(recipe: FlattenRecipe) -> str:
    """Extract ``<outer>(i1)%<member>`` (no member subscripts) from an
    aos_alloc recipe's ``read_exprs[0]``, for presence / size() / lbound()
    queries in the pack-in/pack-out emitters."""
    template = recipe.read_exprs[0]  # "A($i1)%w($i2, ...)"
    return template.rsplit("(", 1)[0].replace("$i1", "i1")


def _aos_alloc_presence(recipe: FlattenRecipe, member: str) -> str:
    """Presence query for one element's member: ``associated`` for a pointer, else ``allocated``."""
    return f"associated({member})" if recipe.member_pointer else f"allocated({member})"


def aos_pack_extent_locals(recipe: FlattenRecipe) -> tuple[str, ...]:
    """Scalar locals holding one element's member extents for the device pack loops (``<flat>_ext1``, ...)."""
    return tuple(f"{recipe.flat_names[0]}_ext{k}" for k in range(1, len(recipe.cap_symbols) + 1))


def _device_pack_loop(recipe: FlattenRecipe, member_at_i: str, assign: str) -> list[str]:
    """An OpenACC loop nest over one present element's member, running ``assign`` per element of it: the buffer
    index of member dim ``k`` is ``i<k+1>`` (``i1`` is the element index), the member's own subscript its lower
    bound plus that.  Member and buffer are both present on the device.  The trip counts are taken on the host into
    scalars first: as loop bounds, ``size(<array>(i1)%<member>, k)`` gives nvfortran a garbage launch geometry."""
    rank = len(recipe.cap_symbols)
    dims = range(1, rank + 1)
    extents = aos_pack_extent_locals(recipe)
    lines = [f"        {ext} = size({member_at_i}, {k})" for k, ext in zip(dims, extents, strict=True)]
    lines.append(f"        !$acc parallel loop collapse({rank}) default(present)")
    for k in reversed(list(dims)):
        lines.append("  " * (rank - k) + f"        do i{k + 1} = 1, {extents[k - 1]}")
    lines.append("  " * rank + "        " + assign)
    lines.extend("  " * (rank - k) + "        end do" for k in dims)
    return lines


def aos_pack_subscripts(recipe: FlattenRecipe) -> tuple[str, str]:
    """``(buffer subscripts, member subscripts)`` of one element's value in a device pack loop: ``i1, i2, i3`` and
    ``lb1 + i2 - 1, lb2 + i3 - 1`` for a rank-2 member, with the recipe's common lower-bound symbols."""
    dims = range(1, len(recipe.cap_symbols) + 1)
    buffer = ", ".join(["i1", *(f"i{k + 1}" for k in dims)])
    member = ", ".join(f"{lb} + i{k + 1} - 1" for k, lb in zip(dims, recipe.lb_symbols, strict=True))
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
    caps, lbs = recipe.cap_symbols, recipe.lb_symbols
    n_extent = recipe.shape_exprs[0] if recipe.shape_exprs else "size(" + outer_expr + ")"
    member_at_i = _aos_alloc_member_at_i(recipe)
    present = _aos_alloc_presence(recipe, member_at_i)
    dims = range(1, len(caps) + 1)
    live = ", ".join(f"1:size({member_at_i}, {k})" for k in dims)
    differs = "error stop 'elements of " + outer_expr + " differ in their lower bounds'"
    if device:
        buffer_idx, member_idx = aos_pack_subscripts(recipe)
        padded = " .or. ".join(f"size({member_at_i}, {k}) < {cap}" for k, cap in zip(dims, caps, strict=True))
        fill = [
            f"    allocate({flat}({n_extent}, {', '.join(caps)}))",
            f"    !$acc enter data create({flat})",
            f"    {flat}_padded = .false.",
            f"    do i1 = 1, {n_extent}",
            f"      if ({present}) then",
            f"        if ({padded}) {flat}_padded = .true.",
            "      end if",
            "    end do",
            f"    if ({flat}_padded) then",
            "      !$acc kernels default(present)",
            f"      {flat} = 0",
            "      !$acc end kernels",
            "    end if",
            f"    do i1 = 1, {n_extent}",
            f"      if ({present}) then",
            *_device_pack_loop(recipe, member_at_i, f"{flat}({buffer_idx}) = {member_at_i}({member_idx})"),
            "      end if",
            "    end do",
        ]
    else:
        fill = [
            f"    allocate({flat}({n_extent}, {', '.join(caps)}))",
            f"    {flat} = 0",
            f"    do i1 = 1, {n_extent}",
            f"      if ({present}) then",
            f"        {flat}(i1, {live}) = {member_at_i}",
            "      end if",
            "    end do",
        ]
    return [
        f"    ! ----- AoS pack-in: {outer_expr} -> {flat} (cap = {', '.join(caps)}){' on the device' if device else ''} -----",
        *(f"    {cap} = 0" for cap in caps),
        *(f"    {lb} = 1" for lb in lbs),
        # The buffer takes the lower bounds of the first present non-empty element (an empty one has lbound 1 in a
        # zero-extent dim); every other non-empty element must agree.
        f"    do i1 = 1, {n_extent}",
        f"      if ({present}) then",
        f"        if (size({member_at_i}) > 0) then",
        *(f"          {lb} = lbound({member_at_i}, {k})" for k, lb in zip(dims, lbs, strict=True)),
        "          exit",
        "        end if",
        "      end if",
        "    end do",
        f"    do i1 = 1, {n_extent}",
        f"      if ({present}) then",
        f"        if (size({member_at_i}) > 0) then",
        *(f"          if (lbound({member_at_i}, {k}) /= {lb}) {differs}" for k, lb in zip(dims, lbs, strict=True)),
        "        end if",
        *(
            f"        if (size({member_at_i}, {k}) > {cap}) {cap} = size({member_at_i}, {k})"
            for k, cap in zip(dims, caps, strict=True)
        ),
        "      end if",
        "    end do",
        # Empty-batch sentinel: keep every cap >= 1 so the buffer is non-degenerate.
        *(f"    if ({cap} == 0) {cap} = 1" for cap in caps),
        *fill,
    ]


def render_aos_alloc_pack_out(recipe: FlattenRecipe, outer_expr: str, device: bool = False) -> list[str]:
    """Copy each present element's live region back from the buffer, free
    the scratch.  No reallocation -- 5c-B kernels don't change
    per-instance sizes (reserved for 5c-C).  ``device`` mirrors
    :func:`render_aos_alloc_pack_in`: OpenACC loops, then the device copy is deleted."""
    if not recipe.aos_alloc:
        raise ValueError("render_aos_alloc_pack_out called on non-aos_alloc recipe")
    flat = recipe.flat_names[0]
    n_extent = recipe.shape_exprs[0] if recipe.shape_exprs else "size(" + outer_expr + ")"
    member_at_i = _aos_alloc_member_at_i(recipe)
    present = _aos_alloc_presence(recipe, member_at_i)
    live = ", ".join(f"1:size({member_at_i}, {k})" for k in range(1, len(recipe.cap_symbols) + 1))
    if device:
        buffer_idx, member_idx = aos_pack_subscripts(recipe)
        unpack = _device_pack_loop(recipe, member_at_i, f"{member_at_i}({member_idx}) = {flat}({buffer_idx})")
    else:
        unpack = [f"        {member_at_i} = {flat}(i1, {live})"]
    return [
        f"    ! ----- AoS pack-out: {flat} -> {outer_expr}{' on the device' if device else ''} -----",
        f"    do i1 = 1, {n_extent}",
        f"      if ({present}) then",
        *unpack,
        "      end if",
        "    end do",
        *render_aos_alloc_release(recipe, device),
    ]


def render_aos_alloc_release(recipe: FlattenRecipe, device: bool = False) -> list[str]:
    """Free the pack-in's buffer, device copy first."""
    flat = recipe.flat_names[0]
    return [*([f"    !$acc exit data delete({flat})"] if device else []), f"    deallocate({flat})"]
