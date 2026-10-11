"""Per-renderer tests for the alias / copy-in / copy-out primitives.

Each renderer tested in isolation: single recipe in, list of Fortran lines out.
"""

import pytest

from dace_fortran.bindings import FlattenRecipe
from dace_fortran.bindings.loop_copy import (
    render_alias_calls,
    render_aos_alloc_pack_in,
    render_aos_alloc_pack_out,
    render_copy_in_loop,
    render_copy_out_loop,
)


def _alias_recipe() -> FlattenRecipe:
    return FlattenRecipe(
        flat_names=("fld_a",),
        read_exprs=("fld%a($i1, $i2)",),
        rank=2,
        shape_exprs=("size(fld%a, dim=1)", "size(fld%a, dim=2)"),
        aliasable=True,
    )


def _complex_recipe() -> FlattenRecipe:
    return FlattenRecipe(
        flat_names=("st_z_re", "st_z_im"),
        read_exprs=(
            "real(st%z($i1,$i2), kind=c_double)",
            "aimag(st%z($i1,$i2))",
        ),
        write_expr="cmplx(st_z_re($i1,$i2), st_z_im($i1,$i2), kind=c_double)",
        rank=2,
        shape_exprs=("size(st%z, dim=1)", "size(st%z, dim=2)"),
        aliasable=False,
    )


# --------------------------------------------------------------------------
# render_alias_calls
# --------------------------------------------------------------------------


def test_render_alias_calls_single_flat():
    lines = render_alias_calls(_alias_recipe())
    assert len(lines) == 1
    assert "call c_f_pointer(c_loc(fld%a), fld_a, [" in lines[0]
    assert "size(fld%a, dim=1)" in lines[0]
    assert "size(fld%a, dim=2)" in lines[0]


def test_render_alias_calls_strips_index_placeholders():
    """``c_loc`` must see the base storage path, not the indexed
    expression  --  strip_index_args handles this."""
    lines = render_alias_calls(_alias_recipe())
    # ``c_loc(fld%a)`` with no ``($i1, $i2)`` suffix.
    assert "c_loc(fld%a)," in lines[0]
    assert "$i" not in "".join(lines)


def test_render_alias_calls_raises_on_non_aliasable():
    with pytest.raises(ValueError, match="non-aliasable"):
        render_alias_calls(_complex_recipe())


def test_render_alias_calls_nested_struct_path():
    recipe = FlattenRecipe(
        flat_names=("st_a_v",),
        read_exprs=("st%a%v($i1, $i2)",),
        rank=2,
        shape_exprs=("size(st%a%v, dim=1)", "size(st%a%v, dim=2)"),
        aliasable=True,
    )
    lines = render_alias_calls(recipe)
    assert "c_loc(st%a%v)" in lines[0]


# --------------------------------------------------------------------------
# render_copy_in_loop
# --------------------------------------------------------------------------


def test_render_copy_in_loop_allocates_each_flat():
    lines = render_copy_in_loop(_complex_recipe())
    joined = "\n".join(lines)
    assert "allocate(st_z_re(size(st%z, dim=1), size(st%z, dim=2)))" in joined
    assert "allocate(st_z_im(size(st%z, dim=1), size(st%z, dim=2)))" in joined


def test_render_copy_in_loop_nested_do_nest():
    lines = render_copy_in_loop(_complex_recipe())
    joined = "\n".join(lines)
    # Two loop heads (one per rank), two ``end do`` markers.
    assert joined.count("do i1 = 1,") == 1
    assert joined.count("do i2 = 1,") == 1
    assert joined.count("end do") == 2


def test_render_copy_in_loop_substitutes_index_names():
    lines = render_copy_in_loop(_complex_recipe())
    joined = "\n".join(lines)
    assert "st_z_re(i1, i2) = real(st%z(i1,i2), kind=c_double)" in joined
    assert "st_z_im(i1, i2) = aimag(st%z(i1,i2))" in joined
    # No raw placeholders survive.
    assert "$i1" not in joined and "$i2" not in joined


def test_render_copy_in_loop_raises_on_aliasable():
    with pytest.raises(ValueError, match="aliasable"):
        render_copy_in_loop(_alias_recipe())


# --------------------------------------------------------------------------
# render_copy_out_loop
# --------------------------------------------------------------------------


def test_render_copy_out_loop_shape():
    lines = render_copy_out_loop(_complex_recipe(), outer_expr="st%z")
    joined = "\n".join(lines)
    # Reconstructs outer element, then deallocates each flat.
    assert "st%z(i1, i2) = cmplx(st_z_re(i1,i2), st_z_im(i1,i2), kind=c_double)" in joined
    assert "deallocate(st_z_re)" in joined
    assert "deallocate(st_z_im)" in joined


def test_render_copy_out_loop_inverse_for_single_flat_empty_write_expr():
    """Single-flat member with no ``write_expr`` copies out as the exact
    inverse of copy-in: ``<read_expr> = <flat>(idx)``, then deallocates."""
    r = FlattenRecipe(
        flat_names=("pts_x",),
        read_exprs=("pts($i1)%x",),
        rank=1,
        shape_exprs=("size(pts, dim=1)",),
        aliasable=False,
    )
    body = "\n".join(render_copy_out_loop(r, outer_expr="pts%x"))
    assert "pts(i1)%x = pts_x(i1)" in body
    assert "deallocate(pts_x)" in body


def test_render_copy_out_loop_raises_on_multi_flat_empty_write_expr():
    """A multi-flat recipe with no ``write_expr`` has no unambiguous inverse
    (the reconstruction must be supplied as ``write_expr``) and must raise."""
    r = FlattenRecipe(
        flat_names=("st_z_re", "st_z_im"),
        read_exprs=("real(st%z($i1))", "aimag(st%z($i1))"),
        rank=1,
        shape_exprs=("size(st%z, dim=1)",),
        aliasable=False,
    )
    with pytest.raises(ValueError, match="empty write_expr"):
        render_copy_out_loop(r, outer_expr="st%z")


# --------------------------------------------------------------------------
# render_aos_alloc_pack_in / render_aos_alloc_pack_out (Phase 5c-B boundary)
# --------------------------------------------------------------------------


def _aos_alloc_recipe() -> FlattenRecipe:
    return FlattenRecipe(
        flat_names=("a_w",),
        read_exprs=("a($i1)%w($i2)",),
        rank=2,
        shape_exprs=("size(a, dim=1)", "cap_a_w"),
        aliasable=False,
        aos_alloc=True,
        cap_symbols=("cap_a_w",),
        lb_symbols=("lb_a_w",),
    )


def test_render_aos_alloc_pack_in_emits_cap_max_loop():
    lines = render_aos_alloc_pack_in(_aos_alloc_recipe(), outer_expr="a")
    joined = "\n".join(lines)
    # Cap = max per-instance size, guarded by allocated() so unalloc'd rows don't poison it.
    assert "cap_a_w = 0" in joined
    assert "if (allocated(a(i1)%w))" in joined
    assert "cap_a_w = max(cap_a_w, ubound(a(i1)%w, 1) - lb_a_w + 1)" in joined
    # The buffer spans the elements' lower bounds: the lowest one is its own.
    assert "lb_a_w = lbound(a(i1)%w, 1)" in joined
    assert "lb_a_w = min(lb_a_w, lbound(a(i1)%w, 1))" in joined
    # Empty-batch sentinel keeps the buffer non-degenerate.
    assert "if (cap_a_w == 0) cap_a_w = 1" in joined
    # Buffer allocate at (N, cap) shape, zero-init, then per-row copy.
    assert "allocate(a_w(size(a, dim=1), cap_a_w))" in joined
    assert "a_w = 0" in joined
    assert "a_w(i1, lbound(a(i1)%w, 1) - lb_a_w + 1:ubound(a(i1)%w, 1) - lb_a_w + 1) = a(i1)%w" in joined


def test_render_aos_alloc_pack_out_copies_back_live_region():
    lines = render_aos_alloc_pack_out(_aos_alloc_recipe(), outer_expr="a")
    joined = "\n".join(lines)
    # Per-row copy-back, guarded by allocated() -- unalloc'd rows stay zeroed (bindings policy).
    assert "if (allocated(a(i1)%w))" in joined
    assert "a(i1)%w = a_w(i1, lbound(a(i1)%w, 1) - lb_a_w + 1:ubound(a(i1)%w, 1) - lb_a_w + 1)" in joined
    assert "deallocate(a_w)" in joined


def test_render_aos_alloc_pack_for_rank3_pointer_member():
    """A rank-3 pointer member: presence by associated(), one cap and lower bound per member dim."""
    r = FlattenRecipe(
        flat_names=("q_sf",),
        read_exprs=("q($i1)%sf($i2, $i3, $i4)",),
        rank=4,
        shape_exprs=("size(q, dim=1)", "cap_q_sf_d0", "cap_q_sf_d1", "cap_q_sf_d2"),
        aliasable=False,
        aos_alloc=True,
        cap_symbols=("cap_q_sf_d0", "cap_q_sf_d1", "cap_q_sf_d2"),
        lb_symbols=("lb_q_sf_d0", "lb_q_sf_d1", "lb_q_sf_d2"),
        member_pointer=True,
    )
    pack_in = "\n".join(render_aos_alloc_pack_in(r, outer_expr="q"))
    assert "if (associated(q(i1)%sf))" in pack_in
    assert "allocated(" not in pack_in
    assert "lb_q_sf_d2 = lbound(q(i1)%sf, 3)" in pack_in
    assert "cap_q_sf_d2 = max(cap_q_sf_d2, ubound(q(i1)%sf, 3) - lb_q_sf_d2 + 1)" in pack_in
    assert "allocate(q_sf(size(q, dim=1), cap_q_sf_d0, cap_q_sf_d1, cap_q_sf_d2))" in pack_in
    own = ("lbound(q(i1)%sf, 1) - lb_q_sf_d0 + 1:ubound(q(i1)%sf, 1) - lb_q_sf_d0 + 1, "
           "lbound(q(i1)%sf, 2) - lb_q_sf_d1 + 1:ubound(q(i1)%sf, 2) - lb_q_sf_d1 + 1, "
           "lbound(q(i1)%sf, 3) - lb_q_sf_d2 + 1:ubound(q(i1)%sf, 3) - lb_q_sf_d2 + 1")
    assert f"q_sf(i1, {own}) = q(i1)%sf" in pack_in
    pack_out = "\n".join(render_aos_alloc_pack_out(r, outer_expr="q"))
    assert f"q(i1)%sf = q_sf(i1, {own})" in pack_out


def test_render_aos_alloc_pack_for_a_rank2_array_of_records():
    """A rank-2 array of records: a loop per record dim (the last outermost), every record dim in the buffer's
    subscripts ahead of the member's, and the device pack loops number the member dims after the record dims."""
    r = FlattenRecipe(
        flat_names=("bc_sf",),
        read_exprs=("bc($i1, $i2)%sf($i3, $i4)",),
        rank=4,
        shape_exprs=("size(bc, dim=1)", "size(bc, dim=2)", "cap_bc_sf_d0", "cap_bc_sf_d1"),
        aliasable=False,
        aos_alloc=True,
        cap_symbols=("cap_bc_sf_d0", "cap_bc_sf_d1"),
        lb_symbols=("lb_bc_sf_d0", "lb_bc_sf_d1"),
    )
    pack_in = render_aos_alloc_pack_in(r, outer_expr="bc")
    assert [ln.strip() for ln in pack_in if ln.strip().startswith("do ")][:2] == [
        "do i2 = 1, size(bc, dim=2)",
        "do i1 = 1, size(bc, dim=1)",
    ]
    joined = "\n".join(pack_in)
    assert "allocate(bc_sf(size(bc, dim=1), size(bc, dim=2), cap_bc_sf_d0, cap_bc_sf_d1))" in joined
    own = ("lbound(bc(i1, i2)%sf, 1) - lb_bc_sf_d0 + 1:ubound(bc(i1, i2)%sf, 1) - lb_bc_sf_d0 + 1, "
           "lbound(bc(i1, i2)%sf, 2) - lb_bc_sf_d1 + 1:ubound(bc(i1, i2)%sf, 2) - lb_bc_sf_d1 + 1")
    assert f"bc_sf(i1, i2, {own}) = bc(i1, i2)%sf" in joined
    assert "exit" not in joined  # an exit would leave only the innermost record loop
    pack_out = "\n".join(render_aos_alloc_pack_out(r, outer_expr="bc"))
    assert f"bc(i1, i2)%sf = bc_sf(i1, i2, {own})" in pack_out
    device = "\n".join(render_aos_alloc_pack_in(r, outer_expr="bc", device=True))
    assert "do i3 = 1, bc_sf_ext1" in device and "do i4 = 1, bc_sf_ext2" in device
    assert ("bc_sf(i1, i2, bc_sf_off1 + i3, bc_sf_off2 + i4) = "
            "bc(i1, i2)%sf(lb_bc_sf_d0 + bc_sf_off1 + i3 - 1, lb_bc_sf_d1 + bc_sf_off2 + i4 - 1)") in device


def test_render_aos_alloc_pack_in_raises_on_non_aos_alloc():
    plain = FlattenRecipe(
        flat_names=("a_w",),
        read_exprs=("a($i1)%w($i2)",),
        rank=2,
        shape_exprs=("size(a, dim=1)", "cap_a_w"),
        aliasable=False,
    )
    with pytest.raises(ValueError, match="non-aos_alloc"):
        render_aos_alloc_pack_in(plain, outer_expr="a")
    with pytest.raises(ValueError, match="non-aos_alloc"):
        render_aos_alloc_pack_out(plain, outer_expr="a")


def test_kind_convert_recipe_rendering():
    """Kind-convert shape: ``real(kind=4)`` outer -> ``real(kind=8)`` SDFG flat,
    via the same machinery -- no dedicated strategy needed."""
    recipe = FlattenRecipe(
        flat_names=("st_x_d",),
        read_exprs=("real(st%x($i1), kind=c_double)",),
        write_expr="real(st_x_d($i1), kind=c_float)",
        rank=1,
        shape_exprs=("size(st%x, dim=1)",),
        aliasable=False,
        scratch_dtype="float64",
    )
    in_lines = render_copy_in_loop(recipe)
    out_lines = render_copy_out_loop(recipe, outer_expr="st%x")
    assert "st_x_d(i1) = real(st%x(i1), kind=c_double)" in "\n".join(in_lines)
    assert "st%x(i1) = real(st_x_d(i1), kind=c_float)" in "\n".join(out_lines)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
