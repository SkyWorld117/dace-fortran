"""N-dim AoS gather + static value-member marshalling (binding gate #12d).

``_render_aos_copy_in``/``_render_aos_copy_out`` flatten a derived-type component
over an N-D record array into SoA; for a fixed-shape VALUE member (e.g. ICON-O's
``p_vn_dual(:,:,:)%x``) all record indices are used and no presence guard is
emitted. Text-only checks on emitted Fortran -- no flang/gfortran needed.
"""

from dace_fortran.bindings.block_builders import _render_aos_copy_in, _render_aos_copy_out
from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenArgKind


def _cartesian_member_arg(is_written=False):
    """``p_diag % p_vn_dual(:,:,:) % x(3)`` -- rank-3 pointer record array, a
    fixed-shape rank-1 VALUE member (companion rank 4, member dim literal 3)."""
    return FrozenArg(
        fortran_name="p_diag_p_vn_dual_x",
        sdfg_name="p_diag_p_vn_dual_x",
        kind=FrozenArgKind.ARRAY,
        dtype="float64",
        rank=4,
        shape=("p_diag_p_vn_dual_x_d0", "p_diag_p_vn_dual_x_d1", "p_diag_p_vn_dual_x_d2", "3"),
        is_written=is_written,
        aos_origin_struct="p_diag % p_vn_dual",
        aos_member_path="x",
        aos_outer_rank=3,
        aos_struct_pointer=True,
        aos_member_pointer=False,
    )


def _becxx_member_arg():
    """QE ``becxx(:) % k(:, :)`` -- 1-D allocatable record array, a rank-2
    ALLOCATABLE member (companion rank 3, symbolic member dims)."""
    return FrozenArg(
        fortran_name="becxx_k",
        sdfg_name="becxx_k",
        kind=FrozenArgKind.ARRAY,
        dtype="float64",
        rank=3,
        shape=("becxx_k_d0", "becxx_k_d1", "becxx_k_d2"),
        aos_origin_struct="becxx",
        aos_member_path="k",
        aos_outer_rank=1,
        aos_struct_pointer=False,
        aos_member_pointer=False,
    )


def test_ndim_cartesian_copy_in_indexes_all_outer_dims():
    """The rank-3 record array is indexed by all three element loops, and the
    rank-4 companion is allocated over the three outer extents + literal 3."""
    src = "\n".join(_render_aos_copy_in(_cartesian_member_arg()))
    # Three nested element loops over the three record-array dims.
    assert "do aos_p_diag_p_vn_dual_x_i0 = lbound(p_diag % p_vn_dual, 1), ubound(p_diag % p_vn_dual, 1)" in src
    assert "do aos_p_diag_p_vn_dual_x_i1 = lbound(p_diag % p_vn_dual, 2), ubound(p_diag % p_vn_dual, 2)" in src
    assert "do aos_p_diag_p_vn_dual_x_i2 = lbound(p_diag % p_vn_dual, 3), ubound(p_diag % p_vn_dual, 3)" in src
    # The element accessor uses all three indices (not a single-index 1/3 ref).
    assert (
        "p_diag % p_vn_dual(aos_p_diag_p_vn_dual_x_i0, aos_p_diag_p_vn_dual_x_i1, aos_p_diag_p_vn_dual_x_i2)%x"
    ) in src
    # Rank-4 allocate: three outer extents + the literal member extent.
    assert (
        "allocate(p_diag_p_vn_dual_x(lbound(p_diag % p_vn_dual, 1):ubound(p_diag % p_vn_dual, 1), "
        "lbound(p_diag % p_vn_dual, 2):ubound(p_diag % p_vn_dual, 2), "
        "lbound(p_diag % p_vn_dual, 3):ubound(p_diag % p_vn_dual, 3), 3)"
    ) in src


def test_static_value_member_emits_no_presence_guard():
    """``x(3)`` is fixed-shape: no allocated/associated guard or bounds scan on
    it, only the struct-present (``associated``) guard remains."""
    src = "\n".join(_render_aos_copy_in(_cartesian_member_arg()))
    assert "allocated(p_diag % p_vn_dual(" not in src  # member is not allocatable
    assert "aos_p_diag_p_vn_dual_x_l0" not in src  # no bounds scan var
    assert "associated(p_diag % p_vn_dual)" in src  # struct-pointer guard kept


def test_static_value_member_copy_out_scatters_all_dims():
    """When written, copy-out scatters back through all three element loops."""
    src = "\n".join(_render_aos_copy_out(_cartesian_member_arg(is_written=True)))
    assert "do aos_p_diag_p_vn_dual_x_i2 = lbound(p_diag % p_vn_dual, 3), ubound(p_diag % p_vn_dual, 3)" in src
    assert (
        "p_diag % p_vn_dual(aos_p_diag_p_vn_dual_x_i0, aos_p_diag_p_vn_dual_x_i1, "
        "aos_p_diag_p_vn_dual_x_i2)%x = p_diag_p_vn_dual_x(aos_p_diag_p_vn_dual_x_i0, "
        "aos_p_diag_p_vn_dual_x_i1, aos_p_diag_p_vn_dual_x_i2, 1:3)"
    ) in src


def test_1d_allocatable_member_unchanged():
    """Regression: QE ``becxx(:)%k`` 1-D allocatable path keeps its single element
    loop, bounds scan, and member ``allocated`` guard after the N-dim generalisation."""
    src = "\n".join(_render_aos_copy_in(_becxx_member_arg()))
    assert "do aos_becxx_k_i0 = lbound(becxx, 1), ubound(becxx, 1)" in src
    assert "allocated(becxx(aos_becxx_k_i0)%k)" in src  # allocatable member guard
    assert "aos_becxx_k_l0 = min(aos_becxx_k_l0, lbound(becxx(aos_becxx_k_i0)%k, 1))" in src
    assert "aos_becxx_k_u1 = max(aos_becxx_k_u1, ubound(becxx(aos_becxx_k_i0)%k, 2))" in src


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))


def test_deferred_member_image_packs_on_the_device():
    """On a device-resident caller the image of a deferred-shape member is filled by OpenACC loops over the
    members' device copies, from a buffer with a device copy; the host buffer is only allocated."""
    src = "\n".join(_render_aos_copy_in(_becxx_member_arg(), device=True))
    assert "!$acc enter data create(becxx_k)" in src
    assert "!$acc kernels default(present)" in src  # the padding is zeroed on the device
    assert "!$acc parallel loop collapse(2) default(present)" in src
    # One element's own bounds are taken on the host before its loop nest.
    assert "aos_becxx_k_el0 = lbound(becxx(aos_becxx_k_i0)%k, 1)" in src
    assert "do aos_becxx_k_m1 = aos_becxx_k_el1, aos_becxx_k_eu1" in src
    assert "becxx_k(aos_becxx_k_i0, aos_becxx_k_m0, aos_becxx_k_m1) = becxx(aos_becxx_k_i0)%k(aos_becxx_k_m0, aos_becxx_k_m1)" in src
    assert "= becxx(aos_becxx_k_i0)%k\n" not in src  # no host array assignment
    assert "becxx_k = 0\n    !$acc end kernels" in src


def test_deferred_member_image_unpacks_on_the_device_and_frees_its_device_copy():
    src = "\n".join(_render_aos_copy_out(_becxx_member_arg(), device=True))
    assert "!$acc parallel loop collapse(2) default(present)" in src
    assert "becxx(aos_becxx_k_i0)%k(aos_becxx_k_m0, aos_becxx_k_m1) = becxx_k(aos_becxx_k_i0, aos_becxx_k_m0, aos_becxx_k_m1)" in src
    assert src.rstrip().endswith("!$acc exit data delete(becxx_k)\n    deallocate(becxx_k)")


def test_static_member_image_stays_on_the_host():
    """A fixed-shape value member is host data: the device flag changes nothing for it."""
    assert _render_aos_copy_in(_cartesian_member_arg(), device=True) == _render_aos_copy_in(_cartesian_member_arg())
