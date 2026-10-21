"""Record images laid out with the member dimensions fastest (``emit_bindings(members_first=...)``,
:mod:`dace_fortran.bindings.record_layout`): the binding allocates, packs, measures and unpacks the buffer in the
permuted order, the SDFG's descriptors (nested ones included) take the matching strides, and over several calls the
routine's results match plain gfortran, whether the buffer is kept between calls or copied back."""

from pathlib import Path

import dace
import numpy as np
import pytest

from dace_fortran.bindings.block_builders import _render_aos_copy_in, _render_aos_copy_out, buffer_dim
from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenArgKind
from dace_fortran.bindings.record_layout import lay_out_members_first, members_first_strides


def _member(**overrides) -> FrozenArg:
    """The buffer of ``acc % vf(:) % sf(:, :, :)``, a module record's array of records of 3-D pointer members."""
    fields = dict(
        fortran_name="acc_vf_sf",
        sdfg_name="acc_vf_sf",
        kind=FrozenArgKind.ARRAY,
        dtype="float64",
        rank=4,
        shape=("acc_vf_sf_d0", "acc_vf_sf_d1", "acc_vf_sf_d2", "acc_vf_sf_d3"),
        is_written=True,
        aos_origin_mod="kept_state",
        aos_origin_struct="acc % vf",
        aos_member_path="sf",
        aos_outer_rank=1,
        aos_member_pointer=True,
    )
    fields.update(overrides)
    return FrozenArg(**fields)


def test_buffer_dim_puts_the_member_dimensions_first():
    a = _member()
    assert [buffer_dim(a, d) for d in range(4)] == [1, 2, 3, 4]
    assert [buffer_dim(a, d, members_first=True) for d in range(4)] == [4, 1, 2, 3]
    two_lead = _member(rank=5, aos_root_rank=1, shape=("a", "b", "c", "d", "e"))
    assert [buffer_dim(two_lead, d, members_first=True) for d in range(5)] == [4, 5, 1, 2, 3]


def test_members_first_strides():
    d0, d1, d2, d3 = (dace.symbol(f"d{i}") for i in range(4))
    assert members_first_strides([d0, d1, d2, d3], 1) == [d1 * d2 * d3, 1, d1, d1 * d2]
    assert members_first_strides([2, 5, 10, 20, 30], 2) == [6000, 12000, 1, 10, 200]


def _line_with(text: list[str], needle: str) -> str:
    lines = [ln.strip() for ln in text if needle in ln]
    assert lines, f"no line with {needle!r} in:\n" + "\n".join(text)
    return lines[0]


def test_copy_in_allocates_packs_and_measures_in_the_permuted_order():
    a = _member()
    plain, first = _render_aos_copy_in(a, kept=True), _render_aos_copy_in(a, kept=True, members_first=True)
    bounds = "aos_acc_vf_sf_l0:aos_acc_vf_sf_u0, aos_acc_vf_sf_l1:aos_acc_vf_sf_u1, aos_acc_vf_sf_l2:aos_acc_vf_sf_u2"
    assert f"allocate(acc_vf_sf(lbound(acc % vf, 1):ubound(acc % vf, 1), {bounds}))" in _line_with(plain, "allocate(acc_vf_sf(lbound")
    assert f"allocate(acc_vf_sf({bounds}, lbound(acc % vf, 1):ubound(acc % vf, 1)))" in _line_with(first, "allocate(acc_vf_sf(aos")
    assert f"allocate(acc_vf_sf({bounds}, 1))" in _line_with(first, "allocate(acc_vf_sf(aos_acc_vf_sf_l0:aos_acc_vf_sf_u0, aos_acc_vf_sf_l1:aos_acc_vf_sf_u1, aos_acc_vf_sf_l2:aos_acc_vf_sf_u2, 1")
    # the host pack: the element's section first, its index last
    pack = _line_with(first, "= acc % vf(aos_acc_vf_sf_i0)%sf")
    assert pack.startswith("acc_vf_sf(lbound(acc % vf(aos_acc_vf_sf_i0)%sf, 1)")
    assert pack.endswith(", aos_acc_vf_sf_i0) = acc % vf(aos_acc_vf_sf_i0)%sf")
    assert _line_with(plain, "= acc % vf(aos_acc_vf_sf_i0)%sf").startswith("acc_vf_sf(aos_acc_vf_sf_i0, lbound(")
    # the kept buffer is measured where each dimension now sits
    assert "lbound(acc_vf_sf, 4) == lbound(acc % vf, 1) .and. ubound(acc_vf_sf, 4) == ubound(acc % vf, 1)" in _line_with(first, "__kept = lbound(acc_vf_sf")
    assert "lbound(acc_vf_sf, 1) == lbound(acc % vf, 1) .and. ubound(acc_vf_sf, 1) == ubound(acc % vf, 1)" in _line_with(plain, "__kept = lbound(acc_vf_sf")
    assert "lbound(acc_vf_sf, 1) == aos_acc_vf_sf_l0" in _line_with(first, "lbound(acc_vf_sf, 1) ==")
    assert "lbound(acc_vf_sf, 2) == aos_acc_vf_sf_l0" in _line_with(plain, "lbound(acc_vf_sf, 2) ==")
    assert "member dimensions first" in first[0]
    # the device pack loop
    device = _render_aos_copy_in(a, device=True, members_first=True)
    assert (
        "acc_vf_sf(aos_acc_vf_sf_m0, aos_acc_vf_sf_m1, aos_acc_vf_sf_m2, aos_acc_vf_sf_i0) = "
        "acc % vf(aos_acc_vf_sf_i0)%sf(aos_acc_vf_sf_m0, aos_acc_vf_sf_m1, aos_acc_vf_sf_m2)"
    ) in _line_with(device, "acc_vf_sf(aos_acc_vf_sf_m0")


def test_copy_out_unpacks_in_the_permuted_order():
    a = _member()
    first = _render_aos_copy_out(a, members_first=True)
    unpack = _line_with(first, "acc % vf(aos_acc_vf_sf_i0)%sf = acc_vf_sf(")
    assert unpack.endswith(", aos_acc_vf_sf_i0)")
    assert _line_with(_render_aos_copy_out(a), "acc % vf(aos_acc_vf_sf_i0)%sf = acc_vf_sf(").startswith(
        "acc % vf(aos_acc_vf_sf_i0)%sf = acc_vf_sf(aos_acc_vf_sf_i0, lbound("
    )
    device = _render_aos_copy_out(a, device=True, members_first=True)
    assert "= acc_vf_sf(aos_acc_vf_sf_m0, aos_acc_vf_sf_m1, aos_acc_vf_sf_m2, aos_acc_vf_sf_i0)" in _line_with(
        device, "%sf(aos_acc_vf_sf_m0, aos_acc_vf_sf_m1, aos_acc_vf_sf_m2) ="
    )
    inside = _render_aos_copy_out(_member(global_alloc_inside=True), members_first=True)
    assert "allocate(acc % vf(aos_acc_vf_sf_i0)%sf(lbound(acc_vf_sf, 1):ubound(acc_vf_sf, 1), " in _line_with(
        inside, "allocate(acc % vf"
    )


def _sdfg_with_nested_reader(inner_shape) -> tuple[dace.SDFG, dace.SDFG]:
    dims = [dace.symbol(f"acc_vf_sf_d{i}") for i in range(4)]
    sdfg = dace.SDFG("outer")
    sdfg.add_array("acc_vf_sf", dims, dace.float64)
    sdfg.add_array("other", dims, dace.float64)
    sdfg.add_array("y", [dims[1]], dace.float64)
    inner = dace.SDFG("inner")
    inner.add_array("buf", inner_shape, dace.float64)
    inner.add_array("out", [dims[1]], dace.float64)
    st = inner.add_state("body")
    t = st.add_tasklet("read", {"v"}, {"w"}, "w = v")
    st.add_edge(st.add_read("buf"), None, t, "v", dace.Memlet("buf[0, 0, 0, 0]"))
    st.add_edge(t, "w", st.add_write("out"), None, dace.Memlet("out[0]"))
    state = sdfg.add_state("call")
    node = state.add_nested_sdfg(inner, {"buf"}, {"out"}, symbol_mapping={str(d): d for d in dims})
    state.add_edge(state.add_read("acc_vf_sf"), None, node, "buf", dace.Memlet.from_array("acc_vf_sf", sdfg.arrays["acc_vf_sf"]))
    state.add_edge(node, "out", state.add_write("y"), None, dace.Memlet.from_array("y", sdfg.arrays["y"]))
    return sdfg, inner


def test_lay_out_members_first_sets_the_strides_down_to_nested_descriptors():
    sdfg, inner = _sdfg_with_nested_reader([dace.symbol(f"acc_vf_sf_d{i}") for i in range(4)])
    d = [dace.symbol(f"acc_vf_sf_d{i}") for i in range(4)]
    assert lay_out_members_first(sdfg, {"acc_vf_sf": 1, "other": 1, "y": 1, "absent": 1}) == ["acc_vf_sf", "other"]
    expected = [d[1] * d[2] * d[3], 1, d[1], d[1] * d[2]]
    assert list(sdfg.arrays["acc_vf_sf"].strides) == expected
    assert list(inner.arrays["buf"].strides) == expected
    assert list(sdfg.arrays["other"].strides) == expected
    sdfg.validate()


def test_lay_out_members_first_leaves_a_buffer_with_an_odd_nested_descriptor(capsys):
    sdfg, inner = _sdfg_with_nested_reader([2, 2, 2, 2])
    before = list(sdfg.arrays["acc_vf_sf"].strides)
    assert lay_out_members_first(sdfg, {"acc_vf_sf": 1}) == []
    assert list(sdfg.arrays["acc_vf_sf"].strides) == before
    assert "acc_vf_sf keeps its layout" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------------------------
# End to end (needs the bridge): the kept-state routine of kept_record_buffer_test with its image laid out members
# first, kept between calls and copied back every call.


def _build(tmp_path: Path, kept: tuple[str, ...], name: str):
    """Build the SDFG (named ``name``: two builds in one process must not share a library), emit the binding with
    the image laid out members first and kept when ``kept`` names it; returns ``(so_path, binding_path)``."""
    from dace_fortran.bindings import FlattenPlan, emit_bindings
    from tests.bindings.kept_record_buffer_test import _SRC
    from dace_fortran.bindings.fortran_interface import build_auto_interface
    from dace_fortran.bindings.record_layout import record_buffers
    from tests._util import build_sdfg

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_SRC, sdfg_dir, name=name, entry="kept_state::accumulate")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = name
    buffers = record_buffers(sdfg._frozen_signature)
    assert buffers == {"acc_vf_sf": 1}
    first = lay_out_members_first(sdfg, buffers)
    assert first == ["acc_vf_sf"]
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)
    binding = tmp_path / f"{name}_bindings.f90"
    emit_bindings(
        sdfg._frozen_signature,
        iface,
        plan,
        str(binding),
        tuple(getattr(compiled, "_sig", None) or ()),
        kept_images=kept,
        members_first=first,
    )
    return so_path, binding


@pytest.mark.parametrize("kept", [("acc_vf_sf",), ()], ids=["kept", "copied_back"])
def test_members_first_image_matches_gfortran_e2e(tmp_path: Path, kept: tuple[str, ...]):
    from tests.bindings.kept_record_buffer_test import _DRIVER, _SRC, _run
    from tests.bindings.module_global_aos_struct_test import _link_pair

    name = "accumulate_kept" if kept else "accumulate_back"
    so_path, binding = _build(tmp_path, kept, name)
    text = binding.read_text()
    assert "member dimensions first" in text
    assert "acc_vf_sf_d0 = int(size(acc_vf_sf, dim=4), c_int)" in text
    assert "acc_vf_sf_d1 = int(size(acc_vf_sf, dim=1), c_int)" in text
    assert "offset_acc_vf_sf_d1 = int(lbound(acc_vf_sf, dim=1), c_int)" in text
    driver = _DRIVER.format(
        suffix="", use_callee=f"use {name}_dace_bindings, only: {name}_dace", call=f"{name}_dace"
    )
    reference = _DRIVER.format(suffix="_ref", use_callee="use kept_state, only: accumulate", call="accumulate")
    sdfg_lib, ref_lib = _link_pair(_SRC, name, driver, reference, so_path, binding, tmp_path)

    rng = np.random.default_rng(1)
    n, calls = 3, 4
    x = np.asfortranarray(rng.standard_normal((n + 1,) * 3))
    ref_out, ref_state = _run(ref_lib.run_accumulate_ref, n, calls, True, x)
    out, state = _run(sdfg_lib.run_accumulate, n, calls, True, x)
    np.testing.assert_allclose(out, ref_out, rtol=1e-14)
    if kept:
        # The state lives in the kept buffer: the members still hold what the driver gave them.
        np.testing.assert_array_equal(state[..., 0], 10)
        np.testing.assert_array_equal(state[..., 1], 20)
    else:
        np.testing.assert_allclose(state, ref_state, rtol=1e-14)
    # Members reallocated to another shape (the buffer is set up again), and more calls after them.
    m = 5
    y = np.asfortranarray(rng.standard_normal((m + 1,) * 3))
    out, state = _run(sdfg_lib.run_accumulate, m, 2, True, y)
    ref_out, ref_state = _run(ref_lib.run_accumulate_ref, m, 2, True, y)
    np.testing.assert_allclose(out, ref_out, rtol=1e-14)
    out, _ = _run(sdfg_lib.run_accumulate, m, 2, False, y)
    ref_out, _ = _run(ref_lib.run_accumulate_ref, m, 2, False, y)
    np.testing.assert_allclose(out, ref_out, rtol=1e-14)


def test_members_first_must_be_module_record_buffers(tmp_path: Path):
    from dace_fortran.bindings import FlattenPlan, emit_bindings
    from dace_fortran.bindings.fortran_interface import build_auto_interface
    from tests._util import build_sdfg
    from tests.bindings.kept_record_buffer_test import _SRC

    builder = build_sdfg(_SRC, tmp_path / "sdfg", name="accumulate_wrong", entry="kept_state::accumulate")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "accumulate_wrong"
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)
    with pytest.raises(ValueError, match="members_first"):
        emit_bindings(sdfg._frozen_signature, iface, plan, str(tmp_path / "b.f90"), (), members_first=("x",))
