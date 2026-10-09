"""Recognition tests for ``hlfir-lift-aos-pointer-records``: pins that the
AoS-of-pointer-records pattern emits ``hlfir.aos_ptr_records.<aos_decl>``, ahead of materialisation.
"""

import subprocess
from pathlib import Path

import pytest

from dace_fortran.build import make_builder
from dace_fortran.build_bridge import hb
from dace_fortran.builder import DEFAULT_PIPELINE
from tests._util import flang_binary, flang_intrinsic_modules_path

_HERE = Path(__file__).resolve().parent


def _post_pass_module(probe_name: str, entry: str, tmp_path):
    """Return the post-pass HLFIR module text for assertion."""
    src = (_HERE / probe_name).read_text()
    builder = make_builder(src, entry=entry, name=entry, out_dir=str(tmp_path))
    return builder.module.dump()


def _has_attr(module_str: str, attr_substring: str) -> bool:
    """The matcher emits ``hlfir.aos_ptr_records.<uniq_name> = [...]``."""
    return any(attr_substring in line for line in module_str.splitlines())


def test_single_pointer_member_recognised(tmp_path):
    mod = _post_pass_module("aos_single_pointer_member_probe.f90", "m::run", tmp_path)
    assert _has_attr(mod, "hlfir.aos_ptr_records."), (
        "matcher did not emit the recognition attribute for single-member shape"
    )


def test_two_pointer_members_recognised(tmp_path):
    """Two pointer members of different rank both surface as rebind entries."""
    mod = _post_pass_module("aos_two_pointer_members_probe.f90", "m::run", tmp_path)
    assert _has_attr(mod, "hlfir.aos_ptr_records."), (
        "matcher did not emit the recognition attribute for two-member shape"
    )
    # both members should appear in the rebind table
    assert 'member = "p"' in mod and 'member = "x"' in mod, "two-member rebind table missing one of the member entries"


def test_runtime_index_recognised(tmp_path):
    """Runtime-index reads don't change matching: the matcher is rebind-driven."""
    mod = _post_pass_module("aos_runtime_index_probe.f90", "m::run", tmp_path)
    assert _has_attr(mod, "hlfir.aos_ptr_records.")


def test_write_through_pointer_recognised(tmp_path):
    """Writes through the alias don't affect matching either."""
    mod = _post_pass_module("aos_write_through_pointer_probe.f90", "m::run", tmp_path)
    assert _has_attr(mod, "hlfir.aos_ptr_records.")


def _materialisation_landed(module_str: str) -> bool:
    """The materialisation step injects a concat declare named ``_QXaos_lift_*``."""
    return any("_QXaos_lift_" in line for line in module_str.splitlines())


def test_materialisation_emits_concat_declare(tmp_path):
    """The pass allocates a per-member concat transient when materialisation fires.

    The recognised candidate's pointer member gets a top-level
    ``hlfir.declare`` whose uniq_name carries the ``_QXaos_lift_<aos>_<member>_<n>``
    convention.  This verifies the pass progresses past the recognition stage
    into the actual IR rewrite.
    """
    mod = _post_pass_module("aos_single_pointer_member_probe.f90", "m::run", tmp_path)
    assert _materialisation_landed(mod), (
        "expected an `_QXaos_lift_` concat declare in the post-pass module; the materialisation step did not emit one"
    )


def test_materialisation_handles_two_members(tmp_path):
    """Both pointer members get their own concat transient."""
    mod = _post_pass_module("aos_two_pointer_members_probe.f90", "m::run", tmp_path)
    concat_lines = [ln for ln in mod.splitlines() if "_QXaos_lift_" in ln]
    # 2 concat declares (one per member)
    assert sum(1 for ln in concat_lines if "hlfir.declare" in ln) >= 2, (
        f"expected 2 concat declares; got: {concat_lines!r}"
    )


def _concat_declares(module_str: str) -> list[str]:
    """The concat ``hlfir.declare`` lines the materialisation emits."""
    return [ln for ln in module_str.splitlines() if "_QXaos_lift_" in ln and "hlfir.declare" in ln]


def test_concat_keeps_member_lower_bounds(tmp_path):
    """A member whose targets start at (0, -2) gets a concat declared with a ``fir.shape_shift``; a member that starts
    at 1 keeps the plain ``fir.shape``."""
    shifted = _concat_declares(_post_pass_module("aos_lower_bound_target_probe.f90", "m::run", tmp_path / "shifted"))
    assert shifted and all("!fir.shapeshift<3>" in ln for ln in shifted), shifted
    plain = _concat_declares(_post_pass_module("aos_single_pointer_member_probe.f90", "m::run", tmp_path / "plain"))
    assert plain and all("!fir.shape<3>" in ln for ln in plain), plain


def _lifted_module(probe_name: str, tmp_path) -> str:
    """The module text right after ``hlfir-lift-aos-pointer-records``: the pipeline up to it, without the bridge."""
    hlfir = tmp_path / "probe.hlfir"
    subprocess.check_call(
        [
            flang_binary(),
            "-fc1",
            "-fintrinsic-modules-path",
            flang_intrinsic_modules_path(),
            "-emit-hlfir",
            str(_HERE / probe_name),
            "-o",
            str(hlfir),
        ],
        cwd=tmp_path,
    )
    mod = hb.HLFIRModule()
    assert mod.parse_file(str(hlfir))
    head, lift, _ = DEFAULT_PIPELINE.partition("hlfir-lift-aos-pointer-records")
    mod.run_passes(head + lift)
    return mod.dump()


def test_unrepresentable_rebinds_declined(tmp_path):
    """Rebinds one gather temp cannot represent are recognised but not lifted: a member pointing at part of its
    target, members whose targets start at different lower bounds, and a runtime remap bound computed after another
    member's rebind.  An array of records starting at 0 is not even recognised."""
    mod = _lifted_module("aos_unliftable_bounds_probe.f90", tmp_path)
    for entry in ("part", "mixed", "late_bound"):
        assert _has_attr(mod, f"hlfir.aos_ptr_records._QMmF{entry}Eq"), f"{entry}: not recognised"
        assert f"_QXaos_lift__QMmF{entry}Eq" not in mod, f"{entry}: lifted"
    assert not _has_attr(mod, "hlfir.aos_ptr_records._QMmFoffset_aosEq"), "offset_aos: recognised"
    assert "_QXaos_lift__QMmFoffset_aosEq" not in mod, "offset_aos: lifted"


@pytest.mark.parametrize("entry", ["m::part", "m::mixed", "m::offset_aos", "m::late_bound"])
def test_unrepresentable_rebinds_rejected_by_bridge(entry, tmp_path):
    """A rebind the pass leaves alone reaches the bridge, which rejects the component pointer assignment instead of
    lowering a copy that reads the wrong elements."""
    with pytest.raises(NotImplementedError, match=r"pointer assignment to the component '\w+' of an array element"):
        _post_pass_module("aos_unliftable_bounds_probe.f90", entry, tmp_path)


def test_allocatable_member_aos_not_matched(tmp_path):
    """An AoS whose member is ALLOCATABLE (``fir.heap``), not POINTER
    (``fir.ptr``), must NOT be matched by ``hlfir-lift-aos-pointer-records``.

    The pass lifts the ``q(c)%m => target`` pointer-rebind pattern; an
    allocatable member is never rebound with ``=>`` -- it is allocated with
    ``allocate(a(i)%w(...))`` and belongs to flatten-structs' Phase 5c-A
    padded-companion path.  Matching it here fires the ``rebinds.empty()``
    "dead exchange scaffolding" branch, which mints a bogus size-1 placeholder
    companion ``_QXaos_lift_..._w_0`` (shape ``(N, 1)``) and rewrites the live
    ``a(i)%w(j)`` reads onto it -- producing an out-of-bounds
    ``a_w_0[1, 1]`` memlet (and, at an SDFG boundary, an unbound
    ``a_w_0_d0`` free symbol).  Regression guard for that double-companion
    bug: assert neither the recognition attribute nor an ``_QXaos_lift_``
    companion is emitted for the allocatable-member shape.
    """
    mod = _post_pass_module("aos_allocatable_member_probe.f90", "m::run", tmp_path)
    assert not _has_attr(mod, "hlfir.aos_ptr_records."), (
        "matcher wrongly recognised an allocatable-member AoS as pointer-records"
    )
    assert not _materialisation_landed(mod), (
        "matcher wrongly minted an _QXaos_lift_ companion for an allocatable-member AoS"
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
