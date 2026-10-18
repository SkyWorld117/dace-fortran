"""Tests of the scratch inference end to end on a small SDFG and its program's sources (CPU).
Run: python -m pytest test_auto_scratch.py"""
from dataclasses import replace
from pathlib import Path

import dace
from dace_fortran.bindings.flatten_plan import SyntheticGlobal
from dace_fortran.bindings.frozen_signature import (FrozenArg, FrozenArgKind, FrozenSignature, ModuleOrigin,
                                                    attach_to_sdfg, get_frozen_signature)

from dace_fortran.substitute.auto_scratch import candidates, infer, scratch_keys
from .test_outside_refs import _FIELDS, _MAIN, _STEPPER

N = 10


def _copy(state, src: str, dst: str):
    state.add_mapped_tasklet("c", {"i": f"0:{N}"}, {"x": dace.Memlet(f"{src}[i]")}, "y = x",
                             {"y": dace.Memlet(f"{dst}[i]")}, external_edges=True)


def _routine() -> dace.SDFG:
    """s_rhs as an SDFG: work = q; kept = work; q = kept; pb%sf = q; q = pb%sf; q = pa%sf (read only)."""
    sdfg = dace.SDFG("s_rhs")
    for name in ("q", "work", "kept", "pa_sf", "pb_sf"):
        sdfg.add_array(name, [N], dace.float64)
    prev = None
    for src, dst in (("q", "work"), ("work", "kept"), ("kept", "q"), ("q", "pb_sf"), ("pb_sf", "q"), ("pa_sf", "q")):
        st = sdfg.add_state(is_start_block=prev is None) if prev is None else sdfg.add_state_after(prev)
        _copy(st, src, dst)
        prev = st

    def image(member: str) -> FrozenArg:
        return FrozenArg(f"{member}%sf", f"{member}_sf", FrozenArgKind.ARRAY, "float64", 1, aos_origin_mod="m_fields",
                         aos_origin_struct=member, aos_member_path="sf")

    attach_to_sdfg(sdfg, FrozenSignature(
        "s_rhs", "_QMm_fieldsPs_rhs",
        (FrozenArg("q", "q", FrozenArgKind.ARRAY, "float64", 1, intent="inout"), image("pa"), image("pb")),
        module_symbol_origins={"work": ModuleOrigin("m_fields", "work"), "kept": ModuleOrigin("m_fields", "kept")}))
    return sdfg


def _sources(tmp_path: Path) -> Path:
    for i, text in enumerate((_FIELDS, _STEPPER, _MAIN)):
        (tmp_path / f"f{i}.fpp.f90").write_text(text)
    return tmp_path


def test_candidates_are_module_arrays_and_record_images():
    assert candidates(_routine()) == {"m_fields::work": ["work"], "m_fields::kept": ["kept"],
                                      "m_fields::pa": ["pa_sf"], "m_fields::pb": ["pb_sf"]}


def test_inference_combines_the_call_and_the_program(tmp_path: Path):
    result = infer(_routine(), "m_fields::s_rhs", None, [_sources(tmp_path)])
    assert result.scratch == ["m_fields::work"]
    assert "s_step" in result.reasons["m_fields::kept"][0]  # read by the time stepper after the call
    assert "pa_sf" in result.reasons["m_fields::pa"][0]  # read before written
    assert "m_fields::pa" in result.reasons["m_fields::pb"][0]  # aliased with pa, which is not scratch
    assert "not scratch: m_fields::kept" in result.report()


def test_a_variable_given_as_scratch_vouches_for_its_aliases(tmp_path: Path):
    result = infer(_routine(), "m_fields::s_rhs", None, [_sources(tmp_path)], asserted=["m_fields::pa"])
    assert result.scratch == ["m_fields::pb", "m_fields::work"]
    assert result.asserted == ["m_fields::pa"]
    assert "given, but not provably scratch: m_fields::pa: pa_sf" in result.report()


def test_without_sources_nothing_is_scratch():
    result = infer(_routine(), "m_fields::s_rhs", None, [])
    assert result.scratch == []
    assert "--fypp" in result.reasons["m_fields::work"][0]


def test_the_analysis_runs_on_the_specialised_copy(tmp_path: Path):
    """The specialisation callback gets a copy: the SDFG the build goes on with is untouched."""
    sdfg = _routine()
    seen = []

    def specialise(copy: dace.SDFG):
        seen.append(copy)
        return {}

    infer(sdfg, "m_fields::s_rhs", specialise, [_sources(tmp_path)])
    assert seen and seen[0] is not sdfg


def test_flattened_record_members_count_under_their_record(tmp_path: Path):
    """``kept_beg`` is the frontend's flattening of a member of the record ``kept`` (made one here): it is judged as
    ``kept``, which the time stepper reads, and make_scratch_transient would match it by its own origin."""
    sdfg = _routine()
    sdfg.add_array("kept_beg", [N], dace.float64)
    frozen = get_frozen_signature(sdfg)
    attach_to_sdfg(sdfg, replace(frozen, module_symbol_origins={**frozen.module_symbol_origins,
                                                                "kept_beg": ModuleOrigin("m_fields", "kept_beg")}))
    synthetic = [SyntheticGlobal("_QMm_fieldsEkept_beg", "m_fields", "kept", "beg", "float64")]
    assert candidates(sdfg, synthetic)["m_fields::kept"] == ["kept", "kept_beg"]
    assert "m_fields::kept_beg" in candidates(sdfg)  # without the plan: a variable of that name
    result = infer(sdfg, "m_fields::s_rhs", None, [_sources(tmp_path)])
    assert "no variable kept_beg declared" in " ".join(result.reasons["m_fields::kept_beg"])
    result = infer(sdfg, "m_fields::s_rhs", None, [_sources(tmp_path)], synthetic=synthetic)
    assert "m_fields::kept" not in result.scratch
    assert scratch_keys(sdfg, ["kept_beg", "pb_sf", "work"]) == ["m_fields::kept_beg", "m_fields::pb", "m_fields::work"]


def test_variables_the_specialised_sdfg_never_accesses_are_left_to_the_build(tmp_path: Path):
    """The specialisation drops every access to ``work``: nothing to gain from making it scratch."""
    def specialise(copy: dace.SDFG):
        for st in copy.states():
            if any(n.data == "work" for n in st.data_nodes()):
                st.remove_nodes_from(list(st.nodes()))
        return {}

    result = infer(_routine(), "m_fields::s_rhs", specialise, [_sources(tmp_path)])
    assert result.scratch == []
    assert "not accessed" in result.reasons["m_fields::work"][0]


def test_variables_whose_shared_storage_the_sdfg_holds_as_one_are_no_alias_group(tmp_path: Path):
    """pb is scratch once the specialised copy accesses the storage it shares with pa through one image only (a
    folded member alias); pa, read before it is written, still is not."""
    result = infer(_routine(), "m_fields::s_rhs", None, [_sources(tmp_path)], associated=[("pa_sf", 0, "pb_sf", 0)])
    assert result.scratch == ["m_fields::pb", "m_fields::work"]
    assert "pa_sf" in result.reasons["m_fields::pa"][0]
