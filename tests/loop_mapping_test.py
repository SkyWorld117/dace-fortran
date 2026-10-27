"""Loop selection skips excluded analysis while retaining dependence checks."""

import dace
import pytest
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap

from dace_fortran.loop_mapping import loop_to_map_type
from dace_fortran.offload import _mapify
from dace_fortran.pipelines import optimize


def graph(carried=False):
    g = dace.SDFG("selected_loop_control")
    g.add_array("A", [100], dace.float32)
    g.add_array("B", [100], dace.float32)
    body = g.add_state("body", is_start_block=True)
    t = body.add_tasklet("copy", {"a"}, {"b"}, "b = a")
    body.add_edge(body.add_read("A"), None, t, "a", dace.Memlet("A[i]"))
    body.add_edge(t, "b", body.add_write("B"), None, dace.Memlet("B[3]" if carried else "B[i]"))
    g.add_loop(None, body, None, "i", "0", "i < 100", "i + 1")
    next(b for b in g.nodes() if isinstance(b, LoopRegion)).label = "columns"
    return g


def test_selected_loop_maps_with_the_original_dependence_check():
    g = graph()
    assert g.apply_transformations_repeated(loop_to_map_type(["col.*"]), validate=False) == 1
    g.validate()


def test_selected_recurrence_still_refused():
    g = graph(carried=True)
    assert g.apply_transformations_repeated(loop_to_map_type(["columns"]), validate=False) == 0


def test_excluded_loop_does_not_run_expensive_analysis(monkeypatch):
    calls = []
    monkeypatch.setattr(LoopToMap, "can_be_applied", lambda *a, **kw: calls.append(True) or False)
    assert graph().apply_transformations_repeated(loop_to_map_type(["other"]), validate=False) == 0
    assert calls == []


def test_none_preserves_default_and_empty_selection_keeps_loops():
    assert loop_to_map_type() is LoopToMap
    g = graph()
    _mapify(g, loop_map_patterns=[])
    assert any(isinstance(b, LoopRegion) for b in g.nodes())


def test_pipeline_and_gpu_fallback_keep_the_selection():
    g = graph()
    with pytest.raises(RuntimeError, match="no top-level map"):
        optimize(g, gpu=True, validate=False, loop_map_patterns=["other"])


@pytest.mark.parametrize("patterns", ["columns", [1], 42])
def test_invalid_selection_refused(patterns):
    with pytest.raises(ValueError):
        loop_to_map_type(patterns)


def test_recursive_mapping_retains_the_label_selection():
    child = graph()
    parent = dace.SDFG("nested_selected_control")
    parent.add_array("A", [100], dace.float32)
    parent.add_array("B", [100], dace.float32)
    state = parent.add_state()
    call = state.add_nested_sdfg(child, {"A"}, {"B"})
    state.add_edge(state.add_read("A"), None, call, "A", dace.Memlet("A[0:100]"))
    state.add_edge(call, "B", state.add_write("B"), None, dace.Memlet("B[0:100]"))
    _mapify(parent, loop_map_patterns=["other"])
    assert any(isinstance(b, LoopRegion) for b in child.nodes())
    _mapify(parent, loop_map_patterns=["columns"])
    assert not any(isinstance(b, LoopRegion) for b in child.nodes())
    parent.validate()
