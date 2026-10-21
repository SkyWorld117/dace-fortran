"""Where the unresolved-free-symbol diagnostic points.

``find_symbol_use_sites`` reports the tasklets that use a name in their code.  A tasklet that reads the name through a
wired connector (a memlet behind it) is not a use of a free symbol and must not be reported, however early it comes in
the SDFG; the tasklet that uses the name with no connector is the culprit.  Plain ``dace`` graphs, no bridge.
"""

import dace

from dace_fortran.symbol_use_sites import find_symbol_use_sites

_NAME = "_in_x_0"


def _two_tasklets(second_inputs: set[str] | None = None) -> dace.SDFG:
    """One state: ``wired`` reads ``_in_x_0`` from ``A`` through a connector and comes first; ``stray`` reads the same
    name in its code, with the connectors given by ``second_inputs`` (none by default, nothing wired)."""
    sdfg = dace.SDFG("use_sites")
    for name in ("A", "B", "C"):
        sdfg.add_array(name, [1], dace.float64)
    state = sdfg.add_state("s")
    a = state.add_read("A")
    b = state.add_write("B")
    c = state.add_write("C")
    wired = state.add_tasklet("wired", {_NAME}, {"_out"}, f"_out = {_NAME} + 1.0")
    stray = state.add_tasklet("stray", set(second_inputs or ()), {"_out"}, f"_out = {_NAME} * 2.0")
    state.add_edge(a, None, wired, _NAME, dace.Memlet("A[0]"))
    state.add_edge(wired, "_out", b, None, dace.Memlet("B[0]"))
    state.add_edge(stray, "_out", c, None, dace.Memlet("C[0]"))
    return sdfg


def test_a_wired_connector_is_not_a_use_site_but_a_stray_name_is():
    sites = find_symbol_use_sites(_two_tasklets(), _NAME)
    assert len(sites) == 1, sites
    assert "tasklet[s::stray]" in sites[0]
    assert "wired" not in sites[0]


def test_a_connector_with_no_edge_is_still_reported():
    sdfg = _two_tasklets(second_inputs={_NAME})  # declared on ``stray`` but nothing feeds it
    sites = find_symbol_use_sites(sdfg, _NAME)
    assert [s.split(" ")[0] for s in sites] == ["tasklet[s::stray]"], sites


def test_a_wired_output_connector_is_not_a_use_site():
    sdfg = dace.SDFG("out_conn")
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state("s")
    b = state.add_write("B")
    t = state.add_tasklet("t", set(), {_NAME}, f"{_NAME} = 3.0")
    state.add_edge(t, _NAME, b, None, dace.Memlet("B[0]"))
    assert find_symbol_use_sites(sdfg, _NAME) == []


def test_the_name_matches_whole_words_only():
    sdfg = _two_tasklets()
    assert find_symbol_use_sites(sdfg, "_in_x") == []
    assert find_symbol_use_sites(sdfg, "x_0") == []


def test_the_limit_caps_the_list():
    sdfg = dace.SDFG("many")
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state("s")
    b = state.add_write("B")
    for i in range(5):
        t = state.add_tasklet(f"stray_{i}", set(), {"_out"}, f"_out = {_NAME}")
        state.add_edge(t, "_out", b, None, dace.Memlet("B[0]"))
    assert len(find_symbol_use_sites(sdfg, _NAME)) == 3
    assert len(find_symbol_use_sites(sdfg, _NAME, limit=2)) == 2
