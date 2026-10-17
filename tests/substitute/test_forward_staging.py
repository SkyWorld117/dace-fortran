"""Forwarding of staging transients (``fuse_stencils.forward_staging``): a transient each state fills by copy maps
from one array before reading it has its readers take that array directly, in every state, including a state whose
single copy leaves the pairing of the constant dims open (decided by the states of the same source that have two).
Run: PYTHONPATH=../../dace python -m pytest test_forward_staging.py"""
import numpy as np
import dace
from dace import nodes

from dace_fortran.substitute.fuse_stencils import forward_copies, forward_staging

N = 6


def _copy(state, name, src, dst, node):
    """``dst = src`` element by element over ``i`` (a copy map as the frontend emits them), into the access node
    ``node`` of the staging array, which its readers then take."""
    state.add_mapped_tasklet(name, {"i": f"0:{N}"}, {"a": dace.Memlet(src)}, "b = a", {"b": dace.Memlet(dst)},
                             external_edges=True, output_nodes={"t": node})


def _staged_sdfg(second_state: bool = True, covered: bool = True) -> dace.SDFG:
    """``q[v, x, z]`` (two components, z of extent 1) staged into ``t[x, z, v]``: state A copies both components and
    reads the slab, state B copies component 1 only and reads it; ``p`` is a second source for state C."""
    sdfg = dace.SDFG("staged")
    sdfg.add_array("q", [2, N, 1], dace.float64)
    sdfg.add_array("p", [2, N, 1], dace.float64)
    sdfg.add_array("out_a", [N], dace.float64)
    sdfg.add_array("out_b", [N], dace.float64)
    sdfg.add_array("out_c", [N], dace.float64)
    sdfg.add_transient("t", [N, 1, 2], dace.float64)
    a = sdfg.add_state("A")
    ta = a.add_access("t")
    _copy(a, "fill_a0", "q[0, i, 0]", "t[i, 0, 0]", ta)
    _copy(a, "fill_a1", "q[1, i, 0]", "t[i, 0, 1]", ta)
    a.add_mapped_tasklet("use_a", {"i": f"0:{N}"}, {"x": dace.Memlet("t[i, 0, 0:2]")}, "y = x[0] + 2 * x[1]",
                         {"y": dace.Memlet("out_a[i]")}, external_edges=True, input_nodes={"t": ta})
    last = a
    if second_state:
        b = sdfg.add_state_after(last, "B")
        tb = b.add_access("t")
        _copy(b, "fill_b1", "q[1, i, 0]", "t[i, 0, 1]", tb)
        read = "t[i, 0, 1]" if covered else "t[i, 0, 0]"
        b.add_mapped_tasklet("use_b", {"i": f"0:{N}"}, {"x": dace.Memlet(read)}, "y = 3 * x",
                             {"y": dace.Memlet("out_b[i]")}, external_edges=True, input_nodes={"t": tb})
        last = b
    c = sdfg.add_state_after(last, "C")
    tc = c.add_access("t")
    _copy(c, "fill_c0", "p[0, i, 0]", "t[i, 0, 0]", tc)
    _copy(c, "fill_c1", "p[1, i, 0]", "t[i, 0, 1]", tc)
    c.add_mapped_tasklet("use_c", {"i": f"0:{N}"}, {"x": dace.Memlet("t[i, 0, 0:2]")}, "y = x[1] - x[0]",
                         {"y": dace.Memlet("out_c[i]")}, external_edges=True, input_nodes={"t": tc})
    return sdfg


def _run(sdfg: dace.SDFG):
    rng = np.random.default_rng(0)
    q, p = rng.standard_normal((2, N, 1)), rng.standard_normal((2, N, 1))
    outs = {k: np.zeros(N) for k in ("out_a", "out_b", "out_c")}
    sdfg.compile()(q=q, p=p, **outs)
    return q, p, outs


def _copies(sdfg: dace.SDFG) -> int:
    return sum(1 for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.Tasklet) and n.code.as_string.strip() == "b = a")


def test_every_state_reads_its_source_directly():
    sdfg = _staged_sdfg()
    sdfg.validate()
    q, p, before = _run(sdfg)
    assert _copies(sdfg) == 5
    report = forward_staging(sdfg)
    assert report == ["t staged: 3 of 3 states read ['p', 'q'] directly"]
    sdfg.validate()
    assert "t" not in sdfg.arrays and _copies(sdfg) == 0
    reads = {st.label: sorted(str(e.data) for e in st.edges() if e.data.data in ("q", "p") and isinstance(e.dst, nodes.MapEntry))
             for st in sdfg.states()}
    assert reads == {"A": ["q[0:2, 0:6, 0]"], "B": ["q[1, 0:6, 0]"], "C": ["p[0:2, 0:6, 0]"]}
    _q, _p, after = _run(sdfg)
    for k in before:
        np.testing.assert_array_equal(after[k], before[k])
    np.testing.assert_allclose(after["out_a"], q[0, :, 0] + 2 * q[1, :, 0])
    np.testing.assert_allclose(after["out_b"], 3 * q[1, :, 0])


def test_a_read_outside_the_fills_keeps_the_staging():
    sdfg = _staged_sdfg(covered=False)  # state B reads component 0, which it never filled
    assert forward_staging(sdfg) == []
    assert "t" in sdfg.arrays and _copies(sdfg) == 5


def test_a_reader_without_fills_keeps_the_staging():
    sdfg = _staged_sdfg(second_state=False)
    d = sdfg.add_state_after(list(sdfg.states())[-1], "D")
    d.add_mapped_tasklet("use_d", {"i": f"0:{N}"}, {"x": dace.Memlet("t[i, 0, 1]")}, "y = x",
                         {"y": dace.Memlet("out_b[i]")}, external_edges=True)
    assert forward_staging(sdfg) == []
    assert "t" in sdfg.arrays


def test_a_reader_is_ordered_after_the_source_writer_of_its_state():
    """The state also writes the source before staging it: the forwarded reader must still run after that writer."""
    sdfg = _staged_sdfg(second_state=False)
    a = sdfg.start_state
    qw = a.add_access("q")
    a.add_mapped_tasklet("make_q", {"i": f"0:{N}"}, {}, "v = 1.0", {"v": dace.Memlet("q[1, i, 0]")},
                         external_edges=True, output_nodes={"q": qw})
    reader = next(n for n in a.nodes() if isinstance(n, nodes.AccessNode) and n.data == "t")
    for e in list(a.in_edges(reader)):  # the fills read q after make_q wrote it
        src_node = next(x.src for x in a.in_edges(a.entry_node(e.src)) if x.data.data == "q")
        if not any(x.dst is src_node for x in a.out_edges(qw)):
            a.add_nedge(qw, src_node, dace.Memlet())
    sdfg.validate()
    assert forward_staging(sdfg)[0].startswith("t staged: 2 of 2")
    sdfg.validate()
    readers = [n for n in a.nodes() if isinstance(n, nodes.AccessNode) and n.data == "q"
               and any(isinstance(e.dst, nodes.MapEntry) for e in a.out_edges(n))]
    assert len(readers) == 1 and any(e.src is qw for e in a.in_edges(readers[0]))
    _q, _p, outs = _run(sdfg)
    np.testing.assert_allclose(outs["out_a"], _q[0, :, 0] + 2 * 1.0)


def test_forward_copies_runs_the_staging_pass_last():
    sdfg = _staged_sdfg()
    assert forward_copies(sdfg)[-1].startswith("t staged: 3 of 3 states")


def test_a_pinned_dim_does_not_leave_the_pairing_open():
    """``q[d, v, x, z]`` has a direction dim fixed at 0 in every copy and ``t`` a z dim fixed at 0 everywhere: two
    copies pair v with v but leave z's partner (d or z) open -- the same elements either way."""
    sdfg = dace.SDFG("pinned")
    sdfg.add_array("q", [1, 2, N, 1], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("t", [N, 1, 2], dace.float64)
    a = sdfg.add_state("A")
    ta = a.add_access("t")
    _copy(a, "fill0", "q[0, 0, i, 0]", "t[i, 0, 0]", ta)
    _copy(a, "fill1", "q[0, 1, i, 0]", "t[i, 0, 1]", ta)
    a.add_mapped_tasklet("use", {"i": f"0:{N}"}, {"x": dace.Memlet("t[i, 0, 0:2]")}, "y = x[0] - x[1]",
                         {"y": dace.Memlet("out[i]")}, external_edges=True, input_nodes={"t": ta})
    sdfg.validate()
    assert forward_staging(sdfg) == ["t staged: 1 of 1 states read ['q'] directly"]
    sdfg.validate()
    assert [str(e.data) for e in a.edges() if e.data.data == "q" and isinstance(e.dst, nodes.MapEntry)] == ["q[0, 0:2, 0:6, 0]"]
    q = np.random.default_rng(1).standard_normal((1, 2, N, 1))
    out = np.zeros(N)
    sdfg.compile()(q=q, out=out)
    np.testing.assert_allclose(out, q[0, 0, :, 0] - q[0, 1, :, 0])


def test_the_recorded_extents_decide_a_pairing_a_copy_per_variable_leaves_open():
    """``t[x, z, v]`` (three variables, z of extent 1) is written back into ``f[d, v, x, z]`` of symbolic extents by
    a copy per variable, each in its own state: the single copy of a state fits v with d, v or z of ``f`` (every
    constant dim). The recorded extents (d 2, v 3, z 1) leave only v: a scratch dim cannot supply a real dim of
    fewer elements."""
    sdfg = dace.SDFG("mirrored")
    d0, d1, d3 = (dace.symbol(s) for s in ("f_d0", "f_d1", "f_d3"))
    sdfg.add_array("f", [d0, d1, N, d3], dace.float64)
    sdfg.add_array("q", [N, 3], dace.float64)
    sdfg.add_transient("t", [N, 1, 3], dace.float64)
    prev = None
    for v in range(3):
        st = sdfg.add_state(f"S{v}") if prev is None else sdfg.add_state_after(prev, f"S{v}")
        tw = st.add_access("t")
        st.add_mapped_tasklet(f"compute{v}", {"i": f"0:{N}"}, {"a": dace.Memlet(f"q[i, {v}]")}, f"b = a * {v + 1}",
                              {"b": dace.Memlet(f"t[i, 0, {v}]")}, external_edges=True, output_nodes={"t": tw})
        st.add_mapped_tasklet(f"copy{v}", {"i": f"0:{N}"}, {"a": dace.Memlet(f"t[i, 0, {v}]")}, "b = a",
                              {"b": dace.Memlet(f"f[0, {v}, i, 0]")}, external_edges=True, input_nodes={"t": tw})
        prev = st
    sdfg.validate()
    known = {"f_d0": 2, "f_d1": 3, "f_d3": 1}
    assert forward_copies(sdfg, pairs=[("t", "f", False)]) == ["t kept: t and f: 3 ways to pair the constant dims"]
    assert forward_copies(sdfg, pairs=[("t", "f", False)], known=known) == ["t -> f (3 copies)"]
    sdfg.validate()
    assert "t" not in sdfg.arrays
    q = np.random.default_rng(2).standard_normal((N, 3))
    f = np.zeros((2, 3, N, 1))
    sdfg.compile()(q=q, f=f, f_d0=2, f_d1=3, f_d3=1)
    np.testing.assert_allclose(f[0, :, :, 0], (q * np.array([1, 2, 3])).T)
