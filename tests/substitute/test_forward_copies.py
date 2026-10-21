"""Copy forwarding (``fuse_stencils.forward_copy``) into a nested SDFG that reads the scratch array in an interstate
assignment, as the slopes of a scheme do: the subscripts follow the element relation like the memlets.
Run: PYTHONPATH=../../dace python -m pytest test_forward_copies.py"""
import numpy as np
import dace

from dace_fortran.substitute.fuse_stencils import forward_copies

N = 6


def _sdfg() -> dace.SDFG:
    """``q[v, x]`` copied into ``t[x, v]``, then ``out[i] = t[i + 1, 1] - t[i, 1]`` computed by an interstate
    assignment in a nested SDFG."""
    sdfg = dace.SDFG("forward_interstate")
    sdfg.add_array("q", [2, N], dace.float64)
    sdfg.add_array("out", [N - 1], dace.float64)
    sdfg.add_transient("t", [N, 2], dace.float64)
    a = sdfg.add_state("A")
    ta = a.add_access("t")
    for v in range(2):
        a.add_mapped_tasklet(f"fill{v}", {"x": f"0:{N}"}, {"a": dace.Memlet(f"q[{v}, x]")}, "b = a",
                             {"b": dace.Memlet(f"t[x, {v}]")}, external_edges=True, output_nodes={"t": ta})

    inner = dace.SDFG("slope")
    inner.add_array("tt", [N, 2], dace.float64)
    inner.add_array("o", [N - 1], dace.float64)
    inner.add_symbol("i", dace.int64)
    inner.add_symbol("d", dace.float64)
    s0 = inner.add_state("s0")
    s1 = inner.add_state("s1")
    inner.add_edge(s0, s1, dace.InterstateEdge(assignments={"d": "tt[i + 1, 1] - tt[i, 1]"}))
    t = s1.add_tasklet("put", {}, {"y"}, "y = d")
    s1.add_edge(t, "y", s1.add_write("o"), None, dace.Memlet("o[i]"))

    b = sdfg.add_state_after(a, "B")
    entry, exit_ = b.add_map("m", {"i": f"0:{N - 1}"})
    node = b.add_nested_sdfg(inner, {"tt"}, {"o"}, {"i": "i"})
    b.add_memlet_path(b.add_read("t"), entry, node, dst_conn="tt", memlet=dace.Memlet(f"t[0:{N}, 0:2]"))
    b.add_memlet_path(node, exit_, b.add_write("out"), src_conn="o", memlet=dace.Memlet(f"out[0:{N - 1}]"))
    return sdfg


def test_an_interstate_read_of_the_scratch_follows_the_relation():
    sdfg = _sdfg()
    sdfg.validate()
    assert forward_copies(sdfg, pairs=[("t", "q", True)]) == ["t -> q (2 copies)"]
    sdfg.validate()
    assert "t" not in sdfg.arrays
    inner = next(n.sdfg for st in sdfg.states() for n in st.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    assert [e.data.assignments["d"] for e in inner.edges()] == ["tt[1, i + 1] - tt[1, i]"]
    q = np.random.default_rng(3).standard_normal((2, N))
    out = np.zeros(N - 1)
    sdfg.compile()(q=q, out=out)
    np.testing.assert_allclose(out, q[1, 1:] - q[1, :-1])
