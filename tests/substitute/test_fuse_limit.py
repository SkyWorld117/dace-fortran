"""fuse_stencil_pairs' tasklet limit: a stencil pair whose producer and consumer bodies hold more tasklets together
than the limit stays two kernels."""
import dace

from dace_fortran.substitute.fuse_stencils import body_tasklets, fuse_stencil_pairs


def _body(name: str, inputs: dict, output: tuple, code: list[str]) -> dace.SDFG:
    """A map body as a nested SDFG: one tasklet per line of ``code`` chained through scalars, reading ``inputs``
    (connector -> memlet) and writing ``output`` (array, memlet)."""
    body = dace.SDFG(name)
    for array in {m.split("[")[0] for m in inputs.values()} | {output[0]}:
        body.add_array(array, [16], dace.float64)
    body.add_symbol("i", dace.int64)
    st = body.add_state(is_start_block=True)
    tasklet = st.add_tasklet(f"{name}0", set(inputs), {"o"}, code[0])
    for conn, memlet in inputs.items():
        st.add_edge(st.add_read(memlet.split("[")[0]), None, tasklet, conn, dace.Memlet(memlet))
    for k, line in enumerate(code[1:], start=1):
        body.add_scalar(f"s{k}", dace.float64, transient=True)
        s_k = st.add_access(f"s{k}")
        st.add_edge(tasklet, "o", s_k, None, dace.Memlet(f"s{k}[0]"))
        tasklet = st.add_tasklet(f"{name}{k}", {"x"}, {"o"}, line)
        st.add_edge(s_k, None, tasklet, "x", dace.Memlet(f"s{k}[0]"))
    st.add_edge(tasklet, "o", st.add_write(output[0]), None, dace.Memlet(output[1]))
    return body


def _stencil_pair() -> dace.SDFG:
    """tmp[i] = 2 * a[i] + 1 over 1..14 (two tasklets), then b[j] = tmp[j + 1] - 1 over 1..13 (one)."""
    sdfg = dace.SDFG("stencil_pair")
    sdfg.add_array("a", [16], dace.float64)
    sdfg.add_array("b", [16], dace.float64)
    sdfg.add_transient("tmp", [16], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    a, tmp, b = state.add_read("a"), state.add_access("tmp"), state.add_write("b")
    pentry, pexit = state.add_map("produce", {"i": "1:15"})
    produce = state.add_nested_sdfg(_body("produce", {"x": "a[i]"}, ("tmp", "tmp[i]"), ["o = 2 * x", "o = x + 1"]),
                                    {"a"}, {"tmp"}, {"i": "i"})
    state.add_memlet_path(a, pentry, produce, dst_conn="a", memlet=dace.Memlet("a[0:16]"))
    state.add_memlet_path(produce, pexit, tmp, src_conn="tmp", memlet=dace.Memlet("tmp[i]"))
    centry, cexit = state.add_map("consume", {"j": "1:14"})
    consume = state.add_nested_sdfg(_body("consume", {"r": "tmp[i + 1]"}, ("b", "b[i]"), ["o = r - 1"]),
                                    {"tmp"}, {"b"}, {"i": "j"})
    state.add_memlet_path(tmp, centry, consume, dst_conn="tmp", memlet=dace.Memlet("tmp[j + 1]"))
    state.add_memlet_path(consume, cexit, b, src_conn="b", memlet=dace.Memlet("b[j]"))
    return sdfg


def _kernels(sdfg: dace.SDFG) -> int:
    """The top-level maps of the SDFG's one state."""
    scopes = sdfg.start_state.scope_dict()
    return sum(1 for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.MapEntry) and scopes[n] is None)


def test_a_pair_over_the_limit_stays_apart_and_one_under_it_fuses():
    kept = _stencil_pair()
    state = kept.start_state
    sizes = sorted(body_tasklets(state, n) for n in state.nodes() if isinstance(n, dace.nodes.MapEntry))
    assert sizes == [1, 2]
    assert fuse_stencil_pairs(kept, max_tasklets=2) == ["tmp (kept apart: 3 tasklets)"]
    assert _kernels(kept) == 2
    fused = _stencil_pair()
    assert fuse_stencil_pairs(fused, max_tasklets=3) == ["tmp (recompute)"]
    assert _kernels(fused) == 1
