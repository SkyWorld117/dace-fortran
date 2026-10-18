"""Tests of small_kernels (CPU).  Run: python -m pytest test_small_kernels.py"""
import numpy as np

import dace
from dace.sdfg import nodes

from dace_fortran.substitute.small_kernels import fuse_small_maps, host_maps_sequential, host_small_work, run_sequentially

N = 1000


def _record_and_field(field_reads_member: bool) -> dace.SDFG:
    """State 1: ``mem[i] = rec[1, i]`` (a map of 3) and ``cnt[0] = rec[0, 0]`` (a free tasklet), then a copy
    ``out_mem = mem``; state 2: ``B[i] = A[i] * 2`` over the field, plus ``mem[0]`` if ``field_reads_member``."""
    sdfg = dace.SDFG("record_and_field" + ("_shared" if field_reads_member else ""))
    sdfg.add_array("rec", [3, 3], dace.float64)
    sdfg.add_array("mem", [3], dace.float64)
    sdfg.add_array("out_mem", [3], dace.float64)
    sdfg.add_array("cnt", [1], dace.float64)
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    s1 = sdfg.add_state("members", is_start_block=True)
    rec = s1.add_read("rec")
    mem = s1.add_access("mem")
    s1.add_mapped_tasklet("copy_member", {"i": "0:3"}, {"a": dace.Memlet("rec[1, i]")}, "b = a",
                          {"b": dace.Memlet("mem[i]")}, input_nodes={rec}, output_nodes={mem}, external_edges=True)
    s1.add_nedge(mem, s1.add_write("out_mem"), dace.Memlet("mem[0:3]"))
    t = s1.add_tasklet("count", {"a"}, {"b"}, "b = a")
    s1.add_edge(rec, None, t, "a", dace.Memlet("rec[0, 0]"))
    s1.add_edge(t, "b", s1.add_write("cnt"), None, dace.Memlet("cnt[0]"))
    s2 = sdfg.add_state_after(s1, "field")
    inputs = {"a": dace.Memlet("A[i]")}
    code = "b = a * 2"
    if field_reads_member:
        inputs["m"] = dace.Memlet("mem[0]")
        code = "b = a * 2 + m"
    s2.add_mapped_tasklet("scale", {"i": f"0:{N}"}, inputs, code, {"b": dace.Memlet("B[i]")}, external_edges=True)
    sdfg.validate()
    return sdfg


def test_small_record_work_stays_on_the_host():
    sdfg = _record_and_field(field_reads_member=False)
    maps, data = host_small_work(sdfg, 64)
    labels = [n.map.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) and n.guid in maps]
    assert labels == ["copy_member_map"]
    assert data == {"rec", "mem", "cnt"}


def test_small_work_whose_data_a_kernel_reads_is_device_work():
    """``mem`` is read by the field kernel: the member copy runs on the device next to it, and so does the
    tasklet reading ``rec``, which the member copy reads too."""
    sdfg = _record_and_field(field_reads_member=True)
    assert host_small_work(sdfg, 64) == ([], set())


def test_host_small_work_off():
    assert host_small_work(_record_and_field(False), 0) == ([], set())


def test_small_work_left_on_the_host_by_the_gpu_transformation():
    """With the maps and data passed on, no device copy of the record is made and the member copy is no kernel;
    the field is copied to the device and scaled there."""
    sdfg = _record_and_field(field_reads_member=False)
    maps, data = host_small_work(sdfg, 64)
    assert run_sequentially(sdfg, maps) == 1
    host = [n for n, d in sdfg.arrays.items() if not d.transient and n not in ("A", "B")]
    sdfg.apply_gpu_transformations(host_data=host + sorted(data - set(host)), host_maps=maps)
    sdfg.validate()
    assert not any(name in sdfg.arrays for name in ("gpu_rec", "gpu_mem", "gpu_cnt", "gpu_out_mem"))
    schedules = {n.map.label: n.map.schedule for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}
    assert schedules["copy_member_map"] == dace.ScheduleType.Sequential
    assert "#pragma omp" not in sdfg.generate_code()[0].clean_code  # no OpenMP runtime for the host side
    assert schedules["scale_map"] == dace.ScheduleType.GPU_Device
    assert not any(label.endswith("_gmap") for label in schedules)





def test_maps_outside_device_code_run_sequentially():
    """A host map left after the GPU transformation (the expansion of a copy of flang's six-element constant
    into a host buffer) becomes sequential, so no OpenMP loop; a map inside a kernel keeps its schedule."""
    sdfg = dace.SDFG("host_and_device_maps")
    sdfg.add_array("ro", [6], dace.int32)
    sdfg.add_array("perm", [6], dace.int32)
    sdfg.add_array("A", [N, 4], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("B", [N, 4], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    state.add_mapped_tasklet("copy", {"k": "0:6"}, {"a": dace.Memlet("ro[k]")}, "b = a", {"b": dace.Memlet("perm[k]")},
                             external_edges=True)
    entry, exit_ = state.add_map("scale", {"i": f"0:{N}"}, schedule=dace.ScheduleType.GPU_Device)
    ientry, iexit = state.add_map("row", {"j": "0:4"})
    t = state.add_tasklet("scale", {"a"}, {"b"}, "b = a * 2")
    state.add_memlet_path(state.add_read("A"), entry, ientry, t, dst_conn="a", memlet=dace.Memlet("A[i, j]"))
    state.add_memlet_path(t, iexit, exit_, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i, j]"))
    sdfg.validate()
    assert host_maps_sequential(sdfg) == 1
    schedules = {n.map.label: n.map.schedule for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)}
    assert schedules == {"copy_map": dace.ScheduleType.Sequential, "scale": dace.ScheduleType.GPU_Device,
                         "row": dace.ScheduleType.Default}
    assert "#pragma omp" not in sdfg.generate_code()[0].clean_code

def _face_copies(fields: int = 4, face: int = 10, volume: bool = True) -> dace.SDFG:
    """One state per field: ``F[f, i, j] = X[i, j, f] + f`` over a face; then, if ``volume``, a state with a
    map over the volume (not small) and one with a second, independent volume map of the same range."""
    sdfg = dace.SDFG("face_copies")
    sdfg.add_array("X", [face, face, fields], dace.float64)
    sdfg.add_array("F", [fields, face, face], dace.float64)
    sdfg.add_array("V", [N], dace.float64)
    sdfg.add_array("W", [N], dace.float64)
    sdfg.add_array("U", [N], dace.float64)
    prev = None
    for f in range(fields):
        st = sdfg.add_state(f"field_{f}", is_start_block=prev is None)
        if prev is not None:
            sdfg.add_edge(prev, st, dace.InterstateEdge())
        st.add_mapped_tasklet(f"copy_{f}", {"i": f"0:{face}", "j": f"0:{face}"}, {"x": dace.Memlet(f"X[i, j, {f}]")},
                              f"y = x + {f}", {"y": dace.Memlet(f"F[{f}, i, j]")}, external_edges=True)
        prev = st
    if volume:
        st = sdfg.add_state_after(prev, "volume")
        st.add_mapped_tasklet("vol_a", {"k": f"0:{N}"}, {"v": dace.Memlet("V[k]")}, "w = v * 3",
                              {"w": dace.Memlet("W[k]")}, external_edges=True)
        st.add_mapped_tasklet("vol_b", {"k": f"0:{N}"}, {"v": dace.Memlet("V[k]")}, "u = v - 1",
                              {"u": dace.Memlet("U[k]")}, external_edges=True)
    sdfg.validate()
    return sdfg


def _run(sdfg: dace.SDFG) -> dict:
    rng = np.random.default_rng(7)
    args = {name: rng.random(desc.shape) for name, desc in sdfg.arrays.items() if not desc.transient}
    sdfg(**args)
    return args


def test_face_copies_fuse_into_one_map_volume_maps_stay():
    ref = _run(_face_copies())
    sdfg = _face_copies()
    states, maps = fuse_small_maps(sdfg, max_iterations=256)
    sdfg.validate()
    labels = sorted(n.map.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert maps == 3 and len([lb for lb in labels if lb.startswith("copy_")]) == 1
    assert "vol_a_map" in labels and "vol_b_map" in labels
    out = _run(sdfg)
    for name in ref:
        assert np.array_equal(ref[name], out[name]), name


def test_fused_body_size_is_bounded():
    sdfg = _face_copies(volume=False)
    _states, maps = fuse_small_maps(sdfg, max_iterations=256, max_tasklets=2)
    labels = [n.map.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert maps == 2 and len(labels) == 2  # pairs of one-tasklet bodies, then nothing more fits


def test_fuse_small_maps_off():
    sdfg = _face_copies()
    assert fuse_small_maps(sdfg, 0) == (0, 0)


def _any_sdfg(n: int, op) -> dace.SDFG:
    """``out = ANY(mask)`` (or ALL) over ``n`` doubles, as the frontend emits the Fortran intrinsic."""
    from dace.libraries.standard.nodes.allany import AllNode, AnyNode

    sdfg = dace.SDFG(f"small_{op}_{n}")
    sdfg.add_array("mask", [n], dace.float64)
    sdfg.add_array("out", [1], dace.bool_)
    state = sdfg.add_state()
    node = (AnyNode if op == "any" else AllNode)(f"{op}_mask")
    state.add_node(node)
    state.add_edge(state.add_read("mask"), None, node, "_mask", dace.Memlet(f"mask[0:{n}]"))
    state.add_edge(node, "_out", state.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


def test_a_small_any_or_all_becomes_a_tasklet_with_its_result():
    import numpy as np

    from dace_fortran.substitute.small_kernels import inline_small_reductions

    for op, ref in (("any", np.any), ("all", np.all)):
        for values in ([0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 3.0, -1.0]):
            sdfg = _any_sdfg(3, op)
            assert inline_small_reductions(sdfg) == 1
            assert not any(isinstance(n, dace.nodes.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
            out = np.zeros(1, dtype=np.bool_)
            sdfg(mask=np.array(values), out=out)
            assert out[0] == ref(np.array(values) != 0)


def test_a_large_any_stays_a_library_node():
    from dace_fortran.substitute.small_kernels import inline_small_reductions

    assert inline_small_reductions(_any_sdfg(100, "any")) == 0


def _ordered_scratch_fields() -> dace.SDFG:
    """One state: ``T[0, i] = a[i]`` and ``T[1, i] = b[i]`` (T a transient), the second ordered after the first by
    the empty edge StateFusionExtended puts on its source node, then ``out[i] = T[0, i] + T[1, i]`` read through
    the second map's node of T: nothing but that edge leads from the first map's node of T to a reader."""
    sdfg = dace.SDFG("ordered_scratch_fields")
    for name in ("a", "b", "out"):
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_transient("T", [2, N], dace.float64)
    state = sdfg.add_state()
    first_t, second_t, b = state.add_access("T"), state.add_access("T"), state.add_read("b")
    state.add_mapped_tasklet("first", {"i": f"0:{N}"}, {"x": dace.Memlet("a[i]")}, "y = x",
                             {"y": dace.Memlet("T[0, i]")}, external_edges=True, output_nodes={"T": first_t})
    state.add_mapped_tasklet("second", {"i": f"0:{N}"}, {"x": dace.Memlet("b[i]")}, "y = 2 * x",
                             {"y": dace.Memlet("T[1, i]")}, external_edges=True, input_nodes={"b": b},
                             output_nodes={"T": second_t})
    state.add_nedge(first_t, b, dace.Memlet())
    state.add_mapped_tasklet("use", {"i": f"0:{N}"}, {"p": dace.Memlet("T[0, i]"), "q": dace.Memlet("T[1, i]")},
                             "o = p + q", {"o": dace.Memlet("out[i]")}, external_edges=True,
                             input_nodes={"T": second_t})
    sdfg.validate()
    return sdfg


def test_maps_whose_fusion_would_orphan_a_scratch_write_stay_apart():
    sdfg = _ordered_scratch_fields()
    _states, maps = fuse_small_maps(sdfg, max_iterations=N)
    sdfg.simplify()
    args = _run(sdfg)
    np.testing.assert_array_equal(args["out"], args["a"] + 2 * args["b"])
    assert maps == 0
