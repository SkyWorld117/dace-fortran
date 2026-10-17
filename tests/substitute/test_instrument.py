"""``--instrument``: every top-level GPU map gets CUDA-event timing; nested maps and host maps are left alone.
Run: PYTHONPATH=../../dace python -m pytest test_instrument.py"""
import dace
import pytest

try:
    from dace_fortran.substitute import make_substitute
except Exception as exc:  # noqa: BLE001 -- importing the tool builds the bridge, which needs LLVM
    pytest.skip(f"make_substitute needs the bridge: {str(exc)[:60]}", allow_module_level=True)


def test_top_level_gpu_maps_get_gpu_events():
    sdfg = dace.SDFG("t")
    sdfg.add_array("a", [16], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("b", [16], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("h", [16], dace.float64)
    st = sdfg.add_state("s")
    st.add_mapped_tasklet("gpu", {"i": "0:16"}, {"x": dace.Memlet("a[i]")}, "y = x + 1", {"y": dace.Memlet("b[i]")},
                          external_edges=True, schedule=dace.ScheduleType.GPU_Device)
    st.add_mapped_tasklet("host", {"i": "0:16"}, {"x": dace.Memlet("h[i]")}, "y = x", {"y": dace.Memlet("h[i]")},
                          external_edges=True)
    assert make_substitute.instrument_kernels(sdfg) == 1
    kinds = {n.map.label: n.map.instrument for n in st.nodes() if isinstance(n, dace.nodes.MapEntry)}
    assert kinds["gpu"] == dace.InstrumentationType.GPU_Events
    assert kinds["host"] == dace.InstrumentationType.No_Instrumentation
