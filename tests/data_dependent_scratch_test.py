"""Overlap snapshots use materialized runtime extents on CPU and GPU."""
import ctypes as ct
import os
import numpy as np
import pytest

from dace_fortran.pipelines import optimize
from tests._util import build_sdfg


@pytest.mark.parametrize('device', [False, True])
def test_overlapping_shift_with_array_derived_extent(tmp_path, device):
    if device and os.environ.get('DACE_TEST_GPU') != '1':
        pytest.skip('requires an allocated GPU')
    source = """
subroutine shift(a,ends)
  real,intent(inout)::a(8,2)
  integer,intent(in)::ends(2)
  integer j
  do j=1,2
    a(2:ends(j),j)=a(1:ends(j)-1,j)+1.0
  end do
end subroutine
"""
    graph = build_sdfg(source, tmp_path/'graph', name='shift', entry='shift').build()
    assert any(name.startswith('array_extent_') for name in graph.symbols)
    optimize(graph, gpu=device, validate=True, unroll_limit=0, device_array_abi=device)
    graph.build_folder = str(tmp_path/'build')
    compiled = graph.compile()
    values = np.array(np.arange(16,dtype=np.float32).reshape((8,2),order='F'),order='F')
    ends = np.array([6,4],dtype=np.int32)
    device_arrays = []
    if device:
        runtime = ct.CDLL('libcudart.so')
        runtime.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
        runtime.cudaFree.argtypes = [ct.c_void_p]
        runtime.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
        class DeviceArray:
            def __init__(self, array):
                self.pointer = ct.c_void_p()
                assert runtime.cudaMalloc(ct.byref(self.pointer),array.nbytes)==0
                self.__cuda_array_interface__ = dict(shape=array.shape,strides=array.strides,
                    typestr=array.dtype.str,data=(self.pointer.value,False),version=3)
        device_arrays = [DeviceArray(array) for array in (values,ends)]
    try:
        for limits in ([6,4],[3,8],[1,2]):
            ends[:] = limits
            expected = values.copy(order='F')
            for column,end in enumerate(ends):
                expected[1:end,column] = values[:end-1,column]+1
            if device:
                for host,gpu in zip((values,ends),device_arrays):
                    assert runtime.cudaMemcpy(gpu.pointer,host.ctypes.data,host.nbytes,1)==0
                compiled(a=device_arrays[0],ends=device_arrays[1])
                assert runtime.cudaMemcpy(values.ctypes.data,device_arrays[0].pointer,values.nbytes,2)==0
            else:
                compiled(a=values,ends=ends)
            np.testing.assert_array_equal(values,expected)
    finally:
        for gpu in device_arrays:
            assert runtime.cudaFree(gpu.pointer)==0
