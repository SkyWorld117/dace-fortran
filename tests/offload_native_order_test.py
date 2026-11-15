"""The bring-up schedule preserves a loop-carried recurrence on the device."""
import os
import ctypes as ct
import dace
import numpy as np
import pytest
from dace_fortran.offload import offload_native_order


def prefix_graph(bounded=False):
    graph = dace.SDFG('native_prefix')
    graph.add_symbol('N',dace.int32)
    graph.add_symbol('offset',dace.int32)
    graph.add_symbol('i',dace.int32)
    graph.add_array('values',['N'],dace.int32)
    graph.add_array('result',['N'],dace.int32)
    graph.add_array('scratch',['local_extent' if bounded else 'N'],dace.int32,transient=True)
    start = graph.add_state('prepare')
    start.add_mapped_tasklet('prepare',{'k':'0:N'}, {},'value = offset + k',
        {'value':dace.Memlet('scratch[k]')},external_edges=True,schedule=dace.ScheduleType.Sequential)
    from dace.sdfg.state import LoopRegion
    loop = LoopRegion('prefix_loop',condition_expr='i < N',loop_var='i',
                      initialize_expr='i = 0',update_expr='i = i + 1')
    graph.add_node(loop)
    graph.add_edge(start,loop,dace.InterstateEdge())
    if bounded:
        allocation = graph.add_state_before(start,'allocation_extent')
        graph.edges_between(allocation,start)[0].data.assignments = {'local_extent':'N'}
    body = loop.add_state('prefix')
    last = graph.add_state('finish')
    graph.add_edge(loop,last,dace.InterstateEdge())
    task = body.add_tasklet('prefix',{'x','s','prev'},{'value'},
                             'value = x + s + (prev if i > 0 else 0)')
    for connector,expression in [('x','values[i]'),('s','scratch[i]'),('prev','result[Max(i-1,0)]')]:
        body.add_edge(body.add_read(expression.split('[')[0]),None,task,connector,dace.Memlet(expression))
    body.add_edge(task,'value',body.add_write('result'),None,dace.Memlet('result[i]'))
    return offload_native_order(graph,{'scratch':'N+3'} if bounded else None)


def test_native_order_retains_loop_and_device_workspace():
    graph = prefix_graph()
    graph.validate()
    assert graph.arrays['scratch'].transient
    assert graph.arrays['scratch'].storage == dace.StorageType.GPU_Global
    assert graph.arrays['scratch'].lifetime == dace.AllocationLifetime.Persistent
    assert 'scratch' not in graph.arglist()
    assert {name for name in graph.arglist()} == {'values','result','N','offset'}
    assert sum(isinstance(n,dace.nodes.MapEntry) and n.schedule == dace.ScheduleType.GPU_Device
               for n,_ in graph.all_nodes_recursive()) == 1


@pytest.mark.skipif(os.environ.get('DACE_TEST_GPU') != '1',reason='requires GPU')
@pytest.mark.parametrize('bounded',[False,True])
def test_native_order_recurrence_repeated_on_gpu(tmp_path,bounded):
    runtime = ct.CDLL('libcudart.so')
    runtime.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p),ct.c_size_t]
    runtime.cudaFree.argtypes = [ct.c_void_p]
    runtime.cudaMemcpy.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int]
    graph = prefix_graph(bounded)
    graph.build_folder = str(tmp_path/'native_prefix')
    compiled = graph.compile()
    class DeviceArray:
        def __init__(self,array):
            self.pointer = ct.c_void_p()
            assert runtime.cudaMalloc(ct.byref(self.pointer),array.nbytes) == 0
            self.__cuda_array_interface__ = dict(shape=array.shape,strides=array.strides,
                typestr=array.dtype.str,data=(self.pointer.value,False),version=3)
    n = 11
    values = np.arange(n,dtype=np.int32)
    result = np.full(n,-19,dtype=np.int32)
    device = [DeviceArray(a) for a in (values,result)]
    try:
        for offset in (3,-7,0,3):
            result[:] = -19
            for host,gpu in zip((values,result),device):
                assert runtime.cudaMemcpy(gpu.pointer,host.ctypes.data,host.nbytes,1) == 0
            compiled(values=device[0],result=device[1],N=np.int32(n),offset=np.int32(offset))
            assert runtime.cudaMemcpy(result.ctypes.data,device[1].pointer,result.nbytes,2) == 0
            np.testing.assert_array_equal(result,np.cumsum(values+offset+np.arange(n),dtype=np.int32))
    finally:
        for gpu in device:
            assert runtime.cudaFree(gpu.pointer) == 0
