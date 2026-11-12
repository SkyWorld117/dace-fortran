"""Keep residual scalar array updates on device without moving host control."""
import copy

import dace
import pytest

from dace_fortran.offload import _device_array_tasklets, offload_device_resident


def graph(code='y = -x',input_kind='device',output_kind='device'):
    sdfg=dace.SDFG('device_element_control')
    sdfg.add_array('A',[8],dace.float32,storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B',[8],dace.float32,storage=dace.StorageType.GPU_Global)
    sdfg.add_array('H',[8],dace.float32)
    sdfg.add_scalar('alpha',dace.float32)
    state=sdfg.add_state()
    tasklet=state.add_tasklet('element',{'x'} if input_kind else set(),{'y'},code)
    if input_kind:
        src='A' if input_kind=='device' else 'alpha' if input_kind=='scalar' else 'H'
        state.add_edge(state.add_read(src),None,tasklet,'x',dace.Memlet(src+'[0]'))
    dest='B' if output_kind=='device' else 'alpha' if output_kind=='scalar' else 'H'
    state.add_edge(tasklet,'y',state.add_write(dest),None,dace.Memlet(dest+'[0]'))
    return sdfg,state,tasklet


def test_array_element_update_moves_without_host_copies():
    sdfg,state,tasklet=graph()
    assert _device_array_tasklets(sdfg)==1
    sdfg.validate()
    assert state.scope_dict()[tasklet].schedule==dace.ScheduleType.GPU_Device
    assert _device_array_tasklets(sdfg)==0
    code='\n'.join(c.code for c in sdfg.generate_code())
    assert 'cudaMemcpy' not in code and '__host_mirror' not in code


@pytest.mark.parametrize('input_kind,code',[(None,'y = 0.0'),('scalar','y = x * 2.0')])
def test_constant_and_by_value_scalar_inputs(input_kind,code):
    sdfg,state,tasklet=graph(code,input_kind)
    assert _device_array_tasklets(sdfg)==1
    sdfg.validate()
    code='\n'.join(c.code for c in sdfg.generate_code())
    assert 'cudaMemcpy' not in code and '__host_mirror' not in code


@pytest.mark.parametrize('input_kind,output_kind',[('host','device'),('device','host'),('device','scalar')])
def test_host_arrays_and_control_outputs_stay_on_host(input_kind,output_kind):
    sdfg,state,tasklet=graph(input_kind=input_kind,output_kind=output_kind)
    assert _device_array_tasklets(sdfg)==0
    assert state.scope_dict()[tasklet] is None


def test_explicit_side_effects_are_not_moved():
    sdfg,state,tasklet=graph();tasklet.side_effects=True
    assert _device_array_tasklets(sdfg)==0


def test_callback_side_effects_are_not_moved():
    sdfg,state,tasklet=graph('y = callback(x)')
    sdfg.add_symbol('callback',dace.callback(dace.float32,dace.float32))
    assert _device_array_tasklets(sdfg)==0


def test_multiple_edges_to_one_array_keep_distinct_memlets():
    sdfg,state,tasklet=graph('y = x + other')
    tasklet.add_in_connector('other')
    state.add_edge(state.add_read('A'),None,tasklet,'other',dace.Memlet('A[1]'))
    assert _device_array_tasklets(sdfg)==1
    sdfg.validate()
    assert len(state.in_edges(tasklet))==2
    assert len({id(e.data) for e in state.edges()})==len(state.edges())


def test_offload_option_is_explicit_and_handles_a_scalar_only_program():
    # Include an ordinary bulk map so the default path has a GPU kernel too.
    sdfg,state,tasklet=graph()
    bulk=sdfg.add_state_before(state,'bulk')
    bulk.add_mapped_tasklet('bulk',{'i':'0:8'},{'a':dace.Memlet('A[i]')},
                           'b = a',{'b':dace.Memlet('B[i]')},external_edges=True)
    default=copy.deepcopy(sdfg)
    offload_device_resident(default)
    assert any(name.startswith('__host_mirror') for name in default.arrays)
    offload_device_resident(sdfg,device_tasklets=True)
    sdfg.validate()
    assert not any(name.startswith('__host_mirror') for name in sdfg.arrays)
    scalar_only,_,_=graph()
    _,maps,_=offload_device_resident(scalar_only,device_tasklets=True)
    assert maps==1


def test_ordered_effects_are_not_moved():
    sdfg,state,tasklet=graph();tasklet.ordered_side_effects=True
    assert _device_array_tasklets(sdfg)==0


def test_non_python_code_requires_an_explicit_purity_declaration():
    sdfg=dace.SDFG('cpp_element_control');sdfg.add_array('A',[8],dace.float32,storage=dace.StorageType.GPU_Global)
    state=sdfg.add_state();tasklet=state.add_tasklet('cpp',set(),{'y'},'y = 0.0f;',language=dace.Language.CPP)
    state.add_edge(tasklet,'y',state.add_write('A'),None,dace.Memlet('A[0]'))
    assert _device_array_tasklets(sdfg)==0
    tasklet.side_effects=False
    assert _device_array_tasklets(sdfg)==1
    sdfg.validate()


@pytest.mark.parametrize('expression', [
    'A[N[0] - 1, k] <= B[k]',
    'A[N[0] - 1, k] + B[M[0], k]',
    'A[(N[0] + M[1]) * 2, k] > 0',
])
def test_host_mirror_rewrites_the_whole_nested_subscript(expression):
    import ast
    from dace_fortran.offload import _rewrite_scalar_subscript
    replaced, count = _rewrite_scalar_subscript(expression, 'A', 'mirror[0]')
    assert count == 1
    ast.parse(replaced, mode='eval')
    assert 'A[' not in replaced
    values = dict(A={(2, 0): 7, (6, 0): 7}, B={(2, 0): 3, 0: 3},
                  N=[3], M=[2, 0], k=0, mirror=[7])
    assert eval(expression, {}, values) == eval(replaced, {}, values)


def test_host_mirror_does_not_replace_a_different_container():
    from dace_fortran.offload import _rewrite_scalar_subscript
    expression = 'A[N[0], 1] + BA[0]'
    assert _rewrite_scalar_subscript(expression, 'B', 'mirror[0]') == (expression, 0)


def nested_index_graph(outer_first=False):
    from dace_fortran.offload import _reroute_host_scalar_accesses
    graph = dace.SDFG('nested_host_mirror_index')
    graph.add_symbol('index', dace.int32)
    graph.add_array('A', [8], dace.int32, transient=True, storage=dace.StorageType.GPU_Global)
    graph.add_array('N', [1], dace.int32, transient=True, storage=dace.StorageType.GPU_Global)
    graph.add_array('result', [1], dace.int32)
    start = graph.add_state('compute')
    start.add_mapped_tasklet('values', {'i': '0:8'}, {}, 'a = 2*i + 5',
                            {'a': dace.Memlet('A[i]')}, external_edges=True,
                            schedule=dace.ScheduleType.GPU_Device)
    start.add_mapped_tasklet('selector', {'one': '0:1'}, {}, 'n = index',
                            {'n': dace.Memlet('N[0]')}, external_edges=True,
                            schedule=dace.ScheduleType.GPU_Device)
    for label, value, condition in [('match', 1, 'A[N[0]] == 11'),
                                     ('other', 0, 'A[N[0]] != 11')]:
        target = graph.add_state(label)
        tasklet = target.add_tasklet(label, {}, {'y'}, f'y = {value}')
        target.add_edge(tasklet, 'y', target.add_write('result'), None, dace.Memlet('result[0]'))
        graph.add_edge(start, target, dace.InterstateEdge(condition))
    if outer_first == 'batch':
        from dace_fortran.offload import _batch_host_scalar_reroutes
        assert _batch_host_scalar_reroutes(graph) == 2
    elif outer_first:
        assert _reroute_host_scalar_accesses(graph, 'A') == 2
        assert _reroute_host_scalar_accesses(graph, 'N') == 0
    else:
        assert _reroute_host_scalar_accesses(graph, 'N') == 2
        assert _reroute_host_scalar_accesses(graph, 'A') == 2
    graph.validate()
    return graph


@pytest.mark.parametrize('outer_first', [False, True, 'batch'])
def test_nested_host_mirror_index_validates_and_generates_code(outer_first):
    code = '\n'.join(part.code for part in nested_index_graph(outer_first).generate_code())
    assert '__host_mirror' in code


@pytest.mark.skipif(__import__('os').environ.get('DACE_TEST_GPU') != '1', reason='requires GPU')
@pytest.mark.parametrize('outer_first', [False, True, 'batch'])
def test_nested_host_mirror_index_changes_between_gpu_calls(outer_first):
    import numpy as np
    graph = nested_index_graph(outer_first)
    compiled = graph.compile()
    for index in (3, 5, 3, 2, 7, 3):
        result = np.full(1, -1, dtype=np.int32)
        compiled(result=result, index=np.int32(index))
        np.testing.assert_array_equal(result, [int(index == 3)])


def reduction_control_graph():
    from dace.libraries.standard import Reduce
    graph = dace.SDFG('gpu_reduction_scalar_control')
    graph.add_symbol('offset', dace.int32)
    graph.add_array('values', [8], dace.float32, transient=True)
    graph.add_scalar('peak', dace.float32, transient=True)
    graph.add_array('result', [1], dace.int32)
    start = graph.add_state('compute')
    start.add_mapped_tasklet('values', {'i': '0:8'}, {}, 'value = float32(i + offset)',
                            {'value': dace.Memlet('values[i]')}, external_edges=True)
    reduce = Reduce('max_value', wcr='lambda a, b: max(a, b)', axes=[0],
                    identity='-3.4028234663852886e38', schedule=dace.ScheduleType.Sequential)
    reduce.add_in_connector('_in')
    reduce.add_out_connector('_out')
    start.add_node(reduce)
    written_values = next(node for node in start.nodes()
                          if isinstance(node, dace.sdfg.nodes.AccessNode) and node.data == 'values')
    start.add_edge(written_values, None, reduce, '_in', dace.Memlet('values[0:8]'))
    start.add_edge(reduce, '_out', start.add_write('peak'), None, dace.Memlet('peak[0]'))
    for label, value, condition in [('positive', 1, 'peak > 0'), ('other', 0, 'peak <= 0')]:
        target = graph.add_state(label)
        tasklet = target.add_tasklet(label, {}, {'y'}, f'y = {value}')
        target.add_edge(tasklet, 'y', target.add_write('result'), None, dace.Memlet('result[0]'))
        graph.add_edge(start, target, dace.InterstateEdge(condition))
    offload_device_resident(graph, device_libnodes=True)
    graph.validate()
    assert graph.arrays['peak'].storage == dace.StorageType.GPU_Global
    assert reduce.schedule == dace.ScheduleType.GPU_Device
    return graph


def test_device_reduction_scalar_can_control_a_host_branch():
    reduction_control_graph().generate_code()


@pytest.mark.skipif(__import__('os').environ.get('DACE_TEST_GPU') != '1', reason='requires GPU')
def test_device_reduction_scalar_host_branch_repeats_on_gpu():
    import numpy as np
    compiled = reduction_control_graph().compile()
    for offset in (-20, 0, -7, 100):
        result = np.full(1, -1, dtype=np.int32)
        compiled(result=result, offset=np.int32(offset))
        np.testing.assert_array_equal(result, [int(offset + 7 > 0)])


def compound_control_graph():
    from dace.sdfg.state import ControlFlowRegion
    from dace_fortran.offload import _batch_host_scalar_reroutes
    graph = dace.SDFG('compound_repeated_index_control')
    graph.add_symbol('index', dace.int32)
    graph.add_array('A', [8], dace.int32, transient=True, storage=dace.StorageType.GPU_Global)
    graph.add_array('N', [1], dace.int32, transient=True, storage=dace.StorageType.GPU_Global)
    graph.add_array('result', [1], dace.int32)
    region = ControlFlowRegion('compound', sdfg=graph)
    graph.add_node(region, is_start_block=True)
    start = region.add_state('compute', is_start_block=True)
    start.add_mapped_tasklet('values', {'i': '0:8'}, {}, 'a = 2*i + 5',
                            {'a': dace.Memlet('A[i]')}, external_edges=True,
                            schedule=dace.ScheduleType.GPU_Device)
    start.add_mapped_tasklet('selector', {'one': '0:1'}, {}, 'n = index',
                            {'n': dace.Memlet('N[0]')}, external_edges=True,
                            schedule=dace.ScheduleType.GPU_Device)
    for label, value, condition in [('match', 1, 'A[N[0]] + A[N[0]] == 22'),
                                     ('other', 0, 'A[N[0]] + A[N[0]] != 22')]:
        target = graph.add_state(label)
        tasklet = target.add_tasklet(label, {}, {'y'}, f'y = {value}')
        target.add_edge(tasklet, 'y', target.add_write('result'), None, dace.Memlet('result[0]'))
        graph.add_edge(region, target, dace.InterstateEdge(condition))
    assert _batch_host_scalar_reroutes(graph) > 0
    graph.validate()
    return graph


def test_mirror_after_compound_block_reuses_repeated_array_element():
    compound_control_graph().generate_code()


@pytest.mark.skipif(__import__('os').environ.get('DACE_TEST_GPU') != '1', reason='requires GPU')
def test_mirror_after_compound_block_repeated_indices_change_on_gpu():
    import numpy as np
    compiled = compound_control_graph().compile()
    for index in (3, 5, 3, 2):
        result = np.full(1, -1, dtype=np.int32)
        compiled(result=result, index=np.int32(index))
        np.testing.assert_array_equal(result, [int(index == 3)])


def distinct_control_graph():
    from dace_fortran.offload import _reroute_host_scalar_accesses
    graph = dace.SDFG('different_control_elements')
    graph.add_symbol('offset', dace.int32)
    graph.add_array('A', [8], dace.int32, transient=True, storage=dace.StorageType.GPU_Global)
    graph.add_array('result', [1], dace.int32)
    start = graph.add_state('compute')
    start.add_mapped_tasklet('values', {'i': '0:8'}, {}, 'a = i + offset',
                            {'a': dace.Memlet('A[i]')}, external_edges=True,
                            schedule=dace.ScheduleType.GPU_Device)
    for label, value, condition in [('match', 1, 'A[0] + A[1] == 1'),
                                     ('other', 0, 'A[0] + A[1] != 1')]:
        end = graph.add_state(label)
        tasklet = end.add_tasklet(label, {}, {'y'}, f'y = {value}')
        end.add_edge(tasklet, 'y', end.add_write('result'), None, dace.Memlet('result[0]'))
        graph.add_edge(start, end, dace.InterstateEdge(condition))
    assert _reroute_host_scalar_accesses(graph, 'A') == 4
    graph.validate()
    return graph


def test_distinct_elements_get_distinct_mirrors():
    distinct_control_graph().generate_code()


@pytest.mark.skipif(__import__('os').environ.get('DACE_TEST_GPU') != '1', reason='requires GPU')
def test_distinct_control_elements_repeat_on_gpu():
    import numpy as np
    compiled = distinct_control_graph().compile()
    for offset in (0, 5, 0, -1):
        result = np.full(1, -1, dtype=np.int32)
        compiled(result=result, offset=np.int32(offset))
        np.testing.assert_array_equal(result, [int(offset == 0)])


def test_targeted_scalar_rewrite_preserves_other_indexed_reads():
    from dace_fortran.offload import _rewrite_scalar_subscript
    text, count = _rewrite_scalar_subscript('a + a[0]', 'a', 'mirror[0]', scalar=True, access='a')
    assert text == 'mirror[0] + a[0]' and count == 1


def array_abi_graph():
    graph = dace.SDFG('host_scalar_reads_device_array_abi')
    graph.add_symbol('index', dace.int32)
    graph.add_array('table', [8], dace.int32)
    graph.add_array('scale', [1], dace.int32)
    graph.add_array('result', [1], dace.int32)
    graph.add_array('bulk', [8], dace.int32, transient=True)
    graph.add_scalar('value', dace.int32, transient=True)
    start = graph.add_state('compute')
    start.add_mapped_tasklet('bulk', {'i': '0:8'}, {}, 'a = i',
                            {'a': dace.Memlet('bulk[i]')}, external_edges=True)
    tasklet = start.add_tasklet('host_value', {'x', 'y'}, {'z'}, 'z = x + y')
    start.add_edge(start.add_read('table'), None, tasklet, 'x', dace.Memlet('table[index]'))
    start.add_edge(start.add_read('scale'), None, tasklet, 'y', dace.Memlet('scale[0]'))
    start.add_edge(tasklet, 'z', start.add_write('value'), None, dace.Memlet('value[0]'))
    for label, val, condition in [('positive', 1, 'value > 3'), ('other', 0, 'value <= 3')]:
        end = graph.add_state(label)
        tasklet = end.add_tasklet(label, {}, {'y'}, f'y = {val}')
        end.add_edge(tasklet, 'y', end.add_write('result'), None, dace.Memlet('result[0]'))
        graph.add_edge(start, end, dace.InterstateEdge(condition))
    offload_device_resident(graph, device_array_abi=True)
    for name in ('table', 'scale', 'result'):
        assert graph.arrays[name].storage == dace.StorageType.GPU_Global
    return graph


def test_device_array_abi_includes_host_only_and_length_one_arguments():
    array_abi_graph().generate_code()


@pytest.mark.skipif(__import__('os').environ.get('DACE_TEST_GPU') != '1', reason='requires GPU')
def test_device_array_abi_scalar_reads_and_writes_repeat_on_gpu():
    import ctypes as ct
    import numpy as np
    runtime = ct.CDLL('libcudart.so')
    runtime.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
    runtime.cudaFree.argtypes = [ct.c_void_p]
    runtime.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
    class DeviceArray:
        def __init__(self, array):
            self.pointer = ct.c_void_p()
            assert runtime.cudaMalloc(ct.byref(self.pointer), array.nbytes) == 0
            self.__cuda_array_interface__ = dict(shape=array.shape, strides=None,
                typestr=array.dtype.str, data=(self.pointer.value, False), version=3)
    arrays = [np.arange(8, dtype=np.int32), np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)]
    device = [DeviceArray(array) for array in arrays]
    try:
        compiled = array_abi_graph().compile()
        for index, scale, offset in [(3, 1, 0), (5, -20, 0), (3, 1, 2), (2, 0, 0)]:
            arrays[0][:] = np.arange(8) + offset
            arrays[1][:] = scale
            arrays[2][:] = -1
            for host, gpu in zip(arrays, device):
                assert runtime.cudaMemcpy(gpu.pointer, host.ctypes.data, host.nbytes, 1) == 0
            compiled(table=device[0], scale=device[1], result=device[2], index=np.int32(index))
            assert runtime.cudaMemcpy(arrays[2].ctypes.data, device[2].pointer, arrays[2].nbytes, 2) == 0
            np.testing.assert_array_equal(arrays[2], [int(index + offset + scale > 3)])
    finally:
        for gpu in device:
            assert runtime.cudaFree(gpu.pointer) == 0


def boundary_singleton_graph():
    graph = dace.SDFG('boundary_singleton_transient')
    graph.add_symbol('offset', dace.int32)
    graph.add_array('temporary', [1], dace.int32, transient=True, storage=dace.StorageType.Register)
    graph.add_array('result', [1], dace.int32)
    start = graph.add_state('device_write')
    start.add_mapped_tasklet('write', {'i': '0:1'}, {}, 'a = offset',
                            {'a': dace.Memlet('temporary[i]')}, external_edges=True)
    end = graph.add_state('host_read')
    graph.add_edge(start, end, dace.InterstateEdge())
    tasklet = end.add_tasklet('read', {'x'}, {'y'}, 'y = x + 1')
    end.add_edge(end.add_read('temporary'), None, tasklet, 'x', dace.Memlet('temporary[0]'))
    end.add_edge(tasklet, 'y', end.add_write('result'), None, dace.Memlet('result[0]'))
    offload_device_resident(graph)
    assert graph.arrays['temporary'].storage == dace.StorageType.GPU_Global
    return graph


def test_boundary_singleton_transient_gets_device_storage():
    boundary_singleton_graph().generate_code()


@pytest.mark.skipif(__import__('os').environ.get('DACE_TEST_GPU') != '1', reason='requires GPU')
def test_boundary_singleton_transient_repeats_on_gpu():
    import numpy as np
    compiled = boundary_singleton_graph().compile()
    for value in (4, -3, 0, 4):
        result = np.full(1, -1, dtype=np.int32)
        compiled(result=result, offset=np.int32(value))
        np.testing.assert_array_equal(result, [value + 1])


def test_private_singleton_inside_one_map_stays_private():
    graph = dace.SDFG('private_singleton')
    graph.add_array('tmp', [1], dace.int32, transient=True, storage=dace.StorageType.Register)
    graph.add_array('output', [8], dace.int32)
    state = graph.add_state()
    entry, exit = state.add_map('columns', {'i': '0:8'})
    write = state.add_tasklet('write', {}, {'x'}, 'x = i')
    read = state.add_tasklet('read', {'x'}, {'y'}, 'y = x + 1')
    tmp = state.add_access('tmp')
    state.add_edge(entry, None, write, None, dace.Memlet())
    state.add_edge(write, 'x', tmp, None, dace.Memlet('tmp[0]'))
    state.add_edge(tmp, None, read, 'x', dace.Memlet('tmp[0]'))
    state.add_memlet_path(read, exit, state.add_write('output'),
                          src_conn='y', memlet=dace.Memlet('output[i]'))
    offload_device_resident(graph)
    assert graph.arrays['tmp'].storage == dace.StorageType.Register
    graph.generate_code()
