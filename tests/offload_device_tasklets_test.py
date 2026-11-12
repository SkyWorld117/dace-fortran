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
