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
