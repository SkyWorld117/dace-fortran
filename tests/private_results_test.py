"""Private branch results stay local without invented initial values."""
import dace
import numpy as np
import pytest
from dace.sdfg.state import ConditionalBlock,ControlFlowRegion
from dace.properties import CodeBlock
from dace_fortran.private_results import restore_private_result_storage


def test_guarded_private_result_does_not_become_an_input(tmp_path):
    graph = dace.SDFG('private_guarded_symbol')
    graph.add_symbol('x',dace.int32)
    graph.add_symbol('__sc_0',dace.int32)
    graph.add_array('out',[1],dace.int32)
    start = graph.add_state('start')
    block = ConditionalBlock('conditional_definition',sdfg=graph)
    graph.add_node(block)
    graph.add_edge(start,block,dace.InterstateEdge())
    branch = ControlFlowRegion('definition',sdfg=graph)
    before = branch.add_state('before')
    after = branch.add_state('after')
    branch.add_edge(before,after,dace.InterstateEdge(assignments={'__sc_0':'x+7'}))
    block.add_branch(CodeBlock('x > 0'),branch)
    last = graph.add_state('publish')
    graph.add_edge(block,last,dace.InterstateEdge())
    tasklet = last.add_tasklet('publish',{}, {'value'}, 'value = __sc_0 if x > 0 else -3')
    last.add_edge(tasklet,'value',last.add_write('out'),None,dace.Memlet('out[0]'))
    assert graph.free_symbols == {'x','__sc_0'}
    assert restore_private_result_storage(graph) == {'__sc_0'}
    assert graph.free_symbols == {'x'}
    assert restore_private_result_storage(graph) == set()
    graph.validate()
    graph.build_folder = str(tmp_path/'build')
    compiled = graph.compile()
    for value in (-2,0,3,9,-1,1):
        out = np.zeros(1,dtype=np.int32)
        compiled(out=out,x=np.int32(value))
        np.testing.assert_array_equal(out,[value+7 if value>0 else -3])


def test_undefined_private_result_is_refused():
    graph = dace.SDFG('undefined_private_result')
    graph.add_symbol('__sc_0',dace.int32)
    graph.add_array('out',[1],dace.int32)
    state = graph.add_state()
    tasklet = state.add_tasklet('read',{}, {'value'}, 'value = __sc_0')
    state.add_edge(tasklet,'value',state.add_write('out'),None,dace.Memlet('out[0]'))
    with pytest.raises(ValueError,match='undefined private SCF'):
        restore_private_result_storage(graph)
    assert '__sc_0' in graph.symbols
