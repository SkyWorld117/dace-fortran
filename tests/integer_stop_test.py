"""Integer STOP remains termination and residual unstructured branches fail loudly."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dace_fortran import build_sdfg_from_hlfir
from tests._util import build_sdfg

# A structured runtime-call fixture, matching the UW error paths after Flang's
# CFG structurization. STOP itself has no scf terminator representation.
STRUCTURED = """module {
  func.func @_QPchecked(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.array<4xf32>>) {
    %n:2 = hlfir.declare %arg0 {uniq_name = "_QFcheckedEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %four = arith.constant 4 : index
    %shape = fir.shape %four : (index) -> !fir.shape<1>
    %a:2 = hlfir.declare %arg1(%shape) {uniq_name = "_QFcheckedEa"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf32>>, !fir.ref<!fir.array<4xf32>>)
    %zero = arith.constant 0 : i32
    %status = arith.constant 7 : i32
    %false = arith.constant false
    %value = fir.load %n#0 : !fir.ref<i32>
    %bad = arith.cmpi slt, %value, %zero : i32
    scf.if %bad {
      fir.call @_FortranAStopStatement(%status, %false, %false) : (i32, i1, i1) -> ()
    }
    return
  }
  func.func private @_FortranAStopStatement(i32, i1, i1)
}
"""


def test_integer_stop_is_retained_and_normal_execution_continues(tmp_path):
    path = tmp_path / 'structured.hlfir'
    path.write_text(STRUCTURED)
    graph = build_sdfg_from_hlfir(path, entry='checked', pipeline='builtin.module(canonicalize)')
    stops = [node for node, _ in graph.all_nodes_recursive()
             if getattr(node, 'label', '').startswith('fortran_stop_')]
    assert len(stops) == 1
    assert stops[0].side_effects and stops[0].ordered_side_effects
    code = '\n'.join(part.code for part in graph.generate_code())
    assert 'std::exit(static_cast<int>(7))' in code and 'trap;' in code
    a = np.arange(4, dtype=np.float32)
    graph(n=np.int32(1), a=a)
    np.testing.assert_array_equal(a, np.arange(4))
    graph_path = tmp_path / 'stop.sdfgz'
    graph.save(str(graph_path), compress=True)
    script = ('import dace, numpy as np; '
              f'g=dace.SDFG.from_file({str(graph_path)!r}); '
              'g(n=np.int32(-1), a=np.zeros(4,dtype=np.float32)); '
              'raise SystemExit(99)')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True)
    assert result.returncode == 7, result.stderr.decode()


def test_noreturn_unstructured_branch_does_not_drop_the_continuation(tmp_path):
    source = """subroutine checked(n, a)
      integer, intent(in) :: n
      real, intent(inout) :: a(4)
      a = a + 1.0
      if (n < 0) stop 7
      a = a * 2.0
    end subroutine
"""
    with pytest.raises(RuntimeError, match='unstructured multi-block entry'):
        build_sdfg(source, tmp_path, name='checked', entry='checked', pipeline='builtin.module(canonicalize)').build()


def test_fresh_fortran_stop_structurizes_without_losing_the_continuation(tmp_path):
    source = """subroutine checked(n, a)
      integer, intent(in) :: n
      real, intent(inout) :: a(4)
      a = a + 1.0
      if (n < 0) stop 7
      a = a * 2.0
    end subroutine
"""
    graph = build_sdfg(source, tmp_path, name='checked', entry='checked').build()
    stops = [node for node, _ in graph.all_nodes_recursive()
             if getattr(node, 'label', '').startswith('fortran_stop_')]
    assert len(stops) == 1
    a = np.arange(4, dtype=np.float32)
    graph(n=np.int32(1), a=a)
    np.testing.assert_array_equal(a, (np.arange(4) + 1) * 2)


@pytest.mark.parametrize('kind', [4, 8])
def test_real_function_with_stop_inlines_without_losing_its_return(tmp_path, kind):
    source = f"""
module helpers
contains
real(kind={kind}) function positive(x)
  real(kind={kind}),intent(in)::x
  if(x<0.0_{kind})stop
  positive=2.0_{kind}*x+1.0_{kind}
end function
end module
subroutine apply_helper(a,b)
  use helpers,only:positive
  real(kind={kind}),intent(in)::a(4)
  real(kind={kind}),intent(out)::b(4)
  integer i
  do i=1,4
    b(i)=positive(a(i))
  end do
end subroutine
"""
    graph = build_sdfg(source, tmp_path/'graph', name='apply_helper', entry='apply_helper').build()
    stops = [node for node, _ in graph.all_nodes_recursive()
             if getattr(node, 'label', '').startswith('fortran_stop_')]
    assert len(stops) == 1
    dtype = np.float32 if kind == 4 else np.float64
    inputs = np.array([1, 2, 3, 4], dtype=dtype)
    output = np.zeros(4, dtype=dtype)
    graph.build_folder = str(tmp_path/'build')
    graph(a=inputs, b=output)
    np.testing.assert_array_equal(output, 2*inputs+1)
    path = tmp_path/'real_stop.sdfgz'
    graph.save(str(path))
    script = ('import dace,numpy as np; '
              f'g=dace.SDFG.from_file({str(path)!r}); '
              f'g(a=np.full(4,-1,dtype=np.float{kind*8}),b=np.zeros(4,dtype=np.float{kind*8})); '
              'raise SystemExit(99)')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
