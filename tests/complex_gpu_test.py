"""Opt-in execution checks for complex arithmetic and intrinsics on CUDA."""
import os
from pathlib import Path

import dace
import numpy as np
import pytest

from dace_fortran.pipelines import optimize
from tests._util import build_sdfg

pytestmark = pytest.mark.skipif(
    os.environ.get('DACE_TEST_GPU') != '1', reason='set DACE_TEST_GPU=1 on a CUDA node'
)


@pytest.mark.parametrize(
    'kind,dtype,real_dtype,rtol',
    [(4, np.complex64, np.float32, 2.e-6), (8, np.complex128, np.float64, 2.e-13)],
)
def test_complex_log_sqrt_division_and_abs_on_gpu(tmp_path: Path, kind, dtype, real_dtype, rtol):
    source = f'''
subroutine main(n,a,b,out,norm)
integer,intent(in)::n
complex({kind}),intent(in)::a(n),b(n)
complex({kind}),intent(out)::out(n)
real({kind}),intent(out)::norm(n)
integer::i
do i=1,n
  out(i)=sqrt(log(a(i)/b(i))+a(i)*a(i))
  norm(i)=abs((a(i)**2-1)/(a(i)**2+2))**2
enddo
end subroutine
'''
    sdfg = build_sdfg(source, tmp_path, name=f'complex_math_gpu_{kind}').build()
    optimize(sdfg, gpu=False)
    sdfg.apply_gpu_transformations()
    assert any(isinstance(node, dace.nodes.MapEntry) and node.schedule == dace.ScheduleType.GPU_Device
               for node, _ in sdfg.all_nodes_recursive())
    sdfg.build_folder = str(tmp_path/'compiled')
    a = np.array([3+4j, .1+.2j, -2+1j, 5-12j, -1-.5j, .7-.1j], dtype=dtype)
    b = np.array([1+.2j, 2+3j, -.5+.1j, 4-2j, 1-1j, -.2-.1j], dtype=dtype)
    out = np.zeros(a.shape, dtype=dtype); norm = np.zeros(a.shape, dtype=real_dtype)
    sdfg(n=len(a), a=a, b=b, out=out, norm=norm)
    np.testing.assert_allclose(out, np.sqrt(np.log(a/b)+a*a), rtol=rtol)
    np.testing.assert_allclose(norm, np.abs((a*a-1)/(a*a+2))**2, rtol=rtol)
