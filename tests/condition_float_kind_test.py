"""Floating conditions retain literal kind and REAL(4) arithmetic at boundaries."""
import os

import numpy as np
import pytest

from dace_fortran.pipelines import optimize
from tests._util import build_sdfg


def execute(source, tmp_path, a, gpu):
    sdfg=build_sdfg(source,tmp_path,name='floating_condition').build()
    if gpu:
        optimize(sdfg,gpu=False)
        sdfg.apply_gpu_transformations()
    sdfg.build_folder=str(tmp_path/'compiled')
    out=np.full(a.size,-1,dtype=np.int32)
    sdfg(a=a,out=out,n=a.size)
    return out


@pytest.mark.parametrize('gpu',[False,True])
@pytest.mark.parametrize('kind,dtype',[(4,np.float32),(8,np.float64)])
@pytest.mark.parametrize('literal',['0.1','0.1d0'])
def test_repeated_array_condition_uses_literal_kind(tmp_path,kind,dtype,literal,gpu):
    if gpu and os.environ.get('DACE_TEST_GPU')!='1':pytest.skip('set DACE_TEST_GPU=1 on a CUDA node')
    source=f'''
subroutine test(n,a,out)
integer,intent(in)::n
real({kind}),intent(in)::a(n)
integer,intent(out)::out(n)
integer::i
do i=1,n
  if(a(i)>=-{literal}.and.a(i)<={literal}) then
    out(i)=1
  else
    out(i)=0
  endif
enddo
end subroutine
'''
    bound=dtype(np.float32(.1)) if literal=='0.1' else dtype(.1)
    a=np.array([-bound,np.nextafter(-bound,dtype(-np.inf)),np.nextafter(-bound,dtype(np.inf)),
                0.,bound,np.nextafter(bound,dtype(np.inf)),np.nextafter(bound,dtype(-np.inf))],dtype=dtype)
    limit=float(np.float32(.1)) if literal=='0.1' else .1
    expected=((a.astype(np.float64)>=-limit)&(a.astype(np.float64)<=limit)).astype(np.int32)
    np.testing.assert_array_equal(execute(source,tmp_path,a,gpu),expected)


@pytest.mark.parametrize('gpu',[False,True])
def test_repeated_float32_condition_rounds_arithmetic_before_comparison(tmp_path,gpu):
    if gpu and os.environ.get('DACE_TEST_GPU')!='1':pytest.skip('set DACE_TEST_GPU=1 on a CUDA node')
    source='''
subroutine test(n,a,out)
integer,intent(in)::n
real,intent(in)::a(n)
integer,intent(out)::out(n)
integer::i
do i=1,n
  if((a(i)+1.0)>1.0.and.(a(i)+1.0)<=2.0) then
    out(i)=1
  else
    out(i)=0
  endif
enddo
end subroutine
'''
    a=np.array([0.,2.**-25,2.**-24,2.**-23,-2.**-25,.5,1.,1.+2.**-23],dtype=np.float32)
    added=a+np.float32(1.)
    expected=((added>np.float32(1.))&(added<=np.float32(2.))).astype(np.int32)
    np.testing.assert_array_equal(execute(source,tmp_path,a,gpu),expected)
