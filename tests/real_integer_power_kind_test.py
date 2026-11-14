"""REAL-valued exponents retain libm rounding; INTEGER exponents retain products."""
import ctypes
import os
import subprocess

import numpy as np
import pytest

from tests._util import build_sdfg


@pytest.mark.parametrize('kind', [4, 8])
def test_integral_real_powers_match_native_fortran(tmp_path, kind):
    source = f"""
subroutine power_kinds(a,scale,b)
  real({kind}),intent(in):: a(64),scale
  real({kind}),intent(out):: b(64,6)
  real({kind}),parameter:: dm=3.
  integer::i
  do i=1,64
    b(i,1)=a(i)**3
    b(i,2)=a(i)**3.0
    b(i,3)=a(i)**4.0
    b(i,4)=a(i)**5.0
    b(i,5)=(a(i)*scale)**dm
    b(i,6)=scale**3.0
  end do
end subroutine
"""
    path=tmp_path/'native.f90';path.write_text(source)
    subprocess.run([os.environ.get('WRF_FC','gfortran'),'-shared','-fPIC','-O2',
                    '-ffp-contract=off','-fno-tree-vectorize',str(path),
                    '-o',str(tmp_path/'native.so')],check=True)
    dtype=np.float32 if kind==4 else np.float64
    values=np.random.default_rng(70).uniform(-4,4,64).astype(dtype)
    scale=np.array(1.61803,dtype=dtype)
    reference=np.zeros((64,6),dtype=dtype,order='F')
    actual=np.zeros_like(reference,order='F')
    native=ctypes.CDLL(str(tmp_path/'native.so')).power_kinds_
    native.argtypes=[ctypes.c_void_p]*3
    native.restype=None
    native(values.ctypes.data,scale.ctypes.data,reference.ctypes.data)
    graph=build_sdfg(source,tmp_path/'graph',name='power_kinds').build()
    graph.build_folder=str(tmp_path/'build')
    graph(a=values,scale=dtype(scale),b=actual)
    assert actual.tobytes(order='F')==reference.tobytes(order='F')
