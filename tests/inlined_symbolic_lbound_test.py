"""Inlined explicit-shape inputs and outputs preserve symbolic caller bounds."""
from pathlib import Path
import numpy as np
import pytest
from tests._util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason='no LLVM flang on PATH')

SOURCE = '''module bounds
contains
subroutine inner(a, b, lo, hi, n)
integer,intent(in)::lo,hi,n
real(8),intent(in)::a(lo:hi,1:n+1)
real(8),intent(out)::b(lo:hi,1:n+1)
integer::i,k
do k=1,n+1
 do i=lo,hi
  b(i,k)=2*a(i,k)+10*k
 enddo
enddo
end subroutine
subroutine outer(a,b,lo,hi,kts,n)
integer,intent(in)::lo,hi,kts,n
real(8),intent(in)::a(lo:hi,kts-1:kts+n-1)
real(8),intent(out)::b(lo:hi,kts-1:kts+n-1)
call inner(a,b,lo,hi,n)
end subroutine
end module
'''


def test_symbolic_lower_bounds_read_and_write(tmp_path: Path):
    graph = build_sdfg(SOURCE, tmp_path, name='outer', entry='outer').build()
    compiled = graph.compile()
    for kts, n in [(1, 3), (4, 7), (-2, 3)]:
        a = np.arange(5*(n+1), dtype=np.float64).reshape((5,n+1), order='F').copy(order='F')
        b = np.full_like(a, -999., order='F')
        args = dict(a=a,b=b,lo=-2,hi=2,kts=kts,n=n)
        for name in graph.arglist():
            if name.startswith('offset_') and name.endswith('_d1'):
                args[name] = kts-1
        compiled(**args)
        np.testing.assert_array_equal(b,2*a+10*np.arange(1,n+2)[None,:])
