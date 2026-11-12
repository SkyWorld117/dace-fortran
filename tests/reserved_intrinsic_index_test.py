"""Intrinsic index calls must not become reserved data identifiers."""
import numpy as np
import pytest
from dace_fortran.builder.access import apply_reserved
from tests._util import build_sdfg


def test_intrinsic_calls_and_reserved_data_names_are_distinct():
    assert apply_reserved('max(im, min(re, limit - 1)) + max[0]') == (
        'max(program_im, min(program_re, program_limit - 1)) + program_max[0]')
    assert apply_reserved('ABS(im - 1)') == 'abs(program_im - 1)'


def test_max_min_in_subscripts_execute(tmp_path):
    source = '''subroutine clamp(a, r)
      real, intent(in) :: a(8)
      real, intent(out) :: r(8)
      integer :: i
      do i = 1, 8
        r(i) = a(max(1,min(8,i-1)))
      end do
end subroutine
'''
    graph = build_sdfg(source, tmp_path, name='clamp', entry='clamp').build()
    code = '\n'.join(part.code for part in graph.generate_code())
    assert 'program_max(' not in code and 'program_min(' not in code
    a = np.arange(8, dtype=np.float32)
    result = np.empty_like(a)
    graph(a=a, r=result)
    np.testing.assert_array_equal(result, a[[0,0,1,2,3,4,5,6]])


def test_colliding_array_name_keeps_the_existing_explicit_rejection(tmp_path):
    source = '''subroutine lookup(max, a, r)
      integer, intent(in) :: max(4)
      real, intent(in) :: a(8)
      real, intent(out) :: r(4)
      integer :: i
      do i = 1, 4
        r(i) = a(max(i))
      end do
end subroutine
'''
    with pytest.raises(RuntimeError, match='collide with bridge-rendered intrinsics'):
        build_sdfg(source, tmp_path, name='lookup', entry='lookup').build()
