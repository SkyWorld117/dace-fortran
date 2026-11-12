"""Reduction destinations preserve Fortran sections and declared lower bounds."""

import numpy as np
import pytest

from tests._util import build_sdfg


@pytest.mark.parametrize("columns", [1, 2])
def test_all_predicate_writes_only_the_selected_output_section(tmp_path, columns):
    source = f"""subroutine check(a,out,j)
      real, intent(in) :: a({columns},3)
      integer, intent(inout) :: out(-1:{columns + 1},2)
      integer, intent(in) :: j
      logical :: mask(-1:{columns + 1},2)
      integer :: i
      mask(0:{columns - 1},j) = all(a >= 0.,2)
      do i=0,{columns - 1}
        out(i,j) = merge(1,0,mask(i,j))
      end do
    end subroutine
"""
    graph = build_sdfg(source, tmp_path, name="check", entry="check").build()
    a = np.ones((columns, 3), dtype=np.float32, order="F")
    a[-1, 0] = -1
    out = np.full((columns + 3, 2), -7, dtype=np.int32, order="F")
    graph(a=a, out=out, j=np.int32(2))
    expected = np.full_like(out, -7)
    expected[1:columns + 1, 1] = np.all(a >= 0, axis=1)
    np.testing.assert_array_equal(out, expected)
