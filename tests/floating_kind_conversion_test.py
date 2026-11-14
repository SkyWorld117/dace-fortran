"""REAL-kind conversions retain both the source rounding and destination arithmetic."""
import numpy as np
import pytest

from tests._util import build_sdfg


@pytest.mark.parametrize('expression,kind,expected', [
    ('dble(a)+dble(b)-dble(a)', 8, 1.),
    ('dble(a+b)', 8, 16777216.),
    ('real(dble(a)+dble(b)-dble(a))', 4, 1.),
])
def test_widen_before_arithmetic_and_round_before_widen(tmp_path, expression, kind, expected):
    source = f"""
subroutine kind_cast_probe(a,b,out)
  real(4),intent(in) :: a,b
  real({kind}),intent(out) :: out
  out={expression}
end subroutine
"""
    graph = build_sdfg(source,tmp_path,name='kind_cast_probe').build()
    graph.build_folder = str(tmp_path/'build')
    result = np.zeros(1,dtype=np.float64 if kind==8 else np.float32)
    graph(a=np.float32(16777216.),b=np.float32(1.),out=result)
    assert result[0] == expected
