"""A module record assigned to itself through an inlined callee's dummy (``mb = m_in`` with ``m_in`` bound to
``mb``) moves no value: no member is copied onto itself.  A scalar member's self-copy would read it before anything
wrote it, and once simplification promotes the scalar to a symbol that read is a free symbol of the SDFG, which no
caller can supply."""

import dace
import numpy as np
from dace.sdfg import nodes

from tests._util import build_sdfg

_MODULE = """
module sm
  implicit none
  type info
    integer :: beg, end
    real(8) :: r
    logical :: flag
  end type
  type(info) :: mb
contains
  subroutine reset(m_in)
    type(info), intent(in) :: m_in
    mb = m_in
  end subroutine reset

  subroutine ks(x, out)
    real(8), intent(in) :: x(10)
    real(8), intent(out) :: out
    integer :: j
    mb%beg = 2
    mb%end = 5
    call reset(mb)
    out = 0
    do j = mb%beg, mb%end
      out = out + x(j)
    end do
  end subroutine ks
end module sm
"""


def test_record_assigned_to_itself_through_a_dummy(tmp_path):
    sdfg = build_sdfg(_MODULE, tmp_path, name="ks", entry="sm::ks").build()
    members = {"mb_r", "mb_flag"}
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.Tasklet):
            ins = {e.data.data for e in parent.in_edges(node)}
            outs = {e.data.data for e in parent.out_edges(node)}
            assert not (ins & outs & members), f"{node.label} copies a member onto itself"
    sdfg.simplify()
    assert not members & {str(s) for s in sdfg.free_symbols}
    x = np.arange(1, 11, dtype=np.float64)
    out = np.zeros(1)
    args = {"x": x, "out": out}
    for name, desc in sdfg.arglist().items():
        if name not in args:
            args[name] = np.zeros(desc.shape, desc.dtype.as_numpy_dtype()) if isinstance(desc, dace.data.Array) else 0
    sdfg(**args)
    assert out[0] == 2 + 3 + 4 + 5
