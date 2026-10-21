"""A whole array assigned from itself, under its own name or through an alias, must not put a copy into the SDFG
whose destination access node feeds its source (the state then has a cycle).

Ways to get there: a plain ``a = a``; a module record assigned from a dummy that the caller associated with that
same record -- once the callee is inlined, the dummy's members are the record's own members, so every array
member becomes a copy onto itself; and an array assigned from a pointer to it, whose view hangs off the array's
access node."""

import numpy as np

from tests._util import build_sdfg

_RECORD_SRC = """
module span_mod
  implicit none
  type span_info
    integer :: lo, hi
    real(8) :: w(3)
  end type
  type(span_info) :: cur
contains
  subroutine take_span(d, x, out)
    type(span_info), intent(in) :: d
    real(8), intent(in) :: x(10)
    real(8), intent(out) :: out
    integer :: j
    cur = d
    out = 0
    do j = cur%lo, cur%hi
      out = out + x(j) * cur%w(1 + mod(j, 3))
    end do
  end subroutine take_span

  subroutine kb(x, out)
    real(8), intent(in) :: x(10)
    real(8), intent(out) :: out
    call take_span(cur, x, out)
  end subroutine kb
end module span_mod
"""

_ARRAY_SRC = """
subroutine self_assign(n, a, out)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  real(8), intent(out) :: out
  integer :: j
  a = a
  out = 0
  do j = 1, n
    out = out + j * a(j)
  end do
end subroutine self_assign
"""

_POINTER_SRC = """
subroutine copy_from_alias(x, out)
  real(8), target, intent(inout) :: x(6)
  real(8), intent(out) :: out
  real(8), pointer :: p(:)
  p => x
  x = p
  out = x(2) + 2 * x(5)
end subroutine copy_from_alias
"""


def _self_copies(sdfg):
    from dace.libraries.standard.nodes import CopyLibraryNode

    found = []
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, CopyLibraryNode):
            sources = {e.src.data for e in state.in_edges(node)}
            targets = {e.dst.data for e in state.out_edges(node)}
            if sources & targets:
                found.append(node.label)
    return found


def test_record_assigned_from_a_dummy_associated_with_it(tmp_path):
    sdfg = build_sdfg(_RECORD_SRC, tmp_path, name="kb", entry="span_mod::kb").build()
    sdfg.validate()
    assert _self_copies(sdfg) == []
    x = np.arange(1, 11, dtype=np.float64)
    out = np.zeros(1)
    w = np.array([0.5, 2.0, -1.0])
    cur_w = w.copy()
    args = dict(x=x, out=out, cur_lo=2, cur_hi=7, cur_w=cur_w)
    sdfg(**{k: v for k, v in args.items() if k in sdfg.arglist()})
    expected = sum(x[j - 1] * w[j % 3] for j in range(2, 8))
    assert out[0] == expected
    np.testing.assert_array_equal(cur_w, w)


def test_whole_array_assigned_from_itself(tmp_path):
    sdfg = build_sdfg(_ARRAY_SRC, tmp_path, name="self_assign").build()
    sdfg.validate()
    assert _self_copies(sdfg) == []
    a = np.array([3.0, -1.0, 4.0, 1.5])
    before = a.copy()
    out = np.zeros(1)
    sdfg(n=4, a=a, out=out)
    np.testing.assert_array_equal(a, before)
    assert out[0] == sum((j + 1) * v for j, v in enumerate(before))


def test_whole_array_assigned_from_a_pointer_to_it(tmp_path):
    sdfg = build_sdfg(_POINTER_SRC, tmp_path, name="copy_from_alias").build()
    sdfg.validate()
    x = np.array([1.0, -2.0, 3.5, 0.25, 8.0, -6.0])
    before = x.copy()
    out = np.zeros(1)
    sdfg(x=x, out=out)
    np.testing.assert_array_equal(x, before)
    assert out[0] == before[1] + 2 * before[4]
