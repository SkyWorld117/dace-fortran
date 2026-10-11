"""An assumed-shape dummy whose declared lower bounds are components of elements of a module array of records
(``a(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)``), specialised to values, used both through a section and whole down a
chain of inlined dummies declared the same way: every dimension keeps its own lower bound -- the literal ``1:`` of
the last one as much as the specialised bounds of the others -- so the kernel's subscripts land on the right
elements."""

import numpy as np
import sympy

import dace
from dace.sdfg import nodes
from dace_fortran.builder import SDFGBuilder
from tests._util import compile_to_hlfir

_SRC = """
module bnd
  implicit none
  type bounds
    integer :: beg, end
  end type bounds
  type(bounds) :: lo(3)
end module bnd

module kern
  use bnd
  implicit none
contains
  subroutine scale(b, n1, n2, n3, nv)
    integer, intent(in) :: n1, n2, n3, nv
    real(8), intent(inout) :: b(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)
    integer :: i, j, k, l
    do i = 1, nv
      do l = lo(3)%beg, lo(3)%beg + n3 - 1
        do k = lo(2)%beg, lo(2)%beg + n2 - 1
          do j = lo(1)%beg, lo(1)%beg + n1 - 1
            b(j, k, l, i) = 2*b(j, k, l, i)
          end do
        end do
      end do
    end do
  end subroutine scale

  subroutine inner(a, n1, n2, n3, nv)
    integer, intent(in) :: n1, n2, n3, nv
    real(8), intent(inout) :: a(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)
    integer :: i, j, k, l
    do i = 1, nv
      do l = lo(3)%beg, lo(3)%beg + n3 - 1
        do k = lo(2)%beg, lo(2)%beg + n2 - 1
          do j = lo(1)%beg, lo(1)%beg + n1 - 2
            a(j, k, l, i) = a(j + 1, k, l, i) + 1000*i + 100*j + 10*k + l
          end do
        end do
      end do
    end do
  end subroutine inner

  subroutine mid(a, n1, n2, n3, nv)
    integer, intent(in) :: n1, n2, n3, nv
    real(8), intent(inout) :: a(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)
    call inner(a, n1, n2, n3, nv)
  end subroutine mid

  subroutine outer(a, n1, n2, n3, nv)
    integer, intent(in) :: n1, n2, n3, nv
    real(8), intent(inout) :: a(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)
    call scale(a(:, :, :, 1:nv), n1, n2, n3, nv)
    call mid(a, n1, n2, n3, nv)
  end subroutine outer
end module kern
"""

LB = (-3, -2, -1)


def _memlets_on(sdfg: dace.SDFG, name: str):
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode) and node.data == name:
            for e in parent.in_edges(node) + parent.out_edges(node):
                if e.data.data == name:
                    yield e.data


def _constant_term(expr) -> int:
    expr = sympy.sympify(dace.symbolic.pystr_to_symbolic(str(expr)))
    return int(expr.subs({s: 0 for s in expr.free_symbols}))


def _reference(a, n1, n2, n3, nv):
    out = 2 * a
    for i in range(1, nv + 1):
        for l in range(LB[2], LB[2] + n3):
            for k in range(LB[1], LB[1] + n2):
                for j in range(LB[0], LB[0] + n1 - 1):
                    out[j - LB[0], k - LB[1], l - LB[2], i - 1] = (
                        out[j + 1 - LB[0], k - LB[1], l - LB[2], i - 1] + 1000 * i + 100 * j + 10 * k + l
                    )
    return out


def test_each_dimension_keeps_its_own_specialised_lower_bound(tmp_path):
    hlfir = compile_to_hlfir(_SRC, tmp_path / "hlfir", name="kern")
    values = {f"bnd::lo({d + 1})%beg": lb for d, lb in enumerate(LB)}
    sdfg = SDFGBuilder(str(hlfir), entry="kern::outer", specialized_globals=values).build()
    sdfg.name = "outer_record_bounds"
    sdfg.build_folder = str(tmp_path / sdfg.name)
    sdfg.save(str(tmp_path / "outer.sdfgz"), compress=True)

    # Every element subscript of ``a`` is rebased by its own dimension's lower bound: ``j + 3`` for ``j`` from -3,
    # ``i - 1`` for the literal ``1:`` of the last dimension.
    seen = 0
    for memlet in _memlets_on(sdfg, "a"):
        begins = [b for b, _e, _s in memlet.subset.ranges]
        if any(len(sympy.sympify(dace.symbolic.pystr_to_symbolic(str(b))).free_symbols) != 1 for b in begins):
            continue  # a whole-array or ranged memlet on an outer scope
        seen += 1
        assert [_constant_term(b) for b in begins[1:]] == [-LB[1], -LB[2], -1], str(memlet.subset)
        assert _constant_term(begins[0]) in (-LB[0], -LB[0] + 1), str(memlet.subset)
    assert seen > 0

    n1, n2, n3, nv = 5, 4, 3, 2
    rng = np.random.default_rng(7)
    a = np.asfortranarray(rng.random((n1, n2, n3, nv)))
    want = _reference(a, n1, n2, n3, nv)
    args = {"a": a, "n1": n1, "n2": n2, "n3": n3, "nv": nv}
    sdfg(**{k: v for k, v in args.items() if k in sdfg.arglist()})
    np.testing.assert_allclose(a, want)
