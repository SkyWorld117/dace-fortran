"""Module allocatable arrays, allocated with lower bounds that are components of elements of a module array of
records (``lo(k)%beg``), passed as actuals to an inlined routine whose dummies declare the same bounds
(``a(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)``) and are accessed both with literal subscripts (``b(0, ...)``,
``a(-1, ...)``) in one callee and with loop subscripts (``a(j + 1, ...)``, ``b(j, ...)``) in another: every access
is rebased by the arrays' own lower bounds, not by a bound guessed from the literal subscripts."""

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

module state
  use bnd
  implicit none
  real(8), allocatable, dimension(:, :, :, :) :: ql, qr
contains
  subroutine init(n1, n2, n3, nv)
    integer, intent(in) :: n1, n2, n3, nv
    allocate(ql(lo(1)%beg:lo(1)%beg + n1 - 1, lo(2)%beg:lo(2)%beg + n2 - 1, lo(3)%beg:lo(3)%beg + n3 - 1, 1:nv))
    allocate(qr(lo(1)%beg:lo(1)%beg + n1 - 1, lo(2)%beg:lo(2)%beg + n2 - 1, lo(3)%beg:lo(3)%beg + n3 - 1, 1:nv))
  end subroutine init
end module state

module kern
  use bnd
  use state
  implicit none
contains
  subroutine edges(a, b, n2, n3, nv)
    integer, intent(in) :: n2, n3, nv
    real(8), intent(inout) :: a(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)
    real(8), intent(inout) :: b(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)
    integer :: i, k, l
    do i = 1, nv
      do l = lo(3)%beg, lo(3)%beg + n3 - 1
        do k = lo(2)%beg, lo(2)%beg + n2 - 1
          b(0, k, l, i) = 2*a(-1, k, l, i)
        end do
      end do
    end do
  end subroutine edges

  subroutine solve(a, b, n1, n2, n3, nv)
    integer, intent(in) :: n1, n2, n3, nv
    real(8), intent(inout) :: a(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)
    real(8), intent(inout) :: b(lo(1)%beg:, lo(2)%beg:, lo(3)%beg:, 1:)
    integer :: i, j, k, l
    call edges(a, b, n2, n3, nv)
    do i = 1, nv
      do l = lo(3)%beg, lo(3)%beg + n3 - 1
        do k = lo(2)%beg, lo(2)%beg + n2 - 1
          do j = lo(1)%beg, lo(1)%beg + n1 - 2
            b(j, k, l, i) = a(j + 1, k, l, i) - b(j, k, l, i)
          end do
        end do
      end do
    end do
  end subroutine solve

  subroutine outer(n1, n2, n3, nv)
    integer, intent(in) :: n1, n2, n3, nv
    call solve(qr, ql, n1, n2, n3, nv)
  end subroutine outer
end module kern
"""

LB = (-3, -2, -1)


def _constant_term(expr) -> int:
    expr = sympy.sympify(dace.symbolic.pystr_to_symbolic(str(expr)))
    return int(expr.subs({s: 0 for s in expr.free_symbols}))


def _element_subsets(sdfg: dace.SDFG, name: str):
    """The subsets of the element accesses of ``name`` (no ranges: those are whole-array memlets on outer scopes)."""
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode) and node.data == name:
            for e in parent.in_edges(node) + parent.out_edges(node):
                if e.data.data != name or e.data.subset is None:
                    continue
                ranges = e.data.subset.ranges
                if any(sympy.sympify(dace.symbolic.pystr_to_symbolic(str(b - en))) != 0 for b, en, _ in ranges):
                    continue
                yield e.data.subset


def _reference(ql, qr, n1, n2, n3, nv):
    """``solve(qr, ql, ...)`` on arrays stored from their lower bounds: a is qr, b is ql."""
    a, b = qr, ql.copy()
    lb0, lb1, lb2 = LB
    for i in range(1, nv + 1):
        for l in range(lb2, lb2 + n3):
            for k in range(lb1, lb1 + n2):
                b[0 - lb0, k - lb1, l - lb2, i - 1] = 2 * a[-1 - lb0, k - lb1, l - lb2, i - 1]
    for i in range(1, nv + 1):
        for l in range(lb2, lb2 + n3):
            for k in range(lb1, lb1 + n2):
                for j in range(lb0, lb0 + n1 - 1):
                    b[j - lb0, k - lb1, l - lb2, i - 1] = (
                        a[j + 1 - lb0, k - lb1, l - lb2, i - 1] - b[j - lb0, k - lb1, l - lb2, i - 1]
                    )
    return b


def test_module_arrays_passed_to_record_bounded_dummies_keep_their_lower_bounds(tmp_path):
    hlfir = compile_to_hlfir(_SRC, tmp_path / "hlfir", name="kern")
    values = {f"bnd::lo({d + 1})%beg": lb for d, lb in enumerate(LB)}
    sdfg = SDFGBuilder(str(hlfir), entry="kern::outer", specialized_globals=values).build()
    sdfg.name = "outer_module_array_bounds"
    sdfg.build_folder = str(tmp_path / sdfg.name)

    # The arrays' lower bounds are the caller's: every dim of both stays a free offset symbol the caller binds, and
    # every subscript is the Fortran index minus that symbol -- the literals 0 and -1 of the edge copy included,
    # which only say what the bounds must at least allow (they are kept as the direct-call defaults, the floors).
    seen = {"ql": 0, "qr": 0}
    for name in ("ql", "qr"):
        for subset in _element_subsets(sdfg, name):
            seen[name] += 1
            for d, (begin, _end, _step) in enumerate(subset.ranges):
                expr = sympy.sympify(dace.symbolic.pystr_to_symbolic(str(begin)))
                symbols = {str(sym): sym for sym in expr.free_symbols}
                offset = symbols.get(f"offset_{name}_d{d}")
                assert offset is not None, f"{name}[{subset}]: dim {d} is not rebased by its offset symbol"
                rebased = sympy.expand(expr + offset)
                assert not any(str(sym).startswith("offset_") for sym in rebased.free_symbols), f"{name}[{subset}]"
                assert _constant_term(rebased) in ((0, 1, -1) if d == 0 else (0,)), f"{name}[{subset}]"
    assert seen["ql"] > 0 and seen["qr"] > 0
    assert sdfg._fortran_offset_floors == {"offset_ql_d0": 0, "offset_qr_d0": -1}

    # Bound to the allocation's bounds, the kernel computes what the Fortran does.
    n1, n2, n3, nv = 6, 4, 3, 2
    rng = np.random.default_rng(11)
    ql = np.asfortranarray(rng.random((n1, n2, n3, nv)))
    qr = np.asfortranarray(rng.random((n1, n2, n3, nv)))
    want = _reference(ql, qr, n1, n2, n3, nv)
    offsets = {f"offset_{name}_d{d}": lb for name in ("ql", "qr") for d, lb in enumerate((*LB, 1))}
    args = {"ql": ql, "qr": qr, "n1": n1, "n2": n2, "n3": n3, "nv": nv, **offsets}
    sdfg(**{k: v for k, v in args.items() if k in sdfg.arglist() or k in {str(s) for s in sdfg.free_symbols}})
    np.testing.assert_allclose(ql, want)
    np.testing.assert_array_equal(qr, np.asfortranarray(qr))
