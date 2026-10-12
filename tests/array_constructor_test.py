"""Array constructors must compute every element, however flang fills their temp.

flang builds a constructor in a heap ``.tmp.arrayctor``: with inline element stores when every item is a scalar (or an
implied-do of known extent), and through the Fortran runtime (``_FortranAInitArrayConstructorVector`` /
``_FortranAPushArrayConstructor*``) when an item is array-valued.  ``hlfir-materialise-array-constructors`` gives both
a named local array.

Before it, the runtime-filled temp was invisible to AST extraction (the assignment came out as ``u = ?``), and the
inline stores were retargeted at the destination without their reads (``_out_b = _in_q_0`` with no memlet behind
``_in_q_0``), at the wrong elements of a section destination, and wrongly when an item reads the destination.

Runtime path: scalar and array items from record members and their sections (also inside a loop), whole arrays with a
lower bound other than 1, a rank-2 item (array element order), a strided section, items reading the destination, a
constructor inside an elementwise expression.  Inline path: element reads of arrays with their own lower bounds into
record members inside a loop, a section destination, items reading the destination, implied-do items (also nested and
inside a loop), whose positions come from a counter flang advances before the store at the old position."""

import numpy as np

from tests._util import build_sdfg

_RECORD_SRC = """
module cell_pack_mod
  implicit none
  type side_pair
    real(8) :: l, r
  end type
  type side_vec3
    real(8) :: l(3), r(3)
  end type
contains
  subroutine pack_cells(n, dens, vel, mag, ener, packed)
    integer, intent(in) :: n
    real(8), intent(in) :: dens(n), vel(3, n), mag(3, n), ener(n)
    real(8), intent(out) :: packed(7, n)
    type(side_pair) :: rho, e
    type(side_vec3) :: v, b
    real(8) :: u(7)
    integer :: i
    do i = 1, n
      rho%l = dens(i)
      v%l = vel(:, i)
      b%l = mag(:, i)
      e%l = ener(i)
      u = [rho%l, rho%l*v%l(1:3), b%l(2:3), e%l]
      packed(:, i) = u
    end do
  end subroutine pack_cells
end module cell_pack_mod
"""

_MIXED_SRC = """
subroutine mix_items(s, w, z, m, t, out)
  real(8), intent(in) :: s, w(5), z(0:3), m(2, 2), t
  real(8), intent(out) :: out(14)
  out = [s, z, m, w(1:5:2), t, 2*s]
end subroutine mix_items
"""

_SELF_SRC = """
subroutine swap_halves(w)
  real(8), intent(inout) :: w(4)
  w = [w(3:4), w(1:2)]
end subroutine swap_halves
"""

_EXPR_SRC = """
subroutine shifted(a, y, z, out)
  real(8), intent(in) :: a, y(3), z(4)
  real(8), intent(out) :: out(3)
  out = y + 2 * [a, z(2:3)]
end subroutine shifted
"""


def test_record_members_and_sections(tmp_path):
    sdfg = build_sdfg(_RECORD_SRC, tmp_path, name="pack_cells", entry="cell_pack_mod::pack_cells").build()
    sdfg.validate()
    rng = np.random.default_rng(7)
    n = 5
    dens = rng.uniform(0.5, 2.0, n)
    vel = np.asfortranarray(rng.standard_normal((3, n)))
    mag = np.asfortranarray(rng.standard_normal((3, n)))
    ener = rng.uniform(1.0, 3.0, n)
    packed = np.asfortranarray(np.zeros((7, n)))
    sdfg(n=n, dens=dens, vel=vel, mag=mag, ener=ener, packed=packed)
    expected = np.empty((7, n))
    expected[0] = dens
    expected[1:4] = dens * vel
    expected[4:6] = mag[1:3]
    expected[6] = ener
    np.testing.assert_allclose(packed, expected, rtol=1e-14, atol=0)


def test_lower_bounds_rank_two_and_strided_items(tmp_path):
    sdfg = build_sdfg(_MIXED_SRC, tmp_path, name="mix_items").build()
    sdfg.validate()
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = np.array([10.0, 11.0, 12.0, 13.0])
    m = np.asfortranarray(np.array([[21.0, 23.0], [22.0, 24.0]]))
    out = np.zeros(14)
    sdfg(s=0.5, w=w, z=z, m=m, t=-7.0, out=out)
    expected = np.concatenate([[0.5], z, m.flatten(order="F"), w[0:5:2], [-7.0, 1.0]])
    np.testing.assert_array_equal(out, expected)


def test_items_reading_the_destination(tmp_path):
    sdfg = build_sdfg(_SELF_SRC, tmp_path, name="swap_halves").build()
    sdfg.validate()
    w = np.array([1.0, 2.0, 3.0, 4.0])
    sdfg(w=w)
    np.testing.assert_array_equal(w, [3.0, 4.0, 1.0, 2.0])


def test_constructor_inside_an_expression(tmp_path):
    sdfg = build_sdfg(_EXPR_SRC, tmp_path, name="shifted").build()
    sdfg.validate()
    y = np.array([1.0, 2.0, 3.0])
    z = np.array([5.0, 6.0, 7.0, 8.0])
    out = np.zeros(3)
    sdfg(a=0.25, y=y, z=z, out=out)
    np.testing.assert_array_equal(out, y + 2 * np.array([0.25, 6.0, 7.0]))


_FACE_SRC = """
module face_fields_mod
  implicit none
  type side_vec3
    real(8) :: l(3), r(3)
  end type
contains
  subroutine face_fields(lo, hi, nv, ib, ql, qr, jump)
    integer, intent(in) :: lo, hi, nv, ib
    real(8), intent(in) :: ql(lo:hi, nv), qr(lo:hi, nv)
    real(8), intent(out) :: jump(lo:hi - 1)
    type(side_vec3) :: b
    integer :: j
    do j = lo, hi - 1
      b%l = [ql(j, ib), ql(j, ib + 1), ql(j, ib + 2)]
      b%r = [qr(j + 1, ib), qr(j + 1, ib + 1), qr(j + 1, ib + 2)]
      jump(j) = (b%l(1) - b%r(1)) + 2 * (b%l(2) - b%r(2)) + 3 * (b%l(3) - b%r(3))
    end do
  end subroutine face_fields
end module face_fields_mod
"""

_SCALAR_ITEMS_SRC = """
subroutine scalar_items(a, w, x)
  real(8), intent(in) :: a
  real(8), intent(inout) :: w(3), x(6)
  x(2:4) = [a, 2 * a, w(3)]
  w = [w(2), w(1), w(3) + a]
end subroutine scalar_items
"""


def test_scalar_element_reads_into_record_members(tmp_path):
    sdfg = build_sdfg(_FACE_SRC, tmp_path, name="face_fields", entry="face_fields_mod::face_fields").build()
    sdfg.validate()
    lo, hi, nv, ib = -2, 5, 6, 3
    rng = np.random.default_rng(11)
    ql = np.asfortranarray(rng.standard_normal((hi - lo + 1, nv)))
    qr = np.asfortranarray(rng.standard_normal((hi - lo + 1, nv)))
    jump = np.zeros(hi - lo)
    sdfg(lo=lo, hi=hi, nv=nv, ib=ib, ql=ql, qr=qr, jump=jump)
    c = ib - 1
    expected = sum((k + 1) * (ql[:-1, c + k] - qr[1:, c + k]) for k in range(3))
    np.testing.assert_allclose(jump, expected, rtol=1e-14, atol=1e-14)


def test_scalar_items_into_a_section_and_reading_the_destination(tmp_path):
    sdfg = build_sdfg(_SCALAR_ITEMS_SRC, tmp_path, name="scalar_items").build()
    sdfg.validate()
    w = np.array([1.0, 2.0, 3.0])
    x = np.arange(10.0, 16.0)
    sdfg(a=0.5, w=w, x=x)
    np.testing.assert_array_equal(x, [10.0, 0.5, 1.0, 3.0, 14.0, 15.0])
    np.testing.assert_array_equal(w, [2.0, 1.0, 3.5])


_IMPLIED_DO_SRC = """
subroutine implied_do_items(a, v)
  real(8), intent(in) :: a
  real(8), intent(out) :: v(5)
  integer :: i
  v = [-1d0, (i * a, i = 1, 3), 2d0]
end subroutine implied_do_items
"""

_NESTED_IMPLIED_DO_SRC = """
subroutine nested_implied_do(a, v)
  real(8), intent(in) :: a
  real(8), intent(out) :: v(6)
  integer :: i, j
  v = [((i * a + 10 * j, i = 1, 2), j = 1, 3)]
end subroutine nested_implied_do
"""

_IMPLIED_DO_IN_LOOP_SRC = """
subroutine implied_do_columns(n, a, m)
  integer, intent(in) :: n
  real(8), intent(in) :: a
  real(8), intent(out) :: m(4, n)
  integer :: i, j
  do j = 1, n
    m(:, j) = [real(j, 8), (i * a * j, i = 1, 3)]
  end do
end subroutine implied_do_columns
"""


def test_implied_do_items(tmp_path):
    """The implied-do's positions come from a counter flang advances before the store at the old position."""
    sdfg = build_sdfg(_IMPLIED_DO_SRC, tmp_path, name="implied_do_items").build()
    sdfg.validate()
    v = np.zeros(5)
    sdfg(a=0.5, v=v)
    np.testing.assert_array_equal(v, [-1.0, 0.5, 1.0, 1.5, 2.0])


def test_nested_implied_do(tmp_path):
    sdfg = build_sdfg(_NESTED_IMPLIED_DO_SRC, tmp_path, name="nested_implied_do").build()
    sdfg.validate()
    a = 0.25
    v = np.zeros(6)
    sdfg(a=a, v=v)
    expected = [i * a + 10 * j for j in range(1, 4) for i in range(1, 3)]
    np.testing.assert_array_equal(v, expected)


def test_implied_do_inside_a_loop(tmp_path):
    sdfg = build_sdfg(_IMPLIED_DO_IN_LOOP_SRC, tmp_path, name="implied_do_columns").build()
    sdfg.validate()
    n, a = 3, 0.5
    m = np.zeros((4, n), order="F")
    sdfg(n=n, a=a, m=m)
    j = np.arange(1, n + 1)
    expected = np.vstack([j, *(i * a * j for i in range(1, 4))])
    np.testing.assert_array_equal(m, expected)
