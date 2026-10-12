"""An array constructor with an array-valued item (``u = [s, s*w(1:3), z(2:3), t]``) is built by flang through the
Fortran runtime (``_FortranAInitArrayConstructorVector`` / ``_FortranAPushArrayConstructor*``) rather than with inline
element stores; the SDFG must still compute every element.

Before ``hlfir-expand-runtime-array-constructors`` the temp the runtime fills was invisible to AST extraction and the
assignment came out as ``u = ?``.  Covered: scalar and array items taken from record members and their sections (also
inside a loop), whole arrays with a lower bound other than 1, a rank-2 item (array element order), a strided section,
a constructor whose items read its own destination, and a constructor used inside an elementwise expression."""

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
