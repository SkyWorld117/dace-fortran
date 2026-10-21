"""A whole-array copy through an inlined dummy bound to a section copies that section.

The dummy of a trivial section (``call op(w(:, q))``) has no array of its own: every element access through it
routes to the actual, at the section's indices. A whole-array copy through it -- ``tmp = v``, ``v = tmp``, and
``v = [...]``, whose constructor is built in a named local array first -- is the copy of the slab the section covers:
here a column of a rank-2 scratch array of a loop, chosen by the loop's own iterator.
"""

import numpy as np

from tests._util import build_sdfg

_SRC = """
module m_column_weights
  implicit none
contains
  subroutine weights(m, v)
    double precision, intent(in) :: m(3)
    double precision, intent(inout) :: v(4)
    double precision :: tmp(4)
    tmp = v
    v = [m(1) * tmp(1), m(2) * tmp(2), m(3) * tmp(3), m(1) + tmp(4)]
  end subroutine weights

  subroutine driver(n, mom, out)
    integer, intent(in) :: n
    double precision, intent(in) :: mom(3, n)
    double precision, intent(out) :: out(4, n)
    double precision :: w(4, 3)
    integer :: i, q, k
    do i = 1, n
      do q = 1, 3
        do k = 1, 4
          w(k, q) = k + 10 * q + mom(2, i)
        end do
        call weights(mom(:, i), w(:, q))
      end do
      do k = 1, 4
        out(k, i) = w(k, 1) + w(k, 2) - w(k, 3)
      end do
    end do
  end subroutine driver
end module m_column_weights
"""


def test_copies_through_a_dummy_bound_to_a_column(tmp_path):
    sdfg = build_sdfg(_SRC, tmp_path, name="column_weights", entry="m_column_weights::driver").build()
    sdfg.validate()
    n = 3
    rng = np.random.default_rng(5)
    mom = np.asfortranarray(rng.uniform(0.5, 2.0, (3, n)))
    out = np.zeros((4, n), order="F")
    sdfg(n=n, mom=mom, out=out)
    expected = np.empty((4, n))
    for i in range(n):
        w = np.empty((4, 3))
        for q in range(3):
            tmp = np.arange(1, 5) + 10 * (q + 1) + mom[1, i]
            w[:, q] = [mom[0, i] * tmp[0], mom[1, i] * tmp[1], mom[2, i] * tmp[2], mom[0, i] + tmp[3]]
        expected[:, i] = w[:, 0] + w[:, 1] - w[:, 2]
    np.testing.assert_allclose(out, expected, rtol=1e-14, atol=0)
