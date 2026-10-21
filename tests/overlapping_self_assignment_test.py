"""An array assignment that reads its own target evaluates the whole right-hand side before it stores.

``v(2:3) = v(1:2) * 0.5`` reads ``v(2)``, which the same statement writes. Fortran evaluates the right-hand side
first, but the loop the bridge lowered the statement to read and wrote in place, so ``v(3)`` got half the NEW
``v(2)``. Every array-assignment form that loops over the target is covered: an elemental expression, a plain
section copy, a scalar broadcast to a section and to the whole array, an element of the target read by every
element, and a two-dimensional shift. A target read only at the element being written stays in place.
"""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg

CASES = {
    # statement, NumPy reference on a copy of v (1-D, length 4) and g (4 x 3)
    "elemental_forward": ("v(2:3) = v(1:2) * 0.5d0", lambda v, g: v.__setitem__(slice(1, 3), v[0:2] * 0.5)),
    "elemental_backward": ("v(1:2) = v(2:3) * 0.5d0", lambda v, g: v.__setitem__(slice(0, 2), v[1:3] * 0.5)),
    "section_copy": ("v(2:4) = v(1:3)", lambda v, g: v.__setitem__(slice(1, 4), v[0:3].copy())),
    "section_broadcast": ("v(1:3) = v(2) * 2d0", lambda v, g: v.__setitem__(slice(0, 3), v[1] * 2.0)),
    "whole_broadcast": ("v = v(2) * 2d0 + 1d0", lambda v, g: v.__setitem__(slice(None), v[1] * 2.0 + 1.0)),
    "column_shift": (
        "g(:, 2:3) = g(:, 1:2) + 1d0",
        lambda v, g: g.__setitem__((slice(None), slice(1, 3)), g[:, 0:2] + 1.0),
    ),
    "element_read": ("v = v / (v(1) + 1d0)", lambda v, g: v.__setitem__(slice(None), v / (v[0] + 1.0))),
    "same_element": ("v = v * 0.5d0 + v", lambda v, g: v.__setitem__(slice(None), v * 0.5 + v)),
}


def source(statement: str) -> str:
    return f"""
module m_overlap
  implicit none
contains
  subroutine driver(v, g)
    double precision, intent(inout) :: v(4), g(4, 3)
    {statement}
  end subroutine driver
end module m_overlap
"""


@pytest.mark.parametrize("case", sorted(CASES))
def test_right_hand_side_is_evaluated_before_the_store(tmp_path: Path, case: str):
    statement, reference = CASES[case]
    sdfg = build_sdfg(source(statement), tmp_path, name=f"overlap_{case}", entry="m_overlap::driver").build()
    v = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array(np.arange(1.0, 13.0).reshape(4, 3, order="F"), order="F")
    want_v, want_g = v.copy(), g.copy()
    reference(want_v, want_g)
    sdfg(v=v, g=g)
    np.testing.assert_array_equal(v, want_v)
    np.testing.assert_array_equal(g, want_g)
