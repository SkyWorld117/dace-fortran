"""A read-modify-write must stay after every earlier read of the value it overwrites.

Consecutive scalar statements lower into one SDFG state. In ``w = p * q`` / ``p = p * 2 + 1`` the update reads
``p`` from the same access node the first statement read it from and writes a new one, so the read node reaches the
written node through the update itself. ``statement_order`` took that path for an ordering of the whole pair and
added nothing, which left the first statement's read free to run after the store: ``w`` saw the updated ``p``.

The entry's scalar dummies and a local scalar are both covered. ``test_matches_reference`` also runs the graph with
every state's nodes re-inserted in reverse, a permutation that is harmless on a correctly ordered graph and flips the
tie-break on a hazardous one.
"""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg
from tests.statement_order_hazard_test import reverse_node_order

SRC = """
module m_read_modify_write
  implicit none
contains
  subroutine driver(p, q, r)
    double precision, intent(inout) :: p, q
    double precision, intent(out) :: r
    double precision :: w, t0, t1, t2, t3
    w = p * q
    p = p * 2d0 + 1d0
    t1 = p - 0.5d0
    q = q * 2d0 + 1d0
    t2 = q - 0.5d0
    t0 = w + 0.25d0
    w = w * 2d0 + 1d0
    t3 = w - 0.5d0
    r = w + t0 + t1 + t2 + t3
  end subroutine driver
end module m_read_modify_write
"""


def reference(p: float, q: float):
    w = p * q
    p = p * 2.0 + 1.0
    t1 = p - 0.5
    q = q * 2.0 + 1.0
    t2 = q - 0.5
    t0 = w + 0.25
    w = w * 2.0 + 1.0
    t3 = w - 0.5
    return p, q, w + t0 + t1 + t2 + t3


@pytest.mark.parametrize("variant", ["asbuilt", "reversed"])
def test_matches_reference(tmp_path: Path, variant: str):
    sdfg = build_sdfg(SRC, tmp_path / variant, name=f"rmw_{variant}", entry="m_read_modify_write::driver").build()
    if variant == "reversed":
        reverse_node_order(sdfg)
    p, q, r = np.array([0.5]), np.array([0.75]), np.zeros(1)
    sdfg(p=p, q=q, r=r)
    np.testing.assert_array_equal([p[0], q[0], r[0]], reference(0.5, 0.75))
