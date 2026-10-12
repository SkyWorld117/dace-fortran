"""An ``if``/``else if``/``else`` chain whose ``else`` holds a nested ``if`` is lowered to ONE conditional block, the
nested test appended to the chain: ``d == 1``, ``d == 2``, ``side == -1``, else. With a literal ``side`` the third
condition folds to true while ``d`` is still a loop variable. A conditional block takes the first branch whose
condition holds, so the earlier branches stay; only the later ones are dead. Simplification once dropped them and ran
the third branch for every ``d``: wrong numbers in a general build, and in a build specialised to two directions an
out-of-bounds subscript into the degenerate third-direction array once the loops are unrolled."""

import numpy as np
import pytest
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll

from dace_fortran.pipelines import specialize
from tests._util import build_sdfg, f2py_compile

SRC = """
subroutine pick_spacing(dir_in, side_in, nx, ny, nz, hx, hy, hz, ws)
  implicit none
  integer, intent(in) :: dir_in, side_in, nx, ny, nz
  real(8), intent(in) :: hx(0:nx), hy(0:ny), hz(0:nz)
  real(8), intent(out) :: ws(0:3)
  integer :: i
  if (dir_in == 1) then
    if (side_in == -1) then
      do i = 0, 3
        ws(i) = hx(i)
      end do
    else
      do i = 0, 3
        ws(i) = hx(nx - i)
      end do
    end if
  else if (dir_in == 2) then
    if (side_in == -1) then
      do i = 0, 3
        ws(i) = hy(i)
      end do
    else
      do i = 0, 3
        ws(i) = hy(ny - i)
      end do
    end if
  else
    if (side_in == -1) then
      do i = 0, 3
        ws(i) = hz(i)
      end do
    else
      do i = 0, 3
        ws(i) = hz(nz - i)
      end do
    end if
  end if
end subroutine pick_spacing

subroutine face_weights(ndims, nx, ny, nz, hx, hy, hz, total)
  implicit none
  integer, intent(in) :: ndims, nx, ny, nz
  real(8), intent(in) :: hx(0:nx), hy(0:ny), hz(0:nz)
  real(8), intent(out) :: total(3)
  real(8) :: ws(0:3)
  integer :: d
  total = 0.0d0
  do d = 1, ndims
    call pick_spacing(d, -1, nx, ny, nz, hx, hy, hz, ws)
    total(d) = ws(0) + 2.0d0*ws(1) + 3.0d0*ws(2) + 4.0d0*ws(3)
  end do
end subroutine face_weights
"""


def _inputs(nx: int, ny: int, nz: int) -> dict:
    rng = np.random.default_rng(7)
    return {
        "hx": rng.uniform(1.0, 2.0, nx + 1),
        "hy": rng.uniform(3.0, 4.0, ny + 1),
        "hz": rng.uniform(5.0, 6.0, nz + 1),
    }


def _reference(tmp_path, ndims: int, nx: int, ny: int, nz: int, arrays: dict) -> np.ndarray:
    ref = f2py_compile(SRC, tmp_path / "ref", "face_weights_ref", only=("face_weights",))
    return ref.face_weights(ndims, arrays["hx"], arrays["hy"], arrays["hz"])


def _chains(sdfg) -> list[int]:
    """Branch counts of the conditional blocks with four branches or more, anywhere in ``sdfg``."""
    return [
        len(n.branches)
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, ConditionalBlock) and len(n.branches) >= 4
    ]


def test_the_nested_test_is_part_of_the_chain(tmp_path):
    """The premise of the tests below: the frontend appends the nested ``if`` of the ``else`` to the chain."""
    sdfg = build_sdfg(SRC, tmp_path / "sdfg", name="face_weights", entry="face_weights").build()
    assert 4 in _chains(sdfg)


def test_a_decided_later_condition_keeps_the_earlier_branches(tmp_path):
    nx, ny, nz = 6, 5, 7
    arrays = _inputs(nx, ny, nz)
    sdfg = build_sdfg(SRC, tmp_path / "sdfg", name="face_weights", entry="face_weights").build()
    sdfg.simplify()
    total = np.zeros(3)
    sdfg(ndims=3, nx=nx, ny=ny, nz=nz, total=total, **arrays)
    np.testing.assert_array_equal(total, _reference(tmp_path, 3, nx, ny, nz, arrays))


@pytest.mark.parametrize("ndims, nz", [(2, 0), (3, 7)])
def test_a_build_specialised_to_fewer_directions_validates_and_runs(tmp_path, ndims, nz):
    """Specialised to two directions with a degenerate third-direction array, the third-direction branch must vanish
    once the direction loop is unrolled, not survive as the only one: its unrolled reads would subscript ``hz``
    outside its one element."""
    nx, ny = 6, 5
    arrays = _inputs(nx, ny, nz)
    sdfg = build_sdfg(SRC, tmp_path / "sdfg", name="face_weights", entry="face_weights").build()
    specialize(sdfg, symbols={"ndims": ndims, "nx": nx, "ny": ny, "nz": nz})
    sdfg.simplify(validate=False, validate_all=False)
    ShortLoopUnroll(8).apply_pass(sdfg, {})
    sdfg.simplify(validate=False, validate_all=False)
    sdfg.validate()
    total = np.zeros(3)
    sdfg(total=total, **arrays)
    np.testing.assert_array_equal(total, _reference(tmp_path, ndims, nx, ny, nz, arrays))
