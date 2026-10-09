"""An array of records whose members the kernel picks by constant element indices is handed over member by member:
its packed companion splits into one array per element, and the binding aliases each element's member in place
instead of packing all of them into a buffer and back."""

import ctypes
from pathlib import Path

import numpy as np

from dace_fortran.bindings.build_fortran_library import build_fortran_library
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.frozen_signature import get_frozen_signature, refreeze
from dace_fortran.bindings.record_split import split_record_companions
from dace_fortran.bindings.specialization import Specialization
from tests._util import build_sdfg

_SRC = """
module field_types
  implicit none
  type field
    real(8), pointer, dimension(:) :: sf => null()
  end type field
end module field_types

module kern
  implicit none
contains
  subroutine pick(q, out)
    use field_types, only: field
    type(field), dimension(:), intent(inout) :: q
    real(8), dimension(:), intent(inout) :: out
    integer :: j
    do j = lbound(q(1)%sf, 1), ubound(q(1)%sf, 1)
      out(j - lbound(q(1)%sf, 1) + 1) = q(1)%sf(j) + 10*q(3)%sf(j)
      q(2)%sf(j) = 2*q(1)%sf(j)
    end do
  end subroutine pick
end module kern
"""

# Three fields over lo:hi; the second one's extent is a parameter, to make it disagree with its siblings.
_SHIM = """
subroutine run_pick(lo, hi, hi2, a, b, c, out, declined_out) bind(c, name="run_pick")
  use iso_c_binding
  use field_types, only: field
  use pick_dace_bindings, only: pick_dace
  integer(c_int), value :: lo, hi, hi2
  real(c_double), intent(inout) :: a(lo:hi), b(lo:hi2), c(lo:hi), out(hi - lo + 1)
  integer(c_int), intent(out) :: declined_out
  type(field) :: q(3)
  real(c_double), target, allocatable :: fa(:), fb(:), fc(:)
  logical :: declined
  allocate(fa(lo:hi), fb(lo:hi2), fc(lo:hi))
  fa = a
  fb = b
  fc = c
  q(1)%sf => fa
  q(2)%sf => fb
  q(3)%sf => fc
  call pick_dace(q, out, dace_declined=declined)
  b = fb
  declined_out = merge(1, 0, declined)
end subroutine run_pick
"""


def _library(tmp_path: Path):
    src = tmp_path / "pick.f90"
    src.write_text(_SRC)
    builder = build_sdfg(_SRC, tmp_path / "sdfg", name="pick", entry="kern::pick")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "pick"
    sdfg.build_folder = str(tmp_path / "dacecache")
    general = get_frozen_signature(sdfg)
    split = split_record_companions(sdfg, plan)
    refreeze(sdfg)
    iface = build_auto_interface(sdfg._fortran_interface_raw, "pick")
    shim = tmp_path / "shim.f90"
    shim.write_text(_SHIM)
    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(tmp_path / "lib"),
        prelude_sources=[src],
        extra_sources=[shim],
        specialization=Specialization({}, general),
    )
    fn = lib.load().run_pick
    fn.restype = None
    fn.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 5
    return split, fn, next((tmp_path / "lib").glob("*_bindings.f90")).read_text()


def test_members_picked_by_constant_index_are_aliased_in_place(tmp_path):
    split, fn, text = _library(tmp_path)
    assert list(split.values()) == [{0: "q_sf_0", 1: "q_sf_1", 2: "q_sf_2"}]
    # No buffer to pack into or back out of: each element's member is the SDFG's array.
    assert "allocate(q_sf(" not in text
    assert "q_sf_1 => q(i1)%sf" in text

    lo, hi = -2, 3
    n = hi - lo + 1
    a, b, c = np.arange(n, dtype=np.float64), np.zeros(n), np.full(n, 0.5)
    out, declined = np.zeros(n), np.zeros(1, dtype=np.int32)
    fn(lo, hi, hi, a.ctypes.data, b.ctypes.data, c.ctypes.data, out.ctypes.data, declined.ctypes.data)
    assert not declined[0]
    np.testing.assert_array_equal(out, a + 5)
    np.testing.assert_array_equal(b, 2 * a)  # written straight into the caller's member


def test_a_member_of_another_shape_declines_the_call(tmp_path):
    _split, fn, _text = _library(tmp_path)
    lo, hi = 1, 4
    a, c = np.ones(4), np.ones(4)
    b = np.full(5, -1.0)  # one element longer than its siblings
    out, declined = np.full(4, -1.0), np.zeros(1, dtype=np.int32)
    fn(lo, hi, hi + 1, a.ctypes.data, b.ctypes.data, c.ctypes.data, out.ctypes.data, declined.ctypes.data)
    assert declined[0]
    np.testing.assert_array_equal(out, -1.0)
    np.testing.assert_array_equal(b, -1.0)
