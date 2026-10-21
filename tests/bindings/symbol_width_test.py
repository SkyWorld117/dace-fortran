"""A 32-bit caller bound promoted to a 64-bit SDFG symbol is passed with the C ABI's width."""

import ctypes
from pathlib import Path

import numpy as np

from dace_fortran.bindings import FlattenPlan, emit_bindings
from dace_fortran.bindings.fortran_interface import build_auto_interface
from tests._util import build_sdfg, gfortran_compile_so


def test_nonunit_lower_bound_through_compiled_binding(tmp_path: Path):
    source = """module shifted_mod
contains
subroutine shifted(a, lo, hi)
  integer, intent(in) :: lo, hi
  real(8), intent(inout) :: a(lo:hi)
  integer :: i
  do i=lo,hi
    a(i)=a(i)+real(i,8)
  end do
end subroutine
end module
"""
    builder = build_sdfg(source, tmp_path / "sdfg", name="shifted", entry="shifted_mod::shifted")
    sdfg = builder.build()
    sdfg.name = "shifted"
    compiled = sdfg.compile()
    library = Path(compiled._lib._library_filename)
    iface = build_auto_interface(builder._fortran_interface_raw, "shifted")
    binding = tmp_path / "binding.f90"
    emit_bindings(sdfg._frozen_signature, iface, FlattenPlan(entries=()), str(binding))
    driver = tmp_path / "driver.f90"
    driver.write_text("""subroutine run(a,lo,hi) bind(c)
use iso_c_binding
use shifted_dace_bindings
integer(c_int), value :: lo,hi
real(c_double) :: a(lo:hi)
call shifted_dace(a,lo,hi)
call shifted_dace_finalize()
end subroutine
""")
    built = tmp_path / "driver.so"
    gfortran_compile_so(built, binding, driver, mod_dir=tmp_path, link_so=library)
    fn = ctypes.CDLL(str(built)).run
    fn.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]
    a = np.arange(7, dtype=np.float64)
    expected = a + np.arange(-3, 4)
    fn(a, -3, 3)
    np.testing.assert_array_equal(a, expected)
