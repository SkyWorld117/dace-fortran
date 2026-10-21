"""``ALLOCATE(x, SOURCE=y)`` and ``MOLD=y`` size ``x`` from ``y``.

Flang lowers both through the Fortran runtime (``_FortranAAllocatableAllocateSource`` / ``ApplyMold``) instead of a
``fir.allocmem``, so the extents of ``x`` were never bound: ``x_d0`` / ``x_d1`` stayed free symbols of the SDFG,
which a caller had to supply and the generated binding passed uninitialised.  They are now assigned the extents of
``y`` at the allocation.
"""

import numpy as np
import pytest

from tests._util import build_sdfg

_SOURCE = """
subroutine scale_source(n, m, y, z)
  integer, intent(in) :: n, m
  real(8), intent(in) :: y(n, m)
  real(8), intent(out) :: z(n, m)
  real(8), allocatable :: x(:, :)
  allocate(x, source=y)
  x = 2.0d0 * x
  z = x
  deallocate(x)
end subroutine scale_source
"""

_MOLD = """
subroutine scale_mold(n, m, y, z)
  integer, intent(in) :: n, m
  real(8), intent(in) :: y(n, m)
  real(8), intent(out) :: z(n, m)
  real(8), allocatable :: x(:, :)
  allocate(x, mold=y)
  x = 3.0d0 * y
  z = x
  deallocate(x)
end subroutine scale_mold
"""


@pytest.mark.parametrize(
    ("source", "entry", "factor"), [(_SOURCE, "scale_source", 2.0), (_MOLD, "scale_mold", 3.0)], ids=["source", "mold"]
)
def test_the_allocation_takes_the_extents_of_its_source(tmp_path, source, entry, factor):
    sdfg = build_sdfg(source, tmp_path, name=entry, entry=entry).build()
    free = {str(s) for s in sdfg.free_symbols}
    assert not {"x_d0", "x_d1"} & free, f"the extents of x are left to the caller: {sorted(free)}"

    n, m = 3, 4
    y = np.asfortranarray(np.arange(n * m, dtype=np.float64).reshape(n, m))
    z = np.zeros((n, m), order="F")
    sdfg(n=n, m=m, y=y, z=z)
    np.testing.assert_array_equal(z, factor * y)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
