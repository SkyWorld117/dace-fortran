# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The AoR gap this suite cannot yet close: a **module-level** array of records.

`array_of_records_test.py` covers `type(t) :: arr(N)` as a DUMMY argument, and the flatten machinery
lowers it (`arr % x` -> `arr_x`). The same declaration at MODULE scope is not lowered, and the
failure is opaque: the pass emits the flat leaf NAMES and never the companion ARRAYS, so the build
ends at `unresolved free symbol(s): ['arr_x', 'arr_y']` -- or, downstream, at a bare `KeyError`.

WHY THIS MATTERS BEYOND THE TEST SUITE: it is what stops MFC's port from being brought current with
its own pinned revision. `m_variables_conversion`'s current `s_reference_curve` reads
`eos_coeffs(i)%r1`, where `eos_coeffs` is `type(eos_coefficients), dimension(num_fluids_max)` at
module scope -- and the ported kernels' call graph reaches it
(`s_compute_speed_of_sound` -> `s_eos_coefficients` -> `s_reference_curve`), so it cannot be scoped
away.

THE MECHANISM, measured, so the fix has a starting point rather than a mystery:

  * the HLFIR for the dummy form and the module form is otherwise IDENTICAL -- both emit
    `hlfir.designate %decl(%i)` and then `hlfir.designate %that{"x"}`;
  * the ONLY difference is that the dummy's `hlfir.declare` carries `dummy_scope`;
  * `passes/FlattenStructs.cpp::splitLocalAoRScalarMembers` (the non-argument path) skips the dummy
    case and then requires the root to be a **local** --
    `mlir::isa_and_nonnull<fir::AllocaOp>(rootDecl.getMemref().getDefiningOp())`. A module variable's
    root is `fir.address_of(@_GLOBAL)` -> `fir.global`, so it is rejected there;
  * and the rewrite half allocates a LOCAL allocatable companion per member, which is per-call and
    meaningless for a module variable -- a global-rooted variant must create a `fir::GlobalOp`
    companion and emit plan metadata so the Python side materialises the SDFG arrays.

These are marked `xfail(strict=False)`: they document the gap and will flip to xpass the moment the
bridge gains module-level AoR flattening, which is the signal to delete the markers.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

REASON = ("module-level array-of-records is not yet flattened: the pass emits the flat leaf names "
          "but not the companion arrays -- see this module's docstring for the located gate")

MODULE_VAR = """
module m
  type :: t
    real(kind=8) :: x
    real(kind=8) :: y
  end type
  type(t), dimension(3) :: arr
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    out = arr(1) % x + arr(2) % y
  end subroutine
end module
"""

# Closest to MFC's actual shape: a runtime index and only reads, as `s_reference_curve` does.
MODULE_VAR_RUNTIME_INDEX = """
module m
  type :: t
    real(kind=8) :: r1
    real(kind=8) :: a
  end type
  type(t), dimension(3) :: eos_coeffs
contains
  subroutine driver(i, v, out)
    integer, intent(in) :: i
    real(kind=8), intent(in) :: v
    real(kind=8), intent(out) :: out
    out = eos_coeffs(i) % a * exp(-(eos_coeffs(i) % r1 * v))
  end subroutine
end module
"""


@pytest.mark.xfail(strict=False, reason=REASON)
def test_module_level_aor_static_index(tmp_path):
    sdfg = build_sdfg(MODULE_VAR, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    assert "arr_x" in sdfg.arrays and "arr_y" in sdfg.arrays


@pytest.mark.xfail(strict=False, reason=REASON)
def test_module_level_aor_runtime_index(tmp_path):
    """MFC's shape. What must hold once this works, and is the whole point of the fix: the SDFG
    arrays are per-MEMBER and 1-D over the record index."""
    sdfg = build_sdfg(MODULE_VAR_RUNTIME_INDEX, tmp_path / "sdfg", name="driver",
                      entry="m::driver").build()
    for leaf in ("eos_coeffs_a", "eos_coeffs_r1"):
        assert leaf in sdfg.arrays, f"{leaf} missing"
        assert len(sdfg.arrays[leaf].shape) == 1, f"{leaf} should be 1-D over the record index"


def test_the_dummy_form_still_works(tmp_path):
    """The control: the form `array_of_records_test.py` covers must keep working, so a future fix
    cannot be merged on the strength of these xfails alone.  This one is NOT xfail."""
    src = """
module m
  type :: t
    real(kind=8) :: x
    real(kind=8) :: y
  end type
contains
  subroutine driver(arr, out)
    type(t), intent(in) :: arr(3)
    real(kind=8), intent(out) :: out
    out = arr(1) % x + arr(2) % y
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    assert "arr_x" in sdfg.arrays and "arr_y" in sdfg.arrays
