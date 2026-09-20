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

# The READ cases now pass: `hlfir-flatten-global-scalar-reads` was extended to accept a STATIC
# module-level array of records (it previously required `isa<fir::RecordType>(g.getType())`, i.e. a
# scalar struct global) and to peel the element select that stands between the component read and the
# global.  The companion global it synthesises carries the record array's shape, so `arr(i)%x`
# becomes `arr_x(i)` -- one level shallower, same index.
WRITE_REASON = ("the pass is READ-ONLY by design (`!writtenGlobals.contains(sym)`): a written member "
                "is rejected rather than flattened, because the struct global would then be stale "
                "for every reader that is not rewritten. This is the pre-existing contract, not a "
                "regression -- MFC's `s_reference_curve` only reads `eos_coeffs`.")

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


def test_module_level_aor_static_index(tmp_path):
    sdfg = build_sdfg(MODULE_VAR, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    assert "arr_x" in sdfg.arrays and "arr_y" in sdfg.arrays


def test_module_level_aor_runtime_index(tmp_path):
    """MFC's shape -- `eos_coeffs(i) % a * exp(-(eos_coeffs(i) % r1 * v))`, read-only, runtime index.
    The point of the fix: the SDFG arrays are per-MEMBER and 1-D over the record index."""
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


@pytest.mark.xfail(strict=False, reason=WRITE_REASON)
def test_module_level_aor_write_is_not_yet_supported(tmp_path):
    """The gap that remains, stated rather than left implied. A member that is WRITTEN cannot be
    flattened while the struct global still exists and other readers may use it -- so the pass
    declines. MFC's `s_reference_curve` needs only reads, which is why the T6.4 blocker is closed
    without this."""
    src = """
module m
  type :: t
    real(kind=8) :: x
    real(kind=8) :: y
  end type
  type(t), dimension(3) :: arr
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    arr(1) % x = 2.0
    out = arr(1) % x + arr(2) % y
  end subroutine
end module
"""
    build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()


def test_the_flattened_read_computes_the_right_number(tmp_path):
    """Building is not enough: the whole risk of a flattening rewrite is that it produces the right
    SHAPE and the wrong ELEMENT.  `arr_x` is 1-D over the record index, so an off-by-one in the
    element select is exactly the mistake to look for -- and it is indistinguishable from a correct
    kernel unless the numbers are checked.

    Fortran's index is 1-based: `arr(1)` is element 0 of the companion."""
    src = """
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    r1 = np.array([0.5, 1.5, 2.5])
    a = np.array([1.0, 2.0, 3.0])
    v = 0.4
    for i in (1, 2, 3):
        out = np.zeros(1)
        sdfg(i=i, v=v, eos_coeffs_r1=r1, eos_coeffs_a=a, out=out)
        assert out[0] == a[i - 1] * np.exp(-(r1[i - 1] * v)), f"element {i} is wrong"
