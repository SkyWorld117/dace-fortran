"""Specialising a binding's SDFG to recorded integer values: a general binding records the symbols of each call, the
SDFG is specialised to the ones that never changed, and the specialised binding runs only for calls that hold those
values -- any other call is declined, so the caller can run its own code.

Each test builds its own entry: two SDFG libraries of one entry name loaded into one process would share the first
one's code."""

import ctypes
from pathlib import Path

import numpy as np
from dace.sdfg.utils import specialize_symbols

from dace_fortran.bindings.build_fortran_library import build_fortran_library
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.frozen_signature import get_frozen_signature, refreeze
from dace_fortran.bindings.specialization import Specialization, recorded_invariants, specializable_symbols
from tests._util import build_sdfg

_N = 6

_SRC = """
module params_{entry}
  implicit none
  integer :: nv = 3
end module params_{entry}

module kern_{entry}
  implicit none
contains
  subroutine {entry}(x, y)
    use params_{entry}, only: nv
    real(8), intent(in) :: x(:)
    real(8), intent(inout) :: y(:)
    integer :: i
    do i = 1, nv
      y(i) = 2*x(i)
    end do
  end subroutine {entry}
end module kern_{entry}
"""

_SHIM = f"""
subroutine run_kern(xin, yout, nv_in, declined_out) bind(c, name="run_kern")
  use iso_c_binding
  use params_{{entry}}, only: nv
  use {{entry}}_dace_bindings, only: {{entry}}_dace
  real(c_double), intent(in) :: xin({_N})
  real(c_double), intent(inout) :: yout({_N})
  integer(c_int), value :: nv_in
  integer(c_int), intent(out) :: declined_out
  logical :: declined
  nv = nv_in
  {{call}}
  declined_out = merge(1, 0, declined)
end subroutine run_kern
"""


def _build(tmp_path: Path, entry: str):
    """The SDFG of ``entry``, its interface and flatten plan, and the kernel source."""
    source = _SRC.format(entry=entry)
    src = tmp_path / f"{entry}.f90"
    src.write_text(source)
    builder = build_sdfg(source, tmp_path / "sdfg", name=entry, entry=f"kern_{entry}::{entry}")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = entry
    sdfg.build_folder = str(tmp_path / "dacecache")
    iface = build_auto_interface(sdfg._fortran_interface_raw, entry)
    return sdfg, iface, plan, src


def _library(tmp_path: Path, sdfg, iface, plan, src: Path, call: str, **binding):
    """Build the library with a shim running ``call``; ``binding`` goes to :func:`build_fortran_library`.  Returns a
    function that runs the kernel with the module variable set to ``nv`` and gives ``(y, declined)``."""
    shim = tmp_path / "shim.f90"
    shim.write_text(_SHIM.format(entry=iface.entry, call=call.format(entry=iface.entry)))
    lib = build_fortran_library(
        sdfg, iface, plan, str(tmp_path / "lib"), prelude_sources=[src], extra_sources=[shim], **binding
    )
    fn = lib.load().run_kern
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

    def run(nv: int) -> tuple[np.ndarray, bool]:
        x = np.arange(1, _N + 1, dtype=np.float64)
        y = np.full(_N, -1.0)
        declined = np.zeros(1, dtype=np.int32)
        fn(x.ctypes.data, y.ctypes.data, nv, declined.ctypes.data)
        return y, bool(declined[0])

    return run


def test_recorded_invariants_keep_the_values_that_never_changed(tmp_path):
    record = tmp_path / "calls.symbols"
    record.write_text("n 3\nm 5\n\nn 3\nm 6\n\n")
    assert recorded_invariants(record) == {"n": 3}


def test_a_general_binding_records_the_symbols_of_every_call(tmp_path):
    sdfg, iface, plan, src = _build(tmp_path, "scale_rec")
    record = tmp_path / "scale_rec.symbols"
    run = _library(
        tmp_path,
        sdfg,
        iface,
        plan,
        src,
        "call {entry}_dace(xin, yout)\n  declined = .false.",
        record_symbols_to=record,
    )
    run(3)
    run(4)
    calls = record.read_text().split("\n\n")
    assert "nv 3" in calls[0].splitlines() and "nv 4" in calls[1].splitlines()
    assert "nv" not in recorded_invariants(record)


def test_a_specialised_binding_runs_its_values_and_declines_others(tmp_path):
    sdfg, iface, plan, src = _build(tmp_path, "scale_spec")
    before = get_frozen_signature(sdfg)
    assert "nv" in specializable_symbols(before, iface, plan)
    specialize_symbols(sdfg, {"nv": 3})
    refreeze(sdfg)
    assert "nv" not in get_frozen_signature(sdfg).free_symbols
    run = _library(
        tmp_path,
        sdfg,
        iface,
        plan,
        src,
        "call {entry}_dace(xin, yout, dace_declined=declined)",
        specialization=Specialization({"nv": 3}, before),
    )
    y, declined = run(3)
    assert not declined
    np.testing.assert_array_equal(y, [2, 4, 6, -1, -1, -1])
    # Another value: the binding declines before touching anything.
    y, declined = run(4)
    assert declined
    np.testing.assert_array_equal(y, -1.0)
    # Without the flag, a mismatch stops naming the symbol; the guard reads the module variable itself, which the
    # specialised SDFG no longer takes.
    text = next((tmp_path / "lib").glob("*_bindings.f90")).read_text()
    assert "if (nv /= 3) dace_mismatch = 'nv'" in text and "error stop" in text
    assert "nv__mod => nv" in text
