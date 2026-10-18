"""Specialising a binding's SDFG to recorded values: a general binding records the integer symbols and scalar inputs
of each call, the SDFG is specialised to the ones that never changed, and the specialised binding runs only for calls
that hold those values -- any other call is declined, so the caller can run its own code.

Each test builds its own entry: two SDFG libraries of one entry name loaded into one process would share the first
one's code."""

import ctypes
from pathlib import Path

import numpy as np

from dace_fortran.bindings.build_fortran_library import build_fortran_library
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import OriginalInterface, build_auto_interface
from dace_fortran.bindings.frozen_signature import FrozenSignature, get_frozen_signature, refreeze
from dace_fortran.bindings.specialization import (
    Specialization,
    build_specialization_guard,
    recorded_invariants,
    specializable,
)
from dace_fortran.pipelines import specialize
from tests._util import build_sdfg

_N = 6

_SRC = """
module params_{entry}
  implicit none
  integer :: nv = 3
  logical :: scaled = .true.
  real(8) :: factor = 2.0d0
end module params_{entry}

module kern_{entry}
  implicit none
contains
  subroutine {entry}(x, y)
    use params_{entry}, only: nv, scaled, factor
    real(8), intent(in) :: x(:)
    real(8), intent(inout) :: y(:)
    integer :: i
    do i = 1, nv
      if (scaled) then
        y(i) = factor*x(i)
      else
        y(i) = x(i)
      end if
    end do
  end subroutine {entry}
end module kern_{entry}
"""

_SHIM = f"""
subroutine run_kern(xin, yout, nv_in, scaled_in, factor_in, declined_out) bind(c, name="run_kern")
  use iso_c_binding
  use params_{{entry}}, only: nv, scaled, factor
  use {{entry}}_dace_bindings, only: {{entry}}_dace
  real(c_double), intent(in) :: xin({_N})
  real(c_double), intent(inout) :: yout({_N})
  integer(c_int), value :: nv_in, scaled_in
  real(c_double), value :: factor_in
  integer(c_int), intent(out) :: declined_out
  logical :: declined
  nv = nv_in
  scaled = scaled_in /= 0
  factor = factor_in
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
    function that runs the kernel with the module variables set as given and returns ``(y, declined)``."""
    shim = tmp_path / "shim.f90"
    shim.write_text(_SHIM.format(entry=iface.entry, call=call.format(entry=iface.entry)))
    lib = build_fortran_library(
        sdfg, iface, plan, str(tmp_path / "lib"), prelude_sources=[src], extra_sources=[shim], **binding
    )
    fn = lib.load().run_kern
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_void_p]

    def run(nv: int = 3, scaled: bool = True, factor: float = 2.0) -> tuple[np.ndarray, bool]:
        x = np.arange(1, _N + 1, dtype=np.float64)
        y = np.full(_N, -1.0)
        declined = np.zeros(1, dtype=np.int32)
        fn(x.ctypes.data, y.ctypes.data, nv, int(scaled), factor, declined.ctypes.data)
        return y, bool(declined[0])

    return run


def test_recorded_invariants_keep_the_values_that_never_changed(tmp_path):
    record = tmp_path / "calls.symbols"
    record.write_text("n 3\nm 5\nf 2.50000000000000000E+000\n\nn 3\nm 6\nf 2.50000000000000000E+000\n\n")
    assert recorded_invariants(record) == {"n": 3, "f": 2.5}


def test_a_specialisation_to_no_values_only_clears_the_flag():
    general = Specialization({}, FrozenSignature(entry="k", mangled="_QPk", args=()))
    lines = build_specialization_guard(general, OriginalInterface(entry="k", args=()), FlattenPlan(entries=()))
    assert [ln.strip() for ln in lines[1:]] == ["if (present(dace_declined)) dace_declined = .false."]


def test_a_general_binding_records_the_values_of_every_call(tmp_path):
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
    run(nv=3, factor=0.1)
    run(nv=4, factor=0.1)
    first, second = (call.splitlines() for call in record.read_text().split("\n\n")[:2])
    assert "nv 3" in first and "nv 4" in second and "scaled 1" in first
    # Reals keep every bit, logicals read 0 or 1, and only what never changed is invariant.
    invariant = recorded_invariants(record)
    assert invariant["scaled"] == 1 and invariant["factor"] == 0.1 and "nv" not in invariant


def test_a_specialised_binding_runs_its_values_and_declines_others(tmp_path):
    sdfg, iface, plan, src = _build(tmp_path, "scale_spec")
    before = get_frozen_signature(sdfg)
    values = {"nv": 3, "scaled": 1, "factor": 2.0}
    assert set(values) <= specializable(before, iface, plan)
    symbols = {s: v for s, v in values.items() if s in before.free_symbols}
    specialize(sdfg, symbols=symbols, scalars={s: v for s, v in values.items() if s not in symbols})
    refreeze(sdfg)
    taken = {a.sdfg_name for a in get_frozen_signature(sdfg).args} | set(get_frozen_signature(sdfg).free_symbols)
    assert not set(values) & taken
    run = _library(
        tmp_path,
        sdfg,
        iface,
        plan,
        src,
        "call {entry}_dace(xin, yout, dace_declined=declined)",
        specialization=Specialization(values, before),
    )
    y, declined = run()
    assert not declined
    np.testing.assert_array_equal(y, [2, 4, 6, -1, -1, -1])
    # Any other value: the binding declines before touching anything.
    for other in ({"nv": 4}, {"scaled": False}, {"factor": 2.5}):
        y, declined = run(**other)
        assert declined, other
        np.testing.assert_array_equal(y, -1.0)
    # Without the flag, a mismatch stops naming the value; the guard reads the module variables themselves, which the
    # specialised SDFG no longer takes.
    text = next((tmp_path / "lib").glob("*_bindings.f90")).read_text()
    assert "if (nv__mod /= 3) dace_mismatch = 'nv'" in text and "error stop" in text
    assert "if (scaled__mod .neqv. .true.) dace_mismatch = 'scaled'" in text
    assert "if (factor__mod /= 2.0_c_double) dace_mismatch = 'factor'" in text


# A small allocatable integer module array the kernel indexes with: the record holds its elements, the guard checks
# them (and its size, and that it is allocated) on every call.
_IDX_SRC = """
module idxs_{entry}
  implicit none
  integer, allocatable :: idx(:)
end module idxs_{entry}

module kern_{entry}
  implicit none
contains
  subroutine {entry}(x, y)
    use idxs_{entry}, only: idx
    real(8), intent(in) :: x(:)
    real(8), intent(inout) :: y(:)
    integer :: k
    do k = 1, 2
      y(idx(k)) = 2*x(idx(k))
    end do
  end subroutine {entry}
end module kern_{entry}
"""

_IDX_SHIM = f"""
subroutine run_kern(xin, yout, n, first, declined_out) bind(c, name="run_kern")
  use iso_c_binding
  use idxs_{{entry}}, only: idx
  use {{entry}}_dace_bindings, only: {{entry}}_dace
  real(c_double), intent(in) :: xin({_N})
  real(c_double), intent(inout) :: yout({_N})
  integer(c_int), value :: n, first
  integer(c_int), intent(out) :: declined_out
  logical :: declined
  integer :: k
  if (allocated(idx)) deallocate(idx)
  if (n > 0) then
    allocate(idx(n))
    idx = [(first + 2 * (k - 1), k = 1, n)]
  end if
  {{call}}
  declined_out = merge(1, 0, declined)
end subroutine run_kern
"""


def _idx_library(tmp_path: Path, entry: str, call: str, **binding):
    """The kernel of _IDX_SRC behind a binding (``binding`` and ``specialization`` of the general SDFG, given as a
    function of its frozen signature); returns a function of the length and first element of ``idx`` returning
    ``(y, declined)``."""
    source = _IDX_SRC.format(entry=entry)
    src = tmp_path / f"{entry}.f90"
    src.write_text(source)
    builder = build_sdfg(source, tmp_path / "sdfg", name=entry, entry=f"kern_{entry}::{entry}")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = entry
    sdfg.build_folder = str(tmp_path / "dacecache")
    iface = build_auto_interface(sdfg._fortran_interface_raw, entry)
    if "specialization" in binding:
        binding["specialization"] = binding["specialization"](get_frozen_signature(sdfg))
    shim = tmp_path / "shim.f90"
    shim.write_text(_IDX_SHIM.format(entry=entry, call=call.format(entry=entry)))
    lib = build_fortran_library(sdfg, iface, plan, str(tmp_path / "lib"), prelude_sources=[src], extra_sources=[shim],
                                **binding)
    fn = lib.load().run_kern
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

    def run(n: int = 2, first: int = 1) -> tuple[np.ndarray, bool]:
        x = np.arange(1, _N + 1, dtype=np.float64)
        y = np.full(_N, -1.0)
        declined = np.zeros(1, dtype=np.int32)
        fn(x.ctypes.data, y.ctypes.data, n, first, declined.ctypes.data)
        return y, bool(declined[0])

    return run


def test_a_general_binding_records_the_elements_of_small_integer_module_arrays(tmp_path):
    record = tmp_path / "idx_rec.symbols"
    run = _idx_library(tmp_path, "idx_rec", "call {entry}_dace(xin, yout)\n  declined = .false.",
                       record_symbols_to=record)
    run(n=2, first=1)
    run(n=2, first=2)
    first, second = (call.splitlines() for call in record.read_text().split("\n\n")[:2])
    assert "idx[0] 1" in first and "idx[1] 3" in first and "idx[0] 2" in second and "idx[1] 4" in second


def test_a_specialised_binding_checks_the_elements_of_small_integer_module_arrays(tmp_path):
    run = _idx_library(tmp_path, "idx_spec", "call {entry}_dace(xin, yout, dace_declined=declined)",
                       specialization=lambda frozen: Specialization({}, frozen, {"idx": (1, 3)}))
    y, declined = run(n=2, first=1)
    assert not declined
    np.testing.assert_array_equal(y, [2, -1, 6, -1, -1, -1])
    for n, first in ((2, 2), (3, 1), (0, 1)):  # other elements, another size, not allocated
        y, declined = run(n=n, first=first)
        assert declined, (n, first)
        np.testing.assert_array_equal(y, -1.0)


def test_a_binding_recording_per_rank_writes_the_file_of_its_rank(tmp_path, monkeypatch):
    sdfg, iface, plan, src = _build(tmp_path, "scale_rank")
    record = tmp_path / "scale_rank.symbols"
    run = _library(
        tmp_path,
        sdfg,
        iface,
        plan,
        src,
        "call {entry}_dace(xin, yout)\n  declined = .false.",
        record_symbols_to=record,
        record_per_rank=True,
    )
    from dace_fortran.bindings.specialization import RANK_VARIABLES

    for var in RANK_VARIABLES:  # the test itself may run under a launcher
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    run(nv=3)
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK")
    run(nv=4)
    assert "nv 3" in (tmp_path / "scale_rank.symbols.3").read_text()
    assert "nv 4" in record.read_text()  # a process no launcher started records to the file itself


def test_extra_procedures_are_part_of_the_binding_module(tmp_path):
    sdfg, iface, plan, src = _build(tmp_path, "scale_extra")
    shim = tmp_path / "shim.f90"
    shim.write_text(_SHIM.format(entry=iface.entry, call=f"call {iface.entry}_dace(xin, yout)\n  declined = .false."))
    extra = (
        "  subroutine extra_probe(v) bind(c, name='extra_probe')\n"
        "    integer(c_int), intent(out) :: v\n"
        "    v = 42\n"
        "  end subroutine extra_probe\n"
    )
    lib = build_fortran_library(
        sdfg, iface, plan, str(tmp_path / "lib"), prelude_sources=[src], extra_sources=[shim], extra_procedures=extra
    )
    assert "extra_probe" in Path(lib.bindings_f90).read_text() if hasattr(lib, "bindings_f90") else True
    probe = lib.load().extra_probe
    probe.restype = None
    v = np.zeros(1, dtype=np.int32)
    probe(v.ctypes.data)
    assert v[0] == 42


def test_recorded_value_sets_are_the_values_that_did_not_stay_the_same(tmp_path):
    from dace_fortran.bindings.specialization import recorded_value_sets

    record = tmp_path / "ranks.symbols"
    record.write_text("bc_x_beg -3\nbc_x_end 0\nm 99\n\nbc_x_beg 0\nbc_x_end -3\nm 99\n\n")
    assert recorded_invariants(record) == {"m": 99}
    assert recorded_value_sets(record) == {"bc_x_beg": [-3, 0], "bc_x_end": [-3, 0]}
