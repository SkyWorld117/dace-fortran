"""A pointer member the binding hands over apart from its record may point into another array the SDFG takes: a
module array or a dummy.  The SDFG takes the two as different containers, so the binding declines such a call and
the caller runs its own code; with the member pointing elsewhere the SDFG runs.  Either way the results are the
compiled kernel's."""

import ctypes

import numpy as np
import pytest

from dace_fortran.bindings.build_fortran_library import build_fortran_library
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.frozen_signature import get_frozen_signature
from dace_fortran.bindings.specialization import Specialization
from tests._util import build_sdfg

_N = 5

# The kernel updates the interior planes 0:1 of the module array t and of the dummy x, fills the boundary planes
# -1 and 2 of whatever g(1)%sf points at from that one's interior, then reads the boundary planes of t and x.
_SRC = """
module share_storage
  implicit none
  type field
    real(8), pointer :: sf(:, :, :) => null()
  end type field
  real(8), allocatable, target :: t(:, :, :)
  type(field), dimension(1) :: g
contains
  subroutine leaf(v, side, k, l)
    type(field), dimension(1:), intent(inout) :: v
    integer, intent(in) :: side, k, l
    if (side == 1) then
      v(1)%sf(k, -1, l) = v(1)%sf(k, 1, l)
    else
      v(1)%sf(k, 2, l) = v(1)%sf(k, 0, l)
    end if
  end subroutine leaf

  subroutine kern(x, n)
    integer, intent(in) :: n
    real(8), intent(inout), target :: x(n, -1:2, 2)
    integer :: k, l, m
    do l = 1, 2
      do m = 0, 1
        do k = 1, n
          t(k, m, l) = t(k, m, l) + 1000
          x(k, m, l) = x(k, m, l) + 2000
        end do
      end do
    end do
    do l = 1, 2
      do k = 1, n
        call leaf(g, 1, k, l)
        call leaf(g, 2, k, l)
      end do
    end do
    do l = 1, 2
      do k = 1, n
        t(k, 0, l) = t(k, 0, l) + t(k, -1, l)
        x(k, 0, l) = x(k, 0, l) + x(k, -1, l)
      end do
    end do
  end subroutine kern
end module share_storage
"""

# run_share calls the binding and, when it declines, the kernel itself; run_reference only the kernel.  which
# selects what g(1)%sf points at: 0 an array of its own, 1 the module array t, 2 the array passed as x.
_SHIM = f"""
module share_driver
  use iso_c_binding
  use share_storage, only: t, g, kern
  implicit none
  real(8), allocatable, target :: own(:, :, :), xs(:, :, :)
contains
  subroutine setup(which)
    integer(c_int), intent(in) :: which
    integer :: j, m, l
    if (allocated(t)) deallocate(t, own, xs)
    allocate(t({_N}, -1:2, 2), own({_N}, -1:2, 2), xs({_N}, -1:2, 2))
    do l = 1, 2
      do m = -1, 2
        do j = 1, {_N}
          t(j, m, l) = 100*l + 10*m + j
          xs(j, m, l) = 0.5d0 + 100*l + 10*m + j
          own(j, m, l) = 0.25d0 + 100*l + 10*m + j
        end do
      end do
    end do
    select case (which)
    case (1)
      g(1)%sf => t
    case (2)
      g(1)%sf => xs
    case default
      g(1)%sf => own
    end select
  end subroutine setup

  subroutine results(out)
    real(c_double), intent(out) :: out({24 * _N})
    out(1:{8 * _N}) = reshape(t, [{8 * _N}])
    out({8 * _N + 1}:{16 * _N}) = reshape(xs, [{8 * _N}])
    out({16 * _N + 1}:{24 * _N}) = reshape(own, [{8 * _N}])
  end subroutine results
end module share_driver

subroutine run_share(which, out, declined_out) bind(c, name="run_share")
  use share_driver
  use share_dace_bindings, only: share_dace
  integer(c_int), value :: which
  real(c_double), intent(out) :: out({24 * _N})
  integer(c_int), intent(out) :: declined_out
  logical :: declined
  call setup(which)
  call share_dace(xs, {_N}, dace_declined=declined)
  if (declined) call kern(xs, {_N})
  call results(out)
  declined_out = merge(1, 0, declined)
end subroutine run_share

subroutine run_reference(which, out) bind(c, name="run_reference")
  use share_driver
  integer(c_int), value :: which
  real(c_double), intent(out) :: out({24 * _N})
  call setup(which)
  call kern(xs, {_N})
  call results(out)
end subroutine run_reference
"""


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """The binding's text and the library's two entries, built once: two SDFG libraries of one entry name loaded
    into one process would share the first one's code."""
    tmp_path = tmp_path_factory.mktemp("share")
    src = tmp_path / "share.f90"
    src.write_text(_SRC)
    builder = build_sdfg(_SRC, tmp_path / "sdfg", name="share", entry="share_storage::kern")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "share"
    sdfg.build_folder = str(tmp_path / "dacecache")
    iface = build_auto_interface(sdfg._fortran_interface_raw, "share")
    shim = tmp_path / "shim.f90"
    shim.write_text(_SHIM)
    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(tmp_path / "lib"),
        prelude_sources=[src],
        extra_sources=[shim],
        specialization=Specialization({}, get_frozen_signature(sdfg)),
    ).load()
    lib.run_share.restype = None
    lib.run_share.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.run_reference.restype = None
    lib.run_reference.argtypes = [ctypes.c_int, ctypes.c_void_p]
    text = next((tmp_path / "lib").glob("*_bindings.f90")).read_text()
    return text, lib


def _run(lib, which: int) -> tuple[np.ndarray, bool, np.ndarray]:
    out, declined, ref = np.zeros(24 * _N), np.zeros(1, dtype=np.int32), np.zeros(24 * _N)
    lib.run_share(which, out.ctypes.data, declined.ctypes.data)
    lib.run_reference(which, ref.ctypes.data)
    return out, bool(declined[0]), ref


def test_the_member_is_handed_over_as_an_image_behind_a_storage_guard(library):
    text, _lib = library
    assert "g_sf(aos_g_sf_i0, " in text  # the member's image, packed from g(:)%sf
    assert "dace_spans_overlap(dace_spans, dace_nspans, dace_tag_a, dace_tag_b)" in text
    assert "dace_span_names(" in text  # the decline message names the two arrays that overlap


def test_a_member_apart_from_the_other_arrays_takes_the_call(library):
    _text, lib = library
    out, declined, ref = _run(lib, 0)
    assert not declined
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("which", [1, 2], ids=["module_array", "dummy"])
def test_a_member_sharing_storage_with_another_array_declines_the_call(library, which):
    _text, lib = library
    out, declined, ref = _run(lib, which)
    assert declined
    np.testing.assert_array_equal(out, ref)


def _signature_with_member() -> tuple:
    """A signature taking the image of a pointer member of a module array of records, a module array and a dummy."""
    from dace_fortran.bindings.fortran_interface import OriginalArg, OriginalInterface
    from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenArgKind, FrozenSignature, ModuleOrigin

    array = FrozenArgKind.ARRAY
    image = FrozenArg(
        "g_sf",
        "g_sf",
        array,
        "float64",
        2,
        ("1", "g_sf_d1"),
        aos_origin_mod="m",
        aos_origin_struct="g",
        aos_member_path="sf",
        aos_outer_rank=1,
        aos_struct_static=True,
        aos_member_pointer=True,
    )
    module_array = FrozenArg("t", "t", array, "float64", 1, ("t_d0",), module_origin_allocatable=True, is_written=True)
    dummy = FrozenArg("x", "x", array, "float64", 1, ("n",), intent="inout")
    n = FrozenArg("n", "n", FrozenArgKind.SYMBOL, "int32", 0, intent="in")
    frozen = FrozenSignature(
        "kern",
        "_QMmPkern",
        (image, module_array, dummy, n),
        free_symbols=("g_sf_d1", "n", "offset_g_sf_d1", "t_d0"),
        module_symbol_origins={"t": ModuleOrigin("m", "t")},
    )
    iface = OriginalInterface(
        "kern",
        (OriginalArg("x", "real(c_double)", 1, ("n",), "inout"), OriginalArg("n", "integer(c_int)", 0, (), "in")),
    )
    return frozen, iface


def test_the_guard_ranges_the_members_against_the_arrays_taken_in_place():
    from dace_fortran.bindings.storage_guard import IN_PLACE_WRITTEN, MEMBER_READ, build_storage_guard

    frozen, iface = _signature_with_member()
    text = "\n".join(build_storage_guard(frozen, iface, FlattenPlan(), declinable=True))
    assert f"call dace_span_add(dace_spans, dace_nspans, {MEMBER_READ}, storage_size(g(aos_g_sf_i0)%sf) / 8" in text
    assert f"call dace_span_add(dace_spans, dace_nspans, {IN_PLACE_WRITTEN}, storage_size(t__mod) / 8" in text
    assert f"call dace_span_add(dace_spans, dace_nspans, {IN_PLACE_WRITTEN}, storage_size(x) / 8" in text
    assert "dace_declined = .true." in text


def test_without_members_apart_or_in_a_sizing_run_there_is_no_guard(tmp_path):
    from dataclasses import replace

    from dace_fortran.bindings.emit_bindings import emit_bindings
    from dace_fortran.bindings.storage_guard import build_storage_guard

    frozen, iface = _signature_with_member()
    plain = replace(frozen, args=frozen.args[1:], free_symbols=("n", "t_d0"))
    assert build_storage_guard(plain, iface, FlattenPlan(), declinable=True) == []
    sizing = emit_bindings(
        frozen,
        iface,
        FlattenPlan(),
        tmp_path / "kern_bindings.f90",
        specialization=Specialization({}, frozen),
        record_symbols_to=tmp_path / "record",
        record_only=True,
    )
    assert "if (dace_spans_overlap(" not in sizing.read_text()  # the alias record shares the helpers, not the guard


def test_a_sizing_run_records_the_members_that_are_another_array(tmp_path):
    """The sizing binding appends ``alias <image> <element> <array>`` to ``<record>.aliases`` for every member of an
    element of a module array of records whose storage is exactly a module array's or a dummy's: the substitute
    folds the two."""
    from dace_fortran.bindings.emit_bindings import emit_bindings
    from dace_fortran.bindings.storage_guard import MEMBER_READ, build_alias_record

    frozen, iface = _signature_with_member()
    lines = build_alias_record(frozen, iface, FlattenPlan(), tmp_path / "record")
    text = "\n".join(lines)
    assert f"call dace_span_add(dace_spans, dace_nspans, {MEMBER_READ}, storage_size(g(aos_g_sf_i0)%sf) / 8" in text
    assert text.count(", aos_g_sf_i0)") == 1  # the element the member belongs to
    assert "'g_sf'" in text and "'t'" in text and "'x'" in text  # the SDFG names the record uses
    assert f"file='{tmp_path / 'record'}.aliases'" in text
    assert "call dace_spans_aliases(dace_spans, dace_nspans, dace_alias_unit, dace_span_sdfg)" in text
    sizing = emit_bindings(
        frozen,
        iface,
        FlattenPlan(),
        tmp_path / "kern_bindings.f90",
        specialization=Specialization({}, frozen),
        record_symbols_to=tmp_path / "record",
        record_only=True,
    ).read_text()
    assert "dace_spans_aliases" in sizing and "subroutine dace_spans_aliases" in sizing
    assert "if (dace_spans_overlap(" not in sizing  # no guard in a sizing run


def test_recorded_aliases_are_those_of_every_call(tmp_path):
    from dace_fortran.bindings.specialization import recorded_aliases

    record = tmp_path / "record"
    assert recorded_aliases(record) == []
    (tmp_path / "record.aliases").write_text("alias g_sf 0 t\nalias g_sf 1 x\n\nalias g_sf 0 t\n\n")
    assert recorded_aliases(record) == [("g_sf", 0, "t")]


def test_recorded_member_aliases_are_those_every_call_reported_alike(tmp_path):
    from dace_fortran.bindings.specialization import recorded_member_aliases
    from dace_fortran.bindings.storage_guard import STORAGE_GUARD_HELPERS

    assert "'same'" in STORAGE_GUARD_HELPERS  # the sizing binding writes them
    record = tmp_path / "record"
    assert recorded_member_aliases(record) == []
    (tmp_path / "record.aliases").write_text("same p_sf 0 c_sf 0\nalias g_sf 0 t\n\nsame p_sf 0 c_sf 0\n\n")
    assert recorded_member_aliases(record) == [("p_sf", 0, "c_sf", 0)]
    (tmp_path / "record.aliases").write_text("same p_sf 0 c_sf 0\n\n\n")  # the second call had none
    assert recorded_member_aliases(record) == []
