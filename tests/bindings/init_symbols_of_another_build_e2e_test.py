"""E2e companion to ``init_symbols_of_another_build_test.py``: two specialisations of one routine, built as SDFGs of
the same name, the second calling an inlined callee once more on a section whose bound is one more symbol.  When the
two compile at the same time into one DaCe build folder, the header a binding reads its ``__dace_init`` parameters
from can be the other build's.  Here each build gets its own folder and the race's effect is reproduced by copying
the other build's header over its own: the emitter must refuse it in both directions -- an extra symbol (the binding
would not compile) and a missing one (it would bind every later argument in the wrong position) -- and accept each
build's own header.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from dace_fortran.bindings import FlattenPlan, SignatureDriftError, emit_bindings
from dace_fortran.bindings.build_fortran_library import compiled_init_symbols
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.frozen_signature import get_frozen_signature
from dace_fortran.build import make_builder

_HELPER = """
module helpers
contains
  subroutine scale(v)
    real(8), intent(inout) :: v(:, :)
    integer :: i, j
    do j = 1, size(v, 2)
      do i = 1, size(v, 1)
        v(i, j) = 2.0d0 * v(i, j)
      end do
    end do
  end subroutine scale
end module helpers
"""

_VARIANTS = {
    "one_call": """
subroutine kernel(n, k, a, b)
  use helpers
  integer, intent(in) :: n, k
  real(8), intent(inout) :: a(n, n), b(n, n)
  call scale(a(:, 1:k))
end subroutine kernel
""",
    "two_calls": """
subroutine kernel(n, k, m, a, b)
  use helpers
  integer, intent(in) :: n, k, m
  real(8), intent(inout) :: a(n, n), b(n, n)
  call scale(a(:, 1:k))
  call scale(b(:, 1:m))
end subroutine kernel
""",
}


class _Build:
    """One specialisation compiled into its own build folder."""

    def __init__(self, tmp_path: Path, tag: str):
        out = tmp_path / tag
        self.sdfg = make_builder(_HELPER + _VARIANTS[tag], entry="kernel", name="kernel", out_dir=str(out)).build()
        self.sdfg.build_folder = str(out / "dacecache" / "kernel")
        self.frozen = get_frozen_signature(self.sdfg)
        self.plan = FlattenPlan.from_dict(self.sdfg._flatten_plan_raw)
        self.iface = build_auto_interface(self.sdfg._fortran_interface_raw, "kernel")
        compiled = self.sdfg.compile()
        self.sig = tuple(compiled._sig)
        self.so = Path(compiled._lib._library_filename)
        self.header = self.so.parent.parent / "include" / "kernel.h"
        self.out = out

    def emit(self) -> Path:
        """The binding against the header now in this build's folder."""
        return emit_bindings(
            self.frozen,
            self.iface,
            self.plan,
            str(self.out / "kernel_bindings.f90"),
            self.sig,
            init_symbols=compiled_init_symbols(self.so, "kernel"),
        )


@pytest.fixture(scope="module")
def builds(tmp_path_factory) -> dict[str, _Build]:
    tmp_path = tmp_path_factory.mktemp("init_symbols")
    return {tag: _Build(tmp_path, tag) for tag in _VARIANTS}


def test_each_build_accepts_its_own_header_and_its_binding_compiles(builds):
    assert compiled_init_symbols(builds["one_call"].so, "kernel") == ("k", "n")
    assert compiled_init_symbols(builds["two_calls"].so, "kernel") == ("k", "m", "n")
    for build in builds.values():
        binding = build.emit()
        subprocess.check_call(
            ["gfortran", "-c", "-ffree-line-length-none", f"-J{build.out}", str(binding), "-o", str(build.out / "b.o")],
            cwd=build.out,
        )


def test_a_header_of_the_build_with_the_extra_call_is_refused(builds, tmp_path):
    one = builds["one_call"]
    own = tmp_path / "own.h"
    shutil.copy(one.header, own)
    try:
        shutil.copy(builds["two_calls"].header, one.header)
        with pytest.raises(SignatureDriftError, match=r"takes \['m'\], which are not symbols of this SDFG"):
            one.emit()
    finally:
        shutil.copy(own, one.header)


def test_a_header_of_the_build_without_the_extra_call_is_refused(builds, tmp_path):
    two = builds["two_calls"]
    own = tmp_path / "own.h"
    shutil.copy(two.header, own)
    try:
        shutil.copy(builds["one_call"].header, two.header)
        with pytest.raises(SignatureDriftError, match=r"lacks \['m'\], which __program takes"):
            two.emit()
    finally:
        shutil.copy(own, two.header)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
