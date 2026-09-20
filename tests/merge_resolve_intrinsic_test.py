# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""`merge_used_modules(resolve_intrinsic=...)`: a VENDORED module whose name is also an intrinsic.

`_INTRINSIC_MODULES` is a heuristic -- "let the compiler supply this" -- and it names `mpi`, because
a real MPI install provides `mpi.mod`.  A project can legitimately vendor its own module under that
name instead: MFC ships a hand-written `mpi` stub carrying the MPI constants its sources use, and the
compiler has no `mpi.mod` to fall back on, so the ``USE`` is left dangling and flang fails with

    Cannot parse module file for module 'mpi': Source file 'mpi.mod' was not found

The default behaviour is unchanged -- without ``resolve_intrinsic`` nothing is inlined -- because
flang really does supply the genuine intrinsics and inlining a project's stub over them would be
wrong for everyone else.
"""
from pathlib import Path

from dace_fortran.preprocess import merge_used_modules

# A root that `USE`s a vendored `mpi`, and the vendored module itself.
ROOT = """\
program p
  use mpi
  implicit none
  integer :: r
  call MPI_COMM_RANK(MPI_COMM_WORLD, r)
end program p
"""

VENDORED = """\
module mpi
  implicit none
  integer, parameter :: MPI_COMM_WORLD = 0, MPI_SUCCESS = 0
  external :: MPI_COMM_RANK
end module mpi
"""


def _stage(tmp_path: Path) -> Path:
    d = tmp_path / "src"
    d.mkdir()
    (d / "mpi.f90").write_text(VENDORED)
    return d


def test_default_leaves_an_intrinsic_named_use_to_the_compiler(tmp_path):
    """Unchanged behaviour: without the opt-in, `use mpi` is not resolved from the search dir."""
    d = _stage(tmp_path)
    out = merge_used_modules(ROOT, search_dirs=[d])
    assert "end module mpi" not in out, (
        "the vendored `mpi` was inlined without being asked for -- the intrinsic set must stay in "
        "charge by default, because flang supplies the real one")
    assert "use mpi" in out


def test_resolve_intrinsic_inlines_the_vendored_module(tmp_path):
    """The opt-in resolves exactly the named module, and it lands before the root that uses it."""
    d = _stage(tmp_path)
    out = merge_used_modules(ROOT, search_dirs=[d], resolve_intrinsic=("mpi",))
    assert "end module mpi" in out, f"the vendored `mpi` was not inlined:\n{out}"
    assert out.index("module mpi") < out.index("program p"), (
        "the inlined module must precede its user -- flang resolves `use` against what it has seen")


def test_resolve_intrinsic_is_name_scoped(tmp_path):
    """Naming one module must not open the door for the rest of the intrinsic set."""
    d = _stage(tmp_path)
    (d / "iso_c_binding.f90").write_text("module iso_c_binding\n  implicit none\nend module "
                                         "iso_c_binding\n")
    root = ROOT.replace("use mpi", "use mpi\n  use iso_c_binding")
    out = merge_used_modules(root, search_dirs=[d], resolve_intrinsic=("mpi",))
    assert "end module mpi" in out
    assert "end module iso_c_binding" not in out, (
        "resolving `mpi` also inlined `iso_c_binding`, which was not named -- the parameter is a "
        "per-name override, not a switch")
