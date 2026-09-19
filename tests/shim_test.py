# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Normalising a Fortran shim's kernel launches."""
from dace_fortran import shim

CALL = "dace_order"
USE = "  use m_dace_order, only: dace_order"


def route(text):
    return shim.route_launches(text, call=CALL, use_line=USE)


def test_a_launch_with_no_ordering_gets_one():
    """The failure this exists for: a DaCe kernel launches on a stream DaCe made NON-blocking, so it
    is unordered against the host runtime's stream in both directions -- a read-after-write hazard
    that looks correct and mostly passes."""
    src = ("module m\ncontains\n  subroutine s(state_x)\n"
           "    call foo_run(state_x, a, b)\n"
           "  end subroutine\nend module\n")
    out, rw = route(src)
    assert any("inserted" in r for r in rw)
    assert "call dace_order(state_x, 'foo')" in out


def test_a_host_blocking_sync_is_replaced_not_duplicated():
    src = ("module m\ncontains\n  subroutine s(state_x)\n"
           "    call foo_run(state_x, a)\n"
           "    ierr = cudaDeviceSynchronize_()\n"
           "    if (ierr /= 0) then\n      print *, 'x'\n      error stop 1\n    end if\n"
           "  end subroutine\nend module\n")
    out, _ = route(src)
    assert "cudaDeviceSynchronize_()" not in out
    assert out.count("call dace_order(") == 1


def test_the_one_line_sync_spelling_is_found():
    """`ierr = cudaDeviceSynchronize_(); call chk(ierr, '...')` is the same barrier on one line, and
    missing a spelling does not fail loudly -- it silently leaves that launch unordered."""
    src = ("module m\ncontains\n  subroutine s(state_x)\n"
           "    call foo_run(state_x, a)\n"
           "    ierr = cudaDeviceSynchronize_(); call chk(ierr, 'sync')\n"
           "  end subroutine\nend module\n")
    out, _ = route(src)
    assert "call chk" not in out
    assert out.count("call dace_order(") == 1


def test_an_interior_sync_is_left_alone():
    """A sync that is neither after a launch nor at an entry tail sits BETWEEN staging steps -- a
    device->host copy after a kernel -- where it may be load bearing for the copy."""
    src = ("module m\ncontains\n  subroutine s(state_x)\n"
           "    ierr = cudaDeviceSynchronize_(); call chk(ierr, 'sync pack')\n"
           "    call foo_run(state_x, a)\n"
           "    ierr = cudaDeviceSynchronize_()\n"
           "    if (ierr /= 0) then\n      print *, 'x'\n    end if\n"
           "    call cudaMemcpy_(dst, src, n, k)\n"
           "  end subroutine\nend module\n")
    out, _ = route(src)
    assert "call chk(ierr, 'sync pack')" in out          # kept
    assert "call dace_order(" in out                      # the launch is routed


def test_an_array_valued_state_keeps_its_subscript():
    """`state_sweeps(3)` is one library per direction.  Dropping the subscript is a rank mismatch the
    compiler catches -- but the CORRECTION matters too: a transformer that merely skipped an
    already-routed call would leave the wrong expression in place forever."""
    src = ("module m\ncontains\n  subroutine s\n"
           "    call sweeps_run_x(state_sweeps(1), a)\n"
           "    call dace_order(state_sweeps, 'sweeps_x')\n"
           "  end subroutine\nend module\n")
    out, rw = route(src)
    assert "call dace_order(state_sweeps(1), 'sweeps_x')" in out
    assert any("corrected" in r for r in rw)


def test_the_infix_run_name_yields_the_right_tag():
    """`_run` may be infixed (`sweeps_run_x`) or suffixed (`rk_run`); stripping it from the END
    leaves the former unchanged."""
    assert shim.tag_of("sweeps_run_x") == "sweeps_x"
    assert shim.tag_of("rk_run") == "rk"
    assert shim.tag_of("mfc_dace_weno_y_run") == "mfc_dace_weno_y"


def test_the_import_goes_in_the_module_header_not_after_a_subroutine_use():
    """A big shim has `use` statements inside its contained subroutines.  An import inserted after
    one of those is at the wrong scope: the routine is not visible at module level and the link
    fails with `undefined reference` -- the compiler says nothing."""
    src = ("module m\n  use iso_c_binding\ncontains\n  subroutine s(state_x)\n"
           "    use something_else\n"
           "    call foo_run(state_x, a)\n"
           "  end subroutine\nend module\n")
    out, _ = route(src)
    lines = out.split("\n")
    assert USE in lines
    assert lines.index(USE) < lines.index("contains")


def test_routing_twice_changes_nothing():
    """Idempotence is not a nicety here: the generators run this on every write, and a pass that
    re-emitted would corrupt its own output."""
    src = ("module m\ncontains\n  subroutine s(state_x)\n    call foo_run(state_x, a)\n"
           "  end subroutine\nend module\n")
    once, _ = route(src)
    twice, rw = route(once)
    assert twice == once
    assert rw == []
