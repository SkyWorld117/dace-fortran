"""Regression tests for the source-scanner heuristics in ``dace_fortran.vacuity``.

The named-``END`` bug: ``_PROC_END`` used to require the keyword *and* name to be absent, so a
source that writes ``END SUBROUTINE foo`` (the WRF/WPS slice style) never closed a nesting level
and :func:`defined_procedures` returned only the outermost procedure name -- making the vacuity
guard falsely report every real callee as an unresolved external.  These tests pin the fix.
"""

from dace_fortran.vacuity import defined_procedures, source_call_sites


def test_defined_procedures_named_end():
    """``END SUBROUTINE name`` / ``END FUNCTION name`` close their level (the GF regression)."""
    src = """
subroutine outer(a, n)
  integer :: n
  real(8) :: a(n)
  call inner(a, n)
  call leaf(a, n)
end subroutine outer

subroutine inner(a, n)
  real(8) :: a(n)
  call leaf(a, n)
end subroutine inner

real(8) function leaf(x, n)
  real(8) :: x(n)
  leaf = x(1)
end function leaf
"""
    assert defined_procedures(src) == {"outer", "inner", "leaf"}


def test_defined_procedures_bare_and_mixed_end():
    """Bare ``END``, keyword-only END and named END all close exactly one level."""
    src = """
subroutine a()
  call b()
end

subroutine b()
  call c()
end subroutine

subroutine c()
end subroutine c
"""
    assert defined_procedures(src) == {"a", "b", "c"}


def test_proc_end_does_not_match_construct_ends():
    """``END DO`` / ``END IF`` / ``END WHERE`` must not be mistaken for a program-unit END."""
    src = """
subroutine a(n, m)
  integer :: n, m, i, j
  do i = 1, n
    if (i > m) then
      exit
    end if
  end do
  where (m > 0)
    j = m
  end where
end subroutine a
"""
    assert defined_procedures(src) == {"a"}


def test_block_data_and_module_end_forms():
    """``END BLOCK DATA name`` / ``END MODULE name`` are recognised program-unit ENDs."""
    src = """
module m
  integer :: k
end module m

block data bd
  common /c/ k
end block data bd

subroutine s()
end subroutine s
"""
    assert defined_procedures(src) == {"s"}


def test_vacuity_named_end_callees_accounted():
    """A named-END callee must satisfy the guard's leg-B accounting, not be flagged unresolved."""
    src = """
subroutine prod(a, n)
  real(8) :: a(n)
  call cup_kbcon(a, n)
end subroutine prod

subroutine cup_kbcon(a, n)
  real(8) :: a(n)
  a(1) = a(1) + 1.0d0
end subroutine cup_kbcon
"""
    defined = defined_procedures(src)
    calls = source_call_sites(src)
    assert calls == {"cup_kbcon"}
    assert calls <= defined  # no unaccounted callee -> leg B passes
