"""startup_constants: which module variables keep their value once a routine has first been called."""
from dace_fortran.substitute.outside_refs import parse_sources
from dace_fortran.substitute.startup_constants import StartupAnalysis

_PARAMS = """
module m_params
    implicit none
    integer :: n_cells, n_steps, counter, in_loop, from_mpi, set_once
    real(8) :: dt, gamma, shown
    real(8), target :: aliased
    real(8), pointer :: p => null()
    integer :: flags(3)
    namelist /inputs/ n_steps, gamma
end module m_params
"""

_RHS = """
module m_rhs
    use m_params
    implicit none
contains
    subroutine s_rhs(q)
        real(8), intent(inout) :: q(:)
        counter = counter + 1
        q = q * gamma + flags(1)
    end subroutine s_rhs
end module m_rhs
"""

_TIME = """
module m_time
    use m_params
    use m_rhs
    implicit none
contains
    subroutine s_step(q)
        real(8), intent(inout) :: q(:)
        call s_adapt(dt)
        call s_show(shown)
        call mpi_bcast(from_mpi, 1)
        call s_rhs(q)
    end subroutine s_step

    subroutine s_adapt(x)
        real(8), intent(inout) :: x
        x = x * 0.5d0
    end subroutine s_adapt

    subroutine s_show(x)
        real(8), intent(in) :: x
        print *, x
    end subroutine s_show

    subroutine s_setup()
        set_once = 3
        p => aliased
    end subroutine s_setup
end module m_time
"""

_MAIN = """
program p_main
    use m_params
    use m_time
    implicit none
    real(8) :: q(10)
    integer :: i
    read (*, nml=inputs)
    flags = 1
    call s_setup()
    do i = 1, n_steps
        in_loop = i
        call s_step(q)
        if (i == 5) n_cells = 2
    end do
end program p_main
"""


def _analysis(tmp_path, main=_MAIN) -> StartupAnalysis:
    paths = []
    for name, text in {"m_params.f90": _PARAMS, "m_rhs.f90": _RHS, "m_time.f90": _TIME, "p_main.f90": main}.items():
        path = tmp_path / name
        path.write_text(text)
        paths.append(path)
    return StartupAnalysis(parse_sources(paths), ("m_rhs", "s_rhs"))


def test_values_set_only_before_the_time_loop_keep_their_value(tmp_path):
    a = _analysis(tmp_path)
    assert a.main is not None and a.main.name == "p_main"
    for name in ("n_steps", "gamma", "flags", "set_once", "shown"):  # read, assigned or set up before the loop
        assert a.changes(("m_params", name)) == [], name


def test_values_code_may_change_after_the_first_call(tmp_path):
    a = _analysis(tmp_path)
    expected = {
        "n_cells": "assigned in p_main",  # in the loop, after the call
        "in_loop": "assigned in p_main",  # in the loop before the call: it runs again after the first call
        "counter": "assigned in m_rhs::s_rhs",  # the routine's own calls
        "dt": "passed to m_time::s_adapt",  # an intent(inout) dummy
        "from_mpi": "passed to mpi_bcast",  # no source: it may change it
        "aliased": "alias in m_time::s_setup",  # pointer-associated, even at start-up
    }
    for name, why in expected.items():
        changes = a.changes(("m_params", name))
        assert any(c.startswith(why) for c in changes), (name, changes)


def test_without_a_loop_start_up_ends_at_the_call(tmp_path):
    main = _MAIN.replace("    do i = 1, n_steps\n", "").replace("    end do\n", "").replace("in_loop = i", "in_loop = 1")
    a = _analysis(tmp_path, main)
    assert a.changes(("m_params", "in_loop")) == []  # before the one call
    assert a.changes(("m_params", "n_cells"))  # after it
