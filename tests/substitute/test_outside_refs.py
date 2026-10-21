"""Tests of the outside-reference scan of a program's sources.  Run: python -m pytest test_outside_refs.py"""
from pathlib import Path

from dace_fortran.substitute.outside_refs import close_alias_groups, exclusive_procedures, outside_readers, parse_sources, source_files

_FIELDS = """
module m_fields
    implicit none
    type vec
        real, pointer :: sf(:) => null()
    end type vec
    real, allocatable, dimension(:) :: work, kept, written_outside
    type(vec) :: pa, pb, pc
    real :: dummy_name(4)
contains
    subroutine s_init()
        allocate (work(10), kept(10), written_outside(10))
        allocate (pa%sf(10))
        pb%sf => pa%sf
        if (allocated(work)) written_outside = 0.0
        print *, 'work is', size(work), lbound(kept, 1)
    end subroutine s_init

    subroutine s_rhs(q)
        real, intent(inout) :: q(10)
        work = q
        call s_helper(q)
        call s_shared(q)
        kept = work
        q = q + kept
        pc%sf => pc%sf
    end subroutine s_rhs

    subroutine s_helper(q)
        real, intent(inout) :: q(10)
        q = q + work
    end subroutine s_helper

    subroutine s_shared(q)
        real, intent(inout) :: q(10)
        q = q + written_outside
    end subroutine s_shared

    subroutine s_shadow(dummy_name)
        real, intent(in) :: dummy_name(4)
        print *, dummy_name
    end subroutine s_shadow
end module m_fields
"""

_STEPPER = """
module m_stepper
    use m_fields, only: s_rhs, s_shared, renamed => kept
    implicit none
contains
    subroutine s_step(q)
        real, intent(inout) :: q(10)
        call s_rhs(q)
        call s_shared(q)
        print *, renamed(1)
    end subroutine s_step
end module m_stepper
"""

_MAIN = """
program p_main
    use m_fields, only: s_init
    use m_stepper, only: s_step
    implicit none
    real :: q(10)
    call s_init()
    call s_step(q)
end program p_main
"""


def _program(tmp_path: Path, *texts: str):
    for i, text in enumerate(texts):
        (tmp_path / f"f{i}.fpp.f90").write_text(text)
    return parse_sources(source_files([tmp_path]))


def _verdicts(prog, keys, outside=()):
    v = outside_readers(prog, ("m_fields", "s_rhs"), [("m_fields", k) for k in keys], outside)
    close_alias_groups(v)
    return {k[1]: x for k, x in v.items()}


def test_exclusive_call_tree(tmp_path: Path):
    prog = _program(tmp_path, _FIELDS, _STEPPER, _MAIN)
    rhs = next(p for p in prog.procedures if p.name == "s_rhs")
    inside = {p.name for p in prog.procedures if id(p) in exclusive_procedures(prog, rhs)}
    # s_shared is called from the time stepper too: it runs outside the routine's calls as well.
    assert inside == {"s_rhs", "s_helper"}


def test_scratch_array_with_only_allocation_and_inquiries_outside(tmp_path: Path):
    v = _verdicts(_program(tmp_path, _FIELDS, _STEPPER, _MAIN), ["work"])
    assert v["work"].scratch, v["work"].reasons


def test_read_through_a_renamed_use_outside(tmp_path: Path):
    v = _verdicts(_program(tmp_path, _FIELDS, _STEPPER, _MAIN), ["kept"])
    assert not v["kept"].scratch
    assert "s_step" in v["kept"].reasons[0]


def test_read_in_a_procedure_the_outside_calls_too(tmp_path: Path):
    """s_shared reads written_outside and the stepper calls it outside s_rhs; s_init only writes it."""
    v = _verdicts(_program(tmp_path, _FIELDS, _STEPPER, _MAIN), ["written_outside"])
    assert not v["written_outside"].scratch
    assert all("s_shared" in r for r in v["written_outside"].reasons)


def test_writes_outside_are_recorded_apart_from_reads(tmp_path: Path):
    """s_init assigns written_outside (harmless for scratch, not for state kept between calls); work is only
    allocated and inquired about outside; s_rhs's own writes do not count."""
    v = _verdicts(_program(tmp_path, _FIELDS, _STEPPER, _MAIN), ["written_outside", "work", "kept"])
    assert [w.split(" ")[0] for w in v["written_outside"].writes] == ["m_fields::s_init"]
    assert v["work"].writes == [] and v["kept"].writes == []


def test_a_dummy_of_the_same_name_shadows(tmp_path: Path):
    v = _verdicts(_program(tmp_path, _FIELDS, _STEPPER, _MAIN), ["dummy_name"])
    assert v["dummy_name"].scratch, v["dummy_name"].reasons


def test_pointer_association_aliases_and_groups(tmp_path: Path):
    """pb%sf => pa%sf (outside): the two are scratch only together."""
    (tmp_path / "a").mkdir()
    v = _verdicts(_program(tmp_path / "a", _FIELDS, _STEPPER, _MAIN), ["pa", "pb"])
    assert v["pa"].scratch and v["pb"].scratch
    assert v["pa"].aliases == {("m_fields", "pb")}
    # A procedure of the main program reads pa: neither is scratch.
    reader = _MAIN.replace("call s_step(q)", "call s_step(q)\n    call s_peek()").replace(
        "end program p_main", "contains\n    subroutine s_peek()\n        use m_fields, only: pa\n"
        "        print *, pa%sf(1)\n    end subroutine s_peek\nend program p_main")
    (tmp_path / "b").mkdir()
    v = _verdicts(_program(tmp_path / "b", _FIELDS, _STEPPER, reader), ["pa", "pb"])
    assert not v["pa"].scratch and not v["pb"].scratch
    assert "s_peek" in v["pa"].reasons[0]
    assert "not scratch" in v["pb"].reasons[0]


def test_alias_with_a_variable_no_candidate_rejects(tmp_path: Path):
    v = _verdicts(_program(tmp_path, _FIELDS, _STEPPER, _MAIN), ["pb"])
    assert not v["pb"].scratch  # pa is no candidate here: what pb points to is unknown to the analysis


def test_dropped_callees_count_outside(tmp_path: Path):
    """s_helper reads work; when its calls are dropped (a stub) it no longer runs inside s_rhs."""
    v = _verdicts(_program(tmp_path, _FIELDS, _STEPPER, _MAIN), ["work"], outside=["s_helper"])
    assert not v["work"].scratch
    assert "s_helper" in v["work"].reasons[0]


def test_generic_interfaces_and_address_taken(tmp_path: Path):
    text = """
module m_gen
    implicit none
    real, allocatable :: buf(:)
    interface s_any
        module procedure s_one
    end interface s_any
contains
    subroutine s_rhs()
        call s_any()
    end subroutine s_rhs
    subroutine s_one()
        buf = 1.0
        print *, buf(1)
    end subroutine s_one
    subroutine s_out()
        call s_register(s_one)
    end subroutine s_out
    subroutine s_register(f)
        external f
    end subroutine s_register
end module m_gen
"""
    prog = _program(tmp_path, text)
    rhs = next(p for p in prog.procedures if p.name == "s_rhs")
    inside = {p.name for p in prog.procedures if id(p) in exclusive_procedures(prog, rhs)}
    # s_one is reached through the generic, and handed to s_register: it may run anywhere.
    assert inside == {"s_rhs"}
    v = outside_readers(prog, ("m_gen", "s_rhs"), [("m_gen", "buf")])
    assert not v[("m_gen", "buf")].scratch


def test_block_and_conditional_declarations_shadow_nothing(tmp_path: Path):
    text = """
module m_blk
    implicit none
    real, allocatable :: buf(:), other(:)
contains
    subroutine s_rhs()
        buf = 1.0
    end subroutine s_rhs
    subroutine s_out()
        block
            real :: buf(3)
            buf = 0.0
        end block
        print *, buf(1)
#ifdef SOMETHING
        real :: other(3)
#endif
        print *, other(1)
        select type (x => buf)
        type is (real)
            print *, 1
        end select
    end subroutine s_out
end module m_blk
"""
    prog = _program(tmp_path, text)
    v = outside_readers(prog, ("m_blk", "s_rhs"), [("m_blk", "buf"), ("m_blk", "other")])
    assert not v[("m_blk", "buf")].scratch and not v[("m_blk", "other")].scratch


def test_passing_to_a_procedure_that_keeps_a_pointer(tmp_path: Path):
    text = """
module m_esc
    implicit none
    real, allocatable, target :: buf(:)
    real, pointer :: saved(:) => null()
contains
    subroutine s_rhs()
        buf = 1.0
        call s_keep(buf)
    end subroutine s_rhs
    subroutine s_keep(x)
        real, target, intent(in) :: x(:)
        call s_store(x)
    end subroutine s_keep
    subroutine s_store(y)
        real, target, intent(in) :: y(:)
        saved => y
    end subroutine s_store
end module m_esc
"""
    prog = _program(tmp_path, text)
    v = outside_readers(prog, ("m_esc", "s_rhs"), [("m_esc", "buf")])
    assert not v[("m_esc", "buf")].scratch
    assert "s_keep" in v[("m_esc", "buf")].reasons[0]
