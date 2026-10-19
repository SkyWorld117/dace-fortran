"""``privatize_loop_local_arrays``: a work array several loop nests share without passing values through it gets one
copy per nest, so every nest becomes a map; an array that does carry a value from one loop to the next keeps its
single container."""

import dace
import numpy as np
import pytest
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap

from dace_fortran.loop_privatization import privatize_loop_local_arrays
from dace_fortran.pipelines import optimize
from tests._util import build_sdfg, f2py_compile

SHARED_SRC = """
subroutine nests(n, idir, a, b)
  implicit none
  integer, intent(in) :: n, idir
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  real(8) :: w(0:1)
  integer :: i
  if (idir == 1) then
    do i = 1, n
      w(0) = a(i)
      w(1) = 2.0d0*a(i)
      b(i) = w(0) + w(1)
    end do
  end if
  if (idir == 2) then
    do i = 1, n
      w(1) = a(i) + 1.0d0
      w(0) = a(i) - 1.0d0
      b(i) = w(0)*w(1)
    end do
  end if
end subroutine nests
"""

BRANCHED_SRC = """
subroutine branched(n, idir, a, b)
  implicit none
  integer, intent(in) :: n, idir
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  real(8) :: t, w(0:1)
  integer :: i
  if (idir == 1) then
    do i = 1, n
      if (a(i) > 0.0d0) then
        t = a(i)
        w(0) = 1.0d0
      else
        t = -a(i)
        w(0) = 2.0d0
      end if
      b(i) = t*w(0)
    end do
  end if
  if (idir == 2) then
    do i = 1, n
      if (a(i) > 1.0d0) then
        t = a(i) - 1.0d0
      else
        t = 1.0d0 - a(i)
      end if
      w(0) = 3.0d0
      b(i) = t + w(0)
    end do
  end if
end subroutine branched
"""

CARRIED_SRC = """
subroutine carried(n, a, b)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  real(8) :: w(0:1)
  integer :: i
  do i = 1, n
    w(0) = a(i)
    w(1) = a(i)*a(i)
  end do
  do i = 1, n
    b(i) = w(0) + w(1) + a(i)
  end do
end subroutine carried
"""


INDEXED_SRC = """
subroutine indexed(n, idir, perm, a, b)
  implicit none
  integer, intent(in) :: n, idir
  integer, intent(in) :: perm(3)
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  real(8) :: w(3)
  integer :: i
  if (idir == 1) then
    do i = 1, n
      w(1) = a(i)
      w(2) = 2.0d0*a(i)
      w(3) = 3.0d0*a(i)
      b(i) = w(perm(1)) - w(perm(2))
    end do
  end if
  if (idir == 2) then
    do i = 1, n
      w(3) = a(i) + 1.0d0
      w(2) = a(i) - 1.0d0
      w(1) = a(i)*a(i)
      b(i) = w(perm(3))*w(perm(1))
    end do
  end if
end subroutine indexed
"""

ZONES_SRC = """
subroutine zones(n, idir, a, b)
  implicit none
  integer, intent(in) :: n, idir
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  real(8) :: w(0:1)
  integer :: i, zone
  if (idir == 1) then
    do i = 1, n
      zone = 0
      if (a(i) > 0.0d0) zone = 1
      if (zone == 0) then
        w(0) = a(i)
        w(1) = 2.0d0*a(i)
      else if (zone == 1) then
        w(0) = -a(i)
        w(1) = a(i)
      end if
      b(i) = w(0) + w(1)
    end do
  else
    do i = 1, n
      zone = 0
      if (a(i) < 1.0d0) zone = 1
      if (zone == 0) then
        w(1) = a(i) + 1.0d0
        w(0) = a(i) - 1.0d0
      else if (zone == 1) then
        w(1) = 3.0d0
        w(0) = a(i)
      end if
      b(i) = w(0)*w(1)
    end do
  end if
end subroutine zones
"""

# The nests of ZONES_SRC under two IFs in sequence instead of one IF/ELSE: the conditions exclude each other.
SEQUENTIAL_IFS_SRC = ZONES_SRC.replace("zones(", "zones2(").replace("end subroutine zones", "end subroutine zones2").replace(
    "  else\n    do i = 1, n", "  end if\n  if (idir == 2) then\n    do i = 1, n")
assert SEQUENTIAL_IFS_SRC.count("if (idir == 2)") == 1


def _loops(sdfg) -> list[LoopRegion]:
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


def test_each_nest_gets_its_own_copy(tmp_path):
    sdfg = build_sdfg(SHARED_SRC, tmp_path / "sdfg", name="nests").build()
    renamed = privatize_loop_local_arrays(sdfg)
    assert list(renamed) == ["w"] and len(renamed["w"]) == 1
    sdfg.validate()


@pytest.mark.parametrize("direction", [1, 2])
def test_shared_work_array_no_longer_keeps_the_nests_sequential(tmp_path, direction):
    sdfg = build_sdfg(SHARED_SRC, tmp_path / "sdfg", name="nests").build()
    optimize(sdfg)
    assert _loops(sdfg) == [], "a nest stayed a loop"
    ref = f2py_compile(SHARED_SRC, tmp_path / "ref", f"nests_ref_{direction}")
    a = np.linspace(-2.0, 3.0, 9).copy()  # linspace returns a view, which DaCe does not take
    b, b_ref = np.zeros(9), np.zeros(9)
    sdfg(n=9, idir=direction, a=a, b=b)
    ref.nests(direction, a, b_ref)
    np.testing.assert_array_equal(b, b_ref)


def test_a_value_carried_between_loops_keeps_one_container(tmp_path):
    sdfg = build_sdfg(CARRIED_SRC, tmp_path / "sdfg", name="carried").build()
    assert privatize_loop_local_arrays(sdfg) == {}
    sdfg = build_sdfg(CARRIED_SRC, tmp_path / "sdfg2", name="carried").build()
    optimize(sdfg)
    ref = f2py_compile(CARRIED_SRC, tmp_path / "ref", "carried_ref")
    a = np.linspace(1.0, 2.0, 5).copy()
    b, b_ref = np.zeros(5), np.zeros(5)
    sdfg(n=5, a=a, b=b)
    ref.carried(a, b_ref)
    np.testing.assert_array_equal(b, b_ref)
    np.testing.assert_allclose(b, a[-1] + a[-1] ** 2 + a)


@pytest.mark.parametrize("direction", [1, 2])
def test_a_temporary_written_on_every_branch_is_local(tmp_path, direction):
    """``t`` and ``w(0)`` are assigned on both branches of an IF before being read, in each nest."""
    sdfg = build_sdfg(BRANCHED_SRC, tmp_path / "sdfg", name="branched").build()
    optimize(sdfg)
    assert _loops(sdfg) == [], "a nest stayed a loop"
    ref = f2py_compile(BRANCHED_SRC, tmp_path / "ref", f"branched_ref_{direction}")
    a = np.linspace(-2.0, 3.0, 9).copy()
    b, b_ref = np.zeros(9), np.zeros(9)
    sdfg(n=9, idir=direction, a=a, b=b)
    ref.branched(direction, a, b_ref)
    np.testing.assert_array_equal(b, b_ref)


def test_a_read_at_a_runtime_index_is_local_once_every_element_is_written(tmp_path):
    """``w(perm(k))`` names no constant element, but each nest has written all of ``w`` before it."""
    sdfg = build_sdfg(INDEXED_SRC, tmp_path / "sdfg", name="indexed").build()
    renamed = privatize_loop_local_arrays(sdfg)
    assert list(renamed) == ["w"] and len(renamed["w"]) == 1
    sdfg = build_sdfg(INDEXED_SRC, tmp_path / "sdfg2", name="indexed").build()
    optimize(sdfg)
    assert _loops(sdfg) == [], "a nest stayed a loop"
    ref = f2py_compile(INDEXED_SRC, tmp_path / "ref", "indexed_ref")
    a = np.linspace(-2.0, 3.0, 9).copy()
    perm = np.array([3, 1, 2], dtype=np.int32)
    for direction in (1, 2):
        b, b_ref = np.zeros(9), np.zeros(9)
        sdfg(n=9, idir=direction, perm=perm, a=a, b=b)
        ref.indexed(direction, perm, a, b_ref)
        np.testing.assert_array_equal(b, b_ref)


N = 7


def _prewritten_sdfg(read_after: bool) -> dace.SDFG:
    """``w(0) = 0`` before a loop that uses ``w`` as scratch, ``b(i) = w(0) + w(1)`` after ``w = (a(i), 2 a(i))``
    (the bridge's copy-in of a callee's ``intent(out)`` argument before the loop); ``read_after``: ``b(0) = w(0)``
    after the loop, which sees the last iteration's value."""
    sdfg = dace.SDFG("prewritten" + ("_read_after" if read_after else ""))
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_transient("w", [2], dace.float64)
    init = sdfg.add_state("init", is_start_block=True)
    zero = init.add_tasklet("zero", {}, {"o"}, "o = 0.0")
    init.add_edge(zero, "o", init.add_write("w"), None, dace.Memlet("w[0]"))
    loop = LoopRegion("l", f"i < {N}", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    body = loop.add_state("body", is_start_block=True)
    w = body.add_access("w")
    fill = body.add_tasklet("fill", {"x"}, {"p", "q"}, "p = x\nq = 2.0 * x")
    body.add_edge(body.add_read("a"), None, fill, "x", dace.Memlet("a[i]"))
    body.add_edge(fill, "p", w, None, dace.Memlet("w[0]"))
    body.add_edge(fill, "q", w, None, dace.Memlet("w[1]"))
    use = body.add_tasklet("use", {"u", "v"}, {"y"}, "y = u + v")
    body.add_edge(w, None, use, "u", dace.Memlet("w[0]"))
    body.add_edge(w, None, use, "v", dace.Memlet("w[1]"))
    body.add_edge(use, "y", body.add_write("b"), None, dace.Memlet("b[i]"))
    if read_after:
        after = sdfg.add_state_after(loop, "after")
        last = after.add_tasklet("last", {"u"}, {"y"}, "y = u")
        after.add_edge(after.add_read("w"), None, last, "u", dace.Memlet("w[0]"))
        after.add_edge(last, "y", after.add_write("b"), None, dace.Memlet("b[0]"))
    return sdfg


def test_an_array_only_written_outside_the_loop_is_local_to_it():
    sdfg = _prewritten_sdfg(read_after=False)
    loop = _loops(sdfg)[0]
    assert not LoopToMap.can_be_applied_to(sdfg, loop=loop)
    renamed = privatize_loop_local_arrays(sdfg)
    assert list(renamed) == ["w"] and len(renamed["w"]) == 1
    sdfg.validate()
    assert LoopToMap.can_be_applied_to(sdfg, loop=loop)
    LoopToMap.apply_to(sdfg, loop=loop)
    a = np.linspace(-1.0, 2.0, N).copy()  # linspace returns a view, which DaCe does not take
    b = np.zeros(N)
    sdfg(a=a, b=b)
    np.testing.assert_array_equal(b, 3.0 * a)


def test_an_array_read_after_the_loop_keeps_one_container():
    sdfg = _prewritten_sdfg(read_after=True)
    assert privatize_loop_local_arrays(sdfg) == {}



def test_nests_that_never_both_run_get_their_own_copies(tmp_path):
    """``w`` is assigned on the branches of an IF chain without an ELSE, so no iteration provably writes it before
    reading; the two nests sit in different branches of one conditional, so no value can pass between them."""
    sdfg = build_sdfg(ZONES_SRC, tmp_path / "sdfg", name="zones").build()
    renamed = privatize_loop_local_arrays(sdfg)
    assert list(renamed) == ["w"] and len(renamed["w"]) == 1
    sdfg.validate()


@pytest.mark.parametrize("direction", [1, 2])
def test_nests_that_never_both_run_are_both_parallel(tmp_path, direction):
    sdfg = build_sdfg(ZONES_SRC, tmp_path / "sdfg", name="zones").build()
    optimize(sdfg)
    assert _loops(sdfg) == [], "a nest stayed a loop"
    ref = f2py_compile(ZONES_SRC, tmp_path / "ref", f"zones_ref_{direction}")
    a = np.linspace(-2.0, 3.0, 9).copy()
    b, b_ref = np.zeros(9), np.zeros(9)
    sdfg(n=9, idir=direction, a=a, b=b)
    ref.zones(direction, a, b_ref)
    np.testing.assert_array_equal(b, b_ref)


@pytest.mark.parametrize("direction", [1, 2])
def test_nests_under_contradicting_conditions_are_both_parallel(tmp_path, direction):
    sdfg = build_sdfg(SEQUENTIAL_IFS_SRC, tmp_path / "sdfg", name="zones2").build()
    assert list(privatize_loop_local_arrays(sdfg)) == ["w"]
    sdfg = build_sdfg(SEQUENTIAL_IFS_SRC, tmp_path / "sdfg2", name="zones2").build()
    optimize(sdfg)
    assert _loops(sdfg) == [], "a nest stayed a loop"
    ref = f2py_compile(SEQUENTIAL_IFS_SRC, tmp_path / "ref", f"zones2_ref_{direction}")
    a = np.linspace(-2.0, 3.0, 9).copy()
    b, b_ref = np.zeros(9), np.zeros(9)
    sdfg(n=9, idir=direction, a=a, b=b)
    ref.zones2(direction, a, b_ref)
    np.testing.assert_array_equal(b, b_ref)
