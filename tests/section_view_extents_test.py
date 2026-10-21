"""A section view's extent the bridge cannot trace is defined inside the SDFG, not left as an input.

An inlined callee called on ``q(:, iv%beg:iv%end)``, with the record ``iv`` set by the routine just before each call,
views ``q`` through a view whose second extent the bridge leaves as the placeholder ``<view>_d1``.  It was a free
symbol of the SDFG, so the binding declared it, had nothing to assign it from (``! TODO: no plan entry gives size``)
and passed an uninitialised value.  It is now assigned the element count of the section, right before the code
of the call that uses it.
"""

import subprocess

import dace
import numpy as np
import pytest
from dace import data, symbolic
from dace.sdfg.state import LoopRegion

from dace_fortran.builder.view_extents import define_section_view_extents
from tests._util import build_sdfg

_MODULE = """
module grid_views
  implicit none
  type bounds_info
    integer :: beg, end
  end type bounds_info
  type(bounds_info) :: iv
  integer :: lb = 0
  real(8), allocatable :: q(:, :)
contains
  subroutine accumulate(v)
    real(8), dimension(lb:, 1:), intent(inout) :: v
    integer :: i, j
    do j = 1, size(v, 2)
      do i = lb, lb + size(v, 1) - 1
        v(i, j) = v(i, j) + j
      end do
    end do
  end subroutine accumulate
  subroutine sweep(w)
    real(8), dimension(lb:, 1:), intent(inout) :: w
    call accumulate(w(:, iv%beg:iv%end))
  end subroutine sweep
end module grid_views
"""

_KERNEL = """
subroutine two_sweeps(nv)
  use grid_views
  integer, intent(in) :: nv
  iv%beg = 2; iv%end = nv
  call sweep(q)
  iv%beg = 1; iv%end = nv - 1
  call sweep(q)
end subroutine two_sweeps
"""


def _hand_built() -> dace.SDFG:
    """``v`` views ``q(:, iv_beg:iv_end)``; a loop bound reads its extent ``v_d1`` before the state linking it."""
    sdfg = dace.SDFG("section_view")
    sdfg.add_symbol("n0", dace.int64)
    sdfg.add_symbol("n1", dace.int64)
    sdfg.add_symbol("v_d1", dace.int64)
    sdfg.add_array(
        "q", [symbolic.symbol("n0"), symbolic.symbol("n1")], dace.float64, strides=[1, symbolic.symbol("n0")]
    )
    sdfg.add_scalar("iv_beg", dace.int32, transient=True)
    sdfg.add_scalar("iv_end", dace.int32, transient=True)
    sdfg.add_view(
        "v", [symbolic.symbol("n0"), symbolic.symbol("v_d1")], dace.float64, strides=[1, symbolic.symbol("n0")]
    )

    init = sdfg.add_state("set_bounds", is_start_block=True)
    for name, value in (("iv_beg", "2"), ("iv_end", "n1")):
        t = init.add_tasklet(f"set_{name}", {}, {"out"}, f"out = {value}")
        init.add_edge(t, "out", init.add_write(name), None, dace.Memlet(f"{name}[0]"))

    loop = LoopRegion("cols", "j < v_d1 + 1", "j", "j = 1", "j = j + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    body = loop.add_state("body", is_start_block=True)
    src = body.add_read("q")
    view = body.add_access("v")
    body.add_edge(
        src, None, view, "views", dace.Memlet(data="q", subset="0:n0, iv_beg - 1:iv_end", other_subset="0:n0, 0:v_d1")
    )
    t = body.add_tasklet("bump", {"a"}, {"b"}, "b = a + j")
    body.add_edge(view, None, t, "a", dace.Memlet("v[0, j - 1]"))
    body.add_edge(t, "b", body.add_write("q"), None, dace.Memlet("q[0, iv_beg + j - 2]"))
    return sdfg


def test_the_placeholder_is_assigned_before_its_first_use():
    sdfg = _hand_built()
    assert "v_d1" in {str(s) for s in sdfg.free_symbols}
    defined = define_section_view_extents(sdfg, {"v": ("q", [":", "iv_beg - 1:iv_end"])})
    assert list(defined) == ["v_d1"]
    assert symbolic.pystr_to_symbolic(defined["v_d1"]) == symbolic.pystr_to_symbolic("iv_end - iv_beg + 1")
    assert "v_d1" not in {str(s) for s in sdfg.free_symbols}
    # Before the loop whose bound reads it, after the state that sets the bounds.
    loop = next(n for n in sdfg.nodes() if isinstance(n, LoopRegion))
    (edge,) = sdfg.in_edges(loop)
    assert set(edge.data.assignments) == {"v_d1"}
    assert sdfg.in_edges(edge.src)[0].src.label == "set_bounds"


def test_a_resolved_extent_is_left_alone():
    sdfg = _hand_built()
    sdfg.arrays["v"].set_shape(
        [symbolic.symbol("n0"), symbolic.pystr_to_symbolic("n1 - 1")], strides=[1, symbolic.symbol("n0")]
    )
    assert define_section_view_extents(sdfg, {"v": ("q", [":", "iv_beg - 1:iv_end"])}) == {}


def test_inlined_calls_on_record_bounded_sections(tmp_path):
    """Two inlined calls of a callee on ``q(:, iv%beg:iv%end)``: no extent is left to the caller, and the views
    cover the right columns."""
    sdfg = build_sdfg(_MODULE + _KERNEL, tmp_path, name="two_sweeps", entry="two_sweeps").build()
    views = [name for name, desc in sdfg.arrays.items() if isinstance(desc, data.View)]
    assert views, "the section is no longer a view: the test does not exercise the placeholder extent"
    free = {str(s) for s in sdfg.free_symbols}
    leaked = sorted(s for s in free for v in views if s.startswith(f"{v}_d"))
    assert not leaked, f"view extents left to the caller: {leaked}"

    n0, nv = 4, 6
    q = np.arange(n0 * nv, dtype=np.float64).reshape(n0, nv, order="F").copy(order="F")
    expected = q.copy()
    for col in range(1, nv):
        expected[:, col] += col  # q(:, 2:nv), column col + 1 of q is column col of the section
    for col in range(1, nv):
        expected[:, col - 1] += col  # q(:, 1:nv-1)
    args = {"q": q, "nv": nv, "lb": 0, "q_d0": n0, "q_d1": nv, "offset_q_d0": 1, "offset_q_d1": 1}
    sdfg(**{k: v for k, v in args.items() if k in sdfg.arglist()})
    np.testing.assert_array_equal(q, expected)


def test_the_binding_assigns_every_symbol_it_passes(tmp_path):
    from dace_fortran.bindings import FlattenPlan, emit_bindings
    from dace_fortran.bindings.build_fortran_library import compiled_init_symbols
    from dace_fortran.bindings.fortran_interface import build_auto_interface
    from dace_fortran.bindings.frozen_signature import get_frozen_signature

    sdfg = build_sdfg(_MODULE + _KERNEL, tmp_path, name="two_sweeps", entry="two_sweeps").build()
    compiled = sdfg.compile()
    so = compiled._lib._library_filename
    from pathlib import Path

    binding = emit_bindings(
        get_frozen_signature(sdfg),
        build_auto_interface(sdfg._fortran_interface_raw, sdfg.name),
        FlattenPlan.from_dict(sdfg._flatten_plan_raw),
        str(tmp_path / "two_sweeps_bindings.f90"),
        tuple(compiled._sig),
        init_symbols=compiled_init_symbols(Path(so), sdfg.name),
        module_allocatables_are_targets=False,
    )
    assert "TODO" not in binding.read_text()
    module = tmp_path / "grid_views.f90"
    module.write_text(_MODULE)
    for src in (module, binding):
        subprocess.check_call(
            ["gfortran", "-c", "-ffree-line-length-none", f"-J{tmp_path}", str(src), "-o", str(src.with_suffix(".o"))],
            cwd=tmp_path,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
