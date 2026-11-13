"""FaCe-native test for the Fortran MERGE intrinsic.

merge(t, f, mask) returns t where mask is true, else f. Flang lowers it to
arith.select (scalars) or hlfir.elemental wrapping arith.select (arrays); the
bridge's generic arith.select fallback in buildExpr emits Python `(t if cond else f)`.
"""

from pathlib import Path

import numpy as np

from tests._util import build_sdfg


def test_merge_scalar(tmp_path: Path):
    """``merge`` on scalar real(8) inputs with a logical mask."""
    src = """
subroutine probe(a, b, mask, out)
  real(8), intent(in)   :: a, b
  logical, intent(in)   :: mask
  real(8), intent(out)  :: out
  out = merge(a, b, mask)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe").build()
    for mask_val in (True, False):
        out = np.zeros(1, dtype=np.float64)
        sdfg(a=1.5, b=2.5, mask=mask_val, out=out)
        expected = 1.5 if mask_val else 2.5
        assert float(out[0]) == expected, f"mask={mask_val}: got {out[0]}, want {expected}"


def test_merge_comparison_as_mask(tmp_path: Path):
    """``merge(a, b, a > b)``  --  an inline comparison as the mask."""
    src = """
subroutine probe(a, b, out)
  real(8), intent(in)  :: a, b
  real(8), intent(out) :: out
  out = merge(a, b, a > b)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="probe").build()
    for a_in, b_in, expected in [(3.0, 2.0, 3.0), (1.0, 5.0, 5.0), (4.0, 4.0, 4.0)]:
        out = np.zeros(1, dtype=np.float64)
        sdfg(a=a_in, b=b_in, out=out)
        assert float(out[0]) == expected


def test_merge_array_via_library_node(tmp_path: Path):
    """Array merge(t, f, mask) -- Flang lowers to hlfir.elemental{hlfir.designate;
    arith.select; yield_element}; bridge detects the simple shape (three loaded
    designates of declared arrays) and routes through MergeLibraryNode. Verified against np.where."""
    src = """
subroutine main(t, f, mask, out, n)
  integer, intent(in)  :: n
  real(8), intent(in)  :: t(n), f(n)
  logical, intent(in)  :: mask(n)
  real(8), intent(out) :: out(n)
  out = MERGE(t, f, mask)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    rng = np.random.default_rng(0)
    n = 12
    t = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    f = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    mask = np.ascontiguousarray(rng.random(n) > 0.5)
    out = np.zeros(n, dtype=np.float64)
    sdfg(t=t, f=f, mask=mask.astype(np.uint32), out=out, n=n)
    np.testing.assert_array_equal(out, np.where(mask, t, f))


def test_merge_array_verifies_libnode_present(tmp_path: Path):
    """Same as above, but also asserts a MergeLibraryNode is present in the
    built SDFG -- pins bridge->library-node routing so a silent fallback to per-element tasklets trips this test."""
    from dace.libraries.standard.nodes import MergeLibraryNode

    src = """
subroutine main(t, f, mask, out, n)
  integer, intent(in)  :: n
  real(8), intent(in)  :: t(n), f(n)
  logical, intent(in)  :: mask(n)
  real(8), intent(out) :: out(n)
  out = MERGE(t, f, mask)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    libnodes = [n for state in sdfg.all_states() for n in state.nodes() if isinstance(n, MergeLibraryNode)]
    assert len(libnodes) == 1, f"expected exactly one MergeLibraryNode, got {len(libnodes)}"


def test_merge_in_place_uses_the_previous_array_value(tmp_path: Path):
    """An operand may also be the destination; subsequent reads use its new value."""
    src = """
subroutine update(a, b, mask, out, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  real(8), intent(in) :: b(n)
  logical, intent(in) :: mask(n)
  real(8), intent(out) :: out(n)
  a = merge(b, a, mask)
  out = a
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="update").build()
    a = np.arange(8, dtype=np.float64)
    b = -a - 1
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint32)
    expected = np.where(mask != 0, b, a)
    out = np.zeros_like(a)
    sdfg(a=a, b=b, mask=mask, out=out, n=8)
    np.testing.assert_array_equal(a, expected)
    np.testing.assert_array_equal(out, expected)


def test_merge_sections_of_different_parent_ranks_preserve_indices(tmp_path: Path):
    """UW-style in-place vectors must retain the parents' scalar dimensions."""
    src = """
subroutine update(a, b, mask)
  real(8), intent(inout) :: a(-1:2,3,2)
  real(8), intent(in) :: b(0:3,3)
  logical, intent(in) :: mask(4,3)
  a(0:1,2,1) = merge(a(0:1,2,1), b(1:2,2), mask(2:3,2))
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="update").build()
    a = np.array(np.arange(24).reshape(4, 3, 2, order="F"), dtype=np.float64, order="F")
    b = np.array(-np.arange(12).reshape(4, 3, order="F") - 1, dtype=np.float64, order="F")
    mask = np.ones((4, 3), dtype=np.uint32, order="F")
    mask[2, 1] = 0
    expected = a.copy(order="F")
    expected[1:3, 1, 0] = np.where(mask[1:3, 1], a[1:3, 1, 0], b[1:3, 1])
    sdfg(a=a, b=b, mask=mask)
    np.testing.assert_array_equal(a, expected)


def test_merge_input_sections_do_not_read_the_entire_parent(tmp_path: Path):
    src = """
subroutine update(a, b, mask, out)
  real(8), intent(in) :: a(4,3), b(4,3)
  logical, intent(in) :: mask(4,3)
  real(8), intent(out) :: out(2)
  out = merge(a(1:3:2,2), b(1:3:2,2), mask(1:3:2,2))
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="update").build()
    a = np.array(np.arange(12).reshape(4, 3, order="F"), dtype=np.float64, order="F")
    b = -a.copy(order="F") - 1
    mask = np.ones((4, 3), dtype=np.uint32, order="F")
    mask[2, 1] = 0
    out = np.zeros(2)
    sdfg(a=a, b=b, mask=mask, out=out)
    np.testing.assert_array_equal(out, [a[0, 1], b[2, 1]])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
