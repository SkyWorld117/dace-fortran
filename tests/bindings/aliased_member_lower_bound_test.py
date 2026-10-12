"""The lower bound of a record member the plan aliases is the member's, not the view's.

An aliasable recipe binds the SDFG's flat companion to the member with ``c_f_pointer``, a view that starts at the
member's first element: its own ``lbound`` is 1 whatever bounds the member was allocated with.  The SDFG subscripts
in the member's frame (``arr[idx - offset]``), so ``offset_<flat>_d<i>`` must read ``lbound`` of the member.  The
emitter did so only when the interface carried the record's member layout; a hand-authored interface without one
fell through to the intrinsic fallback, which reads the view and set every such offset to 1.  While a sub-1 literal
subscript folded the bound to a constant, the fallback was never reached for these members; with the bound free it
indexed a member allocated ``(-16:16)`` and read at ``-10`` eleven elements before its storage.

Binding-side plumbing only: no HLFIR pipeline, bridge or Fortran compiler is involved.
"""

from pathlib import Path

from dace_fortran.bindings import (
    FlattenEntry,
    FlattenPlan,
    FlattenRecipe,
    FrozenArg,
    FrozenSignature,
    OriginalArg,
    OriginalInterface,
    emit_bindings,
)
from dace_fortran.bindings.block_builders import _build_symbol_assigns
from dace_fortran.bindings.frozen_signature import FrozenArgKind


def _frozen() -> FrozenSignature:
    """``p%end_block`` flattened to the rank-1 companion ``p_end_block``, its extent and lower bound free."""
    return FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(
            FrozenArg(
                fortran_name="p_end_block",
                sdfg_name="p_end_block",
                kind=FrozenArgKind.ARRAY,
                dtype="int32",
                rank=1,
                shape=("p_end_block_d0",),
                intent="in",
            ),
        ),
        free_symbols=("offset_p_end_block_d0", "p_end_block_d0"),
    )


def _plan() -> FlattenPlan:
    return FlattenPlan(
        entries=(
            FlattenEntry(
                outer_expr="p%end_block",
                outer_type="integer(c_int)",
                writeback_intent="",
                recipe=FlattenRecipe(
                    flat_names=("p_end_block",),
                    read_exprs=("p%end_block($i1)",),
                    rank=1,
                    shape_exprs=("size(p%end_block, dim=1)",),
                    aliasable=True,
                    scratch_dtype="int32",
                ),
            ),
        )
    )


def _iface() -> OriginalInterface:
    """A hand-authored interface: the record arg names its type, but no member layout is given."""
    return OriginalInterface(
        entry="kernel",
        args=(OriginalArg(name="p", fortran_type="type(t_patch)", rank=0, intent="in", struct_type="t_patch"),),
        struct_types={},
        used_modules={"m_patch": ("t_patch",)},
    )


def test_aliased_member_offset_reads_the_members_lower_bound():
    lines = _build_symbol_assigns(_frozen(), _plan(), {"p"}, _iface())
    text = "\n".join(lines)
    assert "    offset_p_end_block_d0 = int(lbound(p%end_block, dim=1), c_int)" in lines
    assert "    p_end_block_d0 = int(size(p%end_block, dim=1), c_int)" in lines
    assert "lbound(p_end_block," not in text
    assert "TODO: no plan entry" not in text


def test_the_assembled_wrapper_reads_the_members_lower_bound(tmp_path: Path):
    out = tmp_path / "kernel_bindings.f90"
    emit_bindings(_frozen(), _iface(), _plan(), str(out))
    src = out.read_text()
    assert "call c_f_pointer(c_loc(p%end_block), p_end_block, [size(p%end_block, dim=1)])" in src
    assert "offset_p_end_block_d0 = int(lbound(p%end_block, dim=1), c_int)" in src
    assert "lbound(p_end_block," not in src


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
