"""The extent and lower-bound symbols of an array the bridge renamed keep the Fortran name: ``dir`` of
m_acoustic_src reaches the SDFG as ``program_dir`` while its subscripts read ``offset_dir_d0`` and loops ``dir_d0``.
The binding sourced symbols by SDFG name only and refused the whole binding (UnsourcedSymbolError: hypoelastic and
bubble cases); now they are read off the renamed array's own source. Binding-side plumbing only."""
from dace_fortran.bindings import FlattenPlan, FrozenArg, FrozenSignature, OriginalInterface
from dace_fortran.bindings.block_builders import _build_symbol_assigns
from dace_fortran.bindings.frozen_signature import FrozenArgKind, ModuleOrigin


def _frozen() -> FrozenSignature:
    return FrozenSignature(
        entry="kernel",
        mangled="_QPkernel",
        args=(
            FrozenArg(
                fortran_name="dir",
                sdfg_name="program_dir",
                kind=FrozenArgKind.ARRAY,
                dtype="float64",
                rank=1,
                shape=("dir_d0",),
                intent="in",
                module_origin_allocatable=True,
            ),
        ),
        free_symbols=("dir_d0", "offset_dir_d0"),
        module_symbol_origins={"program_dir": ModuleOrigin(module="m_acoustic_src", entity="dir")},
    )


def test_renamed_array_symbols_read_the_arrays_source():
    frozen = _frozen()
    iface = OriginalInterface(entry="kernel", args=(), struct_types={}, used_modules={})
    lines = _build_symbol_assigns(frozen, FlattenPlan(entries=()), set(), iface, passed={"dir_d0", "offset_dir_d0"})
    text = "\n".join(lines)
    assert "offset_dir_d0 = int(lbound(" in text and "dir_d0 = int(size(" in text
    assert "TODO" not in text
