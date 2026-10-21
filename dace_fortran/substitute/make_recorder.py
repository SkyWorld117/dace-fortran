"""Generate the recording stand-in of an MFC routine from its HLFIR.

Linked in the routine's place (link_substitute.sh, record_substitutes.sbatch) for one run of a case, the stand-in
appends the values of the configuration variables the routine reads -- the specialisation candidates dace-fortran
reports -- to a record on every call and then runs the routine itself. The record holds the values a specialised
substitute is built for, exactly as the run has them (derived ones like grid_geometry included).

A public module variable is read through its module; a private one through the accessor of a copy of its module
(module_copies.py), which link_substitute.sh compiles and links in place of the module's object (the "access" entries of
substitute.json). Candidates whose module source is not found, or whose variable cannot be made accessible, are listed
as not recordable.

Usage: make_recorder.py [--fypp <dir> ...] <file.hlfir> <module::procedure> <fypp dir> <out dir> [<record>]
  --fortran-compiler gfortran: GNU caller linker names (default nvfortran); compile the stand-in with the caller.
  <record>: a record of an earlier recorder of the same entry: its values prune the code first, so variables passed on
  only in branches this configuration never runs become recordable too.
  <fypp dir>: the build's preprocessed sources (build/staging/<slug>/fypp/simulation), searched for the modules.
  --fypp <dir>: more directories of the build's sources, searched after <fypp dir> (repeatable): the modules that
  provide the kinds and types of private variables (src/common's m_precision_select, say).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dace_fortran.substitute.link_symbols import procedure_symbol
from dace_fortran.substitute.module_copies import access_entries, module_files, write_access_copies


def _prefix_pipeline() -> str:
    """The default pipeline up to the last run of the specialisation pass (after inlining), which completes the
    candidates."""
    from dace_fortran.builder import DEFAULT_PIPELINE  # noqa: PLC0415

    passes = [p for p in DEFAULT_PIPELINE.split(",") if p.strip()]
    cut = len(passes) - passes[::-1].index("hlfir-specialize-globals")
    return ",".join(passes[:cut])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fortran-compiler", choices=("nvfortran", "gfortran"), default="nvfortran",
                    help="compiler of the caller, generated stand-in and bindings (default nvfortran)")
    ap.add_argument("--fypp", type=Path, action="append", default=[], help="more source directories (repeatable)")
    ap.add_argument("--record-per-rank", action="store_true", help="a process MPI launched records to "
                    "<record>.<rank> (the records of all ranks together are the values equal on every rank)")
    ap.add_argument("hlfir")
    ap.add_argument("entry")
    ap.add_argument("fypp_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("earlier", type=Path, nargs="?")
    args = ap.parse_args()
    hlfir, entry, out_dir, earlier = args.hlfir, args.entry, args.out_dir.resolve(), args.earlier
    from dace_fortran.bindings.fortran_interface import build_auto_interface  # noqa: PLC0415
    from dace_fortran.bindings.module_access import ModuleAccessError, public_variables  # noqa: PLC0415
    from dace_fortran.bindings.standin import RecordedValue, emit_recording_standin, standin_name  # noqa: PLC0415
    from dace_fortran.build_bridge import hb  # noqa: PLC0415
    from dace_fortran.builder import _resolve_entry_symbol  # noqa: PLC0415

    module = hb.HLFIRModule()
    if not module.parse_file(hlfir):
        raise SystemExit(f"cannot parse {hlfir}")
    symbol = _resolve_entry_symbol(module, entry)
    module.set_entry_symbol(symbol)
    known: dict[str, str] = {}  # variable -> module, every configuration candidate of either round
    if earlier is not None:
        from dace_fortran.bindings.specialization import recorded_invariants  # noqa: PLC0415
        from dace_fortran.builder import _global_value_text  # noqa: PLC0415

        invariants = recorded_invariants(earlier)
        module.set_specialized_globals({k: _global_value_text(v) for k, v in invariants.items()})
        known.update({re.split(r"[(%]", k.split("::")[1])[0].lower(): k.split("::")[0] for k in invariants})
    procedure = entry.split("::")[-1]
    iface = build_auto_interface(module.get_fortran_interface(symbol), procedure)
    module.run_passes(_prefix_pipeline())
    candidates = dict(module.get_configuration_candidates())
    known.update({re.split(r"[(%]", k.split("::")[1])[0].lower(): k.split("::")[0] for k in candidates})

    files = module_files([args.fypp_dir, *args.fypp])
    public: dict[str, set[str]] = {}
    values: list = []
    private: dict[str, list] = {}  # module -> values of its private variables
    unrecordable: list[tuple] = []  # (value, why)
    for key, ftype in sorted(candidates.items()):
        mod, name = key.split("::")
        value = RecordedValue(mod, name, ftype)
        if mod not in files:
            unrecordable.append((value, "no source of its module"))
            continue
        if mod not in public:
            public[mod] = public_variables(files[mod].read_text(errors="replace"), mod)
        # A scalar inside a record or array (``bc_x%beg``, ``idwbuff(1)%beg``, ``flags(1)``) is read through its
        # variable.
        if re.split(r"[(%]", name)[0] in public[mod]:
            values.append(value)
        else:
            private.setdefault(mod, []).append(value)

    out_dir.mkdir(parents=True, exist_ok=True)
    # The private variables, through accessors of copies of their modules; a module whose variables cannot all be
    # made accessible is left out whole (its copy would be incomplete).
    access = []
    for mod, mod_values in sorted(private.items()):
        names = sorted({re.split(r"[(%]", v.name)[0] for v in mod_values})
        try:
            copies = write_access_copies({mod: names}, files, out_dir / "access")
        except ModuleAccessError as err:
            unrecordable += [(v, str(err)) for v in mod_values]
            continue
        # Storage that may not exist holds no configuration value (and the stand-in refuses it).
        absent = sorted(n for n, var in copies[0]["variables"].items() if var.allocatable or var.pointer)
        if absent:
            unrecordable += [(v, f"{absent} allocatable or pointer") for v in mod_values]
            continue
        access += copies
        values += mod_values
    record = out_dir / f"{procedure}.record"
    shim = emit_recording_standin(
        iface,
        values,
        record,
        out_dir / f"record_{procedure}.f90",
        private_access={a["module"]: a["variables"] for a in access},
        private_imports=list(dict.fromkeys(line for a in access for line in a["imports"])),
        known=known,
        per_rank=args.record_per_rank,
    )
    # The types of the recorded values: a substitute built from the record checks them at run time
    # (make_substitute.py --globals), and needs each one's type to compare it.
    (out_dir / "recorded_types.json").write_text(
        json.dumps({f"{v.module}::{v.name}": v.fortran_type for v in values}, indent=1)
    )
    module_name = entry.split("::")[0]
    (out_dir / "substitute.json").write_text(
        json.dumps(
            {
                "shim": str(shim),
                "shim_symbol": standin_name("record", procedure) + "_",
                "wrap_symbol": procedure_symbol(module_name, procedure, args.fortran_compiler),
                "original_symbol": standin_name("original", procedure) + "_",
                "record": str(record),
                # Copies of the modules whose private variables the stand-in reads, linked in place of the originals.
                "access": access_entries(access),
            },
            indent=1,
        )
    )
    through = sum(len(private[a["module"]]) for a in access)
    print(f"{len(candidates)} candidates: {len(values)} recorded ({through} private, through accessor copies of "
          f"{len(access)} modules), {len(unrecordable)} not recordable")
    for a in access:
        for v in private[a["module"]]:
            print(f"  private (accessor): {v.module}::{v.name}")
    for v, why in unrecordable:
        print(f"  not recordable: {v.module}::{v.name} ({why})")
    print(f"wrote {shim} and {out_dir / 'substitute.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
