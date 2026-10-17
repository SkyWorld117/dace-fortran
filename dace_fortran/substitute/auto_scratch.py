"""Infer the module arrays a routine only uses as scratch (make_substitute.py --auto-scratch).

A module array, or a module record whose packed image the SDFG takes, is scratch for the routine when

* (in call) every element the routine reads it at was written earlier in the same call -- no value flows in from
  before the call: incall_scratch.uncovered_reads on the SDFG as the substitute runs it (specialised to the recorded
  values the binding checks, so the branches it never takes are gone), and
* (outside) no value it leaves is read after the call: no procedure that runs outside the routine's calls reads it,
  aliases it or takes its address (outside_refs.outside_readers on the build's preprocessed sources).

Then the binding need not copy it in or out, and the SDFG may hold it as a transient. Aliased variables (a pointer
of one associated with the other) are scratch only together.
"""

from __future__ import annotations

import copy
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import dace

from dace_fortran.bindings.frozen_signature import get_frozen_signature

from dace_fortran.substitute import incall_scratch
from dace_fortran.substitute import outside_refs


@dataclass
class Inference:
    """Per candidate (``module::variable``): its SDFG containers and why it is not scratch (empty: it is)."""

    containers: dict[str, list[str]] = field(default_factory=dict)
    reasons: dict[str, list[str]] = field(default_factory=dict)
    asserted: list[str] = field(default_factory=list)  # given as scratch (--scratch), not inferred
    unproven: dict[str, list[str]] = field(default_factory=dict)  # of those, the ones the analysis rejects: why

    used: set[str] = field(default_factory=set)  # containers the analysed SDFG accesses

    @property
    def scratch(self) -> list[str]:
        return sorted(k for k in self.containers if not self.reasons.get(k) and k not in self.asserted)

    def report(self) -> str:
        lines = [f"auto-scratch: {len(self.scratch)} of {len(self.containers)} module variables the SDFG takes are "
                 "scratch:"]
        lines += ["  " + chunk for chunk in textwrap.wrap(" ".join(self.scratch), 116)]
        if self.asserted:
            lines.append(f"  given as scratch (--scratch): {self.asserted}")
        for key, why in sorted(self.unproven.items()):
            if why:
                lines.append(f"  given, but not provably scratch: {key}: {why[0]}")
        for key in sorted(k for k in self.containers if self.reasons.get(k)):
            lines.append(f"  not scratch: {key} ({', '.join(self.containers[key])}): {self.reasons[key][0]}"
                         + (f" (+{len(self.reasons[key]) - 1} more)" if len(self.reasons[key]) > 1 else ""))
        return "\n".join(lines)


def candidates(sdfg: dace.SDFG, synthetic: Iterable = ()) -> dict[str, list[str]]:
    """``module::variable`` -> the non-transient arrays of ``sdfg`` that hold (parts of) it: module arrays (by their
    origin) and the packed images of module records (by the record they were built from). A flattened member of a
    module record (``rec(:)%beg`` as ``rec_beg``) counts under its record, as the frontend's flatten plan
    names it in ``synthetic`` (FlattenPlan.synthetic_globals)."""
    frozen = get_frozen_signature(sdfg)
    if frozen is None:
        return {}
    records = {(g.module.lower(), g.symbol.rsplit("E", 1)[-1].lower()): g.entity.lower() for g in synthetic}
    out: dict[str, list[str]] = defaultdict(list)
    for name, origin in frozen.module_symbol_origins.items():
        desc = sdfg.arrays.get(name)
        if isinstance(desc, dace.data.Array) and not desc.transient:
            module, entity = origin.module.lower(), origin.entity.lower()
            out[f"{module}::{records.get((module, entity), entity)}"].append(name)
    for a in frozen.args:
        struct = getattr(a, "aos_origin_struct", "")
        desc = sdfg.arrays.get(a.sdfg_name)
        if not struct or not a.aos_origin_mod or not isinstance(desc, dace.data.Array) or desc.transient:
            continue
        key = f"{a.aos_origin_mod}::{struct.split('%')[0].strip()}".lower()
        if a.sdfg_name not in out[key]:
            out[key].append(a.sdfg_name)
    return dict(out)


def scratch_keys(sdfg: dace.SDFG, containers: Iterable[str]) -> list[str]:
    """What make_scratch_transient matches for the arrays ``containers``: a module array's origin
    (``module::entity``, flattened names included), a record image's record (``module::record``)."""
    frozen = get_frozen_signature(sdfg)
    images = {a.sdfg_name: f"{a.aos_origin_mod}::{a.aos_origin_struct.split('%')[0].strip()}".lower()
              for a in frozen.args if getattr(a, "aos_origin_struct", "") and a.aos_origin_mod}
    keys = []
    for name in containers:
        origin = frozen.module_symbol_origins.get(name)
        key = f"{origin.module}::{origin.entity}".lower() if origin is not None else images.get(name)
        if key and key not in keys:
            keys.append(key)
    return keys


def accessed(sdfg: dace.SDFG) -> set[str]:
    """Containers some state or interstate edge of ``sdfg`` accesses (nested SDFGs through their outer nodes)."""
    names = {n.data for st in sdfg.all_states() for n in st.data_nodes()}
    for e in sdfg.all_interstate_edges():
        names |= {m.data for m in e.data.get_read_memlets(sdfg.arrays)}
    return names


def infer(
    sdfg: dace.SDFG,
    entry: str,
    specialise: Callable[[dace.SDFG], dict[str, dict[int, str]]] | None,
    source_dirs: Iterable[Path],
    outside_procedures: Iterable[str] = (),
    asserted: Iterable[str] = (),
    synthetic: Iterable = (),
) -> Inference:
    """The scratch module variables of the routine ``entry`` (``module::routine``) that ``sdfg`` (as built) is.

    ``specialise``: applied to a copy of ``sdfg`` before the in-call analysis, as the build will specialise it;
    returns the arrays it split into per-element ones (record_split). ``source_dirs``: the build's (preprocessed)
    sources; without them nothing is scratch (nothing proves the outside condition). ``outside_procedures``: callees
    whose calls the build drops or keeps external -- they count as outside the routine. ``asserted``: variables
    given as scratch (``--scratch``): taken as scratch where another variable is aliased with them. ``synthetic``:
    the flattened members of module records (FlattenPlan.synthetic_globals)."""
    t0 = time.time()
    found = candidates(sdfg, synthetic)
    result = Inference(containers=found)
    if not found:
        return result
    analysed = sdfg
    split: dict[str, dict[int, str]] = {}
    if specialise is not None:
        analysed = copy.deepcopy(sdfg)
        split = specialise(analysed) or {}
    names_of = {key: [n for name in names for n in (split[name].values() if name in split else [name])]
                for key, names in found.items()}
    used = accessed(analysed)
    result.used = {name for key, names in found.items() for name in names
                   if any(n in used for n in (split[name].values() if name in split else [name]))}
    incall = incall_scratch.uncovered_reads(analysed, [n for ns in names_of.values() for n in ns if n in used])
    t1 = time.time()
    for key, names in names_of.items():
        why = [f"{n}: {r}" for n in names for r in incall.get(n, [])]
        if why:
            result.reasons[key] = why
    dirs = [Path(d) for d in source_dirs]
    if not dirs:
        for key in found:
            result.reasons.setdefault(key, []).append("no sources (--fypp) to check what reads it outside the routine")
        return result
    prog = outside_refs.parse_sources(outside_refs.source_files(dirs))
    module, routine = (s.lower() for s in entry.split("::"))
    keys = [tuple(k.split("::")) for k in found]
    verdicts = outside_refs.outside_readers(prog, (module, routine), keys, outside_procedures)
    for (mod, var), v in verdicts.items():
        decl = prog.modules.get(mod)
        if decl is not None and var not in decl.declared | decl.guarded_declared:
            v.reasons.append(f"no variable {var} declared in module {mod} of the sources")
    given = {a.lower() for a in asserted}
    for key, v in verdicts.items():
        k = "::".join(key)
        v.reasons = result.reasons.get(k, []) + v.reasons  # in-call reasons first
        if k in given:
            result.unproven[k] = v.reasons
            v.reasons = []
        result.reasons[k] = v.reasons
    outside_refs.close_alias_groups(verdicts)
    for key, v in verdicts.items():
        result.reasons["::".join(key)] = v.reasons
    for key, names in names_of.items():
        # Nothing to gain: the build drops a module input the specialised SDFG never accesses. (For its aliases it
        # counted as what it is: a variable the routine leaves alone.)
        if not any(n in used for n in names) and not result.reasons.get(key):
            result.reasons[key] = ["not accessed once specialised (dropped as an unread input instead)"]
    result.asserted = sorted(given & set(found))
    print(f"auto-scratch: in-call analysis {t1 - t0:.0f} s, sources {time.time() - t1:.0f} s", flush=True)
    return result


def image_keys(sdfg: dace.SDFG) -> dict[str, list[str]]:
    """``module::record`` -> the buffers of members of the module array of records ``record`` the SDFG takes as
    arguments that the binding can keep between calls (deferred-shape members it packs itself)."""
    from dace_fortran.bindings.block_builders import keepable

    frozen = get_frozen_signature(sdfg)
    out: dict[str, list[str]] = defaultdict(list)
    for a in frozen.args if frozen is not None else ():
        desc = sdfg.arrays.get(a.sdfg_name)
        if keepable(a) and isinstance(desc, dace.data.Array) and not desc.transient:
            out[f"{a.aos_origin_mod}::{a.aos_origin_struct.split('%')[0].strip()}".lower()].append(a.sdfg_name)
    return dict(out)


def infer_kept(sdfg: dace.SDFG, entry: str, source_dirs: Iterable[Path],
               outside_procedures: Iterable[str] = ()) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The module arrays of records only the routine ``entry`` uses, whose buffers the binding can keep between its
    calls (``emit_bindings(kept_images=...)``): no code outside the routine's calls reads them, writes them, aliases
    them or takes their address (outside_refs on the build's sources).  Then the buffer the binding packed once
    evolves exactly as the members would, and nothing ever looks at the members again; unlike scratch, values may flow
    from one call into the next.  Returns ``(kept, rejected)``: ``module::record`` -> its buffers, and -> why not."""
    images = image_keys(sdfg)
    dirs = [Path(d) for d in source_dirs]
    if not images:
        return {}, {}
    if not dirs:
        return {}, {k: ["no sources (--fypp) to check what touches it outside the routine"] for k in images}
    prog = outside_refs.parse_sources(outside_refs.source_files(dirs))
    module, routine = (s.lower() for s in entry.split("::"))
    verdicts = outside_refs.outside_readers(prog, (module, routine), [tuple(k.split("::")) for k in images],
                                            outside_procedures)
    kept, rejected = {}, {}
    for key, v in verdicts.items():
        k = "::".join(key)
        why = list(v.reasons)
        why += [f"written outside the routine's calls in {w}" for w in v.writes]
        why += [f"pointer-associated with {o[0]}::{o[1]}" for o in sorted(v.aliases)]
        if why:
            rejected[k] = why
        else:
            kept[k] = images[k]
    return kept, rejected
