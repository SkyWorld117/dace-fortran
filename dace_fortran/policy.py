# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The project-specific half of TU generation, as DATA.

The three stages before this one -- :mod:`dace_fortran.discovery`, :mod:`dace_fortran.extract`,
:mod:`dace_fortran.tu` -- know Fortran and nothing else.  What they cannot know is which routine in
which file holds the loops, which array becomes which dummy, and what a project calls its field
accesses.  That is the *policy*, and it is the caller's, not this library's.

WHY A FILE AND NOT A FUNCTION.  The porting scripts had it as Python spread across ten generators,
which meant it could only be read by someone reading all ten, and it could only be changed by editing
code.  A declarative file is reviewable in a diff, diffable across a rebase, and -- the reason it
matters here -- it gives the project-specific half a VERSIONED HOME.  A migration that moved the
generic machinery into a library while leaving the policy in an untracked scratch directory would
have moved the risk rather than removed it.

THE RULE THIS SCHEMA EXISTS TO HOLD: this library may know Fortran, never one project's physics.
Anything that would tell dace-fortran what the numbers MEAN -- which row is energy, what an equation
of state is -- is out of scope, and belongs in the prelude the project supplies.

JSON rather than YAML: the only dependency in the standard library, and this file is written by a
generator as often as by a person.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple


class PolicyError(ValueError):
    """The policy is missing something the generator cannot invent, or is self-contradictory.

    Raised at LOAD time rather than at emit time: a policy that names a kernel without saying which
    file it is in should fail the moment it is read, not three stages later with a confusing message
    about a nest that could not be found.
    """


@dataclass
class Kernel:
    """One kernel to generate: where its source lives and what marks its loops."""
    name: str
    fpp: str
    anchor: Optional[str] = None
    routine: Optional[str] = None
    defines: Tuple[str, ...] = ()
    loop_back: Optional[str] = None
    rules: Sequence[Tuple[Pattern, str]] = ()

    @property
    def flatten(self) -> List[Tuple[Pattern, str]]:
        return list(self.rules)


@dataclass
class Policy:
    kernels: Dict[str, Kernel] = field(default_factory=dict)
    # Which caller-supplied extent variable carries each storage dimension.  This is the one fact the
    # ABI manifest cannot supply (see dace_fortran.manifest): it follows from the TU's indexing
    # convention, not from the compiled library.
    extents: Dict[int, str] = field(default_factory=dict)
    access: str = "%sf"

    def kernel(self, name: str) -> Kernel:
        if name not in self.kernels:
            raise PolicyError(f"no kernel {name!r} in the policy; it has "
                              f"{sorted(self.kernels)}")
        return self.kernels[name]


def _compile_rules(raw: Any, where: str) -> List[Tuple[Pattern, str]]:
    out = []
    for i, rule in enumerate(raw or ()):
        if not (isinstance(rule, (list, tuple)) and len(rule) == 2):
            raise PolicyError(f"{where}: rule {i} must be [pattern, replacement], got {rule!r}")
        pat, repl = rule
        try:
            out.append((re.compile(pat), repl))
        except re.error as e:
            raise PolicyError(f"{where}: rule {i} is not a valid regex ({e}): {pat!r}") from e
    return out


def load(path) -> Policy:
    """Read a policy file.  Validates, because every field it can check is a field the generator
    would otherwise have to guess at emit time."""
    p = Path(path)
    if not p.exists():
        raise PolicyError(f"no policy file at {p}")
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise PolicyError(f"{p} is not valid JSON: {e}") from e

    kernels: Dict[str, Kernel] = {}
    for name, spec in (raw.get("kernels") or {}).items():
        if not isinstance(spec, dict):
            raise PolicyError(f"{p}: kernel {name!r} must be an object")
        if "fpp" not in spec and "source" not in spec:
            # The single most likely authoring mistake, and the one whose failure is least legible
            # three stages downstream.
            raise PolicyError(f"{p}: kernel {name!r} names no 'fpp' source file")
        kernels[name] = Kernel(
            name=name,
            fpp=spec.get("fpp") or spec.get("source"),
            anchor=spec.get("anchor"),
            routine=spec.get("routine"),
            defines=tuple(spec.get("defines", ())),
            loop_back=spec.get("loop_back"),
            rules=_compile_rules(spec.get("rules"), f"{p}: kernel {name!r}"),
        )

    extents = {}
    for dim, var in (raw.get("extents") or {}).items():
        try:
            extents[int(dim)] = str(var)
        except ValueError as e:
            raise PolicyError(f"{p}: extent key {dim!r} is not a storage dimension index") from e

    return Policy(kernels=kernels, extents=extents,
                  access=raw.get("access", "%sf"))


def dump(policy: Policy, path) -> None:
    """Write a policy back out (round-trip is exact for the fields the schema covers)."""
    raw = {
        "access": policy.access,
        "extents": {str(k): v for k, v in sorted(policy.extents.items())},
        "kernels": {
            n: {k: v for k, v in {
                "fpp": k_.fpp, "anchor": k_.anchor, "routine": k_.routine,
                "defines": list(k_.defines) or None, "loop_back": k_.loop_back,
                "rules": [[r.pattern, s] for r, s in k_.rules] or None,
            }.items() if v is not None}
            for n, k_ in policy.kernels.items()
        },
    }
    Path(path).write_text(json.dumps(raw, indent=2, sort_keys=False) + "\n")
