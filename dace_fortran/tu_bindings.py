# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The TU binding record: which SOURCE ACCESS becomes which TU DUMMY, as DATA.

A translation unit's dummies are not derivable from the Fortran.  That `dqL_prim_dx_n(2)` is the
level-2 x-gradient of the LEFT state, that `iv%beg` is momentum row 0, that the row loop indexes a
COMPONENT rather than a subscript -- none of that is in the statement, and an emitter that guessed it
would produce a unit that compiles, runs, and computes from the wrong array.  So it is recorded, and
this module is the record: `Binding.from_dict` reads it, `Binding.substitute` applies it, and
`Family.for_statements` SELECTS it.

NOT THE SAME THING AS :mod:`dace_fortran.bindings`, AND THE NAMES ARE ONE LETTER APART.  That package
runs AFTER an SDFG is built and emits a `_bindings.f90` SHIM: its relation is SDFG ABI slot -> the
Fortran the shim calls.  This module's relation is SOURCE ACCESS -> TU DUMMY, on the KERNEL side, and
it is a different record with a different consumer.  A record of one kind that drifted into the other
would still compile and still run.

THE DESIGN RULE THAT KEEPS THIS HONEST: the record holds only what is a fact about the code being
ported.  It NEVER holds arithmetic.  The statements come from the source, and an emitter that composed
them would be the transcription this whole approach replaces.

SELECTION IS BY MATCHING, NOT BY KEY, and that is forced by the data rather than chosen.  A family's
statements can vary in (direction, level written, level read, subscript order) INDEPENDENTLY, so no
key over `(op, dim)` names them -- one direction can appear twice, writing a different level each
time.  `Binding.substitute`'s completeness guard already proves that a table covers a nest, so "the
table that binds" IS the lookup, and a table that does not apply cannot be chosen by accident: it
raises.

THE BINDING IS PER CALL SITE, NOT PER FAMILY, which is why `Family` holds several `shapes`.  One
library serves every loop of one arithmetic form, so the SAME dummy is bound to different arrays at
different call sites -- and a single table per family would be checked against one site while
silently describing neither.

WHAT IS NOT HERE: where the records live (the caller passes a directory), how the statements a record
cannot bind are DERIVED from a second source routine (`append_from_source` is a consumer's, because it
reads that project's own tree), and any arithmetic at all.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Access",
    "Binding",
    "BindingError",
    "Family",
    "ShapeNotRecorded",
    "load_family",
    "rows_used",
]


class BindingError(ValueError):
    """A record that cannot be applied, or an access the record does not cover."""


class ShapeNotRecorded(BindingError):
    """The record has no table for this shape.  Raised so a caller REPORTS rather than approximates.

    The plan's rule for the whole extraction effort: discovery proposes, and a shape the emitter has
    not been generalised for must be reported, never guessed at.  A binding table written from a guess
    is a kernel reading the wrong array with a green build.
    """


def _accesses_of(raw: dict, where: str) -> List["Access"]:
    accs = []
    for a in raw["accesses"]:
        for k in ("base", "dummy", "intent", "rank"):
            if k not in a:
                raise BindingError(f"{where}: access {a.get('base')!r} has no `{k}`")
        if a["intent"] not in ("in", "inout", "out"):
            raise BindingError(f"{where}: bad intent {a['intent']!r}")
        accs.append(Access(a["base"], a["dummy"], a["intent"], int(a["rank"])))
    return accs


@dataclass
class Family:
    """A record covering SEVERAL call sites, because the binding is per call site, not per family.

    `visc_family` emits six shapes and dim0/dim1/dim2 read different accesses (`dqL_prim_dx_n(1)` vs
    `dqL_prim_dy_n(2)` vs `dqL_prim_dz_n(3)`) -- measured.  One table per family cannot say that, and
    the completeness guard inside `Binding.substitute` refuses the nest rather than emitting a unit
    with a bare dummy where a level-2 gradient belongs.
    """
    name: str
    raw: dict

    def keys(self) -> List[str]:
        return [k for k in self.raw.get("shapes", {}) if ":" in k]

    def for_statements(self, stmts) -> "Binding":
        """The table that BINDS this statement set completely.

        Selection by MATCHING rather than by key, and that is a correction the source forced: AVG's
        four statements vary in (direction, level written, level read, subscript order)
        independently -- `dz` appears twice, writing level 1 once and level 2 the other time -- so no
        key over `(op, dim)` can name them.  The guard in `substitute` already proves complete
        coverage, so "the table that binds" IS the lookup, and a table that does not apply cannot be
        chosen by accident: it raises.
        """
        if not self.keys():
            raise ShapeNotRecorded(f"{self.name}: the record has no `shapes` tables")
        fallback = None
        for key in self.keys():
            b = self.by_key(key)
            try:
                for ln in stmts:
                    b.substitute(ln, 0 if b.rows else None)
            except BindingError as _e:
                # A PARTIAL RECORD BINDS A SUBSET: an uncovered statement is DROPPED AND REPORTED by
                # the emit, not a reason to refuse the nest.  This must be honoured HERE as well as
                # there, because selection runs FIRST -- a flag wired only into the emit is inert.
                # MEASURED: that is exactly what happened, and it is the same two-places pattern
                # `rows.loop` needed.
                # ... AND IT IS A FALLBACK, NOT THE FIRST ANSWER.  Returning the first partial table
                # would pick x's for a y nest -- `fdiff_src` has three, one per direction.  A table
                # that binds FULLY always wins; a partial one is used only when none does.
                if b.partial and fallback is None:
                    fallback = b
                import os as _o
                if _o.environ.get("TU_DEBUG_BIND"):
                    print(f"        [bind] {key} refused: {str(_e)[:150]}")
                continue
            return b
        if fallback is not None:
            return fallback
        raise ShapeNotRecorded(
            f"no binding table binds this nest (tried {len(self.keys())}: {self.keys()})")

    def by_key(self, key: str) -> "Binding":
        shapes = self.raw.get("shapes", {})
        if key not in shapes:
            raise ShapeNotRecorded(f"no binding table {key!r}")
        raw = dict(self.raw)
        # A SHAPE WITHOUT `accesses` IS A SHAPE THAT CANNOT SAY WHAT IT BINDS, and it used to be a
        # bare `KeyError` here while the sibling field checks in `_accesses_of` and `from_dict` all
        # named the field they wanted.  The record is DATA, so a missing key is something to report.
        if "accesses" not in shapes[key]:
            raise BindingError(f"{self.name}[{key}]: the shape has no `accesses`")
        raw["accesses"] = shapes[key]["accesses"]
        # the loop-variable rename is per shape too: it is by SUBSCRIPT POSITION, and the source's
        # subscript order differs between dim0/dim1/dim2
        if "loop_vars" in shapes[key]:
            raw["loop_vars"] = shapes[key]["loop_vars"]
        # ... AND SO IS THE LIBRARY BLOCK, for the same reason `accesses` is: it is a property of THIS
        # shape (it names the module this shape is baked as), and `from_dict` reads at family level.
        # MEASURED: without this lift the rename never ran and the unit was written under its
        # per-pair name -- `avg_d1_l141.f90` where `visc_avg_x_mod.f90` was wanted, i.e. a silent
        # no-op that looked like success.
        if "library" in shapes[key]:
            raw["library"] = shapes[key]["library"]
        # ... AND `loop_bounds`, for the same reason.  MEASURED: without this lift the override never
        # applied and the loops rendered from the source's range (`0..m`) instead of the caller's
        # (`0 .. je - jb`) -- a silent no-op that looks like the feature working.
        if "loop_bounds" in shapes[key]:
            raw["loop_bounds"] = shapes[key]["loop_bounds"]
        if "locals" in shapes[key]:
            raw["locals"] = shapes[key]["locals"]
        # ... AND `partial`, WHICH IS WHAT MAKES THE FALLBACK RULE IN `for_statements` REACHABLE.
        # That rule -- "a table that binds FULLY always wins; a partial one is used only when none
        # does" -- was added for a MEASURED defect (returning the first partial table picked the x
        # table for a y nest).  It could not fire: `partial` was read at FAMILY level only, so every
        # table of a given family carried the SAME flag and no table could be full while another was
        # partial.  Lifting it is what makes the rule a rule rather than a branch nobody can take.
        if "partial" in shapes[key]:
            raw["partial"] = shapes[key]["partial"]
        b = Binding.from_dict(raw, where=f"{self.name}[{key}]")
        b.key = key
        return b

    def has(self, op: str, dim) -> bool:
        return f"{op}:{dim}" in self.raw.get("shapes", {})

    @property
    def reshape(self):
        """The family-level reshape spec, if this family is emitted as one kernel per shape.

        On the FAMILY rather than the `Binding` because the driver decides how to emit a shape from
        the record it loaded, before any per-shape table is selected -- and `by_key` copies this
        through to the `Binding` as well, so the emitter reads the same object either way.
        """
        return self.raw.get("reshape")


@dataclass
class Access:
    base: str          # access base, with %(row) for the unrolled row expression
    dummy: str         # dummy name template; {n} is the 1-based row ordinal
    intent: str
    rank: int


@dataclass
class Binding:
    family: str
    accesses: List[Access]
    loop_vars: Dict[str, str]
    bounds: List[str]
    rows: Optional[dict] = None
    source_routine: str = ""
    # `partial` says this record covers a SUBSET of the nest, so an uncovered statement is DROPPED
    # AND REPORTED rather than refusing the whole nest.  Must be honoured in TWO places
    # (`for_statements` selects BEFORE the emit runs), which is the same pattern `rows.loop` needed.
    #
    # THE REASON THIS WAS FIRST WRITTEN FOR MFC's `fdiff_src` WAS WRONG AND IS KEPT ONLY AS A WARNING.
    # It said the uncovered half (the advection source) was DEAD because `fdiff_src_contract()`
    # requires `hypo_nc_mode /= dual_pass`.  That condition belongs to the FLUX nest's own if/else pair
    # and says nothing about the advection arm, which is LIVE -- see `source_append` below and the
    # record's `_partial_why_superseded`.  A record that states a wrong reason is worse than one that
    # states none: it is what made a unit with live arithmetic missing look accounted for.
    partial: bool = False

    # WHICH TABLE THIS IS.  `for_statements` selects a table by MATCHING its accesses against the
    # statements, not by key -- AVG's four statements differ in direction, level written, level read
    # and subscript order independently, so no key over (op, dim) names them.  That is right for
    # SELECTION and useless for NAMING: two tables can match the same `(op, dim)` and the caller has
    # to know WHICH one it got to pick a module name.  MEASURED: the record's `AVG:dx2` and `AVG:dz2`
    # are the same shape and the same (op, dim) -- they differ only in the array they read.
    key: str = ""

    # SOMETIMES THE SAME SHAPE IS BAKED UNDER ANOTHER LIBRARY'S NAMES.  `visc_avg_x/z` and
    # `visc_avg_d0/d2` are the SAME arithmetic form emitted by two different hand generators, and the
    # two shims were generated against the two naming conventions -- `dxl1..oxr3` with
    # `kxb,kxe,lxb,lxe,jlo,jhi,jlb,jre` against `dL1..oR3` with `a_lo..c_hi,ulb,ure`.  Since DaCe
    # sorts the argument list by NAME (measured: `cc` is declared fourth and appears first, `a_hi`
    # before `a_lo`), a substitution is a RENAME and nothing else -- the declaration order is not the
    # ABI.  This is that rename, as data.
    library: Optional[dict] = None

    # `{loop_var: [begin_scalar, end_scalar]}` -- the caller passes BOTH ends and the nest runs
    # `0 .. end - begin`.  A third convention beside the source's own range and `bounds_from_caller`.
    loop_bounds: Optional[dict] = None

    # INTERMEDIATE scalars the body uses and nothing else declares -- the source's hoisted temporaries.
    # MEASURED: without them the emitted unit was INVALID FORTRAN (undeclared `inv_ds`, `flux_face1`,
    # `flux_face2`), which no bit-identity gate could have reported because the unit never compiles.
    locals: Optional[list] = None

    # WHO SUPPLIES A NON-DIFFERENCE AXIS' RANGE -- the caller, or the emitted unit itself.
    #
    # This is an INTERFACE question, not an arithmetic one, and the two families in `visc_family`
    # answer it differently.  The committed units take ALL SIX ranges as arguments and the dispatch
    # passes the SITE's own range, read off the patched source: `s_dace_visc_avg_d0` is called with
    # `is2_viscous%beg + 1, is2_viscous%end - 1`.  An emitted unit that folds the source's loop header
    # into its own text turns `is2_viscous%beg + 1` into `b_lo + 1` and, given the SAME arguments,
    # narrows a SECOND time -- MEASURED as 2214 untouched cells against 1350, worst difference 3.6e+03.
    #
    # `GRAD` never showed it because its transverse axes are already `%beg`/`%end` in the source, so
    # this flag is a NO-OP there -- which is why its device SASS is byte-identical either way.
    #
    # DEFAULT FALSE, and not True, because the opposite arrangement is legitimate: a family whose
    # caller does NOT pass the source's restriction needs the unit to carry it (`periodic` and `pack`
    # take their extents and offsets as scalars for exactly that reason).  It is a property of the
    # CALL SITES, so it belongs in the record.
    bounds_from_caller: bool = False

    # A SHAPE THE RECORD SAYS IS AN OFFSET COPY IS EMITTED DIFFERENTLY FROM A STENCIL PAIR.
    #
    # A stencil pair's arms differ on one axis and share the others, so they go into ONE nest behind
    # guards.  An offset copy's two instances are two FACES: they write disjoint planes, their
    # subscripts are related on no axis, and what they share is a SHAPE -- `dst = doff + gg`,
    # `src = soff + gg` -- with the offsets supplied at the call.  `dace_fortran.reshape` derives those
    # offsets from the source's own subscripts; the record's job is to carry what is NOT derivable:
    # which dimension is unrolled into dummies, the dummy and scalar NAMES, and the caller's values so
    # the derivation can be CHECKED rather than trusted.
    #
    # `None` means "this family is not offset-shaped", which is every family but one.
    reshape: Optional[dict] = None

    # THE RECORD AS LOADED.  `for_statements`/`by_key` build a Binding from the FAMILY's raw dict plus
    # one shape's overrides, so family-level keys survive -- and `append_from_source` needs one
    # (`source_append`) that no per-shape table can express.
    raw: dict = field(default_factory=dict)

    # ---- loading -------------------------------------------------------------------------------
    @staticmethod
    def load(path) -> "Binding":
        raw = json.loads(Path(path).read_text())
        return Binding.from_dict(raw, where=str(path))

    @staticmethod
    def from_dict(raw: dict, where: str = "<dict>") -> "Binding":
        if "accesses" not in raw:
            raise BindingError(f"{where}: no `accesses`")
        accs = _accesses_of(raw, where)
        rows = raw.get("rows")
        if rows is not None:
            for k in ("var", "from", "count"):
                if k not in rows:
                    raise BindingError(f"{where}: `rows` has no `{k}`")
        return Binding(family=raw.get("family", Path(where).stem),
                       accesses=accs,
                       loop_vars=raw.get("loop_vars", {}),
                       bounds=list(raw.get("bounds", [])),
                       rows=rows,
                       source_routine=raw.get("source_routine", ""),
                       partial=bool(raw.get("partial", False)),
                       bounds_from_caller=bool(raw.get("bounds_from_caller", False)),
                       reshape=raw.get("reshape"),
                       library=raw.get("library"),
                       loop_bounds=raw.get("loop_bounds"),
                       locals=list(raw.get("locals") or []),
                       raw=raw)

    # ---- applying ------------------------------------------------------------------------------
    def row_expressions(self) -> List[str]:
        """The row expression per unrolled copy -- `iv%beg + 0`, `iv%beg + 1`, ... .

        ``rows.loop`` INVERTS this: the row stays a LOOP, so `%(row)` expands to the loop VARIABLE
        itself and the accesses index by it directly.  That is a shape the other families do not
        have and MFC's `pack` does -- there the row is a real SUBSCRIPT of a rank-4 destination
        (`v_rs_weno(j, k, l, i) = v_vf(i)(j, k, l)`), so unrolling is the identity and merely DROPS
        the `do i` header while the subscript still names `i`.  MEASURED: the emitted unit had
        `i` undeclared.  A family that must not unroll says so; the default is unchanged.
        """
        if not self.rows:
            return []
        if self.rows.get("loop"):
            return [str(self.rows["var"])]
        base = self.rows["from"]
        return [f"{base} + {r}" if r else base for r in range(int(self.rows["count"]))]

    def _match(self, base: str, row_expr: Optional[str]) -> Optional[Access]:
        for a in self.accesses:
            want = a.base
            if "%(row)" in want:
                if row_expr is None:
                    continue
                want = want.replace("%(row)", row_expr)
            if want == base:
                return a
        return None

    def substitute(self, stmt: str, row_index: Optional[int]) -> str:
        """Replace every bound access in one statement with its dummy, for unrolled copy `row_index`."""
        row_expr = None
        if row_index is not None:
            if not self.rows:
                raise BindingError("a row loop was unrolled but the record has no `rows`")
            row_expr = self.row_expressions()[row_index]
        n = 1 if row_index is None else row_index + 1
        out = stmt
        # EXPAND THE ROW INDEX FIRST.  The statement says `%vf(i)` -- the row VARIABLE -- while the
        # record's base says `%vf(%(row))`, meaning the unrolled expression.  Matching without this
        # step leaves every access unbound, and the guard then refuses the whole nest (which is how
        # this was found: the refusals named the row variable itself).
        if row_expr is not None:
            var = self.rows["var"]
            out = re.sub(rf"%vf\(\s*{re.escape(var)}\s*\)", f"%vf({row_expr})", out)
            # ... AND PLAIN ARRAYS INDEXED BY THE ROW VARIABLE.  `rhs_vf(j)%sf(...)` carries the row
            # in a SUBSCRIPT, not in a `%vf` component, so the expansion above leaves it alone and the
            # access then fails to bind (`the record does not bind these access(es): ['rhs_vf(j)']`).
            # Only arrays the record declares with `%(row)` are touched, so an unrelated `foo(j)` is
            # left as the source wrote it.
            for a in self.accesses:
                if "%(row)" not in a.base:
                    continue
                head = a.base.split("%(row)")[0]
                if not head.endswith("("):
                    continue
                name = head[:-1].split("%")[-1]
                out = re.sub(rf"\b{re.escape(name)}\(\s*{re.escape(var)}\s*\)",
                             f"{name}({row_expr})", out)
        # longest base first: `q_prim_qp%vf(i)%sf` must win over any prefix that also matches
        for a in sorted(self.accesses, key=lambda x: -len(x.base)):
            if "%(row)" in a.base and row_expr is None:
                continue
            want = a.base.replace("%(row)", row_expr) if row_expr is not None else a.base
            dummy = a.dummy.replace("{n}", str(n))
            # the access is `<base>%sf(...)`, and `%sf` is the array's storage component
            out = re.sub(re.escape(want) + r"\s*%sf\s*\(", dummy + "(", out)
        # ... AND THE PLAIN ARRAYS, which carry no `%sf` and so the pass above cannot see.  The GRAD
        # shapes divide by a coordinate spacing that is a MODULE ARRAY (`x_cc(j)`), and leaving it
        # unsubstituted is a TU that does not compile -- "Function 'x_cc' at (1) has no IMPLICIT
        # type" -- which reads like a missing declaration and is a missing SUBSTITUTION.
        for a in sorted(self.accesses, key=lambda x: -len(x.base)):
            if "%" in a.base:
                continue
            want = a.base.replace("%(row)", row_expr) if row_expr is not None else a.base
            dummy = a.dummy.replace("{n}", str(n))
            out = re.sub(r"\b" + re.escape(want) + r"\s*(?=\()", dummy, out)
        # ANY surviving array-component access means the record does not cover the nest.  Refuse
        # rather than emit: an uncovered access becomes an undeclared dummy, and an undeclared dummy
        # is a TU that does not compile -- or worse, one that does and reads the wrong array.
        left = re.findall(r"([A-Za-z_]\w*(?:\([^()]*\))?(?:%\w+(?:\([^()]*\))?)*)%sf\s*\(", out)
        if left:
            # NAME THE STATEMENT, not only the access: the same access may be bound in one arm and
            # not the other, and the bare list gives no way to tell which.  (The access names also
            # appear already-rewritten for the copies that DID match, which reads as a contradiction.)
            raise BindingError(
                f"the record does not bind these access(es): {sorted(set(left))}\n"
                f"        in: {' '.join(out.split())[:220]}")
        # and a loop variable that reaches nothing is the exact failure this record exists to stop
        if self.rows and self.rows["var"] in re.findall(r"\b(\w+)\b", out) and row_index is None:
            raise BindingError(f"row variable {self.rows['var']!r} still appears after substitution")
        return out

    def rename_loops(self, stmt: str) -> str:
        out = stmt
        for src, dst in self.loop_vars.items():
            all_names = set(self.loop_vars) | set(self.loop_vars.values())
            skip = all_names - {src}
            out = re.sub(rf"\b{re.escape(src)}\b",
                         lambda m: m.group(0) if m.group(0) in skip else dst, out)
        return out

    def dummies(self) -> List[Tuple[str, int, str]]:
        """``(name, rank, intent)`` per dummy -- ``dace_fortran.tu.module``'s order."""
        out, seen = [], set()
        n = int(self.rows["count"]) if self.rows else 1
        for a in self.accesses:
            for r in range(n):
                name = a.dummy.replace("{n}", str(r + 1))
                if name not in seen:
                    seen.add(name)
                    out.append((name, a.rank, a.intent))
        return out


def load_family(name: str, directory) -> "Family":
    """The record for a family, which may cover several call sites (`shapes`).

    ``directory`` is the caller's.  The record FORMAT is general; WHERE a project keeps
    its records is not, so this takes the path rather than knowing a layout.
    """
    p = Path(directory) / f"{name}.json"
    if not p.exists():
        raise BindingError(f"no TU binding record at {p}")
    raw = json.loads(p.read_text())
    if "shapes" not in raw:
        raise BindingError(
            f"{p}: a family record needs a `shapes` map -- the binding is per CALL SITE, so a family "
            f"with more than one shape cannot be described by a single access table")
    return Family(name=raw.get("family", p.stem), raw=raw)


def rows_used(stmts: Sequence[str], binding: Binding) -> bool:
    """Whether any statement references the component index that the row loop drives."""
    if not binding.rows:
        return False
    var = binding.rows["var"]
    return any(re.search(rf"%vf\(\s*{re.escape(var)}\s*\)", s) for s in stmts)
