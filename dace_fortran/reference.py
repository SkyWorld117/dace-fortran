# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Call a built SDFG correctly, and compare its results against a reference -- bit-exactly.

Two jobs that every kernel validation needs and that were re-written per kernel:

**Binding the ABI's extents.**  A compiled SDFG's argument list contains, for each array, runtime
extent arguments (``<array>_d0``, ``<array>_d1``, ...) that the caller must fill from the array it is
actually passing -- they are the leading dimensions, so the kernel's index arithmetic strides
correctly.  Getting one wrong is not a crash: it is a plausible-looking wrong answer.  Extending
``offset_`` symbols are bound to 1 for the same reason.

**The comparison itself.**  This is the project's whole correctness bar: the pipeline reorders
statements and forms maps but never reassociates arithmetic, so anything short of BIT-IDENTICAL is a
bug, not rounding.  :func:`compare` is that bar, written once, with the two traps named:

  * a plain ``max|diff|`` is not enough -- a *filtered* comparison (ignore cells below some
    magnitude) passes vacuously on a field whose values all sit under the threshold, which is how one
    investigation spent rounds chasing a difference that was never there;
  * ``nan`` compares unequal to itself, so a NaN in either array must be reported as a mismatch
    rather than silently propagating through the arithmetic.

:func:`compare` returns per-array results and does NOT print or raise: whether a mismatch is fatal is
the caller's call, and a library that decided it would be wrong for the callers that want to report
several kernels before failing.

Related but different: ``pipelines.verify_numerics`` compares TWO SDFGs (pre- vs post-optimization) on
the same inputs.  This compares an SDFG's results against numbers that came from somewhere else --
a gfortran reference, a golden file, another implementation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

EXTENT = re.compile(r'^(\w+)_d(\d+)$')


@dataclass
class Comparison:
    """One array's verdict.  ``equal`` is the whole answer; the rest is for the report."""
    name: str
    equal: bool
    total: int
    mismatches: int
    max_abs: float
    finite: bool

    def line(self, tag: str = "") -> str:
        head = f"{tag}{self.name}" if tag else self.name
        return (f"{head}: bit-exact={self.equal}  finite={self.finite}  "
                f"max|diff|={self.max_abs:.3e}  mismatches={self.mismatches}/{self.total}")


def bind_extents(sdfg, kwargs: Dict[str, Any]) -> None:
    """Fill the extent and offset arguments in ``kwargs`` from the arrays being passed.  In place.

    Duck-typed on ``.shape``, so numpy and cupy both work.  An argument whose base array is not in
    ``kwargs`` is left alone -- it is not ours to invent.
    """
    for k in sdfg.arglist():
        base = re.sub(r"^offset_", "", str(k))
        m = EXTENT.match(base)
        if not m or m.group(1) not in kwargs:
            continue
        shape = getattr(kwargs[m.group(1)], "shape", None)
        if shape is None:
            continue
        dim = int(m.group(2))
        if dim >= len(shape):
            raise ValueError(
                f"{k}: the argument names extent {dim} of {m.group(1)!r}, which has "
                f"{len(shape)} dimension(s) -- the array being passed does not match the kernel's "
                f"assumption about it")
        kwargs[k] = 1 if str(k).startswith("offset_") else int(shape[dim])


def compare(actual: Dict[str, Any], reference: Dict[str, Any],
            names: Optional[List[str]] = None) -> List[Comparison]:
    """Bit-exact comparison of each named array against its reference.

    Arrays present in only one of the two mappings are skipped rather than treated as failures: a
    kernel's argument list and a reference file's contents rarely coincide exactly, and inventing a
    mismatch for an absent reference would be worse than saying nothing.
    """
    keys = list(names) if names is not None else [k for k in reference if k in actual]
    out: List[Comparison] = []
    for k in keys:
        a = np.asarray(actual[k])
        b = np.asarray(reference[k])
        force = a.shape != b.shape
        if not force:
            # `nan != nan`, so a NaN would otherwise count as a mismatch only by luck of the
            # comparison used -- count them explicitly instead.
            both_finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            same_bits = np.array_equal(a, b)
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            out.append(Comparison(k, bool(same_bits and both_finite), a.size,
                                  int(np.count_nonzero(~(a == b))) if both_finite else a.size,
                                  float(diff.max()) if diff.size and np.isfinite(diff).all() else
                                  float("inf"),
                                  both_finite))
            continue
        out.append(Comparison(k, False, a.size, a.size, float("inf"), False))
    return out


def all_equal(results: List[Comparison]) -> bool:
    """The verdict, for a caller that wants one answer rather than a table."""
    return bool(results) and all(c.equal for c in results)


def reference_from_record(path, shape=None, dtype=np.float64) -> np.ndarray:
    """Read a reference a Fortran program wrote (a sequential-unformatted record).

    The usual shape of these files in the codebase this was written for: the TU is compiled with
    gfortran, run on synthetic input, and its outputs dumped for exactly this comparison.
    """
    from dace_fortran import unformatted
    a = unformatted.read_record(path, dtype)
    return a.reshape(shape) if shape is not None else a
