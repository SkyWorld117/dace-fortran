"""Describe a compiled library's ABI, so a language binding does not have to guess.

WHY THIS EXISTS.  The bake has always written two manifests -- ``<name>.args`` (the flat argument
list, one name per line) and ``<name>.arrays.json`` (the sorted array names) -- and they carry
NAMES ONLY.  Everything else a Fortran (or C, or Python) binding needs was reconstructed by hand, in
each binding, from the names:

  * which arguments are arrays and which are scalars -- by a PREFIX HEURISTIC (`a.startswith(("ol",
    "or"))` in the viscous-gradient shim generator),
  * an array's RANK -- by special-casing a name (`if a == "cc"` for the one 1-D array),
  * an argument's INTENT -- by the same prefix heuristic,
  * an array's runtime EXTENTS -- by a regex on the name (`^([a-z0-9]+)_d(\\d+)$`) plus a
    hard-coded mapping of subscript to the caller's extent variable.

Every one of those is a place where a renamed argument silently changes meaning, and one of them
(the extents) already caused a real bug: an extent mapping that was right for one library and wrong
for the next.  ``describe()`` reads the facts off the SDFG instead, where they are true by
construction, and ``write()`` persists them next to the library.

The three older files are left untouched: they are consumed by commit-fed generators and by the
CMake guards, and rewriting their format would be a change with no reader.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import dace
import dace.data as dt

# dace's C ABI names an array's runtime extents `<array>_d<k>`; this is how the flat arglist is
# split back into (arrays + scalars) and (extents).
EXTENT = re.compile(r'^(?P<base>.+)_d(?P<dim>\d+)$')


def _intents(sdfg) -> Dict[str, str]:
    """Read/written/both, per array, from the access nodes.

    An AccessNode's OUT edges mean its data is read by the consumer; its IN edges mean it is
    written.  Both is a read-modify-write (an accumulator, `rh = rh + ...`).  Non-transient arrays
    that never appear are reported as `unknown` rather than guessed at.
    """
    read, written = set(), set()
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.nodes.AccessNode):
            continue
        if any(True for _ in state.out_edges(node)):
            read.add(node.data)
        if any(True for _ in state.in_edges(node)):
            written.add(node.data)
    out = {}
    for name, arr in sdfg.arrays.items():
        if arr.transient or isinstance(arr, dt.Scalar):
            continue
        if name in read and name in written:
            out[name] = "inout"
        elif name in read:
            out[name] = "in"
        elif name in written:
            out[name] = "out"
        else:
            out[name] = "unknown"
    return out


def describe(sdfg) -> Dict[str, Any]:
    """The library's ABI as data: argument order, and for each, its kind/dtype/rank/shape/intent."""
    arglist = [str(a) for a in sdfg.arglist()]

    # Split the flat list into arrays and scalars, and collect each array's extent arguments.  The
    # extents follow their array in the arglist, but the association is by NAME, not position, so a
    # reordering by dace cannot break it.
    arrays, scalars = [], []
    for a in arglist:
        desc = sdfg.arrays.get(a)
        if isinstance(desc, dt.Array) and not isinstance(desc, dt.Scalar):
            arrays.append(a)
        else:
            scalars.append(a)

    # `extents` is keyed by the ARRAY an extent belongs to, so membership in it says nothing about
    # whether an argument IS an extent -- that is `ext_names`.  Conflating the two put every array
    # into the extent branch and crashed on `EXTENT.match(array)` being None.
    array_set = set(arrays)
    extents: Dict[str, List[str]] = {name: [] for name in arrays}
    ext_names = set()
    for a in arglist:
        m = EXTENT.match(a)
        if m and m.group("base") in array_set:
            extents[m.group("base")].append(a)
            ext_names.add(a)

    intents = _intents(sdfg)
    args = []
    for i, name in enumerate(arglist):
        desc = sdfg.arrays.get(name)
        if name in ext_names:
            m = EXTENT.match(name)
            base = m.group("base")
            args.append({
                "index": i,
                "name": name,
                "kind": "extent",
                "of": base,
                "dim": int(m.group("dim")),
                "dtype": "int64",
            })
            continue
        if isinstance(desc, dt.Scalar):
            args.append({
                "index": i,
                "name": name,
                "kind": "scalar",
                "dtype": str(desc.dtype),
            })
            continue
        if isinstance(desc, dt.Array):
            args.append({
                "index": i,
                "name": name,
                "kind": "array",
                "dtype": str(desc.dtype),
                "rank": len(desc.shape),
                "shape": [str(s) for s in desc.shape],
                "intent": intents.get(name, "unknown"),
                "extents": extents[name],
                "is_view": isinstance(desc, dt.View),
            })
            continue
        # Not in `arrays` at all: a symbol, not a buffer.
        args.append({"index": i, "name": name, "kind": "symbol", "dtype": "unknown"})

    return {
        "name": sdfg.name,
        "args": args,
        "order": arglist,
        "arrays": arrays,
        "scalars": scalars,
        "gpu": bool(sdfg.is_gpu_code()) if hasattr(sdfg, "is_gpu_code") else None,
    }


def write(sdfg, out_dir, name: Optional[str] = None) -> Path:
    """Write `<name>.abi.json`.  The older `.args` / `.arrays.json` are NOT touched -- they have
    readers (the commit-fed shim generators, the CMake guards) and this is an addition, not a
    replacement."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or sdfg.name
    path = out_dir / f"{name}.abi.json"
    path.write_text(json.dumps(describe(sdfg), indent=1, sort_keys=False) + "\n")
    return path


def load(lib_dir, name: str) -> Dict[str, Any]:
    """Read a library's rich manifest.  Raises with the fix, not a bare FileNotFoundError."""
    p = Path(lib_dir) / f"{name}.abi.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no rich manifest at {p} -- the library was baked before manifest.describe() existed, "
            f"or by a different tree.  Re-bake it (scripts/mfc_dace_build.py) rather than guessing "
            f"the ABI from the names.")
    return json.loads(p.read_text())
