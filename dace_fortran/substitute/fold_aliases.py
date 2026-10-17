"""Fold the image of a member of an element of a module array of records onto the array it IS.

A program may point ``rec(1)%f`` at the module array ``arr`` and hand ``rec`` to other routines: the SDFG takes the
image ``rec_f`` and ``arr`` as two arrays, and the binding's storage guard declines every call (the two share
storage). The sizing run records the association (``<record>.aliases``, see
``dace_fortran.bindings.storage_guard.build_alias_record``): here every access of the image becomes an access of the
array (its leading element dimension of one dropped), in the nested SDFGs too, the image's extent and offset symbols
become the array's, and the image leaves the SDFG and its frozen signature.
"""

from __future__ import annotations

from dataclasses import replace

import dace
from dace import nodes, subsets, symbolic
from dace.sdfg import dealias
from dace_fortran.bindings.frozen_signature import FrozenSignature, attach_to_sdfg, get_frozen_signature


def _is_zero(expr) -> bool:
    return symbolic.simplify(symbolic.pystr_to_symbolic(expr)) == 0


def _foldable(sdfg: dace.SDFG, image: str, elem: int, array: str) -> str:
    """Why the image cannot be folded onto the array, '' when it can: one element, the array's rank and dtype, every
    access at that element."""
    if image not in sdfg.arrays:
        return "not an array of the SDFG"
    if array not in sdfg.arrays:
        return f"{array} is not an array of the SDFG"
    img, arr = sdfg.arrays[image], sdfg.arrays[array]
    if elem != 0 or not _is_zero(symbolic.pystr_to_symbolic(img.shape[0]) - 1):
        return f"element {elem} of {img.shape[0]}: only a one-element array of records folds"
    if len(img.shape) != len(arr.shape) + 1:
        return f"rank {len(img.shape)} against the array's {len(arr.shape)}"
    if img.dtype != arr.dtype:
        return f"{img.dtype} against the array's {arr.dtype}"
    if img.transient or arr.transient:
        return "a transient"
    for state in sdfg.states():
        for e in state.edges():
            if e.data.data == image and e.data.subset is not None:
                b, en, _s = e.data.subset.ranges[0]
                if not (_is_zero(b) and _is_zero(en)):
                    return f"an access at {e.data.subset} in {state.label}"
            if e.data.other_subset is not None and e.data.data != image and any(
                    isinstance(n, nodes.AccessNode) and n.data == image for n in (e.src, e.dst)):
                b, en, _s = e.data.other_subset.ranges[0]
                if not (_is_zero(b) and _is_zero(en)):
                    return f"a copy at {e.data.other_subset} in {state.label}"
    return ""


def _symbol_replacements(sdfg: dace.SDFG, image: str, array: str, known: dict) -> dict[str, str] | str:
    """The image's extent and offset symbols -> the array's expressions, or why one has no value."""
    arr = sdfg.arrays[array]
    repl: dict[str, str] = {}
    for k in range(1, len(sdfg.arrays[image].shape)):
        d, o = f"{image}_d{k}", f"offset_{image}_d{k}"
        if d in sdfg.symbols:
            repl[d] = str(arr.shape[k - 1])
        if o in sdfg.symbols:
            target = f"offset_{array}_d{k - 1}"
            if target in sdfg.symbols:
                repl[o] = target
            elif target in known:
                repl[o] = str(int(known[target]))
            else:
                return f"no value for {target}"
    return repl


def fold_aliases(sdfg: dace.SDFG, aliases: list[tuple[str, int, str]], known: dict | None = None) -> list[str]:
    """Fold every ``(image, element, array)`` alias that can be; returns a line per alias saying what happened. The
    SDFG's frozen signature loses the images; an array whose image was written counts as written."""
    known = known or {}
    done = []
    frozen = get_frozen_signature(sdfg)
    for image, elem, array in aliases:
        why = _foldable(sdfg, image, elem, array)
        if not why:
            repl = _symbol_replacements(sdfg, image, array, known)
            why = repl if isinstance(repl, str) else ""
        if why:
            done.append(f"{image}[{elem}] kept apart from {array}: {why}")
            continue
        zero = subsets.Range.from_indices(subsets.Indices([0] * len(sdfg.arrays[image].shape)))
        for state in sdfg.states():
            for e in list(state.edges()):
                if e.data.data == image:
                    if e.data.subset is not None:
                        e.data.subset = subsets.Range(e.data.subset.ranges[1:])
                    e.data.data = array
                elif e.data.other_subset is not None and any(
                        isinstance(n, nodes.AccessNode) and n.data == image for n in (e.src, e.dst)):
                    e.data.other_subset = subsets.Range(e.data.other_subset.ranges[1:])
                # The nested SDFG behind the connector takes the array's descriptor, the element dimension squeezed.
                for node, conn in ((e.dst, e.dst_conn), (e.src, e.src_conn)):
                    if isinstance(node, nodes.NestedSDFG) and conn in node.sdfg.arrays and e.data.data == array \
                            and len(node.sdfg.arrays[conn].shape) == len(sdfg.arrays[image].shape):
                        dealias.reduce_connector(node.sdfg, conn, sdfg.arrays[array], offset=zero, squeeze=[0])
            for n in state.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == image:
                    n.data = array
        if repl:
            sdfg.replace_dict(repl)
        for sym in repl:
            if sym in sdfg.symbols:
                sdfg.remove_symbol(sym)
        sdfg.remove_data(image, validate=False)
        if frozen is not None:
            written = any(a.sdfg_name == image and a.is_written for a in frozen.args)
            gone = {image, *repl}  # the image and its extent and offset symbols
            args = tuple(replace(a, is_written=True) if a.sdfg_name == array and written else a
                         for a in frozen.args if a.sdfg_name not in gone)
            frozen = replace(frozen, args=args, free_symbols=tuple(s for s in frozen.free_symbols if s not in gone))
        done.append(f"{image}[{elem}] -> {array}")
    if frozen is not None:
        attach_to_sdfg(sdfg, frozen)
    return done
