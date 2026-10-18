"""Fold the image of a member of an element of a module array of records onto the array it IS.

A program may point ``rec(1)%f`` at the module array ``arr`` and hand ``rec`` to other routines: the SDFG takes the
image ``rec_f`` and ``arr`` as two arrays, and the binding's storage guard declines every call (the two share
storage). The sizing run records the association (``<record>.aliases``, see
``dace_fortran.bindings.storage_guard.build_alias_record``): here every access of the image becomes an access of the
array (its leading element dimension of one dropped), in the nested SDFGs too, the image's extent and offset symbols
become the array's, and the image leaves the SDFG and its frozen signature.

A member may also be the member of an element of another array of records (``p(1)%f => c(1)%f``): every access of
that element of the one image becomes an access of the element of the other (:func:`fold_member_aliases`).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import dace
import networkx as nx
from dace import nodes, subsets, symbolic
from dace.properties import CodeBlock
from dace.sdfg import dealias
from dace.sdfg.state import AbstractControlFlowRegion
from dace_fortran.array_split import _UNKNOWN, _keep_order, _subset_of
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


def _element_range(subset) -> tuple[int, int] | None:
    """The literal first and last element of a subset's leading (element) dimension, or None."""
    if subset is None or subset is _UNKNOWN:
        return None
    begin, end, step = subset.ranges[0]
    try:
        first, last = int(symbolic.simplify(begin)), int(symbolic.simplify(end))
    except (TypeError, ValueError):
        return None
    return (first, last) if step == 1 or first == last else None


def _same_extent(sdfg: dace.SDFG, a: str, b: str, known: dict) -> bool:
    """Whether the arrays ``a`` and ``b`` have the same dtype and, past their element dimension, the same extents and
    lower bounds (``offset_<array>_d<k>`` symbols, with the values ``known`` holds)."""
    da, db = sdfg.arrays[a], sdfg.arrays[b]
    if da.dtype != db.dtype or len(da.shape) != len(db.shape):
        return False
    subs = {symbolic.pystr_to_symbolic(k): v for k, v in known.items()}

    def offset(name: str):  # an image without the symbol starts at 0
        return name if name in sdfg.symbols or name in known else 0

    for k in range(1, len(da.shape)):
        pairs = [(da.shape[k], db.shape[k]), (offset(f"offset_{a}_d{k}"), offset(f"offset_{b}_d{k}"))]
        for x, y in pairs:
            diff = symbolic.pystr_to_symbolic(str(x)).subs(subs) - symbolic.pystr_to_symbolic(str(y)).subs(subs)
            if symbolic.simplify(diff) != 0:
                return False
    return True


def _moved_nodes(sdfg: dace.SDFG, image: str, moves: dict[int, tuple[str, int]]):
    """Per state, the access nodes of ``image`` with an edge at a moved element; None when an access may reach a
    moved element without picking one literal element (a range over elements, a nested SDFG's whole image), or a
    condition or loop header reads the image, or an interstate assignment reads it other than at a literal element
    (one at a moved element :func:`_rewrite_code` rewrites)."""
    found = []
    for state in sdfg.states():
        for edge in state.edges():
            if edge.data.is_empty():
                continue
            subset = _subset_of(edge, state, image)
            if subset is None:
                continue
            rng = _element_range(subset)
            if rng is None or (rng[0] != rng[1] and any(rng[0] <= e <= rng[1] for e in moves)):
                return None
            if edge.data.data != image and edge.data.other_subset is None:
                return None  # a copy whose subset in the image is left implicit
        nodes_ = [n for n in state.data_nodes() if n.data == image and any(
            not e.data.is_empty() and _element_range(_subset_of(e, state, image)) in {(k, k) for k in moves}
            for e in [*state.in_edges(n), *state.out_edges(n)])]
        found += [(state, n) for n in nodes_]
    for edge in sdfg.all_interstate_edges():
        for m in edge.data.get_read_memlets(sdfg.arrays):
            rng = _element_range(m.subset) if m.data == image else ()
            if rng is None or (rng and rng[0] != rng[1]):
                return None
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, AbstractControlFlowRegion) and any(m.data == image for m in block.get_meta_read_memlets()):
            return None
    return found


def _renamed_bounds(sdfg: dace.SDFG, image: str, other: str, known: dict) -> dict[str, str]:
    """The image's extent and offset symbols -> the other image's (of equal value, see :func:`_same_extent`): its
    symbol where the SDFG has it, else the known value; a moved subset then reads the other image's bounds."""
    out = {}
    for k in range(1, len(sdfg.arrays[image].shape)):
        for mine, theirs in ((f"{image}_d{k}", f"{other}_d{k}"), (f"offset_{image}_d{k}", f"offset_{other}_d{k}")):
            if theirs in sdfg.symbols:
                out[mine] = theirs
            elif theirs in known:
                out[mine] = str(known[theirs])
    return out


def _rewrite_code(code: str, image: str, moves: dict[int, tuple[str, int]], rename: dict[str, dict]) -> str:
    """``code`` (an interstate assignment or condition) with every subscript of ``image`` at a moved element made one
    of the other image's element, its bounds symbols those of the other image (``rename``: per other image, symbol ->
    expression)."""
    import ast  # noqa: PLC0415

    class Moved(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if not (isinstance(node.value, ast.Name) and node.value.id == image):
                return node
            index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            try:
                element = int(symbolic.pystr_to_symbolic(ast.unparse(index[0])))
            except (TypeError, ValueError):
                return node
            if element not in moves:
                return node
            other, other_element = moves[element]
            subs = {str(k): str(v) for k, v in rename[other].items()}

            class Bounds(ast.NodeTransformer):
                def visit_Name(self, name):
                    return ast.parse(subs[name.id], mode="eval").body if name.id in subs else name

            rest = [Bounds().visit(i) for i in index[1:]]
            node.value = ast.Name(id=other, ctx=ast.Load())
            node.slice = ast.Tuple(elts=[ast.Constant(other_element), *rest], ctx=ast.Load())
            return node

    return ast.unparse(Moved().visit(ast.parse(code)))


def _ordered(state: dace.SDFGState, a: nodes.AccessNode, b: nodes.AccessNode) -> bool:
    return nx.has_path(state.nx, a, b) or nx.has_path(state.nx, b, a)


def _written(state: dace.SDFGState, node: nodes.AccessNode) -> bool:
    return any(not e.data.is_empty() for e in state.in_edges(node))


def _fold_one(sdfg: dace.SDFG, image: str, moves: dict[int, tuple[str, int]], known: dict) -> bool:
    """Fold the elements ``moves`` of ``image`` onto the other images' (:func:`fold_member_aliases`); False, and
    ``sdfg`` unchanged, when they cannot be."""
    found = _moved_nodes(sdfg, image, moves)
    if not found:
        return False
    targets = {o for o, _ in moves.values()}
    landing = set(moves.values())

    def lands(state: dace.SDFGState, q: nodes.AccessNode) -> bool:
        """Whether the node of an other image accesses an element a move lands on (or may: a range it cannot tell)."""
        for e in [*state.in_edges(q), *state.out_edges(q)]:
            rng = None if e.data.is_empty() else _element_range(_subset_of(e, state, q.data))
            if not e.data.is_empty() and (rng is None or any(
                    o == q.data and rng[0] <= oe <= rng[1] for o, oe in landing)):
                return True
        return False

    if any((_written(st, n) or _written(st, q)) and not _ordered(st, n, q) and lands(st, q)
           for st, n in found for q in st.data_nodes() if q.data in targets):
        return False
    rename = {o: {symbolic.pystr_to_symbolic(k): symbolic.pystr_to_symbolic(v)
                  for k, v in _renamed_bounds(sdfg, image, o, known).items()} for o in targets}

    def moved(rng: subsets.Range, elem: int) -> tuple[str, subsets.Range]:
        other, other_elem = moves[elem]
        out = subsets.Range([(other_elem, other_elem, 1)] + list(rng.ranges[1:]))
        out.replace(rename[other])
        return other, out

    for state, node in found:
        parts = {}
        for edge in [*state.in_edges(node), *state.out_edges(node)]:
            if edge.data.is_empty():
                continue
            rng = _element_range(_subset_of(edge, state, image))
            if rng is None or rng[0] not in moves:
                continue
            memlet = edge.data
            if memlet.data == image:
                other, memlet.subset = moved(memlet.subset, rng[0])
                memlet.data = other
            else:
                other, memlet.other_subset = moved(memlet.other_subset, rng[0])
            if other not in parts:
                parts[other] = state.add_access(other)
            if edge.dst is node:
                state.add_edge(edge.src, edge.src_conn, parts[other], edge.dst_conn, memlet)
            else:
                state.add_edge(parts[other], edge.src_conn, edge.dst, edge.dst_conn, memlet)
            state.remove_edge(edge)
        # What ordered the node orders its parts; the node stays for the elements that stay.
        ordering = [e for e in [*state.in_edges(node), *state.out_edges(node)] if e.data.is_empty()]
        for e in ordering:
            for part in parts.values():
                if e.dst is node:
                    state.add_edge(e.src, e.src_conn, part, None, dace.Memlet())
                else:
                    state.add_edge(part, None, e.dst, e.dst_conn, dace.Memlet())
        if all(e.data.is_empty() for e in [*state.in_edges(node), *state.out_edges(node)]):
            state.remove_node(node)
            _keep_order(state, list(parts.values()))
        else:
            _keep_order(state, [node, *parts.values()])
    # Interstate assignments and conditions that read the image at a moved element.
    for edge in sdfg.all_interstate_edges():
        if not any(m.data == image for m in edge.data.get_read_memlets(sdfg.arrays)):
            continue
        edge.data.assignments = {k: _rewrite_code(v, image, moves, rename) for k, v in edge.data.assignments.items()}
        edge.data.condition = CodeBlock(_rewrite_code(edge.data.condition.as_string, image, moves, rename))
    # Memlets inside scopes that name the image at a moved element (a tasklet's edge to a map exit, say).
    for state in sdfg.states():
        for edge in state.edges():
            memlet = edge.data
            rng = _element_range(memlet.subset) if memlet.data == image else None
            if rng is not None and rng[0] == rng[1] and rng[0] in moves:
                memlet.data, memlet.subset = moved(memlet.subset, rng[0])
    return True


def fold_member_aliases(sdfg: dace.SDFG, aliases: list[tuple[str, int, str, int]], known: dict | None = None
                        ) -> list[tuple[str, int, str, int]]:
    """Fold every ``(image, element, other, other element)`` alias -- the member of ``element`` of the records whose
    SDFG image is ``image`` is that of ``other element`` of ``other`` (``p(1)%f => c(1)%f``) -- that can be: every
    access of the element becomes one of the other image's element, so the SDFG takes the storage as the program
    does. ``known``: values of the extent and offset symbols, to tell that the two images' elements have the same
    shape. The images keep their other elements, in the SDFG and its frozen signature. Returns the aliases folded.

    The elements of an image fold only all together, when every access of the image picks its element by a literal
    (as once the element loops are unrolled), and the image's own and the other image's accesses in a state that one
    of them writes are ordered there: the two were different containers, and nothing ordered a read of the one after
    a write of the other."""
    known = known or {}
    pairs = [(i, e, o, oe) for i, e, o, oe in aliases
             if i != o and i in sdfg.arrays and o in sdfg.arrays and _same_extent(sdfg, i, o, known)]
    folded = []
    # One image's elements onto the other's, or the other way when an access of the one stands in the way.
    for a, b in sorted({(i, o) for i, _e, o, _oe in pairs}):
        forward = {e: (o, oe) for i, e, o, oe in pairs if (i, o) == (a, b)}
        backward = {oe: (i, e) for i, e, o, oe in pairs if (i, o) == (a, b)}
        if _fold_one(sdfg, a, forward, known):
            folded += [(a, e, o, oe) for e, (o, oe) in sorted(forward.items())]
        elif _fold_one(sdfg, b, backward, known):
            folded += [(b, e, o, oe) for e, (o, oe) in sorted(backward.items())]
    return folded
