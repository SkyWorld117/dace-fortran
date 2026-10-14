"""Buffers of members of arrays of records laid out with the member dimensions fastest.

The binding packs the members ``rec(i) % vf(j) % sf(x, y, z)`` of an array of records into one buffer the SDFG reads
as ``buf[i, j, x, y, z]``, a Fortran array of that shape: the record indices run fastest, so the threads of a kernel
at neighbouring points ``x`` read memory ``size(rec) * size(vf)`` elements apart, and a kernel reading two of the
members fetches every one of them.  :func:`lay_out_members_first` gives such a buffer the strides of the Fortran
array ``buf(x, y, z, i, j)`` -- the member dimensions fastest, the record dimensions after them, each group in
column-major order -- in the SDFG (its argument descriptor and every nested descriptor connected to it), and
``emit_bindings(members_first=...)`` has the binding allocate, pack, measure and unpack the buffer in that order
(:func:`dace_fortran.bindings.block_builders.buffer_dim`).  The SDFG's subscripts, symbols and lower bounds are
untouched: only the strides change."""

from __future__ import annotations

from collections.abc import Mapping

import dace
from dace import data, nodes

from dace_fortran.bindings.block_builders import _aos_lead, keepable
from dace_fortran.bindings.frozen_signature import FrozenSignature


def members_first_order(lead: int, rank: int) -> list[int]:
    """The dimensions of a buffer with ``lead`` record dimensions, fastest first: the members', then the records'."""
    return [*range(lead, rank), *range(lead)]


def members_first_strides(shape, lead: int) -> list:
    """Strides of ``shape`` with its ``lead`` leading dimensions slowest, each group in column-major order."""
    strides = [None] * len(shape)
    stride = 1
    for d in members_first_order(lead, len(shape)):
        strides[d] = stride
        stride = stride * shape[d]
    return strides


def record_buffers(frozen: FrozenSignature) -> dict[str, int]:
    """The buffers of deferred-shape members of module arrays of records among ``frozen``'s arguments, with the number
    of their record dimensions: the buffers :func:`lay_out_members_first` lays out."""
    return {a.sdfg_name: _aos_lead(a) for a in frozen.args if keepable(a)}


def lay_out_members_first(sdfg: dace.SDFG, buffers: Mapping[str, int]) -> list[str]:
    """Give the argument arrays ``buffers`` (name -> number of record dimensions) of ``sdfg`` the strides that put the
    member dimensions first, in every descriptor of theirs (nested SDFGs included).  A buffer some nested descriptor
    of which has another shape keeps its layout (a note is printed).  Returns the names laid out, sorted; the binding
    must be emitted with ``members_first`` naming exactly these."""
    done = []
    for name in sorted(buffers):
        desc = sdfg.arrays.get(name)
        if not isinstance(desc, data.Array) or isinstance(desc, data.View) or desc.transient:
            continue
        lead = buffers[name]
        if not 0 < lead < len(desc.shape):
            continue
        shape = tuple(str(s) for s in desc.shape)
        descriptors: list[tuple[dace.SDFG, str]] = []
        odd: list[tuple] = []
        _collect(sdfg, name, shape, descriptors, odd)
        if odd:
            print(f"record image {name} keeps its layout: nested descriptors of another shape: {odd[:3]}", flush=True)
            continue
        for container, array in descriptors:
            d = container.arrays[array]
            d.strides = members_first_strides(list(d.shape), lead)
        done.append(name)
    return done


def _collect(container: dace.SDFG, array: str, shape: tuple[str, ...], out: list, odd: list) -> None:
    """``array`` of ``container`` and every descriptor it reaches through nested-SDFG connectors; one of another shape
    (or a view) goes to ``odd``."""
    out.append((container, array))
    for st in container.all_states():
        for n in st.nodes():
            if not isinstance(n, nodes.NestedSDFG):
                continue
            for e in [*st.in_edges(n), *st.out_edges(n)]:
                conn = e.dst_conn if e.dst is n else e.src_conn
                if e.data.data != array or not conn or conn not in n.sdfg.arrays:
                    continue
                if any(c is n.sdfg and a == conn for c, a in out):
                    continue
                inner = n.sdfg.arrays[conn]
                if isinstance(inner, data.View) or tuple(str(s) for s in inner.shape) != shape:
                    odd.append((n.label, conn, tuple(str(s) for s in inner.shape)))
                    continue
                _collect(n.sdfg, conn, shape, out, odd)
