# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read and write Fortran sequential-unformatted files.

NOT to be confused with ``dace_fortran.libraries.fortran_io``, which is something else entirely: SDFG
LIBRARY NODES (``Read``, ``Write``, ``NamelistRead``) that perform file I/O *at runtime, inside a
generated SDFG*.  This module is the Python side -- reading a file that a Fortran program wrote, from
a test, a validator or a comparison script.  The two are complementary, and they were nearly given
the same name; hence the explicit one here.

WHY THIS IS IN A LIBRARY AND NOT IN EACH PROJECT.  A Fortran program writing

    write(fid) values

produces a file framed as ``int32 length, payload, int32 length`` -- the trailing marker being the
same length, which is how a reader checks it did not lose alignment.  It is Fortran's format, not any
one project's, and it is what a numerical code's restart files, dumps and test references all look
like.

Every one of those is a place a hand-rolled reader gets subtly wrong, and the failure is SILENT: a
reader that does not skip the leading marker returns an array shifted by one element, whose values
look like garbage -- or worse, look plausible.  In the codebase this was written for that happened
five separate times during one investigation, each time costing a round of debugging that ended in
"the baseline is corrupt" ... and each time the baseline was fine and the READING was wrong.

So: one implementation, with the framing asserted rather than assumed.

``length`` may be either a BYTE count (the standard) or an ELEMENT count, depending on the compiler
and the record: both are seen in the wild, and they are told apart by testing whether the declared
length is a multiple of the item size.  That heuristic is the one genuinely non-obvious part, and it
lives here so it is written once.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np

MARKER = np.int32


class UnformattedError(ValueError):
    """The file is not a Fortran unformatted record, or its markers disagree."""


def _payload_len(marker: int, itemsize: int, path) -> int:
    """The record's payload length in BYTES, from its leading marker.

    A marker that is not a multiple of the item size is an ELEMENT count -- some writers emit that --
    so it is scaled.  Getting this wrong is the classic silent failure: the payload is then read at
    the wrong length and every value after it is shifted.
    """
    if marker < 0:
        raise UnformattedError(f"{path}: negative record length {marker}")
    if itemsize > 1 and marker % itemsize:
        return marker * itemsize
    return marker


def read_record(path: Union[str, Path], dtype=np.float64) -> np.ndarray:
    """The single record in a Fortran unformatted file, as an array of ``dtype``.

    Asserts the trailing marker matches the leading one: a mismatch means the byte offsets are wrong,
    and continuing past it would return a plausible-looking array of the wrong data.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise UnformattedError(f"{path}: {len(raw)} bytes is smaller than a record frame")
    itemsize = np.dtype(dtype).itemsize
    head = int(np.frombuffer(raw[0:4], dtype=MARKER)[0])
    n = _payload_len(head, itemsize, path)
    if 4 + n + 4 > len(raw):
        raise UnformattedError(
            f"{path}: record of {n} bytes + framing exceeds the file's {len(raw)} bytes")
    values = np.frombuffer(raw[4:4 + n], dtype=dtype).copy()
    tail = int(np.frombuffer(raw[4 + n:8 + n], dtype=MARKER)[0])
    if tail != head:
        raise UnformattedError(
            f"{path}: record markers disagree (head={head}, tail={tail}); the file is either not "
            f"unformatted Fortran or is truncated")
    if 8 + n != len(raw):
        raise UnformattedError(
            f"{path}: {len(raw) - (8 + n)} trailing byte(s) after the record -- this reader handles "
            f"exactly one record; use read_records() for a multi-record file")
    return values


def read_records(path: Union[str, Path], dtype=np.float64) -> Iterator[np.ndarray]:
    """Every record in the file, in order -- for a file written with several ``write`` statements."""
    path = Path(path)
    raw = path.read_bytes()
    itemsize = np.dtype(dtype).itemsize
    off = 0
    while off < len(raw):
        if off + 4 > len(raw):
            raise UnformattedError(f"{path}: truncated marker at byte {off}")
        head = int(np.frombuffer(raw[off:off + 4], dtype=MARKER)[0])
        n = _payload_len(head, itemsize, path)
        if off + 8 + n > len(raw):
            raise UnformattedError(f"{path}: record at byte {off} runs past the end of the file")
        yield np.frombuffer(raw[off + 4:off + 4 + n], dtype=dtype).copy()
        tail = int(np.frombuffer(raw[off + 4 + n:off + 8 + n], dtype=MARKER)[0])
        if tail != head:
            raise UnformattedError(f"{path}: record at byte {off} has head={head} tail={tail}")
        off += 8 + n


def write_record(path: Union[str, Path], values: Sequence[float], dtype=np.float64) -> None:
    """Write one record in the same framing, so a reference a test writes can be read back by
    :func:`read_record` (and by any Fortran program)."""
    a = np.asarray(values, dtype=dtype)
    payload = a.tobytes()
    n = struct.pack('<i', len(payload))
    Path(path).write_bytes(n + payload + n)


def read_shaped(path: Union[str, Path], shape, dtype=np.float64) -> np.ndarray:
    """A record reshaped to ``shape`` -- a convenience for the common case where the writer emitted
    a multi-dimensional array and the shape is known from the case."""
    a = read_record(path, dtype)
    want = int(np.prod(shape))
    if a.size != want:
        raise UnformattedError(f"{path}: record holds {a.size} value(s), expected {want} for {shape}")
    return a.reshape(shape)
