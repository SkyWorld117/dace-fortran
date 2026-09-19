# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fortran sequential-unformatted framing: read, write, and refuse a broken file.

(Distinct from ``tests/fortran_io_test.py``, which covers the SDFG LIBRARY NODES of
``dace_fortran.libraries.fortran_io`` -- file I/O inside a generated SDFG.  This is the Python side.)
"""
import struct

import numpy as np
import pytest

from dace_fortran import unformatted as uf


def test_round_trip(tmp_path):
    p = tmp_path / "r.dat"
    vals = np.arange(12, dtype=np.float64) * 0.5
    uf.write_record(p, vals)
    assert np.array_equal(uf.read_record(p), vals)


def test_the_framing_is_int32_len_payload_int32_len(tmp_path):
    """What a Fortran `write(fid) values` produces.  Asserting it is the point of the module: a
    reader that does not skip the leading marker returns an array shifted by one element."""
    p = tmp_path / "r.dat"
    uf.write_record(p, [1.0, 2.0])
    raw = p.read_bytes()
    assert struct.unpack('<i', raw[:4])[0] == 16          # 2 doubles
    assert struct.unpack('<i', raw[-4:])[0] == 16         # and the same again
    assert len(raw) == 4 + 16 + 4


def test_an_element_count_marker_is_scaled(tmp_path):
    """Some writers emit the length as an ELEMENT count rather than a byte count.  Reading it as
    bytes gives a payload four or eight times too long -- the classic silent shift."""
    p = tmp_path / "r.dat"
    payload = np.arange(4, dtype=np.float64).tobytes()
    p.write_bytes(struct.pack('<i', 4) + payload + struct.pack('<i', 4))   # 4 ELEMENTS, not 32 bytes
    assert np.array_equal(uf.read_record(p), np.arange(4, dtype=np.float64))


def test_mismatched_markers_are_refused(tmp_path):
    """The trailing marker is how a reader knows it did not lose alignment.  Ignoring it is how a
    truncated or non-Fortran file comes back as plausible-looking garbage."""
    p = tmp_path / "r.dat"
    payload = np.arange(4, dtype=np.float64).tobytes()
    p.write_bytes(struct.pack('<i', len(payload)) + payload + struct.pack('<i', len(payload) + 8))
    with pytest.raises(uf.UnformattedError, match="markers disagree"):
        uf.read_record(p)


def test_a_truncated_record_is_refused(tmp_path):
    p = tmp_path / "r.dat"
    p.write_bytes(struct.pack('<i', 4096) + b"\x00" * 16)
    with pytest.raises(uf.UnformattedError, match="exceeds the file"):
        uf.read_record(p)


def test_extra_trailing_data_says_to_use_read_records(tmp_path):
    p = tmp_path / "r.dat"
    uf.write_record(p, [1.0])
    p.write_bytes(p.read_bytes() * 2)
    with pytest.raises(uf.UnformattedError, match="read_records"):
        uf.read_record(p)


def test_a_multi_record_file_reads_in_order(tmp_path):
    p = tmp_path / "r.dat"
    buf = b""
    for v in ([1.0, 2.0], [3.0], [4.0, 5.0, 6.0]):
        payload = np.asarray(v, dtype=np.float64).tobytes()
        buf += struct.pack('<i', len(payload)) + payload + struct.pack('<i', len(payload))
    p.write_bytes(buf)
    assert [r.tolist() for r in uf.read_records(p)] == [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]


def test_read_shaped_checks_the_size(tmp_path):
    p = tmp_path / "r.dat"
    uf.write_record(p, np.arange(6, dtype=np.float64))
    assert uf.read_shaped(p, (2, 3)).shape == (2, 3)
    with pytest.raises(uf.UnformattedError, match="expected"):
        uf.read_shaped(p, (4, 4))


def test_a_naive_reader_would_be_wrong(tmp_path):
    """The failure this module exists to prevent, shown directly: reading the file with a plain
    fromfile() shifts every value by one element, because the leading marker is read as data.  That
    is not a hypothetical -- it is how this project spent five rounds believing its baseline was
    corrupt."""
    p = tmp_path / "r.dat"
    uf.write_record(p, [1.0, 2.0, 3.0])
    naive = np.fromfile(p, dtype=np.float64)
    assert naive.size == 4                      # 3 values + the marker, read as data
    assert not np.array_equal(naive, [1.0, 2.0, 3.0])
    assert np.array_equal(uf.read_record(p), [1.0, 2.0, 3.0])
