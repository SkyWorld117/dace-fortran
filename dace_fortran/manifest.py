# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deprecated path: the ABI manifest now lives in DaCe core.

It was written here, for the Fortran bindings that needed it first, and then moved --
because nothing in it is Fortran.  It reads ``SDFG.arglist()``, ``SDFG.arrays`` and the access
nodes, so it describes ANY compiled SDFG, and in core every backend gets it rather than only this
one.

This module stays as a re-export so existing bindings keep working; new code should import
``dace.codegen.manifest``.
"""
from dace.codegen.manifest import (  # noqa: F401
    EXTENT,
    describe,
    load,
    write,
)

__all__ = ["EXTENT", "describe", "load", "write"]
