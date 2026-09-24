# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fortran declarations for a compiled library's ABI, from its ABI manifest.

A DaCe-generated library is called from Fortran through a ``bind(C)`` interface block, and every
argument of that block needs a Fortran type.  The manifest (``dace.codegen.manifest``) already says,
per argument, what it IS -- ``kind``, ``dtype``, ``rank``, ``intent`` -- so writing the block is a
pure function of settled data.  This module is that function.

WHY IT LIVES HERE.  The consumer this was written for, a Fortran port of a CFD solver, had SIX
independent copies of this emission across its shim generators, sharing only two ad-hoc helpers, and
none of them read the manifest for a scalar's type: they keyed on the argument's NAME
(``rsz1``/``rsz2``/``swdir``, and the six loop-bound names ``jb``/``je``/``kb``/``ke``/``lb``/``le``)
with a silent ``real(c_double)`` fallback.  A NEW integer scalar reaching one of them would have
been declared as a double -- a wrong declaration that compiles.  The fix is not to merge six
functions; it is to read the fact that is already recorded.

THE TYPE VOCABULARY, AND THE TRAP IN IT.  DaCe renders dtypes with ``str()``, which does NOT match
the ``dace.dtypes`` member name: ``float64`` renders ``"double"``, ``float32`` -> ``"float"``,
``int32`` -> ``"int"``, ``int64`` -> ``"int64_t"``.  MEASURED over a real 33-library port, the whole
vocabulary is the five below, plus the literal ``"int64"`` that ``describe`` writes for an EXTENT.
This matters because dace-fortran's three other dtype maps (``bindings/fortran_interface``,
``bindings/loop_copy``, ``bindings/block_builders``) are keyed on the MEMBER names -- so none of them
can type a manifest argument at all, and they disagree with each other on ``int64``
(``c_int64_t`` / ``c_long`` / ``c_long_long``).  This map is keyed on what the manifest actually
contains.

NOTHING IS GUESSED.  An argument the manifest cannot type raises :class:`UntypedArgument` rather
than falling back to a default.  The consuming port learned this the expensive way: its generators
had FOUR different behaviours for an unknown type -- a silent ``real(c_double)``, a hard exit, a
comment in the emitted text, and a refusal -- and only the refusal is correct, because the other
three produce a compilable interface block that is wrong.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

__all__ = [
    "ARRAY_TYPE",
    "EXTENT_TYPE",
    "BY_VALUE_TYPE",
    "UntypedArgument",
    "UnsupportedKind",
    "by_name",
    "by_value_type",
    "declaration",
    "declarations",
]

#: An argument that is a buffer crosses as an opaque device address, whatever its element type --
#: the shim hands the kernel a device pointer and the pointee is not the interface's business.
ARRAY_TYPE = "type(c_ptr)"

#: A runtime extent.  ``describe`` writes the literal ``"int64"`` for these; the C side is
#: ``int64_t``.
EXTENT_TYPE = "integer(c_int64_t)"

#: DaCe's ``str(dtype)`` -> the Fortran type a PASS-BY-VALUE dummy takes.  See the module docstring
#: for why these keys are not the ``dace.dtypes`` member names.
BY_VALUE_TYPE: Dict[str, str] = {
    "double": "real(c_double)",
    "float": "real(c_float)",
    "int": "integer(c_int)",
    "int64_t": "integer(c_int64_t)",
    "int64": "integer(c_int64_t)",
    "bool": "logical(c_bool)",
}


class UntypedArgument(ValueError):
    """The manifest does not say what this argument is, and a declaration cannot be guessed.

    Raised for ``dtype="unknown"``.  Most often this means the manifest is OLDER than the fix that
    recorded a SYMBOL's dtype -- re-bake the library.  It is a refusal and not a default on purpose:
    a guessed declaration compiles, links, and passes the value wrong.
    """


class UnsupportedKind(ValueError):
    """The manifest names a ``kind`` this emitter does not know."""


def by_value_type(dtype: str, *, name: str = "", library: str = "") -> str:
    """The Fortran type for a pass-by-value dummy of ``dtype``.

    :raises UntypedArgument: for ``"unknown"`` -- see the class docstring.
    :raises UnsupportedKind: for a dtype outside the manifest's vocabulary, because a new one is a
        fact to look up, not to default.
    """
    where = f" for `{name}`" if name else ""
    if library:
        where += f" in {library}"
    if dtype == "unknown":
        raise UntypedArgument(
            f"the ABI records no dtype{where}.  If this is a SYMBOL, the manifest predates the fix "
            f"that types them -- re-bake the library; otherwise the manifest is incomplete.")
    try:
        return BY_VALUE_TYPE[dtype]
    except KeyError:
        raise UnsupportedKind(
            f"unsupported dtype {dtype!r}{where}.  DaCe renders dtypes with `str()`, so these are "
            f"NOT the `dace.dtypes` member names (`float64` renders 'double', `int32` renders "
            f"'int').  Known: {sorted(BY_VALUE_TYPE)}.  Add it deliberately -- a wrong declaration "
            f"compiles.") from None


def declaration(arg: Dict[str, Any], *, indent: str = "      ", library: str = "") -> str:
    """One argument's Fortran dummy declaration.  See :func:`declarations`."""
    kind = arg.get("kind")
    name = arg["name"]
    if kind == "array":
        return f"{indent}{ARRAY_TYPE}, value :: {name}"
    if kind == "extent":
        return f"{indent}{EXTENT_TYPE}, value :: {name}"
    if kind in ("scalar", "symbol"):
        # Both cross by value.  A `symbol` is a free symbol that reached the C ABI as a plain
        # scalar; treating it as anything else is how a loop bound once got declared a double.
        ftype = by_value_type(arg.get("dtype", "unknown"), name=name, library=library)
        return f"{indent}{ftype}, value :: {name}"
    raise UnsupportedKind(
        f"unsupported kind {kind!r} for `{name}`{f' in {library}' if library else ''}.  Known: "
        f"array, extent, scalar, symbol.")


def declarations(abi: Dict[str, Any], *, indent: str = "      ") -> List[str]:
    """The dummy declarations for every argument of ``abi``, in the manifest's own order.

    :param abi: a loaded manifest -- ``dace.codegen.manifest.load(...)``.
    :returns: one Fortran source line per argument, ready to join with newlines.

    The order comes from ``abi["args"]``, which ``describe`` builds by enumerating the SAME flat
    arglist it records as ``abi["order"]``, so positional correspondence with the C function is
    preserved.  That is the one thing a ``bind(C)`` block cannot get wrong quietly.
    """
    library = abi.get("name", "")
    args: Sequence[Dict[str, Any]] = abi.get("args", [])
    return [declaration(a, indent=indent, library=library) for a in args]


def by_name(abi: Dict[str, Any], *, indent: str = "      ") -> Dict[str, str]:
    """``{argument name: its declaration line}`` -- for a caller that emits them one at a time.

    The consumers this was written for build their interface blocks in an f-string, interpolating a
    declaration per name in the order they choose; a name-keyed view lets them keep that shape
    instead of restructuring around a list.
    """
    library = abi.get("name", "")
    return {a["name"]: declaration(a, indent=indent, library=library) for a in abi.get("args", [])}
