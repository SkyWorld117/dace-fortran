"""Fortran dummy declarations from an ABI manifest.

The tests that matter here are the REFUSALS and the SYMBOL.  Everything else is a lookup table, and
a lookup table is not where this went wrong for the consumer it was written for: their generators
keyed a scalar's type on the argument's NAME, with a silent ``real(c_double)`` fallback, so a new
integer scalar would have been declared a double -- a wrong declaration that compiles, links, and
passes the value wrong.  ``test_a_new_int_scalar_is_typed_not_defaulted`` is that case, and it is
the reason the fallback does not exist.
"""
import pytest

from dace_fortran.abi_interface import (ARRAY_TYPE, EXTENT_TYPE, UntypedArgument,
                                        UnsupportedKind, by_value_type, declaration,
                                        declarations)


def _arg(name, kind, dtype=None, **kw):
    a = {"index": 0, "name": name, "kind": kind}
    if dtype is not None:
        a["dtype"] = dtype
    a.update(kw)
    return a


def test_array_is_an_opaque_device_address():
    """Whatever the element type: the shim hands the kernel a device pointer."""
    assert declaration(_arg("q", "array", "double")) == f"      {ARRAY_TYPE}, value :: q"
    assert declaration(_arg("flags", "array", "bool")) == f"      {ARRAY_TYPE}, value :: flags"


def test_extent_is_int64():
    assert declaration(_arg("q_d0", "extent", "int64")) == f"      {EXTENT_TYPE}, value :: q_d0"


def test_scalars_use_the_manifest_dtype():
    assert declaration(_arg("bb", "scalar", "int")) == "      integer(c_int), value :: bb"
    assert declaration(_arg("dt", "scalar", "double")) == "      real(c_double), value :: dt"
    assert declaration(_arg("on", "scalar", "bool")) == "      logical(c_bool), value :: on"


def test_a_symbol_is_typed_from_the_manifest():
    """A free symbol reaches the C ABI as a plain scalar; declaring it by name was the workaround.

    The manifest recorded ``dtype="unknown"`` for every symbol until that was fixed, which is why a
    consumer had to hardcode the six loop-bound names its kernels use.
    """
    assert declaration(_arg("jb", "symbol", "int")) == "      integer(c_int), value :: jb"
    assert declaration(_arg("off", "symbol", "int64_t")) == "      integer(c_int64_t), value :: off"


def test_a_new_int_scalar_is_typed_not_defaulted():
    """The case the name lists could not cover, and the reason there is no fallback.

    A consumer's generators knew `rsz1`/`rsz2` were integers because they were in a list of names.
    Any OTHER integer scalar fell to `real(c_double)`.  Here the manifest answers instead.
    """
    assert by_value_type("int", name="something_new") == "integer(c_int)"
    # and the declaration is what the C side expects, not a double
    assert declaration(_arg("something_new", "scalar", "int")).endswith(
        "integer(c_int), value :: something_new")


def test_dtype_vocabulary_is_daces_rendering_not_the_member_names():
    """`str(dace.float64)` is "double", NOT "float64" -- so the member names would all be rejected.

    `int64` is the exception that proves the point rather than breaking it: it IS in the vocabulary,
    as the literal `describe` writes for an EXTENT, which is a different thing from `int64_t`.
    """
    assert by_value_type("double") == "real(c_double)"
    assert by_value_type("int") == "integer(c_int)"
    assert by_value_type("int64_t") == "integer(c_int64_t)"
    for member_name in ("float64", "float32", "int32"):
        with pytest.raises(UnsupportedKind):
            by_value_type(member_name)


def test_unknown_is_refused_not_guessed():
    with pytest.raises(UntypedArgument) as e:
        by_value_type("unknown", name="jb", library="mfc_dace_conv")
    assert "jb" in str(e.value) and "mfc_dace_conv" in str(e.value)


def test_declarations_keep_the_manifest_order():
    """Positional correspondence with the C function is the one thing a bind(C) block cannot get
    wrong quietly, so the order comes from the manifest and is not sorted or grouped."""
    abi = {
        "name": "lib",
        "args": [
            _arg("q", "array", "double"),
            _arg("jb", "symbol", "int"),
            _arg("q_d0", "extent", "int64"),
            _arg("dt", "scalar", "double"),
        ],
    }
    got = declarations(abi)
    assert [l.split("::")[1].strip() for l in got] == ["q", "jb", "q_d0", "dt"]
    assert [l.split(",")[0].replace("      ", "") for l in got] == [
        "type(c_ptr)", "integer(c_int)", "integer(c_int64_t)", "real(c_double)"]


def test_an_unknown_argument_surfaces_the_library_name():
    abi = {"name": "mfc_dace_x", "args": [_arg("mystery", "scalar", "unknown")]}
    with pytest.raises(UntypedArgument) as e:
        declarations(abi)
    assert "mfc_dace_x" in str(e.value) and "mystery" in str(e.value)


def test_an_unknown_kind_is_refused():
    with pytest.raises(UnsupportedKind):
        declaration(_arg("odd", "pointer-ish", "double"))
