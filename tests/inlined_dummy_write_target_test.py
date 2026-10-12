"""A statement that writes through an inlined dummy must write the actual argument of ITS call site.

After ``hlfir-inline-all`` every inlined copy of a callee's dummy keeps the callee's ``uniq_name``. A dummy bound to
the whole of a caller variable is that variable and is named after it, so the copies of one dummy name different
variables -- one per call site -- although they share one ``uniq_name``. A target named through the shared
``uniq_name`` instead of the copy's own declare lands every copy on one call site's actual: the other actuals never see
the write, and the one that does sees writes meant for the others.

Each test calls one callee at several sites with different actuals. ``whole``: dummies of the entry, a local of the
entry, and a local of a callee that is itself inlined twice (so its copies are distinct locals too). ``passed_on``:
contiguous assumed-shape dummies of an inlined callee that passes them on (bound to a whole array at one call site and
to a section at another, so a dummy bound to the whole of a variable shares its name with one bound to a section), and
sections passed directly.
"""

from pathlib import Path

import numpy as np
import pytest

from tests._helpers import xfail
from tests._util import build_sdfg

_NF = 3

_WHOLE = """
  subroutine via_local(out, src)
    double precision, intent(out) :: out(nf)
    double precision, intent(in) :: src(nf)
    double precision :: tmp(nf)
    integer :: i
    do i = 1, nf
      tmp(i) = src(i) + i
    end do
    call op(tmp{extra})
    do i = 1, nf
      out(i) = tmp(i)
    end do
  end subroutine via_local

  subroutine driver(a, b, c, d, e)
    double precision, intent(inout) :: a(nf), b(nf)
    double precision, intent(out) :: c(nf), d(nf), e(nf)
    double precision :: w(nf)
    integer :: i
    do i = 1, nf
      w(i) = a(i) * b(i)
    end do
    call op(a{extra})
    call op(b{extra})
    call op(w{extra})
    do i = 1, nf
      c(i) = w(i)
    end do
    call via_local(d, a)
    call via_local(e, b)
  end subroutine driver
"""

_PASSED_ON = """
  subroutine via_dummy(x)
    double precision, contiguous, intent(inout) :: x(:)
    call op(x{extra})
  end subroutine via_dummy

  subroutine driver(f, g)
    double precision, intent(inout) :: f(nf), g(nf, 3)
    call via_dummy(f)
    call via_dummy(g(:, 2))
    call op(g(:, 1){extra})
    call op(g(:, 3){extra})
  end subroutine driver
"""

# How the callee declares its array dummy ``v``; ``extra`` is what each call passes after the array.
_SHAPES = {
    "explicit_constant": ("", "double precision, intent(inout) :: v(nf)", ""),
    "explicit_dummy_extent": (", n", "integer, intent(in) :: n\n    double precision, intent(inout) :: v(n)", ", nf"),
    "assumed": ("", "double precision, intent(inout) :: v(:)", ""),
}

# Statement kinds writing through ``v`` (or reading it in a whole-array expression), with a NumPy reference.
_BODIES = {
    "normalise": (
        """
    total = 0d0
    do i = 1, size(v)
      total = total + v(i)
    end do
    v = v / max(total, 1d-16)""",
        lambda v: v / max(v.sum(), 1e-16),
    ),
    "broadcast_scalar": (
        """
    total = 0d0
    do i = 1, size(v)
      total = total + v(i) * v(i)
    end do
    v = total""",
        lambda v: np.full_like(v, (v * v).sum()),
    ),
    "copy_through_local": (
        """
    tmp = v
    do i = 1, nf
      tmp(i) = tmp(i) * i
    end do
    v = tmp""",
        lambda v: v * np.arange(1, _NF + 1),
    ),
    "section": (
        """
    v(2:3) = v(2:3) * v(1)""",
        lambda v: np.concatenate([v[:1], v[1:] * v[0]]),
    ),
    "reduction_read": (
        """
    v = v / sum(v) - maxval(v)""",
        lambda v: v / v.sum() - v.max(),
    ),
    "intrinsic": (
        """
    v = sqrt(abs(v)) + 1d0""",
        lambda v: np.sqrt(np.abs(v)) + 1.0,
    ),
}

# Whole-array copies, broadcasts and reductions through a dummy bound to a section are not lowered at all (they fail
# to build whatever the call sites), so ``passed_on`` covers the statement kinds that are.
_PASSED_ON_BODIES = ("intrinsic", "normalise", "section")


def _source(actuals: str, shape: str, body: str) -> str:
    sig, decl, extra = _SHAPES[shape]
    callee = f"""
  subroutine op(v{sig})
    {decl}
    double precision :: total, tmp(nf)
    integer :: i
    {_BODIES[body][0]}
  end subroutine op
"""
    driver = {"whole": _WHOLE, "passed_on": _PASSED_ON}[actuals]
    return (
        f"module m_inlined_targets\n  implicit none\n  integer, parameter :: nf = {_NF}\ncontains\n"
        + callee
        + driver.format(extra=extra)
        + "end module m_inlined_targets\n"
    )


def _inputs(actuals: str) -> dict:
    if actuals == "whole":
        return {
            "a": np.array([0.5, 1.5, 3.0]),
            "b": np.array([2.0, 0.25, 4.0]),
            "c": np.zeros(_NF),
            "d": np.zeros(_NF),
            "e": np.zeros(_NF),
        }
    return {
        "f": np.array([0.5, 1.5, 3.0]),
        "g": np.array([[2.0, 1.0, 0.75], [0.25, 2.0, 3.0], [4.0, 0.5, 1.25]], order="F"),
    }


def _reference(actuals: str, body: str, inputs: dict) -> dict:
    op = _BODIES[body][1]
    if actuals == "whole":
        a0, b0 = inputs["a"], inputs["b"]
        shift = np.arange(1, _NF + 1, dtype=np.float64)
        a, b = op(a0.copy()), op(b0.copy())
        return {"a": a, "b": b, "c": op(a0 * b0), "d": op(a + shift), "e": op(b + shift)}
    g = inputs["g"].copy()
    for j in range(3):
        g[:, j] = op(g[:, j].copy())
    return {"f": op(inputs["f"].copy()), "g": g}


def _check(tmp_path: Path, actuals: str, shape: str, body: str) -> None:
    src = _source(actuals, shape, body)
    sdfg = build_sdfg(
        src, tmp_path / "sdfg", name=f"inl_target_{actuals}_{body}_{shape}", entry="m_inlined_targets::driver"
    )
    args = _inputs(actuals)
    want = _reference(actuals, body, args)
    sdfg.build()(**args)
    for name, value in want.items():
        np.testing.assert_allclose(args[name], value, rtol=1e-13, atol=1e-13, err_msg=f"{name} ({body}, {shape})")


_WHOLE_CASES = [
    pytest.param(
        body,
        shape,
        marks=[xfail("a scalar variable assigned to a whole assumed-shape dummy is written to its first element only")]
        if (body, shape) == ("broadcast_scalar", "assumed")
        else [],
    )
    for body in sorted(_BODIES)
    for shape in sorted(_SHAPES)
]


@pytest.mark.parametrize(("body", "shape"), _WHOLE_CASES)
def test_write_through_inlined_dummy_reaches_its_own_actual(tmp_path: Path, body: str, shape: str):
    _check(tmp_path, "whole", shape, body)


@pytest.mark.parametrize("shape", sorted(_SHAPES))
@pytest.mark.parametrize("body", _PASSED_ON_BODIES)
def test_write_through_passed_on_dummy_reaches_its_own_actual(tmp_path: Path, body: str, shape: str):
    _check(tmp_path, "passed_on", shape, body)


_SCALAR_SRC = """
module m_inlined_scalar
  implicit none
contains
  subroutine bump(s, t)
    double precision, intent(inout) :: s
    double precision, intent(out) :: t
    s = s * 2d0 + 1d0
    t = s - 0.5d0
  end subroutine bump

  subroutine via_local(out, x)
    double precision, intent(out) :: out
    double precision, intent(in) :: x
    double precision :: loc, aux
    loc = x + 3d0
    call bump(loc, aux)
    out = loc + aux
  end subroutine via_local

  subroutine driver(p, q, r, u, h)
    double precision, intent(inout) :: p, q, h(3)
    double precision, intent(out) :: r, u
    double precision :: w, t1, t2, t3, t4
    w = -3d0
    call bump(p, t1)
    call bump(h(2), t4)
    call bump(q, t2)
    call bump(w, t3)
    r = w + t1 + t2 + t3 + t4
    call via_local(u, p)
    u = u * 10d0
    call via_local(w, q)
    u = u + w
  end subroutine driver
end module m_inlined_scalar
"""


def test_scalar_writes_through_inlined_dummies_reach_their_own_actuals(tmp_path: Path):
    """The scalar twin: an inlined scalar dummy is the caller's variable too, written by an assignment; one call site
    binds it to an array element, a declare of another kind with the same name."""
    sdfg = build_sdfg(_SCALAR_SRC, tmp_path / "sdfg", name="inl_target_scalar", entry="m_inlined_scalar::driver")
    p, q = np.array([1.5]), np.array([-2.0])
    r, u = np.zeros(1), np.zeros(1)
    h = np.array([0.25, 0.5, 0.75])
    sdfg.build()(p=p, q=q, r=r, u=u, h=h)

    def bump(s):
        s = s * 2.0 + 1.0
        return s, s - 0.5

    def via_local(x):
        loc, aux = bump(x + 3.0)
        return loc + aux

    p1, t1 = bump(1.5)
    q1, t2 = bump(-2.0)
    w, t3 = bump(-3.0)
    h1, t4 = bump(0.5)
    want_u = via_local(p1) * 10.0 + via_local(q1)
    np.testing.assert_allclose(
        [p[0], q[0], r[0], u[0], *h], [p1, q1, w + t1 + t2 + t3 + t4, want_u, 0.25, h1, 0.75], rtol=1e-14
    )
