"""E2e numerical correctness for the AoS-of-pointer-records lift.

Each probe under ``tests/aos_pointer_records/`` builds an SDFG via the bridge
and an f2py reference from the same source, runs both with seeded random
inputs, and asserts OUTPUT arrays match elementwise. Regression gate for
``hlfir-lift-aos-pointer-records``.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

import dace_fortran

_HERE = Path(__file__).resolve().parent


def _f2py(src: Path, out_dir: Path, mod_name: str, *, kind_map: dict | None = None):
    """Compile ``src`` via ``numpy.f2py`` and import the resulting module.

    ``kind_map`` writes a ``.f2py_f2cmap`` for symbolic kind aliases (``wp``, ``JPRB``) f2py can't resolve itself.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind_map:
        # f2py reads .f2py_f2cmap from cwd; maps Fortran TypeName -> {kind_alias: ctype}.
        as_dict = {"real": dict(kind_map.items())}
        (out_dir / ".f2py_f2cmap").write_text(repr(as_dict))
    subprocess.check_call([sys.executable, "-m", "numpy.f2py", "-c", str(src), "-m", mod_name, "--quiet"], cwd=out_dir)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def _sdfg(src_text: str, out_dir: Path, entry: str, name: str):
    sdfg = dace_fortran.build_sdfg(src_text, out_dir=str(out_dir), entry=entry, name=name)
    sdfg.validate()
    return sdfg


def test_single_pointer_member_numerical(tmp_path):
    """``q(c)%x => target_c; out = q(1)%x + q(2)%x`` matches gfortran."""
    src_path = _HERE / "aos_single_pointer_member_probe.f90"
    src_text = src_path.read_text()

    mod = _f2py(src_path, tmp_path / "ref", "single_ref")
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    rng = np.random.default_rng(0)
    n, k = 8, 6
    qa = np.asfortranarray(rng.standard_normal((n, k)))
    qb = np.asfortranarray(rng.standard_normal((n, k)))
    qsum_sdfg = np.zeros((n, k), order="F")

    qsum_ref = mod.m.run(qa.copy(order="F"), qb.copy(order="F"))
    sdfg(n=np.int32(n), k=np.int32(k), qa=qa.copy(order="F"), qb=qb.copy(order="F"), qsum=qsum_sdfg)
    np.testing.assert_allclose(qsum_sdfg, qsum_ref, rtol=1e-12, atol=1e-12)


def test_write_through_pointer_numerical(tmp_path):
    """Writes through ``q(c)%x`` propagate back to the targets after lift."""
    src_path = _HERE / "aos_write_through_pointer_probe.f90"
    src_text = src_path.read_text()
    mod = _f2py(src_path, tmp_path / "ref", "write_ref")
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    rng = np.random.default_rng(1)
    n, k = 5, 7
    qa = np.asfortranarray(rng.standard_normal((n, k)))
    qb = np.asfortranarray(rng.standard_normal((n, k)))
    qa_ref, qb_ref = qa.copy(order="F"), qb.copy(order="F")
    qa_sdfg, qb_sdfg = qa.copy(order="F"), qb.copy(order="F")

    mod.m.run(qa_ref, qb_ref)
    sdfg(n=np.int32(n), k=np.int32(k), qa=qa_sdfg, qb=qb_sdfg)
    np.testing.assert_allclose(qa_sdfg, qa_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qb_sdfg, qb_ref, rtol=1e-12, atol=1e-12)


def test_target_and_alias_stay_coherent_numerical(tmp_path):
    """A pointer target read or written directly while ``q(c)%x`` points at it agrees with the alias.

    Covers a write through the alias read through the target, the reverse, and a whole-array write to the target
    read through the alias.  The reference is spelled out in numpy, since the values the source must produce are
    simple and the check does not depend on a Fortran build.
    """
    src_text = (_HERE / "aos_alias_coherence_probe.f90").read_text()
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    rng = np.random.default_rng(4)
    n, k = 6, 5
    qv0 = np.asfortranarray(rng.standard_normal((n, k)))
    qc0 = np.asfortranarray(rng.standard_normal((n, k)))
    qv, qc = qv0.copy(order="F"), qc0.copy(order="F")
    t, u = np.zeros((n, k), order="F"), np.zeros((n, k), order="F")

    sdfg(n=np.int32(n), k=np.int32(k), qv=qv, qc=qc, t=t, u=u)

    np.testing.assert_allclose(t, 2.0 * (qv0 + 1.0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qc, qc0 + 5.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qv, 3.0 * (qv0 + 1.0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u, (qc0 + 5.0) + 3.0 * (qv0 + 1.0), rtol=1e-12, atol=1e-12)


def test_assumed_shape_target_numerical(tmp_path):
    """Assumed-shape pointer targets (``q(iqx)%x => t(:,:)``): gather temp's
    extents come from ``fir.box_dims``.  Regression: unbound extents used to
    default to 1 at call time, under-allocating and overflowing the heap."""
    src_path = _HERE / "aos_assumed_shape_target_probe.f90"
    src_text = src_path.read_text()
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    # Pure runtime-indexed select; reference is just the ix-th input (f2py
    # can't wrap the assumed-shape dummies this probe needs for fir.box_dims).
    rng = np.random.default_rng(3)
    n, k = 7, 5
    targets = [np.asfortranarray(rng.standard_normal((n, k))) for _ in range(4)]
    for ix in (1, 2, 3, 4):
        out_sdfg = np.zeros((n, k), order="F")
        sdfg(
            ix=np.int32(ix),
            qa=targets[0].copy(order="F"),
            qb=targets[1].copy(order="F"),
            qc=targets[2].copy(order="F"),
            qd=targets[3].copy(order="F"),
            out=out_sdfg,
        )
        np.testing.assert_allclose(out_sdfg, targets[ix - 1], rtol=1e-12, atol=1e-12, err_msg=f"mismatch for ix={ix}")


def test_inlined_callee_reads_through_its_dummy_numerical(tmp_path):
    """The lifted array is read in an inlined callee through the callee's own dummy (a re-declare of it)."""
    src_text = (_HERE / "aos_inlined_callee_probe.f90").read_text()
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    # out = qa - qb; f2py cannot wrap the derived-type dummy of ``kernel``.
    rng = np.random.default_rng(15)
    n, k = 5, 4
    qa0 = np.asfortranarray(rng.standard_normal((n, k)))
    qb0 = np.asfortranarray(rng.standard_normal((n, k)))
    qa, qb = qa0.copy(order="F"), qb0.copy(order="F")
    out = np.zeros((n, k), order="F")
    sdfg(n=np.int32(n), k=np.int32(k), qa=qa, qb=qb, out=out)
    np.testing.assert_allclose(out, qa0 - qb0, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(qa, qa0)
    np.testing.assert_array_equal(qb, qb0)


def test_wp_kind_alias_numerical(tmp_path):
    """``REAL(KIND=wp)`` compiles + runs without pre-resolving ``wp`` -- exercises ``normalize_kind_parameters``."""
    src_path = _HERE / "aos_wp_kind_probe.f90"
    src_text = src_path.read_text()
    # f2py can't evaluate SELECTED_REAL_KIND; tell it wp=double to match the
    # bridge's normalize_kind_parameters default (wp -> 8).
    mod = _f2py(src_path, tmp_path / "ref", "wp_ref", kind_map={"wp": "double"})
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    rng = np.random.default_rng(2)
    n, k = 6, 4
    qa = np.asfortranarray(rng.standard_normal((n, k)))
    qb = np.asfortranarray(rng.standard_normal((n, k)))
    qsum_sdfg = np.zeros((n, k), order="F")

    qsum_ref = mod.m.run(qa.copy(order="F"), qb.copy(order="F"))
    sdfg(n=np.int32(n), k=np.int32(k), qa=qa.copy(order="F"), qb=qb.copy(order="F"), qsum=qsum_sdfg)
    np.testing.assert_allclose(qsum_sdfg, qsum_ref, rtol=1e-12, atol=1e-12)


def test_lower_bound_target_numerical(tmp_path):
    """Targets ``qa(0:n, -2:k)``: the members keep the targets' lower bounds, reads and the write-back alike."""
    src_path = _HERE / "aos_lower_bound_target_probe.f90"
    src_text = src_path.read_text()
    mod = _f2py(src_path, tmp_path / "ref", "lower_bound_ref")
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    rng = np.random.default_rng(5)
    n, k = 6, 4  # arrays are (0:n, -2:k), i.e. (n + 1) x (k + 3)
    qa = np.asfortranarray(rng.standard_normal((n + 1, k + 3)))
    qb = np.asfortranarray(rng.standard_normal((n + 1, k + 3)))
    qa_ref, qb_ref = qa.copy(order="F"), qb.copy(order="F")
    qa_sdfg, qb_sdfg = qa.copy(order="F"), qb.copy(order="F")
    out_sdfg = np.zeros((n + 1, k + 3), order="F")

    out_ref = mod.m.run(qa_ref, qb_ref, n=n, k=k)
    sdfg(n=np.int32(n), k=np.int32(k), qa=qa_sdfg, qb=qb_sdfg, out=out_sdfg)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qa_sdfg, qa_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qb_sdfg, qb_ref, rtol=1e-12, atol=1e-12)


def test_assumed_shape_lower_bound_target_numerical(tmp_path):
    """Assumed-shape targets ``qa(0:, -2:)``: extents from the descriptor, lower bounds from the declares.

    The loops run over ``LBOUND`` / ``UBOUND`` / ``SIZE`` of the runtime-selected component, so they read the gather
    temp's slice rebased to the component's lower bounds.
    """
    src_text = (_HERE / "aos_assumed_shape_lower_bound_target_probe.f90").read_text()
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    # A runtime-indexed select of the ix-th input; f2py cannot wrap the assumed-shape dummies.
    rng = np.random.default_rng(6)
    n, k = 5, 6
    targets = [np.asfortranarray(rng.standard_normal((n, k))) for _ in range(2)]
    for ix in (1, 2):
        qa, qb = targets[0].copy(order="F"), targets[1].copy(order="F")
        out_sdfg = np.zeros((n, k), order="F")
        sdfg(ix=np.int32(ix), qa=qa, qb=qb, out=out_sdfg)
        np.testing.assert_allclose(out_sdfg, targets[ix - 1], rtol=1e-12, atol=1e-12, err_msg=f"mismatch for ix={ix}")
        # The copy back must not move the targets' elements.
        np.testing.assert_array_equal(qa, targets[0])
        np.testing.assert_array_equal(qb, targets[1])


def test_runtime_lower_bound_target_numerical(tmp_path):
    """Targets ``qa(m:, m:)`` whose lower bound is a runtime dummy shared by both: one gather temp starting at ``m``."""
    src_text = (_HERE / "aos_runtime_lower_bound_target_probe.f90").read_text()
    sdfg = _sdfg(src_text, tmp_path / "sdfg", "m::run", "run")

    # out(i, j) = q(ix)%x(m + i - 1, m + j - 1) is the ix-th input itself; f2py cannot wrap the assumed-shape dummies.
    rng = np.random.default_rng(8)
    n, k = 4, 6
    targets = [np.asfortranarray(rng.standard_normal((n, k))) for _ in range(2)]
    for ix in (1, 2):
        qa, qb = targets[0].copy(order="F"), targets[1].copy(order="F")
        out_sdfg = np.zeros((n, k), order="F")
        sdfg(m=np.int32(3), ix=np.int32(ix), qa=qa, qb=qb, out=out_sdfg)
        np.testing.assert_allclose(out_sdfg, targets[ix - 1], rtol=1e-12, atol=1e-12, err_msg=f"mismatch for ix={ix}")
        np.testing.assert_array_equal(qa, targets[0])
        np.testing.assert_array_equal(qb, targets[1])


def _bounds_probe(entry: str, tmp_path: Path):
    """The gfortran build and the SDFG of one entry of ``aos_pointer_bounds_probe.f90``."""
    src_path = _HERE / "aos_pointer_bounds_probe.f90"
    mod = _f2py(src_path, tmp_path / "ref", f"bounds_{entry}_ref")
    return getattr(mod.m, entry), _sdfg(src_path.read_text(), tmp_path / "sdfg", f"m::{entry}", entry)


def _check_bounds_entry(entry: str, tmp_path: Path, seed: int, **scalars: int):
    """Run one two-target entry of ``aos_pointer_bounds_probe.f90`` against its gfortran build: output and targets."""
    ref, sdfg = _bounds_probe(entry, tmp_path)
    rng = np.random.default_rng(seed)
    n, k = 5, 3  # arrays are (0:n, -2:k), i.e. (n + 1) x (k + 3)
    qa = np.asfortranarray(rng.standard_normal((n + 1, k + 3)))
    qb = np.asfortranarray(rng.standard_normal((n + 1, k + 3)))
    qa_ref, qb_ref = qa.copy(order="F"), qb.copy(order="F")
    qa_sdfg, qb_sdfg = qa.copy(order="F"), qb.copy(order="F")
    out_sdfg = np.zeros((n + 1, k + 3), order="F")

    out_ref = ref(qa=qa_ref, qb=qb_ref, n=n, k=k, **scalars)
    sdfg(
        n=np.int32(n),
        k=np.int32(k),
        qa=qa_sdfg,
        qb=qb_sdfg,
        out=out_sdfg,
        **{s: np.int32(v) for s, v in scalars.items()},
    )
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qa_sdfg, qa_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qb_sdfg, qb_ref, rtol=1e-12, atol=1e-12)


def test_whole_section_target_numerical(tmp_path):
    """``q(c)%x => qa(:, :)`` restarts the member at 1 although ``qa`` starts at (0, -2)."""
    _check_bounds_entry("section", tmp_path, seed=9)


def test_remapped_lower_bounds_numerical(tmp_path):
    """``q(c)%x(5:, 7:) => qa`` gives the member its own lower bounds, neither 1 nor the target's (0, -2); the loops
    run over ``LBOUND`` / ``UBOUND`` of the member."""
    _check_bounds_entry("remap", tmp_path, seed=10)


def test_runtime_remapped_lower_bounds_numerical(tmp_path):
    """``q(c)%x(m:, m:) => qa`` for both rebinds: a runtime lower bound the rebinds provably share."""
    _check_bounds_entry("remap_runtime", tmp_path, seed=12, m=-4)


def test_rebind_in_nested_block_numerical(tmp_path):
    """Rebinds with lower bounds inside an ``IF``: copy-in and copy-out stay inside the block, taken or not."""
    ref, sdfg = _bounds_probe("nested", tmp_path)
    rng = np.random.default_rng(13)
    n, k = 3, 4
    qa = np.asfortranarray(rng.standard_normal((n + 1, k + 3)))
    qb = np.asfortranarray(rng.standard_normal((n + 1, k + 3)))
    for flag in (1, 0):
        qa_ref, qb_ref, out_ref = qa.copy(order="F"), qb.copy(order="F"), np.zeros((n + 1, k + 3), order="F")
        qa_sdfg, qb_sdfg, out_sdfg = qa.copy(order="F"), qb.copy(order="F"), np.zeros((n + 1, k + 3), order="F")
        ref(flag=flag, qa=qa_ref, qb=qb_ref, out=out_ref, n=n, k=k)
        sdfg(n=np.int32(n), k=np.int32(k), flag=np.int32(flag), qa=qa_sdfg, qb=qb_sdfg, out=out_sdfg)
        np.testing.assert_allclose(out_sdfg, out_ref, rtol=1e-12, atol=1e-12, err_msg=f"flag={flag}")
        np.testing.assert_allclose(qa_sdfg, qa_ref, rtol=1e-12, atol=1e-12, err_msg=f"flag={flag}")
        np.testing.assert_allclose(qb_sdfg, qb_ref, rtol=1e-12, atol=1e-12, err_msg=f"flag={flag}")


def test_static_local_target_with_lower_bounds_numerical(tmp_path):
    """A static local target ``loc(0:3, -2:2)``: a fully static gather temp declared with a ``fir.shape_shift``."""
    ref, sdfg = _bounds_probe("local_target", tmp_path)
    out_sdfg = np.zeros((4, 5), order="F")
    sdfg(out=out_sdfg)
    np.testing.assert_allclose(out_sdfg, ref(), rtol=1e-12, atol=1e-12)


def test_target_window_with_lower_bounds_numerical(tmp_path):
    """Direct accesses to a target ``qv(0:n, -2:k)`` while ``q(1)%x`` points at it, with the target's lower bounds.

    Element accesses are redirected to the gather temp, and the whole-array assignment to the target is bracketed by
    copies.
    """
    ref, sdfg = _bounds_probe("window", tmp_path)
    rng = np.random.default_rng(11)
    n, k = 4, 5
    qv = np.asfortranarray(rng.standard_normal((n + 1, k + 3)))
    u = np.asfortranarray(rng.standard_normal((n + 1, k + 3)))
    qv_ref, qv_sdfg = qv.copy(order="F"), qv.copy(order="F")
    t_sdfg = np.zeros((n + 1, k + 3), order="F")

    t_ref = ref(qv=qv_ref, u=u, n=n, k=k)
    sdfg(n=np.int32(n), k=np.int32(k), qv=qv_sdfg, u=u.copy(order="F"), t=t_sdfg)
    np.testing.assert_allclose(t_sdfg, t_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qv_sdfg, qv_ref, rtol=1e-12, atol=1e-12)


def test_target_window_through_inlined_callee_numerical(tmp_path):
    """An inlined callee reads the target through its own explicit-shape dummy ``a(n, k)`` while ``q(1)%x`` points
    at it: the dummy's re-declare (its extents loaded through the callee's own ``n``, ``k``) is taken to keep the
    target's layout, so its reads go to the gather temp."""
    ref, sdfg = _bounds_probe("callee_window", tmp_path)
    rng = np.random.default_rng(14)
    n, k = 4, 2
    qv = np.asfortranarray(rng.standard_normal((n, k)))
    qv_ref, qv_sdfg = qv.copy(order="F"), qv.copy(order="F")
    t_sdfg = np.zeros((n, k), order="F")

    t_ref = ref(qv=qv_ref, n=n, k=k)
    sdfg(n=np.int32(n), k=np.int32(k), qv=qv_sdfg, t=t_sdfg)
    np.testing.assert_allclose(t_sdfg, t_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(qv_sdfg, qv_ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    import tempfile

    for test in (
        test_single_pointer_member_numerical,
        test_write_through_pointer_numerical,
        test_target_and_alias_stay_coherent_numerical,
        test_assumed_shape_target_numerical,
        test_inlined_callee_reads_through_its_dummy_numerical,
        test_wp_kind_alias_numerical,
        test_lower_bound_target_numerical,
        test_assumed_shape_lower_bound_target_numerical,
        test_runtime_lower_bound_target_numerical,
        test_whole_section_target_numerical,
        test_remapped_lower_bounds_numerical,
        test_runtime_remapped_lower_bounds_numerical,
        test_rebind_in_nested_block_numerical,
        test_static_local_target_with_lower_bounds_numerical,
        test_target_window_with_lower_bounds_numerical,
        test_target_window_through_inlined_callee_numerical,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
