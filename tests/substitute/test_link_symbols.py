"""Wrap symbols match real compiler output; generated shims keep their caller's ABI."""

import shutil
import subprocess

import pytest

from dace_fortran.substitute.link_symbols import procedure_symbol


def test_nvidia_default_is_unchanged():
    assert procedure_symbol("Kernel", "Step") == "kernel_step_"


def test_gnu_symbol_matches_compiled_module(tmp_path):
    if not shutil.which("gfortran"):
        pytest.skip("gfortran unavailable")
    source = tmp_path / "kernel.f90"
    source.write_text("module Kernel\ncontains\nsubroutine Step\nend subroutine\nend module\n")
    obj = tmp_path / "kernel.o"
    subprocess.run(["gfortran", "-c", str(source), "-o", str(obj)], cwd=tmp_path, check=True)
    symbols = subprocess.run(["nm", "--defined-only", str(obj)], capture_output=True, text=True, check=True).stdout
    assert procedure_symbol("Kernel", "Step", "gfortran") in symbols.split()


def test_unknown_compiler_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        procedure_symbol("kernel", "step", "unknown")
