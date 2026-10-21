"""A bridge build respects its allocation even on a much larger machine."""

import os

import pytest

from dace_fortran.llvm_toolchain import build_parallelism


@pytest.fixture(autouse=True)
def allocation(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(128)), raising=False)
    monkeypatch.delenv("CMAKE_BUILD_PARALLEL_LEVEL", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)


def test_default_is_bounded():
    assert build_parallelism() == 32


def test_allocation_and_requested_jobs_both_limit_the_build(monkeypatch):
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "24")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert build_parallelism() == 8
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1})
    assert build_parallelism() == 2


@pytest.mark.parametrize("name", ["CMAKE_BUILD_PARALLEL_LEVEL", "SLURM_CPUS_PER_TASK"])
@pytest.mark.parametrize("value", ["", "0", "-1", "all"])
def test_invalid_limits_fail_before_building(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        build_parallelism()
