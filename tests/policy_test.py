# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The policy loader: the project-specific half of TU generation, as validated data."""
import json

import pytest

from dace_fortran import policy


GOOD = {
    "access": "%sf",
    "extents": {"0": "ext_k", "1": "ext_j"},
    "kernels": {
        "rk_stage": {
            "fpp": "simulation/m_time_steppers.fpp",
            "routine": "s_tvd_rk",
            "anchor": "q_cons_ts(stor)%vf(i)%sf",
            "rules": [[r"q_cons_ts\((\w+)\)%vf\((\w+)\)%sf\(([^()]*)\)", r"q_ts(\1,\2,\3)"]],
        }
    },
}


def write(tmp_path, obj):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(obj))
    return p


def test_a_kernel_without_a_source_file_is_refused_at_load(tmp_path):
    """The likeliest authoring mistake, and the one whose failure is least legible three stages
    downstream -- so it fails at LOAD, not at emit."""
    bad = {"kernels": {"k": {"anchor": "x"}}}
    with pytest.raises(policy.PolicyError, match="names no 'fpp' source file"):
        policy.load(write(tmp_path, bad))


def test_a_bad_regex_is_named_rather_than_compiled_later(tmp_path):
    bad = {"kernels": {"k": {"fpp": "a.fpp", "rules": [["[unclosed", "x"]]}}}
    with pytest.raises(policy.PolicyError, match="not a valid regex"):
        policy.load(write(tmp_path, bad))


def test_a_malformed_rule_is_refused(tmp_path):
    bad = {"kernels": {"k": {"fpp": "a.fpp", "rules": ["not-a-pair"]}}}
    with pytest.raises(policy.PolicyError, match="must be \\[pattern, replacement\\]"):
        policy.load(write(tmp_path, bad))


def test_a_missing_kernel_says_what_is_there(tmp_path):
    p = policy.load(write(tmp_path, GOOD))
    with pytest.raises(policy.PolicyError, match="it has"):
        p.kernel("nope")


def test_extent_keys_are_storage_dimensions(tmp_path):
    bad = {"extents": {"first": "ext_k"}, "kernels": {}}
    with pytest.raises(policy.PolicyError, match="not a storage dimension"):
        policy.load(write(tmp_path, bad))


def test_round_trip_is_exact(tmp_path):
    p = policy.load(write(tmp_path, GOOD))
    out = tmp_path / "again.json"
    policy.dump(p, out)
    assert json.loads(out.read_text()) == GOOD


def test_a_loaded_policy_exposes_compiled_rules(tmp_path):
    k = policy.load(write(tmp_path, GOOD)).kernel("rk_stage")
    assert k.fpp.endswith("m_time_steppers.fpp")
    assert k.routine == "s_tvd_rk"
    pat, repl = k.flatten[0]
    assert pat.sub(repl, "q_cons_ts(1)%vf(3)%sf(j, k, l)") == "q_ts(1,3,j, k, l)"
