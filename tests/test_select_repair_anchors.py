"""Tests for scripts/select_repair_anchors.py."""

from __future__ import annotations

from scripts.select_repair_anchors import build_anchor_map, build_cohort, select_anchor
from src.uncertainty import DEFAULT_ERROR_COSTS


def _problem(task_id: str, passed: list[bool]) -> dict:
    return {
        "task_id": task_id,
        "samples": [{"code": f"code_{i}", "passed": p} for i, p in enumerate(passed)],
    }


def test_build_cohort_excludes_problems_with_any_passing_sample():
    source = {
        "per_problem": [
            _problem("T/0", [False, False, False]),  # sample-0 fails, no sibling passes
            _problem("T/1", [False, True, False]),  # sample-0 fails, a sibling passes
            _problem("T/2", [True, False, False]),  # sample-0 itself passes
        ]
    }

    cohort, excluded = build_cohort(source)

    assert cohort == ["T/0"]
    assert excluded == ["T/1"]


def test_select_anchor_argmin_self_distance_excludes_all_crash_samples():
    # Sample 0 crashes on every probe (all-crash); sample 1 is the true
    # consensus pick; sample 2 disagrees with everything.
    signatures = [
        ["error:unrunnable", "error:unrunnable"],
        ["1", "2"],
        ["1", "9"],
    ]

    anchor = select_anchor(signatures, DEFAULT_ERROR_COSTS)

    assert anchor["sample_index"] == 1
    assert anchor["all_crash"] is False
    assert anchor["fallback_used"] is False
    assert anchor["candidate_pool_size"] == 2  # sample 0 excluded from the pool
    assert anchor["support"] == 2  # both probes ran normally on sample 1


def test_select_anchor_falls_back_to_the_full_pool_when_every_sample_crashes():
    signatures = [
        ["error:unrunnable", "error:unrunnable"],
        ["error:timeout", "error:timeout"],
    ]

    anchor = select_anchor(signatures, DEFAULT_ERROR_COSTS)

    assert anchor["fallback_used"] is True
    assert anchor["candidate_pool_size"] == 2
    assert anchor["all_crash"] is True
    assert anchor["support"] == 0


def test_build_anchor_map_skips_problems_missing_from_the_signature_cache():
    cache = {"T/0": {"signatures": [["1", "2"], ["1", "9"]]}}

    anchors, n_fallback = build_anchor_map(cache, ["T/0", "T/1"], DEFAULT_ERROR_COSTS)

    assert set(anchors) == {"T/0"}
    assert n_fallback == 0
