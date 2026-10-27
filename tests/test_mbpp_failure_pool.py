"""Tests for the Student 2 Week 6 MBPP failure-pool builder."""

from __future__ import annotations

from scripts.build_mbpp_failure_pool import refresh_pool_metadata


def _problem(task_id: str, passed: bool, category: str = "") -> dict:
    return {
        "task_id": task_id,
        "samples": [
            {
                "passed": passed,
                "code": task_id,
                "error_message": category,
                "failure_category": category,
            }
        ],
    }


def test_refresh_pool_metadata_keeps_evaluation_representative_and_splits_disjoint():
    payload = {
        "config": {
            "target_evaluation_failures": 2,
            "target_development_failures": 2,
        },
        "per_problem": [
            _problem("MBPP/1", True),
            _problem("MBPP/2", False, "assertion_error"),
            _problem("MBPP/3", False, "syntax_error"),
            _problem("MBPP/4", False, "name_error"),
            _problem("MBPP/5", False, "assertion_error"),
        ],
    }

    refresh_pool_metadata(payload)

    evaluation = payload["failure_splits"]["evaluation"]
    development = payload["failure_splits"]["development"]
    assert [item["task_id"] for item in evaluation] == ["MBPP/2", "MBPP/3"]
    assert {item["task_id"] for item in development} == {"MBPP/4", "MBPP/5"}
    assert {(item["task_id"], item["sample_index"]) for item in evaluation}.isdisjoint(
        {(item["task_id"], item["sample_index"]) for item in development}
    )
    assert payload["summary"] == {
        "problems_processed": 5,
        "initially_passing": 1,
        "initially_failing": 4,
        "initial_pass_rate": 0.2,
        "initial_failure_categories": {
            "assertion_error": 2,
            "name_error": 1,
            "syntax_error": 1,
        },
        "development_failures": 2,
        "evaluation_failures": 2,
        "target_reached": True,
    }


def test_refresh_pool_metadata_reports_incomplete_targets():
    payload = {
        "config": {
            "target_evaluation_failures": 2,
            "target_development_failures": 1,
        },
        "per_problem": [_problem("MBPP/1", False, "type_error")],
    }

    refresh_pool_metadata(payload)

    assert payload["summary"]["target_reached"] is False
    assert payload["summary"]["evaluation_failures"] == 1
    assert payload["summary"]["development_failures"] == 0
