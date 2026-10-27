"""Unit tests for the Week 4 replay experiment."""

import pytest

from scripts.run_iterative_repair_experiment import (
    _validate_resume_payload,
    _wilson_interval,
    collect_failures,
    select_diverse_failures,
    select_failure_split,
    select_source_order_failures,
    summarize,
)


def test_collect_and_select_failures_prefers_category_and_task_diversity():
    source = {
        "per_problem": [
            {
                "task_id": "HumanEval/0",
                "samples": [
                    {
                        "passed": False,
                        "code": "a",
                        "error_message": "e",
                        "failure_category": "assertion_error",
                    },
                    {
                        "passed": False,
                        "code": "b",
                        "error_message": "e",
                        "failure_category": "syntax_error",
                    },
                ],
            },
            {
                "task_id": "HumanEval/1",
                "samples": [
                    {
                        "passed": False,
                        "code": "c",
                        "error_message": "e",
                        "failure_category": "assertion_error",
                    }
                ],
            },
        ]
    }

    selected = select_diverse_failures(collect_failures(source), 2)

    assert {item["initial_failure_category"] for item in selected} == {
        "assertion_error",
        "syntax_error",
    }


def test_collect_failures_can_filter_one_source_sample_index():
    source = {
        "per_problem": [
            {
                "task_id": "MBPP/1",
                "samples": [
                    {
                        "passed": False,
                        "code": "first",
                        "error_message": "e0",
                        "failure_category": "assertion_error",
                    },
                    {
                        "passed": False,
                        "code": "second",
                        "error_message": "e1",
                        "failure_category": "syntax_error",
                    },
                ],
            },
            {
                "task_id": "MBPP/2",
                "samples": [
                    {
                        "passed": True,
                        "code": "passing",
                        "error_message": "",
                        "failure_category": "",
                    },
                    {
                        "passed": False,
                        "code": "failing",
                        "error_message": "e2",
                        "failure_category": "type_error",
                    },
                ],
            },
        ]
    }

    failures = collect_failures(source, sample_index=0)

    assert [(item["task_id"], item["sample_index"]) for item in failures] == [
        ("MBPP/1", 0)
    ]
    with pytest.raises(ValueError, match="sample_index must be non-negative"):
        collect_failures(source, sample_index=-1)


def test_collect_failures_uses_per_task_anchor_map_and_ignores_sample_index_default():
    source = {
        "per_problem": [
            {
                "task_id": "HumanEval/0",
                "samples": [
                    {
                        "passed": False,
                        "code": "first",
                        "error_message": "e0",
                        "failure_category": "assertion_error",
                    },
                    {
                        "passed": False,
                        "code": "second",
                        "error_message": "e1",
                        "failure_category": "syntax_error",
                    },
                ],
            },
            {
                "task_id": "HumanEval/1",
                "samples": [
                    {
                        "passed": False,
                        "code": "third",
                        "error_message": "e2",
                        "failure_category": "type_error",
                    },
                    {
                        "passed": True,
                        "code": "fourth",
                        "error_message": "",
                        "failure_category": "",
                    },
                ],
            },
            {
                "task_id": "HumanEval/2",
                "samples": [
                    {
                        "passed": False,
                        "code": "fifth",
                        "error_message": "e3",
                        "failure_category": "runtime_error",
                    }
                ],
            },
        ]
    }

    # Anchor map names sample 1 for HumanEval/0 (not the default sample 0) and
    # omits HumanEval/2 entirely, so it is excluded from the run.
    anchor_map = {"HumanEval/0": 1, "HumanEval/1": 0}

    failures = collect_failures(source, anchor_map=anchor_map)

    assert [(item["task_id"], item["sample_index"]) for item in failures] == [
        ("HumanEval/0", 1),
        ("HumanEval/1", 0),
    ]
    assert failures[0]["initial_code"] == "second"


def test_collect_failures_rejects_an_anchor_map_pointing_at_a_passing_sample():
    source = {
        "per_problem": [
            {
                "task_id": "HumanEval/0",
                "samples": [
                    {
                        "passed": True,
                        "code": "a",
                        "error_message": "",
                        "failure_category": "",
                    }
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="is passing, not failing"):
        collect_failures(source, anchor_map={"HumanEval/0": 0})


def test_collect_failures_refuses_sample_index_and_anchor_map_together():
    source = {"per_problem": []}

    with pytest.raises(ValueError, match="mutually exclusive"):
        collect_failures(source, sample_index=0, anchor_map={"HumanEval/0": 0})


def test_summarize_separates_first_step_and_later_successes():
    records = [
        {
            "final_passed": True,
            "success_iteration": 1,
            "initial_failure_category": "syntax_error",
            "final_failure_category": "",
        },
        {
            "final_passed": True,
            "success_iteration": 3,
            "initial_failure_category": "assertion_error",
            "final_failure_category": "",
        },
        {
            "final_passed": False,
            "success_iteration": None,
            "initial_failure_category": "type_error",
            "final_failure_category": "syntax_error",
        },
    ]

    summary = summarize(records)

    assert summary["repair_success_rate"] == 2 / 3
    assert summary["success_after_one_iteration"] == 1
    assert summary["additional_successes_from_iteration_2_plus"] == 1


def test_summarize_reports_week5_candidate_ranking_efficiency():
    records = [
        {
            "final_passed": True,
            "success_iteration": 1,
            "initial_failure_category": "syntax_error",
            "final_failure_category": "",
            "repair_model_calls": 3,
            "duplicate_candidates": 1,
            "oracle_candidate_passed": True,
            "trace": [
                {"iteration": 0, "duplicate": False},
                {"iteration": 1, "duplicate": False},
            ],
        },
        {
            "final_passed": False,
            "success_iteration": None,
            "initial_failure_category": "assertion_error",
            "final_failure_category": "assertion_error",
            "repair_model_calls": 3,
            "duplicate_candidates": 3,
            "oracle_candidate_passed": False,
            "trace": [
                {"iteration": 0, "duplicate": False},
                {"iteration": 1, "duplicate": True},
            ],
        },
    ]

    summary = summarize(records)

    assert summary["repair_model_calls"] == 6
    assert summary["duplicate_candidates"] == 4
    assert summary["duplicate_candidate_rate"] == 4 / 6
    assert summary["unique_candidates_executed"] == 2
    assert summary["oracle_candidate_successes"] == 1
    assert summary["ranking_oracle_gap"] == 0
    assert summary["stagnation_stops"] == 1


def test_select_failure_split_resolves_frozen_coordinates_in_manifest_order():
    failures = [
        {
            "task_id": "MBPP/11",
            "sample_index": 0,
            "initial_failure_category": "syntax_error",
        },
        {
            "task_id": "MBPP/12",
            "sample_index": 0,
            "initial_failure_category": "assertion_error",
        },
    ]

    selected = select_failure_split(
        failures,
        [
            {"task_id": "MBPP/12", "sample_index": 0},
            {"task_id": "MBPP/11", "sample_index": 0},
        ],
    )

    assert [record["task_id"] for record in selected] == ["MBPP/12", "MBPP/11"]
    assert select_source_order_failures(failures, 1) == failures[:1]


def test_summarize_reports_week6_category_budgets_and_stop_reasons():
    records = [
        {
            "final_passed": True,
            "success_iteration": 1,
            "initial_failure_category": "syntax_error",
            "final_failure_category": "",
            "repair_model_calls": 2,
            "duplicate_candidates": 1,
            "unique_candidates_executed": 1,
            "oracle_candidate_passed": True,
            "candidate_generation_stop_reasons": ["passing_candidate"],
            "repair_stop_reason": "success",
            "trace": [],
        },
        {
            "final_passed": False,
            "success_iteration": None,
            "initial_failure_category": "syntax_error",
            "final_failure_category": "syntax_error",
            "repair_model_calls": 4,
            "duplicate_candidates": 2,
            "unique_candidates_executed": 2,
            "oracle_candidate_passed": False,
            "candidate_generation_stop_reasons": ["consecutive_duplicate_limit"],
            "repair_stop_reason": "no_improvement",
            "trace": [],
        },
    ]

    summary = summarize(records)

    category = summary["per_initial_failure_category"]["syntax_error"]
    assert category["failures"] == 2
    assert category["repair_success_rate"] == 0.5
    assert category["repair_model_calls"] == 6
    assert category["duplicate_candidate_rate"] == 0.5
    assert category["unique_candidates_executed"] == 3
    assert summary["repair_stop_reasons"] == {"success": 1, "no_improvement": 1}
    lower, upper = summary["repair_success_wilson_95"]
    assert lower < 0.5 < upper


def test_wilson_interval_and_resume_validation():
    assert _wilson_interval(0, 0) == [0.0, 0.0]
    lower, upper = _wilson_interval(20, 50)
    assert 0.27 < lower < 0.29
    assert 0.53 < upper < 0.55

    _validate_resume_payload({"config": {"temperature": 0.2}}, {"temperature": 0.2})

    with pytest.raises(ValueError, match="Resume config mismatch"):
        _validate_resume_payload({"config": {"temperature": 0.2}}, {"temperature": 0.8})


def test_resume_grandfathers_missing_device_provenance():
    """A pre-Week-8 checkpoint has no device keys and must stay resumable."""
    legacy = {"config": {"temperature": 0.8, "seed": 42}}
    requested = {
        "temperature": 0.8,
        "seed": 42,
        "device_requested": "auto",
        "dtype_requested": "auto",
        "device_resolved": "cuda",
        "dtype_resolved": "torch.float16",
    }
    _validate_resume_payload(legacy, requested)


def test_resume_enforces_device_provenance_once_recorded():
    """Once a checkpoint records its device, a different one must be rejected."""
    recorded = {
        "config": {
            "temperature": 0.8,
            "device_requested": "auto",
            "dtype_requested": "auto",
            "device_resolved": "cpu",
            "dtype_resolved": "torch.float32",
        }
    }
    requested = {
        "temperature": 0.8,
        "device_requested": "auto",
        "dtype_requested": "auto",
        "device_resolved": "cuda",
        "dtype_resolved": "torch.float16",
    }
    with pytest.raises(ValueError, match="device_resolved"):
        _validate_resume_payload(recorded, requested)


def test_resume_grandfathers_a_checkpoint_written_before_context_mode_existed():
    """Pre-Week-9 checkpoints ran the full traceback, so 'error' may resume."""
    legacy = {"config": {"temperature": 0.8, "use_error_context": True}}

    _validate_resume_payload(
        legacy,
        {"temperature": 0.8, "use_error_context": True, "context_mode": "error"},
    )


def test_resume_refuses_to_continue_a_legacy_checkpoint_as_a_different_arm():
    legacy = {"config": {"temperature": 0.8, "use_error_context": True}}

    with pytest.raises(ValueError, match="context_mode"):
        _validate_resume_payload(
            legacy,
            {"temperature": 0.8, "use_error_context": False, "context_mode": "none"},
        )


def test_resume_still_rejects_a_stored_none_that_does_not_match():
    """A recorded None is a value, not an absent key: it must not be defaulted."""
    with pytest.raises(ValueError, match="context_mode"):
        _validate_resume_payload(
            {"config": {"context_mode": None}}, {"context_mode": "error"}
        )
