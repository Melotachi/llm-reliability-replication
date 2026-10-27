"""Tests for Week 6 repair robustness comparison analysis."""

from __future__ import annotations

import json

import pytest

from scripts.analyze_repair_robustness import compare_runs, format_markdown


def _write_run(path, *, selection: str, outcomes: list[bool], temperature: float):
    payload = {
        "config": {
            "selection_sha256": selection,
            "failure_split": "development",
            "temperature": temperature,
            "duplicate_policy": "accept",
            "repair_candidates": 3,
            "max_candidate_calls": 3,
            "stop_on_pass": False,
        },
        "summary": {
            "completed_failures": len(outcomes),
            "successful_repairs": sum(outcomes),
            "repair_success_rate": sum(outcomes) / len(outcomes),
            "repair_model_calls": 3 * len(outcomes),
            "duplicate_candidates": 1,
            "duplicate_candidate_rate": 1 / (3 * len(outcomes)),
            "unique_candidates_executed": 3 * len(outcomes) - 1,
            "model_calls_per_success": 3 * len(outcomes) / sum(outcomes),
            "ranking_oracle_gap": 0,
            "per_initial_failure_category": {},
        },
        "records": [
            {"task_id": f"MBPP/{index}", "sample_index": 0, "final_passed": value}
            for index, value in enumerate(outcomes)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_runs_reports_paired_outcomes(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_run(first, selection="same", outcomes=[True, False], temperature=0.2)
    _write_run(second, selection="same", outcomes=[True, True], temperature=0.8)

    comparison = compare_runs([first, second], require_same_selection=True)

    assert comparison["same_selection"] is True
    paired = comparison["paired_outcomes"][0]
    assert paired["both_success"] == 1
    assert paired["right_only_success"] == 1
    assert paired["left_only_success"] == 0
    assert "Repair Robustness Comparison" in format_markdown(comparison)


def test_compare_runs_rejects_different_selections_when_required(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_run(first, selection="a", outcomes=[True], temperature=0.2)
    _write_run(second, selection="b", outcomes=[True], temperature=0.8)

    with pytest.raises(ValueError, match="same frozen failure selection"):
        compare_runs([first, second], require_same_selection=True)
