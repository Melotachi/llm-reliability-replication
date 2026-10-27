"""Tests for Student 2 Week 7 Smart Repair policy analysis."""

import json
from pathlib import Path

import pytest

from scripts.analyze_smart_repair import (
    align_oof_scores,
    build_analysis,
    counterfactual_stop_on_pass,
    plot_category_efficiency,
    plot_risk_coverage,
    select_cohort,
    select_risk_gate,
)


def _source():
    outcomes = [True, False, False, False]
    return {
        "run_id": "source-run",
        "per_problem": [
            {
                "task_id": f"MBPP/{index}",
                "samples": [
                    {
                        "passed": passed,
                        "failure_category": "" if passed else "assertion_error",
                    }
                ],
            }
            for index, passed in enumerate(outcomes)
        ],
    }


def _predictor(labels=None, probabilities=None):
    return {
        "source_run_id": "source-run",
        "configurations": [
            {
                "task": "sample_level",
                "feature_set": "self_consistency_plus_execution",
                "model_type": "logistic",
                "oof_y": [1, 0, 0, 0] if labels is None else labels,
                "oof_proba": (
                    [0.9, 0.1, 0.2, 0.8] if probabilities is None else probabilities
                ),
            }
        ],
    }


def _repair_run():
    records = []
    for task_id, passed, calls, selected_index in (
        ("MBPP/1", True, 1, 0),
        ("MBPP/2", False, 3, None),
        ("MBPP/3", True, 2, 1),
    ):
        records.append(
            {
                "task_id": task_id,
                "sample_index": 0,
                "final_passed": passed,
                "repair_model_calls": calls,
                "duplicate_candidates": 0,
                "unique_candidates_executed": calls,
                "selected_candidate_indices": [selected_index],
            }
        )
    return {
        "config": {
            "source_results": "source.json",
            "sample_index": 0,
        },
        "records": records,
    }


def test_oof_alignment_and_risk_gate_use_source_order_for_ties():
    scored = align_oof_scores(_source(), _predictor(probabilities=[0.9, 0.1, 0.1, 0.8]))
    cohort = select_cohort(scored)

    assert select_risk_gate(cohort, 2) == {("MBPP/1", 0), ("MBPP/2", 0)}


def test_oof_alignment_rejects_label_or_source_mismatch():
    with pytest.raises(ValueError, match="OOF labels do not align"):
        align_oof_scores(_source(), _predictor(labels=[0, 0, 0, 0]))

    predictor = _predictor()
    predictor["source_run_id"] = "different-run"
    with pytest.raises(ValueError, match="Predictor/source run mismatch"):
        align_oof_scores(_source(), predictor)


def test_build_analysis_reuses_one_repair_pool_for_all_conditions():
    analysis = build_analysis(
        Path("source.json"),
        _source(),
        _predictor(),
        _repair_run(),
        smart_coverage=0.5,
        seed=42,
    )

    conditions = analysis["conditions"]
    assert conditions["baseline"]["final_passed"] == 1
    assert conditions["smart_repair_at_0.50"]["final_passed"] == 2
    assert conditions["repair_all"]["final_passed"] == 3
    assert conditions["smart_repair_at_0.50"]["repair_model_calls"] == 4
    assert conditions["repair_all"]["repair_model_calls"] == 6
    assert conditions["smart_repair_at_0.50"]["repair_all_successes_captured"] == 1
    assert conditions["smart_repair_at_0.50"]["repair_all_success_capture_rate"] == 0.5
    assert conditions["repair_all"]["successful_selected_candidate_indices"] == {
        "0": 1,
        "1": 1,
    }
    assert len(analysis["risk_coverage"]) == 5


def test_analysis_plots_render_without_a_gui(tmp_path):
    analysis = build_analysis(
        Path("source.json"),
        _source(),
        _predictor(),
        _repair_run(),
        smart_coverage=0.5,
        seed=42,
    )
    risk_path = tmp_path / "risk.png"
    category_path = tmp_path / "category.png"

    plot_risk_coverage(analysis, risk_path)
    plot_category_efficiency(analysis, category_path)

    assert risk_path.stat().st_size > 0
    assert category_path.stat().st_size > 0


def test_counterfactual_stop_on_pass_uses_candidate_prefixes():
    run = {
        "records": [
            {
                "final_passed": True,
                "trace": [
                    {"iteration": 0},
                    {
                        "iteration": 1,
                        "candidates": [
                            {"candidate_index": 0, "passed": True},
                            {"candidate_index": 1, "passed": False},
                            {"candidate_index": 2, "passed": False},
                        ],
                    },
                ],
            },
            {
                "final_passed": True,
                "trace": [
                    {"iteration": 0},
                    {
                        "iteration": 1,
                        "candidates": [
                            {"candidate_index": 0, "passed": False},
                            {"candidate_index": 1, "passed": False},
                            {"candidate_index": 2, "passed": True},
                        ],
                    },
                ],
            },
            {
                "final_passed": False,
                "trace": [
                    {"iteration": 0},
                    {
                        "iteration": 1,
                        "candidates": [
                            {"candidate_index": 0, "passed": False},
                            {"candidate_index": 1, "passed": False},
                            {"candidate_index": 2, "passed": False},
                        ],
                    },
                ],
            },
        ]
    }

    comparison = counterfactual_stop_on_pass(run)

    assert comparison["successful_repairs"] == 2
    assert comparison["actual_model_calls"] == 9
    assert comparison["stop_on_pass_model_calls"] == 7


def test_week6_counterfactual_is_exactly_150_to_132_calls():
    root = Path(__file__).resolve().parent.parent
    run_path = root / "results" / "mbpp_week6_evaluation_temp08_accept.json"

    run = json.loads(run_path.read_text(encoding="utf-8"))
    comparison = counterfactual_stop_on_pass(run)

    assert comparison["successful_repairs"] == 12
    assert comparison["actual_model_calls"] == 150
    assert comparison["stop_on_pass_model_calls"] == 132
