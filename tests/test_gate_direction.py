"""Tests for the Week 8 joint gate-direction and near-miss mechanism diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze_smart_repair import align_oof_scores, select_cohort
from scripts.diagnose_gate_direction import (
    budget_matched_point,
    build_analysis,
    gate_direction_sweep,
    mechanism_regression,
    select_inverted_gate,
    sibling_pass_rates,
)


def _source(
    n_problems: int = 8,
    passing: tuple[int, ...] = (0,),
    n_samples: int = 3,
    sibling_passing: dict[int, int] | None = None,
) -> dict:
    """Generation run with `n_samples` per problem; sample 0 carries the cohort.

    `sibling_passing` sets how many of samples 1.. pass for a given problem, so
    the sibling pass rate can be varied independently of sample 0's outcome.
    """
    sibling_passing = sibling_passing or {}
    problems = []
    for index in range(n_problems):
        n_sibling_passes = sibling_passing.get(index, 0)
        samples = [
            {
                "passed": index in passing,
                "failure_category": "" if index in passing else "assertion_error",
            }
        ]
        for sibling_index in range(n_samples - 1):
            samples.append(
                {
                    "passed": sibling_index < n_sibling_passes,
                    "failure_category": (
                        "" if sibling_index < n_sibling_passes else "assertion_error"
                    ),
                }
            )
        problems.append({"task_id": f"UNIT/{index}", "samples": samples})
    return {"run_id": "unit_source", "per_problem": problems}


def _predictor(source: dict, sample0_probabilities: list[float]) -> dict:
    """Predictor JSON whose OOF rows align with the source's flattened order."""
    probabilities, labels = [], []
    for index, problem in enumerate(source["per_problem"]):
        for sample_index, sample in enumerate(problem["samples"]):
            labels.append(int(sample["passed"]))
            probabilities.append(
                sample0_probabilities[index] if sample_index == 0 else 0.5
            )
    return {
        "source_run_id": source["run_id"],
        "configurations": [
            {
                "task": "sample_level",
                "feature_set": "self_consistency_plus_execution",
                "model_type": "logistic",
                "oof_proba": probabilities,
                "oof_y": labels,
            }
        ],
    }


def _repair_pool(
    source: dict,
    source_path: Path,
    repaired: set[int],
    calls: dict[int, int] | None = None,
) -> dict:
    calls = calls or {}
    records = []
    for index, problem in enumerate(source["per_problem"]):
        if problem["samples"][0]["passed"]:
            continue
        records.append(
            {
                "task_id": problem["task_id"],
                "sample_index": 0,
                "initial_failure_category": "assertion_error",
                "repair_model_calls": calls.get(index, 3),
                "duplicate_candidates": 0,
                "unique_candidates_executed": calls.get(index, 3),
                "selected_candidate_indices": [0],
                "final_passed": index in repaired,
                "oracle_candidate_passed": index in repaired,
            }
        )
    return {
        "run_id": "unit_repair",
        "config": {
            "source_results": str(source_path),
            "sample_index": 0,
            "model": "unit/model",
        },
        "records": records,
    }


def _cohort(source: dict, probabilities: list[float]) -> list[dict]:
    return select_cohort(align_oof_scores(source, _predictor(source, probabilities)))


class TestSelectInvertedGate:
    def test_takes_the_highest_probabilities(self):
        cohort = _cohort(_source(n_problems=4, passing=()), [0.1, 0.9, 0.4, 0.8])
        assert select_inverted_gate(cohort, 2) == {("UNIT/1", 0), ("UNIT/3", 0)}

    def test_complements_the_risk_gate(self):
        from scripts.analyze_smart_repair import select_risk_gate

        cohort = _cohort(_source(n_problems=4, passing=()), [0.1, 0.9, 0.4, 0.8])
        assert select_inverted_gate(cohort, 2) & select_risk_gate(cohort, 2) == set()

    def test_rejects_out_of_range_count(self):
        cohort = _cohort(_source(n_problems=2, passing=()), [0.1, 0.9])
        with pytest.raises(ValueError):
            select_inverted_gate(cohort, 3)


class TestSiblingPassRates:
    def test_excludes_the_cohort_sample(self):
        # Problem 0's sample 0 passes; its two siblings both fail.
        source = _source(n_problems=2, passing=(0,), n_samples=3)
        assert sibling_pass_rates(source)["UNIT/0"] == pytest.approx(0.0)

    def test_averages_only_the_siblings(self):
        source = _source(n_problems=2, passing=(), n_samples=3, sibling_passing={1: 1})
        assert sibling_pass_rates(source)["UNIT/1"] == pytest.approx(0.5)

    def test_omits_problems_without_siblings(self):
        assert sibling_pass_rates(_source(n_problems=2, passing=(), n_samples=1)) == {}


class TestMechanismRegression:
    def test_recovers_a_probability_only_signal(self):
        # Repair success tracks P(pass) exactly; siblings are constant.
        n = 12
        probabilities = [0.05 * (index + 1) for index in range(n)]
        source = _source(n_problems=n, passing=(), n_samples=3)
        cohort = _cohort(source, probabilities)
        pool = _repair_pool(source, Path("unit_source.json"), set(range(n // 2, n)))
        records = {(rec["task_id"], 0): rec for rec in pool["records"]}
        result = mechanism_regression(cohort, records, sibling_pass_rates(source))
        assert result["univariate_roc_auc"]["p_pass_oof"] == pytest.approx(1.0)
        assert result["likelihood_ratio"]["p_pass_given_sibling"]["p_value"] < 0.05

    def test_collapses_when_siblings_carry_the_signal(self):
        # Repair success is a deterministic function of the sibling pass rate,
        # and P(pass) is pure noise ordered against it.
        n = 12
        source = _source(
            n_problems=n,
            passing=(),
            n_samples=3,
            sibling_passing={
                index: (2 if index >= n // 2 else 0) for index in range(n)
            },
        )
        probabilities = [0.5 for _ in range(n)]
        cohort = _cohort(source, probabilities)
        pool = _repair_pool(source, Path("unit_source.json"), set(range(n // 2, n)))
        records = {(rec["task_id"], 0): rec for rec in pool["records"]}
        result = mechanism_regression(cohort, records, sibling_pass_rates(source))
        assert result["univariate_roc_auc"]["sibling_pass_rate"] == pytest.approx(1.0)
        assert result["likelihood_ratio"]["p_pass_given_sibling"]["p_value"] > 0.05

    def test_requires_both_outcomes(self):
        source = _source(n_problems=4, passing=(), n_samples=3)
        cohort = _cohort(source, [0.1, 0.2, 0.3, 0.4])
        pool = _repair_pool(source, Path("unit_source.json"), set())
        records = {(rec["task_id"], 0): rec for rec in pool["records"]}
        with pytest.raises(ValueError):
            mechanism_regression(cohort, records, sibling_pass_rates(source))


class TestGateDirectionSweep:
    def test_inverted_wins_when_high_probability_rows_are_repairable(self):
        n = 8
        source = _source(n_problems=n, passing=(), n_samples=3)
        probabilities = [0.1 * (index + 1) for index in range(n)]
        cohort = _cohort(source, probabilities)
        pool = _repair_pool(source, Path("unit_source.json"), {4, 5, 6, 7})
        records = {(rec["task_id"], 0): rec for rec in pool["records"]}
        rows = gate_direction_sweep(
            cohort, records, coverages=(0.5,), n_seeds=5, first_seed=0
        )
        assert rows[0]["inverted"]["added_passes"] == 4
        assert rows[0]["risk"]["added_passes"] == 0

    def test_full_coverage_arms_agree(self):
        n = 6
        source = _source(n_problems=n, passing=(), n_samples=3)
        cohort = _cohort(source, [0.1 * (index + 1) for index in range(n)])
        pool = _repair_pool(source, Path("unit_source.json"), {0, 3})
        records = {(rec["task_id"], 0): rec for rec in pool["records"]}
        row = gate_direction_sweep(
            cohort, records, coverages=(1.0,), n_seeds=3, first_seed=0
        )[0]
        assert (
            row["risk"]["added_passes"]
            == row["inverted"]["added_passes"]
            == row["random_mean"]["added_passes"]
            == 2
        )


class TestBudgetMatchedPoint:
    def test_picks_the_best_affordable_row(self):
        sweep = [
            {
                "coverage": 0.5,
                "gate_count": 5,
                "inverted": {"added_passes": 3, "repair_model_calls": 40},
            },
            {
                "coverage": 0.75,
                "gate_count": 8,
                "inverted": {"added_passes": 5, "repair_model_calls": 90},
            },
        ]
        assert budget_matched_point(sweep, 100)["added_passes"] == 5
        assert budget_matched_point(sweep, 50)["coverage"] == 0.5

    def test_returns_none_when_nothing_is_affordable(self):
        sweep = [
            {
                "coverage": 0.5,
                "gate_count": 5,
                "inverted": {"added_passes": 3, "repair_model_calls": 40},
            }
        ]
        assert budget_matched_point(sweep, 10) is None


class TestBuildAnalysis:
    def test_end_to_end_block_shape(self, tmp_path):
        n = 10
        source = _source(n_problems=n, passing=(0,), n_samples=3)
        probabilities = [0.05 + 0.09 * index for index in range(n)]
        source_path = tmp_path / "unit_source.json"
        pool = _repair_pool(source, source_path, {6, 7, 8, 9})
        analysis = build_analysis(
            source_path,
            source,
            _predictor(source, probabilities),
            pool,
            smart_coverage=0.5,
            coverages=(0.5, 1.0),
            n_seeds=3,
        )
        assert analysis["analysis"] == "joint_gate_direction"
        assert analysis["cohort"]["candidates"] == n
        assert analysis["arms_at_smart_coverage"]["repair_all"]["added_passes"] == 4
        assert len(analysis["direction_sweep"]) == 2
        assert analysis["mechanism_regression"]["n"] == n - 1
