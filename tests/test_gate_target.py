"""Tests for the Week 8 joint gate-target diagnosis and random-gate sweep."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_smart_repair import _stored_basename
from scripts.diagnose_gate_target import build_diagnosis, roc_auc, score_target
from scripts.sweep_random_gate import sweep_random_gates


def _source(n_problems: int = 6, passing: tuple[int, ...] = (0,)) -> dict:
    """Minimal generation run: one sample per problem, `passing` indices pass."""
    return {
        "run_id": "unit_source",
        "per_problem": [
            {
                "task_id": f"UNIT/{index}",
                "samples": [
                    {
                        "passed": index in passing,
                        "failure_category": (
                            "" if index in passing else "assertion_error"
                        ),
                    }
                ],
            }
            for index in range(n_problems)
        ],
    }


def _predictor(source: dict, probabilities: list[float]) -> dict:
    labels = [
        int(sample["passed"])
        for problem in source["per_problem"]
        for sample in problem["samples"]
    ]
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
    source: dict, source_path: Path, repaired: set[int], calls: int = 3
) -> dict:
    records = []
    for index, problem in enumerate(source["per_problem"]):
        if problem["samples"][0]["passed"]:
            continue
        records.append(
            {
                "task_id": problem["task_id"],
                "sample_index": 0,
                "initial_failure_category": "assertion_error",
                "repair_model_calls": calls,
                "duplicate_candidates": 0,
                "unique_candidates_executed": calls,
                "selected_candidate_indices": [0],
                "final_passed": index in repaired,
                "oracle_candidate_passed": index in repaired,
            }
        )
    return {
        "run_id": "unit_repair",
        "config": {"source_results": str(source_path), "sample_index": 0},
        "records": records,
    }


class TestRocAuc:
    def test_perfect_ranking_scores_one(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        assert roc_auc(scores, labels) == pytest.approx(1.0)

    def test_reversed_ranking_scores_zero(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([0, 0, 1, 1])
        assert roc_auc(scores, labels) == pytest.approx(0.0)

    def test_constant_score_is_chance(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0, 1, 0, 1])
        assert roc_auc(scores, labels) == pytest.approx(0.5)

    def test_ties_use_mid_ranks(self):
        # Two positives tied with one negative: the tie contributes 0.5 each.
        scores = np.array([0.1, 0.5, 0.5])
        labels = np.array([0, 0, 1])
        assert roc_auc(scores, labels) == pytest.approx(0.75)

    def test_single_class_raises(self):
        with pytest.raises(ValueError, match="both classes"):
            roc_auc(np.array([0.1, 0.2]), np.array([1, 1]))


class TestScoreTarget:
    def test_permutation_p_is_large_for_chance_score(self):
        rng = np.random.default_rng(0)
        scores = rng.random(80)
        labels = (rng.random(80) < 0.4).astype(int)
        result = score_target(
            scores,
            labels,
            name="noise",
            seed=1,
            n_permutations=400,
            n_bootstrap=200,
        )
        assert result["permutation_p"] > 0.05
        low, high = result["bootstrap_95"]
        assert low <= result["roc_auc"] <= high

    def test_permutation_p_is_small_for_perfect_score(self):
        scores = np.linspace(0.0, 1.0, 40)
        labels = (scores > 0.5).astype(int)
        result = score_target(
            scores,
            labels,
            name="perfect",
            seed=1,
            n_permutations=400,
            n_bootstrap=200,
        )
        assert result["roc_auc"] == pytest.approx(1.0)
        assert result["permutation_p"] < 0.01


class TestBuildDiagnosis:
    def test_separates_a_selection_signal_from_an_allocation_signal(self, tmp_path):
        # Probabilities rank initial correctness perfectly (problem 0 passes and
        # carries the top score) but are unrelated to which failures got fixed.
        source = _source(n_problems=6, passing=(0,))
        source_path = tmp_path / "unit_source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        predictor = _predictor(source, [0.9, 0.5, 0.1, 0.4, 0.2, 0.3])
        repair = _repair_pool(source, source_path, repaired={1, 2})

        result = build_diagnosis(
            source_path,
            source,
            predictor,
            repair,
            n_permutations=200,
            n_bootstrap=200,
        )

        targets = {target["target"]: target for target in result["targets"]}
        assert targets["initial_passed"]["roc_auc"] == pytest.approx(1.0)
        assert result["cohort"] == {
            "candidates": 6,
            "initially_passing": 1,
            "initially_failing": 5,
            "repaired": 2,
        }
        assert targets["repair_success"]["n"] == 5
        assert targets["repair_success"]["n_positive"] == 2
        assert result["target_gap"] == pytest.approx(
            targets["initial_passed"]["roc_auc"] - targets["repair_success"]["roc_auc"]
        )

    def test_reports_repair_rate_per_failure_category(self, tmp_path):
        source = _source(n_problems=4, passing=())
        source_path = tmp_path / "unit_source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        predictor = _predictor(source, [0.1, 0.2, 0.3, 0.4])
        repair = _repair_pool(source, source_path, repaired={0, 1})

        result = build_diagnosis(
            source_path,
            source,
            predictor,
            repair,
            n_permutations=100,
            n_bootstrap=100,
        )

        rates = result["per_initial_failure_category"]["assertion_error"]
        assert rates == {"n": 4, "repaired": 2, "repair_rate": 0.5}

    def test_windows_written_pool_path_validates_on_posix(self, tmp_path):
        source = _source(n_problems=4, passing=())
        source_path = tmp_path / "unit_source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        predictor = _predictor(source, [0.1, 0.2, 0.3, 0.4])
        repair = _repair_pool(source, source_path, repaired={0})
        # The stored Week 6/7 traces carry backslash paths from their Windows run.
        repair["config"]["source_results"] = r"results\unit_source.json"

        result = build_diagnosis(
            source_path,
            source,
            predictor,
            repair,
            n_permutations=100,
            n_bootstrap=100,
        )
        assert result["cohort"]["initially_failing"] == 4

    def test_stored_basename_handles_both_separators(self):
        assert _stored_basename(r"results\run.json") == "run.json"
        assert _stored_basename("results/run.json") == "run.json"
        assert _stored_basename("run.json") == "run.json"


class TestSweepRandomGates:
    def test_smart_gate_is_located_within_the_random_distribution(self, tmp_path):
        source = _source(n_problems=8, passing=(0, 1))
        source_path = tmp_path / "unit_source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        predictor = _predictor(source, [0.95, 0.90, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
        repair = _repair_pool(source, source_path, repaired={2, 3})

        sweep = sweep_random_gates(
            source_path,
            source,
            predictor,
            repair,
            smart_coverage=0.75,
            n_seeds=40,
        )

        assert sweep["config"]["gate_count"] == 6
        assert len(sweep["draws"]) == 40
        assert {draw["seed"] for draw in sweep["draws"]} == set(range(40))
        # Every arm is bounded by the Repair-All ceiling on the shared pool.
        ceiling = sweep["repair_all"]["added_passes"]
        assert sweep["smart"]["added_passes"] <= ceiling
        assert all(draw["added_passes"] <= ceiling for draw in sweep["draws"])
        fractions = sweep["random_distribution"]["added_passes"]
        assert (
            fractions["fraction_below"]
            + fractions["fraction_equal"]
            + fractions["fraction_above"]
        ) == pytest.approx(1.0)

    def test_full_coverage_gate_reproduces_repair_all(self, tmp_path):
        source = _source(n_problems=5, passing=(0,))
        source_path = tmp_path / "unit_source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        predictor = _predictor(source, [0.9, 0.1, 0.2, 0.3, 0.4])
        repair = _repair_pool(source, source_path, repaired={1, 3})

        sweep = sweep_random_gates(
            source_path, source, predictor, repair, smart_coverage=1.0, n_seeds=5
        )

        assert sweep["smart"] == pytest.approx(
            {
                "added_passes": sweep["repair_all"]["added_passes"],
                "repair_model_calls": sweep["repair_all"]["repair_model_calls"],
                "attempted_repairs": 4,
                "final_passed": 3,
            }
        )
        # At full coverage every random draw is the same gate.
        assert {draw["added_passes"] for draw in sweep["draws"]} == {
            sweep["repair_all"]["added_passes"]
        }

    def test_seeds_are_reproducible(self, tmp_path):
        source = _source(n_problems=8, passing=(0,))
        source_path = tmp_path / "unit_source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        predictor = _predictor(source, [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        repair = _repair_pool(source, source_path, repaired={1, 4})

        kwargs = dict(smart_coverage=0.5, n_seeds=10)
        first = sweep_random_gates(source_path, source, predictor, repair, **kwargs)
        second = sweep_random_gates(source_path, source, predictor, repair, **kwargs)
        assert first["draws"] == second["draws"]

    def test_rejects_invalid_coverage(self, tmp_path):
        source = _source(n_problems=4, passing=(0,))
        source_path = tmp_path / "unit_source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        predictor = _predictor(source, [0.9, 0.1, 0.2, 0.3])
        repair = _repair_pool(source, source_path, repaired=set())

        with pytest.raises(ValueError, match="between zero and one"):
            sweep_random_gates(
                source_path, source, predictor, repair, smart_coverage=1.5
            )


class TestGateExclusion:
    def test_successes_concentrated_in_the_excluded_tail_are_flagged(self, tmp_path):
        """All successes sit at the top of the ranking, i.e. in what the gate cuts."""
        source = _source(n_problems=8, passing=())
        source_path = tmp_path / "unit_source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        # Ascending P(pass): problems 6 and 7 rank highest and are excluded at 75%.
        predictor = _predictor(source, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        repair = _repair_pool(source, source_path, repaired={6, 7})

        result = build_diagnosis(
            source_path,
            source,
            predictor,
            repair,
            smart_coverage=0.75,
            n_permutations=100,
            n_bootstrap=100,
        )

        exclusion = result["gate_exclusion"]
        assert exclusion["n_excluded_by_gate"] == 2
        assert exclusion["successes_excluded_observed"] == 2
        assert exclusion["hypergeometric_p_upper"] < 0.05

    def test_successes_spread_evenly_are_not_flagged(self, tmp_path):
        source = _source(n_problems=8, passing=())
        source_path = tmp_path / "unit_source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        predictor = _predictor(source, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        # Both successes sit inside the gated region, not the excluded tail.
        repair = _repair_pool(source, source_path, repaired={0, 1})

        result = build_diagnosis(
            source_path,
            source,
            predictor,
            repair,
            smart_coverage=0.75,
            n_permutations=100,
            n_bootstrap=100,
        )

        exclusion = result["gate_exclusion"]
        assert exclusion["successes_excluded_observed"] == 0
        assert exclusion["hypergeometric_p_upper"] == pytest.approx(1.0)
        assert exclusion["success_ranks_ascending_p_pass"] == [0, 1]
