"""Tests for the selection-experiment helpers in scripts/evaluate_selection.py."""

from __future__ import annotations

import numpy as np

from scripts.evaluate_selection import evaluate_selectors, select_per_problem


def _problem(task_id: str, passed: list[bool]) -> dict:
    return {
        "task_id": task_id,
        "samples": [{"code": f"code_{i}", "passed": p} for i, p in enumerate(passed)],
    }


def _rows(task_id: str, self_distances: list[float], oof: list[float]) -> list[dict]:
    return [
        {
            "task_id": task_id,
            "passed": 0,
            "execution_behavior_self_distance": d,
            "_oof_proba__arm": p,
        }
        for d, p in zip(self_distances, oof)
    ]


def test_select_per_problem_argmin_execution_and_argmax_classifier():
    problem = _problem("T/0", [False, True, False])
    rows = _rows("T/0", self_distances=[0.9, 0.1, 0.5], oof=[0.2, 0.3, 0.8])
    picks = select_per_problem(problem, rows, ["arm"], np.random.default_rng(0))
    assert picks["execution_heuristic"] == 1  # lowest self-distance
    assert picks["classifier__arm"] == 2  # highest OOF P(pass)
    assert 0 <= picks["random_draw"] < 3


def test_select_per_problem_ties_break_to_first_index():
    problem = _problem("T/0", [True, False])
    rows = _rows("T/0", self_distances=[0.5, 0.5], oof=[0.7, 0.7])
    picks = select_per_problem(problem, rows, ["arm"], np.random.default_rng(0))
    assert picks["execution_heuristic"] == 0
    assert picks["classifier__arm"] == 0


def test_select_per_problem_skips_single_sample_and_misaligned_rows():
    single = _problem("T/0", [True])
    rows = _rows("T/0", [0.1], [0.9])
    assert select_per_problem(single, rows, ["arm"], np.random.default_rng(0)) is None
    misaligned = _problem("T/1", [True, False])
    assert (
        select_per_problem(misaligned, rows, ["arm"], np.random.default_rng(0)) is None
    )


def test_select_per_problem_nan_oof_drops_only_that_arm():
    problem = _problem("T/0", [False, True])
    rows = _rows("T/0", self_distances=[0.3, 0.2], oof=[float("nan"), 0.5])
    picks = select_per_problem(problem, rows, ["arm"], np.random.default_rng(0))
    assert "classifier__arm" not in picks
    assert picks["execution_heuristic"] == 1


def test_evaluate_selectors_accuracies_oracle_and_trap():
    # Problem A: solvable, heuristic picks the passing sample.
    # Problem B: unsolvable AND low-diversity -> the consensus trap.
    results = {
        "per_problem": [
            _problem("A", [False, True]),
            _problem("B", [False, False]),
        ]
    }
    sample_rows = _rows("A", [0.9, 0.1], [0.1, 0.8]) + _rows(
        "B", [0.05, 0.05], [0.6, 0.4]
    )
    problem_rows = [
        {"task_id": "A", "execution_behavior_diversity": 0.5},
        {"task_id": "B", "execution_behavior_diversity": 0.05},
    ]
    out = evaluate_selectors(results, sample_rows, problem_rows, ["arm"], seed=0)

    acc = out["selection_accuracy"]
    assert acc["oracle"]["overall"] == 0.5  # only A is solvable
    assert acc["execution_heuristic"]["overall"] == 0.5  # right on A, doomed on B
    assert acc["execution_heuristic"]["conditional_on_solvable"] == 1.0
    assert acc["classifier__arm"]["conditional_on_solvable"] == 1.0
    assert out["expected_random"] == 0.25  # mean of pass rates 0.5 and 0.0
    trap = out["trap_diagnostic"]
    assert trap["n_unsolvable"] == 1
    assert trap["n_unsolvable_low_diversity"] == 1
    assert trap["trapped_task_ids"] == ["B"]
    assert out["n_problems_scored"] == 2
    # Every competing pair gets a McNemar entry with a BH-adjusted p-value.
    assert all("p_bh" in p for p in out["pairwise_mcnemar"])
