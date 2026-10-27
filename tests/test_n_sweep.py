"""Tests for scripts/analyze_n_sweep.py.

The sweep's numbers are only meaningful if the subset scoring reproduces the
canonical metric definitions and the projection arithmetic is right, so both
are pinned here. The expensive parts (metric computation over real pools) are
covered by estimate_reliability's own self-check, which this script reuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_n_sweep import (
    MIN_N_FOR_SPLIT_HALF,
    _correlations_for_subsets,
    anchor_projection,
    build_caches,
    compare_to_prereg,
    infer_cell_key,
    score_budget,
)


def _problem(task_id: str, codes: list[str], passed: list[bool]) -> dict:
    return {
        "task_id": task_id,
        "samples": [{"code": c, "passed": p} for c, p in zip(codes, passed)],
    }


def _results(n_problems: int = 5, n_samples: int = 8) -> dict:
    """A synthetic run where diversity and pass rate are genuinely related."""
    per_problem = []
    for p in range(n_problems):
        # Problem p gets p distinct bodies among n_samples -> rising diversity,
        # and a falling pass rate, so the correlation is real and negative.
        codes = [f"def f(x):\n    return x + {i % (p + 1)}\n" for i in range(n_samples)]
        n_pass = max(0, n_samples - 2 * p)
        passed = [True] * n_pass + [False] * (n_samples - n_pass)
        per_problem.append(_problem(f"T/{p}", codes, passed))
    return {
        "config": {
            "dataset": "humaneval",
            "model": "Salesforce/codegen-350M-mono",
            "n_problems": n_problems,
        },
        "per_problem": per_problem,
    }


def test_build_caches_skips_problems_with_too_few_samples():
    results = _results(n_problems=4, n_samples=8)
    results["per_problem"].append(_problem("T/short", ["def f(): pass"], [True]))
    caches = build_caches(results, prompts={}, representation="full")
    assert [c["task_id"] for c in caches] == ["T/0", "T/1", "T/2", "T/3"]


def test_build_caches_requires_three_problems():
    results = _results(n_problems=2, n_samples=8)
    with pytest.raises(ValueError, match="need >= 3 problems"):
        build_caches(results, prompts={}, representation="full")


def test_correlations_are_none_for_a_constant_metric():
    # Every problem identical -> no variance in any metric or in pass rate.
    codes = ["def f(x):\n    return x\n"] * 6
    results = {
        "config": {},
        "per_problem": [_problem(f"T/{i}", codes, [True] * 6) for i in range(4)],
    }
    caches = build_caches(results, prompts={}, representation="full")
    subsets = [np.arange(6) for _ in caches]
    out = _correlations_for_subsets(caches, subsets, ["ast_node_diversity"])
    assert out["ast_node_diversity"] == {"pearson_r": None, "spearman_rho": None}


def test_score_budget_uses_one_draw_when_the_budget_is_the_whole_pool():
    caches = build_caches(_results(n_samples=8), prompts={}, representation="full")
    rng = np.random.default_rng(0)
    full = score_budget(caches, 8, ["ast_node_diversity"], 25, 5, rng)
    assert full["exhaustive"] is True
    assert full["n_draws"] == 1

    partial = score_budget(caches, 4, ["ast_node_diversity"], 7, 5, rng)
    assert partial["exhaustive"] is False
    assert partial["n_draws"] == 7


def test_score_budget_reports_ceiling_from_both_reliabilities():
    caches = build_caches(_results(n_samples=8), prompts={}, representation="full")
    rng = np.random.default_rng(0)
    scored = score_budget(caches, 8, ["ast_node_diversity"], 4, 20, rng)
    entry = scored["metrics"]["ast_node_diversity"]
    if entry["measured_ceiling"] is not None:
        expected = np.sqrt(
            entry["metric_reliability"] * scored["pass_rate_reliability"]
        )
        assert entry["measured_ceiling"] == pytest.approx(expected)
        assert 0.0 <= entry["measured_ceiling"] <= 1.0


def _sweep_entry(n, metric_rel, pass_rel, observed_r):
    return {
        "n": n,
        "n_draws": 1,
        "exhaustive": False,
        "pass_rate_reliability": pass_rel,
        "metrics": {
            "ast_node_diversity": {
                "pearson": {"mean": observed_r, "std": 0.0, "n_draws_used": 1},
                "spearman": {"mean": observed_r, "std": 0.0, "n_draws_used": 1},
                "metric_reliability": metric_rel,
                "measured_ceiling": float(np.sqrt(metric_rel * pass_rel)),
                "percent_of_measured_ceiling": None,
            }
        },
    }


def test_anchor_projection_predicts_larger_budgets_and_rises():
    sweep = [
        _sweep_entry(4, 0.65, 0.78, -0.50),
        _sweep_entry(10, 0.82, 0.90, -0.62),
        _sweep_entry(20, 0.90, 0.95, -0.68),
    ]
    projection = anchor_projection(sweep, anchor_n=4)
    rows = projection["ast_node_diversity"]

    assert [r["n"] for r in rows] == [10, 20]
    predicted = [r["predicted_abs_pearson_r"] for r in rows]
    # The prediction must climb with N — that is the whole claim.
    assert predicted == sorted(predicted)
    # And it must exceed the anchor's own observed |r|.
    assert predicted[0] > 0.50
    for row in rows:
        assert row["deviation"] == pytest.approx(
            row["observed_abs_pearson_r"] - row["predicted_abs_pearson_r"]
        )


def test_anchor_projection_rejects_an_unscored_anchor():
    sweep = [_sweep_entry(10, 0.8, 0.9, -0.6)]
    with pytest.raises(ValueError, match="was not scored"):
        anchor_projection(sweep, anchor_n=4)


def test_anchor_projection_skips_degenerate_reliability():
    # A reliability of exactly 1.0 inverts to a zero denominator; the metric
    # must be dropped with a warning, not crash the whole sweep.
    sweep = [
        _sweep_entry(4, 1.0, 0.78, -0.50),
        _sweep_entry(10, 0.82, 0.90, -0.62),
    ]
    assert anchor_projection(sweep, anchor_n=4) == {}


def test_compare_to_prereg_accepts_json_string_keys():
    sweep = [_sweep_entry(20, 0.9, 0.95, -0.68)]
    # json.load turns the pre-registration's int budget keys into strings.
    predictions = {
        "ast_node_diversity": {
            "by_target_n": {"20": {"predicted_abs_pearson_r": 0.671, "ceiling": 0.925}}
        }
    }
    rows = compare_to_prereg(sweep, predictions)["ast_node_diversity"]
    assert rows[0]["n"] == 20
    assert rows[0]["observed_abs_pearson_r"] == pytest.approx(0.68)
    assert rows[0]["deviation"] == pytest.approx(0.68 - 0.671)


def test_compare_to_prereg_skips_budgets_without_a_prediction():
    sweep = [_sweep_entry(7, 0.9, 0.95, -0.68)]
    predictions = {
        "ast_node_diversity": {
            "by_target_n": {"20": {"predicted_abs_pearson_r": 0.671, "ceiling": 0.925}}
        }
    }
    assert compare_to_prereg(sweep, predictions) == {}


def test_infer_cell_key_matches_on_model_and_problem_count():
    prereg = {
        "cells": {
            "cg/HE60": {
                "model": "Salesforce/codegen-350M-mono",
                "n_problems": 60,
            },
            "ds/HE60": {
                "model": "deepseek-ai/deepseek-coder-1.3b-base",
                "n_problems": 60,
            },
        }
    }
    results = {"config": {"model": "Salesforce/codegen-350M-mono", "n_problems": 60}}
    assert infer_cell_key(prereg, results) == "cg/HE60"

    results["config"]["n_problems"] = 164
    assert infer_cell_key(prereg, results) is None


def test_min_budget_constant_supports_two_halves():
    assert MIN_N_FOR_SPLIT_HALF >= 4
