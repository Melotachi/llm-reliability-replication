"""Tests for scripts/analyze_consensus_deciles.py and the reliability table.

Both scripts are mostly orchestration; these cover the pure functions that
carry the research claims — quantile grouping, the confidently-wrong rate,
and the attenuation-ceiling arithmetic — with hand-checkable inputs.
"""

from __future__ import annotations

import pytest

from scripts.analyze_consensus_deciles import confidently_wrong_rate, quantile_table
from scripts.estimate_reliability import (
    DEFAULT_METRICS,
    build_ceiling_table,
    build_problem_cache,
    metrics_from_cache,
)
from src.uncertainty import compute_uncertainty


def _rows(values: list[float], solved: list[int]) -> list[dict]:
    return [
        {
            "task_id": f"T/{i}",
            "ast_node_diversity": v,
            "solved": s,
            "pass_rate": float(s),
        }
        for i, (v, s) in enumerate(zip(values, solved))
    ]


# ---------------------------------------------------------------------------
# quantile_table
# ---------------------------------------------------------------------------


def test_quantile_table_group_one_is_the_most_consensus():
    # Diversity ascending => group 1 holds the lowest diversity, i.e. the
    # highest agreement. Solved only in the low-diversity half.
    rows = _rows([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
    table = quantile_table(rows, "ast_node_diversity", n_groups=2)
    assert [group["frac_solved"] for group in table] == [1.0, 0.0]
    assert [group["n_problems"] for group in table] == [2, 2]


def test_quantile_table_records_the_unsolved_ids_per_group():
    rows = _rows([0.1, 0.2, 0.8, 0.9], [1, 0, 0, 0])
    table = quantile_table(rows, "ast_node_diversity", n_groups=2)
    assert table[0]["task_ids_all_fail"] == ["T/1"]
    assert sorted(table[1]["task_ids_all_fail"]) == ["T/2", "T/3"]


def test_quantile_table_returns_none_for_a_constant_feature():
    rows = _rows([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
    assert quantile_table(rows, "ast_node_diversity") is None


def test_quantile_table_returns_none_for_a_missing_feature():
    rows = _rows([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0])
    assert quantile_table(rows, "execution_behavior_diversity") is None


# ---------------------------------------------------------------------------
# confidently_wrong_rate
# ---------------------------------------------------------------------------


def test_confidently_wrong_rate_counts_all_fail_in_the_top_decile():
    # 10 problems -> deciles of 1. The most-consensus problem is unsolved,
    # which is exactly the "confidently wrong" case.
    values = [0.1 * i for i in range(10)]
    solved = [0] + [1] * 9
    rate = confidently_wrong_rate(_rows(values, solved), "ast_node_diversity")
    assert rate["n_problems"] == 1
    assert rate["n_all_fail"] == 1
    assert rate["rate"] == pytest.approx(1.0)
    assert rate["task_ids"] == ["T/0"]


def test_confidently_wrong_rate_is_zero_when_top_consensus_always_solves():
    values = [0.1 * i for i in range(10)]
    solved = [1] * 5 + [0] * 5
    rate = confidently_wrong_rate(_rows(values, solved), "ast_node_diversity")
    assert rate["rate"] == 0.0


# ---------------------------------------------------------------------------
# build_ceiling_table (attenuation arithmetic)
# ---------------------------------------------------------------------------


def test_ceiling_is_the_geometric_mean_of_the_two_reliabilities():
    # sqrt(0.64 * 0.81) = 0.72; an observed r of -0.36 is half of it.
    reliabilities = {
        "pass_rate": {"reliability_mean": 0.81},
        "ast_node_diversity": {"reliability_mean": 0.64},
    }
    observed = {"ast_node_diversity": {"pearson_r": -0.36, "spearman_rho": -0.30}}
    table = build_ceiling_table(reliabilities, observed)
    entry = table["ast_node_diversity"]
    assert entry["ceiling"] == pytest.approx(0.72)
    assert entry["percent_of_ceiling"] == pytest.approx(50.0)


def test_ceiling_table_handles_a_metric_with_no_reliability_estimate():
    reliabilities = {"pass_rate": {"reliability_mean": 0.9}}
    observed = {"lexical_diversity": {"pearson_r": -0.5, "spearman_rho": -0.4}}
    entry = build_ceiling_table(reliabilities, observed)["lexical_diversity"]
    assert entry["ceiling"] is None
    assert entry["percent_of_ceiling"] is None


def test_ceiling_table_handles_a_constant_metric():
    reliabilities = {
        "pass_rate": {"reliability_mean": 0.9},
        "parse_failure_rate": {"reliability_mean": 0.5},
    }
    observed = {"parse_failure_rate": {"pearson_r": None, "spearman_rho": None}}
    entry = build_ceiling_table(reliabilities, observed)["parse_failure_rate"]
    assert entry["ceiling"] is None


# ---------------------------------------------------------------------------
# estimate_reliability's pair-distance cache
# ---------------------------------------------------------------------------


def test_cache_reconstruction_equals_compute_uncertainty():
    # The cache reimplements the cheap metrics as means over a precomputed
    # pair matrix. If that shortcut ever drifts from the canonical
    # definition, every reliability number computed from it is wrong — so
    # the equality is pinned here as well as self-checked at runtime.
    codes = [
        "def f(x):\n    return x + 1\n",
        "def f(x):\n    return x * 2\n",
        "def f(x):\n    total = 0\n    for i in range(x):\n        total += i\n    return total\n",
        "def f(x):\n    return\n",
    ]
    cache = build_problem_cache(codes, None, "full")
    reconstructed = metrics_from_cache(cache, range(len(codes)))
    canonical = compute_uncertainty(codes)
    for metric in DEFAULT_METRICS:
        assert reconstructed[metric] == pytest.approx(canonical[metric], abs=1e-9)


def test_cache_reconstruction_handles_unparseable_samples():
    codes = ["def f():\n    return 1\n", "def f(:\n  ???\n", "def f():\n    return 2\n"]
    cache = build_problem_cache(codes, None, "full")
    reconstructed = metrics_from_cache(cache, range(len(codes)))
    canonical = compute_uncertainty(codes)
    assert reconstructed["parse_failure_rate"] == pytest.approx(1 / 3)
    assert reconstructed["ast_node_diversity"] == pytest.approx(
        canonical["ast_node_diversity"], abs=1e-9
    )


def test_cache_subset_of_one_sample_scores_pairwise_metrics_zero():
    # Matches the canonical contract: a pairwise metric over fewer than two
    # samples is 0.0, not undefined.
    codes = ["def f():\n    return 1\n", "def f():\n    return 2\n"]
    cache = build_problem_cache(codes, None, "full")
    values = metrics_from_cache(cache, [0])
    assert values["ast_node_diversity"] == 0.0
    assert values["token_jaccard_diversity"] == 0.0
    assert values["lexical_diversity"] == 1.0
