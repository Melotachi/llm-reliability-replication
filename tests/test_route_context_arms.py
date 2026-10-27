"""Tests for the Week 10 context-arm router (Objective 1 features -> repair arm)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.route_context_arms import (
    DEFAULT_ARM,
    ROUTED_ARM,
    build_router_rows,
    category_router,
    nested_routing,
    out_of_fold_scores,
    random_routing_control,
    reference_points,
    routing_sweep,
    univariate_aucs,
)


def _source(n_problems: int = 6, n_samples: int = 3) -> dict:
    """Generation run whose sample 0 always fails, so every problem is a coordinate."""
    problems = []
    for index in range(n_problems):
        samples = [{"passed": False, "failure_category": "assertion_error"}]
        for sibling in range(n_samples - 1):
            samples.append(
                {
                    "passed": sibling == 0 and index % 2 == 0,
                    "failure_category": (
                        "" if sibling == 0 and index % 2 == 0 else "type_error"
                    ),
                }
            )
        problems.append(
            {
                "task_id": f"UNIT/{index}",
                "samples": samples,
                "uncertainty": {
                    "lexical_diversity": 1.0,
                    "token_jaccard_diversity": 0.1 * index,
                    "token_cosine_diversity": 0.05 * index,
                    "ast_node_diversity": 0.2 * index,
                    "tree_edit_distance_diversity": 0.15 * index,
                    "pairwise_distance": 10.0 * index,
                    "parse_failure_rate": 0.0,
                },
            }
        )
    return {"run_id": "unit_source", "per_problem": problems}


def _predictor(source: dict) -> dict:
    probabilities, labels = [], []
    for problem in source["per_problem"]:
        for sample_index, sample in enumerate(problem["samples"]):
            labels.append(int(sample["passed"]))
            probabilities.append(0.1 * sample_index + 0.2)
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


def _arms(source: dict, outcomes: dict[str, set[int]]) -> dict:
    """`load_arms`-shaped structure, bypassing file IO."""
    arms = {}
    for mode, repaired in outcomes.items():
        records = {}
        for index, problem in enumerate(source["per_problem"]):
            records[problem["task_id"]] = {
                "task_id": problem["task_id"],
                "sample_index": 0,
                "initial_code": f"def unit_{index}():\n    return {index}\n",
                "initial_failure_category": "assertion_error",
                "final_passed": index in repaired,
            }
        arms[mode] = {"run": {}, "path": f"unit_{mode}.json", "records": records}
    return arms


def _rows(outcomes: dict[str, set[int]], n_problems: int = 6) -> list[dict]:
    source = _source(n_problems=n_problems)
    return build_router_rows(
        source, _predictor(source), _arms(source, outcomes), "base"
    )


def test_build_router_rows_marks_context_only_coordinates():
    rows = _rows({DEFAULT_ARM: {0, 1}, ROUTED_ARM: {1, 2}, "error": {5}})
    flags = {row["task_id"]: row["context_only"] for row in rows}
    assert flags["UNIT/2"] is True  # routed arm rescues it, default does not
    assert flags["UNIT/5"] is True  # a third arm counts too
    assert flags["UNIT/1"] is False  # the default arm already solves it
    assert flags["UNIT/0"] is False


def test_build_router_rows_reads_no_outcome_into_the_features():
    """Two arm assignments over identical source data give identical features."""
    left = _rows({DEFAULT_ARM: set(), ROUTED_ARM: {0, 1, 2}})
    right = _rows({DEFAULT_ARM: {0, 1, 2}, ROUTED_ARM: set()})
    assert [row["features"] for row in left] == [row["features"] for row in right]


def test_build_router_rows_requires_both_routing_arms():
    source = _source()
    with pytest.raises(ValueError, match="missing"):
        build_router_rows(
            source, _predictor(source), _arms(source, {DEFAULT_ARM: set()}), "base"
        )


def test_reference_points_counts_oracle_and_headroom():
    rows = _rows({DEFAULT_ARM: {0}, ROUTED_ARM: {1, 2}, "error": {3}})
    references = reference_points(rows)
    assert references["n_coordinates"] == 6
    assert references["per_arm_passes"] == {DEFAULT_ARM: 1, ROUTED_ARM: 2, "error": 1}
    assert references["oracle_union_passes"] == 4
    assert references["two_arm_oracle_passes"] == 3
    assert references["headroom_routed_arm_only"] == 2
    assert references["headroom_context_only"] == 3


def test_routing_sweep_is_monotone_in_gains_and_losses():
    rows = _rows({DEFAULT_ARM: {0, 1}, ROUTED_ARM: {1, 2}})
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    sweep = routing_sweep(rows, scores)
    assert sweep[0]["passes"] == 2  # the default arm alone
    # Highest-scoring coordinate is UNIT/0: routed arm loses it.
    assert sweep[1]["passes"] == 1 and sweep[1]["lost"] == 1
    # UNIT/1 passes in both arms; UNIT/2 is a gain.
    assert sweep[3]["passes"] == 2 and sweep[3]["gained"] == 1
    assert sweep[-1]["k"] == len(rows)
    assert sweep[-1]["passes"] == 2  # routing everything = the routed arm's total


def test_routing_sweep_end_equals_routed_arm_total():
    rows = _rows({DEFAULT_ARM: {0}, ROUTED_ARM: {1, 2, 3}})
    scores = np.linspace(1.0, 0.0, len(rows))
    sweep = routing_sweep(rows, scores)
    assert sweep[-1]["passes"] == 3


def test_random_routing_control_reports_dominance_against_a_target():
    rows = _rows({DEFAULT_ARM: {0, 1}, ROUTED_ARM: {2, 3}})
    control = random_routing_control(rows, 6, n_seeds=5, seed=42, target=2)
    # Routing every coordinate is deterministic: the routed arm's total.
    assert control["min"] == control["max"] == 2
    assert control["fraction_matching_or_beating_target"] == 1.0
    assert control["fraction_beating_target"] == 0.0


def test_random_routing_control_rejects_out_of_range_k():
    rows = _rows({DEFAULT_ARM: set(), ROUTED_ARM: set()})
    with pytest.raises(ValueError, match="between zero"):
        random_routing_control(rows, len(rows) + 1, n_seeds=2, seed=1)


def test_random_routing_control_is_seed_reproducible():
    rows = _rows({DEFAULT_ARM: {0, 4}, ROUTED_ARM: {1, 2}})
    first = random_routing_control(rows, 3, n_seeds=20, seed=7)
    second = random_routing_control(rows, 3, n_seeds=20, seed=7)
    assert first == second


def test_out_of_fold_scores_are_one_per_row_and_probabilities():
    rows = _rows({DEFAULT_ARM: set(), ROUTED_ARM: {0, 2, 4}}, n_problems=12)
    features = np.array([row["features"] for row in rows], dtype=float)
    labels = np.array([int(row["context_only"]) for row in rows], dtype=int)
    scores = out_of_fold_scores(features, labels, n_splits=3, seed=42)
    assert scores.shape == (len(rows),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_nested_routing_never_reads_the_held_out_coverage():
    """The nested total is a real held-out number, so it may lose to the default."""
    rows = _rows({DEFAULT_ARM: {1, 3}, ROUTED_ARM: {0, 2, 4}}, n_problems=12)
    features = np.array([row["features"] for row in rows], dtype=float)
    labels = np.array([int(row["context_only"]) for row in rows], dtype=int)
    nested = nested_routing(rows, features, labels, n_splits=3, seed=42)
    assert nested["default_arm_passes"] == 2
    assert 0 <= nested["passes"] <= len(rows)
    assert len(nested["fold_coverages"]) == 3


def test_category_router_routes_only_categories_where_the_context_arm_wins():
    rows = _rows({DEFAULT_ARM: {0, 1}, ROUTED_ARM: {0, 1}})
    routed = category_router(rows)
    assert routed["categories_routed"] == []  # a tie is not a win
    assert routed["passes_in_sample"] == 2

    rows = _rows({DEFAULT_ARM: {0}, ROUTED_ARM: {0, 1, 2}})
    routed = category_router(rows)
    assert routed["categories_routed"] == ["assertion_error"]
    assert routed["passes_in_sample"] == 3


def test_univariate_aucs_drop_constant_columns_and_sort_by_deviation():
    rows = _rows({DEFAULT_ARM: set(), ROUTED_ARM: {0, 1, 2}})
    features = np.array([row["features"] for row in rows], dtype=float)
    labels = np.array([int(row["context_only"]) for row in rows], dtype=int)
    scored = univariate_aucs(rows, features, labels)
    names = [entry["feature"] for entry in scored]
    assert "lexical_diversity" not in names  # constant 1.0 in the fixture
    assert "category_assertion_error" not in names  # constant 1.0 too
    deviations = [entry["abs_deviation"] for entry in scored]
    assert deviations == sorted(deviations, reverse=True)
