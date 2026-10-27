"""Tests for scripts/analyze_capability_sweep.py.

Two things carry the weight here and are pinned accordingly: the pre-registered
fits (a scoring bug would silently turn a failed prediction into a passed one),
and the protocol check's distinction between "these cells conflict" and "an
older run never recorded this key" — collapsing those would either cry wolf or
hide a genuine confound.
"""

from __future__ import annotations

import pytest

from scripts.analyze_capability_sweep import (
    P2_SPREAD_BAND,
    P4_PARSE_FAILURE_CEILING,
    PREREGISTERED,
    check_protocol_match,
    observed_quantity,
    score_predictions,
    with_derived_task_ids_hash,
)


def _cell(run_id, model, pass_at_1, *, config=None, rho=-0.6, wrong=0.1, spread=0.5):
    return {
        "run_id": run_id,
        "model": model,
        "config": config or {},
        "pass_at_1": pass_at_1,
        "n_solved": 30,
        "n_problems": 60,
        "correlations": {"ast_node_diversity": {"pearson_r": rho, "spearman_rho": rho}},
        "consensus": {
            "feature": "ast_node_diversity",
            "quartiles": None,
            "spread": spread,
            "confidently_wrong": {"n_problems": 6, "n_all_fail": 1, "rate": wrong},
        },
        "mean_parse_failure_rate": 0.001,
    }


def test_protocol_conflict_is_reported_when_two_cells_disagree():
    cells = [
        _cell("a", "m1", 0.2, config={"temperature": 0.7, "seed": 42}),
        _cell("b", "m2", 0.4, config={"temperature": 0.3, "seed": 42}),
    ]
    conflicts, unrecorded = check_protocol_match(cells)
    assert "temperature" in conflicts
    assert unrecorded == {}


def test_missing_key_is_unrecorded_not_a_conflict():
    # The older run predates --temperature being persisted: unknown, not
    # known-different. Reporting it as a conflict would be a false alarm.
    cells = [
        _cell("old", "m1", 0.2, config={"seed": 42}),
        _cell("new", "m2", 0.4, config={"temperature": 0.7, "seed": 42}),
    ]
    conflicts, unrecorded = check_protocol_match(cells)
    assert conflicts == {}
    assert "temperature" in unrecorded
    assert unrecorded["temperature"] == {"old": None, "new": 0.7}


def test_matching_protocol_reports_neither():
    cells = [
        _cell("a", "m1", 0.2, config={"temperature": 0.7, "seed": 42, "n_samples": 10}),
        _cell("b", "m2", 0.4, config={"temperature": 0.7, "seed": 42, "n_samples": 10}),
    ]
    assert check_protocol_match(cells) == ({}, {})


def test_p1_fit_reproduces_the_two_points_it_was_derived_from():
    # The pre-registration fitted P1 through (0.230, 0.167) and (0.495, 0.000).
    fit = PREREGISTERED["confidently_wrong_rate"]
    assert fit["intercept"] + fit["slope"] * 0.230 == pytest.approx(0.167, abs=1e-9)
    assert fit["intercept"] + fit["slope"] * 0.495 == pytest.approx(0.0, abs=1e-3)


def test_p3_fit_reproduces_the_two_points_it_was_derived_from():
    fit = PREREGISTERED["ast_abs_spearman"]
    assert fit["intercept"] + fit["slope"] * 0.230 == pytest.approx(0.56, abs=5e-3)
    assert fit["intercept"] + fit["slope"] * 0.495 == pytest.approx(0.74, abs=5e-3)


def test_p1_prediction_is_floored_at_zero():
    # A very capable model would otherwise get a negative predicted rate, which
    # is not a possible value of a fraction.
    cells = [_cell("x", "Some/Model", 0.95, wrong=0.0)]
    scored = score_predictions(cells, known_run_ids=set())
    assert scored["confidently_wrong_rate"]["rows"][0]["predicted"] == 0.0


def test_cells_the_fit_came_from_are_excluded_from_the_held_out_count():
    # A two-point fit reproduces its own points exactly; counting them as
    # confirmations would be circular.
    cells = [
        _cell("stored", "Salesforce/codegen-350M-mono", 0.230, wrong=0.167),
        _cell("new", "Qwen/Qwen2.5-Coder-0.5B", 0.350, wrong=0.090),
    ]
    scored = score_predictions(cells, known_run_ids={"stored"})
    block = scored["confidently_wrong_rate"]
    assert block["n_held_out"] == 1
    assert block["n_held_out_within_tolerance"] == 1
    flags = {
        row["model"]: row["fit_was_derived_from_this_cell"] for row in block["rows"]
    }
    assert flags["Salesforce/codegen-350M-mono"] is True
    assert flags["Qwen/Qwen2.5-Coder-0.5B"] is False


def test_tolerance_decides_within_tolerance():
    fit = PREREGISTERED["ast_abs_spearman"]
    inside = fit["intercept"] + fit["slope"] * 0.35 + fit["tolerance"] * 0.5
    outside = fit["intercept"] + fit["slope"] * 0.35 + fit["tolerance"] * 2.0
    for value, expected in ((inside, True), (outside, False)):
        cells = [_cell("x", "Some/Model", 0.35, rho=-value)]
        scored = score_predictions(cells, known_run_ids=set())
        assert scored["ast_abs_spearman"]["rows"][0]["within_tolerance"] is expected


def test_p2_band_membership_and_trend():
    cells = [
        _cell("a", "m1", 0.20, spread=0.60),
        _cell("b", "m2", 0.35, spread=0.50),
        _cell("c", "m3", 0.50, spread=0.55),
    ]
    block = score_predictions(cells, known_run_ids=set())["P2_spread_band"]
    assert all(row["in_band"] for row in block["rows"])
    assert block["trend_vs_pass_at_1"] is not None
    assert block["band"] == list(P2_SPREAD_BAND)


def test_p2_trend_is_absent_with_fewer_than_three_cells():
    cells = [_cell("a", "m1", 0.20, spread=0.60), _cell("b", "m2", 0.50, spread=0.50)]
    block = score_predictions(cells, known_run_ids=set())["P2_spread_band"]
    assert block["trend_vs_pass_at_1"] is None


def test_p2_flags_a_spread_outside_the_committed_band():
    cells = [_cell("a", "m1", 0.20, spread=0.25)]
    block = score_predictions(cells, known_run_ids=set())["P2_spread_band"]
    assert block["rows"][0]["in_band"] is False


def test_p4_ceiling_verdict():
    cells = [_cell("a", "m1", 0.20)]
    cells[0]["mean_parse_failure_rate"] = P4_PARSE_FAILURE_CEILING * 3
    block = score_predictions(cells, known_run_ids=set())["P4_parse_failure"]
    assert block["rows"][0]["below_ceiling"] is False


def test_p5_orders_by_measured_pass_at_1_not_input_order():
    cells = [
        _cell("big", "Big/Model", 0.10),
        _cell("small", "Small/Model", 0.60),
    ]
    rows = score_predictions(cells, known_run_ids=set())["P5_capability_order"]["rows"]
    assert [row["model"] for row in rows] == ["Big/Model", "Small/Model"]
    assert rows[0]["pass_at_1"] < rows[1]["pass_at_1"]


def test_observed_quantity_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="unknown pre-registered quantity"):
        observed_quantity(_cell("a", "m", 0.2), "not_a_prediction")


def test_observed_quantity_returns_absolute_spearman():
    assert observed_quantity(_cell("a", "m", 0.2, rho=-0.73), "ast_abs_spearman") == (
        pytest.approx(0.73)
    )


def _results(task_ids, **config):
    return {
        "config": config,
        "per_problem": [{"task_id": task_id} for task_id in task_ids],
    }


def test_derived_task_ids_hash_separates_different_problem_sets():
    """Week 11: the allowlist path does not identify the cell; the ids do."""
    first = with_derived_task_ids_hash(
        _results(["BigCodeBench/1", "BigCodeBench/2"], task_ids_file="cell.txt")
    )
    second = with_derived_task_ids_hash(
        _results(["BigCodeBench/2", "BigCodeBench/3"], task_ids_file="cell.txt")
    )
    assert first["task_ids_sha256"] != second["task_ids_sha256"]

    cells = [
        {"run_id": "a", "config": first},
        {"run_id": "b", "config": second},
    ]
    conflicts, _ = check_protocol_match(cells)
    assert "task_ids_sha256" in conflicts


def test_derived_task_ids_hash_does_not_override_a_recorded_one():
    """A run that recorded the hash keeps it, even if per_problem is a subset."""
    config = with_derived_task_ids_hash(
        _results(["HumanEval/0"], task_ids_sha256="recorded")
    )
    assert config["task_ids_sha256"] == "recorded"


def test_derived_task_ids_hash_matches_the_recorded_one_for_the_same_ids():
    """Recovered and recorded hashes must agree, or mixing rungs is a false alarm."""
    import hashlib

    ids = ["HumanEval/0", "HumanEval/1"]
    recorded = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    assert with_derived_task_ids_hash(_results(ids))["task_ids_sha256"] == recorded
