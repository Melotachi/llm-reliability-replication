"""Tests for scripts/preregister_n_sweep.py.

The pre-registration's whole value is that its numbers were derived
mechanically and committed before the data existed, so the derivation itself
has to be pinned. These tests check the Spearman-Brown round trip, the
monotonicity the argument depends on, and the degenerate inputs that would
otherwise emit a silently wrong prediction.
"""

from __future__ import annotations

import math

import pytest

from scripts.preregister_n_sweep import (
    SOURCE_N,
    ceiling,
    predict_cell,
    project_reliability,
    single_sample_reliability,
)


def test_spearman_brown_round_trip():
    # Inverting a measured r_10 and projecting it back to 10 must return it.
    for r_10 in (0.5, 0.716, 0.823, 0.919):
        r_1 = single_sample_reliability(r_10, n=SOURCE_N)
        assert project_reliability(r_1, SOURCE_N) == pytest.approx(r_10)


def test_projected_reliability_increases_with_n():
    r_1 = single_sample_reliability(0.823)
    values = [project_reliability(r_1, n) for n in (10, 20, 30, 50, 100)]
    assert values == sorted(values)
    assert values[-1] < 1.0


def test_ceiling_matches_week7_reported_value():
    # cg/HE60 ast_node: r_xx = 0.8225, r_yy = 0.9018 -> 0.8613 in the report.
    assert ceiling(0.8225222938376103, 0.9018156156389731) == pytest.approx(
        0.8613, abs=5e-4
    )


def test_single_sample_reliability_rejects_degenerate_input():
    # A split-half reliability of 1.0 inverts to a zero denominator; the
    # formula must refuse rather than emit an infinite r_1.
    with pytest.raises(ValueError):
        single_sample_reliability(1.0)
    with pytest.raises(ValueError):
        single_sample_reliability(0.0)


def _fake_cell(metric_reliability: float, observed_r: float, pass_rate_rel: float):
    ceil_10 = ceiling(metric_reliability, pass_rate_rel)
    return (
        {"pass_rate": {"reliability_mean": pass_rate_rel}},
        {
            "ast_node_diversity": {
                "metric_reliability": metric_reliability,
                "observed_pearson_r": observed_r,
                "ceiling": ceil_10,
                "percent_of_ceiling": 100 * abs(observed_r) / ceil_10,
            }
        },
    )


def test_predicted_correlation_rises_with_n_and_stays_below_true_r():
    reliabilities, ceiling_table = _fake_cell(0.8225, -0.6243, 0.9018)
    out = predict_cell(reliabilities, ceiling_table, [10, 20, 50, 100])
    pred = out["ast_node_diversity"]

    by_n = pred["by_target_n"]
    predicted = [by_n[n]["predicted_abs_pearson_r"] for n in (10, 20, 50, 100)]
    assert predicted == sorted(predicted)
    # At N = 10 the prediction must reproduce the observed value it came from.
    assert predicted[0] == pytest.approx(abs(-0.6243), abs=1e-6)
    # The disattenuated correlation is the asymptote, never exceeded.
    assert all(p <= pred["implied_true_r"] + 1e-9 for p in predicted)
    assert by_n[100]["ceiling"] > by_n[10]["ceiling"]


def test_prediction_is_capped_at_one():
    # A near-perfect observed correlation must not project above |r| = 1.
    reliabilities, ceiling_table = _fake_cell(0.95, -0.97, 0.98)
    out = predict_cell(reliabilities, ceiling_table, [100])
    assert (
        out["ast_node_diversity"]["by_target_n"][100]["predicted_abs_pearson_r"] <= 1.0
    )


def test_unknown_metric_is_skipped_not_fatal():
    reliabilities = {"pass_rate": {"reliability_mean": 0.9}}
    out = predict_cell(reliabilities, {}, [50])
    assert out == {}


def test_ceiling_is_symmetric_and_bounded():
    assert ceiling(0.8, 0.9) == pytest.approx(ceiling(0.9, 0.8))
    assert ceiling(0.8, 0.9) == pytest.approx(math.sqrt(0.72))
    assert ceiling(1.0, 1.0) == pytest.approx(1.0)
