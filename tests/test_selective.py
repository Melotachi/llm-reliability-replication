"""Tests for src/selective.py (Week 7 risk-coverage / AUGRC analysis).

Style follows tests/test_uncertainty.py: the expected values are derived in
comments from small inputs, and the three reference points of the normalized
scale (oracle = 0, random = 1, inverted ranking > 1) are asserted directly
rather than compared against numbers from a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.selective import (
    coverage_at_risk,
    risk_coverage_curve,
    selective_prediction_metrics,
)


def test_risk_coverage_curve_hand_computed():
    # Ranked by confidence: labels 1, 0, 1, 0 -> cumulative errors 0,1,1,2.
    curve = risk_coverage_curve([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])
    assert curve["n"] == 4
    assert curve["base_error"] == pytest.approx(0.5)
    assert curve["coverage"] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert curve["selective_risk"] == pytest.approx([0.0, 0.5, 1 / 3, 0.5])
    assert curve["generalized_risk"] == pytest.approx([0.0, 0.25, 0.25, 0.5])


def test_risk_coverage_curve_reranks_by_confidence_not_input_order():
    # The two lowest-confidence samples are the failures, even though they
    # come first in input order.
    curve = risk_coverage_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert curve["selective_risk"][:2] == pytest.approx([0.0, 0.0])


def test_risk_coverage_curve_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        risk_coverage_curve([1, 0], [0.5])


def test_risk_coverage_curve_rejects_empty_input():
    with pytest.raises(ValueError):
        risk_coverage_curve([], [])


def test_coverage_at_risk_picks_the_largest_admissible_coverage():
    curve = risk_coverage_curve([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6])
    # Selective risk is 0, 0, 1/3, 0.5 -> at a 0.4 budget the best coverage
    # is 0.75, not 0.5: the curve is not monotone, and the *largest*
    # admissible coverage is what a deployment would actually choose.
    assert coverage_at_risk(curve, 0.4) == pytest.approx(0.75)


def test_coverage_at_risk_returns_zero_when_even_the_top_sample_fails():
    curve = risk_coverage_curve([0, 1, 1, 1], [0.9, 0.8, 0.7, 0.6])
    assert coverage_at_risk(curve, 0.05) == 0.0


def test_normalized_scores_are_zero_for_an_oracle_ranking():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    perfect = [0.9, 0.8, 0.7, 0.3, 0.2, 0.15, 0.1, 0.05]
    metrics = selective_prediction_metrics(y_true, perfect)
    assert metrics["norm_augrc"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["norm_aurc"] == pytest.approx(0.0, abs=1e-9)


def test_normalized_augrc_is_one_for_a_ranking_that_ignores_the_label():
    # Alternating labels down the confidence ranking is the discrete
    # equivalent of "confidence carries no information": the accepted set
    # holds the base error rate at (almost) every coverage.
    y_true = [1, 0] * 8
    confidence = np.linspace(1.0, 0.0, 16).tolist()
    metrics = selective_prediction_metrics(y_true, confidence)
    assert metrics["norm_augrc"] == pytest.approx(1.0, abs=0.15)


def test_inverted_ranking_scores_worse_than_random():
    # Every failure ranked above every success — the "worse than random"
    # regime the project's own cg/MBPP+ execution_only arm lands in.
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    inverted = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.6, 0.5]
    metrics = selective_prediction_metrics(y_true, inverted)
    assert metrics["norm_augrc"] > 1.0


def test_normalized_scores_undefined_when_every_sample_passes():
    # No failures means oracle and random coincide; the normalized score is
    # undefined rather than a division by ~0.
    metrics = selective_prediction_metrics([1, 1, 1], [0.9, 0.5, 0.1])
    assert metrics["norm_augrc"] is None
    assert metrics["norm_aurc"] is None
    assert metrics["base_error"] == 0.0


def test_metrics_are_invariant_under_monotone_recalibration():
    # Stated in src/selective.py and worth pinning: these are rank metrics,
    # so beta/Platt recalibration cannot move them.
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    raw = [0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.4]
    squashed = [p**3 for p in raw]  # strictly increasing transform
    assert selective_prediction_metrics(y_true, raw) == selective_prediction_metrics(
        y_true, squashed
    )


def test_coverage_targets_are_keyed_by_the_requested_budgets():
    metrics = selective_prediction_metrics(
        [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], risk_targets=(0.05, 0.5)
    )
    assert set(metrics["coverage_at_risk"]) == {"0.05", "0.50"}
