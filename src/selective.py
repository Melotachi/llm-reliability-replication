"""Selective prediction: what a predictor's ranking is worth as a decision rule.

Student A, Week 7 (CLAUDE.md Section 10.1 extends naturally from Week 5's
calibration work). Every metric the project has reported so far — correlation,
F1, ROC-AUC, ECE — answers "is the signal good?" None answers the question a
supervisor or a reviewer actually asks: **if I deploy this, how much generated
code can I auto-accept, and at what error rate?**

Risk-coverage analysis answers exactly that. Rank samples by predicted
confidence, accept the top fraction (coverage), and measure the error rate
among the accepted ones (selective risk).

**AUGRC, not AURC.** Traub, Bungert, Lüth, Baumgartner, Maier-Hein,
Maier-Hein & Jaeger (NeurIPS 2024, arXiv 2407.01032) state five requirements
for multi-threshold selective-classification metrics, show that the standard
AURC violates them, and propose the Area Under the Generalized Risk Curve
instead: the joint probability of (failure AND acceptance), interpretable as
*the average risk of undetected failures*. The one-line difference is the
denominator — generalized risk divides the accumulated errors by the total n,
selective risk divides by the k accepted so far — and the paper reports that
switching "substantially changes metric rankings on 5 of 6 datasets". Both are
computed here, with AUGRC primary and AURC secondary, so any ranking
conclusion can state which metric it rests on.

Normalization follows the same paper's interpretation rather than a raw area:
a score of 0.0 is the oracle ranking (every failure sorted below every
success), 1.0 is a random ranking, and **above 1.0 is worse than random** —
which does occur in this project's data and is worth saying plainly.
"""

from __future__ import annotations

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

# Error budgets the coverage table is reported at. 5% is the strict operating
# point a "can I trust this without looking?" workflow needs; 10% is the loose
# one a human-in-the-loop review can absorb.
DEFAULT_RISK_TARGETS = (0.05, 0.10)


def risk_coverage_curve(y_true, y_prob) -> dict:
    """Rank by confidence and walk the accept-the-top-k curve.

    Args:
        y_true: Binary labels, 1 = the sample is correct (passes).
        y_prob: Predicted probability of the positive class. Used only as a
            ranking, so any monotone recalibration (e.g. Week 6's beta
            calibrator) leaves every number here unchanged — a natural
            reviewer question, worth stating explicitly.
    Returns:
        Dict with "coverage", "selective_risk", "generalized_risk" arrays
        (one entry per k = 1..n, as lists) plus "base_error" and "n".
    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if len(y_true) == 0:
        raise ValueError("selective prediction needs at least one observation")

    n = len(y_true)
    # Descending confidence; ties broken deterministically by original index
    # so the curve is reproducible (CLAUDE.md Constraint #4).
    order = np.lexsort((np.arange(n), -y_prob))
    errors = (y_true[order] == 0).astype(float)
    cumulative_errors = np.cumsum(errors)
    k = np.arange(1, n + 1, dtype=float)
    return {
        "n": int(n),
        "base_error": float(errors.mean()),
        "coverage": (k / n).tolist(),
        "selective_risk": (cumulative_errors / k).tolist(),
        "generalized_risk": (cumulative_errors / n).tolist(),
    }


def _oracle_and_random_areas(base_error: float, n: int) -> dict:
    """Reference areas for an oracle ranking and a random one.

    The oracle sorts every correct sample above every incorrect one; a random
    ranking has, in expectation, the base error rate at every coverage. Both
    are computed on the same k = 1..n grid the observed curve uses, so the
    normalization cannot be biased by a discretization mismatch.
    """
    k = np.arange(1, n + 1, dtype=float)
    n_correct = int(round((1.0 - base_error) * n))
    oracle_errors = np.concatenate(
        [np.zeros(n_correct), np.ones(n - n_correct)]
    ).cumsum()
    return {
        "aurc_oracle": float(np.mean(oracle_errors / k)),
        "augrc_oracle": float(np.mean(oracle_errors / n)),
        "aurc_random": float(base_error),
        "augrc_random": float(np.mean(base_error * k / n)),
    }


def coverage_at_risk(curve: dict, target: float) -> float:
    """Largest coverage whose selective risk stays within `target`.

    Args:
        curve: Output of risk_coverage_curve().
        target: Error budget, e.g. 0.05.
    Returns:
        Coverage in [0.0, 1.0]; 0.0 when even accepting the single
        most-confident sample exceeds the budget.
    """
    coverage = np.asarray(curve["coverage"], dtype=float)
    risk = np.asarray(curve["selective_risk"], dtype=float)
    admissible = risk <= target
    return float(coverage[admissible].max()) if admissible.any() else 0.0


def selective_prediction_metrics(
    y_true, y_prob, *, risk_targets: tuple[float, ...] = DEFAULT_RISK_TARGETS
) -> dict:
    """AURC, AUGRC, their normalized forms, and coverage at each risk target.

    Args:
        y_true: Binary labels, 1 = correct.
        y_prob: Predicted probability of the positive class (ranking only).
        risk_targets: Error budgets for the coverage columns.
    Returns:
        Dict with "n", "base_error", "aurc", "augrc", "norm_aurc",
        "norm_augrc" (0.0 = oracle, 1.0 = random, >1.0 = worse than
        random) and "coverage_at_risk" keyed by target.
    """
    curve = risk_coverage_curve(y_true, y_prob)
    aurc = float(np.mean(curve["selective_risk"]))
    augrc = float(np.mean(curve["generalized_risk"]))
    reference = _oracle_and_random_areas(curve["base_error"], curve["n"])

    def _normalize(observed: float, oracle: float, random: float) -> float | None:
        span = random - oracle
        # A degenerate label set (all correct or all incorrect) leaves no
        # gap between oracle and random, so the normalized score is
        # undefined rather than infinite.
        return None if abs(span) < 1e-12 else float((observed - oracle) / span)

    return {
        "n": curve["n"],
        "base_error": curve["base_error"],
        "aurc": aurc,
        "augrc": augrc,
        "norm_aurc": _normalize(
            aurc, reference["aurc_oracle"], reference["aurc_random"]
        ),
        "norm_augrc": _normalize(
            augrc, reference["augrc_oracle"], reference["augrc_random"]
        ),
        "coverage_at_risk": {
            f"{target:.2f}": coverage_at_risk(curve, target) for target in risk_targets
        },
    }
