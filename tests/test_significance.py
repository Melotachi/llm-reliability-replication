"""Tests for the IST manuscript's significance tests (scripts/analyze_significance.py).

Each test pins a property the manuscript's reading depends on: the AUC helper
agrees with sklearn, DeLong agrees with a brute-force reference on a known
case, the cluster bootstrap keeps groups intact and is seeded, the Wilcoxon
selection test drops uninformative problems, and the Monte Carlo rank p
counts ties as extreme.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts.analyze_significance import (
    cluster_bootstrap_delta_auc,
    delong_paired,
    monte_carlo_rank_p,
    roc_auc,
    poisson_binomial_pmf,
    selection_vs_uniform_draw,
)


def test_roc_auc_matches_sklearn_with_ties() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.integers(0, 5, size=200).astype(float)
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_roc_auc_single_class_is_nan() -> None:
    assert math.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_delong_identical_scores_gives_no_difference() -> None:
    y = [0, 0, 1, 1, 0, 1]
    s = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]
    result = delong_paired(y, s, s)
    assert result["delta"] == 0.0
    assert result["p"] == 1.0


def test_delong_variance_matches_reference_formula() -> None:
    # Brute-force DeLong: placement values from pairwise comparisons.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=60).astype(bool)
    a = rng.normal(size=60) + y
    b = rng.normal(size=60) + 0.3 * y

    def placements(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, neg = s[y], s[~y]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        return psi.mean(axis=1), psi.mean(axis=0)

    v10a, v01a = placements(a)
    v10b, v01b = placements(b)
    var = (
        np.var(v10a - v10b, ddof=1) / y.sum()
        + np.var(v01a - v01b, ddof=1) / (~y).sum()
    )
    result = delong_paired(y, a, b)
    assert result["se"] == pytest.approx(math.sqrt(var))
    assert result["auc_a"] == pytest.approx(roc_auc_score(y, a))


def test_cluster_bootstrap_is_seeded_and_detects_a_clear_gap() -> None:
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(40), 10)
    y = rng.integers(0, 2, size=400)
    good = y + rng.normal(scale=0.5, size=400)
    noise = rng.normal(size=400)
    first = cluster_bootstrap_delta_auc(y, good, noise, groups, n_boot=500, seed=7)
    second = cluster_bootstrap_delta_auc(y, good, noise, groups, n_boot=500, seed=7)
    assert first == second
    assert first["delta"] > 0.3
    assert first["p"] == pytest.approx(2 / 501)
    assert first["ci_low"] > 0


def test_cluster_bootstrap_identical_arms_is_not_significant() -> None:
    groups = np.repeat(np.arange(20), 5)
    y = np.tile([0, 1, 0, 1, 1], 20)
    s = np.linspace(0, 1, 100)
    result = cluster_bootstrap_delta_auc(y, s, s, groups, n_boot=200, seed=0)
    assert result["delta"] == 0.0
    assert result["p"] == 1.0


def test_poisson_binomial_pmf_matches_binomial_and_sums_to_one() -> None:
    pmf = poisson_binomial_pmf([0.3] * 6)
    expected = [math.comb(6, k) * 0.3**k * 0.7 ** (6 - k) for k in range(7)]
    assert pmf == pytest.approx(expected)
    assert poisson_binomial_pmf([0.1, 0.9, 0.5, 0.0, 1.0]).sum() == pytest.approx(1.0)


def test_selection_test_ignores_deterministic_problems() -> None:
    passed = [True, False, True, True, False, True]
    rates = [1.0, 0.0, 0.2, 0.3, 0.4, 0.1]
    result = selection_vs_uniform_draw(passed, rates)
    assert result["n_informative"] == 4
    assert result["observed_passes"] == 4
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert 0 < result["p"] < 1


def test_selection_test_holds_its_size_at_low_pass_rates() -> None:
    # The reason Wilcoxon was rejected: at low pass rates its null is
    # asymmetric. The exact mid-p test must stay near nominal size.
    rng = np.random.default_rng(3)
    rates = rng.choice([0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 0.3, 0.6], size=150)
    rejections = sum(
        selection_vs_uniform_draw(rng.random(rates.size) < rates, rates)["p"] < 0.05
        for _ in range(2000)
    )
    assert rejections / 2000 < 0.065


def test_monte_carlo_rank_p_counts_ties_as_extreme() -> None:
    draws = [5, 6, 7, 7, 8]
    assert monte_carlo_rank_p(7, draws, tail="lower") == pytest.approx(5 / 6)
    assert monte_carlo_rank_p(7, draws, tail="upper") == pytest.approx(4 / 6)
    assert monte_carlo_rank_p(100, draws, tail="upper") == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        monte_carlo_rank_p(1, draws, tail="both")
