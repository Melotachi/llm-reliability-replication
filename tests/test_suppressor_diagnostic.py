"""Tests for scripts/diagnose_suppressor.py (Week 7 suppressor re-interpretation)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnose_suppressor import (
    VIF_CEILING,
    analyze_cell,
    parse_run_specs,
    variance_inflation_factors,
)


def test_vif_is_one_for_orthogonal_columns():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    factors = variance_inflation_factors(matrix, ["a", "b"])
    assert factors["a"] == pytest.approx(1.0, abs=1e-6)
    assert factors["b"] == pytest.approx(1.0, abs=1e-6)


def test_vif_flags_near_duplicate_columns():
    rng = np.random.default_rng(0)
    base = rng.normal(size=40)
    matrix = np.column_stack([base, base + rng.normal(scale=0.01, size=40)])
    factors = variance_inflation_factors(matrix, ["a", "a_copy"])
    # Two columns that are the same variable twice are collinearity, which is
    # exactly what VIF_CEILING exists to separate from suppression.
    assert min(factors.values()) > VIF_CEILING


def _rows_with_constructed_suppressor(n: int = 60) -> list[dict]:
    """Rows where the target is signal plus a nuisance the partner also carries.

    pass_rate is driven by -signal. The target is signal + nuisance, so its raw
    correlation is negative but diluted; residualizing it on the partner (which
    carries the same nuisance) removes the nuisance and, because the partner
    also removes some of the shared signal, leaves a positive remainder. That is
    the suppressor signature the diagnostic looks for.
    """
    rng = np.random.default_rng(7)
    signal = rng.normal(size=n)
    nuisance = rng.normal(size=n)
    pass_rate = -signal + 0.05 * rng.normal(size=n)
    rows = []
    for index in range(n):
        rows.append(
            {
                "task_id": f"Task/{index}",
                "pass_rate": float(pass_rate[index]),
                "token_jaccard_diversity": float(0.3 * signal[index] + nuisance[index]),
                "ast_node_diversity": float(signal[index] + 0.9 * nuisance[index]),
            }
        )
    return rows


def test_analyze_cell_reports_the_suppressor_signature():
    rows = _rows_with_constructed_suppressor()
    analysis = analyze_cell(rows, "token_jaccard_diversity", ["ast_node_diversity"])

    assert analysis["available"] is True
    assert analysis["zero_order"]["token_jaccard_diversity"]["pearson_r"] < 0
    assert analysis["residual_correlations"]["resid_on_ast"]["pearson_r"] > 0
    # The reverse residualization must keep its sign, or the flip would be an
    # artefact of residualizing anything at all.
    assert analysis["sanity_controls"]["ast_resid_on_target"]["pearson_r"] < 0
    assert analysis["verdict"].startswith("suppressor")


def test_analyze_cell_reports_unavailable_when_a_feature_is_constant():
    rows = [
        {"task_id": "T/0", "pass_rate": 0.0, "a": 1.0, "b": 3.0},
        {"task_id": "T/1", "pass_rate": 1.0, "a": 1.0, "b": 4.0},
    ]
    analysis = analyze_cell(rows, "a", ["b"])
    assert analysis["available"] is False
    assert "constant" in analysis["reason"]


def test_parse_run_specs_rejects_a_malformed_spec():
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_run_specs(["results/whatever.json"])
