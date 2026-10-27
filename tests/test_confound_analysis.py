"""Tests for the Week 7 confound analysis in scripts/analyze_uncertainty.py.

Style follows tests/test_uncertainty.py: constructed inputs whose expected
answer is known analytically (a variable that is a pure function of the
control must have zero partial correlation; an independent one must keep
its raw correlation), rather than golden numbers from a previous run.

The multi-cell driver's argument parsing (scripts/compare_representations.py)
is tested here too, since it is the only public function in that script that
does not consist of printing.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_uncertainty import (
    _partial_correlation,
    compute_confound_analysis,
    extract_problem_table,
    load_run_prompts,
)
from scripts.compare_representations import parse_run_specs

_PROMPT = 'def f(x):\n    """Doc."""\n'


def _results(dataset: str = "humaneval") -> dict:
    """A minimal Section 8 results dict with three problems x two samples."""
    per_problem = []
    for index, (completion_a, completion_b, passed) in enumerate(
        [
            ("    return x\n", "    return x\n", [True, True]),
            ("    return x + 1\n", "    return x * 2\n", [True, False]),
            ("    return None\n", "    raise ValueError(x)\n", [False, False]),
        ]
    ):
        per_problem.append(
            {
                "task_id": f"Fake/{index}",
                "samples": [
                    {"code": _PROMPT + completion_a, "passed": passed[0]},
                    {"code": _PROMPT + completion_b, "passed": passed[1]},
                ],
            }
        )
    return {"config": {"dataset": dataset}, "per_problem": per_problem}


def _prompts() -> dict[str, str]:
    return {f"Fake/{i}": _PROMPT for i in range(3)}


# ---------------------------------------------------------------------------
# _partial_correlation
# ---------------------------------------------------------------------------


def test_partial_correlation_removes_a_pure_control_effect():
    # y is an exact linear function of the control, so once the control is
    # regressed out there is nothing left for x to correlate with.
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = 3.0 * control
    x = np.array([2.0, 9.0, 4.0, 7.0, 1.0, 5.0])
    r, _ = _partial_correlation(x, y, [control])
    assert r is None or r == pytest.approx(0.0, abs=1e-9)


def test_partial_correlation_keeps_an_independent_relationship():
    # x and y move together; the control is orthogonal noise, so the partial
    # correlation stays essentially the raw one (+1.0 here).
    control = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = 2.0 * x + 1.0
    r, p = _partial_correlation(x, y, [control])
    assert r == pytest.approx(1.0, abs=1e-6)
    assert p is not None and p < 0.05


def test_partial_correlation_rank_mode_matches_spearman_without_controls():
    # With a constant control (no information), the rank-mode partial must
    # reproduce the plain Spearman coefficient of a monotone-but-nonlinear pair.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x**3
    constant = np.zeros_like(x)
    r, _ = _partial_correlation(x, y, [constant], rank=True)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_partial_correlation_without_degrees_of_freedom_returns_no_p_value():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 1.0, 3.0])
    control = np.array([0.0, 1.0, 3.0])
    r, p = _partial_correlation(x, y, [control])
    assert p is None  # df = 3 - 2 - 1 = 0
    assert r is None or isinstance(r, float)


# ---------------------------------------------------------------------------
# compute_confound_analysis
# ---------------------------------------------------------------------------


def test_confound_analysis_unavailable_without_covariates():
    # No prompts passed => extract_problem_table adds no covariates => the
    # analysis must say so rather than silently reporting nothing.
    rows = extract_problem_table(_results(), include_codebert=False)
    analysis = compute_confound_analysis(rows)
    assert analysis["available"] is False
    assert "reason" in analysis


def test_confound_analysis_reports_baseline_and_partials():
    rows = extract_problem_table(_results(), include_codebert=False, prompts=_prompts())
    analysis = compute_confound_analysis(rows)
    assert analysis["available"] is True
    assert analysis["baseline_metric"] == "mean_completion_chars"
    # The free baseline is correlated with pass rate directly, with no
    # diversity metric involved -- that is the row a reviewer adds.
    baseline = analysis["covariate_baselines"]["mean_completion_chars"]
    assert set(baseline) >= {"pearson_r", "pearson_p", "spearman_rho", "mean"}
    entry = analysis["metrics"]["ast_node_diversity"]
    assert "raw_pearson_r" in entry
    for label in ("ctrl_completion_len", "ctrl_both_lengths", "ctrl_prompt_share"):
        assert label in entry


def test_confound_analysis_flags_constant_metrics_instead_of_correlating_them():
    rows = extract_problem_table(_results(), include_codebert=False, prompts=_prompts())
    # Every sample here parses, so parse_failure_rate is constant at 0.0.
    assert analysis_constant(rows, "parse_failure_rate")


def analysis_constant(rows: list[dict], metric: str) -> bool:
    """True if compute_confound_analysis marks `metric` as constant."""
    return compute_confound_analysis(rows)["metrics"][metric].get("constant") is True


# ---------------------------------------------------------------------------
# load_run_prompts
# ---------------------------------------------------------------------------


def test_load_run_prompts_without_a_dataset_field_returns_empty():
    assert load_run_prompts({"config": {}, "per_problem": []}) == {}


# ---------------------------------------------------------------------------
# compare_representations.parse_run_specs
# ---------------------------------------------------------------------------


def test_parse_run_specs_returns_label_path_pairs(tmp_path):
    results_file = tmp_path / "run.json"
    results_file.write_text("{}")
    assert parse_run_specs([f"cell_a={results_file}"]) == [("cell_a", results_file)]


def test_parse_run_specs_rejects_missing_equals(tmp_path):
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_run_specs([str(tmp_path / "run.json")])


def test_parse_run_specs_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        parse_run_specs([f"cell_a={tmp_path / 'absent.json'}"])


def test_parse_run_specs_requires_at_least_one_run():
    with pytest.raises(ValueError, match="at least one"):
        parse_run_specs([])
