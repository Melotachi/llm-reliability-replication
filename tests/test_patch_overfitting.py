"""Tests for scripts/evaluate_patch_overfitting.py (Week 9, Student B)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_patch_overfitting",
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "evaluate_patch_overfitting.py",
)
overfit = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(overfit)


def _unit(population: str, task_id: str, code: str, base: bool, plus: bool) -> dict:
    return {
        "population": population,
        "task_id": task_id,
        "code": code,
        "passed_base": base,
        "passed_plus": plus,
        "plus_failure_category": "" if plus else "assertion_error",
    }


def _pool(records: list[dict]) -> dict:
    return {
        "config": {"source_results": "results/src.json", "model": "m"},
        "records": records,
    }


def _record(
    task_id: str, *, final_passed: bool, candidates: list[tuple[str, bool]]
) -> dict:
    return {
        "task_id": task_id,
        "sample_index": 0,
        "initial_code": f"# initial {task_id}",
        "initial_failure_category": "assertion_error",
        "final_passed": final_passed,
        "final_code": candidates[-1][0] if final_passed else f"# initial {task_id}",
        "repair_model_calls": len(candidates),
        "trace": [
            {
                "iteration": 1,
                "candidates": [
                    {"candidate_index": i, "code": code, "passed": passed}
                    for i, (code, passed) in enumerate(candidates)
                ],
            }
        ],
    }


def test_wilson_interval_brackets_point_estimate():
    low, high = overfit._wilson_interval(5, 54)
    assert low < 5 / 54 < high
    assert overfit._wilson_interval(0, 0) is None


def test_summarize_population_counts_both_directions():
    units = [
        _unit("p", "T/1", "a", True, True),
        _unit("p", "T/2", "b", True, False),
        _unit("p", "T/3", "c", False, True),
        _unit("p", "T/4", "d", False, False),
    ]
    out = overfit.summarize_population(units)
    assert out["confusion"] == {
        "base_pass_plus_pass": 1,
        "base_pass_plus_fail": 1,
        "base_fail_plus_pass": 1,
        "base_fail_plus_fail": 1,
    }
    assert out["false_positive_rate"] == pytest.approx(0.5)
    assert out["base_fail_plus_pass_rate"] == pytest.approx(0.25)
    assert out["plus_failure_categories"] == {"assertion_error": 1}


def test_summarize_population_ignores_base_failures_in_the_rate():
    """The headline rate is P(plus-fail | base-pass), never a raw pass drop."""
    units = [_unit("p", "T/1", f"c{i}", False, False) for i in range(20)]
    units.append(_unit("p", "T/2", "ok", True, True))
    out = overfit.summarize_population(units)
    assert out["n_base_pass"] == 1
    assert out["false_positive_rate"] == 0.0


def test_contrast_reports_fisher_p_and_difference():
    a = overfit.summarize_population(
        [_unit("a", "T/1", "x", True, False)] + [_unit("a", "T/2", "y", True, True)] * 9
    )
    b = overfit.summarize_population([_unit("b", "T/3", "z", True, False)] * 10)
    out = overfit.contrast("a", a, "b", b)
    assert out["table"] == [[1, 9], [10, 0]]
    assert out["fisher_p"] < 0.01
    assert out["fpr_difference"] == pytest.approx(0.1 - 1.0)


def test_contrast_returns_none_when_a_population_has_no_base_passers():
    empty = overfit.summarize_population([_unit("a", "T/1", "x", False, False)])
    other = overfit.summarize_population([_unit("b", "T/2", "y", True, True)])
    assert overfit.contrast("a", empty, "b", other)["fisher_p"] is None


def test_collect_patch_units_covers_initial_selected_and_candidates():
    pool = _pool(
        [_record("T/1", final_passed=True, candidates=[("bad", False), ("good", True)])]
    )
    units = overfit.collect_patch_units(pool, excluded=())
    populations = sorted(u["population"] for u in units)
    assert populations == [
        "initial_failure",
        "patch_candidate",
        "patch_candidate",
        "patch_selected",
    ]
    assert all(
        u["passed_base"] is False for u in units if u["population"] == "initial_failure"
    )


def test_collect_patch_units_honours_exclusions():
    pool = _pool([_record("T/1", final_passed=True, candidates=[("good", True)])])
    assert overfit.collect_patch_units(pool, excluded=("T/1",)) == []


def test_collect_sibling_units_drops_the_repaired_coordinate():
    source = {
        "per_problem": [
            {
                "task_id": "T/1",
                "samples": [
                    {"code": "s0", "passed": False},
                    {"code": "s1", "passed": True},
                ],
            },
            {"task_id": "T/9", "samples": [{"code": "other", "passed": True}]},
        ]
    }
    units = overfit.collect_sibling_units(
        source, {"T/1"}, repaired_sample_index=0, excluded=()
    )
    assert [u["code"] for u in units] == ["s1"]


def test_selection_channel_is_structurally_zero_under_stop_on_first_pass():
    """One base-passing candidate per coordinate leaves ranking no choice."""
    pool = _pool(
        [_record("T/1", final_passed=True, candidates=[("bad", False), ("good", True)])]
    )
    scored = [
        _unit("patch_candidate", "T/1", "bad", False, False),
        _unit("patch_candidate", "T/1", "good", True, True),
        _unit("patch_selected", "T/1", "good", True, True),
    ]
    out = overfit.selection_channel(pool, scored, excluded=())
    assert out["n_decidable_coordinates"] == 0
    assert out["rate_returned_plus_failing"] is None


def test_selection_channel_flags_a_plus_failing_return_when_a_choice_existed():
    record = _record(
        "T/1", final_passed=True, candidates=[("overfit", True), ("correct", True)]
    )
    record["final_code"] = "overfit"
    scored = [
        _unit("patch_candidate", "T/1", "overfit", True, False),
        _unit("patch_candidate", "T/1", "correct", True, True),
    ]
    out = overfit.selection_channel(_pool([record]), scored, excluded=())
    assert out["n_decidable_coordinates"] == 1
    assert out["n_returned_plus_failing"] == 1
    assert out["rate_returned_plus_failing"] == pytest.approx(1.0)


def test_coordinate_labels_never_mark_plus_success_without_base_success():
    record = _record("T/1", final_passed=False, candidates=[("bad", False)])
    rows = overfit.coordinate_labels(_pool([record]), [], excluded=())
    assert rows[0]["repair_success_base"] is False
    assert rows[0]["repair_success_plus"] is False


def test_category_breakdown_rates_are_per_base_success():
    rows = [
        {
            "initial_failure_category": "assertion_error",
            "repair_success_base": True,
            "repair_success_plus": False,
        },
        {
            "initial_failure_category": "assertion_error",
            "repair_success_base": True,
            "repair_success_plus": True,
        },
        {
            "initial_failure_category": "syntax_error",
            "repair_success_base": False,
            "repair_success_plus": False,
        },
    ]
    out = overfit.category_breakdown(rows)
    assert out["assertion_error"]["false_positive_rate"] == pytest.approx(0.5)
    assert out["syntax_error"]["false_positive_rate"] is None
