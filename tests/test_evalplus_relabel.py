"""Tests for scripts/rescore_evalplus_labels.py (Week 7 EvalPlus relabel)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rescore_evalplus_labels import (
    relabelled_run,
    rescore_run,
    summarize_cell,
    trend_across_rungs,
)


def _sample(code: str, passed: bool) -> dict:
    return {"code": code, "passed": passed, "error_message": "", "failure_category": ""}


def _run() -> dict:
    return {
        "run_id": "demo",
        "timestamp": "2026-08-27T00:00:00",
        "config": {"model": "m", "n_problems": 2, "n_samples": 2},
        "summary": {},
        "per_problem": [
            {
                "task_id": "HumanEval/0",
                "samples": [
                    _sample("def f():\n    return 1\n", True),
                    _sample("x", False),
                ],
                "initial_pass_rate": 0.5,
                "n_initial_pass": 1,
                "pass_at_1": 0.5,
                "pass_at_5": 1.0,
                "failure_types": ["syntax_error"],
            },
            {
                "task_id": "HumanEval/1",
                "samples": [_sample("def g():\n    return 2\n", True)] * 2,
                "initial_pass_rate": 1.0,
                "n_initial_pass": 2,
                "pass_at_1": 1.0,
                "pass_at_5": 1.0,
                "failure_types": [],
            },
        ],
    }


def test_rescore_run_relabels_and_skips_unavailable_problems():
    # Only problem 0 has a plus suite; its first sample passes it, the second
    # (unparseable) cannot.
    tests = {"HumanEval/0": "def check(c):\n    assert c() == 1\n\ncheck(f)\n"}
    out = rescore_run(_run(), tests, timeout_s=10.0, workers=2, excluded_tasks=())
    assert out["skipped"] == ["HumanEval/1"]
    assert [s["passed_plus"] for s in out["problems"][0]["samples"]] == [True, False]
    assert [s["passed_base"] for s in out["problems"][0]["samples"]] == [True, False]


def test_rescore_run_honors_excluded_tasks():
    tests = {"HumanEval/0": "check = lambda c: None\ncheck(f)\n"}
    out = rescore_run(
        _run(), tests, timeout_s=10.0, workers=1, excluded_tasks=("HumanEval/0",)
    )
    assert out["problems"] == []
    assert set(out["skipped"]) == {"HumanEval/0", "HumanEval/1"}


def test_summarize_cell_reports_two_directional_confusion():
    rescored = {
        "skipped": [],
        "problems": [
            {
                "task_id": "HumanEval/0",
                "samples": [
                    {
                        "passed_base": True,
                        "passed_plus": True,
                        "plus_failure_category": "",
                    },
                    {
                        "passed_base": True,
                        "passed_plus": False,
                        "plus_failure_category": "assertion_error",
                    },
                    {
                        "passed_base": False,
                        "passed_plus": True,
                        "plus_failure_category": "",
                    },
                    {
                        "passed_base": False,
                        "passed_plus": False,
                        "plus_failure_category": "timeout",
                    },
                ],
            }
        ],
    }
    summary = summarize_cell(rescored)
    assert summary["confusion"] == {
        "base_pass_plus_pass": 1,
        "base_pass_plus_fail": 1,
        "base_fail_plus_pass": 1,
        "base_fail_plus_fail": 1,
    }
    # Headline is P(plus-fail | base-pass), not the pass@1 delta.
    assert summary["false_positive_rate"] == pytest.approx(0.5)
    # Only base-pass failures feed the category counts (the Stroebl population).
    assert summary["plus_failure_categories"] == {"assertion_error": 1}
    assert summary["mean_pass_at_1_base"] == pytest.approx(0.5)
    assert summary["mean_pass_at_1_plus"] == pytest.approx(0.5)


def test_relabelled_run_keeps_schema_and_recomputes_aggregates():
    rescored = {
        "skipped": ["HumanEval/1"],
        "problems": [
            {
                "task_id": "HumanEval/0",
                "samples": [
                    {
                        "passed_base": True,
                        "passed_plus": False,
                        "plus_failure_category": "timeout",
                    },
                    {
                        "passed_base": False,
                        "passed_plus": False,
                        "plus_failure_category": "syntax_error",
                    },
                ],
            }
        ],
    }
    out = relabelled_run(_run(), rescored, "results/original.json")
    assert out["config"]["label_source"] == "evalplus_humanevalplus"
    assert out["config"]["relabel_source"] == "results/original.json"
    assert out["config"]["n_problems"] == 1
    problem = out["per_problem"][0]
    assert [s["passed"] for s in problem["samples"]] == [False, False]
    assert problem["pass_at_1"] == 0.0
    assert problem["n_initial_pass"] == 0
    assert problem["failure_types"] == ["syntax_error", "timeout"]
    assert out["summary"]["mean_pass_at_1"] == 0.0


def test_relabelled_run_does_not_mutate_input():
    run = _run()
    rescored = {
        "skipped": [],
        "problems": [
            {
                "task_id": "HumanEval/0",
                "samples": [
                    {
                        "passed_base": True,
                        "passed_plus": False,
                        "plus_failure_category": "timeout",
                    },
                    {
                        "passed_base": False,
                        "passed_plus": False,
                        "plus_failure_category": "syntax_error",
                    },
                ],
            },
            {
                "task_id": "HumanEval/1",
                "samples": [
                    {
                        "passed_base": True,
                        "passed_plus": True,
                        "plus_failure_category": "",
                    },
                    {
                        "passed_base": True,
                        "passed_plus": True,
                        "plus_failure_category": "",
                    },
                ],
            },
        ],
    }
    relabelled_run(run, rescored, "results/original.json")
    assert run["per_problem"][0]["samples"][0]["passed"] is True


def test_trend_across_rungs_orders_and_correlates():
    cells = [
        {
            "model": "a",
            "mean_pass_at_1_base": 0.2,
            "false_positive_rate": 0.30,
            "relative_pass_at_1_drop": 0.30,
        },
        {
            "model": "b",
            "mean_pass_at_1_base": 0.4,
            "false_positive_rate": 0.20,
            "relative_pass_at_1_drop": 0.20,
        },
        {
            "model": "c",
            "mean_pass_at_1_base": 0.6,
            "false_positive_rate": 0.10,
            "relative_pass_at_1_drop": 0.10,
        },
    ]
    trend = trend_across_rungs(cells)
    assert trend["capability_ordered_models"] == ["a", "b", "c"]
    assert trend["false_positive_rate_spread"] == pytest.approx(0.20)
    assert trend["spearman_rho_fpr_vs_capability"] == pytest.approx(-1.0)


def test_timeout_sensitivity_rescues_slow_but_correct_samples():
    from scripts.rescore_evalplus_labels import timeout_sensitivity

    rescored = {
        "skipped": [],
        "problems": [
            {
                "task_id": "HumanEval/0",
                "samples": [
                    # Slow but correct: fails at a short limit, passes at a long one.
                    {
                        "code": "import time\ndef f():\n    time.sleep(0.3)\n    return 1\n",
                        "passed_base": True,
                        "passed_plus": False,
                        "plus_failure_category": "timeout",
                    },
                    # A genuine wrong answer must not be rescued.
                    {
                        "code": "def f():\n    return 2\n",
                        "passed_base": True,
                        "passed_plus": False,
                        "plus_failure_category": "assertion_error",
                    },
                ],
            }
        ],
    }
    tests = {"HumanEval/0": "def check(c):\n    assert c() == 1\n\ncheck(f)\n"}
    out = timeout_sensitivity(rescored, tests, long_timeout_s=20.0, workers=1)
    assert out["n_timeout_base_pass"] == 1
    assert out["n_rescued"] == 1
    # 2 base-passing samples, 2 plus-failures, 1 rescued -> 1/2.
    assert out["false_positive_rate_at_long_timeout"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Week 9: repair-run relabel
# --------------------------------------------------------------------------- #

from scripts.rescore_evalplus_labels import (  # noqa: E402
    is_repair_run,
    relabelled_repair_run,
    repair_arm_comparison,
    rescore_repair_run,
    shared_initial_pool,
    summarize_repair_cell,
)


def _repair_record(task_id: str, initial: str, final: str, final_passed: bool) -> dict:
    return {
        "task_id": task_id,
        "sample_index": 0,
        "initial_code": initial,
        "initial_error_message": "boom",
        "initial_failure_category": "assertion_error",
        "final_code": final,
        "final_passed": final_passed,
        "final_error_message": "",
        "final_failure_category": None,
        "trace": [],
    }


def _repair_run(mode: str, finals: list[tuple[str, bool]]) -> dict:
    return {
        "run_id": f"arm_{mode}",
        "timestamp": "2026-08-30T00:00:00",
        "config": {"model": "m", "context_mode": mode, "sample_index": 0},
        "summary": {"successful_repairs": sum(p for _, p in finals)},
        "records": [
            _repair_record(f"HumanEval/{i}", "def f():\n    return 0\n", code, passed)
            for i, (code, passed) in enumerate(finals)
        ],
    }


def test_is_repair_run_distinguishes_the_two_schemas():
    assert is_repair_run(_repair_run("none", [("def f():\n    return 1\n", True)]))
    assert not is_repair_run(_run())


def test_shared_initial_pool_requires_identical_failure_pools():
    a = _repair_run("none", [("def f():\n    return 1\n", True)])
    b = _repair_run("error", [("def f():\n    return 2\n", False)])
    pool = shared_initial_pool([a, b], ["a.json", "b.json"])
    assert [p["task_id"] for p in pool] == ["HumanEval/0"]

    b["records"][0]["initial_code"] = "def f():\n    return 99\n"
    with pytest.raises(ValueError, match="identical failure pool"):
        shared_initial_pool([a, b], ["a.json", "b.json"])


def test_rescore_repair_run_scores_final_code_and_skips_missing_suites():
    run = _repair_run(
        "none",
        [("def f():\n    return 1\n", True), ("def f():\n    return 0\n", True)],
    )
    tests = {"HumanEval/0": "def check(c):\n    assert c() == 1\n\ncheck(f)\n"}
    out = rescore_repair_run(run, tests, timeout_s=10.0, workers=1, excluded_tasks=())
    assert out["skipped"] == ["HumanEval/1"]
    assert out["problems"][0]["samples"][0]["passed_plus"] is True
    assert out["problems"][0]["samples"][0]["passed_base"] is True


def test_rescore_repair_run_can_score_the_initial_pool_as_base_failing():
    run = _repair_run("none", [("def f():\n    return 1\n", True)])
    tests = {"HumanEval/0": "def check(c):\n    assert c() == 0\n\ncheck(f)\n"}
    out = rescore_repair_run(
        run,
        tests,
        field="initial_code",
        passed_field="__never__",
        timeout_s=10.0,
        workers=1,
        excluded_tasks=(),
    )
    # The initial program returns 0, so it passes this plus suite even though
    # it is in the pool by virtue of failing the base suite.
    assert out["problems"][0]["samples"][0]["passed_plus"] is True
    assert out["problems"][0]["samples"][0]["passed_base"] is False


def _rescored_repair(rows: list[tuple[str, bool, bool, str]]) -> dict:
    return {
        "skipped": ["HumanEval/32"],
        "problems": [
            {
                "task_id": task_id,
                "sample_index": 0,
                "samples": [
                    {
                        "code": "c",
                        "passed_base": base,
                        "passed_plus": plus,
                        "plus_failure_category": category,
                    }
                ],
            }
            for task_id, base, plus, category in rows
        ],
    }


def test_summarize_repair_cell_reports_both_labels_and_fp_rate():
    rescored = _rescored_repair(
        [
            ("HumanEval/0", True, True, ""),
            ("HumanEval/1", True, False, "assertion_error"),
            ("HumanEval/2", False, True, ""),
            ("HumanEval/3", False, False, "assertion_error"),
        ]
    )
    cell = summarize_repair_cell(rescored, n_coordinates_total=5)
    assert cell["n_coordinates_total"] == 5
    assert cell["n_coordinates_scored"] == 4
    assert cell["skipped_task_ids"] == ["HumanEval/32"]
    assert cell["confusion"] == {
        "base_pass_plus_pass": 1,
        "base_pass_plus_fail": 1,
        "base_fail_plus_pass": 1,
        "base_fail_plus_fail": 1,
    }
    assert cell["successes_base"] == 2
    assert cell["successes_plus"] == 2
    assert cell["false_positive_rate"] == pytest.approx(0.5)
    assert cell["base_fail_plus_pass"] == 1
    lo, hi = cell["false_positive_rate_wilson_95"]
    assert 0.0 <= lo < 0.5 < hi <= 1.0


def test_summarize_repair_cell_adjusts_the_denominator_for_plus_passing_initials():
    rescored = _rescored_repair(
        [
            ("HumanEval/0", True, True, ""),
            ("HumanEval/1", False, False, "assertion_error"),
        ]
    )
    cell = summarize_repair_cell(
        rescored, n_coordinates_total=2, initial_plus_pass_ids={"HumanEval/0"}
    )
    # HumanEval/0 was never a failure under the plus label, so it leaves the
    # denominator — and its success leaves the numerator with it.
    assert cell["n_coordinates_failing_under_plus"] == 1
    assert cell["successes_plus_adjusted"] == 0
    assert cell["repair_success_rate_plus_adjusted"] == pytest.approx(0.0)
    assert cell["repair_success_rate_plus_over_scored"] == pytest.approx(0.5)


def test_relabelled_repair_run_is_additive_and_drops_unscored_coordinates():
    run = _repair_run(
        "code_only",
        [("def f():\n    return 1\n", True), ("def f():\n    return 2\n", False)],
    )
    rescored = _rescored_repair([("HumanEval/0", True, False, "assertion_error")])
    initial = _rescored_repair([("HumanEval/0", False, False, "assertion_error")])
    out = relabelled_repair_run(run, rescored, initial, "results/arm.json")
    assert out["run_id"] == "arm_code_only_evalpluslabel"
    assert len(out["records"]) == 1
    record = out["records"][0]
    # Base label preserved next to the new one — the ladder script switches
    # between them with --label.
    assert record["final_passed"] is True
    assert record["final_passed_plus"] is False
    assert record["initial_passed_plus"] is False
    assert out["config"]["label_source"] == "base+evalplus_humanevalplus"
    assert out["config"]["relabel_source"] == "results/arm.json"
    assert out["summary"]["successful_repairs"] == 1
    assert out["summary"]["successful_repairs_plus"] == 0


def test_repair_arm_comparison_reports_spread_and_overlap():
    cells = [
        {
            "context_mode": "none",
            "false_positive_rate": 0.10,
            "false_positive_rate_wilson_95": [0.05, 0.20],
            "successes_base": 91,
            "successes_plus": 82,
        },
        {
            "context_mode": "error",
            "false_positive_rate": 0.12,
            "false_positive_rate_wilson_95": [0.06, 0.24],
            "successes_base": 54,
            "successes_plus": 48,
        },
    ]
    out = repair_arm_comparison(cells)
    assert out["arms"] == ["none", "error"]
    assert out["false_positive_rate_spread"] == pytest.approx(0.02)
    assert out["wilson_intervals_all_overlap"] is True
    assert out["common_overlap"] == [0.06, 0.20]
    assert out["base_label_success_by_arm"] == [91, 54]
