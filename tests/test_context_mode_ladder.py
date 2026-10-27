"""Tests for scripts/analyze_context_mode_ladder.py (Week 9 context-mode ladder)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_context_mode_ladder import (
    _bh_adjust,
    _roc_auc,
    _wilson_interval,
    arm_summaries,
    build_analysis,
    category_breakdown,
    complementarity,
    contamination,
    entrenchment,
    load_arms,
    ordered_modes,
    paired_comparisons,
    plot_ladder,
    week8_reproduction,
)

INITIAL = "def f(x):\n    return x + 1\n"
PATCHED = "def f(x):\n    return x * 2\n"
FRESH = "def f(value):\n    total = value * 2\n    return total\n"


def _record(
    task_id: str,
    final_code: str,
    passed: bool,
    *,
    category: str = "assertion_error",
    calls: int = 3,
    passed_plus: bool | None = None,
) -> dict:
    record = {
        "task_id": task_id,
        "sample_index": 0,
        "initial_code": INITIAL,
        "initial_failure_category": category,
        "repair_model_calls": calls,
        "final_code": final_code,
        "final_passed": passed,
        "final_failure_category": None if passed else category,
    }
    if passed_plus is not None:
        record["final_passed_plus"] = passed_plus
    return record


def _arm_run(mode: str, records: list[dict], *, calls: int | None = None) -> dict:
    return {
        "run_id": f"arm_{mode}",
        "timestamp": "2026-08-30T00:00:00",
        "config": {
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "context_mode": mode,
            "temperature": 0.8,
            "seed": 42,
            "device_resolved": "cuda",
            "dtype_resolved": "torch.float16",
            "selection_sha256": "abc",
            "stop_on_pass": False,
        },
        "summary": {
            "repair_model_calls": (
                calls
                if calls is not None
                else sum(r["repair_model_calls"] for r in records)
            ),
            "duplicate_candidate_rate": 0.1,
            "stagnation_stops": 2,
            "final_failure_categories": {"assertion_error": 1},
        },
        "records": records,
    }


def _write(tmp_path: Path, name: str, run: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(run), encoding="utf-8")
    return path


def _three_arms(tmp_path: Path) -> list[Path]:
    none = _arm_run(
        "none",
        [
            _record("HumanEval/0", FRESH, True, passed_plus=True),
            _record("HumanEval/1", FRESH, True, passed_plus=False),
            _record(
                "HumanEval/2", FRESH, False, category="name_error", passed_plus=False
            ),
        ],
    )
    code_only = _arm_run(
        "code_only",
        [
            _record("HumanEval/0", PATCHED, True, passed_plus=True),
            _record("HumanEval/1", PATCHED, False, passed_plus=False),
            _record(
                "HumanEval/2", PATCHED, False, category="name_error", passed_plus=False
            ),
        ],
    )
    error = _arm_run(
        "error",
        [
            _record("HumanEval/0", INITIAL, False, passed_plus=False),
            _record("HumanEval/1", PATCHED, False, passed_plus=False),
            _record(
                "HumanEval/2", PATCHED, True, category="name_error", passed_plus=True
            ),
        ],
    )
    return [
        _write(tmp_path, "none.json", none),
        _write(tmp_path, "code_only.json", code_only),
        _write(tmp_path, "error.json", error),
    ]


def test_wilson_interval_brackets_the_point_estimate():
    low, high = _wilson_interval(91, 148)
    assert low < 91 / 148 < high
    assert _wilson_interval(0, 0) == [0.0, 0.0]


def test_bh_adjust_is_monotone_and_bounded():
    adjusted = _bh_adjust([0.01, 0.04, 0.5])
    assert adjusted == sorted(adjusted)
    assert all(a >= p for a, p in zip(adjusted, [0.01, 0.04, 0.5]))
    assert max(adjusted) <= 1.0


def test_roc_auc_matches_hand_computed_cases():
    assert _roc_auc([0.9, 0.1], [True, False]) == pytest.approx(1.0)
    assert _roc_auc([0.1, 0.9], [True, False]) == pytest.approx(0.0)
    # Ties get the mid-rank, so a constant score is chance.
    assert _roc_auc([0.5, 0.5], [True, False]) == pytest.approx(0.5)
    assert _roc_auc([0.5, 0.5], [True, True]) is None


def test_load_arms_orders_the_ladder_and_keys_by_mode(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    assert set(arms) == {"none", "code_only", "error"}
    assert ordered_modes(arms) == ["none", "code_only", "error"]


def test_load_arms_rejects_unpaired_failure_pools(tmp_path):
    paths = _three_arms(tmp_path)
    run = json.loads(paths[1].read_text())
    run["records"][0]["initial_code"] = "def f(x):\n    return 999\n"
    paths[1].write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="not paired"):
        load_arms(paths, "base")


def test_load_arms_rejects_mismatched_coordinates(tmp_path):
    paths = _three_arms(tmp_path)
    run = json.loads(paths[2].read_text())
    run["records"][0]["task_id"] = "HumanEval/99"
    paths[2].write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="identical coordinate set"):
        load_arms(paths, "base")


def test_load_arms_requires_the_plus_label_when_asked_for_it(tmp_path):
    run = _arm_run("none", [_record("HumanEval/0", FRESH, True)])
    path = _write(tmp_path, "none_baseonly.json", run)
    with pytest.raises(ValueError, match="rescore_evalplus_labels"):
        load_arms([path], "plus")


def test_arm_summaries_report_rate_interval_and_cost(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    summaries = arm_summaries(arms, "base")
    assert [s["context_mode"] for s in summaries] == ["none", "code_only", "error"]
    assert summaries[0]["successes"] == 2
    assert summaries[0]["repair_success_rate"] == pytest.approx(2 / 3)
    assert summaries[0]["repair_model_calls"] == 9
    assert summaries[0]["model_calls_per_success"] == pytest.approx(4.5)
    low, high = summaries[0]["repair_success_wilson_95"]
    assert low < 2 / 3 < high


def test_paired_comparisons_cover_every_pair_with_bh(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    comparisons = paired_comparisons(arms, "base")
    assert len(comparisons) == 3
    none_vs_error = next(
        c for c in comparisons if c["left"] == "none" and c["right"] == "error"
    )
    # none solves 0 and 1; error solves only 2 — three discordant coordinates.
    assert none_vs_error["left_only_pass"] == 2
    assert none_vs_error["right_only_pass"] == 1
    assert none_vs_error["both_pass"] == 0
    assert none_vs_error["neither_pass"] == 0
    assert all(c["mcnemar_mid_p_bh"] >= c["mcnemar_mid_p"] for c in comparisons)


def test_paired_comparisons_switch_label(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "plus")
    comparisons = paired_comparisons(arms, "plus")
    none_vs_code = next(
        c for c in comparisons if c["left"] == "none" and c["right"] == "code_only"
    )
    # Under the plus label the none arm loses HumanEval/1, so the two arms tie.
    assert none_vs_code["left_only_pass"] == 0
    assert none_vs_code["right_only_pass"] == 0
    assert none_vs_code["both_pass"] == 1


def test_entrenchment_counts_exact_copies_and_scores_distance(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    cells = {c["context_mode"]: c for c in entrenchment(arms, "base")}
    # The error arm kept the failing program once (no candidate outranked it).
    assert cells["error"]["n_final_identical_to_initial"] == 1
    assert cells["error"]["n_changed"] == 2
    assert cells["none"]["n_final_identical_to_initial"] == 0
    # Resampling from scratch travels further from the initial program than
    # patching it does.
    assert (
        cells["none"]["mean_token_jaccard_to_initial"]
        > cells["code_only"]["mean_token_jaccard_to_initial"]
    )
    assert cells["none"]["roc_auc_token_jaccard_vs_success"] is not None


def test_entrenchment_separates_the_kept_initial_programs(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    cells = {c["context_mode"]: c for c in entrenchment(arms, "base")}
    error = cells["error"]
    # A kept initial program is a guaranteed failure at distance 0.0, so it is
    # a correctly-ranked negative by construction and must never appear in the
    # restricted population.
    assert error["changed_label_crosstab"] == {
        "changed_passed": 1,
        "changed_failed": 1,
        "identical_passed": 0,
        "identical_failed": 1,
    }
    assert error["mean_token_jaccard_to_initial_changed"] > (
        error["mean_token_jaccard_to_initial"]
    )
    # Averaged over the two changed programs only, not the kept initial one.
    assert error["mean_final_chars_changed"] == pytest.approx(len(PATCHED))
    # The inflation this split exists to expose, in miniature: the arm's two
    # changed programs are the same patch, so distance carries no information
    # about which one passed (restricted AUC 0.5), while the kept initial
    # program is a free correctly-ranked negative that lifts the unrestricted
    # number to 0.75.
    assert error["roc_auc_token_jaccard_vs_success"] == pytest.approx(0.75)
    assert error["roc_auc_token_jaccard_vs_success_changed"] == pytest.approx(0.5)
    assert error["n_changed"] == 2


def test_category_breakdown_splits_by_initial_failure_category(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    out = category_breakdown(arms, "base")
    assert out["categories"] == ["assertion_error", "name_error"]
    assert out["by_arm"]["none"]["assertion_error"]["failures"] == 2
    assert out["by_arm"]["none"]["assertion_error"]["successful_repairs"] == 2
    assert out["by_arm"]["error"]["name_error"]["successful_repairs"] == 1


def test_complementarity_separates_context_only_from_context_free_only(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    out = complementarity(arms, "base")
    assert out["n_union"] == 3
    assert out["n_unsolved_by_every_arm"] == 0
    assert out["context_only_task_ids"] == ["HumanEval/2"]
    assert out["none_only_task_ids"] == ["HumanEval/1"]
    assert out["solved_by_arm"] == {"none": 2, "code_only": 1, "error": 1}


def test_contamination_reports_both_reference_points(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    canonical = {
        t: "def f(x):\n    return x * 2\n"
        for t in ("HumanEval/0", "HumanEval/1", "HumanEval/2")
    }
    cells = {c["context_mode"]: c for c in contamination(arms, canonical, "base")}
    # code_only emits exactly the canonical body, so similarity saturates.
    assert cells["code_only"]["mean_similarity_to_canonical"] == pytest.approx(1.0)
    assert cells["code_only"]["n_above_0_9_similarity_to_canonical"] == 3
    # The control is always present next to it.
    assert cells["none"]["mean_similarity_to_initial"] is not None
    assert cells["none"]["mean_similarity_to_canonical"] < 1.0


def test_week8_reproduction_reports_bit_for_bit_agreement(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    week8 = _arm_run("error", json.loads(json.dumps(arms["error"]["run"]["records"])))
    week8["config"]["stop_on_pass"] = True
    week8["summary"]["repair_model_calls"] = 7
    out = week8_reproduction(arms, week8)
    assert out["n_shared_coordinates"] == 3
    assert out["identical_final_code"] == 3
    assert out["agreeing_final_passed"] == 3
    assert out["week8_stop_on_pass"] is True
    assert out["week9_stop_on_pass"] is False
    assert out["week9_repair_model_calls"] != out["week8_repair_model_calls"]


def test_build_analysis_and_figure_end_to_end(tmp_path):
    arms = load_arms(_three_arms(tmp_path), "base")
    analysis = build_analysis(
        arms,
        label="base",
        week8_run=None,
        canonical_by_task=None,
        dataset="humaneval",
    )
    assert analysis["config"]["label"] == "base"
    assert "contamination" not in analysis
    assert "week8_reproduction" not in analysis
    figure = tmp_path / "ladder.png"
    plot_ladder(analysis, figure)
    assert figure.exists() and figure.stat().st_size > 0
