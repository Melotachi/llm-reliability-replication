"""Tests for the Week 12 repairability predictor and its proximity feature.

The load-bearing property is verdict-freeness: proximity must be a function of
probe *outputs* only. Week 7's crash-rate control showed that an agreement
feature which quietly scores never-running samples as maximally confident
degrades in step with the all-crash fraction, so the two undefined strata are
tested explicitly rather than left to the design matrix.
"""

from __future__ import annotations

import pytest

from scripts.diagnose_repairability import (
    _ast_nodes,
    pair_proximity,
    proximity_features,
    strata_report,
)


def test_pair_proximity_scores_only_co_running_probes() -> None:
    candidate = ["1", "2", "error:TypeError", "4"]
    sibling = ["1", "9", "3", "error:ValueError"]
    # Probes 0 and 1 co-run; 0 agrees, 1 does not.
    agreement, shared = pair_proximity(candidate, sibling)
    assert shared == 2
    assert agreement == pytest.approx(0.5)


def test_pair_proximity_is_undefined_without_a_co_running_probe() -> None:
    agreement, shared = pair_proximity(["error:A", "1"], ["2", "error:B"])
    assert agreement is None
    assert shared == 0


def test_pair_proximity_is_one_for_identical_running_behaviour() -> None:
    agreement, shared = pair_proximity(["1", "2"], ["1", "2"])
    assert agreement == pytest.approx(1.0)
    assert shared == 2


def test_proximity_averages_only_over_siblings_that_share_a_probe() -> None:
    candidate = ["1", "2", "error:X"]
    siblings = [
        ["1", "2", "3"],  # agrees on both co-running probes -> 1.0
        ["1", "9", "3"],  # agrees on one of two -> 0.5
        ["error:Y", "error:Y", "3"],  # no co-running probe -> excluded
    ]
    features = proximity_features(candidate, siblings)
    assert features["proximity_support"] == 2
    assert features["proximity"] == pytest.approx(0.75)
    assert features["has_proximity"] == 1
    assert features["crash_rate"] == pytest.approx(1 / 3)
    assert features["all_crash"] == 0


def test_all_crash_candidate_is_flagged_not_scored_as_confident() -> None:
    """The Week 7 trap: a never-running program must not look like agreement."""
    features = proximity_features(["error:A", "error:A"], [["1", "2"], ["1", "2"]])
    assert features["all_crash"] == 1
    assert features["crash_rate"] == pytest.approx(1.0)
    assert features["has_proximity"] == 0
    assert features["proximity_support"] == 0


def test_running_candidate_with_no_co_running_sibling_is_a_separate_stratum() -> None:
    """Distinct from all-crash: this program runs, its siblings just do not."""
    features = proximity_features(["1", "error:A"], [["error:B", "2"]])
    assert features["all_crash"] == 0
    assert features["crash_rate"] == pytest.approx(0.5)
    assert features["has_proximity"] == 0


def test_proximity_is_verdict_free() -> None:
    """Signatures carry outputs only; nothing in the feature reads pass/fail.

    Two candidates with identical probe outputs must get identical features no
    matter what their test verdicts were -- which is why the function takes
    signatures rather than records.
    """
    candidate = ["1", "2", "3"]
    siblings = [["1", "2", "9"]]
    assert proximity_features(candidate, siblings) == proximity_features(
        list(candidate), [list(siblings[0])]
    )


def test_empty_signature_is_treated_as_fully_crashed() -> None:
    features = proximity_features([], [["1"]])
    assert features["all_crash"] == 1
    assert features["has_proximity"] == 0


def test_ast_nodes_counts_and_flags_unparseable_code() -> None:
    assert _ast_nodes("x = 1\n") > 0
    # -1 rather than 0 keeps unparseable programs separable from empty ones.
    assert _ast_nodes("def f(:\n") == -1
    assert _ast_nodes("") == 1  # a bare Module node


def _row(**kwargs) -> dict:
    base = {
        "all_crash": 0,
        "has_proximity": 1,
        "proximity": 0.5,
        "proximity_support": 3,
        "final_passed": 0,
    }
    base.update(kwargs)
    return base


def test_strata_report_partitions_every_row_exactly_once() -> None:
    rows = [
        _row(all_crash=1, has_proximity=0, final_passed=1),
        _row(all_crash=0, has_proximity=0),
        _row(has_proximity=1, final_passed=1),
        _row(has_proximity=1),
    ]
    report = strata_report(rows)
    assert report["all_crash"]["n"] == 1
    assert report["runs_but_no_co_running_sibling"]["n"] == 1
    assert report["proximity_defined"]["n"] == 2
    total = sum(
        report[key]["n"]
        for key in (
            "all_crash",
            "runs_but_no_co_running_sibling",
            "proximity_defined",
        )
    )
    assert total == len(rows)


def test_strata_report_records_each_stratum_base_rate() -> None:
    """An imputed value would hide exactly this: strata with different rates."""
    rows = [
        _row(all_crash=1, has_proximity=0, final_passed=0),
        _row(all_crash=1, has_proximity=0, final_passed=0),
        _row(has_proximity=1, final_passed=1),
    ]
    report = strata_report(rows)
    assert report["all_crash"]["repair_rate"] == pytest.approx(0.0)
    assert report["proximity_defined"]["repair_rate"] == pytest.approx(1.0)
    assert report["mean_support_where_defined"] == pytest.approx(3.0)


def test_strata_report_handles_an_empty_stratum() -> None:
    report = strata_report([_row()])
    assert report["all_crash"]["n"] == 0
    assert report["all_crash"]["repair_rate"] is None


def test_compute_signatures_rejects_ragged_signature_lists(monkeypatch) -> None:
    """zip() in pair_proximity truncates silently, so raggedness must be loud."""
    import scripts.diagnose_repairability as module

    source = {
        "run_id": "unit",
        "config": {"dataset": "humaneval"},
        "per_problem": [
            {"task_id": "UNIT/0", "samples": [{"code": "a"}, {"code": "b"}]}
        ],
    }
    monkeypatch.setattr(
        module,
        "load_dataset_split",
        lambda name: [
            {"task_id": "UNIT/0", "entry_point": "f", "test_code": "assert f(1)"}
        ],
    )
    monkeypatch.setattr(
        module,
        "load_evalplus_probe_inputs",
        lambda name, cap, seed: {"UNIT/0": ["1", "2"]},
    )
    lengths = iter([["1", "2"], ["1"]])
    monkeypatch.setattr(module, "behavior_signature", lambda *a, **k: next(lengths))

    with pytest.raises(ValueError, match="ragged"):
        module.compute_signatures(source, timeout_s=1.0, probe_cap=12, seed=42)
