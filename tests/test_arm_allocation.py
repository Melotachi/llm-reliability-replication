"""Tests for the Week 12 repair-arm allocation diagnosis.

Two things are worth pinning here beyond ordinary coverage.

The stop-on-pass call renormalization is the only reason the Week 9 arms and the
Week 7/8 pools can appear in the same cost table, so its reconstruction is tested
against both policies, including the identity case (a pool that already stopped
on a pass must reconstruct to itself).

The resample control counts *initially failing coordinates rescued by a later
stored sample*, which is easy to get subtly wrong: sample 0 is the coordinate
under repair and must never count towards its own rescue.
"""

from __future__ import annotations

import pytest

from scripts.diagnose_arm_allocation import (
    _run_guards,
    _stop_on_pass_calls,
    budget_matched_resample,
    renormalize_stop_on_pass,
    resample_control,
)


def _record(task_id: str, candidate_passes: list[bool], stored_calls: int) -> dict:
    """One repair record whose single repair iteration holds `candidate_passes`."""
    return {
        "task_id": task_id,
        "sample_index": 0,
        "repair_model_calls": stored_calls,
        "final_passed": any(candidate_passes),
        "trace": [
            {"iteration": 0, "candidates": []},
            {
                "iteration": 1,
                "candidates": [
                    {"candidate_index": index, "passed": passed}
                    for index, passed in enumerate(candidate_passes)
                ],
            },
        ],
    }


def _pool(records: list[dict], *, stop_on_pass: bool) -> dict:
    return {
        "run_id": "unit_pool",
        "config": {"stop_on_pass": stop_on_pass, "selection_sha256": "abc"},
        "summary": {
            "repair_model_calls": sum(r["repair_model_calls"] for r in records)
        },
        "records": records,
    }


def test_stop_on_pass_calls_truncates_at_the_first_pass() -> None:
    assert _stop_on_pass_calls(_record("U/0", [False, True, False], 3)) == 2
    assert _stop_on_pass_calls(_record("U/1", [True, True, True], 3)) == 1
    assert _stop_on_pass_calls(_record("U/2", [False, False, False], 3)) == 3


def test_stop_on_pass_calls_ignores_iterations_without_candidates() -> None:
    record = _record("U/0", [False, True], 2)
    record["trace"].append({"iteration": 2, "candidates": None})
    assert _stop_on_pass_calls(record) == 2


def test_renormalize_rewrites_calls_and_reports_the_delta() -> None:
    pool = _pool(
        [
            _record("U/0", [False, True, False], 3),
            _record("U/1", [False, False, False], 3),
        ],
        stop_on_pass=False,
    )
    renormalized, report = renormalize_stop_on_pass(pool)

    assert [r["repair_model_calls"] for r in renormalized["records"]] == [2, 3]
    assert report["stored_calls"] == 6
    assert report["renormalized_calls"] == 5
    assert report["per_coordinate_delta_histogram"] == {"-1": 1, "0": 1}
    # The stored value survives so a reader can always recover the raw cost.
    assert renormalized["records"][0]["repair_model_calls_as_stored"] == 3
    # And the input is untouched.
    assert pool["records"][0]["repair_model_calls"] == 3


def test_renormalize_is_the_identity_on_a_pool_that_already_stopped() -> None:
    """The faithfulness check: a stop-on-pass pool must reconstruct to itself."""
    pool = _pool(
        [
            _record("U/0", [False, True], 2),
            _record("U/1", [False, False, False], 3),
        ],
        stop_on_pass=True,
    )
    _, report = renormalize_stop_on_pass(pool)
    assert report["per_coordinate_delta_histogram"] == {"0": 2}
    assert report["renormalized_calls"] == report["stored_calls"] == 5


def _cohort(failing: list[str], passing: list[str]) -> list[dict]:
    return [
        {"task_id": task_id, "sample_index": 0, "initial_passed": False}
        for task_id in failing
    ] + [
        {"task_id": task_id, "sample_index": 0, "initial_passed": True}
        for task_id in passing
    ]


def _source_with_samples(passes_by_task: dict[str, list[bool]]) -> dict:
    return {
        "run_id": "unit_source",
        "config": {"model": "unit/model"},
        "per_problem": [
            {
                "task_id": task_id,
                "samples": [{"passed": passed} for passed in passes],
            }
            for task_id, passes in passes_by_task.items()
        ],
    }


def test_resample_control_counts_only_later_samples_of_failing_coordinates() -> None:
    source = _source_with_samples(
        {
            # fails at 0, rescued at sample 2
            "U/0": [False, False, True, False],
            # fails at 0 and never recovers
            "U/1": [False, False, False, False],
            # already passing: must not appear in the denominator or numerator
            "U/2": [True, True, True, True],
        }
    )
    control = resample_control(source, _cohort(["U/0", "U/1"], ["U/2"]), max_k=3)

    assert control["n_failing_coordinates"] == 2
    assert control["calls_go_to"] == "unit/model"
    added = {row["k_extra_samples"]: row["added_passes"] for row in control["rows"]}
    # k=0 is zero by construction and must equal the Baseline row.
    assert added == {0: 0, 1: 0, 2: 1, 3: 1}
    calls = {row["k_extra_samples"]: row["base_model_calls"] for row in control["rows"]}
    assert calls == {0: 0, 1: 2, 2: 4, 3: 6}


def test_resample_control_never_credits_the_coordinate_under_repair() -> None:
    """Sample 0 passing would make the coordinate non-failing; it is excluded."""
    source = _source_with_samples({"U/0": [True, False, False]})
    control = resample_control(source, _cohort([], ["U/0"]), max_k=2)
    assert control["n_failing_coordinates"] == 0
    assert all(row["added_passes"] == 0 for row in control["rows"])


def test_resample_control_calls_per_gain_is_none_before_the_first_rescue() -> None:
    source = _source_with_samples({"U/0": [False, False, True]})
    control = resample_control(source, _cohort(["U/0"], []), max_k=2)
    rows = {row["k_extra_samples"]: row for row in control["rows"]}
    assert rows[1]["calls_per_added_pass"] is None
    assert rows[2]["calls_per_added_pass"] == pytest.approx(2.0)


def test_budget_matched_resample_brackets_rather_than_interpolates() -> None:
    control = {
        "rows": [
            {"k_extra_samples": k, "added_passes": k, "base_model_calls": 100 * k}
            for k in range(4)
        ]
    }
    bracket = budget_matched_resample(control, 150)
    assert bracket["reference_repair_calls"] == 150
    assert bracket["largest_affordable"]["base_model_calls"] == 100
    assert bracket["smallest_unaffordable"]["base_model_calls"] == 200


def test_budget_matched_resample_handles_a_budget_past_the_sweep() -> None:
    control = {
        "rows": [
            {"k_extra_samples": k, "added_passes": k, "base_model_calls": 10 * k}
            for k in range(3)
        ]
    }
    bracket = budget_matched_resample(control, 10_000)
    assert bracket["largest_affordable"]["k_extra_samples"] == 2
    assert bracket["smallest_unaffordable"] is None


def _arm(auc_initial: float, auc_repair: float, *, sha: str, histogram: dict) -> dict:
    return {
        "targets": {
            "initial_passed": {"roc_auc": auc_initial},
            "repair_success": {"roc_auc": auc_repair},
        },
        "selection_sha256": sha,
        "call_renormalization": {
            "stop_on_pass_as_stored": True,
            "stored_calls": 10,
            "renormalized_calls": 10,
            "per_coordinate_delta_histogram": histogram,
        },
    }


def _spec() -> dict:
    return {
        "shipped_arm": "error",
        "shipped_expectations": {"initial_passed": 0.944, "repair_success": 0.518},
    }


def test_guards_pass_when_the_shipped_arm_reproduces() -> None:
    arms = {
        "error": _arm(0.944, 0.518, sha="abc", histogram={"0": 54}),
        "none": _arm(0.944, 0.481, sha="abc", histogram={"0": 54}),
    }
    guards = _run_guards("mbpp", _spec(), arms)
    assert guards["reproduction"]["passed"]
    assert guards["cohort_identity"]["passed"]
    assert all(entry["passed"] for entry in guards["call_renormalization"].values())


def test_guards_flag_a_shipped_arm_that_moved() -> None:
    arms = {"error": _arm(0.944, 0.600, sha="abc", histogram={"0": 54})}
    assert not _run_guards("mbpp", _spec(), arms)["reproduction"]["passed"]


def test_guards_flag_arms_built_over_different_coordinate_selections() -> None:
    arms = {
        "error": _arm(0.944, 0.518, sha="abc", histogram={"0": 54}),
        "none": _arm(0.944, 0.481, sha="DIFFERENT", histogram={"0": 54}),
    }
    assert not _run_guards("mbpp", _spec(), arms)["cohort_identity"]["passed"]


def test_guards_flag_a_stop_on_pass_pool_that_does_not_reconstruct() -> None:
    """A nonzero delta on an already-stopped pool means the reconstruction is wrong."""
    arms = {"error": _arm(0.944, 0.518, sha="abc", histogram={"0": 53, "-1": 1})}
    guards = _run_guards("mbpp", _spec(), arms)
    assert not guards["call_renormalization"]["error"]["passed"]
