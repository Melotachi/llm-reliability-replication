"""Tests for the Week 13 verifier-triggered repair cascade.

Covers the cost/gain identities the cascade's tier-order claim depends on
(gain is order-independent for a fixed tier set; cost is not), the
base-model/repairer-model currency separation, the targeting-control and
marginal-return mechanics, and that pre-registration scoring only reads
already-computed numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_repair_cascade import (
    TIERS,
    all_orders_table,
    budget_sweep_within_failing_set,
    cascade_outcome,
    evaluate_order,
    marginal_return_table,
    nested_three_tier_order_selection,
    score_pre_registrations,
    single_arm_baselines,
    targeting_control,
    three_tier_cascade_table,
)


def _row(
    task_id: str,
    *,
    t0=(False, 0),
    t1=(False, 0),
    t2=(False, 0),
    t3=(False, 0),
    p_pass_oof: float = 0.5,
    initial_chars: float = 100.0,
) -> dict:
    """One cascade row; each t* is a (passed, cost) pair."""
    return {
        "task_id": task_id,
        "sample_index": 0,
        "p_pass_oof": p_pass_oof,
        "initial_chars": initial_chars,
        "failure_category": "assertion_error",
        "source_order": int(task_id.split("/")[1]),
        "tiers": {
            "T0": {"passed": t0[0], "cost": t0[1], "model": "base"},
            "T1": {"passed": t1[0], "cost": t1[1], "model": "repairer"},
            "T2": {"passed": t2[0], "cost": t2[1], "model": "repairer"},
            "T3": {"passed": t3[0], "cost": t3[1], "model": "repairer"},
        },
    }


def test_a_one_tier_cascade_equals_that_arm_exactly():
    rows = [
        _row("UNIT/0", t1=(True, 3)),
        _row("UNIT/1", t1=(False, 3)),
        _row("UNIT/2", t1=(True, 1)),
    ]
    baselines = single_arm_baselines(rows)
    assert baselines["T1"]["added_passes"] == 2
    assert baselines["T1"]["repairer_model_calls"] == 3 + 3 + 1
    assert baselines["T1"]["base_model_calls"] == 0


def test_an_unlimited_budget_cascade_equals_the_union():
    rows = [
        _row("UNIT/0", t1=(False, 3), t2=(True, 2), t3=(False, 3)),
        _row("UNIT/1", t1=(True, 3), t2=(False, 3), t3=(False, 3)),
        _row("UNIT/2", t1=(False, 3), t2=(False, 3), t3=(True, 3)),
        _row("UNIT/3", t1=(False, 3), t2=(False, 3), t3=(False, 3)),
    ]
    full = evaluate_order(rows, TIERS)["prefixes"][-1]
    union = sum(
        1 for row in rows if any(row["tiers"][tier]["passed"] for tier in TIERS)
    )
    assert full["added_passes"] == union == 3


def test_t0_at_k_equals_zero_equals_the_baseline():
    rows = [_row("UNIT/0", t0=(False, 0)), _row("UNIT/1", t0=(False, 0))]
    outcome = cascade_outcome(rows[0], ("T0",))
    assert outcome["passed"] is False
    assert outcome["base_model_calls"] == 0
    assert outcome["repairer_model_calls"] == 0


def test_gain_is_identical_across_orders_sharing_a_tier_set():
    rows = [
        _row("UNIT/0", t1=(False, 3), t2=(True, 2), t3=(False, 3)),
        _row("UNIT/1", t1=(True, 3), t2=(False, 3), t3=(False, 3)),
        _row("UNIT/2", t1=(False, 3), t2=(False, 3), t3=(True, 1)),
        _row("UNIT/3", t1=(False, 3), t2=(False, 3), t3=(False, 3)),
    ]
    gains = set()
    for order in (("T1", "T2", "T3"), ("T2", "T1", "T3"), ("T3", "T2", "T1")):
        gains.add(evaluate_order(rows, order)["prefixes"][-1]["added_passes"])
    assert len(gains) == 1


def test_cost_differs_across_orders_sharing_a_tier_set_when_a_cheap_tier_suffices():
    # UNIT/0 passes T1 cheaply; trying T2/T3 first would waste their cost on it.
    rows = [_row("UNIT/0", t1=(True, 1), t2=(False, 3), t3=(False, 3))]
    cheap_first = evaluate_order(rows, ("T1", "T2", "T3"))["prefixes"][-1]
    expensive_first = evaluate_order(rows, ("T2", "T3", "T1"))["prefixes"][-1]
    assert cheap_first["repairer_model_calls"] < expensive_first["repairer_model_calls"]
    assert cheap_first["added_passes"] == expensive_first["added_passes"] == 1


def test_base_model_calls_and_repairer_calls_are_never_summed_together():
    row = _row("UNIT/0", t0=(False, 5), t1=(False, 3), t2=(True, 2))
    outcome = cascade_outcome(row, ("T0", "T1", "T2"))
    assert outcome["base_model_calls"] == 5
    assert outcome["repairer_model_calls"] == 3 + 2
    assert "calls" not in outcome  # no merged key exists at all


def test_three_tier_cascade_table_finds_t3_dead_weight_when_it_adds_nothing():
    rows = [
        _row("UNIT/0", t1=(False, 3), t2=(True, 2), t3=(False, 3)),
        _row("UNIT/1", t1=(True, 3), t2=(False, 3), t3=(False, 3)),
        _row("UNIT/2", t1=(False, 3), t2=(False, 3), t3=(False, 3)),
    ]
    table = three_tier_cascade_table(rows)
    assert table["reaches_ceiling_without_t3"] is True
    assert table["two_tier_strictly_cheaper_than_every_t3_order"] is True


def test_three_tier_cascade_table_shows_t3_earning_its_place_when_it_adds_a_unique_pass():
    rows = [
        _row("UNIT/0", t1=(False, 3), t2=(False, 3), t3=(True, 3)),
    ]
    table = three_tier_cascade_table(rows)
    assert table["reaches_ceiling_without_t3"] is False
    assert table["ceiling_added_passes"] == 1
    assert table["two_tier_added_passes"] == 0


def test_targeting_control_prefers_the_verifier_over_random_when_gains_are_concentrated():
    # 10 coordinates, T1 fails on 4 of them; T2 rescues exactly one of those 4
    # and would also rescue one T1-passing coordinate if wastefully escalated.
    rows = []
    for index in range(10):
        t1_pass = index not in (0, 1, 2, 3)
        t2_pass = index == 0 or index == 5  # index 5 already passes under T1
        rows.append(_row(f"UNIT/{index}", t1=(t1_pass, 3), t2=(t2_pass, 3)))
    result = targeting_control(rows, n_seeds=50, seed=1)
    assert result["n_escalated"] == 4
    assert result["cascade_added_passes"] == 1  # only index 0 gains
    assert result["oracle_added_passes"] == 1  # matched-n oracle ties the cascade
    assert result["oracle_equals_cascade"] is True
    # Random sometimes escalates index 5, which is a pure loss (T1 passed, T2 fails
    # there) -- so its mean should not exceed the cascade's exact-failure targeting.
    assert result["random_control"]["mean"] <= result["cascade_added_passes"]


def test_targeting_control_is_seed_reproducible():
    rows = [
        _row(f"UNIT/{i}", t1=(i % 3 == 0, 3), t2=(i % 2 == 0, 3)) for i in range(12)
    ]
    first = targeting_control(rows, n_seeds=20, seed=7)
    second = targeting_control(rows, n_seeds=20, seed=7)
    assert first["random_control"] == second["random_control"]


def test_budget_sweep_within_failing_set_reports_attempts_with_every_gain():
    rows = [_row(f"UNIT/{i}", t1=(False, 3), t2=(i < 2, 1)) for i in range(6)]
    sweep = budget_sweep_within_failing_set(rows, coverages=(0.5, 1.0), seed=3)
    for ordering_rows in sweep["sweep"].values():
        for point in ordering_rows:
            assert "attempts" in point and "added" in point
            assert point["attempts"] >= point["added"]


def _arm_records(records: dict) -> dict:
    return {"run": {"records": list(records.values())}}


def _candidate(passed: bool, index: int) -> dict:
    return {"model_call_index": index, "passed": passed}


def test_marginal_return_table_finds_no_crossover_when_deepen_always_wins():
    rows = [_row(f"UNIT/{i}") for i in range(4)]
    current = {
        ("UNIT/0", 0): {
            "task_id": "UNIT/0",
            "sample_index": 0,
            "trace": [{}, {"candidates": [_candidate(True, 0)]}],
        },
        ("UNIT/1", 0): {
            "task_id": "UNIT/1",
            "sample_index": 0,
            "trace": [{}, {"candidates": [_candidate(True, 0)]}],
        },
        ("UNIT/2", 0): {
            "task_id": "UNIT/2",
            "sample_index": 0,
            "trace": [{}, {"candidates": [_candidate(True, 0)]}],
        },
        ("UNIT/3", 0): {
            "task_id": "UNIT/3",
            "sample_index": 0,
            "trace": [{}, {"candidates": [_candidate(True, 0)]}],
        },
    }
    # Next tier never gets a chance because every current-tier candidate 1 passes.
    next_tier = {
        coordinate: {
            "task_id": coordinate[0],
            "sample_index": 0,
            "trace": [{}, {"candidates": [_candidate(False, 0)]}],
        }
        for coordinate in current
    }
    raw_arms = {"T1": _arm_records(current), "T2": _arm_records(next_tier)}
    table = marginal_return_table(
        rows, raw_arms, current_tier="T1", next_tier="T2", max_k=1
    )
    assert table["crossover_k"] is None
    assert table["entries"][0]["deepen_rate"] == 1.0


def test_marginal_return_table_finds_the_crossover_when_switch_overtakes_at_some_k():
    rows = [_row(f"UNIT/{i}") for i in range(4)]
    # Current tier never passes at k=1 or k=2; next tier passes for everyone.
    current = {
        (f"UNIT/{i}", 0): {
            "task_id": f"UNIT/{i}",
            "sample_index": 0,
            "trace": [
                {},
                {"candidates": [_candidate(False, 0), _candidate(False, 1)]},
            ],
        }
        for i in range(4)
    }
    next_tier = {
        (f"UNIT/{i}", 0): {
            "task_id": f"UNIT/{i}",
            "sample_index": 0,
            "trace": [{}, {"candidates": [_candidate(True, 0)]}],
        }
        for i in range(4)
    }
    raw_arms = {"T1": _arm_records(current), "T2": _arm_records(next_tier)}
    table = marginal_return_table(
        rows, raw_arms, current_tier="T1", next_tier="T2", max_k=2
    )
    assert table["crossover_k"] == 1
    for entry in table["entries"]:
        assert entry["deepen_rate"] == 0.0
        assert entry["switch_rate"] == 1.0


def test_nested_order_selection_partition_check_holds_on_a_clean_split():
    rows = [
        _row(f"UNIT/{i}", t1=(i % 2 == 0, 3), t2=(i % 3 == 0, 2), t3=(False, 3))
        for i in range(12)
    ]
    result = nested_three_tier_order_selection(rows, n_splits=3, seed=5)
    assert result["fold_baseline_partition_check"]["passed"] is True
    assert 0 <= result["held_out_added_passes"] <= len(rows)


def test_pre_registration_pr1_falsifies_if_error_arm_solves_a_unique_coordinate():
    rows = [
        _row("UNIT/0", t1=(False, 3), t2=(False, 3), t3=(True, 3)),
        _row("UNIT/1", t1=(True, 3), t2=(False, 3), t3=(False, 3)),
    ]
    three_tier = three_tier_cascade_table(rows)
    targeting = targeting_control(rows, n_seeds=10, seed=1)
    marginal = {"crossover_k": None, "entries": []}
    scored = score_pre_registrations(
        three_tier, targeting, marginal, None, None, is_mbpp=False
    )
    pr1 = next(entry for entry in scored if entry["id"] == "PR-1")
    assert pr1["verdict"] == "falsified"


def test_pre_registration_scoring_reads_only_precomputed_fields():
    rows = [_row("UNIT/0", t1=(True, 3), t2=(False, 3), t3=(False, 3))]
    three_tier = three_tier_cascade_table(rows)
    targeting = targeting_control(rows, n_seeds=5, seed=1)
    marginal = {"crossover_k": 2, "entries": []}
    before = (dict(three_tier), dict(targeting), dict(marginal))
    score_pre_registrations(three_tier, targeting, marginal, None, None, is_mbpp=False)
    # Scoring must not mutate its inputs -- it only reads them.
    assert three_tier.keys() == before[0].keys()
    assert targeting.keys() == before[1].keys()
    assert marginal.keys() == before[2].keys()


def test_pre_registration_pr5_not_scored_without_a_second_cohort():
    rows = [_row("UNIT/0", t1=(True, 3))]
    three_tier = three_tier_cascade_table(rows)
    targeting = targeting_control(rows, n_seeds=5, seed=1)
    marginal = {"crossover_k": None, "entries": []}
    scored = score_pre_registrations(
        three_tier, targeting, marginal, None, None, is_mbpp=False
    )
    pr5 = next(entry for entry in scored if entry["id"] == "PR-5")
    assert pr5["verdict"] == "not scored (needs both cohorts)"


def test_all_orders_table_covers_every_permutation_of_the_four_tiers():
    rows = [_row("UNIT/0", t1=(True, 3))]
    table = all_orders_table(rows)
    assert len(table) == 24
    assert len({entry["order"] for entry in table}) == 24
