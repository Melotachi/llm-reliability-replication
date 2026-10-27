"""Tests for the Week 12 pilot rule that buys the gate direction.

The rule exists to discharge a named threat: the paper's inverted gate chose its
sign after seeing the answer. So the properties that matter are (a) the sign is
read off the pilot alone, (b) the pilot's own repairs count towards the budget
rather than being charged as overhead, and (c) a pilot that cannot produce an AUC
falls back to the shipped direction instead of guessing.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.pilot_gate_direction import (
    _added_at_budget,
    _at_budget,
    _curve,
    evaluate_policies,
    fixed_ordering,
    pilot_ordering,
    sign_resolution_power,
)


def _rows(pairs: list[tuple[float, int]], calls: int = 2) -> list[dict]:
    """Rows from (p_pass, repaired) pairs, uniform cost unless overridden."""
    return [
        {
            "task_id": f"UNIT/{index}",
            "p_pass": p_pass,
            "calls": calls,
            "repaired": repaired,
            "source_order": index,
        }
        for index, (p_pass, repaired) in enumerate(pairs)
    ]


def test_fixed_ordering_risk_attempts_the_lowest_probability_first() -> None:
    rows = _rows([(0.9, 1), (0.1, 0), (0.5, 1)])
    order = [row["p_pass"] for row in fixed_ordering(rows, direction="risk")]
    assert order == [0.1, 0.5, 0.9]


def test_fixed_ordering_inverted_attempts_the_highest_first() -> None:
    rows = _rows([(0.9, 1), (0.1, 0), (0.5, 1)])
    order = [row["p_pass"] for row in fixed_ordering(rows, direction="inverted")]
    assert order == [0.9, 0.5, 0.1]


def test_fixed_ordering_breaks_ties_by_source_order() -> None:
    rows = _rows([(0.5, 1), (0.5, 0), (0.5, 1)])
    assert [row["source_order"] for row in fixed_ordering(rows, direction="risk")] == [
        0,
        1,
        2,
    ]


def test_curve_and_budget_lookup_accumulate_cost_gain_and_attempts() -> None:
    rows = _rows([(0.1, 1), (0.2, 0), (0.3, 1)], calls=3)
    points = _curve(rows)
    assert points == [(0, 0, 0), (3, 1, 1), (6, 1, 2), (9, 2, 3)]
    assert _added_at_budget(points, 0) == 0
    assert _added_at_budget(points, 5) == 1
    assert _added_at_budget(points, 9) == 2
    # A budget past the end cannot buy more than the pool holds.
    assert _added_at_budget(points, 1000) == 2


def test_at_budget_reports_the_attempts_that_bought_the_gain() -> None:
    """Two orderings at equal calls need not be at equal attempts, so both travel."""
    rows = _rows([(0.1, 1), (0.2, 0), (0.3, 1)], calls=3)
    points = _curve(rows)
    assert _at_budget(points, 5) == (1, 1)
    # The third coordinate is what lifts the gain, so attempts must follow it.
    assert _at_budget(points, 9) == (2, 3)


def test_at_budget_counts_every_attempt_the_budget_bought() -> None:
    """A flat run inside the budget is paid for, so it must appear in attempts.

    The alternative -- reporting the first point that reaches the affordable
    maximum -- would credit the policy with an early stop it cannot execute: it
    does not know in advance that coordinates 3 and 4 will add nothing.
    """
    rows = _rows([(0.1, 1), (0.2, 1), (0.3, 0), (0.4, 0)], calls=3)
    points = _curve(rows)
    assert _at_budget(points, 12) == (2, 4)
    # And the gain is still read at the last affordable point, not a later one.
    assert _at_budget(points, 11) == (2, 3)


def test_pilot_reads_the_sign_from_the_pilot_and_orders_the_rest_by_it() -> None:
    """High P(pass) repairs, low does not -> the pilot must pick `inverted`."""
    rows = _rows([(0.1 * index, int(index >= 5)) for index in range(10)])
    ordering, direction, auc = pilot_ordering(
        rows, epsilon=0.5, rng=np.random.default_rng(0)
    )
    assert direction == "inverted"
    assert auc is not None and auc > 0.5
    # Every coordinate is attempted exactly once: the pilot is part of the run.
    assert sorted(row["source_order"] for row in ordering) == list(range(10))
    # The post-pilot tail is ordered by the chosen direction.
    tail = ordering[5:]
    assert [row["p_pass"] for row in tail] == sorted(
        (row["p_pass"] for row in tail), reverse=True
    )


def test_pilot_falls_back_to_the_shipped_direction_when_it_learns_nothing() -> None:
    """A single-class pilot has no AUC; guessing would be worse than the default."""
    rows = _rows([(0.1 * index, 0) for index in range(10)])
    ordering, direction, auc = pilot_ordering(
        rows, epsilon=0.3, rng=np.random.default_rng(0)
    )
    assert auc is None
    assert direction == "risk"
    assert len(ordering) == len(rows)


def test_pilot_never_charges_the_pilot_as_extra_budget() -> None:
    """Total cost is the pool's cost whatever epsilon is: the pilot is not overhead."""
    rows = _rows([(0.1 * index, index % 2) for index in range(10)], calls=4)
    total = sum(row["calls"] for row in rows)
    for epsilon in (0.1, 0.5, 1.0):
        ordering, _, _ = pilot_ordering(
            rows, epsilon=epsilon, rng=np.random.default_rng(1)
        )
        assert _curve(ordering)[-1] == (
            total,
            sum(r["repaired"] for r in rows),
            len(rows),
        )


def test_evaluate_policies_recovers_the_full_pool_at_full_budget() -> None:
    rows = _rows([(0.1 * index, index % 3 == 0) for index in range(12)])
    result = evaluate_policies(
        rows, epsilons=(0.25,), n_seeds=8, first_seed=0, budgets=(0.5, 1.0)
    )
    assert result["total_repairable"] == sum(row["repaired"] for row in rows)
    for values in result["policies"].values():
        # At 100% of the budget every ordering has attempted everything.
        assert values["added_at_budget"][-1] == pytest.approx(
            result["total_repairable"]
        )


def test_evaluate_policies_names_the_full_data_direction() -> None:
    rows = _rows([(0.1 * index, int(index >= 6)) for index in range(12)])
    result = evaluate_policies(
        rows, epsilons=(0.25,), n_seeds=4, first_seed=0, budgets=(1.0,)
    )
    assert result["full_data_allocation_auc"] > 0.5
    assert result["full_data_direction"] == "inverted"


def test_sign_resolution_improves_with_pilot_size_on_a_strong_signal() -> None:
    rows = _rows([(0.01 * index, int(index >= 50)) for index in range(100)])
    resolution = sign_resolution_power(
        rows, pilot_sizes=(4, 40), n_draws=60, first_seed=0
    )
    by_size = {row["pilot_size"]: row for row in resolution["by_pilot_size"]}
    assert by_size[40]["sign_agreement_rate"] >= by_size[4]["sign_agreement_rate"]
    assert by_size[40]["sign_agreement_rate"] > 0.9


def test_sign_resolution_skips_pilot_sizes_larger_than_the_pool() -> None:
    rows = _rows([(0.1 * index, index % 2) for index in range(6)])
    resolution = sign_resolution_power(
        rows, pilot_sizes=(3, 6, 20), n_draws=10, first_seed=0
    )
    assert [row["pilot_size"] for row in resolution["by_pilot_size"]] == [3, 6]


def _rows_with_length(pairs: list[tuple[float, float, int]]) -> list[dict]:
    """Rows carrying both scoring columns: (p_pass, initial_chars, repaired)."""
    return [
        {
            "task_id": f"UNIT/{index}",
            "p_pass": p_pass,
            "initial_chars": chars,
            "calls": 2,
            "repaired": repaired,
            "source_order": index,
        }
        for index, (p_pass, chars, repaired) in enumerate(pairs)
    ]


def test_ordering_follows_the_named_score_column() -> None:
    """The two columns are deliberately anti-correlated here."""
    rows = _rows_with_length([(0.9, 10.0, 1), (0.1, 90.0, 0)])
    by_p_pass = fixed_ordering(rows, direction="risk", score="p_pass")
    by_length = fixed_ordering(rows, direction="risk", score="initial_chars")
    assert [row["source_order"] for row in by_p_pass] == [1, 0]
    assert [row["source_order"] for row in by_length] == [0, 1]


def test_pilot_reads_the_sign_from_the_column_it_was_given() -> None:
    """Short programs repair, and the two columns disagree, so the sign must too."""
    rows = _rows_with_length(
        [(0.9 - 0.08 * i, 10.0 * (i + 1), int(i < 5)) for i in range(10)]
    )
    _, length_direction, length_auc = pilot_ordering(
        rows, epsilon=1.0, rng=np.random.default_rng(0), score="initial_chars"
    )
    _, p_pass_direction, p_pass_auc = pilot_ordering(
        rows, epsilon=1.0, rng=np.random.default_rng(0), score="p_pass"
    )
    assert length_direction == "risk" and length_auc < 0.5
    assert p_pass_direction == "inverted" and p_pass_auc > 0.5


def test_policies_and_resolution_record_which_column_they_scored() -> None:
    rows = _rows_with_length([(0.1 * i, 5.0 * i, i % 2) for i in range(10)])
    policies = evaluate_policies(
        rows,
        epsilons=(0.2,),
        n_seeds=4,
        first_seed=0,
        budgets=(1.0,),
        score="initial_chars",
    )
    resolution = sign_resolution_power(
        rows, pilot_sizes=(4,), n_draws=10, first_seed=0, score="initial_chars"
    )
    assert policies["score_column"] == "initial_chars"
    assert resolution["score_column"] == "initial_chars"
