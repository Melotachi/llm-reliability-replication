"""Week 13 band-gate tests.

Two things these pin that are easy to lose. First, the band family must
*contain* the monotone gates of Weeks 8 and 12 -- if `band_ordering` at
degenerate bounds ever stops reproducing `fixed_ordering`, every earlier number
silently becomes incomparable. Second, `POLICY_GRID` is frozen: the whole
multiple-comparisons argument rests on the grid having been fixed before the
data were looked at, so widening it must show up in a diff.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.diagnose_band_gates import (
    POLICY_GRID,
    _at_attempts,
    band_ordering,
    band_resolution_power,
    band_summary,
    benjamini_hochberg,
    escalation_banding,
    evaluate_grid,
    family_correction,
    grid_for_score,
    marginal_cost_curve,
    neighbourhood_test,
    ordering_margin_permutation,
    permutation_test,
    prefix_equivalence_attempts,
    resolve_bounds,
    select_band_on,
    shape_table,
    stop_at_marginal_cost,
    wilson_interval,
)
from scripts.pilot_gate_direction import _at_budget, _curve, fixed_ordering


def make_rows(
    scores: list[float],
    repaired: list[int] | None = None,
    calls: list[int] | None = None,
) -> list[dict]:
    """Build the row shape build_rows() emits, for a synthetic cohort."""
    n = len(scores)
    repaired = [0] * n if repaired is None else repaired
    calls = [2] * n if calls is None else calls
    return [
        {
            "task_id": f"T/{index}",
            "p_pass": float(scores[index]),
            "initial_chars": float(100 + 10 * index),
            "calls": int(calls[index]),
            "repaired": int(repaired[index]),
            "source_order": index,
        }
        for index in range(n)
    ]


# --------------------------------------------------------------------------
# The family contains the monotone gates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["risk", "inverted"])
def test_degenerate_band_reproduces_fixed_ordering(direction: str) -> None:
    """Unbounded band == the Week 8/12 monotone gate, coordinate for coordinate."""
    rows = make_rows([0.9, 0.1, 0.5, 0.5, 0.02], repaired=[1, 0, 1, 0, 0])
    banded = band_ordering(rows, lo=None, hi=None, inside=direction, outside=direction)
    monotone = fixed_ordering(rows, direction=direction)
    assert [row["task_id"] for row in banded] == [row["task_id"] for row in monotone]


def test_band_partitions_rows_without_loss() -> None:
    """Every coordinate appears exactly once, whatever the bounds."""
    rows = make_rows([0.05, 0.2, 0.45, 0.6, 0.95])
    ordering = band_ordering(rows, lo=0.4, hi=0.55)
    assert sorted(row["task_id"] for row in ordering) == sorted(
        row["task_id"] for row in rows
    )
    assert len(ordering) == len(rows)


def test_band_is_attempted_before_the_fallback() -> None:
    """In-band coordinates come first; the remainder follows in `outside` order."""
    rows = make_rows([0.05, 0.45, 0.50, 0.95])
    ordering = band_ordering(rows, lo=0.4, hi=0.55, inside="inverted", outside="risk")
    assert [row["p_pass"] for row in ordering] == [0.50, 0.45, 0.05, 0.95]


def test_empty_band_falls_through_to_the_fallback() -> None:
    """A band no coordinate satisfies degenerates to the fallback ordering."""
    rows = make_rows([0.01, 0.02, 0.03])
    ordering = band_ordering(rows, lo=0.4, hi=0.55, outside="risk")
    assert [row["p_pass"] for row in ordering] == [0.01, 0.02, 0.03]


# --------------------------------------------------------------------------
# Raw and quantile readings
# --------------------------------------------------------------------------


def test_quantile_bounds_match_the_equivalent_raw_bounds() -> None:
    """The two readings agree once the quantiles are resolved to score units."""
    rows = make_rows([0.0, 0.25, 0.50, 0.75, 1.0])
    low, high = resolve_bounds(rows, lo=0.25, hi=0.75, score="p_pass", scale="quantile")
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.75)
    quantile = band_ordering(rows, lo=0.25, hi=0.75, scale="quantile")
    raw = band_ordering(rows, lo=low, hi=high, scale="raw")
    assert [row["task_id"] for row in quantile] == [row["task_id"] for row in raw]


def test_unbounded_side_resolves_to_infinity() -> None:
    """None means unbounded under both scales, not zero."""
    rows = make_rows([0.1, 0.9])
    assert resolve_bounds(rows, lo=None, hi=None, score="p_pass", scale="raw") == (
        -math.inf,
        math.inf,
    )
    assert resolve_bounds(rows, lo=None, hi=None, score="p_pass", scale="quantile") == (
        -math.inf,
        math.inf,
    )


def test_resolve_bounds_rejects_bad_input() -> None:
    """A typo in the scale or a quantile outside [0, 1] fails loudly."""
    rows = make_rows([0.1, 0.9])
    with pytest.raises(ValueError):
        resolve_bounds(rows, lo=0.1, hi=0.9, score="p_pass", scale="percentile")
    with pytest.raises(ValueError):
        resolve_bounds(rows, lo=-0.1, hi=0.9, score="p_pass", scale="quantile")


def test_grid_for_score_drops_raw_bands_for_non_probability_scores() -> None:
    """Character counts all exceed 0.55, so raw members would degenerate."""
    full = grid_for_score("p_pass")
    reduced = grid_for_score("initial_chars")
    assert full == POLICY_GRID
    assert len(reduced) < len(full)
    assert not any(
        policy["kind"] == "band" and policy["scale"] == "raw" for policy in reduced
    )


def test_policy_grid_is_frozen() -> None:
    """The multiplicity argument rests on the grid predating the data."""
    assert [policy["name"] for policy in POLICY_GRID] == [
        "risk",
        "inverted",
        "thresh>=0.05",
        "thresh>=0.10",
        "thresh>=0.20",
        "thresh>=0.35",
        "raw[0.10,0.40)",
        "raw[0.20,0.60)",
        "raw[0.40,0.55)",
        "raw[0.30,0.70)",
        "q[0.40,0.55)",
        "q[0.50,0.80)",
        "q[0.60,1.00]",
        "q[0.20,0.60)",
    ]


# --------------------------------------------------------------------------
# Curve accounting
# --------------------------------------------------------------------------


def test_at_attempts_boundaries() -> None:
    """Zero attempts buys nothing; over-long requests clip to the full curve."""
    rows = make_rows([0.1, 0.2, 0.3], repaired=[1, 0, 1], calls=[1, 3, 1])
    points = _curve(rows)
    assert _at_attempts(points, 0) == (0, 0)
    assert _at_attempts(points, 2) == (1, 4)
    assert _at_attempts(points, 99) == (2, 5)


def test_at_attempts_agrees_with_at_budget_on_constant_cost() -> None:
    """With equal costs the two axes are the same axis, so they must agree."""
    rows = make_rows([0.1, 0.2, 0.3, 0.4], repaired=[0, 1, 0, 1], calls=[2, 2, 2, 2])
    points = _curve(rows)
    for attempts in range(len(rows) + 1):
        added_attempts, _ = _at_attempts(points, attempts)
        added_budget, attempts_budget = _at_budget(points, 2 * attempts)
        assert added_attempts == added_budget
        assert attempts_budget == attempts


def test_marginal_cost_curve_windows() -> None:
    """The trailing window reads the last `window` attempts, not the prefix."""
    rows = make_rows([0.1] * 4, repaired=[1, 1, 0, 0], calls=[1, 1, 3, 3])
    curve = marginal_cost_curve(rows, window=2)
    assert curve[1]["window_calls_per_success"] == pytest.approx(1.0)
    assert curve[3]["window_successes"] == 0
    assert curve[3]["window_calls_per_success"] is None
    assert curve[-1]["added"] == 2
    assert curve[-1]["calls"] == 8


def test_stop_rule_waits_for_a_full_window() -> None:
    """A barren first attempt cannot stop the rule before the window fills."""
    rows = make_rows([0.1] * 6, repaired=[0, 1, 1, 0, 0, 0], calls=[3, 1, 1, 3, 3, 3])
    stop = stop_at_marginal_cost(rows, window=3, threshold=4.0)
    assert stop["stopped"] is True
    assert stop["attempts"] >= 3
    assert stop["added"] == 2


def test_stop_rule_does_not_fire_while_the_price_is_good() -> None:
    """An ordering that keeps paying runs to the end and says so."""
    rows = make_rows([0.1] * 5, repaired=[1] * 5, calls=[1] * 5)
    stop = stop_at_marginal_cost(rows, window=2, threshold=4.0)
    assert stop["stopped"] is False
    assert stop["attempts"] == 5
    assert stop["added"] == 5


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate() -> None:
    """A known small-n case stays inside [0, 1] and brackets the proportion."""
    low, high = wilson_interval(10, 13)
    assert 0.0 <= low < 10 / 13 < high <= 1.0
    assert low == pytest.approx(0.492, abs=0.01)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_benjamini_hochberg_is_monotone_and_conservative() -> None:
    """Adjusted values never decrease with rank and never fall below the raw."""
    raw = [0.001, 0.01, 0.03, 0.5]
    adjusted = benjamini_hochberg(raw)
    assert adjusted == sorted(adjusted)
    assert all(a >= r for a, r in zip(adjusted, raw))
    assert adjusted[0] == pytest.approx(0.004)
    assert benjamini_hochberg([]) == []


def test_permutation_test_is_seed_reproducible_and_never_zero() -> None:
    """Same seed, same p-values; and the observed arrangement is always counted."""
    scores = [i / 40 for i in range(40)]
    rows = make_rows(scores, repaired=[0] * 20 + [1] * 20)
    kwargs = {
        "score": "p_pass",
        "policies": grid_for_score("p_pass"),
        "n_permutations": 200,
    }
    first = permutation_test(rows, rng=np.random.default_rng(7), **kwargs)
    second = permutation_test(rows, rng=np.random.default_rng(7), **kwargs)
    assert first == second
    assert first["grid_max"]["p_grid_max"] > 0.0
    assert all(0.0 < band["p_single"] <= 1.0 for band in first["bands"])


# --------------------------------------------------------------------------
# Shape, selection and escalation
# --------------------------------------------------------------------------


def test_shape_table_detects_a_monotone_ladder() -> None:
    """A cleanly increasing cohort is reported monotone with no interior winner."""
    rows = make_rows(
        [i / 20 for i in range(20)],
        repaired=[0, 0, 0, 0] + [0, 0, 0, 1] + [0, 0, 1, 1] + [0, 1, 1, 1] + [1] * 4,
    )
    shape = shape_table(rows, score="p_pass")
    assert shape["monotone_non_decreasing"] is True
    assert shape["any_interior_bin_beats_top_beyond_noise"] is False
    assert [row["n"] for row in shape["bins"]] == [4, 4, 4, 4, 4]


def test_shape_table_ignores_wiggles_inside_the_noise() -> None:
    """A single-coordinate dip must not be reported as a sweet spot."""
    rows = make_rows(
        [i / 10 for i in range(10)], repaired=[0, 1, 0, 0, 1, 0, 1, 1, 1, 1]
    )
    shape = shape_table(rows, score="p_pass")
    assert shape["any_interior_bin_beats_top_beyond_noise"] is False


def test_select_band_on_prefers_the_better_priced_band() -> None:
    """Selection reads price per pass, so it lands on the productive coordinates.

    Asserted on the chosen band's membership rather than its name: several grid
    members can resolve to the same coordinate set on a small pilot, and which
    of those names wins the tie is not the behaviour worth pinning.
    """
    rows = make_rows(
        [0.02, 0.03, 0.45, 0.50, 0.95],
        repaired=[0, 0, 1, 1, 0],
        calls=[3, 3, 1, 1, 3],
    )
    policies = grid_for_score("p_pass")
    chosen = select_band_on(rows, score="p_pass", policies=policies)
    policy = next(item for item in policies if item["name"] == chosen)
    summary = band_summary(rows, policy, score="p_pass")
    assert summary["n"] == 2
    assert summary["successes"] == 2


def test_select_band_on_does_not_chase_the_narrowest_band() -> None:
    """A 1-for-1 band must not outprice a 4-for-5 one; that is what shrinkage buys.

    Without `SELECTION_PRIOR_ATTEMPTS` the raw price of the single lucky
    coordinate is unbeatable, and the out-of-fold rule degenerates to picking the
    narrowest member of the grid on every pilot.
    """
    # One lucky coordinate at the very top, a solid block just below it.
    rows = make_rows(
        [0.02, 0.03, 0.05, 0.42, 0.44, 0.46, 0.48, 0.52, 0.95],
        repaired=[0, 0, 0, 1, 1, 1, 1, 0, 1],
        calls=[3, 3, 3, 1, 1, 1, 1, 3, 1],
    )
    policies = grid_for_score("p_pass")
    chosen = select_band_on(rows, score="p_pass", policies=policies)
    policy = next(item for item in policies if item["name"] == chosen)
    summary = band_summary(rows, policy, score="p_pass")
    # A singleton band exists here and goes 1-for-1 at 1 call, so on raw
    # calls-per-success it would be unbeatable at 1.0.
    singleton = next(
        item for item in policies if band_summary(rows, item, score="p_pass")["n"] == 1
    )
    assert band_summary(rows, singleton, score="p_pass")["successes"] == 1
    assert chosen != singleton["name"]
    assert summary["n"] >= 3


def test_band_resolution_power_is_bounded_and_reproducible() -> None:
    """Agreement is a rate in [0, 1] and pilots larger than the cohort are skipped."""
    rows = make_rows([i / 30 for i in range(30)], repaired=[0] * 15 + [1] * 15)
    resolution = band_resolution_power(
        rows,
        score="p_pass",
        pilot_sizes=(5, 10, 999),
        n_draws=20,
        first_seed=0,
        oracle_policy="inverted",
    )
    assert [row["pilot_size"] for row in resolution["by_pilot_size"]] == [5, 10]
    assert all(
        0.0 <= row["band_agreement_rate"] <= 1.0 for row in resolution["by_pilot_size"]
    )


def test_escalation_banding_counts_the_rescue_ceiling() -> None:
    """Population is the base arm's failures; the ceiling is what context adds."""
    base = make_rows([0.05, 0.15, 0.30, 0.60], repaired=[1, 0, 0, 0])
    context = make_rows([0.05, 0.15, 0.30, 0.60], repaired=[0, 0, 1, 1])
    block = escalation_banding({"none": base, "code_only": context})
    assert block["escalation_population"] == 3
    assert block["rescue_ceiling"] == 2
    at_020 = next(row for row in block["by_threshold"] if row["threshold"] == 0.20)
    assert at_020["attempted"] == 2
    assert at_020["rescues_captured"] == 2
    assert at_020["capture_share"] == pytest.approx(1.0)


def test_evaluate_grid_conserves_the_arm_total_at_full_coverage() -> None:
    """No ordering can invent or lose a pass once everything is attempted."""
    rows = make_rows(
        [0.02, 0.10, 0.22, 0.45, 0.52, 0.80],
        repaired=[0, 0, 1, 1, 1, 0],
        calls=[3, 3, 2, 1, 1, 3],
    )
    grid = evaluate_grid(
        rows,
        score="p_pass",
        budgets=(0.5, 1.0),
        attempt_fractions=(0.5, 1.0),
        n_seeds=5,
        first_seed=0,
        stop_window=2,
        stop_thresholds=(6.0,),
    )
    assert grid["total_repairable"] == 3
    for name, values in grid["policies"].items():
        assert values["added_at_budget"][-1] == pytest.approx(3), name
        assert values["added_at_attempts"][-1] == pytest.approx(3), name


# --------------------------------------------------------------------------
# The structural crossover
# --------------------------------------------------------------------------


def test_prefix_equivalence_crossover_is_band_plus_everything_above() -> None:
    """Past `len(band) + len(above band)` a band IS `inverted`, on any dataset.

    The low end of the budget grid exists because of this: a band can only ever
    pay before its crossover, so evaluating one at a large budget compares a
    policy with itself.
    """
    rows = make_rows([0.02, 0.05, 0.42, 0.48, 0.52, 0.60, 0.95])
    policy = {
        "name": "raw[0.40,0.55)",
        "kind": "band",
        "scale": "raw",
        "lo": 0.40,
        "hi": 0.55,
    }
    # Three coordinates in the band, two above it.
    assert prefix_equivalence_attempts(rows, policy, score="p_pass") == 5


def test_prefix_equivalence_is_none_for_the_inverted_policy_itself() -> None:
    """`inverted` never diverges from `inverted`, which is reported as None."""
    rows = make_rows([0.1, 0.4, 0.9])
    policy = {"name": "inverted", "kind": "fixed", "direction": "inverted"}
    assert prefix_equivalence_attempts(rows, policy, score="p_pass") is None


def test_prefix_equivalence_is_none_for_a_top_anchored_band() -> None:
    """A band reaching the maximum has nothing above it, so it never diverges."""
    rows = make_rows([0.1, 0.4, 0.9])
    policy = {
        "name": "thresh>=0.35",
        "kind": "band",
        "scale": "raw",
        "lo": 0.35,
        "hi": None,
    }
    assert prefix_equivalence_attempts(rows, policy, score="p_pass") is None


def test_family_correction_covers_every_cell_and_is_conservative() -> None:
    """Cell-level grid-max p-values are themselves a family and get corrected.

    Without this, searching the grid in twelve cells and quoting the smallest
    reintroduces at the cell level exactly the multiplicity the within-cell
    permutation test absorbs.
    """
    analysis = {
        "cohort_name": "toy",
        "arms": {
            "a": {
                "by_score": {
                    "p_pass": {
                        "permutation": {
                            "grid_max": {
                                "best_band": "raw[0.40,0.55)",
                                "observed_hit_rate": 0.8,
                                "p_grid_max": 0.01,
                            }
                        }
                    },
                    "initial_chars": {
                        "permutation": {
                            "grid_max": {
                                "best_band": "q[0.40,0.55)",
                                "observed_hit_rate": 0.5,
                                "p_grid_max": 0.60,
                            }
                        }
                    },
                }
            }
        },
    }
    correction = family_correction([analysis])
    assert correction["n_cells"] == 2
    assert [cell["cell"] for cell in correction["cells"]] == [
        "toy/a/p_pass",
        "toy/a/initial_chars",
    ]
    assert correction["cells"][0]["p_grid_max_bh"] == pytest.approx(0.02)
    assert correction["cells"][0]["significant_bh_0.05"] is True
    assert correction["cells"][1]["significant_bh_0.05"] is False


def test_neighbourhood_test_compares_the_band_to_the_region_above_it() -> None:
    """Beating the cohort average is not evidence of a band on a monotone gradient.

    The claim "the payoff sits in a band rather than at the top" is only
    supported by the band beating its own upper neighbourhood, which is what this
    test isolates.
    """
    # Band [0.4, 0.55) goes 6-for-7; everything above it goes 0-for-7.
    rows = make_rows(
        [0.01, 0.02, 0.41, 0.42, 0.45, 0.47, 0.50, 0.52, 0.54]
        + [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        repaired=[0, 0, 1, 1, 1, 1, 1, 1, 0] + [0] * 7,
    )
    policy = {
        "name": "raw[0.40,0.55)",
        "kind": "band",
        "scale": "raw",
        "lo": 0.40,
        "hi": 0.55,
    }
    result = neighbourhood_test(rows, policy, score="p_pass")
    assert result["applicable"] is True
    assert result["band_successes"] == 6
    assert result["band_n"] == 7
    assert result["above_successes"] == 0
    assert result["above_n"] == 7
    assert result["p_fisher_two_sided"] < 0.05


def test_neighbourhood_test_is_not_applicable_without_an_upper_neighbourhood() -> None:
    """A top-anchored band and a whole-cohort policy have nothing above them."""
    rows = make_rows([0.1, 0.5, 0.9])
    threshold = {
        "name": "thresh>=0.35",
        "kind": "band",
        "scale": "raw",
        "lo": 0.35,
        "hi": None,
    }
    fixed = {"name": "inverted", "kind": "fixed", "direction": "inverted"}
    assert neighbourhood_test(rows, threshold, score="p_pass")["applicable"] is False
    assert neighbourhood_test(rows, fixed, score="p_pass")["applicable"] is False


def test_ordering_margin_permutation_nulls_the_margin_not_the_hit_rate() -> None:
    """The claimed quantity is the ordering's margin, so that is what is nulled."""
    rows = make_rows(
        [0.01, 0.02, 0.41, 0.45, 0.50, 0.60, 0.70, 0.80],
        repaired=[0, 0, 1, 1, 1, 0, 0, 0],
    )
    policy = {
        "name": "raw[0.40,0.55)",
        "kind": "band",
        "scale": "raw",
        "lo": 0.40,
        "hi": 0.55,
    }
    result = ordering_margin_permutation(
        rows,
        policy,
        score="p_pass",
        attempt_counts=[3],
        n_permutations=200,
        rng=np.random.default_rng(0),
    )
    # Band-first takes the three successes; inverted takes three failures.
    assert result["observed_margin"] == [3]
    assert 0.0 < result["p_margin"][0] <= 1.0
    repeat = ordering_margin_permutation(
        rows,
        policy,
        score="p_pass",
        attempt_counts=[3],
        n_permutations=200,
        rng=np.random.default_rng(0),
    )
    assert repeat == result
