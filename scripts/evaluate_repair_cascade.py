"""Week 13 (joint): a verifier-triggered repair CASCADE, replacing prediction.

This project has failed six times at predicting which failing coordinate is
worth escalating (Week 8's gate, Week 10's router, Week 12's proximity
predictor, and Week 12's ordering diagnostics). The idea here is different:
stop predicting. An execution-guided pipeline already runs a free, exact
verifier -- the test suite. Use IT as the escalation trigger. A CASCADE runs
the cheapest action first and escalates only on coordinates the verifier
still marks failing, exactly as a real deployment would.

Everything here is an exact REPLAY of already-stored outcomes from Week 9's
three repair arms (`none`, `code_only`, `error`) plus the stored base-model
resample siblings. Zero model calls; CPU only.

Four tiers, cheapest first:
    T0 -- reveal stored base-model (codegen-350M-mono) samples 1..k
    T1 -- `none`      (Qwen2.5-Coder-1.5B-Instruct, no failing program shown)
    T2 -- `code_only` (same repairer, shown the failing program)
    T3 -- `error`     (same repairer, shown the failing program + traceback)

T0's calls are spent on the GENERATOR; T1-T3's calls are spent on the
REPAIRER. These are never summed into one budget axis anywhere in this
module -- Week 8/12 established that conflating them hides which model is
actually buying the gain.

Two questions this script answers, matching the two well-posed halves of the
task:

1. Task A (arm level): what does the tier ORDER cost, holding gain fixed?
   Gain from running a fixed tier SET to exhaustion is order-independent
   (arithmetic: every tier in the set is eventually tried on every
   coordinate that needs it). Only COST is order-dependent -- a cascade that
   tries a cheap-and-sufficient tier first saves the calls a later tier would
   have wasted. PR-1 turns this into the score for the `error` tier: since it
   contributes zero coordinates the other two do not already solve, any
   order that tries T3 costs strictly more than the two-tier {T1, T2} order
   for the identical ceiling.

   Separately, PR-2/PR-3 test something orthogonal to ordering: does
   escalating on the VERIFIER's observed failures beat escalating on the
   same number of coordinates chosen by a PREDICTOR (random, or Objective 1's
   own `p_pass_oof` / `initial_chars`)? All policies spend the same
   escalation budget n = the count of T1 failures; they differ only in which
   n coordinates get the T2 call. This isolates conditioning on the observed
   outcome from conditioning on a prediction of it -- exactly the gap every
   earlier prediction-based attempt in this project could not close.

2. Task B (candidate level): each arm stores 3 candidates in call order with
   deterministic seeds, so "attempt k candidates of the current tier, then
   switch" is a valid replay of a policy that would have made exactly those
   calls. Only k <= 3 is testable.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_context_mode_ladder import complementarity, load_arms  # noqa: E402
from scripts.analyze_smart_repair import (  # noqa: E402
    _coordinate,
    align_oof_scores,
    load_json,
    select_cohort,
    validate_repair_pool,
)
from scripts.diagnose_arm_allocation import renormalize_stop_on_pass  # noqa: E402
from scripts.pilot_gate_direction import _at_budget, _curve  # noqa: E402
from src.utils import get_logger, save_results, set_all_seeds  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_FIGURE_DIR = Path("reports/figures")

#: Tier keys, cheapest to most expensive as the project understands cost.
TIERS: tuple[str, ...] = ("T0", "T1", "T2", "T3")
#: Which Week 9 arm each repair tier replays; T0 has no arm (it replays the
#: source run's stored siblings instead).
ARM_BY_TIER: dict[str, str] = {"T1": "none", "T2": "code_only", "T3": "error"}
#: Which model's budget each tier spends. T0 never shares an axis with T1-T3.
CURRENCY_BY_TIER: dict[str, str] = {
    "T0": "base_model_calls",
    "T1": "repairer_model_calls",
    "T2": "repairer_model_calls",
    "T3": "repairer_model_calls",
}

COHORTS: dict[str, dict[str, Any]] = {
    "humaneval": {
        "source": Path("results/humaneval_uncertainty_merged164_20260711.json"),
        "predictor": Path("results/predictor_humaneval_20260712_215312.json"),
        "arms": {
            "none": Path("results/humaneval_week9_arm_resample_sample0_qwen15b.json"),
            "code_only": Path(
                "results/humaneval_week9_arm_code_only_sample0_qwen15b.json"
            ),
            "error": Path(
                "results/humaneval_week9_arm_error_nostop_sample0_qwen15b.json"
            ),
        },
        "expected_complementarity": {
            "n_union": 109,
            "none_only": 39,
            "code_only_only": 6,
            "error_only": 0,
            "n_context_only": 18,
            "n_unsolved_by_every_arm": 39,
        },
        "expected_arm_successes": {"none": 91, "code_only": 65, "error": 54},
    },
    "mbpp": {
        "source": Path("results/mbpp_uncertainty_20260712_024909.json"),
        "predictor": Path("results/predictor_mbpp_20260712_212240.json"),
        "arms": {
            "none": Path("results/mbpp_week9_arm_resample_sample0_qwen15b.json"),
            "code_only": Path("results/mbpp_week9_arm_code_only_sample0_qwen15b.json"),
            "error": Path("results/mbpp_week9_arm_error_nostop_sample0_qwen15b.json"),
        },
        "expected_complementarity": None,  # not pre-registered for this cohort
        "expected_arm_successes": {"none": 27, "code_only": 14, "error": 13},
    },
}

#: PR-5's numeric threshold for "MBPP gains materially" (see the module
#: docstring and score_pre_registrations()). Chosen once, up front, so the
#: verdict is auditable rather than read off the result after the fact.
PR5_MATERIAL_GAIN_COORDINATES = 3

PRE_REGISTRATIONS: list[dict[str, str]] = [
    {
        "id": "PR-1",
        "statement": (
            "Restated per review: the near-trivial half is that the two-tier "
            "{none, code_only} cascade reaches the three-arm union (109/148 "
            "on HumanEval). The non-trivial claim is that "
            "union(none, code_only) == union(none, code_only, error) -- T3 is "
            "provably dead weight -- and that the two-tier order attains that "
            "ceiling at strictly lower repairer-model cost than every order "
            "that tries T3."
        ),
        "falsifier": (
            "T3 solves >= 1 coordinate neither T1 nor T2 solves, OR some "
            "order containing T3 reaches the ceiling at repairer cost <= the "
            "two-tier order's cost."
        ),
    },
    {
        "id": "PR-2",
        "statement": (
            "Escalating exactly on the n coordinates the verifier marks "
            "T1-failing beats escalating on a budget-matched RANDOM "
            "selection of n coordinates from the full cohort, by >= 2x on "
            "net added passes."
        ),
        "falsifier": "ratio (cascade added / random mean added) < 1.3",
    },
    {
        "id": "PR-3",
        "statement": (
            "The verifier-targeted escalation also beats a PREDICTED-failure "
            "escalation of the same size n, chosen by p_pass_oof (lowest "
            "first) and separately by initial_chars (longest first) -- a "
            "pipeline with no verifier has to guess, and p_pass_oof is this "
            "project's best guess."
        ),
        "falsifier": "either predictor's added passes is within 10% of the verifier's",
    },
    {
        "id": "PR-4",
        "statement": (
            "At the margin, switching to the next tier's first candidate "
            "beats deepening the cheap tier's own candidate list, for some "
            "k <= 3."
        ),
        "falsifier": "deepening dominates switching at every testable k <= 3",
    },
    {
        "id": "PR-5",
        "statement": (
            "Cascade gain tracks Week 9's complementarity: MBPP shows ~0 "
            "gain from adding context tiers, since it has only one "
            "context-only coordinate."
        ),
        "falsifier": (
            f"the T1->T2 cascade adds >= {PR5_MATERIAL_GAIN_COORDINATES} "
            "net passes on MBPP"
        ),
    },
]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def _t0_outcome(extra_passed: list[bool], max_k: int) -> tuple[bool, int]:
    """Outcome and cost of revealing stored base-model siblings, in order.

    Args:
        extra_passed: `passed` flags of samples 1..len(extra_passed) for one
            problem, in stored sample order.
        max_k: Budget cap on samples revealed.
    Returns:
        (passed, cost) where cost is the 1-based index of the first passing
        sample, or `max_k` (the whole budget, spent without success) if none
        of the first `max_k` samples pass. `max_k = 0` always returns
        `(False, 0)` -- the degenerate no-budget case.
    """
    if max_k <= 0:
        return False, 0
    capped = extra_passed[:max_k]
    for index, passed in enumerate(capped, start=1):
        if passed:
            return True, index
    return False, max_k


def load_cohort(
    cohort_name: str,
    spec: dict[str, Any],
    *,
    sample_index: int = 0,
    max_k0: int = 9,
) -> dict[str, Any]:
    """Join the source run, predictor, and three Week 9 arms into cascade rows.

    Args:
        cohort_name: "humaneval" or "mbpp", for error messages only.
        spec: One entry of COHORTS.
        sample_index: Candidate index the cohort is built from (Week 9 uses 0).
        max_k0: T0's sibling-reveal budget.
    Returns:
        Dict carrying the per-coordinate `rows` (one per initially-failing
        coordinate, each with every tier's (passed, cost) pair), the raw and
        renormalized arm runs (for guards and candidate-level extraction),
        and the source/predictor run ids.
    """
    source = load_json(spec["source"])
    predictor = load_json(spec["predictor"])
    scored = align_oof_scores(source, predictor)
    cohort = select_cohort(scored, sample_index=sample_index)
    failing = [row for row in cohort if not row["initial_passed"]]

    raw_arms: dict[str, dict[str, Any]] = {}
    renormalized_arms: dict[str, dict[str, Any]] = {}
    renorm_reports: dict[str, Any] = {}
    repair_by_tier: dict[str, dict[tuple[str, int], dict]] = {}
    for tier, arm_name in ARM_BY_TIER.items():
        path = spec["arms"][arm_name]
        raw = load_json(path)
        renormalized, report = renormalize_stop_on_pass(raw)
        raw_arms[tier] = {"run": raw, "path": path, "arm_name": arm_name}
        renormalized_arms[tier] = {"run": renormalized, "path": path}
        renorm_reports[tier] = report
        repair_by_tier[tier] = validate_repair_pool(
            spec["source"], cohort, renormalized, sample_index
        )

    passed_by_task = {
        problem["task_id"]: [bool(s["passed"]) for s in problem["samples"]]
        for problem in source["per_problem"]
    }

    rows: list[dict[str, Any]] = []
    reference_initial_code = None
    for entry in failing:
        coordinate = _coordinate(entry)
        task_id = entry["task_id"]
        records = {tier: repair_by_tier[tier][coordinate] for tier in ARM_BY_TIER}
        codes = {records[tier]["initial_code"] for tier in ARM_BY_TIER}
        if len(codes) != 1:
            raise ValueError(f"{task_id}: arms repair different initial programs")
        if reference_initial_code is None:
            reference_initial_code = next(iter(codes))

        extra = passed_by_task[task_id][1 : 1 + max_k0]
        t0_passed, t0_cost = _t0_outcome(extra, max_k0)

        tiers = {
            "T0": {"passed": t0_passed, "cost": t0_cost, "model": "base"},
        }
        for tier in ("T1", "T2", "T3"):
            record = records[tier]
            tiers[tier] = {
                "passed": bool(record["final_passed"]),
                "cost": int(record["repair_model_calls"]),
                "model": "repairer",
            }
        rows.append(
            {
                "task_id": task_id,
                "sample_index": sample_index,
                "p_pass_oof": float(entry["p_pass_oof"]),
                "initial_chars": float(len(records["T1"]["initial_code"])),
                "failure_category": entry["initial_failure_category"],
                "source_order": entry["source_order"],
                "tiers": tiers,
            }
        )

    hashes = {
        tier: raw_arms[tier]["run"].get("config", {}).get("selection_sha256")
        for tier in ARM_BY_TIER
    }
    return {
        "cohort_name": cohort_name,
        "rows": rows,
        "raw_arms": raw_arms,
        "renormalized_arms": renormalized_arms,
        "renorm_reports": renorm_reports,
        "selection_sha256_by_arm": hashes,
        "source_run_id": source.get("run_id"),
        "sample_index": sample_index,
        "max_k0": max_k0,
    }


# --------------------------------------------------------------------------
# Task A: arm-level cascade
# --------------------------------------------------------------------------


def cascade_outcome(row: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    """Walk one coordinate through a tier order, stopping at the first pass.

    Args:
        row: One `load_cohort` row.
        order: A permutation (or prefix) of tier keys.
    Returns:
        Dict with `passed`, `base_model_calls`, `repairer_model_calls` (never
        summed together), `attempts` (tiers actually tried), and
        `tiers_tried`.
    """
    base_calls = 0
    repairer_calls = 0
    attempts = 0
    passed = False
    tried: list[str] = []
    for tier in order:
        info = row["tiers"][tier]
        attempts += 1
        tried.append(tier)
        if CURRENCY_BY_TIER[tier] == "base_model_calls":
            base_calls += info["cost"]
        else:
            repairer_calls += info["cost"]
        if info["passed"]:
            passed = True
            break
    return {
        "passed": passed,
        "base_model_calls": base_calls,
        "repairer_model_calls": repairer_calls,
        "attempts": attempts,
        "tiers_tried": tuple(tried),
    }


def evaluate_order(
    rows: list[dict[str, Any]], order: tuple[str, ...]
) -> dict[str, Any]:
    """Cumulative cost and gain at every prefix length of one tier order."""
    prefixes = []
    for length in range(1, len(order) + 1):
        prefix = order[:length]
        total = {
            "passed": 0,
            "base_model_calls": 0,
            "repairer_model_calls": 0,
            "attempts": 0,
        }
        for row in rows:
            outcome = cascade_outcome(row, prefix)
            total["passed"] += int(outcome["passed"])
            total["base_model_calls"] += outcome["base_model_calls"]
            total["repairer_model_calls"] += outcome["repairer_model_calls"]
            total["attempts"] += outcome["attempts"]
        prefixes.append(
            {
                "length": length,
                "tiers": prefix,
                "added_passes": total["passed"],
                "base_model_calls": total["base_model_calls"],
                "repairer_model_calls": total["repairer_model_calls"],
                "attempts": total["attempts"],
                "single_currency": len({CURRENCY_BY_TIER[t] for t in prefix}) == 1,
            }
        )
    return {"order": order, "prefixes": prefixes}


def all_orders_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every one of the 4! tier permutations, scored at every prefix length.

    Cost is only comparable across orders that share a tier SET and a single
    currency (see the module docstring); this table reports every order's raw
    numbers without collapsing mixed-currency prefixes to one "best" figure.
    """
    return [evaluate_order(rows, order) for order in itertools.permutations(TIERS)]


def single_arm_baselines(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each tier run alone, as the incumbent baseline (identity guard target)."""
    return {tier: evaluate_order(rows, (tier,))["prefixes"][0] for tier in TIERS}


def three_tier_cascade_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """PR-1's well-posed comparison: all six orders of {T1, T2, T3}.

    Every tier here spends the same currency (repairer_model_calls), so "cost"
    is a single comparable number and "best order" is well-defined -- unlike
    the full 4-tier table, which mixes T0's base-model currency in.
    """
    orders = list(itertools.permutations(("T1", "T2", "T3")))
    scored = [evaluate_order(rows, order) for order in orders]
    ceiling = max(entry["prefixes"][-1]["added_passes"] for entry in scored)
    two_tier = evaluate_order(rows, ("T1", "T2"))["prefixes"][-1]
    at_ceiling = [
        {
            "order": entry["order"],
            "added_passes": entry["prefixes"][-1]["added_passes"],
            "repairer_model_calls": entry["prefixes"][-1]["repairer_model_calls"],
        }
        for entry in scored
        if entry["prefixes"][-1]["added_passes"] == ceiling
    ]
    min_three_tier_cost_at_ceiling = min(
        entry["repairer_model_calls"] for entry in at_ceiling
    )
    return {
        "orders": scored,
        "ceiling_added_passes": ceiling,
        "two_tier_added_passes": two_tier["added_passes"],
        "two_tier_repairer_model_calls": two_tier["repairer_model_calls"],
        "reaches_ceiling_without_t3": two_tier["added_passes"] == ceiling,
        "min_three_tier_cost_at_ceiling": min_three_tier_cost_at_ceiling,
        "two_tier_strictly_cheaper_than_every_t3_order": (
            two_tier["repairer_model_calls"] < min_three_tier_cost_at_ceiling
        ),
    }


def targeting_control(
    rows: list[dict[str, Any]],
    *,
    base_tier: str = "T1",
    escalate_tier: str = "T2",
    n_seeds: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """PR-2/PR-3: escalate on the SAME budget n, chosen four different ways.

    n is fixed to the number of base_tier failures (the cascade's own
    escalation size -- there is no free parameter here, escalating on
    "exactly the verifier failures" is the cascade by definition). Random,
    p_pass_oof, and initial_chars policies choose their n coordinates from
    the FULL cohort (not restricted to the failing subset), so they can waste
    escalation budget on already-passing coordinates -- which is exactly the
    gap the verifier does not have.

    Escalating a coordinate ADDS a candidate on top of base_tier's own
    (already-run, free) attempt; it never discards a working base_tier
    result. This mirrors the project's own repair contract ("passing samples
    are returned unchanged", `src/repair.py`'s `repair_pipeline`) and the
    ranking rule every stored arm already uses (`pass > failure_progress >
    ...`) -- you keep whichever candidate passes. So an escalation outcome is
    `base_pass OR (escalated AND escalate_pass)`: escalating an
    already-passing coordinate can waste a call but can never turn a pass
    into a failure. This is a real modeling choice, not an incidental detail:
    it is what makes "added passes" non-negative by construction and is why
    the random/predicted controls are a fair fight against the verifier
    rather than an inflated one.

    Args:
        rows: load_cohort()'s per-coordinate rows.
        base_tier: The tier whose failures define the escalation budget.
        escalate_tier: The tier escalated to.
        n_seeds: Random-policy draws.
        seed: Base seed (Constraint #4).
    Returns:
        Dict with the cascade's escalation size and net added passes, each
        control's added passes (random as a distribution, others as single
        deterministic values), the oracle upper bound, and the PR-2/PR-3
        ratios.
    """
    base_pass = np.array(
        [row["tiers"][base_tier]["passed"] for row in rows], dtype=bool
    )
    esc_pass = np.array(
        [row["tiers"][escalate_tier]["passed"] for row in rows], dtype=bool
    )
    p_pass = np.array([row["p_pass_oof"] for row in rows], dtype=float)
    chars = np.array([row["initial_chars"] for row in rows], dtype=float)
    n_rows = len(rows)
    baseline_total = int(base_pass.sum())
    n_escalate = int((~base_pass).sum())

    def net_added(mask: np.ndarray) -> int:
        outcome = base_pass | (mask & esc_pass)
        return int(outcome.sum()) - baseline_total

    verifier_mask = ~base_pass
    cascade_added = net_added(verifier_mask)

    rng = np.random.default_rng(seed)
    random_totals = np.empty(n_seeds, dtype=int)
    for draw in range(n_seeds):
        chosen = rng.choice(n_rows, size=n_escalate, replace=False)
        mask = np.zeros(n_rows, dtype=bool)
        mask[chosen] = True
        random_totals[draw] = net_added(mask)
    random_summary = {
        "n_seeds": n_seeds,
        "mean": float(random_totals.mean()),
        "std": float(random_totals.std(ddof=1)) if n_seeds > 1 else 0.0,
        "min": int(random_totals.min()),
        "max": int(random_totals.max()),
    }

    def _mask_from_ranking(scores: np.ndarray, *, descending: bool) -> np.ndarray:
        order = np.argsort(-scores if descending else scores, kind="mergesort")
        mask = np.zeros(n_rows, dtype=bool)
        mask[order[:n_escalate]] = True
        return mask

    predicted_p_pass_added = net_added(_mask_from_ranking(p_pass, descending=False))
    predicted_chars_added = net_added(_mask_from_ranking(chars, descending=True))

    # Under the OR-combine semantics above, escalating can only ever gain or
    # do nothing (never lose), so the per-coordinate value of escalating is
    # simply "would it flip a failure to a pass" -- there is no negative case
    # to rank away from.
    gain_value = (esc_pass & ~base_pass).astype(int)
    oracle_added = net_added(_mask_from_ranking(gain_value, descending=True))

    def _ratio(numerator: float, denominator: float) -> float | None:
        if denominator == 0:
            return None if numerator == 0 else math.inf
        return numerator / denominator

    return {
        "base_tier": base_tier,
        "escalate_tier": escalate_tier,
        "n_coordinates": n_rows,
        "n_escalated": n_escalate,
        "baseline_passes": baseline_total,
        "cascade_added_passes": cascade_added,
        "random_control": random_summary,
        "predicted_p_pass_oof_added_passes": predicted_p_pass_added,
        "predicted_initial_chars_added_passes": predicted_chars_added,
        "oracle_added_passes": oracle_added,
        "oracle_equals_cascade": oracle_added == cascade_added,
        "ratio_vs_random": _ratio(cascade_added, random_summary["mean"]),
        "ratio_vs_predicted_p_pass_oof": _ratio(cascade_added, predicted_p_pass_added),
        "ratio_vs_predicted_initial_chars": _ratio(
            cascade_added, predicted_chars_added
        ),
    }


def budget_sweep_within_failing_set(
    rows: list[dict[str, Any]],
    *,
    base_tier: str = "T1",
    escalate_tier: str = "T2",
    coverages: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
) -> dict[str, Any]:
    """Secondary result: given the escalation budget, which failures first?

    This is the ordering question the review kept explicitly separate from
    PR-2/PR-3: once you have decided to escalate the base_tier-failing
    coordinates, and cannot afford all of them, which subset goes first? Uses
    `_at_budget`'s last-affordable-point definition throughout, so attempts
    travel with every gain (Week 8/12's rule).

    Args:
        rows: load_cohort()'s rows.
        base_tier: Tier whose failures form the population being ordered.
        escalate_tier: Tier the ordering escalates to.
        coverages: Budget fractions of the full escalate_tier cost over the
            failing subset.
        seed: Random-ordering seed.
    Returns:
        Dict with the failing-subset size and, per ordering
        ("p_pass_oof", "initial_chars", "random"), the (added, attempts) pair
        at each requested coverage.
    """
    failing = [row for row in rows if not row["tiers"][base_tier]["passed"]]
    n_failing = len(failing)

    def curve_for(ordering: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
        curve_rows = [
            {
                "calls": row["tiers"][escalate_tier]["cost"],
                "repaired": int(row["tiers"][escalate_tier]["passed"]),
            }
            for row in ordering
        ]
        return _curve(curve_rows)

    by_p_pass = sorted(failing, key=lambda row: row["p_pass_oof"])
    by_chars = sorted(failing, key=lambda row: -row["initial_chars"])
    rng = np.random.default_rng(seed)
    random_order_indices = rng.permutation(n_failing)
    by_random = [failing[index] for index in random_order_indices]

    points = {
        "p_pass_oof": curve_for(by_p_pass),
        "initial_chars": curve_for(by_chars),
        "random": curve_for(by_random),
    }
    full_cost = points["p_pass_oof"][-1][0] if points["p_pass_oof"] else 0

    sweep: dict[str, list[dict[str, Any]]] = {}
    for name, ordering_points in points.items():
        rows_out = []
        for coverage in coverages:
            budget = int(round(coverage * full_cost))
            added, attempts = _at_budget(ordering_points, budget)
            rows_out.append(
                {
                    "coverage": coverage,
                    "budget": budget,
                    "added": added,
                    "attempts": attempts,
                }
            )
        sweep[name] = rows_out

    return {
        "base_tier": base_tier,
        "escalate_tier": escalate_tier,
        "n_failing": n_failing,
        "full_escalate_tier_cost": full_cost,
        "sweep": sweep,
    }


def nested_three_tier_order_selection(
    rows: list[dict[str, Any]], *, n_splits: int = 5, seed: int = 42
) -> dict[str, Any]:
    """Held-out choice of the cheapest {T1,T2,T3} order at the fixed ceiling.

    Mirrors route_context_arms.nested_routing's discipline: the order is
    chosen on training rows only (minimizing repairer_model_calls among
    orders reaching the training ceiling), then applied, unmodified, to
    held-out rows. The partition guard requires the per-fold single-arm
    baseline (T1 alone) to sum to the cohort total.

    Args:
        rows: load_cohort()'s rows.
        n_splits: KFold splits (no binary label to stratify on here).
        seed: Split seed.
    Returns:
        Held-out total added passes and repairer calls under the nested
        order choice, the per-fold chosen orders, and the partition guard.
    """
    n = len(rows)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    folds = np.array_split(shuffled, n_splits)

    total_added = 0
    total_calls = 0
    baseline_sum = 0
    chosen_orders = []
    for fold_index in range(n_splits):
        test_index = folds[fold_index]
        train_index = np.concatenate(
            [folds[i] for i in range(n_splits) if i != fold_index]
        )
        train_rows = [rows[i] for i in train_index]
        test_rows = [rows[i] for i in test_index]

        train_table = three_tier_cascade_table(train_rows)
        ceiling = train_table["ceiling_added_passes"]
        candidates = [
            entry
            for entry in train_table["orders"]
            if entry["prefixes"][-1]["added_passes"] == ceiling
        ]
        best = min(
            candidates,
            key=lambda entry: (
                entry["prefixes"][-1]["repairer_model_calls"],
                entry["order"],
            ),
        )
        order = best["order"]
        chosen_orders.append(order)

        test_outcome = evaluate_order(test_rows, order)["prefixes"][-1]
        total_added += test_outcome["added_passes"]
        total_calls += test_outcome["repairer_model_calls"]
        baseline_sum += sum(1 for row in test_rows if row["tiers"]["T1"]["passed"])

    cohort_baseline = sum(1 for row in rows if row["tiers"]["T1"]["passed"])
    if baseline_sum != cohort_baseline:
        raise ValueError(
            "Nested folds do not partition the cohort: per-fold T1 baseline "
            f"sums to {baseline_sum}, cohort baseline is {cohort_baseline}"
        )
    return {
        "n_splits": n_splits,
        "held_out_added_passes": total_added,
        "held_out_repairer_model_calls": total_calls,
        "fold_orders": [list(order) for order in chosen_orders],
        "fold_baseline_partition_check": {
            "sum": baseline_sum,
            "cohort_total": cohort_baseline,
            "passed": baseline_sum == cohort_baseline,
        },
    }


# --------------------------------------------------------------------------
# Task B: candidate-level frontier
# --------------------------------------------------------------------------


def _candidate_sequence(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The repair round's candidates, sorted into call order."""
    candidates = record["trace"][1]["candidates"] if len(record["trace"]) > 1 else []
    return sorted(candidates, key=lambda candidate: candidate["model_call_index"])


def marginal_return_table(
    rows: list[dict[str, Any]],
    raw_arms: dict[str, dict[str, Any]],
    *,
    current_tier: str = "T1",
    next_tier: str = "T2",
    max_k: int = 3,
) -> dict[str, Any]:
    """Per-call-index return of deepening the current tier vs switching tiers.

    Args:
        rows: load_cohort()'s per-coordinate rows (defines the population:
            every coordinate initially failing overall).
        raw_arms: load_cohort()'s `raw_arms`, keyed by tier.
        current_tier: The tier being deepened.
        next_tier: The tier switched to.
        max_k: Highest candidate index scored (capped at 3: each arm stores
            exactly 3 candidates per coordinate).
    Returns:
        Per-k deepen/switch rates with their support counts, and the
        crossover k (or None if deepening dominates throughout).
    """
    current_records = {
        _coordinate(record): record
        for record in raw_arms[current_tier]["run"]["records"]
    }
    next_records = {
        _coordinate(record): record for record in raw_arms[next_tier]["run"]["records"]
    }

    entries = []
    for k in range(1, max_k + 1):
        deepen_num = deepen_den = 0
        switch_num = switch_den = 0
        for row in rows:
            coordinate = (row["task_id"], row["sample_index"])
            current_candidates = _candidate_sequence(current_records[coordinate])
            if len(current_candidates) < k:
                continue
            prior_failed = all(not c["passed"] for c in current_candidates[: k - 1])
            if not prior_failed:
                continue
            deepen_den += 1
            if current_candidates[k - 1]["passed"]:
                deepen_num += 1

            all_k_failed = all(not c["passed"] for c in current_candidates[:k])
            if all_k_failed:
                next_candidates = _candidate_sequence(next_records[coordinate])
                if next_candidates:
                    switch_den += 1
                    if next_candidates[0]["passed"]:
                        switch_num += 1
        entries.append(
            {
                "k": k,
                "deepen_rate": (deepen_num / deepen_den) if deepen_den else None,
                "deepen_n": deepen_den,
                "switch_rate": (switch_num / switch_den) if switch_den else None,
                "switch_n": switch_den,
            }
        )

    crossover_k = next(
        (
            entry["k"]
            for entry in entries
            if entry["deepen_rate"] is not None
            and entry["switch_rate"] is not None
            and entry["switch_rate"] > entry["deepen_rate"]
        ),
        None,
    )
    return {
        "current_tier": current_tier,
        "next_tier": next_tier,
        "entries": entries,
        "crossover_k": crossover_k,
        "candidate_limit_note": (
            "only k<=3 is testable: each arm stores exactly 3 candidates per "
            "coordinate"
        ),
    }


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def run_guards(
    cohort_name: str, spec: dict[str, Any], loaded: dict[str, Any]
) -> dict[str, Any]:
    """The five verification guards, each reported with an explicit pass flag."""
    rows = loaded["rows"]

    # Guard 1: reproduction of Week 9's complementarity / arm totals.
    arm_paths = list(spec["arms"].values())
    arms = load_arms(arm_paths, "base")
    comp = complementarity(arms, "base")
    observed_successes = {
        mode: comp["solved_by_arm"].get(mode) for mode in spec["arms"]
    }
    expected_successes = spec["expected_arm_successes"]
    successes_match = all(
        observed_successes.get(mode) == expected_successes[mode]
        for mode in expected_successes
    )
    if spec["expected_complementarity"] is not None:
        expected = spec["expected_complementarity"]
        observed = {
            "n_union": comp["n_union"],
            "none_only": len(comp["solved_only_by_arm"]["none"]),
            "code_only_only": len(comp["solved_only_by_arm"]["code_only"]),
            "error_only": len(comp["solved_only_by_arm"]["error"]),
            "n_context_only": comp["n_context_only"],
            "n_unsolved_by_every_arm": comp["n_unsolved_by_every_arm"],
        }
        complementarity_match = observed == expected
    else:
        observed = {
            "n_union": comp["n_union"],
            "none_only": len(comp["solved_only_by_arm"]["none"]),
            "code_only_only": len(comp["solved_only_by_arm"]["code_only"]),
            "error_only": len(comp["solved_only_by_arm"]["error"]),
            "n_context_only": comp["n_context_only"],
            "n_unsolved_by_every_arm": comp["n_unsolved_by_every_arm"],
        }
        complementarity_match = None  # not pre-registered for this cohort
    guard_1 = {
        "observed_arm_successes": observed_successes,
        "expected_arm_successes": expected_successes,
        "observed_complementarity": observed,
        "expected_complementarity": spec["expected_complementarity"],
        "passed": successes_match and (complementarity_match in (True, None)),
    }

    # Guard 2: identity checks.
    single_arm = single_arm_baselines(rows)
    guard_2a = {
        tier: {
            "single_arm_passed": single_arm[tier]["added_passes"],
            "stored_arm_successes": (
                expected_successes[ARM_BY_TIER[tier]] if tier in ARM_BY_TIER else None
            ),
            "passed": (
                single_arm[tier]["added_passes"]
                == expected_successes[ARM_BY_TIER[tier]]
                if tier in ARM_BY_TIER
                else True
            ),
        }
        for tier in TIERS
    }

    union_all = sum(
        1 for row in rows if any(row["tiers"][tier]["passed"] for tier in TIERS)
    )
    unlimited_cascade = evaluate_order(rows, TIERS)["prefixes"][-1]["added_passes"]
    guard_2c = {
        "union_all_tiers": union_all,
        "unlimited_budget_cascade_added_passes": unlimited_cascade,
        "passed": union_all == unlimited_cascade,
    }

    t0_zero_passed, t0_zero_cost = _t0_outcome([], 0)
    guard_2d = {
        "t0_at_k0_passed": t0_zero_passed,
        "t0_at_k0_cost": t0_zero_cost,
        "passed": (t0_zero_passed is False) and (t0_zero_cost == 0),
    }

    guard_2 = {
        "one_tier_equals_arm": guard_2a,
        "full_cascade_equals_union": guard_2c,
        "t0_zero_equals_baseline": guard_2d,
    }
    guard_2_passed = (
        all(entry["passed"] for entry in guard_2a.values())
        and guard_2c["passed"]
        and guard_2d["passed"]
    )

    # Guard 3: selection_sha256 identical across the cohort's three arms.
    hashes = loaded["selection_sha256_by_arm"]
    distinct = {value for value in hashes.values() if value}
    guard_3 = {"selection_sha256_by_arm": hashes, "passed": len(distinct) <= 1}

    return {
        "guard_1_reproduction": guard_1,
        "guard_2_identity": {**guard_2, "passed": guard_2_passed},
        "guard_3_selection_sha256": guard_3,
    }


# --------------------------------------------------------------------------
# Pre-registration scoring
# --------------------------------------------------------------------------


def score_pre_registrations(
    three_tier: dict[str, Any],
    targeting: dict[str, Any],
    marginal: dict[str, Any],
    mbpp_targeting_or_none: dict[str, Any] | None,
    complementarity_guard: dict[str, Any] | None,
    *,
    is_mbpp: bool,
) -> list[dict[str, Any]]:
    """Read already-computed numbers and score each pre-registration.

    Nothing is computed here that was not already computed by the functions
    above -- this only interprets numbers already in hand.
    """
    scored = []
    for entry in PRE_REGISTRATIONS:
        scored.append(
            {**entry, "cohort_scope": None, "observed": None, "verdict": None}
        )

    # PR-1: only meaningfully scored where the three-tier table is available
    # (both cohorts run it; the pre-registration is stated in terms of the
    # HumanEval numbers but the same falsifier applies everywhere).
    pr1 = scored[0]
    pr1["observed"] = {
        "reaches_ceiling_without_t3": three_tier["reaches_ceiling_without_t3"],
        "two_tier_repairer_model_calls": three_tier["two_tier_repairer_model_calls"],
        "min_three_tier_cost_at_ceiling": three_tier["min_three_tier_cost_at_ceiling"],
        "two_tier_strictly_cheaper_than_every_t3_order": three_tier[
            "two_tier_strictly_cheaper_than_every_t3_order"
        ],
    }
    pr1["verdict"] = (
        "held"
        if (
            three_tier["reaches_ceiling_without_t3"]
            and three_tier["two_tier_strictly_cheaper_than_every_t3_order"]
        )
        else "falsified"
    )

    # PR-2 / PR-3: scored from targeting_control().
    pr2 = scored[1]
    pr2["observed"] = {
        "cascade_added_passes": targeting["cascade_added_passes"],
        "random_mean_added_passes": targeting["random_control"]["mean"],
        "ratio_vs_random": targeting["ratio_vs_random"],
    }
    ratio = targeting["ratio_vs_random"]
    pr2["verdict"] = "held" if (ratio is not None and ratio >= 1.3) else "falsified"

    pr3 = scored[2]
    pr3["observed"] = {
        "cascade_added_passes": targeting["cascade_added_passes"],
        "predicted_p_pass_oof_added_passes": targeting[
            "predicted_p_pass_oof_added_passes"
        ],
        "predicted_initial_chars_added_passes": targeting[
            "predicted_initial_chars_added_passes"
        ],
    }
    cascade_val = targeting["cascade_added_passes"]
    within_10pct = lambda value: cascade_val > 0 and value >= 0.9 * cascade_val
    p_pass_close = within_10pct(targeting["predicted_p_pass_oof_added_passes"])
    chars_close = within_10pct(targeting["predicted_initial_chars_added_passes"])
    pr3["verdict"] = "falsified" if (p_pass_close or chars_close) else "held"
    pr3["observed"]["p_pass_within_10pct"] = p_pass_close
    pr3["observed"]["chars_within_10pct"] = chars_close

    # PR-4: from marginal_return_table's crossover.
    pr4 = scored[3]
    pr4["observed"] = {
        "crossover_k": marginal["crossover_k"],
        "entries": marginal["entries"],
    }
    pr4["verdict"] = "held" if marginal["crossover_k"] is not None else "falsified"

    # PR-5: scored only where the calling context supplies MBPP's targeting
    # numbers; for a single-cohort HumanEval-only run it is left unscored.
    pr5 = scored[4]
    if mbpp_targeting_or_none is not None:
        gain = mbpp_targeting_or_none["cascade_added_passes"]
        pr5["observed"] = {"mbpp_cascade_added_passes": gain}
        pr5["verdict"] = (
            "falsified" if gain >= PR5_MATERIAL_GAIN_COORDINATES else "held"
        )
    else:
        pr5["observed"] = None
        pr5["verdict"] = "not scored (needs both cohorts)"

    return scored


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_cohort_analysis(
    cohort_name: str,
    spec: dict[str, Any],
    *,
    sample_index: int = 0,
    max_k0: int = 9,
    n_seeds: int = 200,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run every cascade question for one cohort."""
    loaded = load_cohort(cohort_name, spec, sample_index=sample_index, max_k0=max_k0)
    rows = loaded["rows"]

    single_arm = single_arm_baselines(rows)
    orders_table = all_orders_table(rows)
    three_tier = three_tier_cascade_table(rows)
    targeting = targeting_control(rows, n_seeds=n_seeds, seed=seed)
    budget_sweep = budget_sweep_within_failing_set(rows, seed=seed)
    nested = nested_three_tier_order_selection(rows, n_splits=n_splits, seed=seed)
    marginal = marginal_return_table(rows, loaded["raw_arms"])
    guards = run_guards(cohort_name, spec, loaded)

    return {
        "cohort_name": cohort_name,
        "config": {
            "sample_index": sample_index,
            "max_k0": max_k0,
            "n_seeds": n_seeds,
            "n_splits": n_splits,
            "seed": seed,
            "source_run_id": loaded["source_run_id"],
        },
        "n_coordinates": len(rows),
        "single_arm_baselines": single_arm,
        "all_orders_table": orders_table,
        "three_tier_cascade": three_tier,
        "targeting_control": targeting,
        "budget_sweep_within_failing_set": budget_sweep,
        "nested_three_tier_order_selection": nested,
        "marginal_return_table": marginal,
        "guards": guards,
    }


def plot_cascade(analyses: dict[str, dict[str, Any]], output: Path) -> None:
    """Two panels: the three-tier cost/gain table, and the targeting control."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    axis = axes[0]
    width = 0.38
    names = list(analyses)
    for offset, name in enumerate(names):
        three_tier = analyses[name]["three_tier_cascade"]
        labels = ["T1", "T1->T2", "T1->T2->T3\n(min cost)"]
        values = [
            analyses[name]["single_arm_baselines"]["T1"]["added_passes"],
            three_tier["two_tier_added_passes"],
            three_tier["ceiling_added_passes"],
        ]
        positions = np.arange(len(labels)) + (offset - 0.5) * width
        axis.bar(positions, values, width=width, label=name)
    axis.set_xticks(range(3))
    axis.set_xticklabels(["T1 alone", "T1->T2", "ceiling\n(+T3)"])
    axis.set_ylabel("coordinates repaired")
    axis.set_title("T3 adds zero to the ceiling")
    axis.legend(fontsize=8)

    axis = axes[1]
    x = np.arange(len(names))
    cascade_vals = [
        analyses[n]["targeting_control"]["cascade_added_passes"] for n in names
    ]
    random_means = [
        analyses[n]["targeting_control"]["random_control"]["mean"] for n in names
    ]
    random_stds = [
        analyses[n]["targeting_control"]["random_control"]["std"] for n in names
    ]
    p_pass_vals = [
        analyses[n]["targeting_control"]["predicted_p_pass_oof_added_passes"]
        for n in names
    ]
    chars_vals = [
        analyses[n]["targeting_control"]["predicted_initial_chars_added_passes"]
        for n in names
    ]
    bar_width = 0.2
    axis.bar(x - 1.5 * bar_width, cascade_vals, bar_width, label="verifier (cascade)")
    axis.bar(
        x - 0.5 * bar_width,
        random_means,
        bar_width,
        yerr=random_stds,
        capsize=3,
        label="random",
    )
    axis.bar(x + 0.5 * bar_width, p_pass_vals, bar_width, label="p_pass_oof")
    axis.bar(x + 1.5 * bar_width, chars_vals, bar_width, label="initial_chars")
    axis.set_xticks(x)
    axis.set_xticklabels(names)
    axis.set_ylabel("net added passes at matched escalation size n")
    axis.set_title("Targeting the escalation budget")
    axis.legend(fontsize=8)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _print_summary(analysis: dict[str, Any]) -> None:
    name = analysis["cohort_name"]
    print(f"\n=== {name} ({analysis['n_coordinates']} coordinates)")
    print("single-arm baselines:")
    for tier in TIERS:
        entry = analysis["single_arm_baselines"][tier]
        print(
            f"  {tier:3s} added={entry['added_passes']:4d}  "
            f"base_calls={entry['base_model_calls']:5d}  "
            f"repairer_calls={entry['repairer_model_calls']:5d}"
        )
    three_tier = analysis["three_tier_cascade"]
    print(
        f"three-tier ceiling={three_tier['ceiling_added_passes']}  "
        f"T1->T2={three_tier['two_tier_added_passes']} at "
        f"{three_tier['two_tier_repairer_model_calls']} calls  "
        f"min-cost-with-T3={three_tier['min_three_tier_cost_at_ceiling']} calls"
    )
    targeting = analysis["targeting_control"]
    print(
        f"targeting: cascade={targeting['cascade_added_passes']} vs "
        f"random={targeting['random_control']['mean']:.2f}+-"
        f"{targeting['random_control']['std']:.2f} "
        f"(ratio={targeting['ratio_vs_random']:.2f}) "
        f"p_pass_oof={targeting['predicted_p_pass_oof_added_passes']} "
        f"chars={targeting['predicted_initial_chars_added_passes']}"
    )
    marginal = analysis["marginal_return_table"]
    print(f"deepen-vs-switch crossover k = {marginal['crossover_k']}")
    guards = analysis["guards"]
    print(
        "guards: "
        f"reproduction={guards['guard_1_reproduction']['passed']} "
        f"identity={guards['guard_2_identity']['passed']} "
        f"selection_sha256={guards['guard_3_selection_sha256']['passed']}"
    )


def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Replay the Week 9 repair arms as a verifier-triggered cascade"
    )
    parser.add_argument(
        "--cohort", choices=["humaneval", "mbpp", "both"], default="both"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-k0", type=int, default=9)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--prefix", default="repair_cascade")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    set_all_seeds(args.seed)

    names = ["humaneval", "mbpp"] if args.cohort == "both" else [args.cohort]
    analyses: dict[str, dict[str, Any]] = {}
    for name in names:
        LOGGER.info("scoring cohort %s", name)
        analyses[name] = build_cohort_analysis(
            name,
            COHORTS[name],
            sample_index=args.sample_index,
            max_k0=args.max_k0,
            n_seeds=args.n_seeds,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        _print_summary(analyses[name])

    pre_registrations = score_pre_registrations(
        analyses.get("humaneval", {}).get("three_tier_cascade")
        or next(iter(analyses.values()))["three_tier_cascade"],
        analyses.get("humaneval", {}).get("targeting_control")
        or next(iter(analyses.values()))["targeting_control"],
        analyses.get("humaneval", {}).get("marginal_return_table")
        or next(iter(analyses.values()))["marginal_return_table"],
        analyses.get("mbpp", {}).get("targeting_control"),
        None,
        is_mbpp=False,
    )

    print("\n=== Pre-registrations ===")
    for entry in pre_registrations:
        print(f"{entry['id']}: {entry['verdict']}")
        print(f"  {entry['statement']}")

    figure = args.figure_dir / f"{args.prefix}_cost_and_targeting.png"
    if len(analyses) > 1:
        plot_cascade(analyses, figure)
    else:
        plot_cascade(analyses, figure)

    path = save_results(
        {
            "analysis": "week13_repair_cascade",
            "timestamp": datetime.now().isoformat(),
            "figure": str(figure),
            "cohorts": analyses,
            "pre_registrations": pre_registrations,
        },
        prefix=args.prefix,
    )
    print(f"\nfigure: {figure}")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
