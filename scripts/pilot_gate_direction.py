"""Week 12. Turn the post-hoc gate direction into a rule that pays for itself.

Section 6.5 of the paper reverses the gate's ordering and recovers Repair-All
accuracy at 47% of its budget on HumanEval -- then concedes in Section 8.1 that
"the direction was chosen after seeing that cohort's allocation AUC, on the same
coordinates it is then evaluated over, so its numbers are in-sample for the
choice of sign." That is a real threat and it has never been discharged.

A deployable version cannot look at the answer. It can, however, *buy* it: spend
the first epsilon of the repair budget on a random pilot of failures, read the
sign of the P(pass)-to-repair-success association off the pilot alone, and commit
to that ordering for the remainder. The pilot is not overhead in the usual sense
-- its repairs count towards the total, so the only cost is that those
coordinates were chosen at random rather than by the (as yet unknown) direction.

Two things are measured.

* **Does the rule pay?** Added passes against model calls for the pilot rule at
  several epsilons, against the four fixed policies it has to beat: always-risk
  (the shipped gate), always-inverted (the post-hoc winner, an upper bound the
  rule is not allowed to know), random, and Repair-All.
* **How big must the pilot be?** The sign is a coin flip until the pilot can
  resolve it. Resampling pilots of each size from the frozen pool gives the
  probability that a pilot of that size agrees with the cohort's full-data sign,
  which is the number a deployment actually needs.

Week 12's arm analysis adds a precondition the paper's version did not have: the
sign question is only well posed where an allocation signal exists at all, which
is the *context* arms. On the `none` (resample) arm the association is at chance
in both cohorts, so a pilot there is buying a coin flip -- and the rule is scored
on that arm too, precisely so the failure is visible rather than assumed away.

Usage:
    python scripts/pilot_gate_direction.py --cohort both
    python scripts/pilot_gate_direction.py --cohort humaneval --n-seeds 400
"""

from __future__ import annotations

import argparse
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

from scripts.analyze_smart_repair import (  # noqa: E402
    _coordinate,
    align_oof_scores,
    load_json,
    select_cohort,
    validate_repair_pool,
)
from scripts.diagnose_arm_allocation import (  # noqa: E402
    COHORTS,
    renormalize_stop_on_pass,
)
from scripts.diagnose_gate_target import roc_auc  # noqa: E402
from src.utils import get_logger, save_results, set_all_seeds  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_FIGURE_DIR = Path("reports/figures")
DEFAULT_EPSILONS = (0.10, 0.20, 0.30, 0.50)
DEFAULT_PILOT_SIZES = (5, 10, 15, 20, 30, 40, 50, 75)


def build_rows(
    source: dict,
    predictor: dict,
    repair_run: dict,
    source_path: Path,
    *,
    sample_index: int,
    feature_set: str,
    model_type: str,
) -> list[dict[str, Any]]:
    """One row per initially-failing coordinate: score, cost and outcome.

    Args:
        source: Uncertainty run supplying the cohort.
        predictor: Predictor run supplying out-of-fold probabilities.
        repair_run: Repair pool, already renormalized to stop-on-pass.
        source_path: Path of the uncertainty run, for the pool guards.
        sample_index: Candidate index the cohort is built from.
        feature_set: Predictor arm supplying the probabilities.
        model_type: Predictor model supplying the probabilities.
    Returns:
        Rows with p_pass, calls, repaired and the source order (tie-break).
    """
    scored = align_oof_scores(source, predictor, feature_set, model_type)
    cohort = select_cohort(scored, sample_index=sample_index)
    repair_by_coordinate = validate_repair_pool(
        source_path, cohort, repair_run, sample_index
    )
    rows = []
    for order, entry in enumerate(cohort):
        if entry["initial_passed"]:
            continue
        record = repair_by_coordinate[_coordinate(entry)]
        rows.append(
            {
                "task_id": entry["task_id"],
                "p_pass": float(entry["p_pass_oof"]),
                # Week 12's repairability table found the failing program's own
                # length outranking P(pass) on both HumanEval context arms
                # (univariate 0.736/0.700 sign-reversed against 0.706/0.648), so
                # it is carried here as a rival *ordering*, not just a covariate.
                "initial_chars": float(len(record.get("initial_code", ""))),
                "calls": int(record["repair_model_calls"]),
                "repaired": int(bool(record["final_passed"])),
                "source_order": order,
            }
        )
    return rows


def _curve(ordering: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """Cumulative (calls, added passes, attempts) along an attempt ordering."""
    calls, added, attempts = 0, 0, 0
    points = [(0, 0, 0)]
    for row in ordering:
        calls += row["calls"]
        added += row["repaired"]
        attempts += 1
        points.append((calls, added, attempts))
    return points


def _at_budget(points: list[tuple[int, int, int]], budget: int) -> tuple[int, int]:
    """Added passes and attempts at the LAST point affordable within `budget`.

    Attempts must be reported alongside the gain. Under stop-on-pass a coordinate
    costs 1-3 calls, and any ordering correlated with cost buys more attempts for
    the same budget -- so two orderings compared at equal *calls* are not compared
    at equal *attempts*. Week 8 flagged exactly this for the inverted gate
    ("coverage is over all 164 candidates, so the arms differ in attempts at equal
    coverage"), and an ordering by program length is correlated with cost by
    construction: short programs are cheap to repair and repair early.

    The last affordable point, not the first point reaching the affordable maximum.
    `added` is non-decreasing along an ordering and `calls` strictly increasing, so
    the two agree on the *gain* and differ only in the *attempts* column -- and the
    difference is not a tie-break to be argued about, it is a question of what is
    being counted. An operator holding this budget cannot know which of the
    remaining coordinates will pay, so it attempts every one that fits: those
    attempts are really made and really spent, and a policy whose ordering ends in
    a flat run is genuinely paying for it. Reporting the first point that reaches
    the maximum instead would credit every policy with an early stop none of them
    can execute, and would move the arms unequally (on HumanEval `error` at a 25%
    budget: length 45 -> 44 attempts, inverted 42 -> 38).
    """
    best_added, attempts_there = 0, 0
    for calls, added, attempts in points:
        if calls > budget:
            break
        best_added, attempts_there = added, attempts
    return best_added, attempts_there


def _added_at_budget(points: list[tuple[int, int, int]], budget: int) -> int:
    """Added passes affordable within `budget` calls (attempts discarded)."""
    return _at_budget(points, budget)[0]


def fixed_ordering(
    rows: list[dict[str, Any]], *, direction: str, score: str = "p_pass"
) -> list[dict]:
    """Order coordinates by a scalar score.

    Args:
        rows: Output of build_rows().
        direction: "risk" attempts the lowest score first (the shipped gate's
            ordering when the score is P(pass)); "inverted" attempts the highest
            first.
        score: Row key to order by -- "p_pass" for the shipped gate, or
            "initial_chars" for the free length gate.
    Returns:
        Rows in attempt order, ties broken by source order for reproducibility.
    """
    reverse = direction == "inverted"
    return sorted(
        rows, key=lambda row: (row[score], row["source_order"]), reverse=reverse
    )


def pilot_ordering(
    rows: list[dict[str, Any]],
    *,
    epsilon: float,
    rng: np.random.Generator,
    score: str = "p_pass",
) -> tuple[list[dict], str, float | None]:
    """Attempt a random pilot, read the sign off it, then commit.

    The pilot's own repairs count towards the total, so the rule's cost is not the
    pilot but the fact that the pilot was ordered at random. A pilot whose outcomes
    are single-class cannot produce an AUC; the rule then falls back to the shipped
    risk direction, which is what a deployment without evidence would do.

    Args:
        rows: Output of build_rows().
        epsilon: Fraction of coordinates spent on the pilot.
        rng: Seeded generator for the pilot draw.
        score: Row key the ordering and the sign are read from.
    Returns:
        (attempt ordering, chosen direction, pilot AUC or None).
    """
    n_pilot = max(1, math.ceil(epsilon * len(rows)))
    indices = rng.permutation(len(rows))
    pilot = [rows[index] for index in indices[:n_pilot]]
    rest = [rows[index] for index in indices[n_pilot:]]

    labels = np.array([row["repaired"] for row in pilot], dtype=int)
    scores = np.array([row[score] for row in pilot], dtype=float)
    if labels.sum() in (0, labels.size):
        fallback = fixed_ordering(rest, direction="risk", score=score)
        return pilot + fallback, "risk", None
    auc = roc_auc(scores, labels)
    direction = "inverted" if auc > 0.5 else "risk"
    ordering = pilot + fixed_ordering(rest, direction=direction, score=score)
    return ordering, direction, auc


def evaluate_policies(
    rows: list[dict[str, Any]],
    *,
    epsilons: tuple[float, ...],
    n_seeds: int,
    first_seed: int,
    budgets: tuple[float, ...],
    score: str = "p_pass",
) -> dict[str, Any]:
    """Added passes at fractional budgets for the pilot rule and its baselines.

    Budgets are fractions of the Repair-All cost so the table reads the same on
    both cohorts.

    Args:
        rows: Output of build_rows().
        epsilons: Pilot fractions to evaluate.
        n_seeds: Draws averaged for every stochastic policy.
        first_seed: Lowest seed.
        budgets: Fractions of the full repair budget to report.
    Returns:
        Per-policy added passes at each budget, plus the pilot's sign accuracy.
    """
    total_calls = sum(row["calls"] for row in rows)
    total_repaired = sum(row["repaired"] for row in rows)
    full_auc = roc_auc(
        np.array([row[score] for row in rows], dtype=float),
        np.array([row["repaired"] for row in rows], dtype=int),
    )
    true_direction = "inverted" if full_auc > 0.5 else "risk"
    budget_calls = [int(round(fraction * total_calls)) for fraction in budgets]

    policies: dict[str, Any] = {}
    for direction in ("risk", "inverted"):
        points = _curve(fixed_ordering(rows, direction=direction, score=score))
        at = [_at_budget(points, b) for b in budget_calls]
        policies[direction] = {
            "added_at_budget": [value[0] for value in at],
            "attempts_at_budget": [value[1] for value in at],
            "stochastic": False,
        }

    random_added = {budget: [] for budget in budget_calls}
    random_attempts = {budget: [] for budget in budget_calls}
    for seed in range(first_seed, first_seed + n_seeds):
        rng = np.random.default_rng(seed)
        shuffled = [rows[index] for index in rng.permutation(len(rows))]
        points = _curve(shuffled)
        for budget in budget_calls:
            added, attempts = _at_budget(points, budget)
            random_added[budget].append(added)
            random_attempts[budget].append(attempts)
    policies["random"] = {
        "added_at_budget": [float(np.mean(random_added[b])) for b in budget_calls],
        "added_at_budget_std": [
            float(np.std(random_added[b], ddof=1)) for b in budget_calls
        ],
        "attempts_at_budget": [
            float(np.mean(random_attempts[b])) for b in budget_calls
        ],
        "stochastic": True,
    }

    for epsilon in epsilons:
        added = {budget: [] for budget in budget_calls}
        attempted = {budget: [] for budget in budget_calls}
        agreed = 0
        undefined = 0
        for seed in range(first_seed, first_seed + n_seeds):
            rng = np.random.default_rng(seed)
            ordering, direction, auc = pilot_ordering(
                rows, epsilon=epsilon, rng=rng, score=score
            )
            if auc is None:
                undefined += 1
            agreed += int(direction == true_direction)
            points = _curve(ordering)
            for budget in budget_calls:
                gain, attempts = _at_budget(points, budget)
                added[budget].append(gain)
                attempted[budget].append(attempts)
        policies[f"pilot@{epsilon:.2f}"] = {
            "added_at_budget": [float(np.mean(added[b])) for b in budget_calls],
            "added_at_budget_std": [
                float(np.std(added[b], ddof=1)) for b in budget_calls
            ],
            "attempts_at_budget": [float(np.mean(attempted[b])) for b in budget_calls],
            "sign_agreement_rate": agreed / n_seeds,
            "undefined_pilot_rate": undefined / n_seeds,
            "stochastic": True,
        }

    return {
        "score_column": score,
        "n_coordinates": len(rows),
        "total_repair_calls": total_calls,
        "total_repairable": total_repaired,
        "full_data_allocation_auc": full_auc,
        "full_data_direction": true_direction,
        "budget_fractions": list(budgets),
        "budget_calls": budget_calls,
        "policies": policies,
    }


def sign_resolution_power(
    rows: list[dict[str, Any]],
    *,
    pilot_sizes: tuple[int, ...],
    n_draws: int,
    first_seed: int,
    score: str = "p_pass",
) -> dict[str, Any]:
    """How large a pilot must be before it names the cohort's own sign.

    Resamples pilots of each size without replacement from the frozen pool and
    records how often the pilot's sign matches the full-data sign. This is the
    number a deployment needs, and it is deliberately *not* a closed-form power
    calculation: the pool is small enough that the finite-population behaviour is
    the behaviour of interest.

    Args:
        rows: Output of build_rows().
        pilot_sizes: Pilot sizes in coordinates.
        n_draws: Pilots drawn per size.
        first_seed: Lowest seed.
    Returns:
        Per size: agreement rate, undefined rate, and the mean pilot AUC.
    """
    scores = np.array([row[score] for row in rows], dtype=float)
    labels = np.array([row["repaired"] for row in rows], dtype=int)
    full_auc = roc_auc(scores, labels)
    true_direction = "inverted" if full_auc > 0.5 else "risk"

    out = []
    for size in pilot_sizes:
        if size > len(rows):
            continue
        agreed, undefined, aucs = 0, 0, []
        for seed in range(first_seed, first_seed + n_draws):
            rng = np.random.default_rng(seed)
            index = rng.permutation(len(rows))[:size]
            pilot_labels, pilot_scores = labels[index], scores[index]
            if pilot_labels.sum() in (0, pilot_labels.size):
                undefined += 1
                direction = "risk"
            else:
                auc = roc_auc(pilot_scores, pilot_labels)
                aucs.append(auc)
                direction = "inverted" if auc > 0.5 else "risk"
            agreed += int(direction == true_direction)
        out.append(
            {
                "pilot_size": size,
                "sign_agreement_rate": agreed / n_draws,
                "undefined_rate": undefined / n_draws,
                "mean_pilot_auc": float(np.mean(aucs)) if aucs else None,
            }
        )
    return {
        "score_column": score,
        "full_data_auc": full_auc,
        "full_data_direction": true_direction,
        "by_pilot_size": out,
    }


def build_cohort_analysis(
    cohort_name: str,
    spec: dict[str, Any],
    *,
    sample_index: int = 0,
    epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
    pilot_sizes: tuple[int, ...] = DEFAULT_PILOT_SIZES,
    n_seeds: int = 200,
    n_draws: int = 400,
    first_seed: int = 0,
    budgets: tuple[float, ...] = (0.25, 0.40, 0.50, 0.75, 1.00),
    feature_set: str = "self_consistency_plus_execution",
    model_type: str = "logistic",
    scores: tuple[str, ...] = ("p_pass", "initial_chars"),
) -> dict[str, Any]:
    """Score the pilot rule on every repair arm of one cohort, per ordering.

    Two orderings are evaluated side by side: the shipped `p_pass` gate, and the
    free `initial_chars` one that Week 12's repairability table found outranking
    it. The pilot rule buys the *sign* either way, so a length gate needs no
    trained model at all -- it reads a number already on disk.
    """
    source = load_json(spec["source"])
    predictor = load_json(spec["predictor"])

    arms = {}
    for arm_name, pool_path in spec["arms"]:
        pool, _ = renormalize_stop_on_pass(load_json(pool_path))
        rows = build_rows(
            source,
            predictor,
            pool,
            spec["source"],
            sample_index=sample_index,
            feature_set=feature_set,
            model_type=model_type,
        )
        arms[arm_name] = {
            "pool_file": str(pool_path),
            "by_score": {
                score: {
                    "policies": evaluate_policies(
                        rows,
                        epsilons=epsilons,
                        n_seeds=n_seeds,
                        first_seed=first_seed,
                        budgets=budgets,
                        score=score,
                    ),
                    "sign_resolution": sign_resolution_power(
                        rows,
                        pilot_sizes=pilot_sizes,
                        n_draws=n_draws,
                        first_seed=first_seed,
                        score=score,
                    ),
                }
                for score in scores
            },
        }
    return {
        "analysis": "week12_pilot_gate_direction",
        "cohort_name": cohort_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "source_results": str(spec["source"]),
            "predictor_results": str(spec["predictor"]),
            "sample_index": sample_index,
            "epsilons": list(epsilons),
            "pilot_sizes": list(pilot_sizes),
            "n_seeds": n_seeds,
            "n_draws": n_draws,
            "first_seed": first_seed,
            "score_columns": list(scores),
            "budget_fractions": list(budgets),
            "calls_note": "renormalized to stop-on-pass before any comparison",
        },
        "arms": arms,
    }


def plot_pilot(
    analyses: list[dict[str, Any]], output: Path, *, score_column: str = "p_pass"
) -> None:
    """Sign agreement against pilot size, one line per (cohort, arm)."""
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    styles = {"none": ":", "code_only": "--", "error": "-"}
    for analysis in analyses:
        for arm, arm_block in analysis["arms"].items():
            block = arm_block["by_score"][score_column]
            resolution = block["sign_resolution"]
            sizes = [row["pilot_size"] for row in resolution["by_pilot_size"]]
            rates = [row["sign_agreement_rate"] for row in resolution["by_pilot_size"]]
            axis.plot(
                sizes,
                rates,
                linestyle=styles.get(arm, "-"),
                marker="o",
                markersize=3.5,
                label=(
                    f"{analysis['cohort_name']} `{arm}` "
                    f"(full AUC {resolution['full_data_auc']:.2f})"
                ),
            )
    axis.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axis.set_xlabel("pilot size (coordinates repaired before committing)")
    axis.set_ylabel("P(pilot picks the cohort's own sign)")
    axis.set_title(
        f"How large a pilot must be to buy the gate direction (`{score_column}`)"
    )
    axis.set_ylim(0.0, 1.05)
    axis.legend(fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _print_summary(analysis: dict[str, Any]) -> None:
    """Print, per arm, the budget table and the pilot-size curve."""
    print(f"\n=== {analysis['cohort_name']}")
    for arm, arm_block in analysis["arms"].items():
        for score, block in arm_block["by_score"].items():
            _print_arm(arm, score, block)


def _print_arm(arm: str, score: str, block: dict[str, Any]) -> None:
    """Print one (arm, ordering) block: the budget table and the pilot curve."""
    policies = block["policies"]
    fractions = policies["budget_fractions"]
    print(
        f"\n  arm `{arm}` ordered by `{score}`  n={policies['n_coordinates']} "
        f"repairable={policies['total_repairable']} "
        f"budget={policies['total_repair_calls']} calls  "
        f"full-data AUC={policies['full_data_allocation_auc']:.3f} "
        f"-> {policies['full_data_direction']}"
    )
    header = "  ".join(f"{f:>10.0%}" for f in fractions)
    print(f"    {'policy (added/attempts)':16s} {header}   sign-agreement")
    for name, values in policies["policies"].items():
        row = "  ".join(
            f"{v:5.1f}/{a:<4.0f}"
            for v, a in zip(values["added_at_budget"], values["attempts_at_budget"])
        )
        agreement = values.get("sign_agreement_rate")
        suffix = "" if agreement is None else f"   {agreement:.3f}"
        print(f"    {name:16s} {row}{suffix}")
    sizes = block["sign_resolution"]["by_pilot_size"]
    print(
        "    pilot size -> agreement: "
        + ", ".join(f"{r['pilot_size']}:{r['sign_agreement_rate']:.2f}" for r in sizes)
    )


def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="A pilot rule that buys the gate direction instead of assuming it"
    )
    parser.add_argument(
        "--cohort", choices=["mbpp", "humaneval", "both"], default="both"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--n-draws", type=int, default=400)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--prefix", default="analysis_week12_pilot_direction")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    set_all_seeds(args.seed)

    names = ["mbpp", "humaneval"] if args.cohort == "both" else [args.cohort]
    analyses = []
    for name in names:
        LOGGER.info("scoring cohort %s", name)
        analysis = build_cohort_analysis(
            name,
            COHORTS[name],
            sample_index=args.sample_index,
            n_seeds=args.n_seeds,
            n_draws=args.n_draws,
            first_seed=args.first_seed,
        )
        analyses.append(analysis)
        _print_summary(analysis)

    figures = []
    for score_column in ("p_pass", "initial_chars"):
        figure = args.figure_dir / f"week12_pilot_direction_{score_column}.png"
        plot_pilot(analyses, figure, score_column=score_column)
        figures.append(str(figure))
    figure = figures[0]
    path = save_results(
        {
            "analysis": "week12_pilot_gate_direction",
            "timestamp": datetime.now().isoformat(),
            "figures": figures,
            "cohorts": analyses,
        },
        prefix=args.prefix,
    )
    print(f"\nfigure: {figure}")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
