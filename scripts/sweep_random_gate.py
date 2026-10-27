"""Week 8 (joint): turn Week 7's single random control into a distribution.

Week 7 compared risk-gated Smart Repair against **one** seeded random gate at
the same coverage and lost on both axes -- the control captured 9 of Repair-
All's 10 successes to Smart's 7, and spent 106 model calls to Smart's 120. One
seed cannot say whether that control was typical or lucky, and the Week 7
report says so explicitly.

This script replays the random gate over many seeds against the same frozen
repair trace pool, so the comparison becomes a percentile rather than an
anecdote. Because the pool is fixed and every gate is a subset of coordinates
that Repair-All already repaired, no model is loaded and no code is generated:
each additional seed costs a set intersection.

Both axes are reported jointly. A gate that captures more successes by
spending more calls has not beaten a budget -- the honest question is whether
Smart is dominated, i.e. whether a random draw matches or beats it on successes
*and* costs no more calls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_smart_repair import (  # noqa: E402
    align_oof_scores,
    load_json,
    select_cohort,
    select_random_gate,
    select_risk_gate,
    summarize_condition,
    validate_repair_pool,
)
from src.utils import get_logger, save_results  # noqa: E402

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("results/mbpp_uncertainty_20260712_024909.json")
DEFAULT_PREDICTOR = Path("results/predictor_mbpp_20260712_212240.json")
DEFAULT_REPAIR = Path("results/mbpp_week7_repair_all_sample0_qwen15b.json")
DEFAULT_FIGURE_DIR = Path("reports/figures")


def _percentile_of(values: np.ndarray, observed: float) -> dict[str, float]:
    """Where `observed` sits in `values`, reported without assuming no ties."""
    return {
        "fraction_below": float((values < observed).mean()),
        "fraction_equal": float((values == observed).mean()),
        "fraction_above": float((values > observed).mean()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def sweep_random_gates(
    source_path: Path,
    source: dict,
    predictor: dict,
    repair_run: dict,
    *,
    sample_index: int = 0,
    smart_coverage: float = 0.75,
    n_seeds: int = 200,
    first_seed: int = 0,
    feature_set: str = "self_consistency_plus_execution",
    model_type: str = "logistic",
) -> dict[str, Any]:
    """Replay the random gate over `n_seeds` seeds and locate Smart within it.

    Args:
        source_path: Path of the generation run, used for pool validation.
        source: Loaded generation run JSON.
        predictor: Loaded predictor JSON carrying out-of-fold probabilities.
        repair_run: Loaded Repair-All trace pool.
        sample_index: Source sample index forming the cohort.
        smart_coverage: Gate coverage shared by the smart and random arms.
        n_seeds: Number of random gates to draw.
        first_seed: Lowest seed; seeds run first_seed .. first_seed+n_seeds-1.
        feature_set: Predictor feature set to read.
        model_type: Predictor model to read.
    Returns:
        Result dict ready for save_results().
    """
    if not 0.0 <= smart_coverage <= 1.0:
        raise ValueError("smart_coverage must be between zero and one")
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")

    scored = align_oof_scores(source, predictor, feature_set, model_type)
    cohort = select_cohort(scored, sample_index=sample_index)
    repair_by_coordinate = validate_repair_pool(
        source_path, cohort, repair_run, sample_index
    )
    gate_count = math.ceil(smart_coverage * len(cohort))

    smart_summary, _ = summarize_condition(
        "smart", cohort, select_risk_gate(cohort, gate_count), repair_by_coordinate
    )
    all_summary, _ = summarize_condition(
        "repair_all",
        cohort,
        {(row["task_id"], int(row["sample_index"])) for row in cohort},
        repair_by_coordinate,
    )

    draws = []
    for seed in range(first_seed, first_seed + n_seeds):
        gate = select_random_gate(cohort, gate_count, seed)
        summary, _ = summarize_condition("random", cohort, gate, repair_by_coordinate)
        draws.append(
            {
                "seed": seed,
                "added_passes": summary["added_passes"],
                "repair_model_calls": summary["repair_model_calls"],
                "attempted_repairs": summary["attempted_repairs"],
                "final_passed": summary["final_passed"],
            }
        )

    added = np.array([draw["added_passes"] for draw in draws], dtype=float)
    calls = np.array([draw["repair_model_calls"] for draw in draws], dtype=float)
    smart_added = float(smart_summary["added_passes"])
    smart_calls = float(smart_summary["repair_model_calls"])

    # A draw "dominates" Smart when it is no worse on successes and no worse on
    # cost, with at least one strict -- the only comparison that is fair when
    # the two arms are coverage-matched but not call-matched.
    at_least_as_good = (added >= smart_added) & (calls <= smart_calls)
    strictly_better = at_least_as_good & ((added > smart_added) | (calls < smart_calls))

    return {
        "analysis": "joint_random_gate_sweep",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "source_results": str(source_path),
            "source_run_id": source.get("run_id"),
            "repair_run_id": repair_run.get("run_id"),
            "repair_model": repair_run.get("config", {}).get("model"),
            "predictor_feature_set": feature_set,
            "predictor_model_type": model_type,
            "probability_kind": "out_of_fold_raw_rank_only",
            "sample_index": sample_index,
            "smart_coverage": smart_coverage,
            "gate_count": gate_count,
            "n_seeds": n_seeds,
            "first_seed": first_seed,
            "shared_repair_trace_pool": True,
        },
        "cohort": {
            "candidates": len(cohort),
            "initially_passing": sum(bool(row["initial_passed"]) for row in cohort),
            "initially_failing": sum(not row["initial_passed"] for row in cohort),
        },
        "repair_all": {
            "added_passes": all_summary["added_passes"],
            "repair_model_calls": all_summary["repair_model_calls"],
        },
        "smart": {
            "added_passes": smart_summary["added_passes"],
            "repair_model_calls": smart_summary["repair_model_calls"],
            "attempted_repairs": smart_summary["attempted_repairs"],
            "final_passed": smart_summary["final_passed"],
        },
        "random_distribution": {
            "added_passes": _percentile_of(added, smart_added),
            "repair_model_calls": _percentile_of(calls, smart_calls),
        },
        "joint": {
            "fraction_at_least_as_good_on_both": float(at_least_as_good.mean()),
            "fraction_strictly_dominating_smart": float(strictly_better.mean()),
            # scipy's Spearman, not Pearson-on-argsorted-ranks: added_passes
            # takes ~7 distinct values over 200 draws, so the two disagree.
            # Undefined when either axis is constant, which full coverage makes it.
            "spearman_added_vs_calls": (
                float(spearmanr(added, calls).statistic)
                if added.std() > 0 and calls.std() > 0
                else None
            ),
        },
        "draws": draws,
    }


def plot_sweep(sweep: dict[str, Any], figure_dir: Path) -> Path:
    """Scatter the random draws on the (calls, successes) plane with Smart marked."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    draws = sweep["draws"]
    calls = [draw["repair_model_calls"] for draw in draws]
    added = [draw["added_passes"] for draw in draws]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    rng = np.random.default_rng(0)
    # Jitter only the discrete success axis so overlapping draws stay countable.
    ax.scatter(
        calls,
        np.asarray(added) + rng.uniform(-0.16, 0.16, len(added)),
        s=22,
        alpha=0.45,
        color="#4C72B0",
        edgecolor="none",
        label=f"random gate, {sweep['config']['n_seeds']} seeds",
    )
    ax.scatter(
        [sweep["smart"]["repair_model_calls"]],
        [sweep["smart"]["added_passes"]],
        s=170,
        marker="*",
        color="#C44E52",
        zorder=5,
        label="Smart Repair (risk-gated)",
    )
    ax.axhline(
        sweep["repair_all"]["added_passes"],
        color="#55A868",
        linestyle="--",
        linewidth=1.2,
        label=f"Repair-All ceiling ({sweep['repair_all']['added_passes']} passes,"
        f" {sweep['repair_all']['repair_model_calls']} calls)",
    )
    coverage = sweep["config"]["smart_coverage"]
    dominating = sweep["joint"]["fraction_at_least_as_good_on_both"]
    ax.set_xlabel("repair model calls (lower is better)")
    ax.set_ylabel("added passes (higher is better)")
    ax.set_title(
        f"Risk gate vs. random gate at {coverage:.0%} coverage\n"
        f"{dominating:.1%} of random draws match or beat Smart on both axes"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()

    path = figure_dir / "week8_random_gate_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the Smart Repair random control over many seeds"
    )
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--predictor-results", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--repair-results", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--smart-coverage", type=float, default=0.75)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--feature-set", default="self_consistency_plus_execution")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--prefix", default="analysis_week8_random_gate_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = sweep_random_gates(
        args.source_results,
        load_json(args.source_results),
        load_json(args.predictor_results),
        load_json(args.repair_results),
        sample_index=args.sample_index,
        smart_coverage=args.smart_coverage,
        n_seeds=args.n_seeds,
        first_seed=args.first_seed,
        feature_set=args.feature_set,
        model_type=args.model_type,
    )

    smart = sweep["smart"]
    added = sweep["random_distribution"]["added_passes"]
    calls = sweep["random_distribution"]["repair_model_calls"]
    print(
        f"Smart@{args.smart_coverage:.0%}: {smart['added_passes']} added passes, "
        f"{smart['repair_model_calls']} calls"
    )
    print(
        f"  random added_passes : mean {added['mean']:.2f} +- {added['std']:.2f} "
        f"[{added['min']:.0f}, {added['max']:.0f}]  "
        f"below Smart {added['fraction_below']:.1%} / equal "
        f"{added['fraction_equal']:.1%} / above {added['fraction_above']:.1%}"
    )
    print(
        f"  random model_calls  : mean {calls['mean']:.1f} +- {calls['std']:.1f} "
        f"[{calls['min']:.0f}, {calls['max']:.0f}]  "
        f"below Smart {calls['fraction_below']:.1%}"
    )
    print(
        f"  random draws matching or beating Smart on BOTH axes: "
        f"{sweep['joint']['fraction_at_least_as_good_on_both']:.1%} "
        f"(strictly dominating: "
        f"{sweep['joint']['fraction_strictly_dominating_smart']:.1%})"
    )

    figure = plot_sweep(sweep, args.figure_dir)
    sweep["figure"] = str(figure)
    path = save_results(sweep, prefix=args.prefix)
    print(f"figure: {figure}")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
