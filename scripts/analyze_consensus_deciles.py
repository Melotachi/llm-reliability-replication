"""Week 7: does high consensus mean correctness — and where does it fail?

Student A deliverable. Code-MUE (ISSTA 2026) and every other consensus-based
code-UQ method rests on an assumed "Semantic Consensus Hypothesis": correct
solutions form dense agreement cliques while incorrect ones fragment. Nobody
has tested it below 6.7B. This project's stored runs are at 350M-1.3B, so it
can.

Two numbers per run, per feature:

  1. **Solved fraction by consensus quartile.** If the hypothesis holds, Q1
     (most agreement) should solve far more often than Q4. On the merged-164
     codegen run this spread is 0.537 -> 0.024 even though the model's overall
     pass@1 is 0.091 — consensus works at the small end.
  2. **The confidently-wrong rate**: within the top-consensus decile, how often
     do *all* N samples fail? This is the residual every paper names as its
     limitation and none measures. On MBPP it sits at 1/3 regardless of model
     capability; on HumanEval it falls to 0 as capability rises.

Why both features, side by side: CLAUDE.md Section 13(g) records the
"confidently wrong majority" cell as empty in all four Week 6 cells, but that
used the **execution-behavior** feature at sample level with a stricter joint
definition, while the quartile numbers above use **AST diversity** at problem
level. Different feature, threshold and unit — no contradiction, but a reader
will notice, so this script computes both framings on the same problems and
prints them together. If the execution channel really has a 0% confidently-
wrong rate where AST has 23.5%, that is a stronger argument for it than
anything currently in the Week 6 report.

# TODO: SANDBOX — with --execution this replays stored generated code in
# subprocesses (same caveat as evaluation.run_code). Research/local use only.

Usage:
    python scripts/analyze_consensus_deciles.py --results results/<run>.json
    python scripts/analyze_consensus_deciles.py --results ... --no-execution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_uncertainty import load_run_prompts
from scripts.train_predictor import (
    add_execution_behavior_features,
    build_problem_level_dataset,
    build_sample_level_dataset,
)
from src.uncertainty import (
    DEFAULT_ERROR_COSTS,
    LEGACY_ERROR_COSTS,
    METRIC_REPRESENTATIONS,
)
from src.utils import get_logger, save_results, set_all_seeds

logger = get_logger(__name__)

# The consensus features to bin problems by. Both are "higher = samples
# disagree more", so the most-consensus group is the LOWEST quantile.
CONSENSUS_FEATURES = ["ast_node_diversity", "execution_behavior_diversity"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the consensus-decile entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic Consensus Hypothesis: quartile spread and the "
        "confidently-wrong rate, per consensus feature"
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="Pipeline results JSON to analyze."
    )
    parser.add_argument(
        "--representation",
        choices=METRIC_REPRESENTATIONS,
        default="full",
        help="Metric input representation for the text/structural features "
        "(Week 7 prompt-inclusion confound). The execution feature is "
        "unaffected either way.",
    )
    parser.add_argument(
        "--no-execution",
        action="store_true",
        help="Skip the execution-behavior feature (no subprocess replay).",
    )
    parser.add_argument(
        "--probe-source",
        choices=["test", "evalplus"],
        default="evalplus",
        help="Probe provenance for the execution feature (default: evalplus, "
        "matching the Week 6 four-cell ablation).",
    )
    parser.add_argument("--probe-cap", type=int, default=12, help="Max probes/problem.")
    parser.add_argument(
        "--execution-timeout", type=float, default=5.0, help="Replay limit (s)."
    )
    parser.add_argument(
        "--legacy-error-costs",
        action="store_true",
        help="Use the pre-graded (1, 1, 0) abnormal-outcome costs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print but do not persist the stats JSON.",
    )
    return parser.parse_args()


def quantile_table(
    rows: list[dict], feature: str, n_groups: int = 4
) -> list[dict] | None:
    """Solved fraction and mean pass rate per consensus quantile.

    Args:
        rows: Problem-level rows with the feature, "solved" and "pass_rate".
        feature: Consensus feature to bin by (higher = less consensus).
        n_groups: 4 for quartiles, 10 for deciles.
    Returns:
        One dict per group, group 1 = most consensus; None when the feature
        is absent or constant (nothing to bin).
    """
    values = np.array([row.get(feature, np.nan) for row in rows], dtype=float)
    if np.isnan(values).any() or np.ptp(values) == 0.0:
        return None
    # Rank-based grouping, so ties and skewed feature distributions cannot
    # produce empty groups the way fixed-width bins would.
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, n_groups)
    table = []
    for index, group in enumerate(groups, start=1):
        if len(group) == 0:
            continue
        solved = [rows[i]["solved"] for i in group]
        pass_rates = [rows[i]["pass_rate"] for i in group]
        table.append(
            {
                "group": index,
                "n_problems": int(len(group)),
                "frac_solved": float(np.mean(solved)),
                "mean_pass_rate": float(np.mean(pass_rates)),
                "task_ids_all_fail": [
                    rows[i]["task_id"] for i in group if not rows[i]["solved"]
                ],
            }
        )
    return table


def confidently_wrong_rate(rows: list[dict], feature: str) -> dict | None:
    """Fraction of the top-consensus decile whose samples ALL fail.

    Reported with the denominator explicit: the top decile is 6 problems at
    n=60 and 17 at n=164, so only the larger figure has enough support to
    quote on its own.
    """
    table = quantile_table(rows, feature, n_groups=10)
    if not table:
        return None
    top = table[0]
    return {
        "n_problems": top["n_problems"],
        "n_all_fail": len(top["task_ids_all_fail"]),
        "rate": (
            len(top["task_ids_all_fail"]) / top["n_problems"]
            if top["n_problems"]
            else None
        ),
        "task_ids": top["task_ids_all_fail"],
    }


def print_report(analysis: dict) -> None:
    """Print the quartile spread and confidently-wrong rate per feature."""
    for feature, entry in analysis["features"].items():
        print(f"\n=== consensus feature: {feature} ===")
        if entry.get("unavailable"):
            print(f"  unavailable ({entry['unavailable']})")
            continue
        print(f"{'quartile':<22}{'n':<6}{'frac solved':<14}{'mean pass rate':<16}")
        last = len(entry["quartiles"])
        for group in entry["quartiles"]:
            note = {1: " (most consensus)", last: " (least)"}.get(group["group"], "")
            label = f"Q{group['group']}{note}"
            print(
                f"{label:<22}{group['n_problems']:<6}"
                f"{group['frac_solved']:<14.3f}{group['mean_pass_rate']:<16.3f}"
            )
        spread = (
            entry["quartiles"][0]["frac_solved"] - entry["quartiles"][-1]["frac_solved"]
        )
        # Report the Q1-Q4 *difference*, not the ratio: a Q4 solved-fraction
        # of 0.024 turns the ratio into a 22x artefact of a small denominator.
        print(f"{'Q1 - Q4 spread':<22}{'':<6}{spread:<14.3f}")
        wrong = entry["confidently_wrong"]
        if wrong:
            print(
                f"top-decile all-fail: {wrong['n_all_fail']}/{wrong['n_problems']} "
                f"= {wrong['rate']:.3f}   ids: {wrong['task_ids']}"
            )


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    with args.results.open() as f:
        results = json.load(f)

    prompts = load_run_prompts(results)
    problem_rows = build_problem_level_dataset(
        results,
        include_codebert=False,
        codebert_model="microsoft/codebert-base",
        representation=args.representation,
        prompts=prompts,
    )
    if not args.no_execution:
        logger.info("Computing execution-behavior features (replaying stored samples)")
        add_execution_behavior_features(
            results,
            problem_rows,
            build_sample_level_dataset(results, False, "microsoft/codebert-base"),
            args.execution_timeout,
            probe_source=args.probe_source,
            probe_cap=args.probe_cap,
            error_costs=(
                LEGACY_ERROR_COSTS if args.legacy_error_costs else DEFAULT_ERROR_COSTS
            ),
            seed=args.seed,
        )

    features: dict[str, dict] = {}
    for feature in CONSENSUS_FEATURES:
        quartiles = quantile_table(problem_rows, feature, n_groups=4)
        if quartiles is None:
            features[feature] = {"unavailable": "feature absent or constant"}
            continue
        features[feature] = {
            "quartiles": quartiles,
            "confidently_wrong": confidently_wrong_rate(problem_rows, feature),
        }

    analysis = {
        "analysis": "week7_consensus_deciles",
        "source_results_file": str(args.results),
        "source_run_id": results.get("run_id"),
        "config": {
            "seed": args.seed,
            "representation": args.representation,
            "execution_features": not args.no_execution,
            "probe_source": None if args.no_execution else args.probe_source,
            "probe_cap": args.probe_cap,
        },
        "source_config": results.get("config", {}),
        "n_problems": len(problem_rows),
        "overall_frac_solved": float(np.mean([r["solved"] for r in problem_rows])),
        "features": features,
    }
    print_report(analysis)

    if args.dry_run:
        logger.info("[DRY RUN] skipping stats JSON persistence")
    else:
        out_path = save_results(analysis, prefix="analysis_week7_consensus")
        logger.info("Saved consensus-decile stats to %s", out_path)


if __name__ == "__main__":
    main()
