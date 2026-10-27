"""Week 6 trial (Angle A, 2026-07-14 research session): best-of-N selection.

Reframes the Task B sample-level apparatus from passively PREDICTING
pass/fail into actively SELECTING which one of a problem's N stored
samples to output — the "passive indicator vs actionable signal" reframe
(Rathnasuriya & Yang, ICSE-NIER 2026; selection literature: MBR-Exec,
Shi et al. 2022; CodeT, Zhang et al. 2022; AlphaCode, Li et al. 2022;
Semantic Voting, Jiang et al. 2026). Zero new generation: everything is
recomputed from a stored results/*.json run.

Selectors compared per problem (ground truth samples[i]["passed"] is only
ever used for SCORING a selector's pick, never fed to any selector):

  - random_draw: one uniformly random sample (seeded). Its analytic
    expectation, the run's mean initial pass rate (= pass@1), is reported
    separately as "expected_random" — the run's current no-selection
    baseline, i.e. what the pipeline achieves today by emitting an
    arbitrary sample.
  - execution_heuristic: argmin execution_behavior_self_distance — the
    sample most behaviorally consistent with the other N-1 on shared probe
    inputs (an MBR-exec/consensus-style selector; verdict-free by
    construction, see src/uncertainty.py Week 6).
  - classifier__<arm>: argmax out-of-fold P(pass) from
    evaluate_predictor_cv under each SAMPLE_LEVEL_FEATURE_SETS arm.
    Leakage guard: out-of-fold probabilities only (StratifiedGroupKFold
    keeps a problem's rows in one fold, so the scoring model never saw the
    problem) — never an in-sample refit. Calibration note: monotone
    recalibration provably cannot change an argmax selection (Rabanser et
    al. 2025), so raw OOF probabilities are the correct input here.
  - oracle: ceiling, fraction of problems with >=1 passing sample (the
    empirical pass@N) — not a competing selector.

Pairwise selector differences are tested with McNemar mid-p
(src/evaluation.py, Fagerland et al. 2013) and BH-FDR adjusted across all
pairs (same scipy false_discovery_control pattern as
scripts/analyze_uncertainty.py).

Design decisions (Constraint #8 traceability):
  - --probe-source defaults to "evalplus" (unlike train_predictor.py's
    "test"): the Week 6 4-cell ablation showed EvalPlus probes + graded
    error costs fix the MBPP transfer failure, and the execution heuristic
    depends directly on signature resolution.
  - "Confidently wrong majority" trap diagnostic (SEP's stated limitation;
    DECK's Entrenched regime): reports the fraction of problems where
    execution_behavior_diversity < 0.2 (samples agree on >=80% of probe
    behavior on average) AND no sample passes — problems where consensus
    selection is guaranteed to pick a wrong sample. Per-problem diversity
    and pass_rate are persisted so other thresholds can be re-derived.
  - Selection accuracy is also reported conditional on solvable problems
    (>=1 passing sample), since overall accuracy is ceilinged by the
    oracle and the conditional view isolates selector skill.

Usage:
    python scripts/evaluate_selection.py                # latest run in results/
    python scripts/evaluate_selection.py --results results/humaneval_....json \
        --no-codebert --probe-source evalplus
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_uncertainty import find_latest_results_file
from scripts.train_predictor import (
    SAMPLE_LEVEL_FEATURE_SETS,
    add_execution_behavior_features,
    build_problem_level_dataset,
    build_sample_level_dataset,
)
from src.evaluation import mcnemar_mid_p
from src.uncertainty import (
    DEFAULT_ERROR_COSTS,
    LEGACY_ERROR_COSTS,
    evaluate_predictor_cv,
)
from src.utils import get_logger, save_results, set_all_seeds

logger = get_logger(__name__)

# Below this mean pairwise behavioral disagreement, a problem's samples are
# treated as "in consensus" for the trap diagnostic (see module docstring).
TRAP_DIVERSITY_THRESHOLD = 0.2


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the selection-evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate best-of-N candidate selectors on a stored run"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Pipeline results JSON to evaluate (default: newest run in results/).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="CV fold count for the out-of-fold classifier probabilities.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest"],
        default="logistic",
        help="Classifier for the trained-selector arms (logistic is the "
        "Week 4-6 sample-level winner).",
    )
    parser.add_argument(
        "--codebert-model",
        type=str,
        default="microsoft/codebert-base",
        help="HuggingFace encoder for the CodeBERT-based features.",
    )
    parser.add_argument(
        "--no-codebert",
        action="store_true",
        help="Skip CodeBERT-based features (arms that need them are skipped).",
    )
    parser.add_argument(
        "--execution-timeout",
        type=float,
        default=5.0,
        help="Per-sample wall-clock limit for behavior-signature replay (s).",
    )
    parser.add_argument(
        "--probe-source",
        choices=["test", "evalplus"],
        default="evalplus",
        help="Behavior-probe source (default evalplus; see module docstring).",
    )
    parser.add_argument(
        "--probe-cap",
        type=int,
        default=12,
        help="Max probes per problem for --probe-source evalplus.",
    )
    parser.add_argument(
        "--legacy-error-costs",
        action="store_true",
        help="Use the (1, 1, 0) abnormal-outcome costs instead of the graded "
        "He et al. default (1, 0.8, 0.6). Continuity arm.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the table but skip persisting the stats JSON to results/.",
    )
    return parser.parse_args()


def attach_oof_probabilities(
    sample_rows: list[dict],
    n_splits: int,
    seed: int,
    model_type: str,
) -> dict[str, dict]:
    """Attach each row's out-of-fold P(pass) per feature-set arm, in place.

    Runs evaluate_predictor_cv once per SAMPLE_LEVEL_FEATURE_SETS arm whose
    features are present in every row, and writes each row's held-out
    probability to row["_oof_proba__<arm>"]. oof_proba is aligned to the
    X row order, which is sample_rows order by construction.

    Args:
        sample_rows: Output of build_sample_level_dataset() (mutated).
        n_splits: CV fold count (forwarded to evaluate_predictor_cv).
        seed: Reproducibility seed.
        model_type: "logistic" or "random_forest".
    Returns:
        Dict mapping each evaluated arm name to its evaluate_predictor_cv
        prediction-metric summary (context for the selection numbers).
    """
    arm_summaries: dict[str, dict] = {}
    for arm_name, feature_names in SAMPLE_LEVEL_FEATURE_SETS.items():
        if not all(f in row for row in sample_rows for f in feature_names):
            logger.info(
                "Skipping classifier arm %r: a feature is unavailable "
                "(likely --no-codebert or missing dataset entry)",
                arm_name,
            )
            continue
        X = np.array(
            [[row[f] for f in feature_names] for row in sample_rows], dtype=float
        )
        y = np.array([row["passed"] for row in sample_rows], dtype=int)
        groups = np.array([row["task_id"] for row in sample_rows])
        logger.info("OOF probabilities | %s | %s", arm_name, model_type)
        cv_out = evaluate_predictor_cv(
            X,
            y,
            groups,
            model_type=model_type,
            n_splits=n_splits,
            seed=seed,
            feature_names=feature_names,
        )
        for row, proba in zip(sample_rows, cv_out["oof_proba"]):
            row[f"_oof_proba__{arm_name}"] = proba
        arm_summaries[arm_name] = {
            "summary": cv_out["summary"],
            "n_splits": cv_out["n_splits"],
            "class_balance": cv_out["class_balance"],
        }
    return arm_summaries


def select_per_problem(
    problem: dict,
    rows: list[dict],
    arm_names: list[str],
    rng: np.random.Generator,
) -> dict[str, int] | None:
    """Pick one sample index per selector for a single problem.

    No selector reads any sample's "passed" field; ties break to the first
    (lowest) index deterministically.

    Args:
        problem: One results["per_problem"] entry (needs >=2 samples).
        rows: The problem's sample_rows, in original sample order.
        arm_names: Classifier arms with attached "_oof_proba__<arm>" keys.
        rng: Seeded generator for the random-draw baseline.
    Returns:
        Dict mapping selector name to chosen sample index, or None when the
        problem is unscorable (fewer than 2 samples, or rows misaligned
        with the stored samples).
    """
    samples = problem.get("samples", [])
    n = len(samples)
    if n < 2 or len(rows) != n:
        return None
    picks: dict[str, int] = {"random_draw": int(rng.integers(n))}
    if all("execution_behavior_self_distance" in r for r in rows):
        distances = [r["execution_behavior_self_distance"] for r in rows]
        picks["execution_heuristic"] = int(np.argmin(distances))
    for arm in arm_names:
        key = f"_oof_proba__{arm}"
        probas = [r[key] for r in rows]
        if any(np.isnan(p) for p in probas):
            continue
        picks[f"classifier__{arm}"] = int(np.argmax(probas))
    return picks


def evaluate_selectors(
    results: dict,
    sample_rows: list[dict],
    problem_rows: list[dict],
    arm_names: list[str],
    seed: int,
) -> dict:
    """Score every selector's picks against stored ground truth.

    Args:
        results: Full pipeline results (CLAUDE.md Section 8 schema).
        sample_rows: build_sample_level_dataset() output with execution and
            "_oof_proba__<arm>" columns already attached.
        problem_rows: build_problem_level_dataset() output with
            "execution_behavior_diversity" attached (trap diagnostic).
        arm_names: Classifier arms evaluated by attach_oof_probabilities().
        seed: Seed for the random-draw baseline.
    Returns:
        Dict with per-selector accuracies, the trap diagnostic, pairwise
        McNemar mid-p tests, and the per-problem pick record.
    """
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    for row in sample_rows:
        rows_by_task[row["task_id"]].append(row)
    diversity_by_task = {
        row["task_id"]: row.get("execution_behavior_diversity") for row in problem_rows
    }

    rng = np.random.default_rng(seed)
    # selector -> {task_id: picked sample passed?}; oracle kept alongside.
    outcomes: dict[str, dict[str, bool]] = defaultdict(dict)
    per_problem_records: list[dict] = []
    pass_rates: dict[str, float] = {}
    n_skipped = 0

    for problem in results["per_problem"]:
        task_id = problem["task_id"]
        rows = rows_by_task.get(task_id, [])
        picks = select_per_problem(problem, rows, arm_names, rng)
        if picks is None:
            n_skipped += 1
            continue
        passed = [bool(s["passed"]) for s in problem["samples"]]
        pass_rates[task_id] = sum(passed) / len(passed)
        for selector, index in picks.items():
            outcomes[selector][task_id] = passed[index]
        outcomes["oracle"][task_id] = any(passed)
        per_problem_records.append(
            {
                "task_id": task_id,
                "pass_rate": pass_rates[task_id],
                "execution_behavior_diversity": diversity_by_task.get(task_id),
                "picks": {
                    selector: {"index": index, "passed": passed[index]}
                    for selector, index in picks.items()
                },
            }
        )

    solvable = {t for t, r in pass_rates.items() if r > 0}
    selection_accuracy = {}
    for selector, hits in outcomes.items():
        hit_values = list(hits.values())
        solvable_hits = [ok for t, ok in hits.items() if t in solvable]
        selection_accuracy[selector] = {
            "overall": float(np.mean(hit_values)),
            "n_problems": len(hit_values),
            "conditional_on_solvable": (
                float(np.mean(solvable_hits)) if solvable_hits else None
            ),
            "n_solvable": len(solvable_hits),
        }

    unsolvable = [t for t, r in pass_rates.items() if r == 0]
    trapped = [
        t
        for t in unsolvable
        if diversity_by_task.get(t) is not None
        and diversity_by_task[t] < TRAP_DIVERSITY_THRESHOLD
    ]

    # Pairwise McNemar mid-p over competing selectors (oracle is a ceiling,
    # not a competitor), on each pair's common problems, BH-FDR adjusted.
    selector_names = sorted(n for n in outcomes if n != "oracle")
    pairwise = []
    raw_p = []
    for name_a, name_b in itertools.combinations(selector_names, 2):
        common = sorted(set(outcomes[name_a]) & set(outcomes[name_b]))
        p = mcnemar_mid_p(
            [outcomes[name_a][t] for t in common],
            [outcomes[name_b][t] for t in common],
        )
        pairwise.append({"a": name_a, "b": name_b, "n_common": len(common)})
        raw_p.append(p)
    adjusted = scipy_stats.false_discovery_control(raw_p, method="bh") if raw_p else []
    for entry, p, p_adj in zip(pairwise, raw_p, adjusted):
        entry["p_mid"] = float(p)
        entry["p_bh"] = float(p_adj)

    return {
        "selection_accuracy": selection_accuracy,
        "expected_random": float(np.mean(list(pass_rates.values()))),
        "n_problems_scored": len(pass_rates),
        "n_problems_skipped": n_skipped,
        "trap_diagnostic": {
            "diversity_threshold": TRAP_DIVERSITY_THRESHOLD,
            "n_unsolvable": len(unsolvable),
            "n_unsolvable_low_diversity": len(trapped),
            "trapped_task_ids": trapped,
        },
        "pairwise_mcnemar": pairwise,
        "per_problem": per_problem_records,
    }


def print_selection_table(evaluation: dict) -> None:
    """Print the per-selector accuracy table to stdout."""
    expected_random = evaluation["expected_random"]
    oracle = evaluation["selection_accuracy"].get("oracle", {}).get("overall")
    print(
        f"\nexpected_random (pass@1 baseline) = {expected_random:.3f}"
        + (f"   oracle ceiling = {oracle:.3f}" if oracle is not None else "")
    )
    header = (
        f"{'selector':<45}{'overall':<10}{'vs random':<12}"
        f"{'on solvable':<13}{'n':<6}"
    )
    print(header)
    entries = sorted(
        evaluation["selection_accuracy"].items(),
        key=lambda kv: kv[1]["overall"],
        reverse=True,
    )
    for name, acc in entries:
        conditional = acc["conditional_on_solvable"]
        print(
            f"{name:<45}{acc['overall']:<10.3f}"
            f"{acc['overall'] - expected_random:<+12.3f}"
            f"{(f'{conditional:.3f}' if conditional is not None else 'n/a'):<13}"
            f"{acc['n_problems']:<6}"
        )
    significant = [p for p in evaluation["pairwise_mcnemar"] if p["p_bh"] < 0.05]
    print(f"\nMcNemar mid-p pairs significant at BH-FDR 0.05: {len(significant)}")
    for p in significant:
        print(f"  {p['a']} vs {p['b']}: p_mid={p['p_mid']:.4f} p_bh={p['p_bh']:.4f}")


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    results_path = args.results if args.results else find_latest_results_file()
    logger.info("Evaluating selection on %s", results_path)
    with results_path.open() as f:
        results = json.load(f)

    include_codebert = not args.no_codebert
    problem_rows = build_problem_level_dataset(
        results, include_codebert, args.codebert_model
    )
    sample_rows = build_sample_level_dataset(
        results, include_codebert, args.codebert_model
    )
    error_costs = LEGACY_ERROR_COSTS if args.legacy_error_costs else DEFAULT_ERROR_COSTS
    logger.info("Computing execution-behavior features (replaying stored samples)")
    add_execution_behavior_features(
        results,
        problem_rows,
        sample_rows,
        args.execution_timeout,
        probe_source=args.probe_source,
        probe_cap=args.probe_cap,
        error_costs=error_costs,
        seed=args.seed,
    )

    arm_summaries = attach_oof_probabilities(
        sample_rows, n_splits=args.n_splits, seed=args.seed, model_type=args.model_type
    )
    evaluation = evaluate_selectors(
        results, sample_rows, problem_rows, list(arm_summaries), args.seed
    )
    print_selection_table(evaluation)

    config = results.get("config", {})
    stats = {
        "analysis": "week6_selection_trial",
        "source_results_file": str(results_path),
        "source_run_id": results.get("run_id"),
        "config": {
            "seed": args.seed,
            "n_splits": args.n_splits,
            "model_type": args.model_type,
            "codebert_model": args.codebert_model if include_codebert else None,
            "execution_timeout": args.execution_timeout,
            "probe_source": args.probe_source,
            "probe_cap": args.probe_cap,
            "error_costs": list(error_costs),
        },
        "source_config": config,
        "prediction_metrics_context": arm_summaries,
        **evaluation,
    }
    if args.dry_run:
        logger.info("[DRY RUN] skipping stats JSON persistence")
    else:
        out_json = save_results(
            stats, prefix=f"selection_{config.get('dataset', 'unknown')}"
        )
        logger.info("Saved selection evaluation stats to %s", out_json)


if __name__ == "__main__":
    main()
