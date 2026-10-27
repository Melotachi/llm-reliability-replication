"""Week 7: is the execution-behavior signal really about *agreement*, or about *running*?

Student A deliverable, and a direct check on the Week 6 headline. That week
reported the execution-behavior family as the project's strongest signal
(Task B ROC-AUC 0.89-0.93). A duller explanation was never ruled out: at 350M,
parse-failure rates reach 0.7 and a large share of probe outcomes are
`error:*`, so "behavioral consensus predicts correctness" may be mostly
"runnability predicts correctness".

The crash-rate ablation arms in train_predictor.py answer half of that, and
the first run of them produced a trap worth being explicit about. Restricting
the pairwise distance to probes where *both* samples ran leaves a sample that
never runs with **no co-running probe at all**, and this family's convention
scores an empty comparison as 0.0 — perfect agreement. So the crashiest
samples come out looking like the most confident ones, which can drive that
arm's ROC-AUC to chance for a reason that has nothing to do with the research
question.

This script separates the three quantities on the same replay, so the claim
in the report is about the mechanism rather than about a convention:

  crash_rate           - fraction of probes the sample errors on (the dull half)
  execution_self       - the Week 6 feature, all probes, graded distances
  nocrash_self         - agreement restricted to co-running probes
  nocrash_support      - fraction of this sample's pairs that had any
                         co-running probe; 0.0 means nocrash_self is
                         evidence-free, not evidence of agreement

and reports each one's ROC-AUC twice: over all samples, and over the
**subpopulation with support > 0**, which is the honest form of the question
*"among code that runs, does behavioral agreement predict correctness?"*

# TODO: SANDBOX — replays stored generated code in subprocesses (same caveat
# as evaluation.run_code). Research/local use only.

Usage:
    python scripts/diagnose_crash_control.py --results results/<run>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import behavior_signature, extract_probe_inputs
from src.uncertainty import (
    DEFAULT_ERROR_COSTS,
    execution_behavior_self_consistency,
    execution_behavior_self_consistency_nocrash,
    execution_crash_rate,
)
from src.utils import (
    get_logger,
    load_dataset_split,
    load_evalplus_probe_inputs,
    save_results,
    set_all_seeds,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the crash-control diagnostic."""
    parser = argparse.ArgumentParser(
        description="Separate runnability from behavioral agreement in the "
        "execution-behavior family"
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="Pipeline results JSON to analyze."
    )
    parser.add_argument(
        "--probe-source", choices=["test", "evalplus"], default="evalplus"
    )
    parser.add_argument("--probe-cap", type=int, default=12)
    parser.add_argument("--execution-timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def nocrash_support(signatures: list[list[str]]) -> list[float]:
    """Fraction of a sample's pairs that share at least one co-running probe.

    The companion to execution_behavior_self_consistency_nocrash: without it,
    a score of 0.0 is ambiguous between "agreed everywhere it ran" and "never
    ran anywhere, so nothing was compared".

    Args:
        signatures: One per-probe outcome list per sample.
    Returns:
        One support value per sample in [0.0, 1.0]; all 0.0 when N < 2.
    """
    n = len(signatures)
    if n < 2:
        return [0.0] * n
    counts = [0] * n
    for i, j in combinations(range(n), 2):
        co_running = any(
            not x.startswith("error:") and not y.startswith("error:")
            for x, y in zip(signatures[i], signatures[j])
        )
        if co_running:
            counts[i] += 1
            counts[j] += 1
    return [count / (n - 1) for count in counts]


def collect_sample_features(
    results: dict, probe_source: str, probe_cap: int, timeout_s: float, seed: int
) -> list[dict]:
    """Replay every stored sample and record the four quantities plus the label."""
    dataset_name = results.get("config", {}).get("dataset", "humaneval")
    problems = {p["task_id"]: p for p in load_dataset_split(dataset_name)}
    evalplus_probes = (
        load_evalplus_probe_inputs(dataset_name, cap=probe_cap, seed=seed)
        if probe_source == "evalplus"
        else {}
    )
    rows = []
    for problem in results["per_problem"]:
        task_id = problem["task_id"]
        meta = problems.get(task_id)
        samples = problem.get("samples", [])
        if meta is None or not samples:
            continue
        probes = evalplus_probes.get(task_id) or extract_probe_inputs(
            meta["test_code"], meta["entry_point"]
        )
        signatures = [
            behavior_signature(s["code"], meta["entry_point"], probes, timeout_s)
            for s in samples
        ]
        full = execution_behavior_self_consistency(signatures, DEFAULT_ERROR_COSTS)
        nocrash = execution_behavior_self_consistency_nocrash(signatures)
        support = nocrash_support(signatures)
        for sample, f, nc, sup, sig in zip(samples, full, nocrash, support, signatures):
            rows.append(
                {
                    "task_id": task_id,
                    "passed": int(bool(sample["passed"])),
                    "crash_rate": execution_crash_rate(sig),
                    "execution_self": f,
                    "nocrash_self": nc,
                    "nocrash_support": sup,
                }
            )
    return rows


def _auc(rows: list[dict], feature: str) -> float | None:
    """ROC-AUC of "sample passes" against a feature where higher = worse.

    The diversity features are inverted so every number in the printed table
    reads the same way: 1.0 = perfect ranking of passing samples first.
    """
    labels = [row["passed"] for row in rows]
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, [-row[feature] for row in rows]))


def summarize(rows: list[dict]) -> dict:
    """ROC-AUC of each quantity over all samples and over the support>0 subset."""
    supported = [row for row in rows if row["nocrash_support"] > 0.0]
    features = ["crash_rate", "execution_self", "nocrash_self"]
    return {
        "n_samples": len(rows),
        "n_supported": len(supported),
        "frac_supported": len(supported) / len(rows) if rows else 0.0,
        "pass_rate_all": float(np.mean([r["passed"] for r in rows])) if rows else 0.0,
        "pass_rate_supported": (
            float(np.mean([r["passed"] for r in supported])) if supported else 0.0
        ),
        "mean_crash_rate": float(np.mean([r["crash_rate"] for r in rows])),
        "frac_samples_all_crash": float(
            np.mean([1.0 if r["crash_rate"] == 1.0 else 0.0 for r in rows])
        ),
        "auc_all_samples": {name: _auc(rows, name) for name in features},
        "auc_supported_only": {name: _auc(supported, name) for name in features},
    }


def print_report(summary: dict) -> None:
    """Print the diagnostic table."""
    print(
        f"\nsamples: {summary['n_samples']}  "
        f"with co-running evidence: {summary['n_supported']} "
        f"({summary['frac_supported']:.1%})"
    )
    print(
        f"mean crash rate: {summary['mean_crash_rate']:.3f}   "
        f"samples that crash on EVERY probe: {summary['frac_samples_all_crash']:.1%}"
    )
    print(
        f"pass rate — all: {summary['pass_rate_all']:.3f}   "
        f"support>0 subset: {summary['pass_rate_supported']:.3f}"
    )
    print(f"\n{'feature':<20}{'ROC (all samples)':<22}{'ROC (support>0 only)':<22}")
    for name in ("crash_rate", "execution_self", "nocrash_self"):
        all_auc = summary["auc_all_samples"][name]
        sub_auc = summary["auc_supported_only"][name]
        print(
            f"{name:<20}"
            f"{('n/a' if all_auc is None else f'{all_auc:.3f}'):<22}"
            f"{('n/a' if sub_auc is None else f'{sub_auc:.3f}'):<22}"
        )


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    with args.results.open() as f:
        results = json.load(f)

    logger.info("Replaying stored samples (this is the slow step)")
    rows = collect_sample_features(
        results, args.probe_source, args.probe_cap, args.execution_timeout, args.seed
    )
    summary = summarize(rows)
    print_report(summary)

    stats = {
        "analysis": "week7_crash_control_diagnostic",
        "source_results_file": str(args.results),
        "source_run_id": results.get("run_id"),
        "config": {
            "seed": args.seed,
            "probe_source": args.probe_source,
            "probe_cap": args.probe_cap,
            "execution_timeout": args.execution_timeout,
        },
        "source_config": results.get("config", {}),
        "summary": summary,
        "per_sample": rows,
    }
    if args.dry_run:
        logger.info("[DRY RUN] skipping stats JSON persistence")
    else:
        out_path = save_results(stats, prefix="analysis_week7_crashcontrol")
        logger.info("Saved crash-control diagnostic to %s", out_path)


if __name__ == "__main__":
    main()
