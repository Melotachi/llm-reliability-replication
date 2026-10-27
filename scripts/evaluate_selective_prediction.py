"""Week 7: turn stored predictor rankings into deployment numbers.

Student A deliverable. Re-reads the out-of-fold probabilities that
scripts/train_predictor.py already persists (`oof_proba`/`oof_y`, added in
Week 5 for calibration) and reports, per feature arm, what the ranking is
worth as an accept/reject rule: AUGRC (primary), AURC (secondary), and the
coverage attainable at a 5% and a 10% error budget.

Why this matters more than another ROC-AUC column: on the merged-164 cell the
best sample-level arm scores ROC-AUC 0.934, which sounds close to solved — and
translates to auto-accepting roughly 4% of generated samples at a 5% error
budget. Same model, honest description. See src/selective.py for the metric
definitions and the NeurIPS 2024 citation behind choosing AUGRC over AURC.

No generation, no execution, no re-training: it consumes stored JSON only.

Usage:
    python scripts/evaluate_selective_prediction.py \\
        --predictor cg_he164=results/predictor_humaneval_20260712_215312.json \\
        --predictor ds_he60=results/predictor_humaneval_20260712_212837.json

    # every predictor JSON produced by a Week 7 batch
    python scripts/evaluate_selective_prediction.py \\
        --predictor cg_he164=results/predictor_humaneval_20260804_2300.json \\
        --risk-targets 0.05 0.10 0.20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_representations import parse_run_specs
from src.selective import DEFAULT_RISK_TARGETS, selective_prediction_metrics
from src.utils import get_logger, save_results, set_all_seeds

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the selective-prediction entry point."""
    parser = argparse.ArgumentParser(
        description="Risk-coverage / AUGRC evaluation of stored predictor rankings"
    )
    parser.add_argument(
        "--predictor",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A cell: short label and its train_predictor.py output JSON. "
        "Repeat once per cell.",
    )
    parser.add_argument(
        "--risk-targets",
        type=float,
        nargs="+",
        default=list(DEFAULT_RISK_TARGETS),
        help="Error budgets for the coverage columns (default: 0.05 0.10).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the table but skip persisting the stats JSON.",
    )
    return parser.parse_args()


def evaluate_predictor_file(path: Path, risk_targets: tuple[float, ...]) -> list[dict]:
    """Selective-prediction metrics for every configuration in one JSON.

    Args:
        path: A train_predictor.py output file.
        risk_targets: Error budgets forwarded to the metric computation.
    Returns:
        One dict per configuration that carries out-of-fold probabilities,
        each with task/feature_set/model_type plus the metrics. Arms
        recorded before Week 5 (no oof_proba) are skipped with a warning.
    """
    with path.open() as f:
        data = json.load(f)
    rows = []
    for configuration in data.get("configurations", []):
        proba = configuration.get("oof_proba")
        labels = configuration.get("oof_y")
        if not proba or labels is None:
            logger.warning(
                "%s: %s/%s/%s has no out-of-fold probabilities; skipping",
                path.name,
                configuration.get("task"),
                configuration.get("feature_set"),
                configuration.get("model_type"),
            )
            continue
        metrics = selective_prediction_metrics(labels, proba, risk_targets=risk_targets)
        rows.append(
            {
                "task": configuration.get("task"),
                "feature_set": configuration.get("feature_set"),
                "model_type": configuration.get("model_type"),
                "roc_auc": configuration.get("summary", {}).get("roc_auc_mean"),
                **metrics,
            }
        )
    return rows


def _fmt(value: float | None) -> str:
    """Three-decimal cell, or 'n/a' for an undefined metric."""
    return "n/a" if value is None else f"{value:.3f}"


def print_table(results: dict[str, list[dict]], risk_targets: list[float]) -> None:
    """Print one block per cell, sorted by norm-AUGRC (lower is better)."""
    coverage_headers = "".join(f"{'cov@' + f'{t:.0%}':<10}" for t in risk_targets)
    for label, rows in results.items():
        print(f"\n=== {label} ===")
        print(
            f"{'task':<15}{'feature_set':<34}{'model':<15}"
            f"{'ROC-AUC':<10}{'nAUGRC':<10}{'nAURC':<10}{coverage_headers}"
            f"{'base err':<10}"
        )
        # norm-AUGRC is the primary ranking (NeurIPS 2024); None sorts last.
        for row in sorted(
            rows, key=lambda r: (r["norm_augrc"] is None, r["norm_augrc"] or 0.0)
        ):
            roc_cell = _fmt(row["roc_auc"])
            augrc_cell = _fmt(row["norm_augrc"])
            aurc_cell = _fmt(row["norm_aurc"])
            coverage_cells = "".join(
                "{:<10.3f}".format(row["coverage_at_risk"]["{:.2f}".format(target)])
                for target in risk_targets
            )
            print(
                f"{row['task']:<15}{row['feature_set']:<34}{row['model_type']:<15}"
                f"{roc_cell:<10}{augrc_cell:<10}{aurc_cell:<10}"
                f"{coverage_cells}{row['base_error']:<10.3f}"
            )


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    cells = parse_run_specs(args.predictor)
    risk_targets = tuple(args.risk_targets)

    results = {}
    sources = {}
    for label, path in cells:
        logger.info("Cell %s: %s", label, path)
        results[label] = evaluate_predictor_file(path, risk_targets)
        sources[label] = str(path)

    print_table(results, list(risk_targets))
    print(
        "\nnAUGRC/nAURC: 0.0 = oracle ranking, 1.0 = random, >1.0 = worse than "
        "random.\nBoth are rank-based, so a monotone recalibration (e.g. the "
        "Week 6 beta calibrator)\nleaves every number in this table unchanged."
    )

    stats = {
        "analysis": "week7_selective_prediction",
        "config": {"seed": args.seed, "risk_targets": list(risk_targets)},
        "sources": sources,
        "cells": results,
    }
    if args.dry_run:
        logger.info("[DRY RUN] skipping stats JSON persistence")
    else:
        out_path = save_results(stats, prefix="analysis_week7_selective")
        logger.info("Saved selective-prediction stats to %s", out_path)


if __name__ == "__main__":
    main()
