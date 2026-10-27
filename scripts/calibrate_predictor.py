"""Week 5 calibration analysis: is UncertaintyPredictor.predict_proba() trustworthy?

Student A deliverable (CLAUDE.md Section 10.1 Week 5: "Calibration: Expected
Calibration Error (ECE), reliability diagram").

Weeks 2-4 all asked "does an uncertainty score correlate with pass/fail" --
a ranking question. This script asks a different question about the Week 4
predictor specifically: when it outputs "70% likely to pass," does that
sample (or problem) actually pass about 70% of the time? A model can rank
well (high ROC-AUC) while being badly calibrated (e.g. uniformly
overconfident), and only the second property is safe to use as a literal
probability -- which matters directly for CLAUDE.md Section 10.3's Week 6-7
"Smart Repair" threshold, a decision that reads predict_proba() as a
probability, not just a score to rank by.

Calibrates the Week 4 report's own stated defaults (Section 7 TODO item 1):
Task A `reduced`/logistic (problem level) and Task B
`self_consistency_core`/logistic (sample level) -- not a re-run of all 12
Week 4 configurations, since the point is depth on the two configs already
chosen as this project's best/simplest, not another breadth sweep.

Uses evaluate_predictor_cv's out-of-fold probabilities ("oof_proba"), never
an in-sample fit/score: a model scored on the rows it was trained on looks
artificially well-calibrated, which would silently defeat the point of this
analysis. See src/uncertainty.py's compute_calibration() docstring for the
bin-count/strategy citations (Sun et al. 2023; Nixon et al. 2019).

Produces:
  1. reports/figures/week5_reliability_diagram.png -- two-panel reliability
     diagram (Task A, Task B), each against the y=x perfect-calibration
     reference, marker size proportional to bin count.
  2. A timestamped stats JSON in results/ with both tasks' full
     compute_calibration() output plus the underlying CV summary
     (CLAUDE.md Constraint #5: results are always persisted).

Usage:
    python scripts/calibrate_predictor.py                  # latest run in results/
    python scripts/calibrate_predictor.py --results results/humaneval_uncertainty_....json
    python scripts/calibrate_predictor.py --no-codebert    # skip the ~500 MB encoder
    python scripts/calibrate_predictor.py --n-bins 5        # override the n^(1/3) default
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes PNGs

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import StratifiedGroupKFold

from scripts.analyze_uncertainty import find_latest_results_file
from scripts.train_predictor import (
    build_problem_level_dataset,
    build_sample_level_dataset,
)
from src.uncertainty import (
    UncertaintyPredictor,
    compute_calibration,
    evaluate_predictor_cv,
    fit_beta_calibrator,
)
from src.utils import get_logger, save_results, set_all_seeds

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "reports" / "figures"

# Same validated reference palette as analyze_uncertainty.py/train_predictor.py
# (scripts/validate_palette.js, light mode, surface #fcfcfb) -- unchanged
# instance, not re-validated separately. Task identity keeps the same
# categorical assignment train_predictor.py's comparison figure uses (Task A
# = blue slot 1, Task B = aqua slot 2), so a reader who has already seen that
# figure doesn't need to re-learn the color mapping here.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_TASK_A = "#2a78d6"
_TASK_B = "#1baf7a"
_REFERENCE = "#c3c2b7"  # y=x perfect-calibration line: structural, not data

# Week 4 report Section 7 TODO item 1's own stated defaults -- the configs
# this script calibrates, not a re-sweep of all 12 Week 4 configurations.
TASK_A_FEATURES = [
    "token_jaccard_diversity",
    "ast_node_diversity",
    "codebert_cosine_diversity",
    "pairwise_distance",
]
TASK_B_FEATURES = [
    "token_jaccard_self_distance",
    "token_cosine_self_distance",
    "ast_node_self_distance",
]
# Under --no-codebert, Task A falls back to the encoder-free 4-feature set
# (train_predictor.py's "no_codebert" arm) instead of being skipped.
TASK_A_FEATURES_NO_CODEBERT = [
    "token_jaccard_diversity",
    "token_cosine_diversity",
    "ast_node_diversity",
    "pairwise_distance",
]


def nested_oof_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    recalibrate: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Out-of-fold probabilities, optionally beta-recalibrated without leakage.

    Outer loop: leave-problems-out StratifiedGroupKFold, exactly like
    evaluate_predictor_cv. When recalibrate is set, each outer fold fits a
    beta calibrator (Kull et al., 2017) on the TRAINING side's own inner
    grouped-CV out-of-fold predictions, then applies it once to the outer
    test fold — the calibrator never sees any prediction it is evaluated
    on (the Week 5 out-of-fold rule, applied one level deeper).

    Args:
        X, y, groups: As in evaluate_predictor_cv.
        n_splits: Fold count for both outer and inner splits.
        seed: Reproducibility seed (CLAUDE.md Constraint #4).
        recalibrate: If True, also return beta-recalibrated probabilities.
    Returns:
        (oof_raw, oof_recalibrated) — the second is None when recalibrate
        is False. Both aligned to input row order.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_splits = max(2, min(n_splits, len(np.unique(groups))))
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan) if recalibrate else None
    for train_idx, test_idx in outer.split(X, y, groups):
        predictor = UncertaintyPredictor(model_type="logistic", seed=seed)
        predictor.fit(X[train_idx], y[train_idx])
        p_test = predictor.predict_proba(X[test_idx])
        oof_raw[test_idx] = p_test
        if recalibrate:
            inner = evaluate_predictor_cv(
                X[train_idx],
                y[train_idx],
                groups[train_idx],
                model_type="logistic",
                n_splits=n_splits,
                seed=seed,
            )
            calibrator = fit_beta_calibrator(inner["oof_y"], inner["oof_proba"])
            oof_cal[test_idx] = calibrator(p_test)
    return oof_raw, oof_cal


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the calibration entry point."""
    parser = argparse.ArgumentParser(
        description="Calibrate the Week 4 predictor's default configurations"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Pipeline results JSON to calibrate on (default: newest run in results/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for the output PNG figure (default: reports/figures/).",
    )
    parser.add_argument(
        "--codebert-model",
        type=str,
        default="microsoft/codebert-base",
        help="HuggingFace encoder for the CodeBERT-based Task A feature.",
    )
    parser.add_argument(
        "--no-codebert",
        action="store_true",
        help="Skip codebert_cosine_diversity (avoids the ~500 MB encoder "
        "download; Task A's reduced set needs it, so Task A is skipped).",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="CV fold count (capped at n_groups)."
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Reliability-diagram bin count (default: round(n_rows ** (1/3)), "
        "per Sun et al. 2023 -- see compute_calibration()'s docstring).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--recalibrate",
        choices=["none", "beta"],
        default="none",
        help="Post-hoc recalibration: 'beta' fits a leakage-free nested "
        "beta calibrator (Kull et al. 2017) per outer fold and reports "
        "raw vs recalibrated panels side by side.",
    )
    parser.add_argument(
        "--fig-name",
        type=str,
        default="week5_reliability_diagram.png",
        help="Output figure filename inside --out-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the figure but skip persisting the stats JSON to results/.",
    )
    return parser.parse_args()


def _style_axes(ax: plt.Axes) -> None:
    """Shared recessive-chrome style, matching the project's other figures."""
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_MUTED, labelsize=8.5, length=0)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_reliability_diagrams(
    calibrations: dict[str, dict], out_path: Path, run_label: str
) -> None:
    """Two-panel reliability diagram: mean predicted probability vs. observed
    fraction positive, one panel per task, marker size proportional to bin
    count (so a bin two rows wide doesn't visually compete with one holding
    a third of the data).

    Args:
        calibrations: {"Task A (reduced/logistic)": compute_calibration()
            output, "Task B (self_consistency_core/logistic)": ...}.
        out_path: PNG destination.
        run_label: Dataset/config line for the figure subtitle.
    """
    n_panels = len(calibrations)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.4 * n_panels, 5.0), constrained_layout=True
    )
    fig.set_facecolor(_SURFACE)
    if n_panels == 1:
        axes = [axes]

    # Color by task identity (not by panel index): recalibrated variants of
    # a task keep that task's categorical color, so a 4-panel raw-vs-beta
    # layout stays readable.
    colors = [
        _TASK_A if label.startswith("Task A") else _TASK_B for label in calibrations
    ]
    for ax, color, (label, cal) in zip(axes, colors, calibrations.items()):
        _style_axes(ax)
        ax.plot(
            [0, 1], [0, 1], color=_REFERENCE, linewidth=1.2, linestyle="--", zorder=1
        )

        counts = np.array(cal["bin_count"], dtype=float)
        sizes = 60.0 + 340.0 * (counts / counts.max()) if len(counts) else []
        ax.plot(
            cal["bin_mean_predicted"],
            cal["bin_fraction_positive"],
            color=color,
            linewidth=1.4,
            zorder=2,
        )
        ax.scatter(
            cal["bin_mean_predicted"],
            cal["bin_fraction_positive"],
            s=sizes,
            color=color,
            edgecolor=_SURFACE,
            linewidth=1.0,
            zorder=3,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability", fontsize=9, color=_INK_SECONDARY)
        ax.set_ylabel("Observed fraction positive", fontsize=9, color=_INK_SECONDARY)
        ax.set_title(
            f"{label}\nECE={cal['ece']:.3f}  MCE={cal['mce']:.3f}  "
            f"Brier={cal['brier_score']:.3f}  ({cal['n_bins_actual']} bins, "
            f"n={cal['n_rows']})",
            loc="left",
            fontsize=9.3,
            color=_INK,
            pad=8,
        )

    fig.suptitle(
        "Reliability diagrams: predicted probability vs. observed outcome",
        fontsize=12.5,
        color=_INK,
        ha="left",
        x=0.012,
    )
    fig.text(0.012, -0.02, run_label, fontsize=8, color=_MUTED, ha="left")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    results_path = args.results if args.results else find_latest_results_file()
    logger.info("Calibrating on %s", results_path)
    with results_path.open() as f:
        results = json.load(f)

    include_codebert = not args.no_codebert
    problem_rows = build_problem_level_dataset(
        results, include_codebert, args.codebert_model
    )
    sample_rows = build_sample_level_dataset(
        results, include_codebert, args.codebert_model
    )

    calibrations: dict[str, dict] = {}

    if include_codebert:
        task_a_features, task_a_name = TASK_A_FEATURES, "reduced"
    else:
        task_a_features, task_a_name = TASK_A_FEATURES_NO_CODEBERT, "no_codebert"

    task_specs = []
    if problem_rows:
        task_specs.append(
            (
                f"Task A ({task_a_name} / logistic)",
                np.array(
                    [[row[f] for f in task_a_features] for row in problem_rows],
                    dtype=float,
                ),
                np.array([row["solved"] for row in problem_rows], dtype=int),
                np.arange(len(problem_rows)),
            )
        )
    if sample_rows:
        task_specs.append(
            (
                "Task B (self_consistency_core / logistic)",
                np.array(
                    [[row[f] for f in TASK_B_FEATURES] for row in sample_rows],
                    dtype=float,
                ),
                np.array([row["passed"] for row in sample_rows], dtype=int),
                np.array([row["task_id"] for row in sample_rows]),
            )
        )

    recalibrate = args.recalibrate == "beta"
    for label, X, y, groups in task_specs:
        oof_raw, oof_cal = nested_oof_probabilities(
            X, y, groups, args.n_splits, args.seed, recalibrate
        )
        calibrations[label] = compute_calibration(y, oof_raw, n_bins=args.n_bins)
        if recalibrate:
            calibrations[f"{label} + beta"] = compute_calibration(
                y, oof_cal, n_bins=args.n_bins
            )

    if not calibrations:
        raise ValueError(
            f"No calibratable configuration available from {results_path} "
            "(check --no-codebert and that per_problem samples are present)."
        )

    for label, cal in calibrations.items():
        logger.info(
            "%s: ECE=%.3f MCE=%.3f Brier=%.3f (%d bins, n=%d)",
            label,
            cal["ece"],
            cal["mce"],
            cal["brier_score"],
            cal["n_bins_actual"],
            cal["n_rows"],
        )

    config = results.get("config", {})
    run_label = (
        f"{config.get('dataset', '?')} · {config.get('model', '?')} · "
        f"seed {args.seed} · source: {results.get('run_id', results_path.name)}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.fig_name
    plot_reliability_diagrams(calibrations, out_path, run_label)

    stats = {
        "analysis": "predictor_calibration",
        "source_results_file": str(results_path),
        "source_run_id": results.get("run_id"),
        "config": {
            "seed": args.seed,
            "n_splits": args.n_splits,
            "n_bins": args.n_bins,
            "recalibrate": args.recalibrate,
            "codebert_model": args.codebert_model if include_codebert else None,
        },
        "source_config": config,
        "n_problems": len(problem_rows),
        "n_samples": len(sample_rows),
        "figure": str(out_path),
        "calibrations": calibrations,
    }
    if args.dry_run:
        logger.info("[DRY RUN] skipping stats JSON persistence")
    else:
        out_json = save_results(
            stats, prefix=f"calibration_{config.get('dataset', 'unknown')}"
        )
        logger.info("Saved calibration stats to %s", out_json)


if __name__ == "__main__":
    main()
