"""Week 7: do the project's findings hold across model capability?

Student A deliverable, scored against
`reports/week7_capability_sweep_preregistration.md` (committed before the new
models were run).

THE GAP
-------
Weeks 2-7 rest on two models, a 350M and a 1.3B, and three of the project's most
quotable claims are explicitly about *where on the capability scale* something
happens — the Semantic Consensus Hypothesis's confidently-wrong residual, the
crash-support method trap being "invisible at >= 7B", and the execution
channel's advantage arising "where the model is competent". Each is stated from
two data points, which cannot separate a trend from a coincidence.

RETRACTED 2026-09-01: the second of those three claims is withdrawn. He et al.
(arXiv 2605.09023, Appendix A, Table 8) report a CrashPollutionRate of
0.201-0.269 on GPT-3.5-Turbo, GPT-4o-mini, Gemini-2.5-Flash-Lite, Claude Opus
4.5 and DeepSeek-Coder-V2, against 0.134-0.373 measured here on the *same*
cell-level denominator (diagnose_crash_control.py's `mean_crash_rate`). Crash
pollution is scale-invariant; what is absent at every scale is the *control*
— no paper in that family decomposes the execution signal into runnability and
agreement-among-runnable, which is what report Section 7.3 does. See
`reports/week10_novelty_verification.md` Section 1.2. This script's outputs are
unaffected: the retraction is about how P4 was motivated, not about how any
quantity is computed.

This script takes several runs of the *same cell* under different models and
lines up, against measured pass@1:

  * every diversity metric's correlation with pass rate,
  * the consensus quartile spread and the confidently-wrong rate (report §6),
  * the mean parse-failure rate (the cheap proxy for the §7 crash regime).

WHY pass@1 AND NOT PARAMETER COUNT
----------------------------------
A 2024 0.5B code model can beat a 2022 350M one outright, so parameter count
does not order these models by capability and the x-axis would be measuring the
wrong thing. Using measured pass@1 also makes each prediction checkable the
moment a run finishes, without knowing anything about the model in advance.
Prediction P5 in the pre-registration exists to test this choice rather than
assume it.

Nothing here re-generates. Cells must be otherwise identical (same problems,
N, temperature, seed) or the comparison is confounded; this is checked and
reported, not assumed.

Usage:
    python scripts/analyze_capability_sweep.py --results results/a.json results/b.json
    python scripts/analyze_capability_sweep.py --results ... --metric ast_node_diversity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_consensus_deciles import confidently_wrong_rate, quantile_table
from scripts.analyze_uncertainty import (
    _INK,
    _INK_SECONDARY,
    _MUTED,
    _SERIES,
    _SURFACE,
    _reserve_footnote_band,
    _style_axes,
    load_run_prompts,
)
from scripts.train_predictor import build_problem_level_dataset
from src.uncertainty import METRIC_REPRESENTATIONS
from src.utils import get_logger, save_results, set_all_seeds

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "reports" / "figures"

#: Metrics tracked across the ladder. The embedding families are excluded for
#: the same reason the reliability analysis excludes them: they are the two
#: expensive ones, and no pre-registered prediction concerns them.
TRACKED_METRICS = (
    "lexical_diversity",
    "token_jaccard_diversity",
    "token_cosine_diversity",
    "ast_node_diversity",
    "pairwise_distance",
)

#: The consensus feature every §6 claim is stated over.
CONSENSUS_FEATURE = "ast_node_diversity"

#: Config keys that MUST match across cells for the comparison to be about the
#: model and nothing else.
#:
#: `device_resolved` and `dtype_resolved` were added on 2026-08-31, when the
#: >=7B rung joined. They belong here for exactly the reason the ladder was
#: regenerated device-matched in the first place: `generation._resolve_device`
#: documents that device and dtype both change the sampled tokens at a fixed
#: seed, so a rung that differs in either is confounded with capability. Their
#: absence was a hole, not a decision — the five device-matched rungs all
#: record cuda/float32, so adding the keys leaves their report unchanged while
#: making any future dtype deviation announce itself instead of passing
#: silently.
PROTOCOL_KEYS = (
    "dataset",
    # The hash of the task ids actually evaluated, written by run_pipeline.py
    # from Week 11 on. dataset + n_problems + problem_offset identify a problem
    # set only when the run took a contiguous slice; a --task-ids-file run does
    # not, and the BigCodeBench cell is selected by a gate whose result depends
    # on the executing image. Runs predating the key are not left blank:
    # with_derived_task_ids_hash() recovers the same hash from `per_problem`,
    # so the key is answered for every cell rather than half-populated.
    "task_ids_sha256",
    "n_problems",
    "problem_offset",
    "n_samples",
    "temperature",
    "seed",
    "device_resolved",
    "dtype_resolved",
)

#: The pre-registered two-point fits, as (intercept, slope) in measured pass@1,
#: plus the tolerance each was committed with. Kept here so the scoring cannot
#: quietly drift from the document.
PREREGISTERED = {
    "confidently_wrong_rate": {
        "intercept": 0.167 + 0.630 * 0.230,
        "slope": -0.630,
        "tolerance": 0.10,
        "floor": 0.0,
        "label": "P1 confidently-wrong rate",
    },
    "ast_abs_spearman": {
        "intercept": 0.404,
        "slope": 0.679,
        "tolerance": 0.12,
        "floor": None,
        "label": "P3 |Spearman rho| of ast_node_diversity",
    },
}

#: P2's committed band for the quartile spread, and P4's committed ceiling.
P2_SPREAD_BAND = (0.40, 0.70)
P4_PARSE_FAILURE_CEILING = 0.005


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the capability-sweep entry point."""
    parser = argparse.ArgumentParser(
        description="Metric behavior across model capability, scored against "
        "the pre-registration"
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more pipeline results JSONs of the SAME cell, differing "
        "only in --model.",
    )
    parser.add_argument(
        "--representation",
        choices=METRIC_REPRESENTATIONS,
        default="full",
        help="Metric input representation. Defaults to 'full', matching the "
        "stored cells the pre-registered fits were derived from.",
    )
    parser.add_argument(
        "--metric",
        default=CONSENSUS_FEATURE,
        help=f"Metric to headline in the figure (default: {CONSENSUS_FEATURE}).",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURES_DIR / "week7_capability_sweep.png",
        help="PNG destination.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print but do not persist outputs."
    )
    return parser.parse_args()


def with_derived_task_ids_hash(results: dict) -> dict:
    """Return the run's config with `task_ids_sha256` filled in if it is absent.

    run_pipeline.py records the hash from Week 11 on. Every earlier run stores
    the problem set anyway -- one `per_problem` entry per task id, in
    evaluation order -- so the same hash is recoverable, and recovering it is
    better than leaving the key half-populated: a ladder mixing a pre-Week-11
    rung with a later one would otherwise report the key as unrecorded and say
    nothing about whether the two ran the same problems, which is the single
    thing the key exists to answer.

    Args:
        results: Pipeline results dict (CLAUDE.md Section 8 schema).
    Returns:
        The config dict, copied, with `task_ids_sha256` present.
    """
    config = dict(results.get("config", {}))
    if config.get("task_ids_sha256"):
        return config
    task_ids = [problem.get("task_id", "") for problem in results["per_problem"]]
    config["task_ids_sha256"] = hashlib.sha256("\n".join(task_ids).encode()).hexdigest()
    return config


def check_protocol_match(cells: list[dict]) -> tuple[dict, dict]:
    """Separate real protocol conflicts from keys an older run never recorded.

    A capability sweep is only about capability if everything else is held
    fixed, so this is checked rather than assumed. But the two are not the same
    finding and must not be reported as one:

      * **conflict** — two cells record *different* values. The comparison is
        genuinely confounded.
      * **unrecorded** — *some* cells record the key and others do not, because
        the others predate it being persisted (`--temperature` and
        `--problem-offset` were only added to the results config on
        2026-07-12). The value is *unknown*, not known-different, and
        collapsing that into "conflict" would either raise a false alarm or,
        worse, train the reader to ignore the real one.

    A key absent from *every* cell is neither: nothing was compared, so there
    is nothing to report, and listing it would be pure noise.

    Args:
        cells: Per-cell dicts carrying a "config" block.
    Returns:
        (conflicts, unrecorded), each key -> {run_id: value}. Both empty means
        no key was compared and found wanting.
    """
    conflicts, unrecorded = {}, {}
    for key in PROTOCOL_KEYS:
        values = {cell["run_id"]: cell["config"].get(key) for cell in cells}
        recorded = {k: v for k, v in values.items() if v is not None}
        if len(set(map(repr, recorded.values()))) > 1:
            conflicts[key] = values
        elif 0 < len(recorded) < len(values):
            unrecorded[key] = values
    return conflicts, unrecorded


def score_cell(results: dict, representation: str) -> dict:
    """Compute every tracked quantity for one model's run.

    Args:
        results: Pipeline results dict (CLAUDE.md Section 8 schema).
        representation: Metric input representation.
    Returns:
        A cell record: identity, pass@1, per-metric correlations, the consensus
        quartile table and spread, the confidently-wrong rate, and the mean
        parse-failure rate.
    """
    prompts = load_run_prompts(results)
    rows = build_problem_level_dataset(
        results,
        include_codebert=False,
        codebert_model="microsoft/codebert-base",
        representation=representation,
        prompts=prompts,
        # tree_edit_distance_diversity is not in TRACKED_METRICS, and the
        # Zhang-Shasha computation dominates runtime on library-heavy corpora
        # (hours over the five BigCodeBench rungs). Omitting it leaves every
        # scored column unchanged.
        include_tree_edit=False,
    )

    pass_rates = np.array([row["pass_rate"] for row in rows], dtype=float)
    correlations = {}
    for metric in TRACKED_METRICS:
        values = np.array([row.get(metric, np.nan) for row in rows], dtype=float)
        if np.isnan(values).any() or np.ptp(values) == 0.0:
            correlations[metric] = {"pearson_r": None, "spearman_rho": None}
            continue
        correlations[metric] = {
            "pearson_r": float(scipy_stats.pearsonr(values, pass_rates)[0]),
            "spearman_rho": float(scipy_stats.spearmanr(values, pass_rates).statistic),
        }

    quartiles = quantile_table(rows, CONSENSUS_FEATURE, n_groups=4)
    spread = (
        quartiles[0]["frac_solved"] - quartiles[-1]["frac_solved"]
        if quartiles
        else None
    )
    parse_failures = [
        problem["uncertainty"]["parse_failure_rate"]
        for problem in results["per_problem"]
        if problem.get("uncertainty", {}).get("parse_failure_rate") is not None
    ]

    return {
        "run_id": results.get("run_id"),
        "model": results["config"]["model"],
        "config": with_derived_task_ids_hash(results),
        "pass_at_1": results["summary"]["mean_pass_at_1"],
        "n_solved": int(sum(1 for row in rows if row["solved"])),
        "n_problems": len(rows),
        "correlations": correlations,
        "consensus": {
            "feature": CONSENSUS_FEATURE,
            "quartiles": quartiles,
            "spread": spread,
            "confidently_wrong": confidently_wrong_rate(rows, CONSENSUS_FEATURE),
        },
        "mean_parse_failure_rate": (
            float(np.mean(parse_failures)) if parse_failures else None
        ),
    }


def observed_quantity(cell: dict, key: str) -> float | None:
    """Pull the observed value a pre-registered fit is stated over."""
    if key == "confidently_wrong_rate":
        entry = cell["consensus"]["confidently_wrong"]
        return entry["rate"] if entry else None
    if key == "ast_abs_spearman":
        rho = cell["correlations"][CONSENSUS_FEATURE]["spearman_rho"]
        return abs(rho) if rho is not None else None
    raise ValueError(f"unknown pre-registered quantity: {key}")


def score_predictions(cells: list[dict], known_run_ids: set[str]) -> dict:
    """Score P1-P5 against the pre-registration.

    Cells the fits were *derived from* are marked rather than silently scored:
    a two-point fit reproduces its own two points exactly, so counting them as
    successes would be circular.

    Args:
        cells: Scored cell records.
        known_run_ids: Run ids whose values the fits were derived from.
    Returns:
        Per-prediction results, each with per-cell rows and a verdict.
    """
    scored: dict[str, dict] = {}

    for key, fit in PREREGISTERED.items():
        rows = []
        for cell in cells:
            observed = observed_quantity(cell, key)
            if observed is None:
                continue
            predicted = fit["intercept"] + fit["slope"] * cell["pass_at_1"]
            if fit["floor"] is not None:
                predicted = max(fit["floor"], predicted)
            rows.append(
                {
                    "model": cell["model"],
                    "pass_at_1": cell["pass_at_1"],
                    "predicted": predicted,
                    "observed": observed,
                    "deviation": observed - predicted,
                    "within_tolerance": abs(observed - predicted) <= fit["tolerance"],
                    "fit_was_derived_from_this_cell": cell["run_id"] in known_run_ids,
                }
            )
        held_out = [r for r in rows if not r["fit_was_derived_from_this_cell"]]
        scored[key] = {
            "label": fit["label"],
            "tolerance": fit["tolerance"],
            "rows": rows,
            "n_held_out": len(held_out),
            "n_held_out_within_tolerance": sum(
                1 for r in held_out if r["within_tolerance"]
            ),
        }

    # P2: the spread stays in band, and does not trend with capability.
    spreads = [
        (cell["model"], cell["pass_at_1"], cell["consensus"]["spread"])
        for cell in cells
        if cell["consensus"]["spread"] is not None
    ]
    trend = None
    if len(spreads) >= 3:
        result = scipy_stats.pearsonr([s[1] for s in spreads], [s[2] for s in spreads])
        trend = {"pearson_r": float(result[0]), "p_value": float(result[1])}
    scored["P2_spread_band"] = {
        "label": "P2 quartile spread stays in [0.40, 0.70] with no capability trend",
        "band": list(P2_SPREAD_BAND),
        "rows": [
            {
                "model": model,
                "pass_at_1": pass_at_1,
                "spread": spread,
                "in_band": P2_SPREAD_BAND[0] <= spread <= P2_SPREAD_BAND[1],
            }
            for model, pass_at_1, spread in spreads
        ],
        "trend_vs_pass_at_1": trend,
    }

    # P4: the crash regime disappears.
    scored["P4_parse_failure"] = {
        "label": f"P4 mean parse-failure rate < {P4_PARSE_FAILURE_CEILING}",
        "ceiling": P4_PARSE_FAILURE_CEILING,
        "rows": [
            {
                "model": cell["model"],
                "pass_at_1": cell["pass_at_1"],
                "mean_parse_failure_rate": cell["mean_parse_failure_rate"],
                "below_ceiling": (
                    cell["mean_parse_failure_rate"] < P4_PARSE_FAILURE_CEILING
                    if cell["mean_parse_failure_rate"] is not None
                    else None
                ),
            }
            for cell in cells
        ],
    }

    # P5: capability is not scale — the ladder's order by pass@1.
    scored["P5_capability_order"] = {
        "label": "P5 ranking by measured pass@1 (vs. by parameter count)",
        "rows": [
            {"model": cell["model"], "pass_at_1": cell["pass_at_1"]}
            for cell in sorted(cells, key=lambda c: c["pass_at_1"])
        ],
    }
    return scored


def print_report(
    cells: list[dict], scored: dict, conflicts: dict, unrecorded: dict
) -> None:
    """Print the ladder table and the pre-registration scoring."""
    if conflicts:
        print("\n!! PROTOCOL CONFLICT — these cells differ in more than the model:")
        for key, values in conflicts.items():
            print(f"   {key}: {values}")
        print("   Every comparison below is confounded by the above.\n")
    if unrecorded:
        print(
            "\nNote — not recorded by the older run(s), so unverifiable rather "
            "than known-different:"
        )
        for key, values in unrecorded.items():
            print(f"   {key}: {values}")
        print()

    print("=== the ladder (ordered by measured pass@1) ===")
    header = (
        f"{'model':<34}{'pass@1':<10}{'solved':<10}{'AST rho':<11}"
        f"{'spread':<10}{'conf-wrong':<13}{'parse-fail':<11}"
    )
    print(header)
    print("-" * len(header))
    for cell in sorted(cells, key=lambda c: c["pass_at_1"]):
        rho = cell["correlations"][CONSENSUS_FEATURE]["spearman_rho"]
        spread = cell["consensus"]["spread"]
        wrong = cell["consensus"]["confidently_wrong"]
        parse_failure = cell["mean_parse_failure_rate"]

        name = cell["model"].split("/")[-1]
        solved = f"{cell['n_solved']}/{cell['n_problems']}"
        rho_cell = f"{rho:+.3f}" if rho is not None else "n/a"
        spread_cell = f"{spread:.3f}" if spread is not None else "n/a"
        wrong_cell = f"{wrong['n_all_fail']}/{wrong['n_problems']}" if wrong else "n/a"
        parse_cell = f"{parse_failure:.4f}" if parse_failure is not None else "n/a"

        print(
            f"{name:<34}{cell['pass_at_1']:<10.4f}{solved:<10}"
            f"{rho_cell:<11}{spread_cell:<10}{wrong_cell:<13}{parse_cell:<11}"
        )

    for key in ("confidently_wrong_rate", "ast_abs_spearman"):
        block = scored[key]
        print(f"\n=== {block['label']} (tolerance +-{block['tolerance']}) ===")
        print(
            f"{'model':<34}{'pass@1':<10}{'predicted':<12}{'observed':<12}"
            f"{'dev':<10}{'ok':<5}{'note'}"
        )
        print("-" * 96)
        for row in block["rows"]:
            note = (
                "fit derived from this cell"
                if row["fit_was_derived_from_this_cell"]
                else ""
            )
            print(
                f"{row['model'].split('/')[-1]:<34}{row['pass_at_1']:<10.4f}"
                f"{row['predicted']:<12.3f}{row['observed']:<12.3f}"
                f"{row['deviation']:<+10.3f}"
                f"{('yes' if row['within_tolerance'] else 'NO'):<5}{note}"
            )
        print(
            f"held-out cells within tolerance: "
            f"{block['n_held_out_within_tolerance']}/{block['n_held_out']}"
        )

    block = scored["P2_spread_band"]
    print(f"\n=== {block['label']} ===")
    for row in block["rows"]:
        print(
            f"{row['model'].split('/')[-1]:<34}pass@1={row['pass_at_1']:.4f}  "
            f"spread={row['spread']:.3f}  in band: "
            f"{'yes' if row['in_band'] else 'NO'}"
        )
    if block["trend_vs_pass_at_1"]:
        trend = block["trend_vs_pass_at_1"]
        print(
            f"trend vs pass@1: r={trend['pearson_r']:+.3f}, p={trend['p_value']:.3f}"
            f" ({'no significant trend — P2 holds' if trend['p_value'] > 0.05 else 'SIGNIFICANT — P2 falsified'})"
        )

    block = scored["P4_parse_failure"]
    print(f"\n=== {block['label']} ===")
    for row in block["rows"]:
        value = row["mean_parse_failure_rate"]
        print(
            f"{row['model'].split('/')[-1]:<34}"
            f"{(f'{value:.4f}' if value is not None else 'n/a'):<10}"
            f"{'below' if row['below_ceiling'] else 'ABOVE'}"
        )

    print(f"\n=== {scored['P5_capability_order']['label']} ===")
    for row in scored["P5_capability_order"]["rows"]:
        print(f"  {row['pass_at_1']:.4f}  {row['model']}")


def plot_capability_sweep(cells: list[dict], metric: str, out_path: Path) -> None:
    """Plot each tracked quantity against measured pass@1.

    Args:
        cells: Scored cell records.
        metric: Metric to headline in the correlation panel.
        out_path: PNG destination.
    """
    ordered = sorted(cells, key=lambda c: c["pass_at_1"])
    x = [cell["pass_at_1"] for cell in ordered]
    labels = [cell["model"].split("/")[-1] for cell in ordered]

    panels = [
        (
            f"|Spearman rho|, {metric}",
            [
                (
                    abs(cell["correlations"][metric]["spearman_rho"])
                    if cell["correlations"][metric]["spearman_rho"] is not None
                    else np.nan
                )
                for cell in ordered
            ],
            "P3: predicted to RISE with capability",
        ),
        (
            "confidently-wrong rate (top decile)",
            [
                (
                    cell["consensus"]["confidently_wrong"]["rate"]
                    if cell["consensus"]["confidently_wrong"]
                    else np.nan
                )
                for cell in ordered
            ],
            "P1: predicted to FALL to zero",
        ),
        (
            "Q1-Q4 solved-fraction spread",
            [
                (
                    cell["consensus"]["spread"]
                    if cell["consensus"]["spread"] is not None
                    else np.nan
                )
                for cell in ordered
            ],
            "P2: predicted to stay in [0.40, 0.70], no trend",
        ),
        (
            "mean parse-failure rate",
            [cell["mean_parse_failure_rate"] or np.nan for cell in ordered],
            "P4: predicted below 0.005 at the new rungs",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.6), constrained_layout=True)
    fig.set_facecolor(_SURFACE)
    _reserve_footnote_band(fig)
    for ax, (title, values, subtitle) in zip(np.ravel(axes), panels):
        _style_axes(ax)
        ax.plot(
            x,
            values,
            color=_SERIES,
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.2,
        )
        # The extreme rungs sit against the panel edges, so a centred label
        # runs off the axes. Anchor the first point's label to its right and
        # the last point's to its left; only interior rungs get centred.
        for position, (xi, yi, label) in enumerate(zip(x, values, labels)):
            if np.isnan(yi):
                continue
            if position == 0:
                alignment, offset = "left", (4, 7)
            elif position == len(x) - 1:
                alignment, offset = "right", (-4, 7)
            else:
                alignment, offset = "center", (0, 8)
            ax.annotate(
                label,
                (xi, yi),
                textcoords="offset points",
                xytext=offset,
                fontsize=6.5,
                color=_MUTED,
                ha=alignment,
            )
        # pad leaves room for the subtitle drawn just below the title; without
        # it the two overprint each other.
        ax.set_title(title, fontsize=10, color=_INK, loc="left", pad=20)
        ax.set_xlabel(
            "measured pass@1 (capability)", fontsize=8.5, color=_INK_SECONDARY
        )
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            fontsize=7.5,
            color=_MUTED,
            ha="left",
        )
        # Headroom so the topmost point's label stays inside the panel.
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.14 * (high - low))

    fig.suptitle(
        "Do the findings hold across model capability?",
        fontsize=12.5,
        color=_INK,
        x=0.012,
        ha="left",
    )
    fig.text(
        0.012,
        0.012,
        "x-axis is measured pass@1, not parameter count — see the "
        "pre-registration's P5.",
        fontsize=7.5,
        color=_MUTED,
        ha="left",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main() -> None:
    """Score every supplied run as one rung of the capability ladder."""
    args = parse_args()
    set_all_seeds(args.seed)

    if len(args.results) < 2:
        raise SystemExit("a sweep needs at least two runs")

    cells = []
    for path in args.results:
        results = json.loads(path.read_text())
        logger.info("scoring %s (%s)", path.name, results["config"]["model"])
        cells.append(score_cell(results, args.representation))

    conflicts, unrecorded = check_protocol_match(cells)
    if conflicts:
        logger.warning("protocol conflict across cells: %s", sorted(conflicts))
    if unrecorded:
        logger.info("keys unrecorded by an older run: %s", sorted(unrecorded))

    # The two stored HE60 cells are what the two-point fits were derived from;
    # scoring them as confirmations would be circular, so they are flagged.
    derived_from = {
        cell["run_id"]
        for cell in cells
        if cell["model"]
        in ("Salesforce/codegen-350M-mono", "deepseek-ai/deepseek-coder-1.3b-base")
    }
    scored = score_predictions(cells, derived_from)
    print_report(cells, scored, conflicts, unrecorded)

    stats = {
        "analysis": "week7_capability_sweep",
        "source_results_files": [str(p) for p in args.results],
        "config": {
            "seed": args.seed,
            "representation": args.representation,
            "headline_metric": args.metric,
            "consensus_feature": CONSENSUS_FEATURE,
            "tracked_metrics": list(TRACKED_METRICS),
        },
        "protocol_conflicts": conflicts,
        "protocol_unrecorded": unrecorded,
        "cells": cells,
        "prereg_scoring": scored,
    }

    if args.dry_run:
        logger.info("--dry-run: skipping figure and stats JSON")
        return

    plot_capability_sweep(cells, args.metric, args.figure)
    path = save_results(stats, prefix="analysis_week7_capability")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
