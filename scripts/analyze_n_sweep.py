"""Week 7: does the correlation rise with the sample budget, as predicted?

Student A deliverable, and the check the pre-registration
(`reports/week7_n_sweep_preregistration.md`, committed before any of this data
existed) exists to be scored against.

THE ARGUMENT
------------
Week 7 measured that every correlation this project reports is attenuated,
because `pass_rate` and the diversity metric are both estimated from the same
ten samples. That gives a ceiling on |r| of ~0.83-0.90 at N = 10, and several
metrics already sit at 85-94% of it. Read as a caveat, this only says "our
numbers are smaller than the true ones". Read as a *model*, it makes a
prediction: raise N and every observed |r| must climb along a specific curve.

This script scores that prediction. Given one large sample pool (N = 50),
it subsamples the pool at N = 5, 10, 20, ... and, at each budget, recomputes

  * every metric and each problem's pass rate on the subset,
  * the metric-vs-pass-rate correlation across problems,
  * the split-half reliability of both, hence the *measured* ceiling at that N,

then plots observed |r| against the pre-registered prediction and against both
ceilings. Three outcomes, all reportable:

  * observed tracks the prediction  -> the measurement model is validated, and
    the gap to the big-model literature's rho ~ -0.99 is a sample-budget
    effect rather than a difference in method;
  * observed rises but undershoots  -> attenuation is part of the story, not
    all of it; the residual is the interesting quantity;
  * observed stays flat             -> the model is wrong and the correlations
    are bounded by something else. This is the falsification condition, and it
    is stated in the pre-registration rather than discovered here.

WHY THIS IS AFFORDABLE
----------------------
Nothing is re-generated and no metric is recomputed per draw. Each problem's
N x N pair-distance matrices are built once by
`estimate_reliability.build_problem_cache`, and every subset's score is then a
mean over a submatrix (`metrics_from_cache`). Both helpers are reused rather
than reimplemented, so the metric definitions here are identical to
`compute_uncertainty()`'s by construction — and that identity is re-checked at
runtime against the canonical implementation before any number is reported.

Usage:
    python scripts/analyze_n_sweep.py --results results/<n50-run>.json
    python scripts/analyze_n_sweep.py --results ... --sweep 5 10 20 30 40 50
    python scripts/analyze_n_sweep.py --results ... --representation corrected
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The palette and axis chrome are imported, not copied: they were contrast-
# validated once in analyze_uncertainty.py (see its comment block) and a
# second divergent copy would silently drift out of that validation.
from scripts.analyze_uncertainty import (
    _FAIL,
    _INK,
    _INK_SECONDARY,
    _MUTED,
    _SERIES,
    _SURFACE,
    _reserve_footnote_band,
    _style_axes,
    load_run_prompts,
)
from scripts.estimate_reliability import (
    DEFAULT_METRICS,
    _verify_cache_matches_canonical_metrics,
    build_problem_cache,
    metrics_from_cache,
)
from src.uncertainty import METRIC_REPRESENTATIONS
from src.utils import get_logger, save_results, set_all_seeds

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "reports" / "figures"

#: Sample budgets scored by default. Clipped at runtime to the pool actually
#: available, and any budget below 4 is dropped (a split-half needs two halves
#: of at least two samples each).
DEFAULT_SWEEP = (5, 10, 20, 30, 40, 50)

#: Minimum subset size that supports a split-half reliability estimate.
MIN_N_FOR_SPLIT_HALF = 4


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the N-sweep entry point."""
    parser = argparse.ArgumentParser(
        description="Observed |r| vs sample budget, against the pre-registration"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Pipeline results JSON with a large per-problem sample pool.",
    )
    parser.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWEEP),
        help=f"Sample budgets to score (default: {' '.join(map(str, DEFAULT_SWEEP))}).",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=25,
        help="Random subsets per budget (default: 25). The full-pool budget "
        "has exactly one subset and ignores this.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=50,
        help="Half-splits per subset for the reliability estimate (default: 50).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to score (default: the cheap ones, as in estimate_reliability).",
    )
    parser.add_argument(
        "--representation",
        choices=METRIC_REPRESENTATIONS,
        default="full",
        help="Metric input representation. Defaults to 'full' because the "
        "pre-registration's reliabilities were measured on 'full'; scoring a "
        "corrected sweep against a full-representation prediction would be "
        "comparing two different quantities.",
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path("results/prereg_n_sweep.json"),
        help="Pre-registration JSON from scripts/preregister_n_sweep.py. "
        "Pass a nonexistent path to score the sweep without a prediction.",
    )
    parser.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Pre-registration cell key (e.g. 'cg/HE60'). Inferred from the "
        "run's model and problem count when omitted.",
    )
    parser.add_argument(
        "--anchor-n",
        type=int,
        default=None,
        help="Budget to fit the attenuation model at for the internal "
        "consistency check (default: the smallest scored budget). The model "
        "is fit there and scored against every larger budget in the same "
        "sweep, which needs no pre-registration and no new data.",
    )
    parser.add_argument(
        "--no-anchor-check",
        action="store_true",
        help="Skip the internal anchor projection.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="PNG destination (default: reports/figures/week7_n_sweep_<label>.png).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print but do not persist outputs."
    )
    return parser.parse_args()


def build_caches(results: dict, prompts: dict[str, str], representation: str) -> list:
    """Build the per-problem pair-distance cache used by every budget.

    Args:
        results: Pipeline results dict (CLAUDE.md Section 8 schema).
        prompts: task_id -> prompt, for the corrected representations.
        representation: Metric input representation.
    Returns:
        One entry per usable problem, each with its cache, per-sample pass
        flags, and the inputs the canonical-metric self-check needs.
    Raises:
        ValueError: If fewer than three problems carry enough samples.
    """
    caches = []
    for problem in results["per_problem"]:
        samples = problem.get("samples", [])
        if len(samples) < MIN_N_FOR_SPLIT_HALF:
            continue
        codes = [s["code"] for s in samples]
        prompt = prompts.get(problem["task_id"])
        # A stored candidate is `prompt + completion`; if that does not hold
        # (e.g. a chat-template run), fall back to scoring the full string
        # rather than mis-splitting it.
        if prompt is not None and not all(c.startswith(prompt) for c in codes):
            prompt = None
        caches.append(
            {
                "task_id": problem["task_id"],
                "passed": [bool(s["passed"]) for s in samples],
                "cache": build_problem_cache(codes, prompt, representation),
                "codes": codes,
                "prompt": prompt,
            }
        )
    if len(caches) < 3:
        raise ValueError(
            f"need >= 3 problems with >= {MIN_N_FOR_SPLIT_HALF} samples, got "
            f"{len(caches)}"
        )
    return caches


def _correlations_for_subsets(
    caches: list, subsets: list, metrics: list[str]
) -> dict[str, dict[str, float | None]]:
    """Correlate each metric against pass rate across problems, for one draw.

    Args:
        caches: Output of build_caches().
        subsets: Per-problem index arrays, aligned to `caches`.
        metrics: Metric keys to score.
    Returns:
        metric -> {"pearson_r", "spearman_rho"}, with None where the metric is
        constant across problems (correlation undefined, not zero).
    """
    pass_rates = []
    values: dict[str, list[float]] = {m: [] for m in metrics}
    for entry, indices in zip(caches, subsets):
        passed = [entry["passed"][i] for i in indices]
        pass_rates.append(sum(passed) / len(passed))
        scored = metrics_from_cache(entry["cache"], indices)
        for metric in metrics:
            values[metric].append(scored[metric])

    pass_rates = np.asarray(pass_rates, dtype=float)
    out: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        column = np.asarray(values[metric], dtype=float)
        if np.ptp(column) == 0.0 or np.ptp(pass_rates) == 0.0:
            out[metric] = {"pearson_r": None, "spearman_rho": None}
            continue
        out[metric] = {
            "pearson_r": float(scipy_stats.pearsonr(column, pass_rates)[0]),
            "spearman_rho": float(scipy_stats.spearmanr(column, pass_rates).statistic),
        }
    return out


def _reliabilities_for_subsets(
    caches: list, subsets: list, metrics: list[str], n_permutations: int, rng
) -> dict[str, float | None]:
    """Split-half reliability of pass rate and each metric, within the subsets.

    Same estimator as scripts/estimate_reliability.py — random half-split,
    correlate halves across problems, Spearman-Brown correct — but restricted
    to the subset that defines the budget being scored, which is what makes it
    a reliability *at N* rather than at the pool size.

    Args:
        caches: Output of build_caches().
        subsets: Per-problem index arrays defining the budget.
        metrics: Metric keys to score.
        n_permutations: Half-splits to average over.
        rng: Seeded numpy Generator.
    Returns:
        "pass_rate" plus one entry per metric, each the mean corrected
        reliability, or None when no permutation yielded a usable estimate.
    """
    tracked = ["pass_rate", *metrics]
    collected: dict[str, list[float]] = {name: [] for name in tracked}

    for _ in range(n_permutations):
        halves: dict[str, tuple[list[float], list[float]]] = {
            name: ([], []) for name in tracked
        }
        for entry, indices in zip(caches, subsets):
            order = rng.permutation(np.asarray(indices))
            half = len(order) // 2
            for side, group in enumerate((order[:half], order[half : 2 * half])):
                passed = [entry["passed"][i] for i in group]
                halves["pass_rate"][side].append(sum(passed) / len(passed))
                scored = metrics_from_cache(entry["cache"], group)
                for metric in metrics:
                    halves[metric][side].append(scored[metric])

        for name in tracked:
            left, right = (np.asarray(v, dtype=float) for v in halves[name])
            if len(left) < 3 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
                continue
            half_r = float(np.corrcoef(left, right)[0, 1])
            corrected = 2.0 * half_r / (1.0 + half_r) if half_r > -1.0 else 0.0
            collected[name].append(max(0.0, min(1.0, corrected)))

    return {
        name: float(np.mean(vals)) if vals else None for name, vals in collected.items()
    }


def score_budget(
    caches: list,
    n: int,
    metrics: list[str],
    n_draws: int,
    n_permutations: int,
    rng,
) -> dict:
    """Score one sample budget: correlations, reliabilities, measured ceiling.

    Args:
        caches: Output of build_caches().
        n: The sample budget.
        metrics: Metric keys to score.
        n_draws: Random subsets to average over. Forced to 1 when `n` equals
            the pool size, since only one subset exists.
        n_permutations: Half-splits per subset for the reliability estimate.
        rng: Seeded numpy Generator.
    Returns:
        Dict with per-metric mean/std Pearson and Spearman across draws, the
        mean reliabilities, and the measured ceiling sqrt(r_xx * r_yy).
    """
    pool = caches[0]["cache"]["n"]
    if any(entry["cache"]["n"] != pool for entry in caches):
        # Not fatal, but the budget then means different things per problem.
        logger.warning("problems have unequal pool sizes; using each problem's own")
    exhaustive = all(entry["cache"]["n"] == n for entry in caches)
    draws = 1 if exhaustive else max(1, n_draws)

    per_draw_pearson: dict[str, list[float]] = {m: [] for m in metrics}
    per_draw_spearman: dict[str, list[float]] = {m: [] for m in metrics}
    reliability_draws: dict[str, list[float]] = {m: [] for m in ["pass_rate", *metrics]}

    for _ in range(draws):
        subsets = [
            rng.choice(
                entry["cache"]["n"], size=min(n, entry["cache"]["n"]), replace=False
            )
            for entry in caches
        ]
        correlations = _correlations_for_subsets(caches, subsets, metrics)
        for metric, stats in correlations.items():
            if stats["pearson_r"] is not None:
                per_draw_pearson[metric].append(stats["pearson_r"])
                per_draw_spearman[metric].append(stats["spearman_rho"])

        reliabilities = _reliabilities_for_subsets(
            caches, subsets, metrics, n_permutations, rng
        )
        for name, value in reliabilities.items():
            if value is not None:
                reliability_draws[name].append(value)

    def _summarize(values: list[float]) -> dict:
        return {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "n_draws_used": len(values),
        }

    pass_rate_reliability = (
        float(np.mean(reliability_draws["pass_rate"]))
        if reliability_draws["pass_rate"]
        else None
    )

    per_metric = {}
    for metric in metrics:
        metric_reliability = (
            float(np.mean(reliability_draws[metric]))
            if reliability_draws[metric]
            else None
        )
        ceiling = (
            float(np.sqrt(metric_reliability * pass_rate_reliability))
            if metric_reliability is not None and pass_rate_reliability is not None
            else None
        )
        pearson = _summarize(per_draw_pearson[metric])
        percent = (
            100.0 * abs(pearson["mean"]) / ceiling
            if pearson["mean"] is not None and ceiling
            else None
        )
        per_metric[metric] = {
            "pearson": pearson,
            "spearman": _summarize(per_draw_spearman[metric]),
            "metric_reliability": metric_reliability,
            "measured_ceiling": ceiling,
            "percent_of_measured_ceiling": percent,
        }

    return {
        "n": n,
        "n_draws": draws,
        "exhaustive": exhaustive,
        "pass_rate_reliability": pass_rate_reliability,
        "metrics": per_metric,
    }


def infer_cell_key(prereg: dict, results: dict) -> str | None:
    """Match this run to a pre-registration cell by model and problem count.

    Args:
        prereg: Loaded pre-registration JSON.
        results: Pipeline results dict.
    Returns:
        The matching cell key, or None when nothing matches unambiguously.
    """
    model = results["config"]["model"]
    n_problems = results["config"]["n_problems"]
    matches = [
        key
        for key, cell in prereg["cells"].items()
        if cell["model"] == model and cell["n_problems"] == n_problems
    ]
    if len(matches) == 1:
        return matches[0]
    logger.warning(
        "could not infer the pre-registration cell for model=%s n_problems=%s "
        "(candidates: %s); pass --cell",
        model,
        n_problems,
        matches or list(prereg["cells"]),
    )
    return None


def compare_to_prereg(sweep: list[dict], predictions: dict) -> dict:
    """Score observed |r| against the pre-registered prediction at each budget.

    Args:
        sweep: Per-budget results from score_budget().
        predictions: The matched cell's "predictions" block.
    Returns:
        metric -> list of per-budget comparison rows. Budgets with no
        prediction (not in the pre-registration's target list) are skipped.
    """
    comparison: dict[str, list[dict]] = {}
    for metric, predicted in predictions.items():
        rows = []
        for budget in sweep:
            entry = budget["metrics"].get(metric)
            if entry is None or entry["pearson"]["mean"] is None:
                continue
            # The pre-registration keys its targets by int; JSON round-trips
            # them to str.
            by_n = predicted["by_target_n"]
            key = str(budget["n"]) if str(budget["n"]) in by_n else budget["n"]
            if key not in by_n:
                continue
            observed = abs(entry["pearson"]["mean"])
            expected = by_n[key]["predicted_abs_pearson_r"]
            rows.append(
                {
                    "n": budget["n"],
                    "observed_abs_pearson_r": observed,
                    "predicted_abs_pearson_r": expected,
                    "deviation": observed - expected,
                    "predicted_ceiling": by_n[key]["ceiling"],
                    "measured_ceiling": entry["measured_ceiling"],
                }
            )
        if rows:
            comparison[metric] = rows
    return comparison


def anchor_projection(sweep: list[dict], anchor_n: int) -> dict:
    """Fit the attenuation model at one budget and score it at the larger ones.

    This is the pre-registration's argument turned inward: instead of comparing
    a new run against a prediction made from an old one, it fits the model at
    the sweep's smallest budget and asks whether it predicts the sweep's own
    larger budgets. It needs no pre-registration and no new generation, so it
    can be run on any existing pool — including, before spending compute on a
    bigger run, on the N = 10 pools this project already has.

    A model that cannot predict N = 10 from N = 4 within a pool has no business
    predicting N = 50 from N = 10 across pools.

    Args:
        sweep: Per-budget results from score_budget(), ascending in n.
        anchor_n: The budget to fit at.
    Returns:
        metric -> list of rows for every budget above the anchor, each with the
        predicted and observed |Pearson r| and their difference. Metrics whose
        anchor reliability is degenerate are omitted.
    """
    from scripts.preregister_n_sweep import (
        ceiling,
        project_reliability,
        single_sample_reliability,
    )

    by_n = {budget["n"]: budget for budget in sweep}
    if anchor_n not in by_n:
        raise ValueError(f"anchor budget {anchor_n} was not scored")
    anchor = by_n[anchor_n]
    if anchor["pass_rate_reliability"] is None:
        logger.warning("anchor budget %d has no pass-rate reliability", anchor_n)
        return {}

    try:
        pass_rate_r1 = single_sample_reliability(
            anchor["pass_rate_reliability"], n=anchor_n
        )
    except ValueError as exc:
        logger.warning("anchor pass-rate reliability unusable: %s", exc)
        return {}

    projection: dict[str, list[dict]] = {}
    for metric, entry in anchor["metrics"].items():
        if entry["metric_reliability"] is None or entry["pearson"]["mean"] is None:
            continue
        try:
            metric_r1 = single_sample_reliability(
                entry["metric_reliability"], n=anchor_n
            )
        except ValueError as exc:
            logger.warning("anchor reliability for %s unusable: %s", metric, exc)
            continue
        anchor_ceiling = ceiling(
            entry["metric_reliability"], anchor["pass_rate_reliability"]
        )
        if not anchor_ceiling:
            continue
        implied_true_r = abs(entry["pearson"]["mean"]) / anchor_ceiling

        rows = []
        for budget in sweep:
            if budget["n"] <= anchor_n:
                continue
            target = budget["metrics"].get(metric)
            if target is None or target["pearson"]["mean"] is None:
                continue
            projected_ceiling = ceiling(
                project_reliability(metric_r1, budget["n"]),
                project_reliability(pass_rate_r1, budget["n"]),
            )
            predicted = min(implied_true_r * projected_ceiling, 1.0)
            observed = abs(target["pearson"]["mean"])
            rows.append(
                {
                    "n": budget["n"],
                    "predicted_abs_pearson_r": predicted,
                    "observed_abs_pearson_r": observed,
                    "deviation": observed - predicted,
                }
            )
        if rows:
            projection[metric] = rows
    return projection


def print_anchor_table(projection: dict, anchor_n: int) -> None:
    """Print the internal anchor-projection check."""
    if not projection:
        print("\n(anchor projection unavailable)")
        return
    print(
        f"\n=== internal check: model fit at N={anchor_n}, scored at larger budgets ==="
    )
    print(f"{'metric':<30}{'N':<6}{'predicted':<12}{'observed':<12}{'deviation':<12}")
    print("-" * 72)
    deviations = []
    for metric, rows in projection.items():
        for row in rows:
            deviations.append(abs(row["deviation"]))
            print(
                f"{metric:<30}{row['n']:<6}"
                f"{row['predicted_abs_pearson_r']:<12.3f}"
                f"{row['observed_abs_pearson_r']:<12.3f}"
                f"{row['deviation']:<+12.3f}"
            )
    print(
        f"\nmean |deviation| = {np.mean(deviations):.3f}, "
        f"max = {np.max(deviations):.3f} over {len(deviations)} predictions"
    )


def print_sweep_table(sweep: list[dict], comparison: dict, metrics: list[str]) -> None:
    """Print the observed-vs-predicted table (scripts may print; src may not)."""
    print("\n=== observed |Pearson r| by sample budget ===")
    header = f"{'metric':<30}" + "".join(f"N={b['n']:<10}" for b in sweep)
    print(header)
    print("-" * len(header))
    for metric in metrics:
        cells = []
        for budget in sweep:
            entry = budget["metrics"].get(metric)
            mean = entry["pearson"]["mean"] if entry else None
            cells.append(f"{abs(mean):<12.3f}" if mean is not None else f"{'n/a':<12}")
        print(f"{metric:<30}" + "".join(cells))

    if not comparison:
        print("\n(no pre-registration matched; nothing to score against)")
        return

    print("\n=== against the pre-registration (observed - predicted) ===")
    print(f"{'metric':<30}{'N':<6}{'observed':<12}{'predicted':<12}{'deviation':<12}")
    print("-" * 72)
    for metric, rows in comparison.items():
        for row in rows:
            print(
                f"{metric:<30}{row['n']:<6}"
                f"{row['observed_abs_pearson_r']:<12.3f}"
                f"{row['predicted_abs_pearson_r']:<12.3f}"
                f"{row['deviation']:<+12.3f}"
            )


def plot_n_sweep(
    sweep: list[dict],
    comparison: dict,
    metrics: list[str],
    out_path: Path,
    run_label: str,
) -> None:
    """Plot observed |r| vs budget against prediction and both ceilings.

    One panel per metric. The prediction is drawn as a dashed line so the
    question the figure answers — does the solid line follow the dashed one —
    is readable without the caption.

    Args:
        sweep: Per-budget results from score_budget().
        comparison: Output of compare_to_prereg() (may be empty).
        metrics: Metrics to panel, in order.
        out_path: PNG destination.
        run_label: Subtitle line identifying the run.
    """
    panelled = [m for m in metrics if any(b["metrics"].get(m) for b in sweep)]
    if not panelled:
        raise ValueError("no metric has a scored budget to plot")

    ncols = 2
    nrows = int(np.ceil(len(panelled) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9.4, 3.4 * nrows + 0.8), constrained_layout=True
    )
    fig.set_facecolor(_SURFACE)
    _reserve_footnote_band(fig)
    axes = np.atleast_1d(axes).ravel()

    budgets = [b["n"] for b in sweep]
    for ax, metric in zip(axes, panelled):
        _style_axes(ax)
        observed, errors = [], []
        for budget in sweep:
            entry = budget["metrics"].get(metric)
            mean = entry["pearson"]["mean"] if entry else None
            observed.append(abs(mean) if mean is not None else np.nan)
            errors.append((entry["pearson"]["std"] or 0.0) if entry else 0.0)
        observed = np.asarray(observed)
        errors = np.asarray(errors)

        measured_ceiling = np.asarray(
            [
                (budget["metrics"].get(metric) or {}).get("measured_ceiling") or np.nan
                for budget in sweep
            ]
        )
        ax.plot(
            budgets,
            measured_ceiling,
            color=_MUTED,
            linewidth=1.2,
            linestyle=":",
            label="measured ceiling",
        )

        rows = comparison.get(metric, [])
        if rows:
            ax.plot(
                [r["n"] for r in rows],
                [r["predicted_abs_pearson_r"] for r in rows],
                color=_FAIL,
                linewidth=1.8,
                linestyle="--",
                marker="s",
                markersize=4.5,
                label="pre-registered",
            )

        ax.fill_between(
            budgets,
            observed - errors,
            observed + errors,
            color=_SERIES,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            budgets,
            observed,
            color=_SERIES,
            linewidth=2.0,
            marker="o",
            markersize=5,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.2,
            label="observed",
        )

        ax.set_title(metric, fontsize=10, color=_INK, loc="left")
        ax.set_xlabel("samples per problem (N)", fontsize=8.5, color=_INK_SECONDARY)
        ax.set_ylabel("|Pearson r| with pass rate", fontsize=8.5, color=_INK_SECONDARY)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(budgets)
        ax.legend(
            frameon=False, fontsize=7.5, labelcolor=_INK_SECONDARY, loc="lower right"
        )

    for ax in axes[len(panelled) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Does the correlation climb with the sample budget?",
        fontsize=12.5,
        color=_INK,
        x=0.012,
        ha="left",
    )
    # Two lines, not one: at 9.4in and 7.5pt a single-line footnote runs off
    # the right edge and the legend key is what gets cut.
    fig.text(
        0.012,
        0.026,
        "solid: observed (band = SD across subsets)   ·   dashed: pre-registered "
        "prediction   ·   dotted: measured ceiling",
        fontsize=7.5,
        color=_MUTED,
        ha="left",
    )
    fig.text(0.012, 0.008, run_label, fontsize=7.5, color=_MUTED, ha="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main() -> None:
    """Score the sweep, compare it to the pre-registration, persist both."""
    args = parse_args()
    set_all_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    results = json.loads(args.results.read_text())
    pool = max(len(p.get("samples", [])) for p in results["per_problem"])
    budgets = sorted({min(n, pool) for n in args.sweep if n >= MIN_N_FOR_SPLIT_HALF})
    if not budgets:
        raise SystemExit(
            f"no budget >= {MIN_N_FOR_SPLIT_HALF} in --sweep; a split-half "
            "reliability needs two halves of at least two samples"
        )
    logger.info("pool=%d samples/problem; scoring budgets %s", pool, budgets)
    if pool < max(args.sweep):
        logger.warning(
            "requested budgets above the pool size were clipped to %d — this "
            "run does not have the samples to score them",
            pool,
        )

    prompts = load_run_prompts(results)
    caches = build_caches(results, prompts, args.representation)
    logger.info("built caches for %d problems", len(caches))
    # Fail loudly if the submatrix shortcut has drifted from the canonical
    # metric definitions; every number below depends on that identity.
    _verify_cache_matches_canonical_metrics(
        caches[0], args.metrics, args.representation
    )

    sweep = [
        score_budget(caches, n, args.metrics, args.n_draws, args.n_permutations, rng)
        for n in budgets
    ]
    for budget in sweep:
        logger.info(
            "N=%d scored (%d draw%s)",
            budget["n"],
            budget["n_draws"],
            "" if budget["n_draws"] == 1 else "s",
        )

    comparison: dict = {}
    cell_key = None
    if args.prereg.exists():
        prereg = json.loads(args.prereg.read_text())
        cell_key = args.cell or infer_cell_key(prereg, results)
        if cell_key and cell_key in prereg["cells"]:
            comparison = compare_to_prereg(
                sweep, prereg["cells"][cell_key]["predictions"]
            )
            if not comparison:
                logger.warning(
                    "cell %r matched, but none of the scored budgets %s appear "
                    "in the pre-registered targets %s — re-run with a --sweep "
                    "that includes them",
                    cell_key,
                    budgets,
                    prereg["target_n"],
                )
        elif cell_key:
            logger.warning("cell %r not in the pre-registration", cell_key)
    else:
        logger.warning(
            "no pre-registration at %s; reporting observed only", args.prereg
        )

    projection: dict = {}
    anchor_n = args.anchor_n if args.anchor_n is not None else budgets[0]
    if not args.no_anchor_check and len(budgets) > 1:
        projection = anchor_projection(sweep, anchor_n)

    print_sweep_table(sweep, comparison, args.metrics)
    if projection:
        print_anchor_table(projection, anchor_n)

    model_label = results["config"]["model"].split("/")[-1]
    run_label = (
        f"{results['config']['dataset']} · {model_label} · "
        f"{results['config']['n_problems']} problems · pool N={pool} · "
        f"representation={args.representation}"
    )
    figure_path = args.figure or (
        DEFAULT_FIGURES_DIR
        / f"week7_n_sweep_{results['config']['dataset']}_{model_label}.png"
    )

    stats = {
        "analysis": "week7_n_sweep",
        "source_results_file": str(args.results),
        "source_run_id": results.get("run_id"),
        "source_config": results["config"],
        "config": {
            "seed": args.seed,
            "sweep": budgets,
            "pool": pool,
            "n_draws": args.n_draws,
            "n_permutations": args.n_permutations,
            "metrics": args.metrics,
            "representation": args.representation,
            "prereg_file": str(args.prereg) if args.prereg.exists() else None,
            "prereg_cell": cell_key,
            "anchor_n": anchor_n if projection else None,
        },
        "sweep": sweep,
        "prereg_comparison": comparison,
        "anchor_projection": projection,
    }

    if args.dry_run:
        logger.info("--dry-run: skipping figure and stats JSON")
        return

    plot_n_sweep(sweep, comparison, args.metrics, figure_path, run_label)
    path = save_results(stats, prefix="analysis_week7_nsweep")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
