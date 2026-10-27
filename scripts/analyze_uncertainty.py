"""Correlation analysis: uncertainty metrics vs execution outcomes.

Student A deliverable (CLAUDE.md Section 13; introduced Week 2, extended
Week 3 with tree edit distance + CodeBERT embedding diversity, extended again
post-Week-3 with the inter-metric redundancy heatmap).

Loads a pipeline results JSON (CLAUDE.md Section 8 schema), *recomputes* all
uncertainty metrics from the stored per-sample code (so runs recorded before
a metric existed are analyzable without re-generating), and produces:

  1. reports/figures/{prefix}_metric_vs_passrate.png — per-problem scatter of
     each metric against initial pass rate, with a least-squares trend line
     and Pearson/Spearman coefficients.
  2. reports/figures/{prefix}_metric_by_outcome.png — per-sample distribution
     of each metric grouped by pass/fail (box + jittered strip), with the
     point-biserial correlation.
  3. reports/figures/{prefix}_correlation_summary.png — signed Spearman ρ per
     metric, the "which metric is most predictive so far" view.
  4. reports/figures/{prefix}_correlation_heatmap.png — pairwise Spearman ρ
     between every metric, the "which metrics are redundant with each other"
     view (backs the report's redundancy observations with a figure instead
     of prose-only numbers).
  5. A timestamped stats JSON in results/ with every coefficient, p-value,
     the inter-metric correlation matrix, and per-problem metric table
     (CLAUDE.md Constraint #5: results are always persisted).

Week 7 additions (the prompt-inclusion confound — see src/uncertainty.py's
"prompt-inclusion confound" section for the mechanism and the measured
impact):

  6. --representation selects which string each metric family is computed
     over. Default "full" = the Weeks 2-6 behavior, so every prior figure
     and number reproduces unchanged; "corrected" removes the shared prompt
     prefix per family.
  7. A confound table (stdout + "confound_analysis" in the stats JSON) with
     the completion-length baseline row and partial correlations
     controlling for length and for prompt share. This is the row a
     reviewer would add: mean completion length alone recovers 71-96% of
     ast_node_diversity's correlation magnitude across this project's six
     regime cells, so the metric's claimable contribution is the partial
     that survives the control, not the raw coefficient.

Usage:
    python scripts/analyze_uncertainty.py                  # latest run in results/
    python scripts/analyze_uncertainty.py --results results/humaneval_uncertainty_20260702_120000.json
    python scripts/analyze_uncertainty.py --fig-prefix week3   # names figures + stats JSON
    python scripts/analyze_uncertainty.py --no-codebert    # skip the ~500 MB encoder
    python scripts/analyze_uncertainty.py --representation corrected  # Week 7 fix
    python scripts/analyze_uncertainty.py --dry-run        # figures only, skip stats JSON
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes PNGs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.uncertainty import (
    METRIC_REPRESENTATIONS,
    compute_uncertainty,
    compute_uncertainty_prompt_aware,
    prompt_confound_covariates,
)
from src.utils import (
    RESULTS_DIR,
    get_logger,
    load_dataset_split,
    save_results,
    set_all_seeds,
)

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "reports" / "figures"

# Canonical metric order everywhere (figures, tables, stats JSON). Metrics
# not present in a given analysis (e.g. codebert under --no-codebert) are
# skipped wherever this order is consumed.
METRIC_ORDER = [
    "lexical_diversity",
    "token_jaccard_diversity",
    "token_cosine_diversity",
    "ast_node_diversity",
    "tree_edit_distance_diversity",
    "codebert_cosine_diversity",
    "codebert_bertscore_diversity",
    "pairwise_distance",
    "parse_failure_rate",
]
METRIC_LABELS = {
    "lexical_diversity": "Lexical diversity",
    "token_jaccard_diversity": "Token Jaccard diversity",
    "token_cosine_diversity": "Token cosine diversity",
    "ast_node_diversity": "AST node diversity",
    "tree_edit_distance_diversity": "Tree edit distance diversity",
    "codebert_cosine_diversity": "CodeBERT cosine diversity (mean-pooled)",
    "codebert_bertscore_diversity": "CodeBERT BERTScore diversity (per-token)",
    "pairwise_distance": "Pairwise length distance (legacy)",
    "parse_failure_rate": "Parse failure rate",
}

# Week 7 confound covariates (src/uncertainty.py's prompt-inclusion section).
# Not diversity metrics: they never enter METRIC_ORDER, the figures or the
# FDR family. They are controls, plus one free baseline.
#
#   mean_completion_chars — the trivial baseline. Across this project's six
#     regime cells it recovers 71-96% of ast_node_diversity's correlation
#     magnitude with pass rate, needing no sampling, parsing, embedding or
#     execution. It is a *substantive* confound (longer solutions are
#     genuinely harder and genuinely more varied) and plausibly on the causal
#     path, so it is controlled for and reported, never stripped.
#   prompt_share_char — the *measurement* artefact: the fraction of the
#     compared string that is identical prompt. Fixed by --representation,
#     quantified here.
CONFOUND_BASELINE = "mean_completion_chars"
COVARIATE_ORDER = [
    "mean_completion_chars",
    "mean_completion_tokens",
    "prompt_share_char",
    "prompt_share_token",
    "prompt_share_ast",
    "prompt_chars",
    "prompt_tokens",
]
# Control sets for the partial-correlation columns, in report order.
PARTIAL_CONTROL_SETS = {
    "ctrl_completion_len": ["mean_completion_chars"],
    "ctrl_both_lengths": ["mean_completion_chars", "prompt_chars"],
    "ctrl_prompt_share": ["prompt_share_char"],
}

# Colors from the validated data-viz reference palette (light mode).
# Scatter = one series → categorical slot 1; trend = a darker step of the
# same blue ramp (sequential 550 — a ramp step, not a second identity).
# Pass/fail groups *mean* good/bad, so they wear the status tokens
# (good/critical), never plain series hues; group identity is also carried
# by axis position + text labels, never by color alone. Signed correlation
# bars encode polarity → the diverging blue/red pole pair.
#
# Validator record (scripts/validate_palette.js, --mode light, surface
# #fcfcfb, re-run 2026-07-05 — computed, not eyeballed):
#   status pair  #0ca30c/#d03b3b  worst ΔE 12.4 (deutan)  ≥ 12 target  PASS
#   diverging    #2a78d6/#e34948  worst ΔE 74.6 (protan)              PASS
#   series blue  #2a78d6          ≥ 3:1 on surface                    PASS
# Scope notes (static light-mode PNG for the printed report): no hover
# layer or dark variant — the table-view twin is print_correlation_table()
# + the persisted stats JSON + the report's markdown table. Deviation from
# the bar mark spec: data-ends are square, not 4px-rounded (matplotlib
# needs post-layout transform math for that; revisit if figures move to
# HTML/SVG).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SERIES = "#2a78d6"
_TREND = "#1c5cab"
_PASS = "#0ca30c"
_FAIL = "#d03b3b"
_NEG_POLE = "#2a78d6"
_POS_POLE = "#e34948"

# Sequential blue ramp, steps 100->700 (reference palette's "Sequential hue"
# table) — the magnitude channel for the inter-metric redundancy heatmap.
# Same source palette as every other color in this file; not re-validated
# separately, since it's the reference instance unchanged, not a new palette.
_SEQUENTIAL_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the analysis entry point."""
    parser = argparse.ArgumentParser(
        description="Correlate uncertainty metrics with execution outcomes"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Pipeline results JSON to analyze (default: newest run in results/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for the output PNG figures (default: reports/figures/).",
    )
    parser.add_argument(
        "--fig-prefix",
        type=str,
        default="week3",
        help="Prefix naming the figures and the stats JSON (default: week3).",
    )
    parser.add_argument(
        "--representation",
        choices=METRIC_REPRESENTATIONS,
        default="full",
        help="Which string each metric family is computed over (Week 7 "
        "prompt-inclusion confound; see src/uncertainty.py). 'full' is the "
        "Weeks 2-6 behavior and stays the default so every prior figure "
        "reproduces; 'corrected' puts the token/lexical/embedding families "
        "on completion-only strings and the structural families on "
        "docstring-stripped signature + completion.",
    )
    parser.add_argument(
        "--codebert-model",
        type=str,
        default="microsoft/codebert-base",
        help="HuggingFace encoder for codebert_cosine_diversity.",
    )
    parser.add_argument(
        "--no-codebert",
        action="store_true",
        help="Skip codebert_cosine_diversity (avoids the ~500 MB encoder "
        "download; all cheap metrics are still analyzed).",
    )
    parser.add_argument(
        "--no-tree-edit",
        action="store_true",
        help="Skip tree_edit_distance_diversity. Zhang-Shasha dominates "
        "runtime on library-heavy corpora (minutes per problem on "
        "BigCodeBench, whose runs are generated with --no-tree-edit anyway); "
        "no other metric column changes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for strip jitter.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write figures but skip persisting the stats JSON to results/.",
    )
    return parser.parse_args()


def find_latest_results_file() -> Path:
    """Return the newest pipeline results JSON in results/.

    Skips non-run JSON files (e.g. this script's own analysis outputs) by
    requiring the Section 8 schema's "per_problem" and "config" keys.
    """
    candidates = sorted(
        RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for path in candidates:
        try:
            with path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if "per_problem" in data and "config" in data:
            return path
    raise FileNotFoundError(
        f"No pipeline results JSON found in {RESULTS_DIR}. "
        "Run scripts/run_pipeline.py first."
    )


def load_run_prompts(results: dict) -> dict[str, str]:
    """Map task_id -> prompt for the dataset a stored run was generated from.

    The Section 8 results schema records prompt_length but not the prompt
    itself, so the prompt is recovered from the dataset and validated
    against that stored length. Needed by everything Week 7 added: the
    prompt-stripped representations and the completion-length baseline both
    require knowing where the prompt ends.

    Args:
        results: Full pipeline results (CLAUDE.md Section 8 schema).
    Returns:
        task_id -> prompt for the run's dataset; empty dict if the dataset
        cannot be loaded (offline with no cache), which callers treat as
        "fall back to uncorrected analysis".
    """
    dataset = results.get("config", {}).get("dataset")
    if not dataset:
        logger.warning("Run config has no dataset field; cannot recover prompts")
        return {}
    try:
        problems = load_dataset_split(dataset)
    except Exception as exc:  # dataset download/cache failure is not fatal
        logger.warning("Could not load %s to recover prompts: %s", dataset, exc)
        return {}
    return {p["task_id"]: p["prompt"] for p in problems}


def extract_problem_table(
    results: dict,
    include_codebert: bool = True,
    codebert_model: str = "microsoft/codebert-base",
    representation: str = "full",
    prompts: dict[str, str] | None = None,
    include_tree_edit: bool = True,
) -> list[dict]:
    """Build one analysis row per problem from a pipeline results dict.

    Metrics are recomputed from the stored sample code, so the analysis
    always reflects the current metric implementations regardless of when
    the run was recorded — this is how Week 3 metrics get scored on runs
    recorded in Week 2, and how Week 7's corrected representations get
    scored on every run recorded before the confound was found.

    Args:
        results: Full pipeline results (CLAUDE.md Section 8 schema).
        include_codebert: Forwarded to the metric computation; False skips
            the embedding metric and its encoder download.
        codebert_model: HuggingFace encoder for the embedding metric.
        representation: One of src.uncertainty.METRIC_REPRESENTATIONS.
            "full" reproduces every pre-Week-7 number exactly.
        prompts: task_id -> prompt (load_run_prompts()). Required for any
            representation other than "full", and for the Week 7 confound
            covariates. When None or missing a problem's task_id, that
            problem falls back to "full" with no covariates and a warning.
        include_tree_edit: Forwarded to the metric computation. False drops
            tree_edit_distance_diversity, which dominates the runtime on the
            larger runs; callers that do not read that column (e.g. the
            suppressor diagnostic) can turn it off.
    Returns:
        List of dicts with task_id, pass_rate, passed flags, one key per
        uncertainty metric, and (when the prompt is known) the Week 7
        confound covariates.
    """
    prompts = prompts or {}
    rows = []
    for problem in results["per_problem"]:
        samples = problem.get("samples", [])
        if not samples:
            logger.warning("Skipping %s: no stored samples", problem.get("task_id"))
            continue
        codes = [s["code"] for s in samples]
        passed = [bool(s["passed"]) for s in samples]
        row = {
            "task_id": problem["task_id"],
            "pass_rate": sum(passed) / len(passed),
            "passed_flags": passed,
        }
        prompt = prompts.get(problem["task_id"])
        # A prompt that does not prefix its own samples means the stored run
        # and the dataset have diverged; degrade to the uncorrected analysis
        # rather than silently mis-splitting the strings.
        if prompt is not None and not all(c.startswith(prompt) for c in codes):
            logger.warning(
                "%s: stored samples do not start with the dataset prompt; "
                "falling back to representation='full' for this problem",
                problem["task_id"],
            )
            prompt = None
        if prompt is None:
            if representation != "full":
                logger.warning(
                    "%s: no usable prompt, computing metrics on full strings",
                    problem["task_id"],
                )
            row.update(
                compute_uncertainty(
                    codes,
                    include_codebert=include_codebert,
                    codebert_model_name=codebert_model,
                    include_tree_edit=include_tree_edit,
                )
            )
        else:
            row.update(
                compute_uncertainty_prompt_aware(
                    codes,
                    prompt,
                    representation=representation,
                    include_codebert=include_codebert,
                    codebert_model_name=codebert_model,
                    include_tree_edit=include_tree_edit,
                )
            )
            row.update(prompt_confound_covariates(codes, prompt))
        rows.append(row)
    return rows


def _fmt_p(p: float | None) -> str:
    """Compact p-value phrase ("p < 0.001" / "p = 0.005") for labels."""
    if p is None or math.isnan(p):
        return "p n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def compute_correlations(rows: list[dict]) -> dict[str, dict]:
    """Correlate each metric with correctness at problem and sample level.

    Problem level: metric value vs the problem's initial pass rate
    (Pearson r, Spearman ρ, and Kendall's tau-b, over n_problems points).
    Kendall's tau-b (Week 5 addition) is reported alongside Spearman rather
    than replacing it: Arndt et al. (1999) bootstrap-compared the three at
    n=8-50 (this project's exact range) and found Pearson's r "often
    unstable" and Spearman's rho "a biased estimate" at this scale, while
    Kendall's tau-b "maintained adequate control of type I errors" and gave
    tighter confidence intervals — a third, better-behaved measure to check
    the other two against, not a replacement (|tau| <= |rho| always holds by
    construction, so a smaller tau alone is not a discrepancy). Sample
    level: point-biserial correlation between each sample's binary pass
    flag and its problem's metric value (metric values are per-problem, so
    samples from one problem share the same x — stated in the report).

    After every metric's raw p-values are computed, a final pass adds
    Benjamini-Hochberg FDR-adjusted p-values ("pearson_p_fdr",
    "spearman_p_fdr") for the Pearson and Spearman families separately —
    this project tests nine metrics simultaneously against the same outcome
    with no multiple-comparisons correction anywhere in the codebase prior
    to Week 5 (confirmed absent via grep across every prior report and this
    script). Benjamini & Hochberg's procedure is the right choice over
    Bonferroni given the strong positive dependency these metrics exhibit
    (inter-metric Spearman rho up to 0.99 — see the correlation heatmap).

    Args:
        rows: Output of extract_problem_table().
    Returns:
        Dict keyed by metric name with coefficient/p-value entries; metrics
        with zero variance are flagged {"constant": True} and get None
        coefficients instead of NaNs.
    """
    pass_rates = np.array([r["pass_rate"] for r in rows], dtype=float)
    sample_flags = np.array(
        [flag for r in rows for flag in r["passed_flags"]], dtype=float
    )
    correlations: dict[str, dict] = {}
    # Only metrics actually present in the rows (codebert is absent under
    # --no-codebert); all rows share one compute_uncertainty() key set.
    metrics_present = [m for m in METRIC_ORDER if m in rows[0]]
    for metric in metrics_present:
        values = np.array([r[metric] for r in rows], dtype=float)
        sample_values = np.array(
            [r[metric] for r in rows for _ in r["passed_flags"]], dtype=float
        )
        entry: dict = {
            "n_problems": int(len(values)),
            "n_samples": int(len(sample_values)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "constant": bool(np.ptp(values) == 0.0),
        }
        if entry["constant"] or np.ptp(pass_rates) == 0.0:
            entry.update(
                pearson_r=None,
                pearson_p=None,
                spearman_rho=None,
                spearman_p=None,
                kendall_tau=None,
                kendall_p=None,
            )
        else:
            pearson = scipy_stats.pearsonr(values, pass_rates)
            spearman = scipy_stats.spearmanr(values, pass_rates)
            kendall = scipy_stats.kendalltau(values, pass_rates)
            entry.update(
                pearson_r=float(pearson.statistic),
                pearson_p=float(pearson.pvalue),
                spearman_rho=float(spearman.statistic),
                spearman_p=float(spearman.pvalue),
                kendall_tau=float(kendall.statistic),
                kendall_p=float(kendall.pvalue),
            )
        if entry["constant"] or np.ptp(sample_flags) == 0.0:
            entry.update(pointbiserial_r=None, pointbiserial_p=None)
        else:
            biserial = scipy_stats.pointbiserialr(sample_flags, sample_values)
            entry.update(
                pointbiserial_r=float(biserial.statistic),
                pointbiserial_p=float(biserial.pvalue),
            )
        correlations[metric] = entry
    _apply_fdr_correction(correlations)
    return correlations


def _apply_fdr_correction(correlations: dict[str, dict]) -> None:
    """Add Benjamini-Hochberg FDR-adjusted p-values, in place, per metric family.

    Corrected separately for the Pearson and Spearman families (each is one
    set of simultaneously-tested hypotheses over the same nine metrics);
    Kendall's tau and point-biserial are left uncorrected since they are
    reported as cross-checks on the other two, not as an independent family
    of primary claims. Metrics with a None p-value (constant/zero-variance)
    keep None rather than being silently dropped from the correction set.

    Args:
        correlations: The dict compute_correlations() is about to return;
            mutated in place by adding "pearson_p_fdr"/"spearman_p_fdr".
    """
    for p_key, fdr_key in (
        ("pearson_p", "pearson_p_fdr"),
        ("spearman_p", "spearman_p_fdr"),
    ):
        metrics = [m for m, e in correlations.items() if e.get(p_key) is not None]
        if not metrics:
            continue
        raw_p = [correlations[m][p_key] for m in metrics]
        adjusted = scipy_stats.false_discovery_control(raw_p, method="bh")
        for m, p_adj in zip(metrics, adjusted):
            correlations[m][fdr_key] = float(p_adj)
        for m, e in correlations.items():
            e.setdefault(fdr_key, None)


# A partial correlation is undefined once a variable is (numerically) a pure
# function of its controls. Residuals whose standard deviation has shrunk by
# more than this factor relative to the original variable are treated as
# degenerate rather than correlated — see _partial_correlation().
_RESIDUAL_DEGENERACY_RTOL = 1e-10


def _residualize(values: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    """Residuals of `values` after least-squares regression on `controls`."""
    design = np.column_stack([np.ones(len(values))] + controls)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _partial_correlation(
    x: np.ndarray, y: np.ndarray, controls: list[np.ndarray], *, rank: bool = False
) -> tuple[float | None, float | None]:
    """Partial correlation of x and y controlling for `controls`.

    Both variables are residualized on the controls (with an intercept) and
    the residuals are correlated — the textbook construction. The p-value
    comes from the standard t statistic with df = n - 2 - k, k being the
    number of controls; this is the same test scipy runs for an ordinary
    Pearson correlation, with the degrees of freedom the controls consume
    subtracted.

    Args:
        x, y: Equal-length observation vectors.
        controls: Control vectors, same length.
        rank: If True, rank-transform everything first, giving the
            Spearman-family partial correlation instead of the Pearson one.
    Returns:
        (coefficient, p_value); (None, None) when a residual vector is
        constant (nothing left to correlate) and (r, None) when the degrees
        of freedom run out.
    """
    if rank:
        x = scipy_stats.rankdata(x)
        y = scipy_stats.rankdata(y)
        controls = [scipy_stats.rankdata(c) for c in controls]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res_x = _residualize(x, controls)
    res_y = _residualize(y, controls)
    # A variable fully explained by the controls leaves floating-point noise
    # (std ~1e-15), not an exact zero, and correlating with that noise
    # produces a large, entirely meaningless coefficient. Compare each
    # residual's scale against its own variable's scale rather than against
    # an absolute epsilon, so the check is unit-independent.
    for residual, original in ((res_x, x), (res_y, y)):
        if np.std(residual) <= _RESIDUAL_DEGENERACY_RTOL * max(
            np.std(original), 1e-300
        ):
            return None, None
    r = float(np.corrcoef(res_x, res_y)[0, 1])
    df = len(res_x) - 2 - len(controls)
    if df <= 0:
        return r, None
    t = r * math.sqrt(df / max(1.0 - r * r, 1e-12))
    return r, float(2.0 * scipy_stats.t.sf(abs(t), df))


def compute_confound_analysis(rows: list[dict]) -> dict:
    """Week 7: the length baseline and the partial-correlation columns.

    Answers the two questions the raw correlation table cannot, both found
    by re-analysing this project's own stored runs (see src/uncertainty.py's
    prompt-inclusion section for the measured numbers):

      1. *Does the apparatus beat a free baseline?* Mean completion length
         needs no sampling, parsing, embedding or execution, and recovers
         71-96% of ast_node_diversity's correlation magnitude in every one
         of this project's six regime cells. Any claim that the diversity
         metrics earn their cost must be made against this row, and the
         claimable quantity is the partial correlation that survives it
         (-0.30 to -0.50 for AST/TED), not the raw one.
      2. *How much of a metric's correlation is prompt share?* Controlling
         for prompt share collapses token_jaccard_diversity from -0.219 to
         -0.032 at n=164 while barely moving AST or lexical uniqueness.

    Length is reported both ways on purpose: it is not a pure nuisance
    variable — an uncertain model plausibly rambles, producing longer and
    worse output — so over-controlling can hide real signal. The raw
    coefficient, the partials, and the ambiguity all belong in the report.

    Args:
        rows: Output of extract_problem_table(). Rows must carry the Week 7
            covariates, i.e. the run's prompts had to be recoverable.
    Returns:
        {"available": False, ...} when covariates are missing; otherwise
        per-covariate baselines and, per metric, the raw and partial
        coefficients plus the metric's own correlation with prompt share.
    """
    covariates = [c for c in COVARIATE_ORDER if c in rows[0]]
    if CONFOUND_BASELINE not in covariates:
        return {
            "available": False,
            "reason": "no prompt-derived covariates in rows (prompts unavailable)",
        }
    pass_rates = np.array([r["pass_rate"] for r in rows], dtype=float)
    values = {c: np.array([r[c] for r in rows], dtype=float) for c in covariates}
    # A covariate that is constant or undefined across the run (e.g. an AST
    # share that could not be computed) cannot serve as a control.
    usable = {
        c: v for c, v in values.items() if not np.isnan(v).any() and np.ptp(v) > 0.0
    }

    baselines = {}
    for name, vector in values.items():
        if name not in usable:
            baselines[name] = {"constant_or_undefined": True}
            continue
        pearson = scipy_stats.pearsonr(vector, pass_rates)
        spearman = scipy_stats.spearmanr(vector, pass_rates)
        baselines[name] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "mean": float(vector.mean()),
        }

    metrics_present = [m for m in METRIC_ORDER if m in rows[0]]
    metric_entries: dict[str, dict] = {}
    for metric in metrics_present:
        metric_values = np.array([r[metric] for r in rows], dtype=float)
        if np.ptp(metric_values) == 0.0:
            metric_entries[metric] = {"constant": True}
            continue
        entry: dict = {
            "raw_pearson_r": float(scipy_stats.pearsonr(metric_values, pass_rates)[0]),
            "raw_spearman_rho": float(
                scipy_stats.spearmanr(metric_values, pass_rates).statistic
            ),
        }
        # The measurement-artefact diagnostic: how strongly does the metric
        # track the share of the compared string that is identical prompt?
        if "prompt_share_char" in usable:
            entry["r_with_prompt_share"] = float(
                scipy_stats.pearsonr(metric_values, usable["prompt_share_char"])[0]
            )
        for label, control_names in PARTIAL_CONTROL_SETS.items():
            controls = [usable[c] for c in control_names if c in usable]
            if len(controls) != len(control_names):
                entry[label] = {"unavailable": True}
                continue
            r, p = _partial_correlation(metric_values, pass_rates, controls)
            rho, rho_p = _partial_correlation(
                metric_values, pass_rates, controls, rank=True
            )
            entry[label] = {
                "controls": control_names,
                "pearson_r": r,
                "pearson_p": p,
                "spearman_rho": rho,
                "spearman_p": rho_p,
            }
        metric_entries[metric] = entry

    return {
        "available": True,
        "n_problems": len(rows),
        "baseline_metric": CONFOUND_BASELINE,
        "covariate_baselines": baselines,
        "metrics": metric_entries,
    }


def print_confound_table(analysis: dict) -> None:
    """Print the Week 7 baseline + partial-correlation table to stdout."""
    if not analysis.get("available"):
        print(f"\n[confound analysis unavailable: {analysis.get('reason')}]")
        return

    print("\nCovariate baselines vs pass rate (no diversity metric involved):")
    print(f"{'covariate':<26}{'pearson r (p)':<24}{'spearman ρ (p)':<24}{'mean':<12}")
    for name in COVARIATE_ORDER:
        entry = analysis["covariate_baselines"].get(name)
        if not entry or entry.get("constant_or_undefined"):
            continue
        pearson_cell = "{:+.3f} ({})".format(
            entry["pearson_r"], _fmt_p(entry["pearson_p"])
        )
        spearman_cell = "{:+.3f} ({})".format(
            entry["spearman_rho"], _fmt_p(entry["spearman_p"])
        )
        print(f"{name:<26}{pearson_cell:<24}{spearman_cell:<24}{entry['mean']:<12.2f}")

    print(
        "\nMetric correlations with pass rate, raw vs controlled "
        "(Pearson r; the claimable quantity is the partial, not the raw):"
    )
    header = (
        f"{'metric':<28}{'raw r':<10}{'ctrl compl.len':<16}{'ctrl both len':<16}"
        f"{'ctrl prompt share':<20}{'r(metric, share)':<18}"
    )
    print(header)
    for metric in (m for m in METRIC_ORDER if m in analysis["metrics"]):
        entry = analysis["metrics"][metric]
        if entry.get("constant"):
            print(f"{metric:<28}constant")
            continue

        def _cell(label: str) -> str:
            sub = entry.get(label, {})
            if sub.get("unavailable") or sub.get("pearson_r") is None:
                return "n/a"
            return f"{sub['pearson_r']:+.3f}"

        share = entry.get("r_with_prompt_share")
        share_cell = f"{share:+.3f}" if share is not None else "n/a"
        print(
            f"{metric:<28}"
            f"{entry['raw_pearson_r']:<+10.3f}"
            f"{_cell('ctrl_completion_len'):<16}"
            f"{_cell('ctrl_both_lengths'):<16}"
            f"{_cell('ctrl_prompt_share'):<20}"
            f"{share_cell:<18}"
        )


def _reserve_footnote_band(fig: plt.Figure) -> None:
    """Keep constrained_layout off the bottom strip where the footnote sits."""
    fig.get_layout_engine().set(rect=(0.0, 0.05, 1.0, 0.95))


def _style_axes(ax: plt.Axes) -> None:
    """Apply the shared recessive-chrome style: hairline grid, muted ticks."""
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_MUTED, labelsize=8.5, length=3)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _variable_metrics(correlations: dict[str, dict]) -> list[str]:
    """Analyzed metrics with any variance across problems, in canonical order."""
    return [
        m for m in METRIC_ORDER if m in correlations and not correlations[m]["constant"]
    ]


def plot_metric_vs_passrate(
    rows: list[dict],
    correlations: dict[str, dict],
    out_path: Path,
    run_label: str,
) -> None:
    """Scatter each variable metric against per-problem pass rate.

    Args:
        rows: Output of extract_problem_table().
        correlations: Output of compute_correlations().
        out_path: PNG destination.
        run_label: Dataset/model/config line for the figure subtitle.
    """
    metrics = _variable_metrics(correlations)
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9.4, 3.6 * nrows + 0.7), constrained_layout=True
    )
    fig.set_facecolor(_SURFACE)
    _reserve_footnote_band(fig)
    axes = np.atleast_1d(axes).ravel()

    pass_rates = np.array([r["pass_rate"] for r in rows], dtype=float)
    for ax, metric in zip(axes, metrics):
        values = np.array([r[metric] for r in rows], dtype=float)
        _style_axes(ax)
        # Trend line under the dots; dots carry a 2px surface ring so
        # overlapping problems stay distinguishable.
        slope, intercept = np.polyfit(values, pass_rates, deg=1)
        xs = np.linspace(values.min(), values.max(), 2)
        ax.plot(xs, slope * xs + intercept, color=_TREND, linewidth=2, alpha=0.8)
        ax.scatter(
            values,
            pass_rates,
            s=52,
            color=_SERIES,
            edgecolors=_SURFACE,
            linewidths=1.5,  # ~2px surface ring at typical report embed scale
            zorder=3,
        )
        stats = correlations[metric]
        annotation = (
            f"Spearman ρ = {stats['spearman_rho']:+.2f} ({_fmt_p(stats['spearman_p'])})\n"
            f"Pearson r = {stats['pearson_r']:+.2f} ({_fmt_p(stats['pearson_p'])})"
        )
        # Put the annotation in the emptier corner: negative trends leave
        # the top-right free, positive trends the top-left.
        if slope <= 0:
            xy, ha = (0.97, 0.96), "right"
        else:
            xy, ha = (0.03, 0.96), "left"
        ax.text(
            *xy,
            annotation,
            transform=ax.transAxes,
            ha=ha,
            va="top",
            fontsize=8.5,
            color=_INK_SECONDARY,
        )
        ax.set_title(METRIC_LABELS[metric], loc="left", fontsize=10.5, color=_INK)
        ax.set_ylim(-0.06, 1.06)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    for ax in axes[: len(metrics)]:
        ax.set_xlabel("metric value", fontsize=9, color=_INK_SECONDARY)
    for ax in axes[::ncols]:
        ax.set_ylabel("initial pass rate", fontsize=9, color=_INK_SECONDARY)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Uncertainty metrics vs correctness — one dot per problem",
        fontsize=12.5,
        color=_INK,
        ha="left",
        x=0.012,
    )
    fig.text(0.012, 0.005, run_label, fontsize=8, color=_MUTED, ha="left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_metric_by_outcome(
    rows: list[dict],
    correlations: dict[str, dict],
    out_path: Path,
    run_label: str,
    seed: int,
) -> None:
    """Box + jittered strip of each metric split by per-sample pass/fail.

    Each dot is one generated sample, carrying its problem's metric value;
    pass/fail wear the reserved status colors and are named on the axis so
    color never carries the grouping alone.

    Args:
        rows: Output of extract_problem_table().
        correlations: Output of compute_correlations().
        out_path: PNG destination.
        run_label: Dataset/model/config line for the figure subtitle.
        seed: RNG seed for the horizontal jitter (reproducibility).
    """
    rng = np.random.default_rng(seed)
    metrics = _variable_metrics(correlations)
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9.4, 3.6 * nrows + 0.7), constrained_layout=True
    )
    fig.set_facecolor(_SURFACE)
    _reserve_footnote_band(fig)
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, metrics):
        _style_axes(ax)
        groups = {True: [], False: []}
        for row in rows:
            for flag in row["passed_flags"]:
                groups[flag].append(row[metric])
        for position, (flag, color) in enumerate(((True, _PASS), (False, _FAIL))):
            values = np.array(groups[flag], dtype=float)
            if len(values) == 0:
                continue
            box = ax.boxplot(
                [values],
                positions=[position],
                widths=0.42,
                showfliers=False,
                patch_artist=True,
            )
            box["boxes"][0].set(facecolor=color, alpha=0.10, edgecolor=color)
            box["boxes"][0].set_linewidth(1.4)
            for part in ("whiskers", "caps"):
                for artist in box[part]:
                    artist.set(color=color, linewidth=1.2)
            box["medians"][0].set(color=_INK, linewidth=1.1)
            jitter = rng.uniform(-0.13, 0.13, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=14,
                color=color,
                alpha=0.5,
                edgecolors=_SURFACE,
                linewidths=0.6,
                zorder=3,
            )
        n_pass, n_fail = len(groups[True]), len(groups[False])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"pass (n={n_pass})", f"fail (n={n_fail})"], color=_INK)
        ax.set_xlim(-0.55, 1.55)
        stats = correlations[metric]
        if stats["pointbiserial_r"] is not None:
            # Inside the axes, not on the title line: a long METRIC_LABELS
            # entry (e.g. "CodeBERT BERTScore diversity (per-token)") can run
            # the full loc="left" title into a same-row annotation — an
            # opaque surface-colored backing patch is what makes an in-plot
            # position collision-safe against a strip plot's own data, which
            # can otherwise reach any corner.
            ax.text(
                0.98,
                0.97,
                f"$r_{{pb}}$ = {stats['pointbiserial_r']:+.2f} "
                f"({_fmt_p(stats['pointbiserial_p'])})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color=_INK_SECONDARY,
                bbox=dict(facecolor=_SURFACE, alpha=0.88, edgecolor="none", pad=2.0),
                zorder=4,
            )
        ax.set_title(METRIC_LABELS[metric], loc="left", fontsize=10.5, color=_INK)
    for ax in axes[::ncols]:
        ax.set_ylabel("metric value", fontsize=9, color=_INK_SECONDARY)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Uncertainty metrics by sample outcome — one dot per generated sample "
        "($r_{pb}$ = point-biserial correlation)",
        fontsize=12.5,
        color=_INK,
        ha="left",
        x=0.012,
    )
    fig.text(0.012, 0.005, run_label, fontsize=8, color=_MUTED, ha="left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_correlation_summary(
    correlations: dict[str, dict], out_path: Path, run_label: str
) -> None:
    """Signed horizontal bars of problem-level Spearman ρ per metric.

    Bars encode polarity around a zero baseline with the diverging pole
    pair (negative = blue, positive = red); values sit at the bar tips.

    Args:
        correlations: Output of compute_correlations().
        out_path: PNG destination.
        run_label: Dataset/model/config line for the figure subtitle.
    """
    entries = [
        (metric, correlations[metric])
        for metric in METRIC_ORDER
        if metric in correlations
        and correlations[metric].get("spearman_rho") is not None
    ]
    entries.sort(key=lambda item: abs(item[1]["spearman_rho"]), reverse=True)
    skipped = [
        METRIC_LABELS[m]
        for m in METRIC_ORDER
        if m in correlations and correlations[m].get("spearman_rho") is None
    ]

    fig, ax = plt.subplots(
        figsize=(8.6, 0.62 * len(entries) + 1.9), constrained_layout=True
    )
    fig.set_facecolor(_SURFACE)
    _reserve_footnote_band(fig)
    _style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    positions = np.arange(len(entries))[::-1]  # strongest at the top
    values = [entry["spearman_rho"] for _, entry in entries]
    colors = [_NEG_POLE if v < 0 else _POS_POLE for v in values]
    # Thin marks: cap the bar well under the row band and let the leftover
    # read as air — a 0.55 band-fraction renders ~74px at dpi=200, three
    # times the ≤24px mark-spec cap, and reads as loud saturated blocks.
    ax.barh(positions, values, height=0.28, color=colors)
    ax.axvline(0.0, color=_MUTED, linewidth=1.0, zorder=3)

    limit = max(0.5, max(abs(v) for v in values) * 1.2)
    ax.set_xlim(-limit, limit)
    ax.set_yticks(positions)
    ax.set_yticklabels([METRIC_LABELS[m] for m, _ in entries], fontsize=9.5, color=_INK)
    ax.set_xlabel(
        "Spearman ρ vs per-problem pass rate", fontsize=9, color=_INK_SECONDARY
    )
    # Each row has one bar extending one way from zero, so the opposite
    # side of the baseline is guaranteed empty — anchor the value label
    # there, where it can collide with neither the bar nor the y labels.
    for pos, value, (_, entry) in zip(positions, values, entries):
        offset = 0.025 * limit
        ax.text(
            offset if value < 0 else -offset,
            pos,
            f"{value:+.2f} ({_fmt_p(entry['spearman_p'])})",
            va="center",
            ha="left" if value < 0 else "right",
            fontsize=8.5,
            color=_INK_SECONDARY,
        )

    subtitle = "negative = more diverse samples ↔ lower pass rate (self-consistency hypothesis)"
    if skipped:
        subtitle += f"\nconstant this run, excluded: {', '.join(skipped)}"
    fig.suptitle(
        "Which uncertainty metric tracks correctness so far?",
        fontsize=12.5,
        color=_INK,
        ha="left",
        x=0.012,
    )
    ax.set_title(subtitle, loc="left", fontsize=8.5, color=_INK_SECONDARY, pad=10)
    fig.text(0.012, 0.005, run_label, fontsize=8, color=_MUTED, ha="left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def compute_inter_metric_matrix(
    rows: list[dict], correlations: dict[str, dict]
) -> tuple[list[str], np.ndarray]:
    """Pairwise Spearman rho between every pair of variable metrics.

    This is the redundancy view the report's "which metrics are actually
    independent information" observations describe in prose (e.g.
    ast_node_diversity vs tree_edit_distance_diversity rho=+0.99,
    codebert_bertscore_diversity vs the structural family rho~0.87-0.88) —
    computed here from the same per-problem metric table those numbers came
    from, so plot_metric_correlation_heatmap() and the prose can never drift
    apart silently.

    Args:
        rows: Output of extract_problem_table().
        correlations: Output of compute_correlations(); only used to select
            the variable (non-constant), canonically-ordered metric set —
            a constant metric has no correlation with anything.
    Returns:
        (metric_names, matrix) where matrix[i, j] is Spearman rho between
        metric_names[i] and metric_names[j] over the n_problems per-problem
        values; matrix[i, i] == 1.0 by construction (self-correlation).
    """
    metrics = _variable_metrics(correlations)
    values = {m: np.array([r[m] for r in rows], dtype=float) for m in metrics}
    n = len(metrics)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = float(
                scipy_stats.spearmanr(values[metrics[i]], values[metrics[j]]).statistic
            )
            matrix[i, j] = matrix[j, i] = rho
    return metrics, matrix


def plot_metric_correlation_heatmap(
    rows: list[dict], correlations: dict[str, dict], out_path: Path, run_label: str
) -> None:
    """Pairwise Spearman rho between every metric — the redundancy view.

    Backs the report's redundancy observations with a figure instead of
    leaving them as prose-only numbers pulled from the stats JSON by hand.
    Sequential single-hue ramp (a magnitude job, not polarity — every
    observed pair here is positively correlated, so a diverging pair with an
    unused negative half would be the wrong tool). Every cell is direct-
    labeled: an 8-9 metric grid has no overplotting to avoid the way a
    scatter does, and a correlation matrix's cells already are the "table
    view" a chart normally needs alongside it.

    Args:
        rows: Output of extract_problem_table().
        correlations: Output of compute_correlations().
        out_path: PNG destination.
        run_label: Dataset/model/config line for the figure subtitle.
    """
    metrics, matrix = compute_inter_metric_matrix(rows, correlations)
    n = len(metrics)
    cmap = LinearSegmentedColormap.from_list("sequential_blue", _SEQUENTIAL_RAMP)

    fig, ax = plt.subplots(
        figsize=(1.02 * n + 2.1, 1.02 * n + 1.3), constrained_layout=True
    )
    fig.set_facecolor(_SURFACE)
    _reserve_footnote_band(fig)

    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
    labels = [METRIC_LABELS[m] for m in metrics]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5, color=_INK)
    ax.set_yticklabels(labels, fontsize=8.5, color=_INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    # Thin surface-colored gridlines between cells (same 2px-surface-gap
    # convention as adjacent bars/fills elsewhere in this file).
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=_SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            # Light cells (low redundancy) read best with dark ink, dark
            # cells with light ink — mirrors the ramp's own light->dark walk.
            text_color = _INK if value < 0.55 else _SURFACE
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.6,
                color=text_color,
                fontweight="bold" if i == j else "normal",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Spearman ρ between metrics", fontsize=9, color=_INK_SECONDARY)
    cbar.ax.tick_params(colors=_MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "How redundant are these metrics with each other?",
        fontsize=12.5,
        color=_INK,
        ha="left",
        x=0.012,
    )
    ax.set_title(
        "higher = more redundant · diagonal = 1.0 by construction",
        loc="left",
        fontsize=8.5,
        color=_INK_SECONDARY,
        pad=8,
    )
    fig.text(0.012, 0.005, run_label, fontsize=8, color=_MUTED, ha="left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def print_correlation_table(correlations: dict[str, dict]) -> None:
    """Print the correlation summary table to stdout (the table-view twin).

    Week 5 addition: a Kendall's tau-b column (a third, better-behaved
    measure at this project's n — see compute_correlations()'s docstring)
    and a Spearman FDR-adjusted-p column (Benjamini-Hochberg, corrected
    jointly across all metrics in this table — see _apply_fdr_correction()).
    """
    header = (
        f"\n{'metric':<28}{'pearson r (p)':<22}{'spearman ρ (p)':<22}"
        f"{'kendall τ (p)':<22}{'pt-biserial r (p)':<22}"
        f"{'spearman FDR-p':<16}{'range':<18}"
    )
    print(header)
    for metric in (m for m in METRIC_ORDER if m in correlations):
        entry = correlations[metric]

        def _cell(r_key: str, p_key: str) -> str:
            r = entry[r_key]
            if r is None:
                return "constant"
            return f"{r:+.2f} ({_fmt_p(entry[p_key])})"

        fdr_p = entry.get("spearman_p_fdr")
        fdr_cell = _fmt_p(fdr_p) if fdr_p is not None else "constant"
        value_range = f"[{entry['min']:.2f}, {entry['max']:.2f}]"
        print(
            f"{metric:<28}"
            f"{_cell('pearson_r', 'pearson_p'):<22}"
            f"{_cell('spearman_rho', 'spearman_p'):<22}"
            f"{_cell('kendall_tau', 'kendall_p'):<22}"
            f"{_cell('pointbiserial_r', 'pointbiserial_p'):<22}"
            f"{fdr_cell:<16}"
            f"{value_range:<18}"
        )


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    results_path = args.results if args.results else find_latest_results_file()
    logger.info("Analyzing %s", results_path)
    with results_path.open() as f:
        results = json.load(f)

    prompts = load_run_prompts(results)
    rows = extract_problem_table(
        results,
        include_codebert=not args.no_codebert,
        codebert_model=args.codebert_model,
        representation=args.representation,
        prompts=prompts,
        include_tree_edit=not args.no_tree_edit,
    )
    if len(rows) < 3:
        raise ValueError(
            f"Only {len(rows)} problems with samples in {results_path}; "
            "need at least 3 for a meaningful correlation."
        )
    correlations = compute_correlations(rows)
    confound_analysis = compute_confound_analysis(rows)

    config = results.get("config", {})
    run_label = (
        f"{config.get('dataset', '?')} · {config.get('model', '?')} · "
        f"{len(rows)} problems × {config.get('n_samples', '?')} samples · "
        f"seed {config.get('seed', '?')} · source: {results.get('run_id', results_path.name)}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "metric_vs_passrate": args.out_dir
        / f"{args.fig_prefix}_metric_vs_passrate.png",
        "metric_by_outcome": args.out_dir / f"{args.fig_prefix}_metric_by_outcome.png",
        "correlation_summary": args.out_dir
        / f"{args.fig_prefix}_correlation_summary.png",
        "correlation_heatmap": args.out_dir
        / f"{args.fig_prefix}_correlation_heatmap.png",
    }
    plot_metric_vs_passrate(
        rows, correlations, figure_paths["metric_vs_passrate"], run_label
    )
    plot_metric_by_outcome(
        rows, correlations, figure_paths["metric_by_outcome"], run_label, args.seed
    )
    plot_correlation_summary(
        correlations, figure_paths["correlation_summary"], run_label
    )
    plot_metric_correlation_heatmap(
        rows, correlations, figure_paths["correlation_heatmap"], run_label
    )

    print_correlation_table(correlations)
    print_confound_table(confound_analysis)

    inter_metric_names, inter_metric_matrix = compute_inter_metric_matrix(
        rows, correlations
    )
    stats = {
        "analysis": f"{args.fig_prefix}_uncertainty_correlation",
        "source_results_file": str(results_path),
        "source_run_id": results.get("run_id"),
        "config": {
            "seed": args.seed,
            "out_dir": str(args.out_dir),
            "fig_prefix": args.fig_prefix,
            "codebert_model": None if args.no_codebert else args.codebert_model,
            "representation": args.representation,
            "include_tree_edit": not args.no_tree_edit,
        },
        "source_config": config,
        "n_problems": len(rows),
        "n_samples_total": int(sum(len(r["passed_flags"]) for r in rows)),
        "n_samples_pass": int(sum(sum(r["passed_flags"]) for r in rows)),
        "correlations": correlations,
        "confound_analysis": confound_analysis,
        "inter_metric_correlations": {
            "metrics": inter_metric_names,
            "spearman_rho": inter_metric_matrix.tolist(),
        },
        "figures": {name: str(path) for name, path in figure_paths.items()},
        "per_problem_metrics": [
            {k: v for k, v in row.items() if k != "passed_flags"} for row in rows
        ],
    }
    if args.dry_run:
        logger.info("[DRY RUN] skipping stats JSON persistence")
    else:
        out_path = save_results(
            stats,
            prefix=f"analysis_{args.fig_prefix}_{config.get('dataset', 'unknown')}",
        )
        logger.info("Saved correlation stats to %s", out_path)


if __name__ == "__main__":
    main()
