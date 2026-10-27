"""Week 11 field audit: do published metric definitions change conclusion?

Runs the definitions re-implemented in ``src/published_metrics.py`` through
the measurement controls this project already built, over the seven stored
cells frozen in ``reports/week11_field_audit_preregistration.md``. No model
is loaded and nothing is generated: every number is recomputed from samples
already on disk.

What it answers, and why it is not the same question as every previous week:
Weeks 2-7 audited *this project's* metrics and found five ways the
measurement moves. That is a self audit. This asks whether the same controls
move *published* definitions -- which is the difference between "our
coefficients were fragile" and "this subfield's coefficients are fragile".

Usage::

    python scripts/audit_published_metrics.py --jobs 8
    python scripts/audit_published_metrics.py --cells cg_he60 ds_he60 --no-tsed

Output: ``results/analysis_week11_field_audit_<timestamp>.json`` plus a
printed table. The pre-registration's criteria (FDR significance flip, or
|delta rho| > 0.12) are applied by ``scripts/score_field_audit.py``, not
here -- this script only measures.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import published_metrics as pm  # noqa: E402
from src.uncertainty import (  # noqa: E402
    prepare_metric_inputs,
    prompt_confound_covariates,
)
from src.utils import get_logger, save_results  # noqa: E402

# NOTE (macOS, multiprocessing): `scripts.analyze_uncertainty` is imported
# lazily inside the functions that need it, never at module level. Workers
# are spawned (the default start method on macOS), so each one re-executes
# this file as `__mp_main__` -- and that module pulls in matplotlib, whose
# import can deadlock in a spawned child before any GUI backend selection
# happens. The symptom is silent: the parent sits at 0% CPU with no live
# children and no traceback. Only the parent process needs these helpers,
# so keeping them out of module scope removes the hazard entirely rather
# than papering over it with a backend environment variable.

logger = get_logger(__name__)

#: The seven cells frozen in the pre-registration (§3). Names are the keys
#: used throughout the report; the seventh is scale-matched to the audited
#: paper's own 7-8B models.
CELLS: dict[str, str] = {
    "cg_he60": "humaneval_uncertainty_20260710_014920.json",
    "cg_he164": "humaneval_uncertainty_merged164_20260711.json",
    "cg_he_t03": "humaneval_t03_60_merged_20260712.json",
    "ds_he60": "humaneval_deepseek60_merged_20260712.json",
    "cg_mbppplus": "mbpp_uncertainty_20260712_024909.json",
    "ds_mbppplus": "mbpp_uncertainty_20260712_085004.json",
    "qwen7b_he60": "humaneval_uncertainty_20260901_004849.json",
}

#: Parameter grids, frozen in pre-registration §5. Nothing here may be
#: widened after seeing a result; anything added later is labelled post-hoc.
TSED_WEIGHT_GRID = {
    "paper": pm.TSED_WEIGHTS_PAPER,
    "unit": pm.TSED_WEIGHTS_UNIT,
}
SENT_DEPTH_GRID = (1, 2, 3, 4)
SENT_EPSILON_GRID = (1e-3, 1e-6, 1e-9)
SENT_ENCODING_GRID = ("struct", "value")

REPRESENTATIONS = ("full", "corrected")


def parse_args() -> argparse.Namespace:
    """CLI arguments (CLAUDE.md Section 12: argparse in scripts, never in src)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(CELLS),
        choices=list(CELLS),
        help="Which pre-registered cells to run (default: all seven).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory holding the stored run JSONs.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes. TSED is super-quadratic and pure Python, so "
        "this is the difference between minutes and hours.",
    )
    parser.add_argument(
        "--no-tsed",
        action="store_true",
        help="Skip the TSED definition (by far the most expensive one).",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=list(REPRESENTATIONS),
        choices=list(REPRESENTATIONS),
        help="Control C1: which metric-input representations to measure.",
    )
    return parser.parse_args()


def _sent_configs() -> list[tuple[str, int, str, float]]:
    """(name, depth, encoding, epsilon) for the frozen structural-entropy grid."""
    configs = []
    for depth in SENT_DEPTH_GRID:
        for encoding in SENT_ENCODING_GRID:
            for epsilon in SENT_EPSILON_GRID:
                exponent = int(round(-np.log10(epsilon)))
                configs.append(
                    (f"d{depth}_{encoding}_eps{exponent}", depth, encoding, epsilon)
                )
    return configs


def _resolve_representation(representation: str) -> str:
    """Map the audit's C1 setting onto a concrete metric-input representation.

    ``uncertainty.prepare_metric_inputs`` deliberately refuses "corrected":
    in this project that is a *per-family* choice resolved one level up, with
    the token/lexical/embedding families reading "completion" (prompt removed
    outright) and the structural families reading "signature" (the prompt's
    docstrings and comments stripped, the signature kept so the code still
    parses).

    Every definition audited here is AST-based -- TSED parses trees, and both
    structural-entropy encodings walk an AST -- so none of them can read a
    bare completion: a function body with no ``def`` line does not parse, and
    the metric would collapse to its unparseable constant for every sample.
    "corrected" therefore resolves to "signature" for all of them.

    This is worth stating in the report rather than treating as plumbing: the
    strongest available correction, removing the prompt entirely, is
    unavailable to any metric that must parse. A metric family's own design
    constrains which confound correction it can accept, and the two published
    AST metrics inherit that constraint from their definitions, not from us.
    Note the consequence for the sigma_value encoding in particular: it reads
    lexemes, so it dilutes like the token family, but it must parse, so it can
    only receive the structural family's partial correction.
    """
    return "signature" if representation == "corrected" else representation


def problem_metrics(payload: tuple) -> dict[str, float]:
    """Every frozen metric configuration for one problem's N samples.

    Runs in a worker process. Structural-entropy distributions are built
    once per (depth, encoding) and shared across the three epsilon values
    and both metrics -- the smoothing is applied downstream of extraction,
    so recomputing per epsilon would be pure waste.

    Args:
        payload: (metric_inputs, include_tsed) for one problem.
    Returns:
        config name -> metric value.
    """
    codes, include_tsed = payload
    values: dict[str, float] = {}

    if include_tsed:
        for label, weights in TSED_WEIGHT_GRID.items():
            values[f"tsed_{label}"] = pm.tsed_diversity(codes, weights=weights)

    n = len(codes)
    for depth in SENT_DEPTH_GRID:
        for encoding in SENT_ENCODING_GRID:
            distributions = [
                pm.subtree_distribution(code, depth=depth, encoding=encoding)
                for code in codes
            ]
            n_bad = sum(1 for dist in distributions if dist is None)
            if n < 2:
                for epsilon in SENT_EPSILON_GRID:
                    exponent = int(round(-np.log10(epsilon)))
                    suffix = f"d{depth}_{encoding}_eps{exponent}"
                    values[f"sent_jsd_{suffix}"] = 0.0
                    values[f"sent_sce_{suffix}"] = 0.0
                continue
            # Unparseable members are maximally divergent (declared
            # convention; the paper specifies none).
            base = float(n_bad * (n - n_bad) + n_bad * (n_bad - 1) // 2)
            n_pairs = n * (n - 1) // 2
            valid = [dist for dist in distributions if dist is not None]
            supports = []
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    supports.append(
                        (
                            valid[i],
                            valid[j],
                            sorted(set(valid[i]) | set(valid[j]), key=repr),
                        )
                    )
            for epsilon in SENT_EPSILON_GRID:
                exponent = int(round(-np.log10(epsilon)))
                suffix = f"d{depth}_{encoding}_eps{exponent}"
                jsd_total = base
                sce_total = base
                for dist_a, dist_b, support in supports:
                    p = pm._smoothed(dist_a, support, epsilon)
                    q = pm._smoothed(dist_b, support, epsilon)
                    jsd_total += pm._jensen_shannon(p, q)
                    sce_total += 1.0 - 0.5 * (pm._sce(p, q) + pm._sce(q, p))
                values[f"sent_jsd_{suffix}"] = jsd_total / n_pairs
                values[f"sent_sce_{suffix}"] = sce_total / n_pairs
    return values


def build_cell_table(
    results: dict,
    prompts: dict[str, str],
    representation: str,
    *,
    include_tsed: bool,
    jobs: int,
) -> list[dict]:
    """Per-problem rows: pass rate, length covariates, every metric config.

    Args:
        results: Stored run JSON (CLAUDE.md Section 8 schema).
        prompts: task_id -> prompt, from load_run_prompts().
        representation: Control C1 -- "full" reproduces Weeks 2-6 exactly.
        include_tsed: Whether to compute the expensive TSED family.
        jobs: Worker processes.
    Returns:
        One dict per problem with samples.
    """
    problems = [p for p in results["per_problem"] if p.get("samples")]
    payloads = []
    meta = []
    for problem in problems:
        full_codes = [s["code"] for s in problem["samples"]]
        prompt = prompts.get(problem["task_id"], "")
        if prompt and representation != "full":
            codes = prepare_metric_inputs(
                full_codes,
                prompt,
                representation=_resolve_representation(representation),
            )
        else:
            codes = list(full_codes)
        covariates = (
            prompt_confound_covariates(full_codes, prompt)
            if prompt
            else {"mean_completion_chars": float(np.mean([len(c) for c in full_codes]))}
        )
        payloads.append((codes, include_tsed))
        meta.append(
            {
                "task_id": problem["task_id"],
                "pass_rate": float(problem["initial_pass_rate"]),
                "mean_completion_chars": covariates.get("mean_completion_chars"),
                "prompt_chars": covariates.get("prompt_chars"),
            }
        )

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            computed = list(pool.map(problem_metrics, payloads, chunksize=1))
    else:
        computed = [problem_metrics(payload) for payload in payloads]

    return [{**row, **metrics} for row, metrics in zip(meta, computed)]


def correlate(rows: list[dict], metric: str) -> dict:
    """Spearman and Pearson against pass rate, plus the length partial (C2).

    Bootstrap CIs at 2,000 resamples, as the pre-registration fixes -- the
    project's earlier weeks reported point estimates only, and a rank
    comparison without an interval is exactly what finding #5 says not to
    trust.
    """
    x = np.array([row[metric] for row in rows], dtype=float)
    y = np.array([row["pass_rate"] for row in rows], dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"spearman_rho": None, "spearman_p": None, "constant": True}
    spearman = scipy_stats.spearmanr(x, y)
    pearson = scipy_stats.pearsonr(x, y)

    from scripts.analyze_uncertainty import _partial_correlation

    lengths = np.array(
        [row.get("mean_completion_chars") or 0.0 for row in rows], dtype=float
    )
    controls = [lengths] if np.std(lengths) > 0 else []
    partial_r, partial_p = (
        _partial_correlation(x, y, controls, rank=True) if controls else (None, None)
    )

    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(x), len(x))
        if np.std(x[idx]) == 0.0 or np.std(y[idx]) == 0.0:
            continue
        boot.append(scipy_stats.spearmanr(x[idx], y[idx]).statistic)
    ci = (
        [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        if boot
        else None
    )
    return {
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "partial_rho_length": partial_r,
        "partial_p_length": partial_p,
        "spearman_ci95": ci,
        "n": int(len(x)),
    }


def main() -> None:
    """Measure every frozen configuration on every requested cell."""
    args = parse_args()
    from scripts.analyze_uncertainty import _apply_fdr_correction, load_run_prompts

    sent_names = [
        f"{metric}_{name}"
        for metric in ("sent_jsd", "sent_sce")
        for name, *_ in _sent_configs()
    ]
    metric_names = sent_names + (
        [] if args.no_tsed else [f"tsed_{label}" for label in TSED_WEIGHT_GRID]
    )

    analysis: dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "preregistration": "reports/week11_field_audit_preregistration.md",
        "definitions": "reports/week11_published_metric_definitions.md",
        "grids": {
            "tsed_weights": {k: list(v) for k, v in TSED_WEIGHT_GRID.items()},
            "sent_depth": list(SENT_DEPTH_GRID),
            "sent_epsilon": list(SENT_EPSILON_GRID),
            "sent_encoding": list(SENT_ENCODING_GRID),
            "representations": list(args.representations),
        },
        "cells": {},
    }

    for cell in args.cells:
        path = args.results_dir / CELLS[cell]
        if not path.exists():
            logger.error("Cell %s: missing run file %s", cell, path)
            continue
        results = json.loads(path.read_text())
        prompts = load_run_prompts(results)
        entry: dict = {
            "run_file": CELLS[cell],
            "model": results["config"].get("model"),
            "dataset": results["config"].get("dataset"),
            "temperature": results["config"].get("temperature"),
            "device_resolved": results["config"].get("device_resolved"),
            "dtype_resolved": results["config"].get("dtype_resolved"),
            "representations": {},
        }
        for representation in args.representations:
            logger.info("Cell %s / %s: computing", cell, representation)
            rows = build_cell_table(
                results,
                prompts,
                representation,
                include_tsed=not args.no_tsed,
                jobs=args.jobs,
            )
            correlations = {name: correlate(rows, name) for name in metric_names}
            _apply_fdr_correction(correlations)
            entry["representations"][representation] = {
                "n_problems": len(rows),
                "correlations": correlations,
                "per_problem": rows,
            }
            logger.info(
                "Cell %s / %s: %d problems, %d configurations",
                cell,
                representation,
                len(rows),
                len(metric_names),
            )
        analysis["cells"][cell] = entry

    path = save_results(analysis, prefix="analysis_week11_field_audit")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
