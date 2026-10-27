"""Week 7: the family-transfer matrix, recomputed under corrected metric inputs.

Student A deliverable. Answers one question across every regime cell the
project has run: **how much of the Weeks 2-6 "which uncertainty signals
transfer?" result survives once the prompt-inclusion confound is removed?**

Background (src/uncertainty.py, "prompt-inclusion confound" section):
run_pipeline.py assembles `prompt + sample` before executing — necessary,
since a bare completion is not standalone Python — and hands the same strings
to the diversity metrics. Roughly 70% of every compared string is therefore
identical by construction, which dilutes the token-overlap family ~3x more
than the structural one (0.693 of the word tokens vs 0.239 of the AST nodes).
The Weeks 2-6 headline — "the structural family transfers, the lexical/token
family does not" — is partly an artefact of that asymmetry.

This script re-runs the whole matrix under each requested representation and
prints them side by side, plus the free length baseline that any claim about
the apparatus has to beat. It is pure re-analysis: it reads stored run JSONs
and generates nothing.

Usage:
    python scripts/compare_representations.py \\
        --run cg_he164=results/humaneval_uncertainty_merged164_20260711.json \\
        --run ds_he60=results/humaneval_deepseek60_merged_20260712.json \\
        --no-codebert

    # every representation, not just full vs corrected
    python scripts/compare_representations.py --run cg_he60=... \\
        --representations full completion signature corrected
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_uncertainty import (
    CONFOUND_BASELINE,
    METRIC_ORDER,
    compute_confound_analysis,
    compute_correlations,
    extract_problem_table,
    load_run_prompts,
)
from src.uncertainty import METRIC_REPRESENTATIONS
from src.utils import get_logger, save_results, set_all_seeds

logger = get_logger(__name__)

# Benjamini-Hochberg level for the "does this cell's correlation survive
# multiple-comparison correction?" annotation, matching the alpha
# analyze_uncertainty.py's FDR column is read against in the Week 6 report.
FDR_ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the representation-comparison entry point."""
    parser = argparse.ArgumentParser(
        description="Recompute the family-transfer matrix under corrected "
        "metric input representations (Week 7 prompt-inclusion confound)"
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A regime cell: short label and its pipeline results JSON. "
        "Repeat once per cell (e.g. --run ds_he60=results/....json).",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=METRIC_REPRESENTATIONS,
        default=["full", "corrected"],
        help="Representations to compute, in report order (default: the "
        "before/after pair, full then corrected).",
    )
    parser.add_argument(
        "--codebert-model",
        type=str,
        default="microsoft/codebert-base",
        help="HuggingFace encoder for the CodeBERT metrics.",
    )
    parser.add_argument(
        "--no-codebert",
        action="store_true",
        help="Skip the CodeBERT metrics (~500 MB encoder). Recommended for "
        "a full multi-cell sweep, where the encoder dominates the runtime.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the tables but skip persisting the stats JSON.",
    )
    return parser.parse_args()


def parse_run_specs(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated --run LABEL=PATH arguments into (label, path) pairs.

    Args:
        specs: Raw --run values.
    Returns:
        List of (label, path), input order preserved.
    Raises:
        ValueError: On a malformed spec or a missing results file.
    """
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--run expects LABEL=PATH, got {spec!r}")
        label, _, raw_path = spec.partition("=")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"results file for cell {label!r} not found: {path}")
        parsed.append((label.strip(), path))
    if not parsed:
        raise ValueError("at least one --run LABEL=PATH is required")
    return parsed


def analyze_cell(
    results: dict,
    representation: str,
    include_codebert: bool,
    codebert_model: str,
    prompts: dict[str, str],
) -> dict:
    """Correlations plus confound analysis for one cell under one representation.

    Args:
        results: Full pipeline results (CLAUDE.md Section 8 schema).
        representation: One of src.uncertainty.METRIC_REPRESENTATIONS.
        include_codebert: Whether to compute the embedding metrics.
        codebert_model: HuggingFace encoder for those metrics.
        prompts: task_id -> prompt for the run's dataset.
    Returns:
        {"n_problems", "correlations", "confound_analysis"}.
    """
    rows = extract_problem_table(
        results,
        include_codebert=include_codebert,
        codebert_model=codebert_model,
        representation=representation,
        prompts=prompts,
    )
    return {
        "n_problems": len(rows),
        "correlations": compute_correlations(rows),
        "confound_analysis": compute_confound_analysis(rows),
    }


def _rho_cell(entry: dict | None) -> str:
    """One matrix cell: Spearman rho, or n.s.(rho) when it fails BH-FDR."""
    if entry is None:
        return "-"
    rho = entry.get("spearman_rho")
    if rho is None:
        return "const"
    fdr_p = entry.get("spearman_p_fdr")
    if fdr_p is not None and fdr_p > FDR_ALPHA:
        return f"n.s.({rho:+.2f})"
    return f"{rho:+.2f}"


def print_matrix(analysis: dict, representation: str) -> None:
    """Print the metric x cell Spearman matrix for one representation."""
    cells = list(analysis["cells"])
    print(f"\n=== representation: {representation} ===")
    print(
        "Spearman ρ vs pass rate; n.s.(ρ) = failed Benjamini-Hochberg FDR "
        f"at {FDR_ALPHA}"
    )
    header = f"{'metric':<30}" + "".join(f"{cell:<15}" for cell in cells)
    print(header)
    for metric in METRIC_ORDER:
        entries = [
            analysis["cells"][cell]
            .get(representation, {})
            .get("correlations", {})
            .get(metric)
            for cell in cells
        ]
        if all(e is None for e in entries):
            continue
        print(f"{metric:<30}" + "".join(f"{_rho_cell(e):<15}" for e in entries))


def print_baseline_row(analysis: dict, representation: str) -> None:
    """Print the free length baseline and the structural partial beside it.

    The baseline is representation-independent by construction (completion
    length does not depend on how the metrics are computed); it is printed
    with each matrix so the comparison a reviewer wants is on one screen.

    Every row here is a **Pearson r**, including the baseline — the matrix
    above is Spearman, and mixing the two in one comparison block would
    make the "how much of the metric is just length?" ratio meaningless.
    """
    cells = list(analysis["cells"])
    print(f"\n{'-- length baseline --':<30}" + "".join(f"{cell:<15}" for cell in cells))
    rows: dict[str, list[str]] = {
        "mean_completion_chars (r)": [],
        "ast_node_diversity (raw r)": [],
        "ast_node_diversity (r|length)": [],
        "length as % of |ast raw r|": [],
    }
    for cell in cells:
        confound = (
            analysis["cells"][cell].get(representation, {}).get("confound_analysis", {})
        )
        if not confound.get("available"):
            for values in rows.values():
                values.append("-")
            continue
        base = confound["covariate_baselines"].get(CONFOUND_BASELINE, {})
        ast = confound["metrics"].get("ast_node_diversity", {})
        partial = ast.get("ctrl_completion_len", {})
        base_r = base.get("pearson_r")
        ast_r = ast.get("raw_pearson_r")
        partial_r = partial.get("pearson_r")
        rows["mean_completion_chars (r)"].append(
            f"{base_r:+.2f}" if base_r is not None else "-"
        )
        rows["ast_node_diversity (raw r)"].append(
            f"{ast_r:+.2f}" if ast_r is not None else "-"
        )
        rows["ast_node_diversity (r|length)"].append(
            f"{partial_r:+.2f}" if partial_r is not None else "-"
        )
        rows["length as % of |ast raw r|"].append(
            f"{100.0 * abs(base_r) / abs(ast_r):.0f}%"
            if base_r is not None and ast_r
            else "-"
        )
    for label, values in rows.items():
        print(f"{label:<30}" + "".join(f"{value:<15}" for value in values))


def print_delta_matrix(analysis: dict, before: str, after: str) -> None:
    """Print the change in |Spearman rho| between two representations.

    This is the table that says whether the Weeks 2-6 transfer conclusion
    holds: a metric whose magnitude jumps once the shared prompt prefix is
    removed was measuring prompt share, not divergence.
    """
    cells = list(analysis["cells"])
    print(f"\n=== Δ|ρ| ({after} − {before}) — positive = stronger once corrected ===")
    print(f"{'metric':<30}" + "".join(f"{cell:<15}" for cell in cells))
    for metric in METRIC_ORDER:
        row = []
        any_value = False
        for cell in cells:
            per_rep = analysis["cells"][cell]
            before_entry = per_rep.get(before, {}).get("correlations", {}).get(metric)
            after_entry = per_rep.get(after, {}).get("correlations", {}).get(metric)
            if not before_entry or not after_entry:
                row.append("-")
                continue
            b, a = before_entry.get("spearman_rho"), after_entry.get("spearman_rho")
            if b is None or a is None:
                row.append("-")
                continue
            any_value = True
            row.append(f"{abs(a) - abs(b):+.2f}")
        if any_value:
            print(f"{metric:<30}" + "".join(f"{c:<15}" for c in row))


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    runs = parse_run_specs(args.run)
    include_codebert = not args.no_codebert

    analysis: dict = {"cells": {}, "sources": {}}
    for label, path in runs:
        logger.info("Cell %s: %s", label, path)
        with path.open() as f:
            results = json.load(f)
        prompts = load_run_prompts(results)
        if not prompts:
            logger.warning(
                "Cell %s: no prompts recovered — corrected representations "
                "will silently fall back to full strings",
                label,
            )
        config = results.get("config", {})
        analysis["sources"][label] = {
            "results_file": str(path),
            "run_id": results.get("run_id"),
            "dataset": config.get("dataset"),
            "model": config.get("model"),
            "temperature": config.get("temperature"),
            "n_problems": len(results.get("per_problem", [])),
        }
        analysis["cells"][label] = {}
        for representation in args.representations:
            logger.info("  representation=%s", representation)
            analysis["cells"][label][representation] = analyze_cell(
                results,
                representation,
                include_codebert,
                args.codebert_model,
                prompts,
            )

    for representation in args.representations:
        print_matrix(analysis, representation)
        print_baseline_row(analysis, representation)
    if len(args.representations) >= 2:
        print_delta_matrix(analysis, args.representations[0], args.representations[-1])

    stats = {
        "analysis": "week7_representation_family_transfer_matrix",
        "config": {
            "seed": args.seed,
            "representations": args.representations,
            "codebert_model": None if args.no_codebert else args.codebert_model,
            "fdr_alpha": FDR_ALPHA,
        },
        "sources": analysis["sources"],
        "cells": analysis["cells"],
    }
    if args.dry_run:
        logger.info("[DRY RUN] skipping stats JSON persistence")
    else:
        out_path = save_results(stats, prefix="analysis_week7_representations")
        logger.info("Saved representation comparison to %s", out_path)


if __name__ == "__main__":
    main()
