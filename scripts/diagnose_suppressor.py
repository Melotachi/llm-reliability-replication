"""Week 7: is ``token_jaccard_diversity`` a genuine suppressor, or was it the confound?

Student A deliverable, closing the Week 7 report's first "next week" item.

Background. Every diversity feature in the project's predictor arms carries a
negative logistic coefficient — more disagreement, less likely to pass — except
``token_jaccard_diversity``, which is positive in every model, at both task
levels, in every cell. Read together with its weak standalone correlation, that
is the textbook signature of a **suppressor variable**: a feature that carries
little criterion variance itself but improves the fit by absorbing variance the
other features share (Friedman & Wall, 2005, "Graphical Views of Suppression and
Multicollinearity"). Akinwande, Dikko & Samson (2015) give the checkable
condition for calling it that rather than plain multicollinearity: the variance
inflation factor must stay in the acceptable range (VIF < 5), otherwise the
positive sign is an unstable-coefficient artefact.

The reason this needs re-running: the Week 7 prompt-inclusion confound means the
original suppressor evidence was computed on a metric whose input strings were
~70% shared prompt. A diluted measurement behaves erratically in a multivariate
model, so "suppressor" and "confounded measurement" were not distinguishable at
the time. Under ``--representation corrected`` the dilution is gone, and the
question becomes answerable: if the effect is the confound, it disappears; if it
survives, the suppressor reading stands on its own.

Pure re-analysis — reads stored run JSONs, generates nothing.

Usage:
    python scripts/diagnose_suppressor.py \\
        --run cg_he164=results/humaneval_uncertainty_merged164_20260711.json \\
        --run ds_he60=results/humaneval_deepseek60_merged_20260712.json \\
        --no-codebert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_uncertainty import (
    _residualize,
    extract_problem_table,
    load_run_prompts,
)
from src.uncertainty import METRIC_REPRESENTATIONS
from src.utils import get_logger, save_results, set_all_seeds

logger = get_logger(__name__)

# The feature whose positive coefficient prompted this diagnostic.
DEFAULT_TARGET = "token_jaccard_diversity"
# The variables the target would be suppressing variance in: the Week 4
# "reduced" feature set plus the legacy pairwise metric. This is exactly the
# control set the Week 6 residualization used, so the "full" representation
# reproduces its published numbers.
DEFAULT_PARTNERS = [
    "lexical_diversity",
    "token_cosine_diversity",
    "ast_node_diversity",
    "pairwise_distance",
]
# Akinwande et al. (2015): VIF at or above this is multicollinearity, not
# suppression, and a positive coefficient there means an unstable estimate.
VIF_CEILING = 5.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the suppressor diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Cell label and stored results JSON. Repeatable.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["full", "corrected"],
        choices=list(METRIC_REPRESENTATIONS),
        help="Metric input representations to compare (default: full corrected).",
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="Suspected suppressor."
    )
    parser.add_argument(
        "--partners",
        nargs="+",
        default=DEFAULT_PARTNERS,
        help="Features the target is tested against.",
    )
    parser.add_argument("--no-codebert", action="store_true", help="Skip the encoder.")
    parser.add_argument(
        "--no-tree-edit",
        action="store_true",
        help="Skip tree_edit_distance_diversity (not read by this diagnostic; "
        "it dominates the runtime on the larger runs).",
    )
    parser.add_argument(
        "--codebert-model",
        default="microsoft/codebert-base",
        help="Encoder for the embedding metrics when they are enabled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    return parser.parse_args()


def parse_run_specs(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated --run LABEL=PATH arguments into (label, path) pairs.

    Args:
        specs: Raw --run values.
    Returns:
        List of (label, path) in input order.
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


def variance_inflation_factors(
    matrix: np.ndarray, names: list[str]
) -> dict[str, float]:
    """VIF of every column against all the others.

    VIF_j = 1 / (1 - R^2_j), where R^2_j regresses column j on the rest. The
    suppressor claim requires these to stay below VIF_CEILING (Akinwande et
    al., 2015); above it, a sign reversal is an unstable-coefficient artefact
    of collinearity rather than suppression.

    Args:
        matrix: (n_samples, n_features) feature matrix.
        names: Feature names, one per column.
    Returns:
        name -> VIF. A column with no residual variance maps to inf.
    """
    factors: dict[str, float] = {}
    for index, name in enumerate(names):
        target = matrix[:, index]
        others = [matrix[:, j] for j in range(matrix.shape[1]) if j != index]
        if not others:
            factors[name] = 1.0
            continue
        residual = _residualize(target, others)
        total = float(np.sum((target - target.mean()) ** 2))
        unexplained = float(np.sum(residual**2))
        if total == 0.0:
            factors[name] = float("inf")
            continue
        r_squared = 1.0 - unexplained / total
        factors[name] = (
            float("inf") if r_squared >= 1.0 else float(1.0 / (1.0 - r_squared))
        )
    return factors


def _correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson and Spearman coefficients of two vectors, as plain floats."""
    return {
        "pearson_r": float(scipy_stats.pearsonr(x, y).statistic),
        "spearman_rho": float(scipy_stats.spearmanr(x, y).statistic),
    }


def analyze_cell(rows: list[dict], target: str, partners: list[str]) -> dict:
    """Run the full suppressor diagnostic on one cell's problem table.

    Three independent checks, all on problem-level rows:

    1. **Zero-order vs residual sign.** The suppressor reading predicts that
       the component of the target orthogonal to its partners correlates with
       pass rate in the *opposite* direction to its raw coefficient. The
       reverse residualization (a partner residualized on the target) is the
       sanity control: it must keep its sign, otherwise the flip is an
       artefact of residualizing anything.
    2. **VIF.** Below VIF_CEILING the sign reversal is suppression; above it,
       collinearity.
    3. **Standardized logistic coefficients** at problem level (label: did any
       sample pass), which is where the positive sign was observed.

    Args:
        rows: extract_problem_table() output for one run and representation.
        target: Suspected suppressor feature name.
        partners: The other features in the set.
    Returns:
        Diagnostic dict; {"available": False, ...} when a feature is missing
        or constant across the run.
    """
    present = [name for name in [target] + partners if name in rows[0]]
    if target not in present or len(present) < 2:
        return {"available": False, "reason": "target or partner features missing"}

    pass_rates = np.array([r["pass_rate"] for r in rows], dtype=float)
    columns = {n: np.array([r[n] for r in rows], dtype=float) for n in present}
    constant = [n for n, v in columns.items() if np.ptp(v) == 0.0]
    if constant:
        return {"available": False, "reason": f"constant features: {constant}"}

    partner_names = [n for n in present if n != target]
    target_values = columns[target]

    residual_sets = {
        "resid_on_ast": [n for n in partner_names if n == "ast_node_diversity"],
        "resid_on_all_partners": partner_names,
    }
    residuals: dict[str, dict] = {}
    for label, control_names in residual_sets.items():
        if not control_names:
            continue
        controls = [columns[n] for n in control_names]
        residuals[label] = {
            "controls": control_names,
            **_correlations(_residualize(target_values, controls), pass_rates),
        }

    # Sanity control: the reverse direction must NOT flip.
    sanity = {}
    if "ast_node_diversity" in partner_names:
        sanity["ast_resid_on_target"] = _correlations(
            _residualize(columns["ast_node_diversity"], [target_values]), pass_rates
        )

    matrix = np.column_stack([columns[n] for n in present])
    vif = variance_inflation_factors(matrix, present)

    # Problem-level "did any sample pass" label — the Task A framing in which
    # the positive coefficient was first seen.
    labels = (pass_rates > 0).astype(int)
    coefficients: dict[str, float] | dict[str, str]
    if len(set(labels.tolist())) < 2:
        coefficients = {"unavailable": "single-class label"}
    else:
        scaled = StandardScaler().fit_transform(matrix)
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(scaled, labels)
        coefficients = {
            name: float(value) for name, value in zip(present, model.coef_[0])
        }

    return {
        "available": True,
        "n_problems": len(rows),
        "features": present,
        "zero_order": {n: _correlations(columns[n], pass_rates) for n in present},
        "residual_correlations": residuals,
        "sanity_controls": sanity,
        "vif": vif,
        "vif_ceiling": VIF_CEILING,
        "max_vif": max(vif.values()),
        "logistic_coefficients": coefficients,
        "verdict": _verdict(target, residuals, vif),
    }


def _verdict(target: str, residuals: dict[str, dict], vif: dict[str, float]) -> str:
    """One-line reading of the three checks, in the report's own vocabulary."""
    if not residuals:
        return "inconclusive: no residual set available"
    raw_flip = all(entry["pearson_r"] > 0 for entry in residuals.values())
    collinear = vif.get(target, float("inf")) >= VIF_CEILING
    if raw_flip and not collinear:
        return "suppressor: residual sign flips positive, VIF below ceiling"
    if raw_flip and collinear:
        return "collinearity, not suppression: sign flips but VIF at/above ceiling"
    return "not a suppressor here: residual keeps its raw sign"


def print_cell(label: str, representation: str, analysis: dict, target: str) -> None:
    """Print one cell's diagnostic block to stdout."""
    header = f"{label} / {representation}"
    print(f"\n=== {header} ===")
    if not analysis.get("available"):
        print(f"  unavailable: {analysis['reason']}")
        return
    zero = analysis["zero_order"][target]
    print(
        f"  {target}: raw r = {zero['pearson_r']:+.3f}, rho = {zero['spearman_rho']:+.3f}"
    )
    for name, entry in analysis["residual_correlations"].items():
        print(
            f"    {name:<24} r = {entry['pearson_r']:+.3f}  "
            f"rho = {entry['spearman_rho']:+.3f}   (controls: {', '.join(entry['controls'])})"
        )
    for name, entry in analysis["sanity_controls"].items():
        print(f"    {name:<24} r = {entry['pearson_r']:+.3f}  (must stay negative)")
    vif_cells = ", ".join(f"{n}={v:.2f}" for n, v in analysis["vif"].items())
    print(f"  VIF: {vif_cells}")
    coefficients = analysis["logistic_coefficients"]
    if "unavailable" not in coefficients:
        coef_cells = ", ".join(f"{n}={v:+.3f}" for n, v in coefficients.items())
        print(f"  standardized logistic coefficients: {coef_cells}")
    print(f"  verdict: {analysis['verdict']}")


def main() -> None:
    """CLI entry point: diagnose each cell under each representation."""
    args = parse_args()
    set_all_seeds(args.seed)
    runs = parse_run_specs(args.run)

    stats: dict = {
        "analysis": "week7_suppressor",
        "config": {
            "seed": args.seed,
            "target": args.target,
            "partners": args.partners,
            "representations": args.representations,
            "include_codebert": not args.no_codebert,
            "include_tree_edit": not args.no_tree_edit,
            "vif_ceiling": VIF_CEILING,
        },
        "source_results_files": {label: str(path) for label, path in runs},
        "cells": {},
    }

    for label, path in runs:
        results = json.loads(path.read_text())
        prompts = load_run_prompts(results)
        stats["cells"][label] = {
            "run_id": results.get("run_id"),
            "model": results.get("config", {}).get("model"),
            "representations": {},
        }
        for representation in args.representations:
            rows = extract_problem_table(
                results,
                include_codebert=not args.no_codebert,
                codebert_model=args.codebert_model,
                representation=representation,
                prompts=prompts,
                include_tree_edit=not args.no_tree_edit,
            )
            analysis = analyze_cell(rows, args.target, args.partners)
            stats["cells"][label]["representations"][representation] = analysis
            print_cell(label, representation, analysis, args.target)

    out_path = save_results(stats, prefix="analysis_week7_suppressor")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
