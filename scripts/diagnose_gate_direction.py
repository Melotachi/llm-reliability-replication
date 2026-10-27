"""Week 8 (joint): is the gate's ordering wrong, and is its signal its own?

`diagnose_gate_target.py` scored the stored out-of-fold ``P(pass)`` against
both targets. On the Week 7 MBPP cohort the repairability AUC was 0.518 and the
gate looked *orthogonal* to what it was allocating. On the Week 8 HumanEval
cohort the same measurement returns 0.706 with a permutation p of 1e-4, and the
gate-exclusion test concentrates successes in the discarded tail
(hypergeometric p = 4.4e-4). Two questions follow, and neither is answered by
another AUC:

1. **Is the signal the gate's own?** ``P(pass)`` ranks *initial correctness* at
   AUC 0.947 on this cohort. Among the initially failing candidates, a high
   ``P(pass)`` marks a near-miss -- the other samples of that problem mostly
   passed and this one did not -- and near-misses are the failures a repairer
   one-shots. So "P(pass) predicts repairability" may be nothing more than
   "P(pass) predicts problem difficulty", which the self-consistency features
   carry by construction (they are leave-one-out distances to the sibling
   samples). ``mechanism_regression`` separates the two by nesting ``P(pass)``
   against the sibling pass rate -- the same problem's samples 1..N-1, which
   excludes the repaired coordinate and so cannot leak its outcome.

2. **What does reversing the ordering buy?** ``gate_direction_sweep`` replays
   the frozen Repair-All trace pool under the risk ordering (lowest ``P(pass)``
   first, what Week 7 shipped), the inverted ordering (highest first) and a
   seeded random control, at every coverage. Exact, not estimated: the pool
   stores per-coordinate ``repair_model_calls`` and ``final_passed``, so each
   arm is a set intersection and no model is loaded.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_smart_repair import (  # noqa: E402
    _coordinate,
    align_oof_scores,
    load_json,
    select_cohort,
    select_random_gate,
    select_risk_gate,
    summarize_condition,
    validate_repair_pool,
)
from scripts.diagnose_gate_target import roc_auc  # noqa: E402
from src.utils import get_logger, save_results  # noqa: E402

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("results/humaneval_uncertainty_merged164_20260711.json")
DEFAULT_PREDICTOR = Path("results/predictor_humaneval_20260712_215312.json")
DEFAULT_REPAIR = Path("results/humaneval_week8_repair_all_sample0_qwen15b.json")
DEFAULT_FIGURE_DIR = Path("reports/figures")
DEFAULT_COVERAGES = tuple(round(0.05 * step, 2) for step in range(1, 21))


def select_inverted_gate(cohort: list[dict], count: int) -> set[tuple[str, int]]:
    """Select the highest P(pass) rows -- `select_risk_gate` with the sign flipped.

    Args:
        cohort: Scored cohort rows carrying "p_pass_oof" and "source_order".
        count: Number of coordinates to gate.
    Returns:
        Set of (task_id, sample_index) coordinates.
    """
    if not 0 <= count <= len(cohort):
        raise ValueError("gate count must be between zero and cohort size")
    ranked = sorted(cohort, key=lambda row: (-row["p_pass_oof"], row["source_order"]))
    return {_coordinate(row) for row in ranked[:count]}


def sibling_pass_rates(source: dict, sample_index: int = 0) -> dict[str, float]:
    """Per-problem pass rate over every sample *except* `sample_index`.

    The excluded sample is the one the cohort repairs, so this covariate is
    computed without reading the outcome it is used to explain.

    Args:
        source: Loaded generation run JSON.
        sample_index: Sample index held out of the average.
    Returns:
        Mapping task_id -> sibling pass rate; problems with no sibling samples
        are omitted.
    """
    rates: dict[str, float] = {}
    for problem in source.get("per_problem", []):
        siblings = [
            bool(sample["passed"])
            for index, sample in enumerate(problem.get("samples", []))
            if index != sample_index
        ]
        if siblings:
            rates[problem["task_id"]] = sum(siblings) / len(siblings)
    return rates


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Unpenalized logistic fit with Wald standard errors and log-likelihood.

    Standard errors come from the observed information matrix
    (X'WX)^-1, W = diag(p(1-p)); statsmodels is not a project dependency and
    the model here is at most three columns wide.
    """
    model = LogisticRegression(penalty=None, max_iter=10_000)
    model.fit(features, labels)
    probabilities = model.predict_proba(features)[:, 1]
    design = np.column_stack([np.ones(len(features)), features])
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (design * weights[:, None])
    try:
        covariance = np.linalg.inv(information)
        errors = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:  # pragma: no cover - singular design
        errors = np.full(design.shape[1], float("nan"))
    coefficients = np.concatenate([model.intercept_, model.coef_[0]])
    epsilon = 1e-12
    log_likelihood = float(
        np.sum(
            labels * np.log(probabilities + epsilon)
            + (1 - labels) * np.log(1.0 - probabilities + epsilon)
        )
    )
    z_scores = coefficients / errors
    return {
        "coefficients": [float(value) for value in coefficients],
        "standard_errors": [float(value) for value in errors],
        "wald_z": [float(value) for value in z_scores],
        "wald_p": [float(2.0 * (1.0 - _normal_cdf(abs(z)))) for z in z_scores],
        "log_likelihood": log_likelihood,
        "n": int(len(labels)),
    }


def _normal_cdf(value: float) -> float:
    """Standard normal CDF via the error function (no scipy round-trip needed)."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _likelihood_ratio(full: dict, reduced: dict, df: int = 1) -> dict[str, float]:
    statistic = 2.0 * (full["log_likelihood"] - reduced["log_likelihood"])
    return {
        "statistic": float(statistic),
        "df": df,
        "p_value": float(chi2.sf(max(statistic, 0.0), df)),
    }


def mechanism_regression(
    cohort: list[dict],
    repair_by_coordinate: dict[tuple[str, int], dict],
    sibling_rates: dict[str, float],
) -> dict[str, Any]:
    """Does P(pass) rank repairability beyond the problem's sibling pass rate?

    Fits three nested logistic models of repair success over the initially
    failing coordinates -- P(pass) alone, sibling pass rate alone, and both --
    on z-scored predictors, and reports the likelihood-ratio test for each
    predictor given the other.

    Args:
        cohort: Scored cohort rows.
        repair_by_coordinate: Frozen Repair-All records keyed by coordinate.
        sibling_rates: Output of sibling_pass_rates().
    Returns:
        Result block with univariate AUCs, the three fits and both LR tests.
    """
    probabilities, siblings, outcomes = [], [], []
    for row in cohort:
        if row["initial_passed"]:
            continue
        task_id = row["task_id"]
        if task_id not in sibling_rates:
            continue
        probabilities.append(row["p_pass_oof"])
        siblings.append(sibling_rates[task_id])
        outcomes.append(bool(repair_by_coordinate[_coordinate(row)]["final_passed"]))

    p_pass = np.asarray(probabilities, dtype=float)
    sibling = np.asarray(siblings, dtype=float)
    labels = np.asarray(outcomes, dtype=int)
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        raise ValueError("mechanism regression needs both repair outcomes present")

    def standardize(values: np.ndarray) -> np.ndarray:
        spread = values.std(ddof=0)
        return (values - values.mean()) / spread if spread > 0 else values * 0.0

    z_pass = standardize(p_pass)
    z_sibling = standardize(sibling)
    fit_pass = _fit_logistic(z_pass[:, None], labels)
    fit_sibling = _fit_logistic(z_sibling[:, None], labels)
    fit_both = _fit_logistic(np.column_stack([z_pass, z_sibling]), labels)

    # A constant predictor makes Pearson's r undefined (nan), which would then
    # travel into the results JSON; report None instead, as the Week 8 sweep
    # does for Spearman.
    correlated = p_pass.std(ddof=0) > 0 and sibling.std(ddof=0) > 0
    return {
        "n": int(labels.size),
        "n_repaired": int(labels.sum()),
        "predictor_correlation_pearson": (
            float(np.corrcoef(p_pass, sibling)[0, 1]) if correlated else None
        ),
        "univariate_roc_auc": {
            "p_pass_oof": roc_auc(p_pass, labels),
            "sibling_pass_rate": roc_auc(sibling, labels),
        },
        "mean_by_outcome": {
            "p_pass_oof": {
                "repaired": float(p_pass[labels == 1].mean()),
                "unrepaired": float(p_pass[labels == 0].mean()),
            },
            "sibling_pass_rate": {
                "repaired": float(sibling[labels == 1].mean()),
                "unrepaired": float(sibling[labels == 0].mean()),
            },
        },
        "models": {
            "p_pass_only": {"terms": ["intercept", "p_pass_oof"], **fit_pass},
            "sibling_only": {
                "terms": ["intercept", "sibling_pass_rate"],
                **fit_sibling,
            },
            "both": {
                "terms": ["intercept", "p_pass_oof", "sibling_pass_rate"],
                **fit_both,
            },
        },
        "likelihood_ratio": {
            "p_pass_given_sibling": _likelihood_ratio(fit_both, fit_sibling),
            "sibling_given_p_pass": _likelihood_ratio(fit_both, fit_pass),
        },
    }


def gate_direction_sweep(
    cohort: list[dict],
    repair_by_coordinate: dict[tuple[str, int], dict],
    *,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
    n_seeds: int = 200,
    first_seed: int = 0,
) -> list[dict[str, Any]]:
    """Exact passes and calls for the risk, inverted and random gates by coverage.

    Args:
        cohort: Scored cohort rows.
        repair_by_coordinate: Frozen Repair-All records keyed by coordinate.
        coverages: Gate coverages to evaluate.
        n_seeds: Random draws averaged per coverage.
        first_seed: Lowest random seed.
    Returns:
        One row per coverage with added passes and model calls per arm.
    """
    rows = []
    for coverage in coverages:
        count = math.ceil(coverage * len(cohort))
        risk, _ = summarize_condition(
            "risk", cohort, select_risk_gate(cohort, count), repair_by_coordinate
        )
        inverted, _ = summarize_condition(
            "inverted",
            cohort,
            select_inverted_gate(cohort, count),
            repair_by_coordinate,
        )
        random_passes, random_calls = [], []
        for seed in range(first_seed, first_seed + n_seeds):
            drawn, _ = summarize_condition(
                "random",
                cohort,
                select_random_gate(cohort, count, seed),
                repair_by_coordinate,
            )
            random_passes.append(drawn["added_passes"])
            random_calls.append(drawn["repair_model_calls"])
        rows.append(
            {
                "coverage": float(coverage),
                "gate_count": count,
                "risk": {
                    "added_passes": risk["added_passes"],
                    "repair_model_calls": risk["repair_model_calls"],
                    "attempted_repairs": risk["attempted_repairs"],
                },
                "inverted": {
                    "added_passes": inverted["added_passes"],
                    "repair_model_calls": inverted["repair_model_calls"],
                    "attempted_repairs": inverted["attempted_repairs"],
                },
                "random_mean": {
                    "added_passes": float(np.mean(random_passes)),
                    "repair_model_calls": float(np.mean(random_calls)),
                    "added_passes_std": (
                        float(np.std(random_passes, ddof=1)) if n_seeds > 1 else 0.0
                    ),
                },
            }
        )
    return rows


def budget_matched_point(
    sweep: list[dict], reference_calls: int
) -> dict[str, Any] | None:
    """Best inverted-gate row that spends no more calls than `reference_calls`."""
    affordable = [
        row for row in sweep if row["inverted"]["repair_model_calls"] <= reference_calls
    ]
    if not affordable:
        return None
    best = max(affordable, key=lambda row: row["inverted"]["added_passes"])
    return {
        "coverage": best["coverage"],
        "gate_count": best["gate_count"],
        "added_passes": best["inverted"]["added_passes"],
        "repair_model_calls": best["inverted"]["repair_model_calls"],
        "reference_calls": int(reference_calls),
    }


def build_analysis(
    source_path: Path,
    source: dict,
    predictor: dict,
    repair_run: dict,
    *,
    sample_index: int = 0,
    smart_coverage: float = 0.75,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
    n_seeds: int = 200,
    first_seed: int = 0,
    feature_set: str = "self_consistency_plus_execution",
    model_type: str = "logistic",
) -> dict[str, Any]:
    """Run both diagnostics against one frozen repair trace pool."""
    scored = align_oof_scores(source, predictor, feature_set, model_type)
    cohort = select_cohort(scored, sample_index=sample_index)
    repair_by_coordinate = validate_repair_pool(
        source_path, cohort, repair_run, sample_index
    )
    gate_count = math.ceil(smart_coverage * len(cohort))
    smart, _ = summarize_condition(
        "smart", cohort, select_risk_gate(cohort, gate_count), repair_by_coordinate
    )
    repair_all, _ = summarize_condition(
        "repair_all",
        cohort,
        {_coordinate(row) for row in cohort},
        repair_by_coordinate,
    )
    inverted, _ = summarize_condition(
        "inverted",
        cohort,
        select_inverted_gate(cohort, gate_count),
        repair_by_coordinate,
    )
    sweep = gate_direction_sweep(
        cohort,
        repair_by_coordinate,
        coverages=coverages,
        n_seeds=n_seeds,
        first_seed=first_seed,
    )
    return {
        "analysis": "joint_gate_direction",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "source_results": str(source_path),
            "source_run_id": source.get("run_id"),
            "repair_run_id": repair_run.get("run_id"),
            "repair_model": repair_run.get("config", {}).get("model"),
            "predictor_feature_set": feature_set,
            "predictor_model_type": model_type,
            "probability_kind": "out_of_fold_raw_rank_only",
            "sample_index": sample_index,
            "smart_coverage": smart_coverage,
            "n_seeds": n_seeds,
            "first_seed": first_seed,
        },
        "cohort": {
            "candidates": len(cohort),
            "initially_passing": smart["initial_passed"],
            "initially_failing": len(cohort) - smart["initial_passed"],
            "repaired_in_pool": repair_all["successful_repairs"],
        },
        "arms_at_smart_coverage": {
            "risk": {
                "added_passes": smart["added_passes"],
                "repair_model_calls": smart["repair_model_calls"],
                "attempted_repairs": smart["attempted_repairs"],
                "repair_success_rate": smart["repair_success_rate"],
            },
            "inverted": {
                "added_passes": inverted["added_passes"],
                "repair_model_calls": inverted["repair_model_calls"],
                "attempted_repairs": inverted["attempted_repairs"],
                "repair_success_rate": inverted["repair_success_rate"],
            },
            "repair_all": {
                "added_passes": repair_all["added_passes"],
                "repair_model_calls": repair_all["repair_model_calls"],
                "attempted_repairs": repair_all["attempted_repairs"],
                "repair_success_rate": repair_all["repair_success_rate"],
            },
        },
        "mechanism_regression": mechanism_regression(
            cohort, repair_by_coordinate, sibling_pass_rates(source, sample_index)
        ),
        "direction_sweep": sweep,
        "budget_matched_inverted": budget_matched_point(
            sweep, smart["repair_model_calls"]
        ),
    }


def plot_gate_direction(analysis: dict, output: Path) -> None:
    """Passes-versus-calls frontier for the three gate orderings."""
    sweep = analysis["direction_sweep"]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for arm, label, style in (
        (
            "risk",
            "risk gate (lowest P(pass) first)",
            {"color": "#c0392b", "marker": "o"},
        ),
        (
            "inverted",
            "inverted gate (highest P(pass) first)",
            {"color": "#1f6f3f", "marker": "s"},
        ),
        (
            "random_mean",
            "random gate (mean of seeds)",
            {"color": "#5b7fb4", "marker": "^"},
        ),
    ):
        calls = [row[arm]["repair_model_calls"] for row in sweep]
        passes = [row[arm]["added_passes"] for row in sweep]
        axis.plot(calls, passes, linewidth=1.6, markersize=4, label=label, **style)
    ceiling = analysis["arms_at_smart_coverage"]["repair_all"]
    axis.axhline(
        ceiling["added_passes"],
        linestyle="--",
        color="#2e8b57",
        linewidth=1.2,
        label=(
            f"Repair-All ceiling ({ceiling['added_passes']} passes, "
            f"{ceiling['repair_model_calls']} calls)"
        ),
    )
    axis.set_xlabel("repair model calls (lower is better)")
    axis.set_ylabel("added passes (higher is better)")
    axis.set_title("Gate ordering at matched coverage, HumanEval sample-0 cohort")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate ordering and near-miss mechanism diagnostics"
    )
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--predictor-results", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--repair-results", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--smart-coverage", type=float, default=0.75)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--feature-set", default="self_consistency_plus_execution")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--prefix", default="analysis_week8_gate_direction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = build_analysis(
        args.source_results,
        load_json(args.source_results),
        load_json(args.predictor_results),
        load_json(args.repair_results),
        sample_index=args.sample_index,
        smart_coverage=args.smart_coverage,
        n_seeds=args.n_seeds,
        first_seed=args.first_seed,
        feature_set=args.feature_set,
        model_type=args.model_type,
    )
    figure_path = args.figure_dir / "week8_gate_direction.png"
    plot_gate_direction(analysis, figure_path)
    analysis["figure"] = str(figure_path)
    out_path = save_results(analysis, prefix=args.prefix)

    regression = analysis["mechanism_regression"]
    both = regression["models"]["both"]
    logger.info("Wrote %s", out_path)
    logger.info(
        "P(pass) AUC %.3f | sibling AUC %.3f | corr %s",
        regression["univariate_roc_auc"]["p_pass_oof"],
        regression["univariate_roc_auc"]["sibling_pass_rate"],
        regression["predictor_correlation_pearson"],
    )
    logger.info(
        "joint model coefficients %s (Wald p %s)",
        [round(value, 3) for value in both["coefficients"]],
        [round(value, 4) for value in both["wald_p"]],
    )
    logger.info(
        "LR p_pass|sibling p=%.4f ; sibling|p_pass p=%.4f",
        regression["likelihood_ratio"]["p_pass_given_sibling"]["p_value"],
        regression["likelihood_ratio"]["sibling_given_p_pass"]["p_value"],
    )
    arms = analysis["arms_at_smart_coverage"]
    for name, arm in arms.items():
        logger.info(
            "%s: %s passes / %s calls",
            name,
            arm["added_passes"],
            arm["repair_model_calls"],
        )
    logger.info("budget-matched inverted: %s", analysis["budget_matched_inverted"])


if __name__ == "__main__":
    main()
