"""Week 8 (joint): does the risk gate rank the target it is being asked to rank?

Week 7's Smart Repair gate orders candidates by Student A's out-of-fold
``P(pass)`` and repairs the lowest-scoring ones. A seeded random control beat it
(9/10 vs 7/10 of Repair-All's successes, at fewer model calls), which the Week 7
report recorded as a negative result without a mechanism.

This script supplies the mechanism. ``P(pass)`` was trained and validated as a
predictor of **initial correctness**. Gating repair asks it a different
question -- *which failure is worth spending repair calls on* -- whose target is
**repairability**. The two are only the same signal if a sample's probability of
being right also ranks its probability of being fixable.

The diagnostic scores the same stored probabilities against both targets on the
same cohort:

* ``initial_passed`` over the whole cohort -- the task the predictor was built
  for, and a sanity check that the alignment is correct;
* ``final_passed`` over the initially failing subset -- the task the gate
  actually performs.

Both come with a permutation test and a bootstrap interval, because the second
target has few positives (10 of 54 on the Week 7 MBPP cohort) and a point
estimate alone would over-claim.

No model is loaded and no code is generated: this is a re-analysis of
``analyze_smart_repair.py``'s inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_smart_repair import (  # noqa: E402
    align_oof_scores,
    load_json,
    select_cohort,
    select_risk_gate,
    validate_repair_pool,
)
from src.utils import get_logger, save_results  # noqa: E402

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("results/mbpp_uncertainty_20260712_024909.json")
DEFAULT_PREDICTOR = Path("results/predictor_mbpp_20260712_212240.json")
DEFAULT_REPAIR = Path("results/mbpp_week7_repair_all_sample0_qwen15b.json")


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC with mid-ranks for ties.

    Implemented directly (Mann-Whitney U form) rather than pulled from sklearn
    so the permutation loop stays cheap: it is called tens of thousands of
    times per target.

    Args:
        scores: Predicted scores, higher meaning more likely positive.
        labels: Binary labels in {0, 1}, same length as scores.
    Returns:
        Area under the ROC curve; 0.5 for a non-discriminating score.
    Raises:
        ValueError: If labels are single-class, where AUC is undefined.
    """
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC needs both classes present")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    # Mid-rank ties so duplicated probabilities cannot inflate the statistic.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if unique.size != scores.size:
        sums = np.zeros(unique.size, dtype=float)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _permutation_p(
    scores: np.ndarray, labels: np.ndarray, observed: float, rng, n: int
) -> float:
    """Two-sided permutation p-value for |AUC - 0.5| under label exchange."""
    shuffled = labels.copy()
    extreme = 0
    for _ in range(n):
        rng.shuffle(shuffled)
        if abs(roc_auc(scores, shuffled) - 0.5) >= abs(observed - 0.5) - 1e-12:
            extreme += 1
    return (extreme + 1) / (n + 1)


def _bootstrap_interval(
    scores: np.ndarray, labels: np.ndarray, rng, n: int
) -> list[float]:
    """Percentile bootstrap 95% interval, skipping single-class resamples."""
    draws = []
    size = labels.size
    for _ in range(n):
        index = rng.integers(0, size, size)
        resampled = labels[index]
        if resampled.min() == resampled.max():
            continue
        draws.append(roc_auc(scores[index], resampled))
    if not draws:
        return [float("nan"), float("nan")]
    array = np.asarray(draws)
    return [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))]


def score_target(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    seed: int,
    n_permutations: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Score one (score, label) pair with AUC, permutation p, and bootstrap CI.

    A single-class target is reported as undefined rather than raised: a hard
    cohort can legitimately contain no initial passes or no repair successes,
    and one undefined target should not lose the other one's result.

    Args:
        scores: Predicted P(pass) values.
        labels: Binary target.
        name: Target label recorded in the result dict.
        seed: Seed for the permutation and bootstrap generators.
        n_permutations: Label shuffles for the p-value.
        n_bootstrap: Resamples for the interval.
    Returns:
        Dict with the AUC, its p-value, its interval, and class counts.
    """
    rng = np.random.default_rng(seed)
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return {
            "target": name,
            "n": int(labels.size),
            "n_positive": positives,
            "n_negative": int(labels.size - positives),
            "roc_auc": None,
            "undefined_reason": "single-class target; ROC-AUC is not defined",
            "permutation_p": None,
            "bootstrap_95": [None, None],
            "mean_score_positive": (
                float(scores[labels == 1].mean()) if positives else None
            ),
            "mean_score_negative": (
                float(scores[labels == 0].mean()) if positives != labels.size else None
            ),
        }
    observed = roc_auc(scores, labels)
    return {
        "target": name,
        "n": int(labels.size),
        "n_positive": positives,
        "n_negative": int(labels.size - positives),
        "roc_auc": observed,
        "permutation_p": _permutation_p(scores, labels, observed, rng, n_permutations),
        "n_permutations": n_permutations,
        "bootstrap_95": _bootstrap_interval(scores, labels, rng, n_bootstrap),
        "n_bootstrap": n_bootstrap,
        "mean_score_positive": float(scores[labels == 1].mean()),
        "mean_score_negative": float(scores[labels == 0].mean()),
    }


def gate_exclusion_test(
    failing: list[dict],
    repair_by_coordinate: dict[tuple[str, int], dict],
    gate: set[tuple[str, int]],
) -> dict[str, Any]:
    """Do the repairable failures concentrate in the region the gate cuts off?

    A near-chance ROC-AUC over *all* failures is not by itself the operative
    result, because the gate does not act on all of them: it repairs the
    lowest-scoring ones and discards the top of the ranking. An overall AUC of
    0.5 can coexist with an ordering effect concentrated exactly in that
    discarded tail, which would be a stronger and different claim than
    orthogonality -- so it is tested rather than assumed.

    The statistic is the hypergeometric tail: given `K` repair successes among
    `N` failures, how surprising is it that `observed` of them landed in the
    `n` coordinates the gate excluded?

    Args:
        failing: Initially failing cohort rows, any order.
        repair_by_coordinate: Repair records keyed by (task_id, sample_index).
        gate: Coordinates the risk gate selected for repair.
    Returns:
        Dict with the excluded/observed/expected counts and the tail p-value.
    """
    ranked = sorted(failing, key=lambda row: (row["p_pass_oof"], row["source_order"]))
    coordinates = [(row["task_id"], row["sample_index"]) for row in ranked]
    success_ranks = [
        index
        for index, coordinate in enumerate(coordinates)
        if repair_by_coordinate[coordinate]["final_passed"]
    ]
    excluded_ranks = [
        index for index, coordinate in enumerate(coordinates) if coordinate not in gate
    ]

    n_failures = len(ranked)
    n_successes = len(success_ranks)
    n_excluded = len(excluded_ranks)
    observed = len(set(success_ranks) & set(excluded_ranks))
    expected = n_excluded * n_successes / n_failures if n_failures else 0.0

    result = {
        "n_failures": n_failures,
        "n_successes": n_successes,
        "n_excluded_by_gate": n_excluded,
        "successes_excluded_observed": observed,
        "successes_excluded_expected": expected,
        "success_ranks_ascending_p_pass": success_ranks,
        "mean_success_rank": (
            sum(success_ranks) / n_successes if n_successes else None
        ),
        "midpoint_rank": (n_failures - 1) / 2 if n_failures else None,
    }
    if n_successes and n_excluded:
        # scipy is already a project dependency (Week 5 statistics).
        from scipy.stats import hypergeom

        result["hypergeometric_p_upper"] = float(
            hypergeom.sf(observed - 1, n_failures, n_successes, n_excluded)
        )
    else:
        result["hypergeometric_p_upper"] = None
    return result


def _category_repair_rates(records: list[dict]) -> dict[str, dict[str, Any]]:
    """Repair success rate per initial failure category (a reference signal)."""
    grouped: dict[str, list[int]] = {}
    for record in records:
        category = record.get("initial_failure_category") or "unknown"
        grouped.setdefault(category, []).append(int(record["final_passed"]))
    return {
        category: {
            "n": len(values),
            "repaired": sum(values),
            "repair_rate": sum(values) / len(values),
        }
        for category, values in sorted(grouped.items())
    }


def build_diagnosis(
    source_path: Path,
    source: dict,
    predictor: dict,
    repair_run: dict,
    *,
    sample_index: int = 0,
    smart_coverage: float = 0.75,
    feature_set: str = "self_consistency_plus_execution",
    model_type: str = "logistic",
    seed: int = 42,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    """Score the stored gate probabilities against both decision targets.

    Args:
        source_path: Path of the generation run, used for pool validation.
        source: Loaded generation run JSON.
        predictor: Loaded predictor JSON carrying out-of-fold probabilities.
        repair_run: Loaded Repair-All trace pool.
        sample_index: Source sample index forming the cohort (Week 7 uses 0).
        smart_coverage: Gate coverage whose excluded tail is tested.
        feature_set: Predictor feature set to read.
        model_type: Predictor model to read.
        seed: Seed for permutation and bootstrap resampling.
        n_permutations: Label shuffles per target.
        n_bootstrap: Resamples per target.
    Returns:
        Result dict ready for save_results().
    """
    scored = align_oof_scores(source, predictor, feature_set, model_type)
    cohort = select_cohort(scored, sample_index=sample_index)
    repair_by_coordinate = validate_repair_pool(
        source_path, cohort, repair_run, sample_index
    )

    selection_scores = np.array([row["p_pass_oof"] for row in cohort], dtype=float)
    selection_labels = np.array(
        [int(row["initial_passed"]) for row in cohort], dtype=int
    )

    failing = [row for row in cohort if not row["initial_passed"]]
    allocation_scores = np.array([row["p_pass_oof"] for row in failing], dtype=float)
    records = [
        repair_by_coordinate[(row["task_id"], row["sample_index"])] for row in failing
    ]
    allocation_labels = np.array(
        [int(record["final_passed"]) for record in records], dtype=int
    )
    oracle_labels = np.array(
        [
            int(record.get("oracle_candidate_passed", record["final_passed"]))
            for record in records
        ],
        dtype=int,
    )

    targets = [
        score_target(
            selection_scores,
            selection_labels,
            name="initial_passed",
            seed=seed,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
        ),
        score_target(
            allocation_scores,
            allocation_labels,
            name="repair_success",
            seed=seed,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
        ),
    ]
    # oracle_candidate_passed differs from final_passed only when the ranker
    # missed an available passing candidate; Week 7's oracle gap was zero, so
    # this arm is a guard against reading a ranker artefact as a gate result.
    if not np.array_equal(oracle_labels, allocation_labels):
        targets.append(
            score_target(
                allocation_scores,
                oracle_labels,
                name="repair_oracle_available",
                seed=seed,
                n_permutations=n_permutations,
                n_bootstrap=n_bootstrap,
            )
        )

    return {
        "analysis": "joint_gate_target_diagnosis",
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
            "seed": seed,
            "n_permutations": n_permutations,
            "n_bootstrap": n_bootstrap,
        },
        "cohort": {
            "candidates": len(cohort),
            "initially_passing": int(selection_labels.sum()),
            "initially_failing": len(failing),
            "repaired": int(allocation_labels.sum()),
        },
        "gate_exclusion": gate_exclusion_test(
            failing,
            repair_by_coordinate,
            select_risk_gate(cohort, math.ceil(smart_coverage * len(cohort))),
        ),
        "targets": targets,
        "target_gap": (
            targets[0]["roc_auc"] - targets[1]["roc_auc"]
            if targets[0]["roc_auc"] is not None and targets[1]["roc_auc"] is not None
            else None
        ),
        "per_initial_failure_category": _category_repair_rates(records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the Smart Repair gate against selection and allocation targets"
    )
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--predictor-results", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--repair-results", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--smart-coverage", type=float, default=0.75)
    parser.add_argument("--feature-set", default="self_consistency_plus_execution")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--prefix", default="analysis_week8_gate_target")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnosis = build_diagnosis(
        args.source_results,
        load_json(args.source_results),
        load_json(args.predictor_results),
        load_json(args.repair_results),
        sample_index=args.sample_index,
        smart_coverage=args.smart_coverage,
        feature_set=args.feature_set,
        model_type=args.model_type,
        seed=args.seed,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
    )

    print(f"cohort: {diagnosis['cohort']}")
    for target in diagnosis["targets"]:
        if target["roc_auc"] is None:
            print(
                f"  {target['target']:<26} undefined "
                f"({target['n_positive']}/{target['n']} positive)"
            )
            continue
        low, high = target["bootstrap_95"]
        print(
            f"  {target['target']:<26} AUC={target['roc_auc']:.4f}  "
            f"p={target['permutation_p']:.3f}  95% CI=[{low:.3f}, {high:.3f}]  "
            f"({target['n_positive']}/{target['n']} positive)"
        )
    if diagnosis["target_gap"] is not None:
        print(f"  selection - allocation gap: {diagnosis['target_gap']:.4f}")
    exclusion = diagnosis["gate_exclusion"]
    print(
        f"  gate-excluded tail: {exclusion['successes_excluded_observed']} of "
        f"{exclusion['n_successes']} successes in the {exclusion['n_excluded_by_gate']} "
        f"discarded coordinates (expected "
        f"{exclusion['successes_excluded_expected']:.2f}, hypergeometric "
        f"p={exclusion['hypergeometric_p_upper']:.4f})"
    )

    path = save_results(diagnosis, prefix=args.prefix)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
