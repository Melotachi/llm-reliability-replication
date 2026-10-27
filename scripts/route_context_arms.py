"""Week 10 (joint): can Objective 1's features route between Week 9's repair arms?

Week 9 established that repair context is *uniformly* mis-applied at this
capability gap: withholding the failing program repairs 91 of 148 HumanEval
coordinates, showing it repairs 65, adding the traceback 54. But the arms are
not nested. Their union is 109, so 18 coordinates are solved only when the
failing program is in the prompt. That is the routing headroom, and Week 9
named it the follow-up: does anything *predictable before the repair call*
mark those 18?

This script answers it with re-analysis only -- no model is loaded, no
candidate is generated. Every arm's outcome is already stored per coordinate.

Three questions, in increasing strength:

1. **Rankability.** Do Objective 1's own features (the problem-level diversity
   metrics, the sample-level out-of-fold ``P(pass)``, the failing program's
   length, its failure category) rank the 18 context-only coordinates above the
   other 130? Reported as a cross-validated ROC-AUC with a permutation p, and
   as a univariate AUC per feature -- at 18 positives a single ranking column is
   the more trustworthy estimate.

2. **Payoff.** Ranked by that score, route the top k coordinates to
   ``code_only`` and the rest to ``none``, and count passes. The reference
   points are the best single arm (91) and the oracle router (109). The sweep's
   best k is selected in-sample and is reported as such; the honest number is
   the nested one, where each fold's coverage is chosen on the other folds.

3. **Control.** The same routing policy with the coordinates chosen at random,
   over many seeds -- the Week 8 control design. A router that cannot beat its
   own random control has not earned its complexity, whatever its AUC.

The population is the 148 base-failing sample-0 coordinates shared by all three
arms; ``load_arms`` asserts the arms repair byte-identical initial programs, so
the outcomes are paired. ``--label plus`` repeats everything on the EvalPlus
relabelling as a sensitivity pass.

Power caveat, load-bearing: 18 positives. This experiment can exclude a large
routing signal; it cannot establish a small one.

One feature is not out of fold with respect to *this* target. ``p_pass_oof`` is
out of fold for its own pass/fail predictor, which was fit over all 164
problems, and it is reused here as a fixed covariate while the router re-splits
the same rows. It therefore leaks mildly into the router's test folds -- which
can only inflate the reported AUC, and the reported AUC is a null, so the
direction of the flaw strengthens the conclusion rather than threatening it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_context_mode_ladder import _passed_key, load_arms  # noqa: E402
from scripts.analyze_smart_repair import align_oof_scores, load_json  # noqa: E402
from scripts.diagnose_gate_direction import sibling_pass_rates  # noqa: E402
from scripts.diagnose_gate_target import roc_auc  # noqa: E402
from src.utils import get_logger, save_results  # noqa: E402

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("results/humaneval_uncertainty_merged164_20260711.json")
DEFAULT_PREDICTOR = Path("results/predictor_humaneval_20260712_215312.json")
DEFAULT_FIGURE_DIR = Path("reports/figures")

#: Arm the router may divert to. ``none`` is the default arm (Week 9's winner).
DEFAULT_ARM = "none"
ROUTED_ARM = "code_only"

#: Problem-level metrics read from the source run's stored ``uncertainty`` dict.
UNCERTAINTY_FEATURES = (
    "lexical_diversity",
    "token_jaccard_diversity",
    "token_cosine_diversity",
    "ast_node_diversity",
    "tree_edit_distance_diversity",
    "pairwise_distance",
    "parse_failure_rate",
)

#: Failure categories one-hot encoded as router inputs (Section 9 taxonomy).
FAILURE_CATEGORIES = (
    "assertion_error",
    "name_error",
    "runtime_error",
    "syntax_error",
    "type_error",
    "timeout",
)


def build_router_rows(
    source: dict,
    predictor: dict,
    arms: dict[str, Any],
    label: str,
    *,
    sample_index: int = 0,
) -> list[dict[str, Any]]:
    """Join each repaired coordinate to its pre-repair Objective 1 features.

    Every feature here is computable before the repair call: the problem-level
    diversity metrics and the sample-level out-of-fold ``P(pass)`` come from the
    generation run, and the length and failure category describe the program
    being repaired. No arm outcome enters the feature vector.

    Args:
        source: Loaded generation run JSON (the arms' shared source).
        predictor: Loaded predictor JSON carrying sample-level OOF probabilities.
        arms: ``load_arms`` output, keyed by context mode.
        label: "base" or "plus"; picks which stored outcome is read.
        sample_index: Cohort sample index (Week 9 uses 0).
    Returns:
        One row per coordinate, carrying "features", "feature_names", each arm's
        outcome, and the derived routing targets.
    Raises:
        ValueError: If an arm coordinate has no matching source problem, or the
            default/routed arms are absent.
    """
    for mode in (DEFAULT_ARM, ROUTED_ARM):
        if mode not in arms:
            raise ValueError(f"Arm {mode!r} missing; routing needs it")
    passed_key = _passed_key(label)
    scored = align_oof_scores(source, predictor)
    by_coordinate = {
        (row["task_id"], row["sample_index"]): row
        for row in scored
        if row["sample_index"] == sample_index
    }
    uncertainty = {
        problem["task_id"]: problem.get("uncertainty", {})
        for problem in source.get("per_problem", [])
    }
    siblings = sibling_pass_rates(source, sample_index=sample_index)

    feature_names = [
        "p_pass_oof",
        "sibling_pass_rate",
        "initial_code_chars",
        *UNCERTAINTY_FEATURES,
        *[f"category_{name}" for name in FAILURE_CATEGORIES],
    ]
    rows: list[dict[str, Any]] = []
    reference = arms[DEFAULT_ARM]["records"]
    for task_id in sorted(reference, key=lambda tid: (len(tid), tid)):
        coordinate = (task_id, sample_index)
        if coordinate not in by_coordinate or task_id not in uncertainty:
            raise ValueError(f"{task_id} has no matching source coordinate")
        record = reference[task_id]
        metrics = uncertainty[task_id]
        category = record.get("initial_failure_category", "")
        values = [
            by_coordinate[coordinate]["p_pass_oof"],
            siblings.get(task_id, 0.0),
            float(len(record["initial_code"])),
            *[float(metrics.get(name, 0.0)) for name in UNCERTAINTY_FEATURES],
            *[1.0 if category == name else 0.0 for name in FAILURE_CATEGORIES],
        ]
        outcomes = {
            mode: bool(arm["records"][task_id][passed_key])
            for mode, arm in arms.items()
        }
        rows.append(
            {
                "task_id": task_id,
                "sample_index": sample_index,
                "initial_failure_category": category,
                "features": [float(value) for value in values],
                "feature_names": feature_names,
                "outcomes": outcomes,
                # The routing target: a context arm rescues this coordinate and
                # the default arm does not.
                "context_only": bool(
                    not outcomes[DEFAULT_ARM]
                    and any(
                        passed
                        for mode, passed in outcomes.items()
                        if mode != DEFAULT_ARM
                    )
                ),
                "routed_arm_only": bool(
                    outcomes[ROUTED_ARM] and not outcomes[DEFAULT_ARM]
                ),
            }
        )
    return rows


def reference_points(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm totals, the oracle union, and the headroom a router can win."""
    modes = sorted(rows[0]["outcomes"])
    per_arm = {mode: int(sum(row["outcomes"][mode] for row in rows)) for mode in modes}
    oracle = int(sum(any(row["outcomes"].values()) for row in rows))
    best_arm = max(per_arm, key=lambda mode: (per_arm[mode], mode == DEFAULT_ARM))
    return {
        "n_coordinates": len(rows),
        "per_arm_passes": per_arm,
        "best_single_arm": best_arm,
        "best_single_arm_passes": per_arm[best_arm],
        "oracle_union_passes": oracle,
        "default_arm": DEFAULT_ARM,
        "default_arm_passes": per_arm[DEFAULT_ARM],
        "routed_arm": ROUTED_ARM,
        "two_arm_oracle_passes": int(
            sum(
                row["outcomes"][DEFAULT_ARM] or row["outcomes"][ROUTED_ARM]
                for row in rows
            )
        ),
        "headroom_context_only": int(sum(row["context_only"] for row in rows)),
        "headroom_routed_arm_only": int(sum(row["routed_arm_only"] for row in rows)),
    }


def out_of_fold_scores(
    features: np.ndarray, labels: np.ndarray, *, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    """Cross-validated P(context_only) for every row, aligned to input order.

    Each coordinate is its own problem (one sample per task), so a plain
    stratified split already satisfies the project's leave-problems-out rule.

    Args:
        features: (n_rows, n_features) design matrix.
        labels: Binary routing target.
        n_splits: Stratified folds.
        seed: Split and solver seed (Constraint #4).
    Returns:
        Out-of-fold probabilities, one per row.
    """
    scores = np.zeros(len(labels), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(features, labels):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=10_000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        )
        model.fit(features[train_index], labels[train_index])
        scores[test_index] = model.predict_proba(features[test_index])[:, 1]
    return scores


def _permutation_p(
    scores: np.ndarray, labels: np.ndarray, *, n_permutations: int, seed: int
) -> float:
    """One-sided permutation p for an observed AUC, labels shuffled."""
    observed = roc_auc(scores, labels)
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    at_least = 1
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        if roc_auc(scores, shuffled) >= observed:
            at_least += 1
    return at_least / (n_permutations + 1)


def routing_sweep(
    rows: list[dict[str, Any]], scores: np.ndarray
) -> list[dict[str, Any]]:
    """Passes obtained when the k highest-scoring coordinates go to the routed arm.

    Args:
        rows: Router rows carrying each arm's outcome.
        scores: Per-row routing score; ties broken by row order.
    Returns:
        One entry per k in 0..n, with the pass count and the arms' composition.
    """
    order = sorted(range(len(rows)), key=lambda index: (-scores[index], index))
    default_passes = [bool(row["outcomes"][DEFAULT_ARM]) for row in rows]
    routed_passes = [bool(row["outcomes"][ROUTED_ARM]) for row in rows]
    total = int(sum(default_passes))
    sweep = [{"k": 0, "passes": total, "gained": 0, "lost": 0}]
    gained = lost = 0
    for position, index in enumerate(order, start=1):
        if routed_passes[index] and not default_passes[index]:
            total += 1
            gained += 1
        elif default_passes[index] and not routed_passes[index]:
            total -= 1
            lost += 1
        sweep.append({"k": position, "passes": total, "gained": gained, "lost": lost})
    return sweep


def nested_routing(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Route with both the score and the coverage chosen out of fold.

    The sweep's best k is selected on the same rows it is scored on and is
    optimistic. Here each fold's coverage *fraction* is chosen on the training
    rows only, then applied to the held-out rows, so the reported total contains
    no in-sample choice at all.

    Args:
        rows: Router rows.
        features: Design matrix aligned to rows.
        labels: Routing target.
        n_splits: Stratified folds.
        seed: Split and solver seed.
    Returns:
        Total passes, the per-fold coverages chosen, and the default-arm total.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    total = 0
    baseline = 0
    coverages: list[float] = []
    for train_index, test_index in splitter.split(features, labels):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=10_000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        )
        model.fit(features[train_index], labels[train_index])
        train_rows = [rows[index] for index in train_index]
        train_scores = model.predict_proba(features[train_index])[:, 1]
        train_sweep = routing_sweep(train_rows, train_scores)
        best = max(train_sweep, key=lambda entry: (entry["passes"], -entry["k"]))
        coverage = best["k"] / len(train_rows)
        coverages.append(coverage)

        test_rows = [rows[index] for index in test_index]
        test_scores = model.predict_proba(features[test_index])[:, 1]
        test_sweep = routing_sweep(test_rows, test_scores)
        k = min(len(test_rows), int(round(coverage * len(test_rows))))
        total += test_sweep[k]["passes"]
        baseline += test_sweep[0]["passes"]

    # The folds partition the cohort, so the per-fold default totals must sum to
    # the cohort default -- otherwise the routed total is being compared against
    # a different denominator and the headline difference is meaningless.
    default_passes = int(sum(row["outcomes"][DEFAULT_ARM] for row in rows))
    if baseline != default_passes:
        raise ValueError(
            "Nested folds do not partition the cohort: per-fold default arm sums "
            f"to {baseline}, cohort default is {default_passes}"
        )
    return {
        "passes": int(total),
        "fold_default_arm_passes": int(baseline),
        "fold_coverages": [float(value) for value in coverages],
        "default_arm_passes": default_passes,
    }


def random_routing_control(
    rows: list[dict[str, Any]],
    k: int,
    *,
    n_seeds: int,
    seed: int,
    target: int | None = None,
) -> dict[str, Any]:
    """Route k coordinates chosen uniformly at random, over many seeds.

    The Week 8 control design: a policy that cannot beat a budget-matched
    random draw of the same size has not earned its complexity.

    Args:
        rows: Router rows.
        k: Number of coordinates diverted to the routed arm.
        n_seeds: Draws.
        seed: Base seed.
        target: Pass count to compare the draws against (the router's). When
            given, the result carries the fraction of draws that match or beat
            it.
    Returns:
        Mean, standard deviation, range and quantiles of the pass count, plus
        the dominance fractions when `target` is given.
    """
    if not 0 <= k <= len(rows):
        raise ValueError("k must be between zero and the cohort size")
    rng = np.random.default_rng(seed)
    default_passes = np.array(
        [bool(row["outcomes"][DEFAULT_ARM]) for row in rows], dtype=bool
    )
    routed_passes = np.array(
        [bool(row["outcomes"][ROUTED_ARM]) for row in rows], dtype=bool
    )
    totals = np.empty(n_seeds, dtype=int)
    for draw in range(n_seeds):
        chosen = rng.choice(len(rows), size=k, replace=False)
        outcome = default_passes.copy()
        outcome[chosen] = routed_passes[chosen]
        totals[draw] = int(outcome.sum())
    control = {
        "k": int(k),
        "n_seeds": int(n_seeds),
        "mean": float(totals.mean()),
        "std": float(totals.std(ddof=1)) if n_seeds > 1 else 0.0,
        "min": int(totals.min()),
        "max": int(totals.max()),
        "quantiles": {
            "p05": float(np.quantile(totals, 0.05)),
            "p50": float(np.quantile(totals, 0.50)),
            "p95": float(np.quantile(totals, 0.95)),
        },
    }
    if target is not None:
        control["target_passes"] = int(target)
        control["fraction_matching_or_beating_target"] = float(
            np.mean(totals >= target)
        )
        control["fraction_beating_target"] = float(np.mean(totals > target))
    return control


def category_router(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The zero-feature baseline: route whole failure categories, chosen in sample.

    Week 9 read the ladder's category table as a hint that ``runtime_error``
    might favour the context arms. This scores that hint as a policy. It is
    in-sample by construction -- the categories are chosen on the same rows --
    so it is an upper bound on what a category rule could deliver, and it is
    reported as such.
    """
    categories = sorted({row["initial_failure_category"] for row in rows})
    chosen: list[str] = []
    total = int(sum(row["outcomes"][DEFAULT_ARM] for row in rows))
    per_category = {}
    for category in categories:
        members = [row for row in rows if row["initial_failure_category"] == category]
        default = sum(row["outcomes"][DEFAULT_ARM] for row in members)
        routed = sum(row["outcomes"][ROUTED_ARM] for row in members)
        per_category[category] = {
            "n": len(members),
            "default_arm_passes": int(default),
            "routed_arm_passes": int(routed),
        }
        if routed > default:
            chosen.append(category)
            total += routed - default
    return {
        "per_category": per_category,
        "categories_routed": chosen,
        "passes_in_sample": int(total),
    }


def univariate_aucs(
    rows: list[dict[str, Any]], features: np.ndarray, labels: np.ndarray
) -> list[dict[str, Any]]:
    """Per-feature ranking AUC against the routing target, most extreme first."""
    names = rows[0]["feature_names"]
    scored = []
    for index, name in enumerate(names):
        column = features[:, index]
        if float(np.ptp(column)) == 0.0:
            continue
        auc = roc_auc(column, labels)
        scored.append(
            {"feature": name, "auc": float(auc), "abs_deviation": abs(auc - 0.5)}
        )
    return sorted(scored, key=lambda entry: -entry["abs_deviation"])


def build_analysis(
    source_path: Path,
    predictor_path: Path,
    arm_paths: list[Path],
    *,
    label: str = "base",
    sample_index: int = 0,
    n_splits: int = 5,
    n_permutations: int = 1000,
    n_seeds: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Run every routing question and return one persistable analysis dict."""
    source = load_json(source_path)
    predictor = load_json(predictor_path)
    arms = load_arms(arm_paths, label)
    rows = build_router_rows(source, predictor, arms, label, sample_index=sample_index)
    references = reference_points(rows)

    features = np.array([row["features"] for row in rows], dtype=float)
    labels = np.array([int(row["context_only"]) for row in rows], dtype=int)
    n_positive = int(labels.sum())
    if n_positive < n_splits:
        raise ValueError(
            f"Only {n_positive} routing positives; cannot run {n_splits} folds"
        )

    scores = out_of_fold_scores(features, labels, n_splits=n_splits, seed=seed)
    auc = float(roc_auc(scores, labels))
    permutation_p = _permutation_p(
        scores, labels, n_permutations=n_permutations, seed=seed
    )
    sweep = routing_sweep(rows, scores)
    best = max(sweep, key=lambda entry: (entry["passes"], -entry["k"]))
    control = random_routing_control(
        rows, best["k"], n_seeds=n_seeds, seed=seed, target=best["passes"]
    )
    nested = nested_routing(rows, features, labels, n_splits=n_splits, seed=seed)

    return {
        "analysis": "week10_context_arm_router",
        "label": label,
        "source_results_file": str(source_path),
        "predictor_file": str(predictor_path),
        "arm_files": {mode: arm["path"] for mode, arm in arms.items()},
        "config": {
            "sample_index": sample_index,
            "n_splits": n_splits,
            "n_permutations": n_permutations,
            "n_seeds": n_seeds,
            "seed": seed,
            "default_arm": DEFAULT_ARM,
            "routed_arm": ROUTED_ARM,
            "feature_names": rows[0]["feature_names"],
        },
        "reference_points": references,
        "target": {
            "name": "context_only",
            "n_positive": n_positive,
            "n_rows": len(rows),
            "positives": [row["task_id"] for row in rows if row["context_only"]],
        },
        "rankability": {
            "cv_roc_auc": auc,
            "permutation_p": float(permutation_p),
            "univariate": univariate_aucs(rows, features, labels),
        },
        "routing_sweep": sweep,
        "best_in_sample": {
            **best,
            "note": "k selected on the same rows; optimistic by construction",
        },
        "random_control": control,
        "nested_routing": nested,
        "category_router": category_router(rows),
        "rows": [
            {
                "task_id": row["task_id"],
                "initial_failure_category": row["initial_failure_category"],
                "outcomes": row["outcomes"],
                "context_only": row["context_only"],
                "router_score": float(score),
            }
            for row, score in zip(rows, scores)
        ],
    }


def plot_routing(analysis: dict[str, Any], output: Path) -> None:
    """Routed passes against coverage, with the control band and both ceilings."""
    sweep = analysis["routing_sweep"]
    references = analysis["reference_points"]
    ks = [entry["k"] for entry in sweep]
    passes = [entry["passes"] for entry in sweep]

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(ks, passes, color="#245B78", linewidth=1.8, label="router (OOF score)")
    axis.axhline(
        references["default_arm_passes"],
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label=f"default arm ({references['default_arm']})",
    )
    axis.axhline(
        references["two_arm_oracle_passes"],
        color="#2C7A4B",
        linestyle=":",
        linewidth=1.4,
        label="two-arm oracle",
    )
    control = analysis["random_control"]
    axis.errorbar(
        [control["k"]],
        [control["mean"]],
        yerr=[
            [control["mean"] - control["quantiles"]["p05"]],
            [control["quantiles"]["p95"] - control["mean"]],
        ],
        fmt="o",
        color="#D05A3A",
        capsize=4,
        label=f"random control at k={control['k']} (5--95%)",
    )
    axis.set_xlabel("coordinates routed to the context arm (k)")
    axis.set_ylabel("coordinates repaired")
    axis.set_title(
        "Routing between Week 9's repair arms "
        f"({analysis['label']} label, n={references['n_coordinates']})"
    )
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--predictor", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument(
        "--arms",
        type=Path,
        nargs="+",
        required=True,
        help="The context-mode arm result JSONs (one per arm).",
    )
    parser.add_argument("--label", choices=("base", "plus"), default="base")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="analysis_week10_router")
    parser.add_argument("--figure", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = build_analysis(
        args.source,
        args.predictor,
        list(args.arms),
        label=args.label,
        sample_index=args.sample_index,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        n_seeds=args.n_seeds,
        seed=args.seed,
    )
    figure = args.figure or (
        DEFAULT_FIGURE_DIR / f"week10_context_arm_router_{args.label}.png"
    )
    plot_routing(analysis, figure)
    analysis["figure"] = str(figure)
    path = save_results(analysis, prefix=f"{args.prefix}_{args.label}")

    references = analysis["reference_points"]
    print(f"coordinates            : {references['n_coordinates']}")
    print(f"per-arm passes         : {references['per_arm_passes']}")
    print(f"two-arm oracle         : {references['two_arm_oracle_passes']}")
    print(f"routing headroom       : {references['headroom_routed_arm_only']}")
    print(
        "router CV ROC-AUC      : "
        f"{analysis['rankability']['cv_roc_auc']:.3f} "
        f"(perm p = {analysis['rankability']['permutation_p']:.4f})"
    )
    best = analysis["best_in_sample"]
    control = analysis["random_control"]
    print(
        f"best in-sample k={best['k']:<4d} : {best['passes']} passes "
        f"(+{best['gained']} / -{best['lost']})"
    )
    print(
        f"random control at k={control['k']:<4d}: "
        f"{control['mean']:.1f} +- {control['std']:.1f} "
        f"[{control['min']}, {control['max']}]"
    )
    nested = analysis["nested_routing"]
    print(
        f"nested (honest) router : {nested['passes']} passes vs "
        f"{nested['default_arm_passes']} default"
    )
    print(f"figure                 : {figure}")
    print(f"saved                  : {path}")


if __name__ == "__main__":
    main()
