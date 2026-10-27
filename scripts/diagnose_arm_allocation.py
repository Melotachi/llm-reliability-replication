"""Week 12. Does the repair *arm* decide whether P(pass) can allocate?

Objective 3 spent Objective 1's out-of-fold P(pass) as a repair gate and lost to
a budget-matched random gate on both cohorts. The shipped diagnosis is
correlational -- "predicting failure is not predicting repairability" -- and it
was measured on exactly one repair arm, the traceback-context (`error`) one,
because that is the only arm Weeks 7-8 had built.

Week 9 later built two more arms over the *identical* coordinates (`code_only`,
and `none`, which resamples with no failure context at all). This script scores
the same stored probabilities against all of them, which turns a single number
into a manipulation: as the failing program is removed from the repair prompt,
does the allocation signal survive?

It also ships the control Objective 3 never had. Every cohort stores N=10
samples per problem and the joint experiment uses only sample 0, so "spend a
model call drawing another sample" replays exactly, for free, by revealing
stored samples 1..k. That is the repair-versus-resampling comparison the
self-repair literature asks for (Olausson et al., 2023, `pass@t`), and this
project had never run it.

Two measurement hazards are handled explicitly rather than assumed away.

1. **Call counts are not comparable as stored.** The Week 7/8 pools ran
   `stop_on_pass=True` and therefore stopped calling the model mid candidate
   list; every Week 9 arm ran `stop_on_pass=False` and always spent its full
   budget. Candidate seeds are index-deterministic, so the stop-on-pass count is
   reconstructable exactly from the stored candidate list, and
   `renormalize_stop_on_pass` does that. The reconstruction is *checked*, not
   trusted: the `error` arm must reproduce the Week 8 pool's 368 calls with a
   per-coordinate delta of zero.
2. **The resample control changes which model is called.** Revealed samples come
   from the generator (codegen-350M-mono); repair calls go to the repairer
   (Qwen2.5-Coder-1.5B-Instruct). A call is not interchangeable currency across
   that gap, so the control is reported as calls to the base model against calls
   to the repairer, never merged into one budget axis.

Usage:
    python scripts/diagnose_arm_allocation.py --cohort humaneval
    python scripts/diagnose_arm_allocation.py --cohort mbpp
    python scripts/diagnose_arm_allocation.py --cohort both --n-seeds 200
"""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_smart_repair import (  # noqa: E402
    _coordinate,
    align_oof_scores,
    load_json,
    select_cohort,
    summarize_condition,
    validate_repair_pool,
)
from scripts.diagnose_gate_direction import (  # noqa: E402
    DEFAULT_COVERAGES,
    budget_matched_point,
    build_analysis,
)
from scripts.diagnose_gate_target import score_target  # noqa: E402
from src.utils import get_logger, save_results, set_all_seeds  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_FIGURE_DIR = Path("reports/figures")

# Arms are ordered by how much of the failing program the repair prompt carries:
# none < code_only < error. That ordering is the manipulation, so it is fixed
# here rather than taken from the command line.
COHORTS: dict[str, dict[str, Any]] = {
    "mbpp": {
        "source": Path("results/mbpp_uncertainty_20260712_024909.json"),
        "predictor": Path("results/predictor_mbpp_20260712_212240.json"),
        "arms": [
            ("none", Path("results/mbpp_week9_arm_resample_sample0_qwen15b.json")),
            (
                "code_only",
                Path("results/mbpp_week9_arm_code_only_sample0_qwen15b.json"),
            ),
            ("error", Path("results/mbpp_week7_repair_all_sample0_qwen15b.json")),
        ],
        # The shipped Objective 3 pool, used only as the reproduction guard.
        "shipped_arm": "error",
        "shipped_expectations": {"initial_passed": 0.944, "repair_success": 0.518},
    },
    "humaneval": {
        "source": Path("results/humaneval_uncertainty_merged164_20260711.json"),
        "predictor": Path("results/predictor_humaneval_20260712_215312.json"),
        "arms": [
            ("none", Path("results/humaneval_week9_arm_resample_sample0_qwen15b.json")),
            (
                "code_only",
                Path("results/humaneval_week9_arm_code_only_sample0_qwen15b.json"),
            ),
            (
                "error",
                Path("results/humaneval_week8_repair_all_sample0_qwen15b.json"),
            ),
        ],
        "shipped_arm": "error",
        "shipped_expectations": {"initial_passed": 0.947, "repair_success": 0.706},
    },
}


def renormalize_stop_on_pass(repair_run: dict) -> tuple[dict, dict[str, Any]]:
    """Rewrite each record's call count as if generation had stopped on a pass.

    The Week 9 arms generated every candidate unconditionally; the Week 7/8 pools
    stopped at the first passing candidate. Candidate seeds are index
    deterministic, so the same candidate list exists under both policies and the
    stop-on-pass cost is the index of the first passing candidate plus one.
    Rewriting the field (rather than the analysis) keeps every downstream helper
    -- `summarize_condition` and everything built on it -- unchanged.

    Args:
        repair_run: A loaded repair pool. Not mutated.
    Returns:
        (renormalized pool, report) where the report carries the stored and
        recomputed totals and the per-coordinate delta histogram. A pool that
        already ran stop-on-pass is returned with an all-zero histogram, which
        is the check that the reconstruction is faithful.
    """
    renormalized = copy.deepcopy(repair_run)
    deltas: dict[int, int] = {}
    stored_total = 0
    recomputed_total = 0
    for record in renormalized["records"]:
        stored = int(record.get("repair_model_calls", 0))
        recomputed = _stop_on_pass_calls(record)
        stored_total += stored
        recomputed_total += recomputed
        delta = recomputed - stored
        deltas[delta] = deltas.get(delta, 0) + 1
        record["repair_model_calls"] = recomputed
        record["repair_model_calls_as_stored"] = stored
    renormalized.setdefault("summary", {})
    renormalized["summary"]["repair_model_calls_as_stored"] = stored_total
    renormalized["summary"]["repair_model_calls"] = recomputed_total
    report = {
        "stored_calls": stored_total,
        "renormalized_calls": recomputed_total,
        "stop_on_pass_as_stored": bool(
            repair_run.get("config", {}).get("stop_on_pass", False)
        ),
        "per_coordinate_delta_histogram": {
            str(key): value for key, value in sorted(deltas.items())
        },
        "n_records": len(renormalized["records"]),
    }
    return renormalized, report


def _stop_on_pass_calls(record: dict) -> int:
    """Model calls this coordinate would have cost under stop-on-pass."""
    total = 0
    for iteration in record.get("trace", []):
        candidates = iteration.get("candidates") or []
        if not candidates:
            continue
        used = len(candidates)
        for index, candidate in enumerate(candidates):
            if candidate.get("passed"):
                used = index + 1
                break
        total += used
    return total


def resample_control(
    source: dict, cohort: list[dict], *, max_k: int = 9
) -> dict[str, Any]:
    """Replay "spend a call drawing another base-model sample" from stored data.

    Objective 3 uses `sample_index = 0` only, but the source run stored N=10
    samples per problem. Revealing samples 1..k is therefore an exact, zero-cost
    replay of a resampling policy at a known call cost, and it is the control the
    joint experiment never shipped.

    Args:
        source: The uncertainty run holding all N samples per problem.
        cohort: Scored cohort rows (one per candidate at the chosen sample index).
        max_k: Largest number of extra samples to reveal.
    Returns:
        One row per k with added passes, base-model calls, and calls per gain.
        `added_passes` counts initially-failing coordinates rescued, so k=0 is
        zero by construction and matches the Baseline row.
    """
    passed_by_task = {
        problem["task_id"]: [bool(sample["passed"]) for sample in problem["samples"]]
        for problem in source["per_problem"]
    }
    failing = [row for row in cohort if not row["initial_passed"]]
    rows = []
    for k in range(0, max_k + 1):
        added = sum(
            1 for row in failing if any(passed_by_task[row["task_id"]][1 : 1 + k])
        )
        calls = k * len(failing)
        rows.append(
            {
                "k_extra_samples": k,
                "added_passes": added,
                "base_model_calls": calls,
                "calls_per_added_pass": (calls / added) if added else None,
            }
        )
    return {
        "n_failing_coordinates": len(failing),
        "max_k": max_k,
        "calls_go_to": source.get("config", {}).get("model"),
        "rows": rows,
    }


def budget_matched_resample(
    control: dict[str, Any], reference_calls: int
) -> dict[str, Any]:
    """Bracket the resample curve around a repair budget.

    The curve is a step function in k, so a repair budget almost never lands on a
    grid point. Returning the rows on either side is honest where interpolating a
    single number is not.

    Args:
        control: Output of resample_control().
        reference_calls: Repair-model calls the arm actually spent.
    Returns:
        The largest affordable row and the smallest unaffordable one (either may
        be None at the ends of the sweep).
    """
    affordable = [
        row for row in control["rows"] if row["base_model_calls"] <= reference_calls
    ]
    over = [row for row in control["rows"] if row["base_model_calls"] > reference_calls]
    return {
        "reference_repair_calls": int(reference_calls),
        "largest_affordable": max(
            affordable, key=lambda row: row["base_model_calls"], default=None
        ),
        "smallest_unaffordable": min(
            over, key=lambda row: row["base_model_calls"], default=None
        ),
    }


def score_arm(
    source_path: Path,
    source: dict,
    predictor: dict,
    repair_run: dict,
    *,
    sample_index: int,
    smart_coverage: float,
    n_seeds: int,
    first_seed: int,
    feature_set: str,
    model_type: str,
    seed: int,
    n_permutations: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Score one repair arm on both targets, the mechanism control and the sweep.

    Args:
        source_path: Path of the uncertainty run (used by the pool guards).
        source: Loaded uncertainty run.
        predictor: Loaded predictor run holding the out-of-fold probabilities.
        repair_run: A repair pool, already renormalized to stop-on-pass.
        sample_index: Candidate index the cohort is built from.
        smart_coverage: Gate coverage of the shipped Smart Repair policy.
        n_seeds: Random-gate draws averaged per coverage.
        first_seed: Lowest random-gate seed.
        feature_set: Predictor arm supplying the probabilities.
        model_type: Predictor model supplying the probabilities.
        seed: Seed for the permutation and bootstrap generators.
        n_permutations: Label shuffles behind each permutation p-value.
        n_bootstrap: Resamples behind each interval.
    Returns:
        A result block combining the two-target diagnosis, the sibling-pass-rate
        control, the gate-direction sweep and the budget-matched inverted point.
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
    allocation_labels = np.array(
        [
            int(repair_by_coordinate[_coordinate(row)]["final_passed"])
            for row in failing
        ],
        dtype=int,
    )

    targets = {
        "initial_passed": score_target(
            selection_scores,
            selection_labels,
            name="initial_passed",
            seed=seed,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
        ),
        "repair_success": score_target(
            allocation_scores,
            allocation_labels,
            name="repair_success",
            seed=seed,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
        ),
    }

    # build_analysis already fits the free baseline -- the nested logistic of
    # repair success on P(pass) and the problem's own sibling pass rate. "Beats a
    # random gate" is not the claim that survives review; "beats the sibling pass
    # rate" is, so that block is surfaced rather than recomputed.
    direction = build_analysis(
        source_path,
        source,
        predictor,
        repair_run,
        sample_index=sample_index,
        smart_coverage=smart_coverage,
        coverages=DEFAULT_COVERAGES,
        n_seeds=n_seeds,
        first_seed=first_seed,
        feature_set=feature_set,
        model_type=model_type,
    )
    repair_all, _ = summarize_condition(
        "repair_all",
        cohort,
        {_coordinate(row) for row in cohort},
        repair_by_coordinate,
    )
    return {
        "targets": targets,
        "mechanism_regression": direction["mechanism_regression"],
        "arms_at_smart_coverage": direction["arms_at_smart_coverage"],
        "gate_direction_sweep": direction["direction_sweep"],
        "budget_matched_inverted": budget_matched_point(
            direction["direction_sweep"], repair_all["repair_model_calls"]
        ),
        "repair_all": {
            "added_passes": repair_all["added_passes"],
            "repair_model_calls": repair_all["repair_model_calls"],
            "attempted_repairs": repair_all["attempted_repairs"],
            "repair_success_rate": repair_all["repair_success_rate"],
        },
    }


def build_cohort_analysis(
    cohort_name: str,
    spec: dict[str, Any],
    *,
    sample_index: int = 0,
    smart_coverage: float = 0.75,
    n_seeds: int = 200,
    first_seed: int = 0,
    feature_set: str = "self_consistency_plus_execution",
    model_type: str = "logistic",
    seed: int = 42,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
    max_k: int = 9,
) -> dict[str, Any]:
    """Run every arm of one cohort plus the shared resample control."""
    source = load_json(spec["source"])
    predictor = load_json(spec["predictor"])
    scored = align_oof_scores(source, predictor, feature_set, model_type)
    cohort = select_cohort(scored, sample_index=sample_index)

    control = resample_control(source, cohort, max_k=max_k)

    arms: dict[str, Any] = {}
    for arm_name, pool_path in spec["arms"]:
        pool = load_json(pool_path)
        renormalized, renorm_report = renormalize_stop_on_pass(pool)
        result = score_arm(
            spec["source"],
            source,
            predictor,
            renormalized,
            sample_index=sample_index,
            smart_coverage=smart_coverage,
            n_seeds=n_seeds,
            first_seed=first_seed,
            feature_set=feature_set,
            model_type=model_type,
            seed=seed,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
        )
        result["arm"] = arm_name
        result["pool_file"] = str(pool_path)
        result["pool_run_id"] = pool.get("run_id")
        result["context_mode_as_stored"] = pool.get("config", {}).get("context_mode")
        result["device_resolved"] = pool.get("config", {}).get("device_resolved")
        result["dtype_resolved"] = pool.get("config", {}).get("dtype_resolved")
        result["selection_sha256"] = pool.get("config", {}).get("selection_sha256")
        result["call_renormalization"] = renorm_report
        result["resample_control_at_this_budget"] = budget_matched_resample(
            control, result["repair_all"]["repair_model_calls"]
        )
        arms[arm_name] = result

    guards = _run_guards(cohort_name, spec, arms)
    return {
        "analysis": "week12_arm_allocation",
        "cohort_name": cohort_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "source_results": str(spec["source"]),
            "source_run_id": source.get("run_id"),
            "predictor_results": str(spec["predictor"]),
            "predictor_feature_set": feature_set,
            "predictor_model_type": model_type,
            "probability_kind": "out_of_fold_raw_rank_only",
            "sample_index": sample_index,
            "smart_coverage": smart_coverage,
            "n_seeds": n_seeds,
            "first_seed": first_seed,
            "seed": seed,
            "n_permutations": n_permutations,
            "n_bootstrap": n_bootstrap,
            "arm_order_note": (
                "arms are ordered by how much of the failing program the repair "
                "prompt carries: none < code_only < error"
            ),
        },
        "cohort": {
            "candidates": len(cohort),
            "initially_failing": sum(1 for row in cohort if not row["initial_passed"]),
        },
        "resample_control": control,
        "arms": arms,
        "guards": guards,
    }


def _run_guards(
    cohort_name: str, spec: dict[str, Any], arms: dict[str, Any]
) -> dict[str, Any]:
    """Check the reproduction, renormalization and cohort-identity guards.

    Every guard is reported rather than asserted so one failure does not hide the
    other results, but each carries an explicit pass flag.
    """
    shipped = arms.get(spec["shipped_arm"], {})
    expectations = spec["shipped_expectations"]
    observed = {
        name: shipped.get("targets", {}).get(name, {}).get("roc_auc")
        for name in expectations
    }
    reproduction = {
        "arm": spec["shipped_arm"],
        "expected": expectations,
        "observed": observed,
        "passed": all(
            observed[name] is not None
            and abs(observed[name] - expectations[name]) < 5e-4
            for name in expectations
        ),
    }

    renormalization = {}
    for arm_name, result in arms.items():
        report = result["call_renormalization"]
        histogram = report["per_coordinate_delta_histogram"]
        renormalization[arm_name] = {
            "stop_on_pass_as_stored": report["stop_on_pass_as_stored"],
            "stored_calls": report["stored_calls"],
            "renormalized_calls": report["renormalized_calls"],
            "per_coordinate_delta_histogram": histogram,
            # A pool that already ran stop-on-pass must reconstruct to itself.
            "passed": (
                list(histogram) == ["0"] if report["stop_on_pass_as_stored"] else True
            ),
        }

    hashes = {name: result["selection_sha256"] for name, result in arms.items()}
    distinct = {value for value in hashes.values() if value}
    return {
        "reproduction": reproduction,
        "call_renormalization": renormalization,
        "cohort_identity": {
            "selection_sha256_by_arm": hashes,
            "passed": len(distinct) <= 1,
            "note": (
                "all arms must have been generated over the same frozen "
                "coordinate selection"
            ),
        },
        "cohort_name": cohort_name,
    }


def plot_arm_allocation(analyses: list[dict[str, Any]], output: Path) -> None:
    """Allocation ROC per arm, and the resample control against Repair-All."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    order = ["none", "code_only", "error"]

    axis = axes[0]
    width = 0.36
    for offset, analysis in enumerate(analyses):
        values, errors, positions = [], [[], []], []
        for index, arm in enumerate(order):
            target = (
                analysis["arms"].get(arm, {}).get("targets", {}).get("repair_success")
            )
            if not target or target.get("roc_auc") is None:
                continue
            auc = target["roc_auc"]
            low, high = target["bootstrap_95"]
            positions.append(index + (offset - 0.5) * width)
            values.append(auc)
            errors[0].append(auc - low)
            errors[1].append(high - auc)
        axis.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=analysis["cohort_name"],
        )
    axis.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(["none\n(resample)", "code_only", "error\n(traceback)"])
    axis.set_ylabel("ROC-AUC, P(pass) vs repair success")
    axis.set_title("Allocation signal by repair context")
    axis.legend(fontsize=8)

    # Markers distinguish the arms; colour stays with the cohort, so the reader
    # compares each cohort's repair arms against that cohort's own resample curve.
    axis = axes[1]
    arm_markers = {"none": "*", "code_only": "s", "error": "^"}
    for analysis in analyses:
        control = analysis["resample_control"]
        (line,) = axis.plot(
            [row["base_model_calls"] for row in control["rows"]],
            [row["added_passes"] for row in control["rows"]],
            marker="o",
            markersize=3.5,
            label=f"{analysis['cohort_name']}: resample, base model",
        )
        for arm, marker in arm_markers.items():
            result = analysis["arms"].get(arm)
            if not result:
                continue
            axis.scatter(
                [result["repair_all"]["repair_model_calls"]],
                [result["repair_all"]["added_passes"]],
                marker=marker,
                s=170 if marker == "*" else 60,
                color=line.get_color(),
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
                label=f"{analysis['cohort_name']}: repair `{arm}`",
            )
    axis.set_xlabel("model calls -- base model and repairer are NOT the same model")
    axis.set_ylabel("added passes")
    axis.set_title("The control Objective 3 never shipped")
    axis.legend(fontsize=6.5, ncol=2, loc="lower right")

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _print_summary(analysis: dict[str, Any]) -> None:
    """Print the two rows a reader needs before opening the JSON."""
    name = analysis["cohort_name"]
    print(
        f"\n=== {name}  ({analysis['cohort']['candidates']} candidates, "
        f"{analysis['cohort']['initially_failing']} failing)"
    )
    print(
        f"{'arm':10s} {'pos':>4s} {'ROC':>6s} {'perm p':>8s} {'boot95':>16s} "
        f"{'added':>6s} {'calls':>6s}  sibling-LR p"
    )
    for arm in ["none", "code_only", "error"]:
        result = analysis["arms"].get(arm)
        if not result:
            continue
        target = result["targets"]["repair_success"]
        auc = target["roc_auc"]
        low, high = target["bootstrap_95"]
        lr = (
            result["mechanism_regression"]
            .get("likelihood_ratio", {})
            .get("p_pass_given_sibling", {})
            .get("p_value")
        )
        print(
            f"{arm:10s} {target['n_positive']:4d} "
            f"{auc:6.3f} {target['permutation_p']:8.4f} "
            f"[{low:.3f}, {high:.3f}] "
            f"{result['repair_all']['added_passes']:6d} "
            f"{result['repair_all']['repair_model_calls']:6d}  "
            f"{'n/a' if lr is None else f'{lr:.4f}'}"
        )
    guards = analysis["guards"]
    print(
        f"guards: reproduction={guards['reproduction']['passed']} "
        f"cohort_identity={guards['cohort_identity']['passed']} "
        f"renormalization="
        f"{all(g['passed'] for g in guards['call_renormalization'].values())}"
    )


def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Score the repair gate against every Week 9 repair arm"
    )
    parser.add_argument(
        "--cohort", choices=["mbpp", "humaneval", "both"], default="both"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--smart-coverage", type=float, default=0.75)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--feature-set", default="self_consistency_plus_execution")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--max-k", type=int, default=9)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--prefix", default="analysis_week12_arm_allocation")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    set_all_seeds(args.seed)

    names = ["mbpp", "humaneval"] if args.cohort == "both" else [args.cohort]
    analyses = []
    for name in names:
        LOGGER.info("scoring cohort %s", name)
        analysis = build_cohort_analysis(
            name,
            COHORTS[name],
            sample_index=args.sample_index,
            smart_coverage=args.smart_coverage,
            n_seeds=args.n_seeds,
            first_seed=args.first_seed,
            feature_set=args.feature_set,
            model_type=args.model_type,
            seed=args.seed,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            max_k=args.max_k,
        )
        analyses.append(analysis)
        _print_summary(analysis)

    figure = args.figure_dir / "week12_arm_allocation.png"
    plot_arm_allocation(analyses, figure)

    path = save_results(
        {
            "analysis": "week12_arm_allocation",
            "timestamp": datetime.now().isoformat(),
            "figure": str(figure),
            "cohorts": analyses,
        },
        prefix=args.prefix,
    )
    print(f"\nfigure: {figure}")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
