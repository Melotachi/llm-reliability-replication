"""Analyze Student 2 Week 7 budgeted Smart Repair from shared repair traces.

The script never generates code. It aligns Student 1's persisted out-of-fold
pass probabilities with their source samples, then evaluates Baseline,
Repair-All, and risk-gated Smart Repair against one shared Student 2 repair
trace pool. Reusing the same repair outcomes removes sampling noise from the
policy comparison and avoids a second model run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import mcnemar_mid_p

DEFAULT_SOURCE = Path("results/mbpp_uncertainty_20260712_024909.json")
DEFAULT_PREDICTOR = Path("results/predictor_mbpp_20260712_212240.json")
DEFAULT_REPAIR = Path("results/mbpp_week7_repair_all_sample0_qwen15b.json")
DEFAULT_WEEK6 = Path("results/mbpp_week6_evaluation_temp08_accept.json")
DEFAULT_OUTPUT = Path("results/mbpp_week7_smart_repair_comparison.json")
DEFAULT_FIGURE_DIR = Path("reports/figures")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _stored_basename(stored_path: Any) -> str:
    """Basename of a path string that may have been written on another OS.

    The Week 6/7 repair traces were generated on Windows, so their configs
    carry `results\\file.json`. `PurePosixPath(...).name` treats that whole
    string as one filename, which made every stored trace fail validation on
    macOS and Linux. PureWindowsPath accepts both separators, so it reads
    Windows-written and POSIX-written configs alike.
    """
    return PureWindowsPath(str(stored_path)).name


def _coordinate(record: dict) -> tuple[str, int]:
    return record["task_id"], int(record["sample_index"])


def _predictor_configuration(
    predictor: dict,
    feature_set: str,
    model_type: str,
) -> dict:
    matches = [
        config
        for config in predictor.get("configurations", [])
        if config.get("task") == "sample_level"
        and config.get("feature_set") == feature_set
        and config.get("model_type") == model_type
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one sample-level predictor configuration for "
            f"feature_set={feature_set!r}, model_type={model_type!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def align_oof_scores(
    source: dict,
    predictor: dict,
    feature_set: str = "self_consistency_plus_execution",
    model_type: str = "logistic",
) -> list[dict]:
    """Attach OOF pass probabilities to source coordinates with leakage guards."""
    source_run_id = source.get("run_id")
    predictor_source_run_id = predictor.get("source_run_id")
    if predictor_source_run_id != source_run_id:
        raise ValueError(
            "Predictor/source run mismatch: "
            f"predictor={predictor_source_run_id!r}, source={source_run_id!r}"
        )

    config = _predictor_configuration(predictor, feature_set, model_type)
    probabilities = list(config.get("oof_proba", []))
    labels = list(config.get("oof_y", []))
    flattened = []
    for problem in source.get("per_problem", []):
        for sample_index, sample in enumerate(problem.get("samples", [])):
            flattened.append(
                {
                    "task_id": problem["task_id"],
                    "sample_index": sample_index,
                    "initial_passed": bool(sample["passed"]),
                    "initial_failure_category": sample.get("failure_category", ""),
                    "source_order": len(flattened),
                }
            )

    if len(probabilities) != len(flattened) or len(labels) != len(flattened):
        raise ValueError(
            "Predictor/source row-count mismatch: "
            f"source={len(flattened)}, probabilities={len(probabilities)}, "
            f"labels={len(labels)}"
        )

    source_labels = [int(record["initial_passed"]) for record in flattened]
    predictor_labels = [int(label) for label in labels]
    if predictor_labels != source_labels:
        mismatch = next(
            index
            for index, (expected, actual) in enumerate(
                zip(source_labels, predictor_labels)
            )
            if expected != actual
        )
        raise ValueError(
            "Predictor OOF labels do not align with source sample order; "
            f"first mismatch at row {mismatch}"
        )

    for record, probability in zip(flattened, probabilities):
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid OOF probability: {probability!r}")
        record["p_pass_oof"] = value
    return flattened


def select_cohort(scored_rows: list[dict], sample_index: int = 0) -> list[dict]:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    cohort = [row for row in scored_rows if row["sample_index"] == sample_index]
    task_ids = [row["task_id"] for row in cohort]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Cohort must contain at most one sample per task")
    if not cohort:
        raise ValueError(f"No source rows found for sample_index={sample_index}")
    return cohort


def select_risk_gate(cohort: list[dict], count: int) -> set[tuple[str, int]]:
    """Select the lowest P(pass) rows with source order as a stable tie-break."""
    if not 0 <= count <= len(cohort):
        raise ValueError("gate count must be between zero and cohort size")
    ranked = sorted(cohort, key=lambda row: (row["p_pass_oof"], row["source_order"]))
    return {_coordinate(row) for row in ranked[:count]}


def select_random_gate(
    cohort: list[dict], count: int, seed: int
) -> set[tuple[str, int]]:
    if not 0 <= count <= len(cohort):
        raise ValueError("gate count must be between zero and cohort size")
    indices = random.Random(seed).sample(range(len(cohort)), count)
    return {_coordinate(cohort[index]) for index in indices}


def validate_repair_pool(
    source_path: Path,
    cohort: list[dict],
    repair_run: dict,
    sample_index: int,
) -> dict[tuple[str, int], dict]:
    config = repair_run.get("config", {})
    repair_source = _stored_basename(config.get("source_results", ""))
    if repair_source != source_path.name:
        raise ValueError(
            "Repair/source file mismatch: "
            f"repair={repair_source!r}, source={source_path.name!r}"
        )
    if config.get("sample_index") != sample_index:
        raise ValueError(
            "Repair/source sample-index mismatch: "
            f"repair={config.get('sample_index')!r}, requested={sample_index}"
        )

    repair_by_coordinate = {
        _coordinate(record): record for record in repair_run.get("records", [])
    }
    if len(repair_by_coordinate) != len(repair_run.get("records", [])):
        raise ValueError("Repair results contain duplicate source coordinates")
    expected_failures = {
        _coordinate(row) for row in cohort if not row["initial_passed"]
    }
    missing = sorted(expected_failures - repair_by_coordinate.keys())
    extra = sorted(repair_by_coordinate.keys() - expected_failures)
    if missing or extra:
        raise ValueError(
            f"Repair pool must exactly cover cohort failures; missing={missing}, "
            f"extra={extra}"
        )
    return repair_by_coordinate


def _category_metrics(
    cohort: list[dict],
    gate: set[tuple[str, int]],
    repair_by_coordinate: dict[tuple[str, int], dict],
) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in cohort:
        if not row["initial_passed"]:
            grouped[row["initial_failure_category"]].append(row)

    output = {}
    for category, rows in sorted(grouped.items()):
        attempted = [row for row in rows if _coordinate(row) in gate]
        repair_records = [repair_by_coordinate[_coordinate(row)] for row in attempted]
        successes = sum(bool(record["final_passed"]) for record in repair_records)
        calls = sum(
            int(record.get("repair_model_calls", 0)) for record in repair_records
        )
        output[category] = {
            "initial_failures": len(rows),
            "attempted_repairs": len(attempted),
            "successful_repairs": successes,
            "captured_failure_rate": successes / len(rows) if rows else 0.0,
            "repair_success_rate": successes / len(attempted) if attempted else 0.0,
            "repair_model_calls": calls,
            "calls_per_success": calls / successes if successes else None,
        }
    return output


def summarize_condition(
    name: str,
    cohort: list[dict],
    gate: set[tuple[str, int]],
    repair_by_coordinate: dict[tuple[str, int], dict],
) -> tuple[dict, list[bool]]:
    final_outcomes = []
    attempted_records = []
    for row in cohort:
        coordinate = _coordinate(row)
        if row["initial_passed"]:
            final_outcomes.append(True)
        elif coordinate in gate:
            repair_record = repair_by_coordinate[coordinate]
            attempted_records.append(repair_record)
            final_outcomes.append(bool(repair_record["final_passed"]))
        else:
            final_outcomes.append(False)

    initial_passed = sum(bool(row["initial_passed"]) for row in cohort)
    successful_repairs = sum(
        bool(record["final_passed"]) for record in attempted_records
    )
    calls = sum(
        int(record.get("repair_model_calls", 0)) for record in attempted_records
    )
    duplicates = sum(
        int(record.get("duplicate_candidates", 0)) for record in attempted_records
    )
    unique_executed = sum(
        int(record.get("unique_candidates_executed", 0)) for record in attempted_records
    )
    selected_candidate_indices = Counter(
        str(record["selected_candidate_indices"][0])
        for record in attempted_records
        if record.get("final_passed") and record.get("selected_candidate_indices")
    )
    final_passed = sum(final_outcomes)
    summary = {
        "name": name,
        "candidates": len(cohort),
        "gated_candidates": len(gate),
        "gate_coverage": len(gate) / len(cohort),
        "initial_passed": initial_passed,
        "initial_accuracy": initial_passed / len(cohort),
        "attempted_repairs": len(attempted_records),
        "successful_repairs": successful_repairs,
        "repair_success_rate": (
            successful_repairs / len(attempted_records) if attempted_records else 0.0
        ),
        "repair_success_wilson_95": _wilson_interval(
            successful_repairs, len(attempted_records)
        ),
        "final_passed": final_passed,
        "final_accuracy": final_passed / len(cohort),
        "added_passes": final_passed - initial_passed,
        "repair_model_calls": calls,
        "duplicate_candidates": duplicates,
        "duplicate_candidate_rate": duplicates / calls if calls else 0.0,
        "unique_repair_candidates_executed": unique_executed,
        "initial_candidate_executions": len(gate),
        "total_execution_budget": len(gate) + unique_executed,
        "model_calls_per_added_pass": (
            calls / successful_repairs if successful_repairs else None
        ),
        "successful_selected_candidate_indices": dict(selected_candidate_indices),
        "per_initial_failure_category": _category_metrics(
            cohort, gate, repair_by_coordinate
        ),
    }
    return summary, final_outcomes


def _bh_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 1.0
    total = len(p_values)
    for rank_index in range(total - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[original_index] * total / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def paired_comparisons(outcomes: dict[str, list[bool]]) -> list[dict]:
    names = list(outcomes)
    comparisons = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_values = outcomes[left]
            right_values = outcomes[right]
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "both_pass": sum(
                        a and b for a, b in zip(left_values, right_values)
                    ),
                    "left_only_pass": sum(
                        a and not b for a, b in zip(left_values, right_values)
                    ),
                    "right_only_pass": sum(
                        b and not a for a, b in zip(left_values, right_values)
                    ),
                    "neither_pass": sum(
                        not a and not b for a, b in zip(left_values, right_values)
                    ),
                    "mcnemar_mid_p": mcnemar_mid_p(left_values, right_values),
                }
            )
    adjusted = _bh_adjust([item["mcnemar_mid_p"] for item in comparisons])
    for item, value in zip(comparisons, adjusted):
        item["mcnemar_mid_p_bh"] = value
    return comparisons


def counterfactual_stop_on_pass(run: dict) -> dict:
    """Compute exact prefix-call savings from stored ordered candidate traces."""
    actual_calls = 0
    counterfactual_calls = 0
    actual_successes = 0
    counterfactual_successes = 0
    first_passing_candidate_indices = Counter()
    for record in run.get("records", []):
        record_has_pass = False
        for step in record.get("trace", [])[1:]:
            candidates = list(step.get("candidates", []))
            actual_calls += len(candidates)
            passing = [candidate for candidate in candidates if candidate.get("passed")]
            if passing:
                first = min(passing, key=lambda candidate: candidate["candidate_index"])
                counterfactual_calls += int(first["candidate_index"]) + 1
                first_passing_candidate_indices[str(first["candidate_index"])] += 1
                record_has_pass = True
                break
            counterfactual_calls += len(candidates)
        actual_success = bool(record.get("final_passed"))
        actual_successes += int(actual_success)
        counterfactual_successes += int(record_has_pass or actual_success)

    if actual_successes != counterfactual_successes:
        raise ValueError(
            "Stored trace does not support an outcome-preserving stop-on-pass replay"
        )
    saved = actual_calls - counterfactual_calls
    return {
        "completed_failures": len(run.get("records", [])),
        "successful_repairs": actual_successes,
        "actual_model_calls": actual_calls,
        "stop_on_pass_model_calls": counterfactual_calls,
        "saved_model_calls": saved,
        "relative_call_savings": saved / actual_calls if actual_calls else 0.0,
        "actual_calls_per_success": (
            actual_calls / actual_successes if actual_successes else None
        ),
        "stop_on_pass_calls_per_success": (
            counterfactual_calls / actual_successes if actual_successes else None
        ),
        "first_passing_candidate_indices": dict(first_passing_candidate_indices),
    }


def build_analysis(
    source_path: Path,
    source: dict,
    predictor: dict,
    repair_run: dict,
    sample_index: int = 0,
    smart_coverage: float = 0.75,
    seed: int = 42,
    feature_set: str = "self_consistency_plus_execution",
    model_type: str = "logistic",
    week6_run: dict | None = None,
) -> dict:
    if not 0.0 <= smart_coverage <= 1.0:
        raise ValueError("smart_coverage must be between zero and one")
    scored = align_oof_scores(source, predictor, feature_set, model_type)
    cohort = select_cohort(scored, sample_index=sample_index)
    repair_by_coordinate = validate_repair_pool(
        source_path, cohort, repair_run, sample_index
    )

    smart_count = math.ceil(smart_coverage * len(cohort))
    all_gate = {_coordinate(row) for row in cohort}
    smart_gate = select_risk_gate(cohort, smart_count)
    random_gate = select_random_gate(cohort, smart_count, seed)
    gates = {
        "baseline": set(),
        "repair_all": all_gate,
        f"smart_repair_at_{smart_coverage:.2f}": smart_gate,
        f"random_repair_at_{smart_coverage:.2f}": random_gate,
    }

    conditions = {}
    outcomes = {}
    for name, gate in gates.items():
        conditions[name], outcomes[name] = summarize_condition(
            name, cohort, gate, repair_by_coordinate
        )
    repair_all = conditions["repair_all"]
    for condition in conditions.values():
        condition["model_call_savings_vs_repair_all"] = (
            repair_all["repair_model_calls"] - condition["repair_model_calls"]
        )
        condition["relative_model_call_savings_vs_repair_all"] = (
            (repair_all["repair_model_calls"] - condition["repair_model_calls"])
            / repair_all["repair_model_calls"]
            if repair_all["repair_model_calls"]
            else 0.0
        )
        condition["repair_all_successes_captured"] = condition["successful_repairs"]
        condition["repair_all_success_capture_rate"] = (
            condition["successful_repairs"] / repair_all["successful_repairs"]
            if repair_all["successful_repairs"]
            else 0.0
        )

    baseline_accuracy = conditions["baseline"]["final_accuracy"]
    smart_accuracy = conditions[f"smart_repair_at_{smart_coverage:.2f}"][
        "final_accuracy"
    ]
    repair_all_accuracy = repair_all["final_accuracy"]
    if not baseline_accuracy <= smart_accuracy <= repair_all_accuracy:
        raise ValueError(
            "Expected Baseline accuracy <= Smart accuracy <= Repair-All accuracy"
        )

    risk_coverage = []
    for count in range(len(cohort) + 1):
        gate = select_risk_gate(cohort, count)
        summary, _ = summarize_condition(
            f"risk_gate_{count}", cohort, gate, repair_by_coordinate
        )
        risk_coverage.append(
            {
                "gate_count": count,
                "coverage": count / len(cohort),
                "final_error_risk": 1.0 - summary["final_accuracy"],
                "final_accuracy": summary["final_accuracy"],
                "repair_model_calls": summary["repair_model_calls"],
                "total_execution_budget": summary["total_execution_budget"],
                "added_passes": summary["added_passes"],
            }
        )
    area = sum(
        (right["coverage"] - left["coverage"])
        * (left["final_error_risk"] + right["final_error_risk"])
        / 2.0
        for left, right in zip(risk_coverage, risk_coverage[1:])
    )

    successful_indices = Counter(
        str(record.get("selected_candidate_indices", [None])[0])
        for record in repair_run.get("records", [])
        if record.get("final_passed")
    )
    return {
        "analysis": "student_b_week7_smart_repair",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "source_run_id": source.get("run_id"),
            "predictor_source_run_id": predictor.get("source_run_id"),
            "predictor_feature_set": feature_set,
            "predictor_model_type": model_type,
            "probability_kind": "out_of_fold_raw_rank_only",
            "sample_index": sample_index,
            "smart_coverage": smart_coverage,
            "smart_gate_count": smart_count,
            "seed": seed,
            "shared_repair_trace_pool": True,
        },
        "cohort": {
            "candidates": len(cohort),
            "initially_passing": sum(row["initial_passed"] for row in cohort),
            "initially_failing": sum(not row["initial_passed"] for row in cohort),
            "coordinates": [
                {
                    "task_id": row["task_id"],
                    "sample_index": row["sample_index"],
                    "p_pass_oof": row["p_pass_oof"],
                    "initial_passed": row["initial_passed"],
                    "smart_gated": _coordinate(row) in smart_gate,
                    "random_gated": _coordinate(row) in random_gate,
                }
                for row in cohort
            ],
        },
        "conditions": conditions,
        "paired_comparisons": paired_comparisons(outcomes),
        "risk_coverage": risk_coverage,
        "area_under_final_risk_coverage": area,
        "repair_all_selected_candidate_indices": dict(successful_indices),
        "week6_stop_on_pass_counterfactual": (
            counterfactual_stop_on_pass(week6_run) if week6_run is not None else None
        ),
    }


def plot_risk_coverage(analysis: dict, output: Path) -> None:
    points = analysis["risk_coverage"]
    conditions = analysis["conditions"]
    smart_name = next(name for name in conditions if name.startswith("smart_repair"))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)

    axes[0].plot(
        [point["coverage"] for point in points],
        [point["final_accuracy"] for point in points],
        color="#245b78",
        linewidth=2.0,
    )
    axes[0].scatter(
        conditions[smart_name]["gate_coverage"],
        conditions[smart_name]["final_accuracy"],
        color="#d05a3a",
        zorder=3,
        label="Smart Repair@75",
    )
    axes[0].set_xlabel("Risk-gate coverage")
    axes[0].set_ylabel("Final pass rate")
    axes[0].set_title("Risk–coverage")
    axes[0].legend(frameon=False)

    axes[1].plot(
        [point["repair_model_calls"] for point in points],
        [point["final_accuracy"] for point in points],
        color="#245b78",
        linewidth=2.0,
    )
    for name, label, color in (
        ("baseline", "Baseline", "#777777"),
        (smart_name, "Smart@75", "#d05a3a"),
        ("repair_all", "Repair-All", "#2e8b57"),
    ):
        condition = conditions[name]
        axes[1].scatter(
            condition["repair_model_calls"],
            condition["final_accuracy"],
            color=color,
            zorder=3,
            label=label,
        )
    axes[1].set_xlabel("Repair model calls")
    axes[1].set_ylabel("Final pass rate")
    axes[1].set_title("Accuracy–cost frontier")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_category_efficiency(analysis: dict, output: Path) -> None:
    conditions = analysis["conditions"]
    smart_name = next(name for name in conditions if name.startswith("smart_repair"))
    repair_all = conditions["repair_all"]["per_initial_failure_category"]
    smart = conditions[smart_name]["per_initial_failure_category"]
    categories = sorted(repair_all)
    positions = list(range(len(categories)))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)

    axes[0].bar(
        [position - width / 2 for position in positions],
        [repair_all[category]["captured_failure_rate"] for category in categories],
        width,
        label="Repair-All",
        color="#2e8b57",
    )
    axes[0].bar(
        [position + width / 2 for position in positions],
        [smart[category]["captured_failure_rate"] for category in categories],
        width,
        label="Smart@75",
        color="#d05a3a",
    )
    axes[0].set_ylabel("Recovered fraction of initial failures")
    axes[0].set_title("Repair outcome by initial failure")
    axes[0].legend(frameon=False)

    axes[1].bar(
        [position - width / 2 for position in positions],
        [repair_all[category]["repair_model_calls"] for category in categories],
        width,
        label="Repair-All",
        color="#2e8b57",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        [smart[category]["repair_model_calls"] for category in categories],
        width,
        label="Smart@75",
        color="#d05a3a",
    )
    axes[1].set_ylabel("Repair model calls")
    axes[1].set_title("Call budget by initial failure")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xticks(positions, categories, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--predictor-results", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--repair-results", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--week6-results", type=Path, default=DEFAULT_WEEK6)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--smart-coverage", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-set", default="self_consistency_plus_execution")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_json(args.source_results)
    predictor = load_json(args.predictor_results)
    repair_run = load_json(args.repair_results)
    week6_run = load_json(args.week6_results) if args.week6_results else None
    analysis = build_analysis(
        args.source_results,
        source,
        predictor,
        repair_run,
        sample_index=args.sample_index,
        smart_coverage=args.smart_coverage,
        seed=args.seed,
        feature_set=args.feature_set,
        model_type=args.model_type,
        week6_run=week6_run,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    risk_path = args.figure_dir / "week7_student_b_risk_coverage.png"
    category_path = args.figure_dir / "week7_student_b_category_efficiency.png"
    plot_risk_coverage(analysis, risk_path)
    plot_category_efficiency(analysis, category_path)
    print(f"Saved: {args.output}")
    print(f"Saved: {risk_path}")
    print(f"Saved: {category_path}")
    for name, condition in analysis["conditions"].items():
        print(
            f"{name}: final={condition['final_passed']}/{condition['candidates']}, "
            f"added={condition['added_passes']}, "
            f"calls={condition['repair_model_calls']}, "
            f"saved_vs_all={condition['model_call_savings_vs_repair_all']}"
        )


if __name__ == "__main__":
    main()
