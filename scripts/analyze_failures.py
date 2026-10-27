"""Summarize execution failure types from a pipeline results JSON file.

Student 2 owns this helper. It does not run generation or repair; it only
reads the JSON produced by scripts/run_pipeline.py and turns the per-sample
failure categories into a compact, reproducible failure analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _pct(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def summarize_failures(results: dict[str, Any]) -> dict[str, Any]:
    """Aggregate before/after repair failure categories from pipeline results."""
    initial_failures: Counter[str] = Counter()
    repaired_failures: Counter[str] = Counter()
    repair_transitions: Counter[str] = Counter()
    per_task: list[dict[str, Any]] = []

    total_samples = 0
    initially_passing = 0
    repair_attempts = 0
    successful_repairs = 0

    for problem in results.get("per_problem", []):
        task_id = problem.get("task_id", "")
        task_initial_failures: Counter[str] = Counter()
        task_successful_repairs = 0
        task_repair_attempts = 0

        for sample in problem.get("samples", []):
            total_samples += 1
            passed = bool(sample.get("passed"))
            if passed:
                initially_passing += 1
                continue

            initial_category = sample.get("failure_category") or "unknown"
            initial_failures[initial_category] += 1
            task_initial_failures[initial_category] += 1

            if "repaired_passed" not in sample:
                continue

            task_repair_attempts += 1
            repair_attempts += 1
            repaired_passed = bool(sample.get("repaired_passed"))
            repaired_category = (
                ""
                if repaired_passed
                else sample.get("repaired_failure_category") or "unknown"
            )
            if repaired_passed:
                successful_repairs += 1
                task_successful_repairs += 1
            else:
                repaired_failures[repaired_category] += 1
            repair_transitions[
                f"{initial_category}->{repaired_category or 'pass'}"
            ] += 1

        per_task.append(
            {
                "task_id": task_id,
                "initial_failures": dict(sorted(task_initial_failures.items())),
                "repair_attempts": task_repair_attempts,
                "successful_repairs": task_successful_repairs,
            }
        )

    initially_failing = total_samples - initially_passing
    return {
        "run_id": results.get("run_id", ""),
        "total_samples": total_samples,
        "initially_passing": initially_passing,
        "initially_failing": initially_failing,
        "initial_pass_rate": _pct(initially_passing, total_samples),
        "repair_attempts": repair_attempts,
        "successful_repairs": successful_repairs,
        "repair_success_rate": _pct(successful_repairs, repair_attempts),
        "initial_failure_counts": dict(sorted(initial_failures.items())),
        "repaired_failure_counts": dict(sorted(repaired_failures.items())),
        "repair_transitions": dict(sorted(repair_transitions.items())),
        "per_task": per_task,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    """Format a failure summary as a small Markdown report."""
    lines = [
        f"# Failure Analysis: {summary['run_id'] or 'unknown run'}",
        "",
        "## Summary",
        "",
        f"- Total samples: {summary['total_samples']}",
        f"- Initially passing: {summary['initially_passing']}",
        f"- Initially failing: {summary['initially_failing']}",
        f"- Initial pass rate: {summary['initial_pass_rate']:.2%}",
        f"- Repair attempts: {summary['repair_attempts']}",
        f"- Successful repairs: {summary['successful_repairs']}",
        f"- Repair success rate: {summary['repair_success_rate']:.2%}",
        "",
        "## Initial Failure Types",
        "",
    ]
    initial_counts = summary["initial_failure_counts"]
    if initial_counts:
        lines.extend(f"- {name}: {count}" for name, count in initial_counts.items())
    else:
        lines.append("- None")

    lines.extend(["", "## Repair Transitions", ""])
    transitions = summary["repair_transitions"]
    if transitions:
        lines.extend(f"- {name}: {count}" for name, count in transitions.items())
    else:
        lines.append("- No repair transition data")

    return "\n".join(lines) + "\n"


def load_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pipeline failure types")
    parser.add_argument("results_json", type=Path, help="Path to a results/*.json file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of Markdown",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = summarize_failures(load_results(args.results_json))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_markdown(summary), end="")


if __name__ == "__main__":
    main()
