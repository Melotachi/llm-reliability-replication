"""Compare Student 2 Week 6 repair robustness experiment checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_run(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _arm_name(path: Path, run: dict[str, Any]) -> str:
    config = run["config"]
    policy = config.get("duplicate_policy", "accept")
    temperature = config.get("temperature")
    return f"{path.stem}:temp={temperature}:policy={policy}"


def _record_outcomes(run: dict[str, Any]) -> dict[tuple[str, int], bool]:
    return {
        (record["task_id"], int(record["sample_index"])): bool(record["final_passed"])
        for record in run.get("records", [])
    }


def compare_runs(paths: list[Path], require_same_selection: bool = False) -> dict:
    """Build arm summaries and paired outcome counts from result JSON files."""
    if not paths:
        raise ValueError("At least one result path is required")
    loaded = [(path, load_run(path)) for path in paths]
    selection_hashes = {
        run.get("config", {}).get("selection_sha256") for _, run in loaded
    }
    if require_same_selection and len(selection_hashes) != 1:
        raise ValueError("Runs do not use the same frozen failure selection")

    arms = []
    outcomes = {}
    for path, run in loaded:
        name = _arm_name(path, run)
        config = run["config"]
        summary = run["summary"]
        arms.append(
            {
                "name": name,
                "path": str(path),
                "selection_sha256": config.get("selection_sha256"),
                "failure_split": config.get("failure_split"),
                "temperature": config.get("temperature"),
                "duplicate_policy": config.get("duplicate_policy", "accept"),
                "repair_candidates": config.get("repair_candidates"),
                "max_candidate_calls": config.get("max_candidate_calls"),
                "stop_on_pass": config.get("stop_on_pass", False),
                "completed_failures": summary.get("completed_failures", 0),
                "successful_repairs": summary.get("successful_repairs", 0),
                "repair_success_rate": summary.get("repair_success_rate", 0.0),
                "repair_success_wilson_95": summary.get("repair_success_wilson_95"),
                "repair_model_calls": summary.get("repair_model_calls", 0),
                "duplicate_candidates": summary.get("duplicate_candidates", 0),
                "duplicate_candidate_rate": summary.get(
                    "duplicate_candidate_rate", 0.0
                ),
                "unique_candidates_executed": summary.get(
                    "unique_candidates_executed", 0
                ),
                "model_calls_per_success": summary.get("model_calls_per_success"),
                "ranking_oracle_gap": summary.get("ranking_oracle_gap", 0),
                "per_initial_failure_category": summary.get(
                    "per_initial_failure_category", {}
                ),
            }
        )
        outcomes[name] = _record_outcomes(run)

    paired = []
    for left_index, left in enumerate(arms):
        for right in arms[left_index + 1 :]:
            left_outcomes = outcomes[left["name"]]
            right_outcomes = outcomes[right["name"]]
            common = sorted(left_outcomes.keys() & right_outcomes.keys())
            paired.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "common_failures": len(common),
                    "both_success": sum(
                        left_outcomes[key] and right_outcomes[key] for key in common
                    ),
                    "left_only_success": sum(
                        left_outcomes[key] and not right_outcomes[key] for key in common
                    ),
                    "right_only_success": sum(
                        right_outcomes[key] and not left_outcomes[key] for key in common
                    ),
                    "neither_success": sum(
                        not left_outcomes[key] and not right_outcomes[key]
                        for key in common
                    ),
                }
            )

    return {
        "same_selection": len(selection_hashes) == 1,
        "selection_hashes": sorted(str(value) for value in selection_hashes),
        "arms": arms,
        "paired_outcomes": paired,
    }


def format_markdown(comparison: dict) -> str:
    lines = [
        "# Repair Robustness Comparison",
        "",
        f"- Same frozen selection: {comparison['same_selection']}",
        "",
        "| Arm | Success | Calls | Duplicates | Unique executed | Calls/success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in comparison["arms"]:
        calls_per_success = arm["model_calls_per_success"]
        calls_per_success_text = (
            "n/a" if calls_per_success is None else f"{calls_per_success:.2f}"
        )
        lines.append(
            f"| {arm['name']} | {arm['successful_repairs']}/"
            f"{arm['completed_failures']} ({arm['repair_success_rate']:.1%}) | "
            f"{arm['repair_model_calls']} | {arm['duplicate_candidates']} "
            f"({arm['duplicate_candidate_rate']:.1%}) | "
            f"{arm['unique_candidates_executed']} | {calls_per_success_text} |"
        )

    if comparison["paired_outcomes"]:
        lines.extend(["", "## Paired outcomes", ""])
        for paired in comparison["paired_outcomes"]:
            lines.append(
                f"- {paired['left']} vs {paired['right']}: "
                f"both={paired['both_success']}, "
                f"left-only={paired['left_only_success']}, "
                f"right-only={paired['right_only_success']}, "
                f"neither={paired['neither_success']} "
                f"(n={paired['common_failures']})"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--require-same-selection", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_runs(
        args.results, require_same_selection=args.require_same_selection
    )
    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        print(format_markdown(comparison), end="")


if __name__ == "__main__":
    main()
