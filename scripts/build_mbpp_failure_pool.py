"""Build a frozen, checkpointed MBPP failure pool for Student 2 Week 6.

Problems are shuffled once with a recorded seed so the pool is not the first
contiguous MBPP slice. One initial solution is generated per task and executed
against the stock MBPP tests. The first failures in that frozen task order form
the held-out evaluation split; a category-diverse development split is selected
from later failures, keeping policy selection separate from final evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# HuggingFace reads its offline flags while its modules are imported, so set
# them before importing the shared generation/dataset helpers.
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_iterative_repair_experiment import (
    collect_failures,
    select_diverse_failures,
    write_checkpoint,
)
from src.evaluation import classify_failure, run_code
from src.generation import load_model, multi_sample_generation
from src.utils import load_dataset_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic MBPP failure pool for repair experiments"
    )
    parser.add_argument("--evaluation-failures", type=int, default=50)
    parser.add_argument("--development-failures", type=int, default=12)
    parser.add_argument("--max-problems", type=int, default=100)
    parser.add_argument("--model", default="Salesforce/codegen-350M-mono")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only locally cached HuggingFace datasets and models.",
    )
    return parser.parse_args()


def _coordinates(failures: list[dict]) -> list[dict]:
    return [
        {"task_id": failure["task_id"], "sample_index": failure["sample_index"]}
        for failure in failures
    ]


def refresh_pool_metadata(payload: dict) -> None:
    """Refresh summary and disjoint frozen-split coordinates in-place."""
    failures = collect_failures(payload)
    evaluation_target = payload["config"]["target_evaluation_failures"]
    development_target = payload["config"]["target_development_failures"]

    evaluation = failures[:evaluation_target]
    evaluation_coordinates = {
        (failure["task_id"], failure["sample_index"]) for failure in evaluation
    }
    development_candidates = [
        failure
        for failure in failures
        if (failure["task_id"], failure["sample_index"]) not in evaluation_coordinates
    ]
    development = select_diverse_failures(
        development_candidates, limit=development_target
    )

    samples = [
        sample
        for problem in payload["per_problem"]
        for sample in problem.get("samples", [])
    ]
    passing = sum(bool(sample["passed"]) for sample in samples)
    categories = Counter(
        sample["failure_category"] for sample in samples if not sample["passed"]
    )
    target_reached = (
        len(evaluation) == evaluation_target and len(development) == development_target
    )
    payload["failure_splits"] = {
        "development": _coordinates(development),
        "evaluation": _coordinates(evaluation),
    }
    payload["summary"] = {
        "problems_processed": len(payload["per_problem"]),
        "initially_passing": passing,
        "initially_failing": len(samples) - passing,
        "initial_pass_rate": passing / len(samples) if samples else 0.0,
        "initial_failure_categories": dict(sorted(categories.items())),
        "development_failures": len(development),
        "evaluation_failures": len(evaluation),
        "target_reached": target_reached,
    }


def _validate_positive_args(args: argparse.Namespace) -> None:
    for name in ("evaluation_failures", "development_failures", "max_problems"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_problems > 500:
        raise ValueError("max-problems cannot exceed the 500-task MBPP test split")


def _expected_config(args: argparse.Namespace) -> dict:
    return {
        "dataset": "mbpp",
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_problems": args.max_problems,
        "target_development_failures": args.development_failures,
        "target_evaluation_failures": args.evaluation_failures,
        "task_order_policy": "random.Random(seed).shuffle(test_split)",
        "generation_seed_policy": "seed+shuffled_task_index",
        "samples_per_problem": 1,
    }


def _validate_resume_config(payload: dict, expected: dict) -> None:
    actual = payload.get("config", {})
    mismatches = {
        key: {"checkpoint": actual.get(key), "requested": value}
        for key, value in expected.items()
        if key != "max_problems" and actual.get(key) != value
    }
    if expected["max_problems"] < len(payload.get("per_problem", [])):
        mismatches["max_problems"] = {
            "checkpoint_records": len(payload.get("per_problem", [])),
            "requested": expected["max_problems"],
        }
    if mismatches:
        raise ValueError(f"Resume config mismatch: {mismatches}")
    payload["config"]["max_problems"] = expected["max_problems"]


def main() -> None:
    args = parse_args()
    _validate_positive_args(args)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.output or Path("results") / f"mbpp_week6_failure_pool_{timestamp}.json"
    )
    config = _expected_config(args)

    problems = load_dataset_split("mbpp")
    random.Random(args.seed).shuffle(problems)
    problems = problems[: args.max_problems]

    if output.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists: {output}. Pass --resume to continue it."
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
        _validate_resume_config(payload, config)
    else:
        if args.resume:
            raise FileNotFoundError(f"Cannot resume missing output: {output}")
        payload = {
            "run_id": output.stem,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "summary": {},
            "failure_splits": {"development": [], "evaluation": []},
            "per_problem": [],
        }
        refresh_pool_metadata(payload)
        write_checkpoint(output, payload)

    existing_task_ids = [problem["task_id"] for problem in payload["per_problem"]]
    expected_task_ids = [problem["task_id"] for problem in problems]
    if existing_task_ids != expected_task_ids[: len(existing_task_ids)]:
        raise ValueError("Checkpoint tasks are not a prefix of the frozen task order")
    refresh_pool_metadata(payload)
    if payload["summary"]["target_reached"]:
        write_checkpoint(output, payload)
        print(f"Already complete: {output}")
        print(json.dumps(payload["summary"], indent=2))
        return

    tokenizer, model = load_model(args.model)
    for task_index in range(len(payload["per_problem"]), len(problems)):
        problem = problems[task_index]
        generation_seed = args.seed + task_index
        completion = multi_sample_generation(
            problem["prompt"],
            tokenizer,
            model,
            n=1,
            temperature=args.temperature,
            seed=generation_seed,
        )[0]
        code = problem["prompt"] + completion
        passed, error_message = run_code(code, problem["test_code"])
        payload["per_problem"].append(
            {
                "task_id": problem["task_id"],
                "prompt_length": len(problem["prompt"]),
                "generation_seed": generation_seed,
                "samples": [
                    {
                        "code": code,
                        "passed": passed,
                        "error_message": error_message,
                        "failure_category": classify_failure(error_message),
                    }
                ],
            }
        )
        refresh_pool_metadata(payload)
        write_checkpoint(output, payload)
        print(
            f"[{task_index + 1}/{len(problems)}] {problem['task_id']} -> "
            f"{'PASS' if passed else classify_failure(error_message)} "
            f"(failures={payload['summary']['initially_failing']})",
            flush=True,
        )
        if payload["summary"]["target_reached"]:
            break

    refresh_pool_metadata(payload)
    write_checkpoint(output, payload)
    print(f"Saved: {output}")
    print(json.dumps(payload["summary"], indent=2))
    if not payload["summary"]["target_reached"]:
        raise RuntimeError(
            "Failure targets were not reached before max-problems; resume the same "
            "output with a larger --max-problems value"
        )


if __name__ == "__main__":
    main()
