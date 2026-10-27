"""Trace one generated sample through the Student 2 repair pipeline.

This is a debugging/learning helper. It can replay a sample stored in a
results/*.json file and print the prompt, tests, initial execution result,
repair prompt, repaired code, and post-repair execution result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import classify_failure, run_code
from src.generation import generate_code, load_model
from src.repair import (
    _assemble_repaired_code,
    _build_baseline_repair_prompt,
    _build_error_context_repair_prompt,
)
from src.utils import load_dataset_split


def _print_section(title: str, body: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")
    print(body.rstrip() if body else "<empty>")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _find_problem(results: dict[str, Any], task_id: str) -> dict[str, Any]:
    for problem in results.get("per_problem", []):
        if problem.get("task_id") == task_id:
            return problem
    raise ValueError(f"Task id {task_id!r} not found in results file")


def _find_dataset_problem(dataset_name: str, task_id: str) -> dict[str, str]:
    for problem in load_dataset_split(dataset_name):
        if problem["task_id"] == task_id:
            return problem
    raise ValueError(f"Task id {task_id!r} not found in {dataset_name} dataset")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace one repair sample")
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--task-id", default="HumanEval/1")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--use-error-context",
        action="store_true",
        help="Show the Week 3 traceback-aware repair prompt.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Call the real model again instead of replaying saved repaired_code.",
    )
    parser.add_argument("--model", default="Salesforce/codegen-350M-mono")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use HuggingFace offline/cache mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    results = _load_json(args.results_json)
    result_problem = _find_problem(results, args.task_id)
    sample = result_problem["samples"][args.sample_index]
    dataset_problem = _find_dataset_problem(args.dataset, args.task_id)

    prompt = dataset_problem["prompt"]
    test_code = dataset_problem["test_code"]
    initial_code = sample["code"]

    _print_section("1. Dataset prompt sent to generation", prompt)
    _print_section("2. Dataset tests used by evaluation", test_code)
    _print_section("3. Generated candidate code before repair", initial_code)

    passed, error_message = run_code(initial_code, test_code)
    initial_summary = (
        f"passed: {passed}\n"
        f"failure_category: {classify_failure(error_message)}\n"
        f"error_message:\n{error_message or '<empty>'}"
    )
    _print_section("4. Initial execution result", initial_summary)

    repair_error_message = error_message if args.use_error_context else ""
    repair_prompt = (
        _build_error_context_repair_prompt(prompt, initial_code, repair_error_message)
        if repair_error_message.strip()
        else _build_baseline_repair_prompt(prompt, initial_code)
    )
    _print_section("5. Repair prompt sent to the model", repair_prompt)

    if args.generate:
        tokenizer, model = load_model(args.model)
        completion = generate_code(
            repair_prompt,
            tokenizer,
            model,
            temperature=args.temperature,
        )
        repaired_code = _assemble_repaired_code(prompt, completion)
        _print_section("6. Fresh model repair completion", completion)
        _print_section("7. Fresh assembled repaired code", repaired_code)
    else:
        repaired_code = sample.get("repaired_code", "")
        _print_section("6. Saved repaired_code from results JSON", repaired_code)

    repaired_passed, repaired_error = run_code(repaired_code, test_code)
    repaired_summary = (
        f"repaired_passed: {repaired_passed}\n"
        f"repaired_failure_category: {classify_failure(repaired_error)}\n"
        f"repaired_error_message:\n{repaired_error or '<empty>'}"
    )
    _print_section("8. Post-repair execution result", repaired_summary)


if __name__ == "__main__":
    main()
