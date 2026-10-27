"""Replay real failing LLM outputs through Week 4-6 repair.

This avoids regenerating an existing HumanEval sample pool. It selects
failures across distinct tasks and failure categories, generates and ranks
repair candidates, and checkpoints the JSON output after every source failure.
Week 6 options freeze named development/evaluation splits, resume interrupted
runs, and separate model-call, duplicate, and unique-execution budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# HuggingFace reads its offline flags while its modules are imported, so set
# them before importing the shared model/dataset helpers.
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import classify_failure
from src.generation import (
    DEVICE_CHOICES,
    DTYPE_CHOICES,
    load_model,
    resolve_runtime,
)
from src.repair import REPAIR_CONTEXT_MODES, iterative_repair
from src.utils import get_logger, load_dataset_split

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay real LLM failures through iterative repair"
    )
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--n-failures", type=int, default=8)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Optional source sample index to retain before failure selection. "
        "For example, --sample-index 0 selects at most one candidate per problem "
        "from an N-sample source run (Week 7 Smart Repair cohort). Mutually "
        "exclusive with --anchor-map.",
    )
    parser.add_argument(
        "--anchor-map",
        type=Path,
        default=None,
        help="Path to a scripts/select_repair_anchors.py anchor-map JSON "
        "(task_id -> chosen sample_index, e.g. the Week 6 execution-behavior "
        "consensus selector). Restricts the run to exactly the map's task_ids "
        "and repairs each one's named sample index, ignoring --sample-index, "
        "--failure-split, --selection-mode, and --n-failures (all --n-failures "
        "coordinates from the map are used). Mutually exclusive with "
        "--sample-index.",
    )
    parser.add_argument(
        "--failure-split",
        help="Named source-results failure_splits entry; uses the complete frozen split.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["diverse", "source_order"],
        default="diverse",
        help="Selection used when --failure-split is not provided.",
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--repair-candidates",
        type=int,
        default=1,
        help="Repair candidates generated and ranked per iteration.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--duplicate-policy",
        choices=["accept", "resample"],
        default="accept",
    )
    parser.add_argument(
        "--max-candidate-calls",
        type=int,
        help="Per-iteration model-call cap; defaults to --repair-candidates.",
    )
    parser.add_argument(
        "--stop-on-pass",
        action="store_true",
        help="Stop a candidate batch as soon as one generated program passes.",
    )
    parser.add_argument(
        "--max-consecutive-duplicates",
        type=int,
        help="Stop a candidate batch after this many consecutive duplicates.",
    )
    parser.add_argument(
        "--context-mode",
        choices=REPAIR_CONTEXT_MODES,
        default="error",
        help="How much of the failure each candidate call is shown (Week 9 "
        "ablation ladder). 'error' is the shipped behavior; 'code_only' "
        "withholds the traceback; 'none' withholds the failing program too, "
        "making the run a same-model, same-budget resampling control.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Salesforce/codegen-350M-mono")
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Inference device (default: auto = CUDA when present, else CPU). "
        "Every stored Week 4-7 repair trace was generated on CPU/float32, so "
        "a GPU rerun is NOT numerically comparable to them: re-generate every "
        "arm you intend to compare on the same device.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="auto",
        help="Model weight dtype (default: auto = float16 on CUDA, float32 "
        "elsewhere). Pin to float32 on GPU to stay closer to the stored "
        "CPU runs, at a large speed cost.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing --output checkpoint after validating its config.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only locally cached HuggingFace datasets and models.",
    )
    return parser.parse_args()


def collect_failures(
    source: dict,
    sample_index: int | None = None,
    anchor_map: dict[str, int] | None = None,
) -> list[dict]:
    """Flatten failing samples, optionally restricted to one selection mode.

    Args:
        source: A pipeline results dict (CLAUDE.md Section 8 schema).
        sample_index: Optional global sample index to retain (e.g. the
            existing sample-0 Smart Repair cohort). Mutually exclusive with
            anchor_map.
        anchor_map: Optional task_id -> sample_index map (e.g.
            scripts/select_repair_anchors.py's execution-behavior consensus
            selector). When given, every other task_id is excluded and the
            named sample index is repaired for each mapped task, regardless
            of sample_index. Raises if a mapped sample is not actually
            failing -- the anchor map is expected to be built only from
            already-failing coordinates, so this signals a stale or
            mismatched map rather than being an expected path.
    """
    if sample_index is not None and anchor_map is not None:
        raise ValueError("--sample-index and --anchor-map are mutually exclusive")
    if sample_index is not None and sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    failures = []
    for problem in source["per_problem"]:
        if anchor_map is not None:
            task_id = problem["task_id"]
            if task_id not in anchor_map:
                continue
            anchor_index = anchor_map[task_id]
            sample = problem["samples"][anchor_index]
            if sample["passed"]:
                raise ValueError(
                    f"Anchor for {task_id} (sample {anchor_index}) is passing, "
                    "not failing -- anchor map does not match this source-results"
                )
            failures.append(
                {
                    "task_id": task_id,
                    "sample_index": anchor_index,
                    "initial_code": sample["code"],
                    "initial_error_message": sample["error_message"],
                    "initial_failure_category": sample["failure_category"],
                }
            )
            continue
        for current_sample_index, sample in enumerate(problem["samples"]):
            if sample_index is not None and current_sample_index != sample_index:
                continue
            if sample["passed"]:
                continue
            failures.append(
                {
                    "task_id": problem["task_id"],
                    "sample_index": current_sample_index,
                    "initial_code": sample["code"],
                    "initial_error_message": sample["error_message"],
                    "initial_failure_category": sample["failure_category"],
                }
            )
    return failures


def select_diverse_failures(failures: list[dict], limit: int) -> list[dict]:
    """Select deterministically across categories, preferring distinct tasks."""
    if limit < 0:
        raise ValueError("limit must be non-negative")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for failure in failures:
        by_category[failure["initial_failure_category"]].append(failure)

    categories = sorted(by_category, key=lambda key: (len(by_category[key]), key))
    selected: list[dict] = []
    used_tasks: set[str] = set()

    while len(selected) < limit:
        made_progress = False
        for category in categories:
            candidates = by_category[category]
            index = next(
                (
                    i
                    for i, candidate in enumerate(candidates)
                    if candidate["task_id"] not in used_tasks
                ),
                0 if candidates else None,
            )
            if index is None:
                continue
            candidate = candidates.pop(index)
            selected.append(candidate)
            used_tasks.add(candidate["task_id"])
            made_progress = True
            if len(selected) == limit:
                break
        if not made_progress:
            break
    return selected


def select_source_order_failures(failures: list[dict], limit: int) -> list[dict]:
    """Select the first failures from an already-frozen source ordering."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return failures[:limit]


def select_failure_split(failures: list[dict], split_entries: list[dict]) -> list[dict]:
    """Resolve a frozen list of task/sample coordinates against source failures."""
    by_coordinate = {
        (failure["task_id"], int(failure["sample_index"])): failure
        for failure in failures
    }
    selected = []
    missing = []
    for entry in split_entries:
        coordinate = (entry["task_id"], int(entry["sample_index"]))
        failure = by_coordinate.get(coordinate)
        if failure is None:
            missing.append(coordinate)
        else:
            selected.append(failure)
    if missing:
        raise ValueError(f"Failure split coordinates missing from source: {missing}")
    if len(selected) != len(set((r["task_id"], r["sample_index"]) for r in selected)):
        raise ValueError("Failure split contains duplicate coordinates")
    return selected


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
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


def _category_summary(records: list[dict]) -> dict:
    """Aggregate success, duplicate, and budget metrics for one record group."""
    failures = len(records)
    successful = sum(bool(record["final_passed"]) for record in records)
    model_calls = sum(int(record.get("repair_model_calls", 0)) for record in records)
    duplicates = sum(int(record.get("duplicate_candidates", 0)) for record in records)
    unique_executed = sum(
        int(
            record.get(
                "unique_candidates_executed",
                record.get("repair_model_calls", 0)
                - record.get("duplicate_candidates", 0),
            )
        )
        for record in records
    )
    return {
        "failures": failures,
        "successful_repairs": successful,
        "repair_success_rate": successful / failures if failures else 0.0,
        "repair_success_wilson_95": _wilson_interval(successful, failures),
        "repair_model_calls": model_calls,
        "duplicate_candidates": duplicates,
        "duplicate_candidate_rate": duplicates / model_calls if model_calls else 0.0,
        "unique_candidates_executed": unique_executed,
        "model_calls_per_failure": model_calls / failures if failures else 0.0,
        "model_calls_per_success": model_calls / successful if successful else None,
    }


def summarize(records: list[dict]) -> dict:
    completed = len(records)
    successful = sum(record["final_passed"] for record in records)
    first_step = sum(record["success_iteration"] == 1 for record in records)
    later_step = sum(
        record["success_iteration"] is not None and record["success_iteration"] > 1
        for record in records
    )
    model_calls = sum(record.get("repair_model_calls", 0) for record in records)
    duplicate_candidates = sum(
        record.get("duplicate_candidates", 0) for record in records
    )
    oracle_successes = sum(
        bool(record.get("oracle_candidate_passed", record["final_passed"]))
        for record in records
    )
    stagnation_stops = sum(
        record.get("repair_stop_reason") == "no_improvement"
        or (
            "repair_stop_reason" not in record
            and bool(record.get("trace"))
            and bool(record["trace"][-1].get("duplicate", False))
        )
        for record in records
    )
    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_category[record["initial_failure_category"]].append(record)
    overall = _category_summary(records)
    return {
        "completed_failures": completed,
        "successful_repairs": successful,
        "repair_success_rate": successful / completed if completed else 0.0,
        "repair_success_wilson_95": _wilson_interval(successful, completed),
        "success_after_one_iteration": first_step,
        "additional_successes_from_iteration_2_plus": later_step,
        "repair_model_calls": model_calls,
        "duplicate_candidates": duplicate_candidates,
        "duplicate_candidate_rate": (
            duplicate_candidates / model_calls if model_calls else 0.0
        ),
        "unique_candidates_executed": sum(
            int(
                record.get(
                    "unique_candidates_executed",
                    record.get("repair_model_calls", 0)
                    - record.get("duplicate_candidates", 0),
                )
            )
            for record in records
        ),
        "model_calls_per_failure": overall["model_calls_per_failure"],
        "model_calls_per_success": overall["model_calls_per_success"],
        "oracle_candidate_successes": oracle_successes,
        "ranking_oracle_gap": oracle_successes - successful,
        "stagnation_stops": stagnation_stops,
        "candidate_generation_stop_reasons": dict(
            Counter(
                reason
                for record in records
                for reason in record.get("candidate_generation_stop_reasons", [])
            )
        ),
        "repair_stop_reasons": dict(
            Counter(record.get("repair_stop_reason", "unknown") for record in records)
        ),
        "initial_failure_categories": dict(
            Counter(record["initial_failure_category"] for record in records)
        ),
        "final_failure_categories": dict(
            Counter(
                record["final_failure_category"]
                for record in records
                if not record["final_passed"]
            )
        ),
        "per_initial_failure_category": {
            category: _category_summary(category_records)
            for category, category_records in sorted(by_category.items())
        },
    }


def write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _selection_coordinates(selected: list[dict]) -> list[dict]:
    return [
        {"task_id": item["task_id"], "sample_index": int(item["sample_index"])}
        for item in selected
    ]


def _selection_sha256(coordinates: list[dict]) -> str:
    encoded = json.dumps(coordinates, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_anchor_map(path: Path) -> dict[str, int]:
    """Load a scripts/select_repair_anchors.py anchor-map JSON.

    Args:
        path: Path to the anchor-map JSON (top-level "anchors" key mapping
            task_id -> {"sample_index": ..., ...}).
    Returns:
        task_id -> sample_index, the only field this script consumes.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = payload.get("anchors")
    if anchors is None:
        raise ValueError(f"{path} has no top-level 'anchors' key")
    return {task_id: int(entry["sample_index"]) for task_id, entry in anchors.items()}


def _anchor_map_sha256(anchor_map: dict[str, int]) -> str:
    encoded = json.dumps(anchor_map, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


#: Config keys introduced after the Week 4-7 checkpoints were written. A
#: checkpoint that predates them cannot state what it ran on, so requiring a
#: match would make every stored trace unresumable. They are enforced only once
#: the checkpoint carries them.
_PROVENANCE_KEYS_ADDED_IN_WEEK8 = frozenset(
    {"device_requested", "dtype_requested", "device_resolved", "dtype_resolved"}
)

#: Anchor-map keys added when per-task anchor selection was introduced. A
#: checkpoint written before this feature existed is, by definition, a
#: sample_index (or diverse/source_order) run and never carried these keys --
#: grandfathered the same way the Week 8 provenance keys are.
_ANCHOR_KEYS_ADDED_AFTER_WEEK9 = frozenset(
    {"anchor_map_path", "anchor_map_sha256", "anchor_map_size"}
)

#: Week 9 config keys whose absence from a checkpoint pins down exactly one
#: value: every pre-Week-9 run generated candidates with the full traceback.
#: A checkpoint missing the key may therefore be resumed as "error" and must
#: NOT be resumable as any other arm.
_DEFAULTS_FOR_KEYS_ADDED_AFTER_WEEK8 = {"context_mode": "error"}


#: Sentinel distinguishing "absent from the checkpoint" from a stored None.
_MISSING = object()


def _validate_resume_payload(payload: dict, expected_config: dict) -> None:
    """Reject a resume whose scientific configuration no longer matches."""
    actual = payload.get("config", {})
    grandfathered = sorted(
        key
        for key in (_PROVENANCE_KEYS_ADDED_IN_WEEK8 | _ANCHOR_KEYS_ADDED_AFTER_WEEK9)
        if key not in actual
    )
    mismatches = {
        key: {"checkpoint": actual.get(key), "requested": value}
        for key, value in expected_config.items()
        if key not in grandfathered
        and actual.get(key, _DEFAULTS_FOR_KEYS_ADDED_AFTER_WEEK8.get(key, _MISSING))
        != value
    }
    if mismatches:
        raise ValueError(f"Resume config mismatch: {mismatches}")
    if grandfathered:
        logger.warning(
            "Resuming a checkpoint written before device provenance was "
            "recorded (%s missing). Its completed records were generated on an "
            "unrecorded device; verify it matches the current run before "
            "comparing the two halves.",
            ", ".join(grandfathered),
        )


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    if args.n_failures < 1:
        raise ValueError("n_failures must be positive")
    if args.sample_index is not None and args.sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if args.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if args.repair_candidates < 1:
        raise ValueError("repair_candidates must be positive")
    max_candidate_calls = (
        args.repair_candidates
        if args.max_candidate_calls is None
        else args.max_candidate_calls
    )
    if max_candidate_calls < args.repair_candidates:
        raise ValueError("max-candidate-calls must be at least repair-candidates")
    if (
        args.max_consecutive_duplicates is not None
        and args.max_consecutive_duplicates < 1
    ):
        raise ValueError("max-consecutive-duplicates must be positive")
    if args.anchor_map is not None and args.sample_index is not None:
        raise ValueError("--sample-index and --anchor-map are mutually exclusive")
    if args.anchor_map is not None and args.failure_split:
        raise ValueError("--anchor-map and --failure-split are mutually exclusive")

    anchor_map: dict[str, int] | None = None
    if args.anchor_map is not None:
        anchor_map = _load_anchor_map(args.anchor_map)
        logger.info(
            "--anchor-map given (%d task_ids): --selection-mode and --n-failures "
            "are ignored, every mapped task_id is repaired at its named sample index",
            len(anchor_map),
        )

    source = json.loads(args.source_results.read_text(encoding="utf-8"))
    available_failures = collect_failures(
        source, sample_index=args.sample_index, anchor_map=anchor_map
    )
    if anchor_map is not None:
        missing = sorted(
            set(anchor_map) - {failure["task_id"] for failure in available_failures}
        )
        if missing:
            raise ValueError(
                f"Anchor map task_ids missing from source-results: {missing}"
            )
        selected = sorted(available_failures, key=lambda failure: failure["task_id"])
    elif args.failure_split:
        split_entries = source.get("failure_splits", {}).get(args.failure_split)
        if split_entries is None:
            raise ValueError(
                f"Failure split {args.failure_split!r} missing from source-results"
            )
        selected = select_failure_split(available_failures, split_entries)
    elif args.selection_mode == "source_order":
        selected = select_source_order_failures(
            available_failures, limit=args.n_failures
        )
    else:
        selected = select_diverse_failures(available_failures, limit=args.n_failures)
    if not selected:
        raise ValueError("No failures selected")
    selection_coordinates = _selection_coordinates(selected)
    selection_sha256 = _selection_sha256(selection_coordinates)
    problems = {
        problem["task_id"]: problem for problem in load_dataset_split(args.dataset)
    }
    missing = sorted({item["task_id"] for item in selected} - problems.keys())
    if missing:
        raise ValueError(f"Tasks missing from {args.dataset}: {missing}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Resolved before the model loads so the config records the concrete
    # device/dtype even when the run is later resumed or interrupted.
    _device, _dtype = resolve_runtime(args.device, args.dtype)
    _resolved_device, _resolved_dtype = _device, str(_dtype)
    default_label = (
        "anchor_selection"
        if anchor_map is not None
        else (
            "robustness_week6"
            if (
                args.failure_split
                or args.duplicate_policy != "accept"
                or args.stop_on_pass
                or args.max_candidate_calls is not None
                or args.max_consecutive_duplicates is not None
            )
            else (
                "candidate_ranking_week5"
                if args.repair_candidates > 1
                else "iterative_repair_week4"
            )
        )
    )
    output = args.output or Path("results") / (
        f"{args.dataset}_{default_label}_{timestamp}.json"
    )
    config = {
        "dataset": args.dataset,
        "model": args.model,
        "source_results": str(args.source_results),
        "available_source_failures": len(available_failures),
        "selected_failures": len(selected),
        "failure_split": args.failure_split,
        "selection_mode": (
            "anchor_map"
            if anchor_map is not None
            else (args.selection_mode if not args.failure_split else "frozen_split")
        ),
        "selection_sha256": selection_sha256,
        "selected_failure_coordinates": selection_coordinates,
        "max_repair_iterations": args.max_iterations,
        "repair_candidates": args.repair_candidates,
        "duplicate_policy": args.duplicate_policy,
        "max_candidate_calls": max_candidate_calls,
        "stop_on_pass": args.stop_on_pass,
        "max_consecutive_duplicates": args.max_consecutive_duplicates,
        "temperature": args.temperature,
        # Device/dtype provenance: "auto" alone does not identify a run, and
        # both change the sampled tokens for a fixed seed. Recorded so a
        # cross-device comparison is detectable instead of silent.
        "device_requested": args.device,
        "dtype_requested": args.dtype,
        "device_resolved": _resolved_device,
        "dtype_resolved": _resolved_dtype,
        "seed": args.seed,
        "candidate_seed_policy": (
            "record_seed=seed+record_index; "
            "candidate_seed=record_seed+model_call_index"
        ),
        "ranking_strategy": (
            "pass>failure_progress>unique>changed>lower_candidate_index"
        ),
        "context_mode": args.context_mode,
        "use_error_context": args.context_mode == "error",
    }
    if args.sample_index is not None:
        config["sample_index"] = args.sample_index
    if anchor_map is not None:
        config["anchor_map_path"] = str(args.anchor_map)
        config["anchor_map_sha256"] = _anchor_map_sha256(anchor_map)
        config["anchor_map_size"] = len(anchor_map)
    if output.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists: {output}. Pass --resume to continue it."
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
        _validate_resume_payload(payload, config)
    else:
        if args.resume:
            raise FileNotFoundError(f"Cannot resume missing output: {output}")
        payload = {
            "run_id": output.stem,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "summary": {},
            "records": [],
        }
        write_checkpoint(output, payload)

    existing_coordinates = [
        {"task_id": record["task_id"], "sample_index": record["sample_index"]}
        for record in payload["records"]
    ]
    if existing_coordinates != selection_coordinates[: len(existing_coordinates)]:
        raise ValueError("Checkpoint records are not a prefix of the frozen selection")
    if len(payload["records"]) == len(selected):
        print(f"Already complete: {output}")
        print(json.dumps(payload["summary"], indent=2))
        return

    tokenizer, model = load_model(args.model, device=args.device, dtype=args.dtype)
    for index in range(len(payload["records"]), len(selected)):
        failure = selected[index]
        problem = problems[failure["task_id"]]
        # Preserve the Week 4 paired baseline: candidate 0 for record ``index``
        # uses the original ``seed + index``. Additional Week 5 candidates use
        # the following seeds inside iterative_repair().
        record_seed = args.seed + index
        trace: list[dict] = []
        final_code = iterative_repair(
            problem["prompt"],
            failure["initial_code"],
            problem["test_code"],
            tokenizer,
            model,
            max_iterations=args.max_iterations,
            use_error_context=args.context_mode == "error",
            context_mode=args.context_mode,
            temperature=args.temperature,
            n_candidates=args.repair_candidates,
            duplicate_policy=args.duplicate_policy,
            max_candidate_calls=max_candidate_calls,
            stop_on_pass=args.stop_on_pass,
            max_consecutive_duplicates=args.max_consecutive_duplicates,
            seed=record_seed,
            trace=trace,
        )
        final = trace[-1]
        success_iteration = next(
            (step["iteration"] for step in trace if step["passed"]), None
        )
        repair_steps = trace[1:]
        generated_candidates = [
            candidate
            for step in repair_steps
            for candidate in step.get("candidates", [])
        ]
        oracle_success_iteration = next(
            (
                step["iteration"]
                for step in repair_steps
                if any(candidate["passed"] for candidate in step.get("candidates", []))
            ),
            None,
        )
        record = {
            **failure,
            "attempts_used": len(trace) - 1,
            "repair_rounds_used": len(repair_steps),
            "repair_model_calls": len(generated_candidates),
            "duplicate_candidates": sum(
                candidate["duplicate"] for candidate in generated_candidates
            ),
            "unique_candidates_executed": sum(
                int(step.get("unique_candidates_executed", 0)) for step in repair_steps
            ),
            "candidate_generation_stop_reasons": [
                step.get("candidate_generation_stop_reason", "unknown")
                for step in repair_steps
            ],
            "repair_stop_reason": final.get("repair_stop_reason", "unknown"),
            "oracle_candidate_passed": oracle_success_iteration is not None,
            "oracle_success_iteration": oracle_success_iteration,
            "selected_candidate_indices": [
                step.get("selected_candidate_index") for step in repair_steps
            ],
            "success_iteration": success_iteration,
            "final_passed": final["passed"],
            "final_code": final_code,
            "final_error_message": final["error_message"],
            "final_failure_category": classify_failure(final["error_message"]),
            "trace": trace,
        }
        payload["records"].append(record)
        payload["summary"] = summarize(payload["records"])
        write_checkpoint(output, payload)
        print(
            f"[{index + 1}/{len(selected)}] {failure['task_id']} "
            f"{failure['initial_failure_category']} -> "
            f"{'PASS' if final['passed'] else 'FAIL'} "
            f"(iteration={success_iteration}, "
            f"selected={final.get('selected_candidate_index')})",
            flush=True,
        )

    print(f"Saved: {output}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
