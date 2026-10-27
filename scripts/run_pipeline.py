"""CLI entry point integrating generation, uncertainty, evaluation, and repair.

Usage examples (see CLAUDE.md Section 5):
    python scripts/run_pipeline.py --dataset humaneval --n-problems 3 --n-samples 2 --dry-run
    python scripts/run_pipeline.py --dataset humaneval --n-problems 10 --n-samples 5 \
        --model Salesforce/codegen-350M-mono --seed 42 --mode full
    python scripts/run_pipeline.py --mode uncertainty --dataset humaneval --n-problems 10
    python scripts/run_pipeline.py --mode repair --dataset humaneval --n-problems 10
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    aggregate_results,
    classify_failure,
    evaluate_outputs,
    pass_at_k,
)
from src.generation import (
    DEVICE_CHOICES,
    DEVICE_MAP_CHOICES,
    DTYPE_CHOICES,
    load_model,
    multi_sample_generation,
    resolve_runtime,
)
from src.repair import repair_pipeline
from src.uncertainty import compute_uncertainty
from src.utils import get_logger, load_dataset_split, save_results, set_all_seeds

logger = get_logger(__name__)


def _format_duration(seconds: float) -> str:
    """Render a duration as H:MM:SS for the per-problem progress heartbeat."""
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _serializable_device_map(model) -> dict[str, str] | None:
    """Return accelerate's placement plan as JSON-writable strings, or None.

    `hf_device_map` values are ints for CUDA ordinals and strings for "cpu" /
    "disk"; the whole dict is absent on the single-device path. Coerced here
    rather than at read time so the results JSON never depends on how a given
    accelerate version spelled a device.
    """
    plan = getattr(model, "hf_device_map", None)
    return None if not plan else {str(k): str(v) for k, v in plan.items()}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pipeline entry point."""
    parser = argparse.ArgumentParser(description="Reliable code generation pipeline")
    parser.add_argument(
        "--dataset",
        choices=["humaneval", "mbpp", "bigcodebench"],
        default="humaneval",
    )
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument(
        "--problem-offset",
        type=int,
        default=0,
        help="Skip the first N dataset problems before taking --n-problems "
        "(e.g. --problem-offset 60 --n-problems 104 runs HumanEval/60..163, "
        "complementing an existing first-60 run without re-generating it).",
    )
    parser.add_argument(
        "--task-ids-file",
        type=Path,
        default=None,
        help="Optional allowlist file (one task_id per line, e.g. MBPP/2). "
        "Problems are filtered to this set BEFORE --problem-offset/--n-problems "
        "slicing. Week 6 use case: restrict MBPP to MBPP+ member problems so "
        "EvalPlus-derived behavior probes cover every problem in the run.",
    )
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation budget per sample (default: 256, what every stored "
        "HumanEval/MBPP run used). Raise it on BigCodeBench, whose canonical "
        "solutions are ~2x longer; the run then reports truncation_rate, "
        "since truncated code fails to parse and unparseable samples count as "
        "maximally distant in every diversity metric.",
    )
    parser.add_argument(
        "--exec-timeout",
        type=float,
        default=5.0,
        help="Per-sample execution limit in seconds (default: 5.0). "
        "BigCodeBench needs more: a pandas/sklearn cold import alone can "
        "exceed 5 s, and its canonical solutions measure p95 ~6 s locally.",
    )
    parser.add_argument("--model", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["full", "uncertainty", "repair"],
        default="full",
        help="full: generate+uncertainty+evaluate+repair. "
        "uncertainty: generate+uncertainty+evaluate, skip repair (Student A). "
        "repair: generate+evaluate+repair, skip uncertainty (Student B).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Inference device (default: auto = CUDA when present, else CPU — "
        "the pre-Week-7 behavior every stored run used). 'auto' never picks "
        "MPS, because Metal kernels change the sampled tokens for a fixed seed.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="auto",
        help="Model weight dtype (default: auto = float16 on CUDA, float32 "
        "elsewhere). Pin to float32 on GPU when a run must stay numerically "
        "comparable to this project's CPU-generated runs.",
    )
    parser.add_argument(
        "--device-map",
        choices=DEVICE_MAP_CHOICES,
        default="none",
        help="Weight placement strategy (default: none = the single-device "
        "path every stored run used). 'auto' hands placement to accelerate, "
        "which streams the checkpoint straight to the GPU and offloads any "
        "layer that does not fit to CPU instead of raising OutOfMemoryError. "
        "Offloading is numerically neutral — needed for the >=7B rung of the "
        "capability ladder, where the weights alone nearly fill the card.",
    )
    parser.add_argument(
        "--codebert-model",
        type=str,
        default="microsoft/codebert-base",
        help="HuggingFace encoder for the Week 3 embedding-diversity metric.",
    )
    parser.add_argument(
        "--no-codebert",
        action="store_true",
        help="Skip codebert_cosine_diversity and codebert_bertscore_diversity "
        "(avoids the ~500 MB encoder download; all cheap metrics are still "
        "computed).",
    )
    parser.add_argument(
        "--no-tree-edit",
        action="store_true",
        help="Skip tree_edit_distance_diversity. Zhang-Shasha is superlinear "
        "in tree size over O(N^2) pairs: at N=50 it costs ~99 s per problem "
        "against ~0.3 s for every other metric combined, and it dominates a "
        "large-N run's wall clock. Safe whenever the downstream analysis does "
        "not read it (estimate_reliability.py and analyze_n_sweep.py exclude "
        "it by default).",
    )
    parser.add_argument(
        "--use-error-context",
        action="store_true",
        help="Pass execution error messages into repair prompts (Week 3+).",
    )
    parser.add_argument(
        "--max-repair-iterations",
        type=int,
        default=3,
        help="Maximum repair attempts per failing sample (Week 4; default: 3). "
        "Use 1 to reproduce the Week 2/3 one-step experiments.",
    )
    parser.add_argument(
        "--repair-candidates",
        type=_positive_int,
        default=1,
        help="Independently sampled repair candidates ranked per iteration "
        "(Week 5; default: 1 for backward compatibility).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline as configured but skip writing results/ JSON.",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> dict:
    """Run the configured pipeline stages over a dataset subset.

    Args:
        args: Parsed CLI arguments (see parse_args()).
    Returns:
        The full results dict matching the CLAUDE.md Section 8 schema.
    """
    set_all_seeds(args.seed)

    logger.info("Loading dataset split: %s", args.dataset)
    data = load_dataset_split(args.dataset)
    if args.task_ids_file is not None:
        allowed = {
            line.strip()
            for line in args.task_ids_file.read_text().splitlines()
            if line.strip()
        }
        data = [p for p in data if p["task_id"] in allowed]
        logger.info(
            "Task-id allowlist %s: %d problems match", args.task_ids_file, len(data)
        )
    problems = data[args.problem_offset : args.problem_offset + args.n_problems]

    logger.info("Loading model: %s", args.model)
    resolved_device, resolved_dtype = resolve_runtime(args.device, args.dtype)
    tokenizer, model = load_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
    )

    compute_uncertainty_stage = args.mode in ("full", "uncertainty")
    compute_repair_stage = args.mode in ("full", "repair")

    per_problem = []
    # tqdm alone is not enough on a long run: it redraws with "\r", which a
    # notebook or CI log capturing a piped subprocess never renders, so an
    # hours-long run looks frozen at whatever line last ended in "\n". The
    # newline-terminated heartbeat below is what actually shows up there.
    started_at = time.monotonic()
    for index, problem in enumerate(
        tqdm(problems, desc=f"{args.dataset}/{args.mode}"), start=1
    ):
        prompt = problem["prompt"]
        test_code = problem["test_code"]

        samples = multi_sample_generation(
            prompt,
            tokenizer,
            model,
            n=args.n_samples,
            temperature=args.temperature,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
        # HF completions continue *from* the prompt; they are not standalone
        # runnable code on their own. Assemble the full candidate before
        # executing or AST-parsing it (see src/uncertainty.py docstring).
        full_codes = [prompt + sample for sample in samples]

        execution_results = evaluate_outputs(
            full_codes, test_code, timeout_s=args.exec_timeout
        )
        # Proxy, and named as one: a sample whose completion tokenizes to the
        # full budget was almost certainly cut off rather than finished, since
        # early stopping truncates at the first stop sequence.
        truncated = sum(
            1
            for sample in samples
            if len(tokenizer(sample).input_ids) >= args.max_new_tokens
        )
        n_initial_pass = sum(1 for success, _ in execution_results if success)
        initial_pass_rate = n_initial_pass / len(execution_results)
        failure_types = sorted(
            {classify_failure(err) for success, err in execution_results if not success}
        )

        uncertainty = (
            compute_uncertainty(
                full_codes,
                include_codebert=not args.no_codebert,
                include_tree_edit=not args.no_tree_edit,
                codebert_model_name=args.codebert_model,
            )
            if compute_uncertainty_stage
            else {}
        )

        # Per-sample records; repair fields are appended below when the
        # repair stage runs, so Student B can build before/after tables per
        # failure category without re-running anything.
        sample_records = [
            {
                "code": code,
                "passed": success,
                "error_message": err,
                "failure_category": classify_failure(err),
            }
            for code, (success, err) in zip(full_codes, execution_results)
        ]

        post_repair_pass_rate = None
        n_post_repair_pass = None
        if compute_repair_stage:
            repair_traces: list[list[dict]] = []
            repaired_codes = repair_pipeline(
                prompt,
                full_codes,
                test_code,
                tokenizer,
                model,
                use_error_context=args.use_error_context,
                max_iterations=args.max_repair_iterations,
                temperature=args.temperature,
                n_candidates=args.repair_candidates,
                seed=args.seed,
                traces=repair_traces,
            )
            repaired_results = [
                (bool(trace[-1]["passed"]), str(trace[-1]["error_message"]))
                for trace in repair_traces
            ]
            n_post_repair_pass = sum(1 for success, _ in repaired_results if success)
            post_repair_pass_rate = n_post_repair_pass / len(repaired_results)
            for record, r_code, (r_success, r_err), repair_trace in zip(
                sample_records, repaired_codes, repaired_results, repair_traces
            ):
                record["repaired_code"] = r_code
                record["repaired_passed"] = r_success
                record["repaired_failure_category"] = classify_failure(r_err)
                record["repair_trace"] = repair_trace

        n_samples = len(full_codes)
        pass_at_1 = pass_at_k(n_samples, n_initial_pass, 1)
        pass_at_5 = pass_at_k(n_samples, n_initial_pass, 5) if n_samples >= 5 else None

        per_problem.append(
            {
                "task_id": problem["task_id"],
                "prompt_length": len(prompt),
                "uncertainty": uncertainty,
                "samples": sample_records,
                "initial_pass_rate": initial_pass_rate,
                "post_repair_pass_rate": post_repair_pass_rate,
                "failure_types": failure_types,
                "pass_at_1": pass_at_1,
                "pass_at_5": pass_at_5,
                "n_samples": n_samples,
                "n_initial_pass": n_initial_pass,
                "n_post_repair_pass": n_post_repair_pass,
                "truncation_rate": truncated / max(1, len(samples)),
            }
        )

        elapsed = time.monotonic() - started_at
        remaining = (elapsed / index) * (len(problems) - index)
        logger.info(
            "[%d/%d] %s pass_rate=%.2f | elapsed %s | eta %s",
            index,
            len(problems),
            problem["task_id"],
            initial_pass_rate,
            _format_duration(elapsed),
            _format_duration(remaining),
        )

    results = {
        "run_id": f"{args.dataset}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset,
            "model": args.model,
            "n_samples": args.n_samples,
            "n_problems": args.n_problems,
            "max_new_tokens": args.max_new_tokens,
            "exec_timeout": args.exec_timeout,
            "problem_offset": args.problem_offset,
            "task_ids_file": str(args.task_ids_file) if args.task_ids_file else None,
            # The path does not identify the problem set: two runs can name the
            # same allowlist file and evaluate different problems, which is
            # exactly how a capability ladder joins rungs that were never
            # comparable. The hash of the ids actually evaluated does identify
            # it, and analyze_capability_sweep.py treats it as a protocol key.
            "task_ids_sha256": hashlib.sha256(
                "\n".join(p["task_id"] for p in problems).encode()
            ).hexdigest(),
            "temperature": args.temperature,
            "seed": args.seed,
            "mode": args.mode,
            # Requested vs. resolved are both recorded: "auto" alone does not
            # identify a run, and device/dtype change the sampled tokens.
            "device_requested": args.device,
            "dtype_requested": args.dtype,
            "device_resolved": resolved_device,
            "dtype_resolved": str(resolved_dtype),
            # Requested vs. actual layout, for the same reason device and
            # dtype are both recorded (Week 8 (d)(ii)): "auto" names a
            # strategy, not a placement, and whether a layer was offloaded is
            # what explains the wall clock. None when the single-device path
            # ran, so every pre-existing run compares equal on this key.
            "device_map_requested": args.device_map,
            "device_map_resolved": _serializable_device_map(model),
            # What the weights actually are, not what was asked for. See
            # generation.load_model: newer transformers deprecate the
            # argument this project passes, and every checkpoint here
            # declares a non-float32 dtype in its own config.
            "dtype_actual": str(next(model.parameters()).dtype),
            "codebert_model": (
                args.codebert_model
                if compute_uncertainty_stage and not args.no_codebert
                else None
            ),
            "tree_edit_distance": not args.no_tree_edit,
            "use_error_context": args.use_error_context,
            "max_repair_iterations": args.max_repair_iterations,
            "repair_candidates": args.repair_candidates,
        },
        "summary": aggregate_results(per_problem),
        "per_problem": per_problem,
    }
    return results


def print_summary(results: dict) -> None:
    """Print a human-readable summary table to stdout."""
    print(f"\nRun: {results['run_id']}")
    print(f"Config: {results['config']}\n")
    summary = results["summary"]
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(
        f"\n{'task_id':<16}{'initial_pass':<14}{'pass@1':<10}{'lex_div':<10}"
        f"{'jaccard':<10}{'tok_cos':<10}{'ast_div':<10}{'ted_div':<10}"
        f"{'codebert':<10}{'bertscore':<11}{'pairwise':<10}{'parse_fail':<12}"
    )
    for problem in results["per_problem"]:
        unc = problem["uncertainty"]
        print(
            f"{problem['task_id']:<16}"
            f"{problem['initial_pass_rate']:<14.2f}"
            f"{problem['pass_at_1']:<10.2f}"
            f"{unc.get('lexical_diversity', float('nan')):<10.2f}"
            f"{unc.get('token_jaccard_diversity', float('nan')):<10.2f}"
            f"{unc.get('token_cosine_diversity', float('nan')):<10.2f}"
            f"{unc.get('ast_node_diversity', float('nan')):<10.2f}"
            f"{unc.get('tree_edit_distance_diversity', float('nan')):<10.2f}"
            f"{unc.get('codebert_cosine_diversity', float('nan')):<10.2f}"
            f"{unc.get('codebert_bertscore_diversity', float('nan')):<11.2f}"
            f"{unc.get('pairwise_distance', float('nan')):<10.2f}"
            f"{unc.get('parse_failure_rate', float('nan')):<12.2f}"
        )


def main() -> None:
    args = parse_args()
    results = run_pipeline(args)
    print_summary(results)
    if args.dry_run:
        logger.info("[DRY RUN] skipping results/ persistence")
    else:
        out_path = save_results(results, prefix=f"{args.dataset}_{args.mode}")
        logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
