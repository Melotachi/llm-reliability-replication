"""Week 11: the canonical gate that must pass before any BigCodeBench GPU run.

BigCodeBench is the project's harder cell: HumanEval-60 is saturated (the 7B
rung solves 59/60), so a further capability rung there measures nothing.
BigCodeBench ships 1,140 library-heavy function-completion problems whose
tests are `unittest.TestCase` subclasses -- which is exactly why this script
exists first. Four things are validated with **zero GPU**, by executing each
problem's own canonical solution through the project's normal harness:

1. **The driver is correct.** `unittest` collects failures into a result
   object rather than raising, so a naive runner reports success for broken
   code. Canonical solutions passing at ~100% is the only proof the driver
   discriminates; a mutation control (one canonical per problem, deliberately
   broken) proves it is not merely lenient. Same discipline as the Week 6
   pass@1 sanity check.
2. **Which problems this environment cannot run.** 271 of 1,140 tests touch
   the filesystem, 198 use `unittest.mock`, 40 pull `requests`. Those announce
   themselves here as canonical failures, and the evaluation subsample is
   drawn **only from canonical-passing problems**, with the exclusion count
   recorded -- otherwise the cell's pass@1 carries an unknown floor
   contribution from environment breakage and "harder cell" silently becomes
   "partly broken cell".
3. **The Week 7 representation split still applies.** `split_prompt_completion`
   raises unless the assembled candidate starts with the prompt; the whole
   `corrected`-representation apparatus depends on it.
4. **The real timeout.** A pandas/sklearn cold import can exceed the project's
   5 s default on its own, so the value is measured here rather than assumed.

The output JSON carries the eligible task ids and the chosen subsample, which
is what the generation runs consume.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import classify_failure, run_code  # noqa: E402
from src.uncertainty import split_prompt_completion  # noqa: E402
from src.utils import get_logger, load_dataset_split, save_results  # noqa: E402

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 60.0
DEFAULT_SUBSAMPLE = 60


def check_problem(problem: dict, timeout_s: float) -> dict[str, Any]:
    """Execute one problem's canonical solution and time it.

    Args:
        problem: Normalized problem dict from `load_dataset_split`.
        timeout_s: Wall-clock limit passed to `run_code`.
    Returns:
        Record with the verdict, the failure category, the elapsed seconds and
        whether the prompt/completion split holds.
    """
    candidate = problem["prompt"] + problem["canonical_solution"]
    split_ok = True
    try:
        split_prompt_completion(candidate, problem["prompt"])
    except ValueError:
        split_ok = False

    started = time.perf_counter()
    passed, error = run_code(candidate, problem["test_code"], timeout_s=timeout_s)
    elapsed = time.perf_counter() - started
    return {
        "task_id": problem["task_id"],
        "canonical_passed": bool(passed),
        "failure_category": "" if passed else classify_failure(error),
        "error_head": "" if passed else error.strip().splitlines()[-1][:200],
        "elapsed_s": round(elapsed, 3),
        "prompt_split_ok": split_ok,
        "prompt_chars": len(problem["prompt"]),
        "canonical_chars": len(problem["canonical_solution"]),
    }


def mutate(solution: str, entry_point: str, kind: str) -> str:
    """Break a canonical solution so that its tests cannot pass.

    Shadowing the entry point after the canonical body is used rather than
    editing the body: it is syntax-safe for every problem and guaranteed to
    change behavior, which is what a driver check needs. An in-body edit is
    not -- the first attempt at this control replaced `return X` with
    `return None or X`, which is a no-op for truthy X, and all six mutants
    "survived" for that reason rather than because the driver was broken.

    Args:
        solution: The canonical solution body.
        entry_point: Function the tests call.
        kind: "none" returns None from the entry point; "raise" makes it raise.
    Returns:
        The mutated solution.
    Raises:
        ValueError: On an unknown `kind`.
    """
    if kind == "none":
        body = "    return None"
    elif kind == "raise":
        body = '    raise RuntimeError("mutant")'
    else:
        raise ValueError(f"Unknown mutation kind: {kind!r}")
    return f"{solution}\n\n\ndef {entry_point}(*args, **kwargs):\n{body}\n"


def mutation_control(
    problems: list[dict], eligible: list[str], timeout_s: float, n: int, seed: int
) -> dict[str, Any]:
    """Break `n` canonical-passing solutions; every one of them must now fail.

    A driver that swallows failures would report these as passing, which is the
    exact defect this control exists to catch.
    """
    by_id = {problem["task_id"]: problem for problem in problems}
    chosen = random.Random(seed).sample(eligible, min(n, len(eligible)))
    records = []
    for task_id in chosen:
        problem = by_id[task_id]
        for kind in ("none", "raise"):
            mutated = mutate(
                problem["canonical_solution"], problem["entry_point"], kind
            )
            passed, error = run_code(
                problem["prompt"] + mutated, problem["test_code"], timeout_s=timeout_s
            )
            records.append(
                {
                    "task_id": task_id,
                    "kind": kind,
                    "mutant_passed": bool(passed),
                    "failure_category": "" if passed else classify_failure(error),
                }
            )
    survived = [
        f"{record['task_id']}:{record['kind']}"
        for record in records
        if record["mutant_passed"]
    ]
    return {
        "n_mutated": len(records),
        "n_survived": len(survived),
        "survivors": survived,
        "records": records,
    }


def build_validation(
    *,
    timeout_s: float = DEFAULT_TIMEOUT,
    limit: int | None = None,
    subsample: int = DEFAULT_SUBSAMPLE,
    n_mutants: int = 12,
    seed: int = 42,
    eligible_file: Path | None = None,
) -> dict[str, Any]:
    """Run the whole gate and return one persistable dict.

    `eligible_file` restricts the check to an allowlist produced by an earlier
    run of this gate elsewhere. Runnability is a property of the *environment*,
    and the environment of record is whichever machine executes the samples, so
    the evaluation subsample is drawn from problems that pass canonically in
    both -- the intersection, computed here.
    """
    problems = load_dataset_split("bigcodebench")
    if eligible_file is not None:
        allowed = {
            line.strip()
            for line in eligible_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not allowed:
            raise ValueError(f"{eligible_file} lists no task ids")
        problems = [p for p in problems if p["task_id"] in allowed]
        logger.info("Restricted to %d problems from %s", len(problems), eligible_file)
    if limit is not None:
        problems = problems[:limit]
    logger.info("Executing %d canonical solutions", len(problems))

    records = [check_problem(problem, timeout_s) for problem in problems]
    eligible = [r["task_id"] for r in records if r["canonical_passed"]]
    failed = [r for r in records if not r["canonical_passed"]]
    categories: dict[str, int] = {}
    for record in failed:
        categories[record["failure_category"]] = (
            categories.get(record["failure_category"], 0) + 1
        )
    elapsed = sorted(record["elapsed_s"] for record in records)
    control = mutation_control(problems, eligible, timeout_s, n_mutants, seed)
    chosen = sorted(
        random.Random(seed).sample(eligible, min(subsample, len(eligible))),
        key=lambda task_id: int(task_id.split("/")[-1]),
    )

    return {
        "analysis": "week11_bigcodebench_canonical_gate",
        "config": {
            "dataset": "bigcodebench",
            "split": "v0.1.4",
            "timeout_s": timeout_s,
            "limit": limit,
            "subsample": subsample,
            "n_mutants": n_mutants,
            "seed": seed,
            "eligible_file": str(eligible_file) if eligible_file else None,
        },
        "summary": {
            "n_problems": len(records),
            "n_canonical_passed": len(eligible),
            "canonical_pass_rate": round(len(eligible) / max(1, len(records)), 4),
            "n_prompt_split_failures": sum(
                1 for record in records if not record["prompt_split_ok"]
            ),
            "failure_categories": categories,
            "elapsed_s": {
                "median": elapsed[len(elapsed) // 2] if elapsed else 0.0,
                "p95": elapsed[int(0.95 * (len(elapsed) - 1))] if elapsed else 0.0,
                "max": elapsed[-1] if elapsed else 0.0,
            },
        },
        "mutation_control": control,
        "eligible_task_ids": eligible,
        "subsample_task_ids": chosen,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only check the first N problems."
    )
    parser.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE)
    parser.add_argument("--n-mutants", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eligible-file",
        type=Path,
        default=None,
        help="Restrict the check to these task ids (one per line) -- the "
        "allowlist an earlier run of this gate wrote elsewhere. The subsample "
        "is then drawn from problems runnable in BOTH environments.",
    )
    parser.add_argument(
        "--eligible-out",
        type=Path,
        default=Path("data/bigcodebench_eligible_task_ids.txt"),
        help="Where to write every canonically-passing task id.",
    )
    parser.add_argument("--prefix", default="analysis_week11_bcb_gate")
    parser.add_argument(
        "--task-ids-out",
        type=Path,
        default=Path("data/bigcodebench_cell_task_ids.txt"),
        help="Where to write the chosen subsample, one task id per line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = build_validation(
        timeout_s=args.timeout,
        limit=args.limit,
        subsample=args.subsample,
        n_mutants=args.n_mutants,
        seed=args.seed,
        eligible_file=args.eligible_file,
    )
    summary = validation["summary"]
    control = validation["mutation_control"]
    print(f"problems checked        : {summary['n_problems']}")
    print(
        "canonical pass rate     : "
        f"{summary['canonical_pass_rate']:.3f} "
        f"({summary['n_canonical_passed']} eligible)"
    )
    print(f"prompt-split failures   : {summary['n_prompt_split_failures']}")
    print(f"failure categories      : {summary['failure_categories']}")
    print(f"elapsed s (med/p95/max) : {summary['elapsed_s']}")
    print(
        "mutation control        : "
        f"{control['n_survived']}/{control['n_mutated']} survived "
        "(must be 0)"
    )
    print(f"subsample               : {len(validation['subsample_task_ids'])} problems")

    args.eligible_out.parent.mkdir(parents=True, exist_ok=True)
    args.eligible_out.write_text(
        "\n".join(validation["eligible_task_ids"]) + "\n", encoding="utf-8"
    )
    args.task_ids_out.parent.mkdir(parents=True, exist_ok=True)
    args.task_ids_out.write_text(
        "\n".join(validation["subsample_task_ids"]) + "\n", encoding="utf-8"
    )
    path = save_results(validation, prefix=args.prefix)
    print(f"eligible ids            : {args.eligible_out}")
    print(f"task ids                : {args.task_ids_out}")
    print(f"saved                   : {path}")


if __name__ == "__main__":
    main()
