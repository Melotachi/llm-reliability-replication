"""Week 9 (Student B): does execution-guided repair overfit the suite that guides it?

A *re-analysis*, not a re-run: every repair candidate the Week 8 pool ever
generated is stored, so only the correctness label changes.

THE HYPOTHESIS
--------------
Classical program repair separates a *plausible* patch (passes the given test
suite) from a *correct* one (matches the intended behavior); the gap between
them is patch overfitting (Qi et al., 2015, ISSTA; Smith et al., 2015, FSE).
LLM self-repair inherits the same exposure through two distinct channels, and
this project's pipeline uses both:

  * generation — `repair_code()` puts the failing suite's traceback into the
    repair prompt, so the model is told exactly which assertion to satisfy;
  * selection  — `rank_repair_candidates()` orders candidates by execution
    against that same suite, so the patch that is *returned* is chosen by the
    oracle it may be overfitting.

Neither channel exists for the initial, unguided samples. If overfitting is
real here, patches that pass the base suite should fail EvalPlus's expanded
suite (Liu et al., 2023, NeurIPS) more often than base-passing samples that
were never repaired.

WHAT IS MEASURED
----------------
The headline is the false-positive rate, `P(plus-fail | base-pass)`, for the
same reason `scripts/rescore_evalplus_labels.py` uses it: an absolute drop in
pass rate is bounded above by the base rate and would manufacture a
capability-shaped finding. Three populations are scored under both labels:

  1. `patch_selected`   — the returned patch of every repaired coordinate;
  2. `patch_candidate`  — every repair candidate generated (the generation
     channel alone, independent of what ranking picked);
  3. `unrepaired_sibling` — base-passing samples 1-9 of the *same problems*,
     generated with no error context and no execution-based selection.

Population 3 comes from a different model than 1-2 (CodeGen-350M generator vs.
Qwen2.5-Coder-1.5B-Instruct repairer), which is a real confound and is reported
as such. Its direction is conservative for the hypothesis: the repairer is the
stronger model, so a *higher* false-positive rate for its patches cannot be
explained by capability.

The selection channel gets its own, model-free contrast: coordinates whose
trace holds two or more base-passing candidates that disagree under the plus
label, where base-guided ranking had a choice to make and could take the wrong
one.

Labels are two-directional throughout (base asserts exact equality, EvalPlus
compares through `np.allclose`), so `base-fail & plus-pass` is counted, never
assumed away.

Outputs one analysis JSON under `results/`, including per-coordinate
`repair_success_base` / `repair_success_plus` so the Week 8 gate diagnostics
can be re-scored against the stricter label without re-executing anything.

Usage:
    python scripts/evaluate_patch_overfitting.py \
        --repair-pool results/humaneval_week8_repair_all_sample0_qwen15b.json \
        --source-results results/humaneval_uncertainty_merged164_20260711.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath

from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import classify_failure, run_code
from src.utils import get_logger, load_evalplus_test_code, save_results, set_all_seeds

logger = get_logger(__name__)


def _stored_basename(stored_path: object) -> str:
    """Basename of a path string that may have been written on another OS.

    Same defect `analyze_smart_repair._stored_basename` documents: the Week 6/7
    traces were generated on Windows and carry `results\\file.json`, which
    PurePosixPath reads as one filename.
    """
    return PureWindowsPath(str(stored_path)).name


#: Same mirror defect the Week 7 relabel excludes: HumanEval/32's HumanEval+
#: harness unpacks a float and raises TypeError even for EvalPlus's own
#: canonical solution. Excluded from BOTH labels so the comparison is
#: like-for-like.
DEFAULT_EXCLUDED_TASKS = ("HumanEval/32",)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for a binomial proportion (None when total == 0)."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def collect_patch_units(pool: dict, *, excluded: tuple[str, ...]) -> list[dict]:
    """Flatten a repair pool into one scoring unit per generated program.

    Args:
        pool: A loaded `run_iterative_repair_experiment.py` output JSON.
        excluded: Task ids to drop entirely.
    Returns:
        List of dicts with population, task_id, sample_index, code, and the
        base-suite label already recorded by the repair run.
    """
    units: list[dict] = []
    for record in pool["records"]:
        task_id = record["task_id"]
        if task_id in excluded:
            continue
        sample_index = record["sample_index"]
        units.append(
            {
                "population": "initial_failure",
                "task_id": task_id,
                "sample_index": sample_index,
                "code": record["initial_code"],
                "passed_base": False,
                "initial_failure_category": record["initial_failure_category"],
            }
        )
        if record["final_passed"]:
            units.append(
                {
                    "population": "patch_selected",
                    "task_id": task_id,
                    "sample_index": sample_index,
                    "code": record["final_code"],
                    "passed_base": True,
                    "initial_failure_category": record["initial_failure_category"],
                }
            )
        for step in record["trace"]:
            for candidate in step.get("candidates", []):
                units.append(
                    {
                        "population": "patch_candidate",
                        "task_id": task_id,
                        "sample_index": sample_index,
                        "code": candidate["code"],
                        "passed_base": bool(candidate["passed"]),
                        "iteration": step["iteration"],
                        "candidate_index": candidate["candidate_index"],
                        "duplicate": bool(candidate.get("duplicate", False)),
                        "initial_failure_category": record["initial_failure_category"],
                    }
                )
    return units


def collect_sibling_units(
    source_run: dict,
    task_ids: set[str],
    *,
    repaired_sample_index: int,
    excluded: tuple[str, ...],
) -> list[dict]:
    """Collect the unrepaired base-passing samples of the same problems.

    Args:
        source_run: The generation run the repair pool draws its failures from.
        task_ids: Problems in the repair cohort (matching keeps problem
            difficulty comparable across populations).
        repaired_sample_index: Sample index the repair pool consumed; excluded
            so no program is counted in two populations.
        excluded: Task ids to drop entirely.
    Returns:
        List of scoring units in the same shape as collect_patch_units().
    """
    units: list[dict] = []
    for problem in source_run["per_problem"]:
        task_id = problem["task_id"]
        if task_id in excluded or task_id not in task_ids:
            continue
        for s_idx, sample in enumerate(problem["samples"]):
            if s_idx == repaired_sample_index:
                continue
            units.append(
                {
                    "population": "unrepaired_sibling",
                    "task_id": task_id,
                    "sample_index": s_idx,
                    "code": sample["code"],
                    "passed_base": bool(sample["passed"]),
                }
            )
    return units


def score_units(
    units: list[dict],
    test_code_by_task: dict[str, str],
    *,
    timeout_s: float = 30.0,
    workers: int = 8,
) -> tuple[list[dict], list[str]]:
    """Execute each distinct (task_id, code) once against its plus suite.

    Args:
        units: Scoring units from the collect_* helpers.
        test_code_by_task: task_id -> plus test_code.
        timeout_s: Wall-clock limit per execution.
        workers: Thread-pool size (each worker blocks on a child process).
    Returns:
        (units annotated with `passed_plus` / `plus_failure_category`,
        task ids skipped for want of a plus suite).
    """
    skipped = sorted(
        {u["task_id"] for u in units if u["task_id"] not in test_code_by_task}
    )
    scoped = [u for u in units if u["task_id"] in test_code_by_task]
    distinct = sorted({(u["task_id"], u["code"]) for u in scoped})
    logger.info(
        "scoring %d distinct programs (%d units, %d task ids skipped)",
        len(distinct),
        len(scoped),
        len(skipped),
    )

    def _one(job: tuple[str, str]) -> tuple[tuple[str, str], bool, str]:
        task_id, code = job
        ok, err = run_code(code, test_code_by_task[task_id], timeout_s=timeout_s)
        return job, ok, err

    outcomes: dict[tuple[str, str], tuple[bool, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (job, ok, err) in enumerate(pool.map(_one, distinct), start=1):
            outcomes[job] = (ok, err)
            if done % 50 == 0:
                logger.info("  %d/%d distinct programs scored", done, len(distinct))

    for unit in scoped:
        ok, err = outcomes[(unit["task_id"], unit["code"])]
        unit["passed_plus"] = bool(ok)
        unit["plus_failure_category"] = classify_failure(err) if not ok else ""
    return scoped, skipped


def summarize_population(units: list[dict]) -> dict:
    """Two-by-two confusion, false-positive rate and Wilson interval."""
    both = base_only = plus_only = neither = 0
    plus_fail_categories: dict[str, int] = {}
    for unit in units:
        if unit["passed_base"] and unit["passed_plus"]:
            both += 1
        elif unit["passed_base"]:
            base_only += 1
            cat = unit["plus_failure_category"] or "unknown"
            plus_fail_categories[cat] = plus_fail_categories.get(cat, 0) + 1
        elif unit["passed_plus"]:
            plus_only += 1
        else:
            neither += 1
    n_base_pass = both + base_only
    n_total = both + base_only + plus_only + neither
    return {
        "n_units": n_total,
        "n_base_pass": n_base_pass,
        "confusion": {
            "base_pass_plus_pass": both,
            "base_pass_plus_fail": base_only,
            "base_fail_plus_pass": plus_only,
            "base_fail_plus_fail": neither,
        },
        "false_positive_rate": (base_only / n_base_pass) if n_base_pass else None,
        "false_positive_wilson_95": _wilson_interval(base_only, n_base_pass),
        "base_fail_plus_pass_rate": (plus_only / n_total) if n_total else None,
        "plus_failure_categories": plus_fail_categories,
    }


def contrast(name_a: str, a: dict, name_b: str, b: dict) -> dict:
    """Fisher exact test on two populations' base-pass x plus-outcome tables."""
    a_fail = a["confusion"]["base_pass_plus_fail"]
    a_pass = a["confusion"]["base_pass_plus_pass"]
    b_fail = b["confusion"]["base_pass_plus_fail"]
    b_pass = b["confusion"]["base_pass_plus_pass"]
    table = [[a_fail, a_pass], [b_fail, b_pass]]
    if min(a_fail + a_pass, b_fail + b_pass) == 0:
        return {"a": name_a, "b": name_b, "table": table, "fisher_p": None}
    odds, p = scipy_stats.fisher_exact(table)
    return {
        "a": name_a,
        "b": name_b,
        "table": table,
        "fpr_a": a["false_positive_rate"],
        "fpr_b": b["false_positive_rate"],
        "fpr_difference": (a["false_positive_rate"] or 0.0)
        - (b["false_positive_rate"] or 0.0),
        "odds_ratio": float(odds),
        "fisher_p": float(p),
    }


def selection_channel(
    pool: dict, scored: list[dict], *, excluded: tuple[str, ...]
) -> dict:
    """Score base-guided ranking where it actually had a choice to make.

    A coordinate is *decidable* when its trace holds at least two base-passing
    candidates whose plus labels disagree: there, and only there, ranking on
    the base suite could return an overfitting patch while a correct one was
    available in the same batch.

    Args:
        pool: The repair pool JSON.
        scored: Units annotated by score_units().
        excluded: Task ids dropped from scoring.
    Returns:
        Counts of decidable coordinates and how often the returned patch was
        the plus-failing one.
    """
    plus_by_code: dict[tuple[str, str], bool] = {
        (u["task_id"], u["code"]): u["passed_plus"] for u in scored
    }
    decidable = 0
    wrong_choice = 0
    detail = []
    for record in pool["records"]:
        task_id = record["task_id"]
        if task_id in excluded or not record["final_passed"]:
            continue
        base_passers = []
        for step in record["trace"]:
            for candidate in step.get("candidates", []):
                key = (task_id, candidate["code"])
                if candidate["passed"] and key in plus_by_code:
                    base_passers.append((candidate["code"], plus_by_code[key]))
        labels = {flag for _, flag in base_passers}
        if len(base_passers) < 2 or len(labels) < 2:
            continue
        decidable += 1
        chosen_key = (task_id, record["final_code"])
        chosen_plus = plus_by_code.get(chosen_key)
        if chosen_plus is False:
            wrong_choice += 1
        detail.append(
            {
                "task_id": task_id,
                "n_base_passing_candidates": len(base_passers),
                "n_plus_passing_among_them": sum(flag for _, flag in base_passers),
                "returned_patch_plus_passed": chosen_plus,
            }
        )
    return {
        "n_decidable_coordinates": decidable,
        "n_returned_plus_failing": wrong_choice,
        "rate_returned_plus_failing": (wrong_choice / decidable) if decidable else None,
        "coordinates": detail,
    }


def coordinate_labels(
    pool: dict, scored: list[dict], *, excluded: tuple[str, ...]
) -> list[dict]:
    """Per-coordinate base and plus repair-success labels for gate re-scoring."""
    plus_by_code: dict[tuple[str, str], bool] = {
        (u["task_id"], u["code"]): u["passed_plus"] for u in scored
    }
    rows = []
    for record in pool["records"]:
        task_id = record["task_id"]
        if task_id in excluded:
            continue
        base_success = bool(record["final_passed"])
        plus_success = (
            bool(plus_by_code.get((task_id, record["final_code"]), False))
            if base_success
            else False
        )
        rows.append(
            {
                "task_id": task_id,
                "sample_index": record["sample_index"],
                "initial_failure_category": record["initial_failure_category"],
                "repair_success_base": base_success,
                "repair_success_plus": plus_success,
                "repair_model_calls": record["repair_model_calls"],
                "initial_plus_passed": plus_by_code.get(
                    (task_id, record["initial_code"])
                ),
            }
        )
    return rows


def category_breakdown(rows: list[dict]) -> dict:
    """Repair success under both labels, split by the initial failure kind."""
    out: dict[str, dict] = {}
    for row in rows:
        cat = row["initial_failure_category"] or "unknown"
        cell = out.setdefault(cat, {"n": 0, "base_success": 0, "plus_success": 0})
        cell["n"] += 1
        cell["base_success"] += int(row["repair_success_base"])
        cell["plus_success"] += int(row["repair_success_plus"])
    for cell in out.values():
        cell["false_positive_rate"] = (
            (cell["base_success"] - cell["plus_success"]) / cell["base_success"]
            if cell["base_success"]
            else None
        )
    return out


def validate_plus_harness(
    task_ids: list[str],
    test_code_by_task: dict[str, str],
    *,
    dataset_name: str = "humaneval",
    timeout_s: float = 30.0,
    workers: int = 8,
) -> dict:
    """Run EvalPlus's own reference solutions through the same scorer.

    A zero count of `base-fail & plus-pass` units is indistinguishable from a
    broken harness unless the reference solutions are known to pass, so this
    is run alongside the populations rather than trusted from the Week 7
    relabel.

    Args:
        task_ids: Problems in the cohort.
        test_code_by_task: task_id -> plus test_code.
        dataset_name: Which plus mirror the reference solutions come from.
        timeout_s: Wall-clock limit per execution.
        workers: Thread-pool size.
    Returns:
        Dict with the pass count and the ids of any reference solution that
        fails its own suite.
    """
    from src.utils import load_evalplus_canonical_solutions

    solutions = load_evalplus_canonical_solutions(dataset_name)
    scoped = [t for t in task_ids if t in solutions and t in test_code_by_task]

    def _one(task_id: str) -> tuple[str, bool]:
        ok, _ = run_code(
            solutions[task_id], test_code_by_task[task_id], timeout_s=timeout_s
        )
        return task_id, ok

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task_id, ok in pool.map(_one, scoped):
            if not ok:
                failures.append(task_id)
    return {
        "n_scored": len(scoped),
        "n_passing": len(scoped) - len(failures),
        "failing_task_ids": sorted(failures),
    }


def main() -> None:
    """CLI entry point: score three populations under both labels and persist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-pool", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="analysis_week9_patch_overfitting")
    parser.add_argument(
        "--no-validate-canonical",
        action="store_true",
        help="Skip re-running EvalPlus reference solutions through the scorer.",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=None,
        help="Task ids excluded from both labels (default: HumanEval/32).",
    )
    args = parser.parse_args()
    set_all_seeds(args.seed)

    if args.exclude_task is not None:
        excluded = tuple(args.exclude_task)
    else:
        # The default exclusion is a HumanEval+ mirror defect and has no MBPP+
        # counterpart, so it must not silently drop an MBPP task id.
        excluded = DEFAULT_EXCLUDED_TASKS if args.dataset == "humaneval" else ()
    pool = json.loads(args.repair_pool.read_text())
    source_run = json.loads(args.source_results.read_text())

    pool_source = pool["config"].get("source_results", "")
    if _stored_basename(pool_source) != args.source_results.name:
        raise SystemExit(
            f"repair pool was built from {pool_source!r}, not {args.source_results}"
        )

    repaired_indices = {record["sample_index"] for record in pool["records"]}
    if len(repaired_indices) != 1:
        raise SystemExit(f"expected one repaired sample index, got {repaired_indices}")
    repaired_sample_index = repaired_indices.pop()

    test_code_by_task = load_evalplus_test_code(args.dataset)
    task_ids = {record["task_id"] for record in pool["records"]}

    units = collect_patch_units(pool, excluded=excluded)
    units += collect_sibling_units(
        source_run,
        task_ids,
        repaired_sample_index=repaired_sample_index,
        excluded=excluded,
    )
    scored, skipped = score_units(
        units,
        test_code_by_task,
        timeout_s=args.timeout_s,
        workers=args.workers,
    )

    populations = {}
    for name in (
        "patch_selected",
        "patch_candidate",
        "unrepaired_sibling",
        "initial_failure",
    ):
        populations[name] = summarize_population(
            [u for u in scored if u["population"] == name]
        )

    # Under `stop_on_first_pass`, generation halts at the first base-passing
    # candidate, so every base-passing candidate is also the returned patch and
    # the two populations share one base-passing set. Only one contrast is
    # therefore informative; the identity is asserted rather than assumed.
    candidate_equals_selected = (
        populations["patch_candidate"]["confusion"]["base_pass_plus_pass"]
        == populations["patch_selected"]["confusion"]["base_pass_plus_pass"]
        and populations["patch_candidate"]["confusion"]["base_pass_plus_fail"]
        == populations["patch_selected"]["confusion"]["base_pass_plus_fail"]
    )
    contrasts = [
        contrast(
            "patch_selected",
            populations["patch_selected"],
            "unrepaired_sibling",
            populations["unrepaired_sibling"],
        )
    ]
    if not candidate_equals_selected:
        contrasts.append(
            contrast(
                "patch_candidate",
                populations["patch_candidate"],
                "unrepaired_sibling",
                populations["unrepaired_sibling"],
            )
        )
    rows = coordinate_labels(pool, scored, excluded=excluded)
    n_base = sum(r["repair_success_base"] for r in rows)
    n_plus = sum(r["repair_success_plus"] for r in rows)

    results = {
        "config": {
            "repair_pool": str(args.repair_pool),
            "source_results": str(args.source_results),
            "dataset": args.dataset,
            "timeout_s": args.timeout_s,
            "seed": args.seed,
            "excluded_tasks": list(excluded),
            "repair_model": pool["config"].get("model"),
            "generation_model": source_run["config"].get("model"),
            "repaired_sample_index": repaired_sample_index,
            "skipped_task_ids": skipped,
        },
        "harness_validation": (
            None
            if args.no_validate_canonical
            else validate_plus_harness(
                sorted(task_ids - set(excluded)),
                test_code_by_task,
                dataset_name=args.dataset,
                timeout_s=args.timeout_s,
                workers=args.workers,
            )
        ),
        "patch_candidate_equals_patch_selected": candidate_equals_selected,
        "populations": populations,
        "contrasts": contrasts,
        "selection_channel": selection_channel(pool, scored, excluded=excluded),
        "repair_success": {
            "n_failures_scored": len(rows),
            "base_label": n_base,
            "plus_label": n_plus,
            "base_rate": n_base / len(rows) if rows else None,
            "plus_rate": n_plus / len(rows) if rows else None,
            "base_rate_wilson_95": _wilson_interval(n_base, len(rows)),
            "plus_rate_wilson_95": _wilson_interval(n_plus, len(rows)),
        },
        "by_initial_failure_category": category_breakdown(rows),
        "coordinates": rows,
    }
    path = save_results(results, prefix=args.prefix)

    print(f"\nSaved: {path}")
    print(f"Repair success  base {n_base}/{len(rows)}  plus {n_plus}/{len(rows)}")
    for name, cell in populations.items():
        fpr = cell["false_positive_rate"]
        ci = cell["false_positive_wilson_95"]
        print(
            f"{name:>20}: base-pass {cell['n_base_pass']:>4}/{cell['n_units']:<4} "
            f"FPR {fpr if fpr is None else round(fpr, 4)} "
            f"CI {None if ci is None else [round(c, 3) for c in ci]}"
        )
    for c in contrasts:
        print(
            f"contrast {c['a']} vs {c['b']}: "
            f"{c.get('fpr_a')} vs {c.get('fpr_b')}, fisher p = {c['fisher_p']}"
        )
    if results["harness_validation"]:
        hv = results["harness_validation"]
        print(
            f"harness check: {hv['n_passing']}/{hv['n_scored']} EvalPlus reference "
            f"solutions pass their own plus suite"
        )
    sel = results["selection_channel"]
    print(
        f"selection channel: {sel['n_returned_plus_failing']}/"
        f"{sel['n_decidable_coordinates']} decidable coordinates returned a "
        "plus-failing patch"
    )


if __name__ == "__main__":
    main()
