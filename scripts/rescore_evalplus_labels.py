"""Week 7: re-score the capability ladder against HumanEval+ labels.

Student A deliverable. A *re-analysis*, not a re-run: the five capability
rungs' per-sample code is already stored in `results/`, so only the
correctness label changes.

THE THREAT
----------
Stroebl et al. (2024) argue that benchmark verifiers with weak test suites
produce false positives — samples that pass the shipped tests while being
wrong — and that the false-positive rate need not be constant across models.
Section 9 of the Week 7 report scores five pre-registered predictions whose
x-axis is measured pass@1 on the *base* HumanEval suite. If the false-positive
rate grows with capability, the ladder is stretched non-uniformly and the
trend claims (P1, P3) are contaminated.

WHAT IS MEASURED
----------------
Every stored sample is re-executed against EvalPlus's HumanEval+ suite
(~80x the input volume, Liu et al., 2023, NeurIPS), giving each sample two
labels. The headline is deliberately NOT the drop in pass@1: an absolute
drop is bounded above by base pass@1, so it is mechanically
capability-correlated and would manufacture the finding. The confound-free
quantity, and the one Stroebl's mechanism is about, is

    false_positive_rate = P(plus-fail | base-pass)

computed per rung over that rung's base-passing samples. Relative drop is
reported as secondary, absolute pp as descriptive context only.

The relabel is two-directional and is reported as a full 2x2: base
HumanEval asserts exact equality, while EvalPlus compares through
`np.allclose`, so a float-precision sample can fail base and pass plus.
Nothing is pruned on the assumption that plus-failure implies base-failure.

Outputs (both under `results/`):
  * one re-labelled run JSON per input run, same schema as the generation
    runs so `analyze_capability_sweep.py` consumes them unchanged. They
    carry `config["label_source"] = "evalplus_humanevalplus"` and
    `config["relabel_source"]` so they can never be mistaken for
    generation runs;
  * one analysis JSON with the per-rung 2x2, false-positive rates,
    timeout diagnostics and the across-rung trend.

WEEK 9 EXTENSION — REPAIR RUNS
------------------------------
The same threat applies, with a sharper edge, to the Week 9 context-mode
ladder (`--repair-results`). Its three arms are scored on the *same* 148
base-failing coordinates but emit systematically different programs: the
`none` arm resamples from scratch (mean 324 chars) while `code_only` and
`error` patch the failing program (mean 714 chars). A weak verifier's
false-positive rate therefore need not be equal across arms, and the
61.5 / 43.9 / 36.5% ladder cannot be reported until that is measured.

PRE-REGISTERED READING (written before the run, per the Week 7 Section 9 /
Week 8 Section (e) convention):
  * if P(plus-fail | base-pass) is flat across the three arms, the ladder
    stands as measured and the base-label numbers are the headline;
  * if it is markedly higher for `none`, the ladder narrows and the
    plus label becomes primary in the report;
  * the decisive quantity is not the three independent rates (denominators
    91 / 65 / 54, +-7pp) but the *paired* discordant counts between arms
    under the plus label — computed downstream by
    `scripts/analyze_context_mode_ladder.py --label plus`.

Two repair-specific denominators are reported and must not be conflated:
`HumanEval/32` is excluded from BOTH labels (see DEFAULT_EXCLUDED_TASKS) so
plus rates run over 147 coordinates where base rates run over 148; and
because the relabel is two-directional, a coordinate's *initial* (base-failing)
program can pass the plus suite, in which case it was never a failure under
that label. The initial programs are identical across arms by construction
(same `selection_sha256`), so they are scored once and the shared result
gives every arm the same adjusted denominator.

Usage:
    python scripts/rescore_evalplus_labels.py --results results/a.json ...
    python scripts/rescore_evalplus_labels.py --repair-results results/arm_*.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import classify_failure, pass_at_k, run_code
from src.utils import get_logger, load_evalplus_test_code, save_results, set_all_seeds

logger = get_logger(__name__)

#: HumanEval/32's HumanEval+ harness in the HuggingFace mirror asserts
#: `_poly(*candidate(*inp), inp)`, which unpacks a float and raises
#: TypeError even for EvalPlus's own canonical solution (verified: 59/60
#: canonical solutions pass, this one cannot). Excluded from BOTH labels so
#: the comparison stays like-for-like; excluding it is a mirror-defect
#: workaround, not a difficulty filter.
DEFAULT_EXCLUDED_TASKS = ("HumanEval/32",)


def rescore_run(
    run: dict,
    test_code_by_task: dict[str, str],
    *,
    timeout_s: float = 30.0,
    workers: int = 8,
    excluded_tasks: tuple[str, ...] = DEFAULT_EXCLUDED_TASKS,
) -> dict:
    """Re-execute every stored sample of one run against the plus suites.

    Args:
        run: A loaded results JSON (CLAUDE.md Section 8 schema).
        test_code_by_task: task_id -> plus test_code, from
            utils.load_evalplus_test_code().
        timeout_s: Wall-clock limit per sample, forwarded to run_code().
        workers: Thread-pool size. run_code() already blocks on a child
            process, so threads only wait on subprocesses.
        excluded_tasks: Task ids skipped entirely (see DEFAULT_EXCLUDED_TASKS).
    Returns:
        Dict with "problems" (per-problem re-labelled records) and "skipped"
        (task ids with no usable plus suite).
    """
    jobs: list[tuple[int, int, str, str]] = []
    problems = []
    skipped = []
    for p_idx, problem in enumerate(run["per_problem"]):
        task_id = problem["task_id"]
        if task_id in excluded_tasks or task_id not in test_code_by_task:
            skipped.append(task_id)
            continue
        problems.append(problem)
        for s_idx, sample in enumerate(problem["samples"]):
            jobs.append(
                (len(problems) - 1, s_idx, sample["code"], test_code_by_task[task_id])
            )

    def _one(job: tuple[int, int, str, str]) -> tuple[int, int, bool, str]:
        p_i, s_i, code, test_code = job
        ok, err = run_code(code, test_code, timeout_s=timeout_s)
        return p_i, s_i, ok, err

    outcomes: dict[tuple[int, int], tuple[bool, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (p_i, s_i, ok, err) in enumerate(pool.map(_one, jobs), start=1):
            outcomes[(p_i, s_i)] = (ok, err)
            if done % 200 == 0:
                logger.info("  re-scored %d/%d samples", done, len(jobs))

    records = []
    for p_i, problem in enumerate(problems):
        samples = []
        for s_i, sample in enumerate(problem["samples"]):
            ok, err = outcomes[(p_i, s_i)]
            samples.append(
                {
                    "code": sample["code"],
                    "passed_base": bool(sample["passed"]),
                    "passed_plus": bool(ok),
                    "plus_failure_category": classify_failure(err),
                }
            )
        records.append({"task_id": problem["task_id"], "samples": samples})
    return {"problems": records, "skipped": skipped}


def timeout_sensitivity(
    rescored: dict,
    test_code_by_task: dict[str, str],
    *,
    long_timeout_s: float = 300.0,
    workers: int = 4,
) -> dict:
    """Re-run every timed-out base-passing sample under a much longer limit.

    A timeout is the one plus-failure mode that can be an artefact of this
    script rather than a property of the sample: HumanEval+ runs ~80x the
    base input volume, so a slow-but-correct sample could be scored wrong.
    Any sample rescued here was a false negative of the main pass, and the
    false-positive rate must be reported at the longer limit instead.

    Args:
        rescored: The return value of rescore_run().
        test_code_by_task: task_id -> plus test_code.
        long_timeout_s: The generous limit to retry under.
        workers: Thread-pool size.
    Returns:
        Dict with the number retried, the number rescued, and the
        false-positive rate recomputed at the longer limit.
    """
    jobs = []
    for problem in rescored["problems"]:
        for sample in problem["samples"]:
            if (
                sample["passed_base"]
                and not sample["passed_plus"]
                and sample["plus_failure_category"] == "timeout"
            ):
                jobs.append((problem["task_id"], sample["code"]))

    rescued = 0
    if jobs:

        def _one(job: tuple[str, str]) -> bool:
            task_id, code = job
            ok, _ = run_code(code, test_code_by_task[task_id], timeout_s=long_timeout_s)
            return ok

        with ThreadPoolExecutor(max_workers=workers) as pool:
            rescued = sum(pool.map(_one, jobs))

    base_pass = sum(
        s["passed_base"] for p in rescored["problems"] for s in p["samples"]
    )
    plus_fail_among_base_pass = sum(
        s["passed_base"] and not s["passed_plus"]
        for p in rescored["problems"]
        for s in p["samples"]
    )
    return {
        "long_timeout_s": long_timeout_s,
        "n_timeout_base_pass": len(jobs),
        "n_rescued": int(rescued),
        "false_positive_rate_at_long_timeout": (
            (plus_fail_among_base_pass - rescued) / base_pass if base_pass else None
        ),
    }


def summarize_cell(rescored: dict) -> dict:
    """Compute one rung's 2x2, false-positive rate and pass@1 under both labels.

    Args:
        rescored: The return value of rescore_run().
    Returns:
        Dict of counts, rates and per-label pass@1 / pass@5, all restricted
        to the problems actually scored under both labels.
    """
    both = base_only = plus_only = neither = 0
    timeouts_among_plus_fail = 0
    plus_fail_categories: dict[str, int] = {}
    p1_base, p1_plus, p5_base, p5_plus = [], [], [], []
    for problem in rescored["problems"]:
        n = len(problem["samples"])
        c_base = sum(s["passed_base"] for s in problem["samples"])
        c_plus = sum(s["passed_plus"] for s in problem["samples"])
        p1_base.append(pass_at_k(n, c_base, 1))
        p1_plus.append(pass_at_k(n, c_plus, 1))
        p5_base.append(pass_at_k(n, c_base, 5))
        p5_plus.append(pass_at_k(n, c_plus, 5))
        for s in problem["samples"]:
            if s["passed_base"] and s["passed_plus"]:
                both += 1
            elif s["passed_base"]:
                base_only += 1
                cat = s["plus_failure_category"] or "unknown"
                plus_fail_categories[cat] = plus_fail_categories.get(cat, 0) + 1
                if cat == "timeout":
                    timeouts_among_plus_fail += 1
            elif s["passed_plus"]:
                plus_only += 1
            else:
                neither += 1

    n_base_pass = both + base_only
    n_samples = both + base_only + plus_only + neither
    return {
        "n_problems_scored": len(rescored["problems"]),
        "n_samples_scored": n_samples,
        "confusion": {
            "base_pass_plus_pass": both,
            "base_pass_plus_fail": base_only,
            "base_fail_plus_pass": plus_only,
            "base_fail_plus_fail": neither,
        },
        "false_positive_rate": (base_only / n_base_pass) if n_base_pass else None,
        "base_fail_plus_pass_rate": (plus_only / n_samples) if n_samples else None,
        "mean_pass_at_1_base": float(np.mean(p1_base)) if p1_base else None,
        "mean_pass_at_1_plus": float(np.mean(p1_plus)) if p1_plus else None,
        "mean_pass_at_5_base": float(np.mean(p5_base)) if p5_base else None,
        "mean_pass_at_5_plus": float(np.mean(p5_plus)) if p5_plus else None,
        "absolute_pass_at_1_drop": (
            float(np.mean(p1_base) - np.mean(p1_plus)) if p1_base else None
        ),
        "relative_pass_at_1_drop": (
            float((np.mean(p1_base) - np.mean(p1_plus)) / np.mean(p1_base))
            if p1_base and np.mean(p1_base) > 0
            else None
        ),
        "timeout_fraction_among_plus_failures": (
            timeouts_among_plus_fail / base_only if base_only else 0.0
        ),
        "plus_failure_categories": plus_fail_categories,
        "skipped_task_ids": rescored["skipped"],
    }


def relabelled_run(run: dict, rescored: dict, source_path: str) -> dict:
    """Rebuild an input run with plus labels, in the generation-run schema.

    Only scored problems are kept, `samples[].passed` becomes the plus
    label, and the per-problem/summary aggregates are recomputed so
    downstream scripts need no changes.

    Args:
        run: The original loaded run JSON.
        rescored: The return value of rescore_run() for that run.
        source_path: Path of the original run, recorded in the config.
    Returns:
        A new run dict; never overwrites the input.
    """
    plus_by_task = {p["task_id"]: p for p in rescored["problems"]}
    out_problems = []
    for problem in run["per_problem"]:
        scored = plus_by_task.get(problem["task_id"])
        if scored is None:
            continue
        new_problem = json.loads(json.dumps(problem))
        n = len(new_problem["samples"])
        for sample, relabelled in zip(new_problem["samples"], scored["samples"]):
            sample["passed"] = relabelled["passed_plus"]
            sample["failure_category"] = relabelled["plus_failure_category"]
        c = sum(s["passed"] for s in new_problem["samples"])
        new_problem["initial_pass_rate"] = c / n if n else 0.0
        new_problem["n_initial_pass"] = c
        new_problem["pass_at_1"] = pass_at_k(n, c, 1)
        new_problem["pass_at_5"] = pass_at_k(n, c, 5)
        new_problem["failure_types"] = sorted(
            {
                s["failure_category"]
                for s in new_problem["samples"]
                if s["failure_category"]
            }
        )
        out_problems.append(new_problem)

    out = {
        "run_id": f"{run.get('run_id', 'run')}_evalpluslabel",
        "timestamp": run.get("timestamp"),
        "config": {
            **run["config"],
            "label_source": "evalplus_humanevalplus",
            "relabel_source": source_path,
            "n_problems": len(out_problems),
        },
        "summary": {
            "mean_pass_at_1": float(np.mean([p["pass_at_1"] for p in out_problems])),
            "mean_pass_at_5": float(np.mean([p["pass_at_5"] for p in out_problems])),
            "mean_initial_pass_rate": float(
                np.mean([p["initial_pass_rate"] for p in out_problems])
            ),
            "mean_post_repair_pass_rate": None,
            "repair_success_rate": None,
        },
        "per_problem": out_problems,
    }
    return out


def trend_across_rungs(cells: list[dict]) -> dict:
    """Correlate each rung's false-positive rate with its base pass@1.

    With five rungs a Spearman p-value bottoms out at 0.0167 even under
    perfect monotonicity, so the ordered rates and their spread are the
    readable evidence; the coefficients are reported alongside, not instead.

    Args:
        cells: Per-rung summary dicts (each with "model", "mean_pass_at_1_base"
            and "false_positive_rate").
    Returns:
        Dict of Pearson/Spearman coefficients, p-values, and the spread.
    """
    x = [c["mean_pass_at_1_base"] for c in cells]
    y = [c["false_positive_rate"] for c in cells]
    out: dict = {
        "capability_ordered_models": [c["model"] for c in cells],
        "capability_ordered_base_pass_at_1": x,
        "capability_ordered_false_positive_rate": y,
        "false_positive_rate_spread": (max(y) - min(y)) if y else None,
    }
    if len(x) >= 3:
        pr, pp = scipy_stats.pearsonr(x, y)
        sr, sp = scipy_stats.spearmanr(x, y)
        rel = [c["relative_pass_at_1_drop"] for c in cells]
        rr, rp = scipy_stats.pearsonr(x, rel)
        out.update(
            {
                "pearson_r_fpr_vs_capability": float(pr),
                "pearson_p_fpr_vs_capability": float(pp),
                "spearman_rho_fpr_vs_capability": float(sr),
                "spearman_p_fpr_vs_capability": float(sp),
                "pearson_r_relative_drop_vs_capability": float(rr),
                "pearson_p_relative_drop_vs_capability": float(rp),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Week 9: repair runs (scripts/run_iterative_repair_experiment.py schema)
# --------------------------------------------------------------------------- #


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for a binomial proportion (same form as the
    repair experiment's, so the two files' intervals are comparable)."""
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


def is_repair_run(run: dict) -> bool:
    """True when a loaded JSON is a repair run rather than a generation run.

    Args:
        run: A loaded results JSON.
    Returns:
        True if it carries `records[].final_code` (the repair-experiment
        schema) rather than `per_problem[].samples[].code`.
    """
    records = run.get("records")
    return bool(records) and "final_code" in records[0]


def shared_initial_pool(runs: list[dict], sources: list[str]) -> list[dict]:
    """Return the repair pool's initial programs, asserting arm identity.

    The context-mode arms repair the *same* coordinates with the *same*
    failing programs (identical `selection_sha256`), so the initial programs
    need scoring only once. Divergence would silently invalidate every paired
    comparison downstream, so it raises rather than warns.

    Args:
        runs: Loaded repair runs, in the order given on the command line.
        sources: Their paths, for the error message.
    Returns:
        List of {"task_id", "sample_index", "code"}, in the first run's order.
    Raises:
        ValueError: If the coordinates or the initial programs differ.
    """
    reference = [
        {
            "task_id": r["task_id"],
            "sample_index": r["sample_index"],
            "code": r["initial_code"],
        }
        for r in runs[0]["records"]
    ]
    for run, source in zip(runs[1:], sources[1:]):
        other = [
            {
                "task_id": r["task_id"],
                "sample_index": r["sample_index"],
                "code": r["initial_code"],
            }
            for r in run["records"]
        ]
        if other != reference:
            raise ValueError(
                f"Repair arms do not share an identical failure pool: "
                f"{sources[0]} vs {source}. Paired comparison is invalid."
            )
    return reference


def rescore_repair_run(
    run: dict,
    test_code_by_task: dict[str, str],
    *,
    field: str = "final_code",
    passed_field: str = "final_passed",
    timeout_s: float = 30.0,
    workers: int = 8,
    excluded_tasks: tuple[str, ...] = DEFAULT_EXCLUDED_TASKS,
) -> dict:
    """Re-execute one repair arm's programs against the plus suite.

    Shaped exactly like :func:`rescore_run`'s return value — one "problem"
    per repair coordinate, holding one sample — so
    :func:`timeout_sensitivity` consumes it unchanged.

    Args:
        run: A loaded repair run.
        test_code_by_task: task_id -> plus test_code.
        field: Which stored program to score ("final_code" or "initial_code").
        passed_field: The stored base label for that program; when absent
            from the record (initial programs are failures by construction)
            it is taken as False.
        timeout_s: Wall-clock limit per execution, forwarded to run_code().
        workers: Thread-pool size.
        excluded_tasks: Task ids skipped entirely.
    Returns:
        Dict with "problems" and "skipped", as in rescore_run().
    """
    jobs: list[tuple[int, str, str]] = []
    kept = []
    skipped = []
    for record in run["records"]:
        task_id = record["task_id"]
        if task_id in excluded_tasks or task_id not in test_code_by_task:
            skipped.append(task_id)
            continue
        kept.append(record)
        jobs.append((len(kept) - 1, record[field], test_code_by_task[task_id]))

    def _one(job: tuple[int, str, str]) -> tuple[int, bool, str]:
        idx, code, test_code = job
        ok, err = run_code(code, test_code, timeout_s=timeout_s)
        return idx, ok, err

    outcomes: dict[int, tuple[bool, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (idx, ok, err) in enumerate(pool.map(_one, jobs), start=1):
            outcomes[idx] = (ok, err)
            if done % 50 == 0:
                logger.info("  re-scored %d/%d programs", done, len(jobs))

    problems = []
    for idx, record in enumerate(kept):
        ok, err = outcomes[idx]
        problems.append(
            {
                "task_id": record["task_id"],
                "sample_index": record["sample_index"],
                "samples": [
                    {
                        "code": record[field],
                        "passed_base": bool(record.get(passed_field, False)),
                        "passed_plus": bool(ok),
                        "plus_failure_category": classify_failure(err),
                    }
                ],
            }
        )
    return {"problems": problems, "skipped": skipped}


def summarize_repair_cell(
    rescored: dict,
    *,
    n_coordinates_total: int,
    initial_plus_pass_ids: set[str] | None = None,
) -> dict:
    """Compute one arm's 2x2, false-positive rate and both-label success rates.

    Deliberately does not reuse :func:`summarize_cell`: a repair coordinate
    holds a single program, so that function's pass@1 / pass@5 would be a
    restatement of the label rather than an estimator.

    Args:
        rescored: The return value of rescore_repair_run() on the arm's
            final programs.
        n_coordinates_total: Coordinates in the arm before exclusions — the
            denominator every base-label number in the arm JSON already uses.
        initial_plus_pass_ids: Task ids whose *initial* program passes the
            plus suite, i.e. that were never failures under that label. When
            given, an adjusted plus rate over the remaining coordinates is
            reported alongside the unadjusted one.
    Returns:
        Dict of counts, rates and Wilson intervals under both labels.
    """
    both = base_only = plus_only = neither = 0
    plus_fail_categories: dict[str, int] = {}
    timeouts_among_plus_fail = 0
    adjusted_scored = adjusted_plus_pass = 0
    initial_plus_pass_ids = initial_plus_pass_ids or set()
    for problem in rescored["problems"]:
        sample = problem["samples"][0]
        if sample["passed_base"] and sample["passed_plus"]:
            both += 1
        elif sample["passed_base"]:
            base_only += 1
            cat = sample["plus_failure_category"] or "unknown"
            plus_fail_categories[cat] = plus_fail_categories.get(cat, 0) + 1
            if cat == "timeout":
                timeouts_among_plus_fail += 1
        elif sample["passed_plus"]:
            plus_only += 1
        else:
            neither += 1
        if problem["task_id"] not in initial_plus_pass_ids:
            adjusted_scored += 1
            adjusted_plus_pass += int(sample["passed_plus"])

    n_scored = both + base_only + plus_only + neither
    n_base_pass = both + base_only
    n_plus_pass = both + plus_only
    return {
        "n_coordinates_total": n_coordinates_total,
        "n_coordinates_scored": n_scored,
        "skipped_task_ids": rescored["skipped"],
        "confusion": {
            "base_pass_plus_pass": both,
            "base_pass_plus_fail": base_only,
            "base_fail_plus_pass": plus_only,
            "base_fail_plus_fail": neither,
        },
        "successes_base": n_base_pass,
        "repair_success_rate_base_over_scored": (
            n_base_pass / n_scored if n_scored else None
        ),
        "repair_success_rate_base_wilson_95": _wilson_interval(n_base_pass, n_scored),
        "successes_plus": n_plus_pass,
        "repair_success_rate_plus_over_scored": (
            n_plus_pass / n_scored if n_scored else None
        ),
        "repair_success_rate_plus_wilson_95": _wilson_interval(n_plus_pass, n_scored),
        "false_positive_rate": (base_only / n_base_pass) if n_base_pass else None,
        "false_positive_rate_wilson_95": _wilson_interval(base_only, n_base_pass),
        "base_fail_plus_pass": plus_only,
        "base_fail_plus_pass_rate": (plus_only / n_scored) if n_scored else None,
        "n_coordinates_failing_under_plus": adjusted_scored,
        "successes_plus_adjusted": adjusted_plus_pass,
        "repair_success_rate_plus_adjusted": (
            adjusted_plus_pass / adjusted_scored if adjusted_scored else None
        ),
        "timeout_fraction_among_plus_failures": (
            timeouts_among_plus_fail / base_only if base_only else 0.0
        ),
        "plus_failure_categories": plus_fail_categories,
    }


def relabelled_repair_run(
    run: dict,
    rescored: dict,
    initial_rescored: dict,
    source_path: str,
) -> dict:
    """Rebuild a repair run carrying BOTH labels, in the repair-run schema.

    Additive rather than overwriting (unlike :func:`relabelled_run`, whose
    consumers expect the generation schema's single `passed` key): the
    downstream ladder analysis switches between labels with `--label`, so
    both must survive in one file. Coordinates with no plus suite are dropped
    and listed in the config.

    Args:
        run: The original loaded repair run.
        rescored: rescore_repair_run() over the arm's final programs.
        initial_rescored: rescore_repair_run() over the shared initial pool.
        source_path: Path of the original arm run, recorded in the config.
    Returns:
        A new run dict; never overwrites the input.
    """
    final_by_task = {p["task_id"]: p["samples"][0] for p in rescored["problems"]}
    initial_by_task = {
        p["task_id"]: p["samples"][0] for p in initial_rescored["problems"]
    }
    records = []
    for record in run["records"]:
        scored = final_by_task.get(record["task_id"])
        if scored is None:
            continue
        new_record = json.loads(json.dumps(record))
        new_record["final_passed_plus"] = scored["passed_plus"]
        new_record["final_plus_failure_category"] = scored["plus_failure_category"]
        initial = initial_by_task.get(record["task_id"])
        new_record["initial_passed_plus"] = (
            bool(initial["passed_plus"]) if initial else None
        )
        records.append(new_record)

    successes_plus = sum(r["final_passed_plus"] for r in records)
    return {
        "run_id": f"{run.get('run_id', 'run')}_evalpluslabel",
        "timestamp": run.get("timestamp"),
        "config": {
            **{k: v for k, v in run["config"].items()},
            "label_source": "base+evalplus_humanevalplus",
            "relabel_source": source_path,
            "relabel_excluded_task_ids": sorted(set(rescored["skipped"])),
            "n_coordinates_relabelled": len(records),
        },
        "summary": {
            **{k: v for k, v in run["summary"].items()},
            "successful_repairs_plus": int(successes_plus),
            "repair_success_rate_plus": (
                successes_plus / len(records) if records else None
            ),
            "repair_success_rate_plus_wilson_95": _wilson_interval(
                int(successes_plus), len(records)
            ),
        },
        "records": records,
    }


def repair_arm_comparison(cells: list[dict]) -> dict:
    """Report whether the false-positive rate is flat across the arms.

    Three independent rates on denominators of ~50-90 have roughly +-7pp
    intervals, so this block answers only "is there a *large* arm-dependent
    verifier effect"; the paired discordant counts under the plus label —
    computed by scripts/analyze_context_mode_ladder.py — are what decide the
    ladder itself.

    Args:
        cells: Per-arm summary dicts from summarize_repair_cell(), each with
            "context_mode" and "false_positive_rate".
    Returns:
        Dict of the ordered rates, their spread, and the intervals' overlap.
    """
    rates = [
        c["false_positive_rate"] for c in cells if c["false_positive_rate"] is not None
    ]
    intervals = [
        c["false_positive_rate_wilson_95"]
        for c in cells
        if c["false_positive_rate"] is not None
    ]
    overlap_low = max((i[0] for i in intervals), default=None)
    overlap_high = min((i[1] for i in intervals), default=None)
    return {
        "arms": [c["context_mode"] for c in cells],
        "false_positive_rate_by_arm": [c["false_positive_rate"] for c in cells],
        "false_positive_rate_spread": (max(rates) - min(rates)) if rates else None,
        "wilson_intervals_all_overlap": (
            bool(overlap_low is not None and overlap_low <= overlap_high)
        ),
        "common_overlap": (
            [overlap_low, overlap_high]
            if overlap_low is not None and overlap_low <= overlap_high
            else None
        ),
        "base_label_success_by_arm": [c["successes_base"] for c in cells],
        "plus_label_success_by_arm": [c["successes_plus"] for c in cells],
    }


def run_repair_relabel(
    args: argparse.Namespace, test_code_by_task: dict[str, str]
) -> dict:
    """Relabel every repair arm and assemble the Week 9 analysis dict.

    Args:
        args: Parsed CLI arguments.
        test_code_by_task: task_id -> plus test_code.
    Returns:
        The analysis dict written to results/.
    """
    paths = [str(p) for p in args.repair_results]
    runs = [json.loads(Path(p).read_text()) for p in paths]
    for path, run in zip(paths, runs):
        if not is_repair_run(run):
            raise ValueError(f"{path} is not a repair run (no records[].final_code)")

    pool = shared_initial_pool(runs, paths)
    logger.info(
        "scoring the shared initial pool once (%d coordinates, identical across "
        "%d arms)",
        len(pool),
        len(runs),
    )
    initial_rescored = rescore_repair_run(
        runs[0],
        test_code_by_task,
        field="initial_code",
        passed_field="__never__",
        timeout_s=args.timeout,
        workers=args.workers,
    )
    initial_plus_pass_ids = {
        p["task_id"]
        for p in initial_rescored["problems"]
        if p["samples"][0]["passed_plus"]
    }
    logger.info(
        "  initial programs passing the plus suite: %d/%d",
        len(initial_plus_pass_ids),
        len(initial_rescored["problems"]),
    )

    cells = []
    written = []
    for path, run in zip(paths, runs):
        mode = run["config"].get("context_mode", "unknown")
        logger.info("re-scoring arm %s (%s)", mode, Path(path).name)
        rescored = rescore_repair_run(
            run,
            test_code_by_task,
            timeout_s=args.timeout,
            workers=args.workers,
        )
        cell = summarize_repair_cell(
            rescored,
            n_coordinates_total=len(run["records"]),
            initial_plus_pass_ids=initial_plus_pass_ids,
        )
        cell["context_mode"] = mode
        cell["run_id"] = run.get("run_id")
        cell["source_results_file"] = path
        if args.sensitivity_timeout > 0:
            cell["timeout_sensitivity"] = timeout_sensitivity(
                rescored,
                test_code_by_task,
                long_timeout_s=args.sensitivity_timeout,
                workers=max(1, args.workers // 2),
            )
            logger.info("  timeout sensitivity: %s", cell["timeout_sensitivity"])
        logger.info(
            "  base %d -> plus %d of %d scored | FP rate %s",
            cell["successes_base"],
            cell["successes_plus"],
            cell["n_coordinates_scored"],
            (
                f"{cell['false_positive_rate']:.4f}"
                if cell["false_positive_rate"] is not None
                else "n/a"
            ),
        )
        cells.append(cell)
        if not args.no_write_runs:
            out = relabelled_repair_run(run, rescored, initial_rescored, path)
            stem = Path(path).stem
            written.append(str(save_results(out, prefix=f"{stem}_evalpluslabel")))

    return {
        "config": {
            "dataset": args.dataset,
            "timeout_s": args.timeout,
            "seed": args.seed,
            "excluded_tasks": list(DEFAULT_EXCLUDED_TASKS),
            "label": "evalplus_plus_suite",
            "input_kind": "repair_runs",
        },
        "source_results_files": paths,
        "relabelled_run_files": written,
        "initial_pool": {
            "n_coordinates": len(initial_rescored["problems"]),
            "skipped_task_ids": sorted(set(initial_rescored["skipped"])),
            "n_initial_plus_pass": len(initial_plus_pass_ids),
            "initial_plus_pass_task_ids": sorted(initial_plus_pass_ids),
        },
        "cells": cells,
        "arm_comparison": repair_arm_comparison(cells),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        type=Path,
        default=[],
        help="Generation runs (CLAUDE.md Section 8 schema) to relabel.",
    )
    parser.add_argument(
        "--repair-results",
        nargs="+",
        type=Path,
        default=[],
        help="Week 9: repair-experiment runs (records[].final_code) to relabel. "
        "Given several arms of one ablation, their shared initial pool is "
        "verified identical and scored once.",
    )
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sensitivity-timeout",
        type=float,
        default=0.0,
        help="If > 0, retry every timed-out base-passing sample at this limit "
        "and report the false-positive rate recomputed there (guards against "
        "slow-but-correct samples being scored as plus-failures).",
    )
    parser.add_argument(
        "--no-write-runs",
        action="store_true",
        help="Skip writing the re-labelled run JSONs (analysis only).",
    )
    args = parser.parse_args()
    if not args.results and not args.repair_results:
        parser.error("pass --results and/or --repair-results")

    set_all_seeds(args.seed)
    test_code_by_task = load_evalplus_test_code(args.dataset)

    if args.repair_results:
        analysis = run_repair_relabel(args, test_code_by_task)
        out_path = save_results(
            analysis, prefix=f"analysis_week9_evalplus_relabel_{args.dataset}"
        )
        logger.info("wrote %s", out_path)
        if not args.results:
            return

    cells = []
    written = []
    for path in args.results:
        run = json.loads(path.read_text())
        model = run["config"]["model"]
        logger.info("re-scoring %s (%s)", path.name, model)
        rescored = rescore_run(
            run,
            test_code_by_task,
            timeout_s=args.timeout,
            workers=args.workers,
        )
        cell = summarize_cell(rescored)
        cell["model"] = model
        if args.sensitivity_timeout > 0:
            cell["timeout_sensitivity"] = timeout_sensitivity(
                rescored,
                test_code_by_task,
                long_timeout_s=args.sensitivity_timeout,
                workers=max(1, args.workers // 2),
            )
            logger.info("  timeout sensitivity: %s", cell["timeout_sensitivity"])
        cell["source_results_file"] = str(path)
        cells.append(cell)
        logger.info(
            "  pass@1 %.4f -> %.4f | FP rate %.4f",
            cell["mean_pass_at_1_base"],
            cell["mean_pass_at_1_plus"],
            cell["false_positive_rate"],
        )
        if not args.no_write_runs:
            out = relabelled_run(run, rescored, str(path))
            written.append(
                str(save_results(out, prefix=f"{args.dataset}_evalpluslabel"))
            )

    cells.sort(key=lambda c: c["mean_pass_at_1_base"])
    analysis = {
        "config": {
            "dataset": args.dataset,
            "timeout_s": args.timeout,
            "seed": args.seed,
            "excluded_tasks": list(DEFAULT_EXCLUDED_TASKS),
            "label": "evalplus_humanevalplus",
        },
        "source_results_files": [str(p) for p in args.results],
        "relabelled_run_files": written,
        "cells": cells,
        "trend": trend_across_rungs(cells),
    }
    out_path = save_results(analysis, prefix=f"analysis_week7_evalplus_relabel")
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
