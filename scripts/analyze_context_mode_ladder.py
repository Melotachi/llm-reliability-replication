"""Week 9 (Student B): what does repair *context* buy at this capability gap?

A re-analysis of three stored repair arms — no model, no GPU.

THE QUESTION
------------
Every repair result this project has reported so far holds the context fixed:
the model is shown the failing program and its traceback, and asked to fix it.
That bundles two things a repair loop actually does — *drawing another sample*
and *conditioning that sample on the failure* — and only their sum was ever
measured. Week 8 needed the decomposition: its gate allocates repair attempts,
and an attempt whose context is net harmful is a different economic object
from one whose context helps.

The three arms are the ladder, run on the same 148 base-failing HumanEval
coordinates with the same seeds, temperature, candidate budget and ranking
rule; only the repair prompt changes:

    none        resample from the original problem prompt (no failing code,
                no traceback) — the control the repair literature rarely runs
    code_only   the failing program, no traceback
    error       the failing program AND its traceback (Weeks 2-8 behavior)

WHAT IS MEASURED
----------------
  * per-arm repair success with Wilson intervals, and model calls per success;
  * the *paired* comparison — the arms share coordinates, so the evidence is
    the discordant cell, scored with McNemar's mid-p (Fagerland et al., 2013)
    over all three pairs with Benjamini-Hochberg adjustment, not three
    independent proportions;
  * a bit-for-bit reproduction check against the Week 8 pool (identical
    `final_code` on every coordinate). The mechanism: candidate seeds are
    index-deterministic, so Week 8's `--stop-on-pass` truncates *generation*
    without changing which candidates exist. Agreement is therefore expected
    and its absence would indicate a harness change, not sampling noise;
  * entrenchment — how far each arm's patch travels from the program it was
    asked to repair (exact-copy count, token-Jaccard and AST distance to the
    initial program), and whether that distance ranks the patch's own success
    within the arm (ROC-AUC);
  * the per-initial-failure-category breakdown, since Week 8 established that
    repairability is governed by failure *kind*;
  * complementarity: which coordinates only the context arms solve, and which
    only the context-free arm solves;
  * a contamination check — similarity of each patch to the docstring-stripped
    canonical solution. REPORTED ALONGSIDE similarity-to-initial, which is the
    control: the patched arms inherit the failing program's structure by
    construction, so a lower canonical similarity there is the entrenchment
    finding, not evidence about memorization.

LABELS
------
`--label base` scores the shipped HumanEval suite; `--label plus` scores the
EvalPlus HumanEval+ labels written by `scripts/rescore_evalplus_labels.py
--repair-results`. The arms emit systematically different programs (the `none`
arm writes from scratch, the others patch), so a weak verifier's
false-positive rate need not be equal across them; whichever label the relabel
pass makes primary, the other stays computable from the same inputs.

Usage:
    python scripts/analyze_context_mode_ladder.py \
        --arms results/humaneval_week9_arm_{resample,code_only,error_nostop}_*.json \
        --week8-results results/humaneval_week8_repair_all_sample0_qwen15b.json \
        --label base
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import mcnemar_mid_p
from src.uncertainty import (
    ast_node_diversity,
    strip_prompt_docstring,
    token_jaccard_diversity,
)
from src.utils import get_logger, save_results

logger = get_logger(__name__)

DEFAULT_FIGURE_DIR = Path("reports/figures")

#: Display order of the ladder: least context first. Fixed rather than derived
#: from the input order so figures and tables are stable across invocations.
ARM_ORDER = ("none", "code_only", "error")

ARM_LABELS = {
    "none": "none (resample)",
    "code_only": "code_only",
    "error": "error + code",
}


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Wilson score interval, matching run_iterative_repair_experiment.py."""
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


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjustment, matching analyze_smart_repair.py."""
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


def _roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-based ROC-AUC; None when either class is absent.

    Written here rather than imported so this script has no sklearn
    dependency for one number; ties get the mid-rank, as sklearn does.
    """
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        return None
    order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = np.asarray(scores, dtype=float)[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = float(sum(r for r, y in zip(ranks, labels) if y))
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _passed_key(label: str) -> str:
    """Record key holding the requested correctness label."""
    return "final_passed" if label == "base" else "final_passed_plus"


def load_arms(paths: list[Path], label: str) -> dict[str, dict[str, Any]]:
    """Load the arm runs, keyed by context mode, and validate their alignment.

    Args:
        paths: Arm JSONs (raw, or EvalPlus-relabelled when label == "plus").
        label: "base" or "plus"; decides which record key must be present.
    Returns:
        Dict context_mode -> {"run", "path", "records"} with records keyed by
        task_id.
    Raises:
        ValueError: If a mode repeats, the requested label is missing, or the
            arms do not cover an identical coordinate set (which would make
            every paired comparison invalid).
    """
    key = _passed_key(label)
    arms: dict[str, dict[str, Any]] = {}
    for path in paths:
        run = load_json(path)
        mode = run["config"].get("context_mode")
        if mode is None:
            raise ValueError(f"{path} has no config.context_mode")
        if mode in arms:
            raise ValueError(f"Two arms share context_mode {mode!r}")
        if key not in run["records"][0]:
            raise ValueError(
                f"{path} carries no {key!r}; for --label plus pass the outputs of "
                f"rescore_evalplus_labels.py --repair-results"
            )
        arms[mode] = {
            "run": run,
            "path": str(path),
            "records": {r["task_id"]: r for r in run["records"]},
        }

    coordinate_sets = [set(a["records"]) for a in arms.values()]
    if any(s != coordinate_sets[0] for s in coordinate_sets):
        raise ValueError("Arms do not share an identical coordinate set")
    for mode, arm in arms.items():
        for task_id, record in arm["records"].items():
            reference = next(iter(arms.values()))["records"][task_id]["initial_code"]
            if record["initial_code"] != reference:
                raise ValueError(
                    f"Arm {mode} repairs a different program at {task_id}; "
                    "the arms are not paired"
                )
    return arms


def ordered_modes(arms: dict[str, Any]) -> list[str]:
    """Arms in ladder order, with any unrecognized mode appended."""
    known = [m for m in ARM_ORDER if m in arms]
    return known + sorted(set(arms) - set(known))


def arm_summaries(arms: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """Per-arm success, interval and cost.

    Args:
        arms: The return value of load_arms().
        label: "base" or "plus".
    Returns:
        One dict per arm, in ladder order.
    """
    key = _passed_key(label)
    out = []
    for mode in ordered_modes(arms):
        arm = arms[mode]
        records = list(arm["records"].values())
        successes = sum(bool(r[key]) for r in records)
        calls = sum(r["repair_model_calls"] for r in records)
        summary = arm["run"]["summary"]
        out.append(
            {
                "context_mode": mode,
                "run_id": arm["run"].get("run_id"),
                "source_results_file": arm["path"],
                "n_coordinates": len(records),
                "successes": successes,
                "repair_success_rate": successes / len(records) if records else None,
                "repair_success_wilson_95": _wilson_interval(successes, len(records)),
                "repair_model_calls": calls,
                "model_calls_per_success": (calls / successes) if successes else None,
                "duplicate_candidate_rate": summary.get("duplicate_candidate_rate"),
                "stagnation_stops": summary.get("stagnation_stops"),
                "final_failure_categories": summary.get("final_failure_categories"),
            }
        )
    return out


def paired_comparisons(arms: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """McNemar mid-p over every arm pair, BH-adjusted.

    Args:
        arms: The return value of load_arms().
        label: "base" or "plus".
    Returns:
        One dict per unordered pair, carrying the 2x2's discordant cells.
    """
    key = _passed_key(label)
    modes = ordered_modes(arms)
    task_ids = sorted(next(iter(arms.values()))["records"])
    outcomes = {m: [bool(arms[m]["records"][t][key]) for t in task_ids] for m in modes}
    comparisons = []
    for i, left in enumerate(modes):
        for right in modes[i + 1 :]:
            a, b = outcomes[left], outcomes[right]
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "both_pass": sum(x and y for x, y in zip(a, b)),
                    "left_only_pass": sum(x and not y for x, y in zip(a, b)),
                    "right_only_pass": sum(y and not x for x, y in zip(a, b)),
                    "neither_pass": sum((not x) and (not y) for x, y in zip(a, b)),
                    "mcnemar_mid_p": mcnemar_mid_p(a, b),
                }
            )
    adjusted = _bh_adjust([c["mcnemar_mid_p"] for c in comparisons])
    for comparison, p_adj in zip(comparisons, adjusted):
        comparison["mcnemar_mid_p_bh"] = p_adj
    return comparisons


def week8_reproduction(arms: dict[str, Any], week8_run: dict[str, Any]) -> dict:
    """Check the `error` arm reproduces the Week 8 pool bit-for-bit.

    Week 8 ran the same protocol with `--stop-on-pass`, which truncates
    candidate *generation* (368 calls vs 444) without changing which
    candidates exist, because candidate seeds are index-deterministic. Every
    coordinate's selected program should therefore be identical; disagreement
    would indicate a harness change rather than sampling noise.

    Args:
        arms: The return value of load_arms().
        week8_run: The loaded Week 8 pool JSON.
    Returns:
        Dict of agreement counts, or a "skipped" note when no `error` arm is
        present.
    """
    if "error" not in arms:
        return {"skipped": "no error arm in this invocation"}
    week9 = arms["error"]["records"]
    week8 = {r["task_id"]: r for r in week8_run["records"]}
    shared = sorted(set(week9) & set(week8))
    return {
        "n_week9_coordinates": len(week9),
        "n_week8_coordinates": len(week8),
        "n_shared_coordinates": len(shared),
        "identical_final_code": sum(
            week9[t]["final_code"] == week8[t]["final_code"] for t in shared
        ),
        "agreeing_final_passed": sum(
            bool(week9[t]["final_passed"]) == bool(week8[t]["final_passed"])
            for t in shared
        ),
        "week9_successes": sum(bool(week9[t]["final_passed"]) for t in shared),
        "week8_successes": sum(bool(week8[t]["final_passed"]) for t in shared),
        "week9_repair_model_calls": arms["error"]["run"]["summary"][
            "repair_model_calls"
        ],
        "week8_repair_model_calls": week8_run["summary"]["repair_model_calls"],
        "week8_stop_on_pass": week8_run["config"].get("stop_on_pass"),
        "week9_stop_on_pass": arms["error"]["run"]["config"].get("stop_on_pass"),
    }


def _pair_distances(left: str, right: str) -> dict[str, float]:
    """Token and AST distance between two programs (Week 2/3 metrics)."""
    return {
        "token_jaccard": float(token_jaccard_diversity([left, right])),
        "ast_distance": float(ast_node_diversity([left, right])),
    }


def entrenchment(arms: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """How far each arm's patch travels from the program it was asked to repair.

    READ THE RESTRICTED ROW, NOT THE FULL ONE, FOR THE RANKING CLAIM. When no
    candidate outranks the current program the harness keeps it, so
    `final_code == initial_code`; those coordinates have distance exactly 0.0
    *and* are guaranteed failures (the coordinate is in the pool because that
    program failed, and no initial program passes the plus suite either). They
    are therefore correctly-ranked negatives by construction — a third of the
    context arms' pairs — and they inflate the unrestricted ROC-AUC without
    measuring anything. `changed_label_crosstab` makes the size of that block
    visible; every distance statistic is reported twice, over all coordinates
    and over the coordinates where the harness actually emitted a changed
    program. Structurally the same trap as the Week 7 Section (h) crash
    control and handled the same way (see scripts/analyze_repair_entrenchment.py).

    The restricted row is a different population, not a fixed one: it drops
    negatives only, so its base rate is higher.

    Args:
        arms: The return value of load_arms().
        label: "base" or "plus".
    Returns:
        One dict per arm: the stagnation count, the crosstab, and mean
        distances plus within-arm ROC-AUC over both populations.
    """
    key = _passed_key(label)
    out = []
    for mode in ordered_modes(arms):
        records = list(arms[mode]["records"].values())
        rows = []
        for record in records:
            distances = _pair_distances(record["initial_code"], record["final_code"])
            distances["passed"] = bool(record[key])
            distances["changed"] = record["final_code"] != record["initial_code"]
            distances["final_chars"] = len(record["final_code"])
            rows.append(distances)
        changed_rows = [r for r in rows if r["changed"]]
        cell: dict[str, Any] = {
            "context_mode": mode,
            "n_coordinates": len(rows),
            "n_final_identical_to_initial": sum(not r["changed"] for r in rows),
            "n_changed": len(changed_rows),
            "changed_label_crosstab": {
                "changed_passed": sum(r["changed"] and r["passed"] for r in rows),
                "changed_failed": sum(r["changed"] and not r["passed"] for r in rows),
                "identical_passed": sum(
                    (not r["changed"]) and r["passed"] for r in rows
                ),
                "identical_failed": sum(
                    (not r["changed"]) and not r["passed"] for r in rows
                ),
            },
            "mean_final_chars": float(np.mean([r["final_chars"] for r in rows])),
            "mean_final_chars_changed": (
                float(np.mean([r["final_chars"] for r in changed_rows]))
                if changed_rows
                else None
            ),
        }
        for population, subset, suffix in (
            ("all", rows, ""),
            ("changed", changed_rows, "_changed"),
        ):
            passed = [r for r in subset if r["passed"]]
            failed = [r for r in subset if not r["passed"]]
            for metric in ("token_jaccard", "ast_distance"):
                values = [r[metric] for r in subset]
                cell[f"mean_{metric}_to_initial{suffix}"] = (
                    float(np.mean(values)) if values else None
                )
                cell[f"mean_{metric}_to_initial_passed{suffix}"] = (
                    float(np.mean([r[metric] for r in passed])) if passed else None
                )
                cell[f"mean_{metric}_to_initial_failed{suffix}"] = (
                    float(np.mean([r[metric] for r in failed])) if failed else None
                )
                cell[f"roc_auc_{metric}_vs_success{suffix}"] = _roc_auc(
                    values, [r["passed"] for r in subset]
                )
        out.append(cell)
    return out


def category_breakdown(arms: dict[str, Any], label: str) -> dict[str, Any]:
    """Success by initial failure category, per arm.

    Week 8 established that repairability is governed by failure *kind*, so
    an arm-level difference that is concentrated in one category is a
    different claim from one spread across all of them.

    Args:
        arms: The return value of load_arms().
        label: "base" or "plus".
    Returns:
        Dict with the category order and per-arm counts/rates.
    """
    key = _passed_key(label)
    reference = next(iter(arms.values()))["records"]
    categories = sorted({r["initial_failure_category"] for r in reference.values()})
    rows = {}
    for mode in ordered_modes(arms):
        records = arms[mode]["records"]
        per_category = {}
        for category in categories:
            subset = [
                r for r in records.values() if r["initial_failure_category"] == category
            ]
            successes = sum(bool(r[key]) for r in subset)
            per_category[category] = {
                "failures": len(subset),
                "successful_repairs": successes,
                "repair_success_rate": successes / len(subset) if subset else None,
                "repair_success_wilson_95": _wilson_interval(successes, len(subset)),
            }
        rows[mode] = per_category
    return {"categories": categories, "by_arm": rows}


def complementarity(arms: dict[str, Any], label: str) -> dict[str, Any]:
    """Which coordinates each arm — and only that arm — solves.

    Args:
        arms: The return value of load_arms().
        label: "base" or "plus".
    Returns:
        Dict of the union, the intersection, and the two directional
        differences between the context-free arm and the context arms.
    """
    key = _passed_key(label)
    solved = {
        mode: {t for t, r in arm["records"].items() if bool(r[key])}
        for mode, arm in arms.items()
    }
    all_ids = set(next(iter(arms.values()))["records"])
    union = set().union(*solved.values()) if solved else set()
    intersection = set.intersection(*solved.values()) if len(solved) > 1 else set(union)
    context_modes = [m for m in solved if m != "none"]
    context_union = (
        set().union(*(solved[m] for m in context_modes)) if context_modes else set()
    )
    none_solved = solved.get("none", set())
    return {
        "n_coordinates": len(all_ids),
        "solved_by_arm": {m: len(v) for m, v in solved.items()},
        "n_union": len(union),
        "union_rate": len(union) / len(all_ids) if all_ids else None,
        "n_intersection": len(intersection),
        "n_unsolved_by_every_arm": len(all_ids - union),
        "n_context_only": len(context_union - none_solved),
        "context_only_task_ids": sorted(context_union - none_solved),
        "n_none_only": len(none_solved - context_union),
        "none_only_task_ids": sorted(none_solved - context_union),
        "solved_only_by_arm": {
            m: sorted(v - set().union(*(solved[o] for o in solved if o != m)))
            for m, v in solved.items()
        },
    }


def contamination(
    arms: dict[str, Any],
    canonical_by_task: dict[str, str],
    label: str,
) -> list[dict[str, Any]]:
    """Similarity of each arm's patches to the reference solution.

    Both sides are docstring- and comment-stripped first (Week 7's
    `strip_prompt_docstring`), so the shared problem prompt text does not
    inflate the number.

    READ THIS ONLY NEXT TO similarity-to-initial. The patched arms inherit the
    failing program's structure by construction, so a lower canonical
    similarity there is the entrenchment finding restated, not evidence about
    memorization. The comparison that is informative about contamination is
    *within* an arm: whether the coordinates it solves are the ones whose
    patches sit unusually close to the canonical solution.

    Args:
        arms: The return value of load_arms().
        canonical_by_task: task_id -> reference solution (prompt + body).
        label: "base" or "plus".
    Returns:
        One dict per arm.
    """
    key = _passed_key(label)
    out = []
    for mode in ordered_modes(arms):
        canonical_similarity, initial_similarity, labels = [], [], []
        for task_id, record in arms[mode]["records"].items():
            canonical = canonical_by_task.get(task_id)
            if canonical is None:
                continue
            final = strip_prompt_docstring(record["final_code"])
            canonical_similarity.append(
                1.0
                - token_jaccard_diversity([final, strip_prompt_docstring(canonical)])
            )
            initial_similarity.append(
                1.0
                - token_jaccard_diversity(
                    [final, strip_prompt_docstring(record["initial_code"])]
                )
            )
            labels.append(bool(record[key]))
        passed = [s for s, y in zip(canonical_similarity, labels) if y]
        failed = [s for s, y in zip(canonical_similarity, labels) if not y]
        out.append(
            {
                "context_mode": mode,
                "n_scored": len(canonical_similarity),
                "mean_similarity_to_canonical": (
                    float(np.mean(canonical_similarity))
                    if canonical_similarity
                    else None
                ),
                "mean_similarity_to_canonical_passed": (
                    float(np.mean(passed)) if passed else None
                ),
                "mean_similarity_to_canonical_failed": (
                    float(np.mean(failed)) if failed else None
                ),
                "max_similarity_to_canonical": (
                    float(np.max(canonical_similarity))
                    if canonical_similarity
                    else None
                ),
                "n_above_0_9_similarity_to_canonical": int(
                    sum(s >= 0.9 for s in canonical_similarity)
                ),
                "mean_similarity_to_initial": (
                    float(np.mean(initial_similarity)) if initial_similarity else None
                ),
                "roc_auc_canonical_similarity_vs_success": _roc_auc(
                    canonical_similarity, labels
                ),
            }
        )
    return out


def build_analysis(
    arms: dict[str, Any],
    *,
    label: str,
    week8_run: dict[str, Any] | None,
    canonical_by_task: dict[str, str] | None,
    dataset: str,
) -> dict[str, Any]:
    """Assemble every block into the analysis dict written to results/."""
    analysis = {
        "config": {
            "dataset": dataset,
            "label": label,
            "arms": {m: arms[m]["path"] for m in ordered_modes(arms)},
            "model": next(iter(arms.values()))["run"]["config"].get("model"),
            "device_resolved": next(iter(arms.values()))["run"]["config"].get(
                "device_resolved"
            ),
            "dtype_resolved": next(iter(arms.values()))["run"]["config"].get(
                "dtype_resolved"
            ),
            "temperature": next(iter(arms.values()))["run"]["config"].get(
                "temperature"
            ),
            "seed": next(iter(arms.values()))["run"]["config"].get("seed"),
            "selection_sha256": next(iter(arms.values()))["run"]["config"].get(
                "selection_sha256"
            ),
        },
        "arms": arm_summaries(arms, label),
        "paired_comparisons": paired_comparisons(arms, label),
        "entrenchment": entrenchment(arms, label),
        "category_breakdown": category_breakdown(arms, label),
        "complementarity": complementarity(arms, label),
    }
    if week8_run is not None:
        analysis["week8_reproduction"] = week8_reproduction(arms, week8_run)
    if canonical_by_task:
        analysis["contamination"] = contamination(arms, canonical_by_task, label)
    return analysis


def plot_ladder(analysis: dict[str, Any], output: Path) -> None:
    """Three-panel figure: the ladder, its complementarity, and entrenchment."""
    arms = analysis["arms"]
    modes = [a["context_mode"] for a in arms]
    names = [ARM_LABELS.get(m, m) for m in modes]
    colors = ["#2e8b57", "#245b78", "#d05a3a"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    rates = [a["repair_success_rate"] for a in arms]
    lows = [r - a["repair_success_wilson_95"][0] for r, a in zip(rates, arms)]
    highs = [a["repair_success_wilson_95"][1] - r for r, a in zip(rates, arms)]
    axes[0].bar(names, rates, color=colors[: len(names)], width=0.6)
    axes[0].errorbar(
        names, rates, yerr=[lows, highs], fmt="none", ecolor="#333333", capsize=4
    )
    for index, arm in enumerate(arms):
        axes[0].text(
            index,
            arm["repair_success_wilson_95"][1] + 0.02,
            f"{arm['successes']}/{arm['n_coordinates']}",
            ha="center",
            fontsize=9,
        )
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel(f"Repair success ({analysis['config']['label']} label)")
    axes[0].set_title("Context-mode ladder")

    overlap = analysis["complementarity"]
    bars = {
        "context-free only": overlap["n_none_only"],
        "context only": overlap["n_context_only"],
        "both": overlap["n_union"] - overlap["n_none_only"] - overlap["n_context_only"],
        "unsolved": overlap["n_unsolved_by_every_arm"],
    }
    axes[1].bar(
        list(bars),
        list(bars.values()),
        color=["#2e8b57", "#d05a3a", "#245b78", "#999999"],
    )
    for index, value in enumerate(bars.values()):
        axes[1].text(index, value + 1, str(value), ha="center", fontsize=9)
    axes[1].set_ylabel("Coordinates")
    axes[1].set_title(f"Who solves what (union {overlap['n_union']})")
    axes[1].tick_params(axis="x", labelrotation=20)

    width = 0.36
    positions = np.arange(len(modes))
    passed = [
        c["mean_token_jaccard_to_initial_passed"] for c in analysis["entrenchment"]
    ]
    failed = [
        c["mean_token_jaccard_to_initial_failed"] for c in analysis["entrenchment"]
    ]
    axes[2].bar(
        positions - width / 2, passed, width, label="repair passed", color="#2e8b57"
    )
    axes[2].bar(
        positions + width / 2, failed, width, label="repair failed", color="#d05a3a"
    )
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(names)
    axes[2].set_ylabel("Token-Jaccard distance to initial")
    axes[2].set_title("Entrenchment")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.25, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, type=Path)
    parser.add_argument("--label", choices=("base", "plus"), default="base")
    parser.add_argument("--week8-results", type=Path, default=None)
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument(
        "--no-contamination",
        action="store_true",
        help="Skip the canonical-similarity block (avoids the EvalPlus download).",
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--figure-name", default=None)
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    arms = load_arms(args.arms, args.label)
    week8_run = load_json(args.week8_results) if args.week8_results else None

    canonical_by_task = None
    if not args.no_contamination:
        from src.utils import load_evalplus_canonical_solutions

        canonical_by_task = load_evalplus_canonical_solutions(args.dataset)

    analysis = build_analysis(
        arms,
        label=args.label,
        week8_run=week8_run,
        canonical_by_task=canonical_by_task,
        dataset=args.dataset,
    )

    figure_name = args.figure_name or (
        f"week9_context_mode_ladder_{args.dataset}"
        + ("" if args.label == "base" else f"_{args.label}")
        + ".png"
    )
    figure_path = args.figure_dir / figure_name
    plot_ladder(analysis, figure_path)
    analysis["figure"] = str(figure_path)

    prefix = (
        args.prefix or f"analysis_week9_context_mode_ladder_{args.dataset}_{args.label}"
    )
    out_path = save_results(analysis, prefix=prefix)
    logger.info("wrote %s", out_path)
    logger.info("wrote %s", figure_path)

    for arm in analysis["arms"]:
        logger.info(
            "%s: %d/%d = %.4f [%.3f, %.3f] | %d calls | %.2f calls/success",
            arm["context_mode"],
            arm["successes"],
            arm["n_coordinates"],
            arm["repair_success_rate"],
            *arm["repair_success_wilson_95"],
            arm["repair_model_calls"],
            arm["model_calls_per_success"] or float("nan"),
        )
    for comparison in analysis["paired_comparisons"]:
        logger.info(
            "%s vs %s: %d/%d discordant, mid-p = %.4g (BH %.4g)",
            comparison["left"],
            comparison["right"],
            comparison["left_only_pass"],
            comparison["right_only_pass"],
            comparison["mcnemar_mid_p"],
            comparison["mcnemar_mid_p_bh"],
        )


if __name__ == "__main__":
    main()
