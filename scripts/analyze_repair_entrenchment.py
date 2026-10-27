"""Week 9 (Student B): does a repairer that stays near the failing program fail?

A re-analysis of stored repair traces — no model, no GPU.

THE QUESTION
------------
Week 6 (Student A) found that behavioral *consensus* among independent samples
ranks correctness, and that the feared "confidently wrong majority" was empty
in every cell. Repair asks a differently-shaped question: the model is not
sampling freely, it is being shown a wrong program and asked to change it. The
failure mode with a name in the LLM-repair literature is entrenchment — the
model re-emits the failing behavior with cosmetic edits instead of a different
solution.

That predicts something measurable in traces this project already has: the
*distance between a repair candidate and the program it was asked to repair*
should carry the candidate's own chance of passing, without executing any
test. If it does, it is a test-free signal about a patch — the quantity a
deployed repair loop needs and the one `rank_repair_candidates()` currently
gets from the oracle.

WHAT IS MEASURED
----------------
Level 1 (primary, no selection bias): the FIRST candidate of every coordinate.
Every coordinate has one regardless of how the run stopped, so the population
is not shaped by `--stop-on-pass`. Three distances from that candidate to the
failing program it replaces:

    ast_distance          ast_node_diversity over the pair (structural)
    token_jaccard         token_jaccard_diversity over the pair (lexical)
    behavior_disagreement execution_behavior_diversity over the pair's
                          EvalPlus probe signatures (functional, verdict-free)

scored against whether that candidate passed. Crash support is reported
alongside the behavioral number, per the Week 7 §(h) trap: a pair where
neither program ran carries no behavioral evidence at all, and the metric
still returns a number for it — under DEFAULT_ERROR_COSTS the constant 0.6
(same error type on every probe), under the legacy corner 0.0, i.e. perfect
agreement. Either way that value is an artefact of the cost table rather than
a measurement, so the pairs with shared running probes are scored separately.

READ THE RESTRICTED ROW AS A DIFFERENT POPULATION, NOT A FIXED ONE. A
candidate that passes the suite necessarily runs, so `all_crash` can only be
true for a *failure*: the restriction removes negatives only and raises the
base rate (measured on the HumanEval cohort: 29/148 = 19.6% to 29/90 =
32.2%). `crash_label_crosstab` is emitted next to the scored blocks so this
is visible in the output rather than inferred. The AST channel is the one to
watch, because `ast_node_diversity` scores an unparseable program at the
maximum distance 1.0: on the full population "large AST distance" largely
means "the patch did not parse", and the difference between the two rows is
which negatives were dropped.

Level 2 carries none of that entanglement — its label (did a LATER candidate
pass) is independent of the two candidates whose distance is measured — so it
is the cleaner test of the behavioral channel.

Level 2 (secondary, size-controlled): among coordinates whose first candidate
failed AND that generated a second one, the diversity *between* the first two
candidates, scored against whether any later candidate passed. Candidate count
is confounded with success under `--stop-on-pass` (a coordinate that succeeds
immediately never generates a third), so the set size is held at two.

Both levels report ROC-AUC with a permutation p-value and a bootstrap
interval, reusing `scripts/diagnose_gate_target.score_target` so the Week 8
and Week 9 tables are computed by the same code.

Usage:
    python scripts/analyze_repair_entrenchment.py \
        --repair-pool results/humaneval_week8_repair_all_sample0_qwen15b.json \
        --dataset humaneval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnose_gate_target import score_target
from src.evaluation import behavior_signature, extract_probe_inputs
from src.uncertainty import (
    ast_node_diversity,
    execution_behavior_diversity,
    token_jaccard_diversity,
)
from src.utils import (
    get_logger,
    load_dataset_split,
    load_evalplus_probe_inputs,
    save_results,
    set_all_seeds,
)

logger = get_logger(__name__)

#: A probe outcome that means the program never ran there. Two such outcomes
#: are *equal*, which is why they must not be read as agreement.
_CRASH_PREFIX = "error:unrunnable"


def _first_candidate(record: dict) -> dict | None:
    """Return the first generated candidate of a coordinate, if any."""
    for step in record["trace"]:
        for candidate in step.get("candidates", []):
            return candidate
    return None


def _candidates(record: dict) -> list[dict]:
    """Return every generated candidate of a coordinate, in generation order."""
    out = []
    for step in record["trace"]:
        out.extend(step.get("candidates", []))
    return out


def shared_probe_support(signatures: list[list[str]]) -> int:
    """Count probes where every program actually ran.

    Args:
        signatures: One probe-outcome list per program.
    Returns:
        Number of probe positions no program marked unrunnable.
    """
    if not signatures or not signatures[0]:
        return 0
    return sum(
        1
        for outcomes in zip(*signatures)
        if not any(str(o).startswith(_CRASH_PREFIX) for o in outcomes)
    )


def pair_features(
    programs: list[str],
    entry_point: str,
    probes: list[str],
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Structural, lexical and behavioral distance across a set of programs.

    Args:
        programs: Two or more complete runnable programs.
        entry_point: Function name the probes call.
        probes: Probe argument-list strings.
        timeout_s: Per-program replay limit.
    Returns:
        Dict of distances plus the shared probe support and an
        all-programs-crashed flag, so the behavioral number is never read
        without knowing whether anything ran.
    """
    signatures = [
        behavior_signature(program, entry_point, probes, timeout_s=timeout_s)
        for program in programs
    ]
    support = shared_probe_support(signatures)
    return {
        "ast_distance": ast_node_diversity(programs),
        "token_jaccard": token_jaccard_diversity(programs),
        "behavior_disagreement": execution_behavior_diversity(signatures),
        "identical": len(set(programs)) == 1,
        "probe_support": support,
        "n_probes": len(probes),
        "all_crash": support == 0,
    }


def build_rows(
    pool: dict,
    problems: dict[str, dict],
    probes_by_task: dict[str, list[str]],
    *,
    timeout_s: float = 5.0,
) -> tuple[list[dict], list[dict]]:
    """Build the Level 1 and Level 2 row sets from one repair pool.

    Args:
        pool: A loaded repair-experiment JSON.
        problems: task_id -> dataset record (entry_point, test_code).
        probes_by_task: task_id -> EvalPlus probe inputs.
        timeout_s: Per-program replay limit.
    Returns:
        (level 1 rows, level 2 rows). Coordinates with no probes are kept with
        `behavior_disagreement` scored over an empty probe set (0.0) and
        `n_probes` 0, so they can be excluded downstream rather than silently.
    """
    level1: list[dict] = []
    level2: list[dict] = []
    for index, record in enumerate(pool["records"]):
        task_id = record["task_id"]
        problem = problems.get(task_id)
        if problem is None:
            continue
        probes = probes_by_task.get(task_id) or extract_probe_inputs(
            problem["test_code"], problem["entry_point"]
        )
        candidates = _candidates(record)
        first = _first_candidate(record)
        if first is None:
            continue
        features = pair_features(
            [record["initial_code"], first["code"]],
            problem["entry_point"],
            probes,
            timeout_s=timeout_s,
        )
        level1.append(
            {
                "task_id": task_id,
                "sample_index": record["sample_index"],
                "initial_failure_category": record["initial_failure_category"],
                "candidate_passed": bool(first["passed"]),
                **features,
            }
        )
        if not first["passed"] and len(candidates) >= 2:
            pair = pair_features(
                [candidates[0]["code"], candidates[1]["code"]],
                problem["entry_point"],
                probes,
                timeout_s=timeout_s,
            )
            level2.append(
                {
                    "task_id": task_id,
                    "sample_index": record["sample_index"],
                    "later_candidate_passed": any(
                        candidate["passed"] for candidate in candidates[1:]
                    ),
                    **pair,
                }
            )
        if (index + 1) % 25 == 0:
            logger.info(
                "  %d/%d coordinates processed", index + 1, len(pool["records"])
            )
    return level1, level2


def crash_label_crosstab(rows: list[dict], label_key: str) -> dict[str, Any]:
    """Cross-tabulate all-crash pairs against the label.

    Makes the population shift explicit: a passing candidate must run, so
    every all-crash row is a negative and restricting to supported pairs
    deletes negatives only.

    Args:
        rows: Feature rows.
        label_key: Key holding the binary label.
    Returns:
        Counts per (all_crash, label) cell plus the base rate before and
        after the restriction.
    """
    cells = {
        "all_crash_positive": 0,
        "all_crash_negative": 0,
        "supported_positive": 0,
        "supported_negative": 0,
    }
    for row in rows:
        prefix = "all_crash" if row["all_crash"] else "supported"
        suffix = "positive" if row[label_key] else "negative"
        cells[f"{prefix}_{suffix}"] += 1
    supported = cells["supported_positive"] + cells["supported_negative"]
    total = supported + cells["all_crash_positive"] + cells["all_crash_negative"]
    return {
        **cells,
        "base_rate_all_rows": (
            (cells["all_crash_positive"] + cells["supported_positive"]) / total
            if total
            else None
        ),
        "base_rate_supported_rows": (
            cells["supported_positive"] / supported if supported else None
        ),
    }


def score_features(
    rows: list[dict],
    label_key: str,
    feature_keys: tuple[str, ...],
    *,
    seed: int,
    n_permutations: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Score each feature as a ranker of one binary label.

    Args:
        rows: Feature rows.
        label_key: Key holding the binary label.
        feature_keys: Features to score.
        seed: Seed forwarded to the permutation/bootstrap generators.
        n_permutations: Label shuffles per feature.
        n_bootstrap: Resamples per feature.
    Returns:
        Dict with per-feature AUC blocks and per-class means.
    """
    labels = np.array([bool(row[label_key]) for row in rows], dtype=bool)
    out: dict[str, Any] = {
        "n": len(rows),
        "n_positive": int(labels.sum()),
        "features": {},
    }
    for key in feature_keys:
        scores = np.array([float(row[key]) for row in rows], dtype=float)
        block = score_target(
            scores,
            labels,
            name=key,
            seed=seed,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
        )
        # score_target already carries mean_score_positive / mean_score_negative
        # and the bootstrap interval; nothing is recomputed here so the Week 8
        # and Week 9 tables cannot drift apart.
        out["features"][key] = block
    return out


def main() -> None:
    """CLI entry point: score entrenchment distances and persist the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-pool", type=Path, required=True)
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--probe-cap", type=int, default=12)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="analysis_week9_repair_entrenchment")
    args = parser.parse_args()
    set_all_seeds(args.seed)

    pool = json.loads(args.repair_pool.read_text())
    problems = {p["task_id"]: p for p in load_dataset_split(args.dataset)}
    probes_by_task = load_evalplus_probe_inputs(
        args.dataset, cap=args.probe_cap, seed=args.seed
    )
    level1, level2 = build_rows(
        pool, problems, probes_by_task, timeout_s=args.timeout_s
    )

    feature_keys = ("ast_distance", "token_jaccard", "behavior_disagreement")
    supported = [row for row in level1 if not row["all_crash"] and row["n_probes"]]
    results = {
        "config": {
            "repair_pool": str(args.repair_pool),
            "dataset": args.dataset,
            "repair_model": pool["config"].get("model"),
            "device_resolved": pool["config"].get("device_resolved"),
            "probe_cap": args.probe_cap,
            "timeout_s": args.timeout_s,
            "seed": args.seed,
            "n_permutations": args.n_permutations,
            "n_bootstrap": args.n_bootstrap,
        },
        "level1_first_candidate": score_features(
            level1,
            "candidate_passed",
            feature_keys,
            seed=args.seed,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
        ),
        # A RESTRICTED POPULATION, not a corrected one: see the module
        # docstring and the crosstabs below. Every dropped row is a negative.
        "level1_probe_supported": (
            score_features(
                supported,
                "candidate_passed",
                feature_keys,
                seed=args.seed,
                n_permutations=args.n_permutations,
                n_bootstrap=args.n_bootstrap,
            )
            if supported
            else None
        ),
        "level2_candidate_pair": (
            score_features(
                level2,
                "later_candidate_passed",
                feature_keys,
                seed=args.seed,
                n_permutations=args.n_permutations,
                n_bootstrap=args.n_bootstrap,
            )
            if level2
            else None
        ),
        "crash_label_crosstab_level1": crash_label_crosstab(level1, "candidate_passed"),
        "crash_label_crosstab_level2": (
            crash_label_crosstab(level2, "later_candidate_passed") if level2 else None
        ),
        "identical_to_initial": {
            "n": sum(row["identical"] for row in level1),
            "of": len(level1),
        },
        "all_crash_coordinates": sum(row["all_crash"] for row in level1),
        "rows_level1": level1,
        "rows_level2": level2,
    }
    path = save_results(results, prefix=f"{args.prefix}_{args.dataset}")

    print(f"\nSaved: {path}")
    cross = results["crash_label_crosstab_level1"]
    print(
        f"all-crash pairs: {cross['all_crash_negative']} negative, "
        f"{cross['all_crash_positive']} positive -> base rate "
        f"{cross['base_rate_all_rows']:.3f} over all rows vs "
        f"{cross['base_rate_supported_rows']:.3f} over supported rows"
    )
    for name in (
        "level1_first_candidate",
        "level1_probe_supported",
        "level2_candidate_pair",
    ):
        block = results[name]
        if not block:
            print(f"{name}: empty")
            continue
        print(f"\n{name}: n={block['n']}, positives={block['n_positive']}")
        for key, cell in block["features"].items():
            auc = cell.get("roc_auc")
            print(
                f"  {key:>22}: AUC {auc if auc is None else round(auc, 4)} "
                f"p={cell.get('permutation_p')} "
                f"CI={cell.get('bootstrap_95')} "
                f"means pass/fail: {cell.get('mean_score_positive')} / "
                f"{cell.get('mean_score_negative')}"
            )


if __name__ == "__main__":
    main()
