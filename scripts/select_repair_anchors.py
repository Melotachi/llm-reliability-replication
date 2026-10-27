"""Objective 3 follow-up: pick a per-problem repair ANCHOR, not always sample 0.

Every Objective 3 repair pool to date has repaired `sample_index = 0`, an
arbitrary draw from the pipeline's 10 stored samples per problem. Week 6's
execution-behavior consensus selector (argmin
`execution_behavior_self_distance`, verdict-free) lifts pass@1 0.495 -> 0.700
on one cell but was never combined with repair -- Week 6's own lesson was
"prediction != selection: this apparatus works as a SELECTOR and fails as an
ALLOCATOR." So use it where it works: to choose which of a problem's N
failing samples the repairer sees, instead of always sample 0.

Cohort: sample-0-failing HumanEval coordinates from a stored uncertainty run,
further restricted to problems with **no** passing sibling at all -- on a
problem with a passing sibling, simply revealing that sibling already solves
it for free, so repairing anything there is the wrong action and it is
excluded (CLAUDE.md/task framing: this is expected to drop ~148 to ~128).

Anchor selection reuses the exact selector logic in
`scripts/evaluate_selection.py`'s `select_per_problem()`
(argmin `execution_behavior_self_consistency`, `src/uncertainty.py`) and the
signature cache already built by `scripts/diagnose_repairability.py`
(`compute_signatures()` / `signature_cache_path()`, keyed on
run_id/probe_cap/timeout_s/seed) -- reused here via import, not
reimplemented.

Known hazard (Week 7 crash-rate control, documented in CLAUDE.md): a sample
that crashes on every probe has no co-running probe with most siblings, and
DEFAULT_ERROR_COSTS' graded costs dull but do not remove the "crashiest
sample looks most confident" trap for such a sample (two same-type crashes
still cost only 0.6, not 1.0). So a sample with `execution_crash_rate == 1.0`
is excluded from the candidate pool unless *every* sample of that problem is
all-crash, in which case the pool falls back to all samples and the case is
counted separately (`n_all_crash_fallback`) rather than silently repaired.

Usage:
    python scripts/select_repair_anchors.py
    python scripts/select_repair_anchors.py --source-results results/... \
        --probe-cap 12 --execution-timeout 5.0 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnose_repairability import (  # noqa: E402
    compute_signatures,
    signature_cache_path,
)
from src.uncertainty import (  # noqa: E402
    DEFAULT_ERROR_COSTS,
    LEGACY_ERROR_COSTS,
    execution_behavior_self_consistency,
    execution_crash_rate,
)
from src.utils import get_logger, save_results, set_all_seeds  # noqa: E402

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("results/humaneval_uncertainty_merged164_20260711.json")
DEFAULT_CACHE_DIR = Path("results/signature_cache")


def build_cohort(source: dict) -> tuple[list[str], list[str]]:
    """Sample-0-failing problems with no passing sibling anywhere.

    A problem with any passing sample is excluded even though sample 0
    itself fails: revealing that passing sibling already solves the problem
    for free, so repairing anything there is the wrong action to measure.

    Args:
        source: A pipeline results dict (CLAUDE.md Section 8 schema).
    Returns:
        (cohort_task_ids, excluded_task_ids_with_a_passing_sample), both
        sorted for determinism.
    """
    cohort: list[str] = []
    excluded: list[str] = []
    for problem in source["per_problem"]:
        samples = problem.get("samples", [])
        if not samples or samples[0]["passed"]:
            continue  # not a sample-0 failure at all
        if any(s["passed"] for s in samples):
            excluded.append(problem["task_id"])
        else:
            cohort.append(problem["task_id"])
    return sorted(cohort), sorted(excluded)


def select_anchor(
    signatures: list[list[str]],
    error_costs: tuple[float, float, float],
) -> dict[str, Any]:
    """Argmin execution-behavior self-distance anchor, with the crash guard.

    Replicates `scripts/evaluate_selection.py`'s `select_per_problem()`
    execution_heuristic (argmin `execution_behavior_self_consistency`), but
    additionally excludes all-crash samples from the candidate pool -- see
    module docstring for why.

    Args:
        signatures: One outcome list per sample (behavior_signature()
            output), all over the same probes.
        error_costs: Graded abnormal-outcome costs (DEFAULT_ERROR_COSTS or
            LEGACY_ERROR_COSTS).
    Returns:
        Dict with sample_index, selector_score (the winning self-distance),
        all_crash (whether the winner itself is all-crash -- only True when
        the fallback fired), support (count of probes on which the winning
        sample produced a non-error outcome, i.e. how much real evidence
        backs its score), candidate_pool_size, and fallback_used.
    """
    n = len(signatures)
    crash_rates = [execution_crash_rate(s) for s in signatures]
    self_distances = execution_behavior_self_consistency(signatures, error_costs)
    candidate_pool = [i for i in range(n) if crash_rates[i] < 1.0]
    fallback_used = False
    if not candidate_pool:
        candidate_pool = list(range(n))
        fallback_used = True
    best_index = min(candidate_pool, key=lambda i: (self_distances[i], i))
    support = sum(
        1 for outcome in signatures[best_index] if not outcome.startswith("error:")
    )
    return {
        "sample_index": best_index,
        "selector_score": float(self_distances[best_index]),
        "all_crash": bool(crash_rates[best_index] == 1.0),
        "support": support,
        "candidate_pool_size": len(candidate_pool),
        "fallback_used": fallback_used,
    }


def build_anchor_map(
    cache: dict[str, dict[str, Any]],
    cohort: list[str],
    error_costs: tuple[float, float, float],
) -> tuple[dict[str, dict[str, Any]], int]:
    """Select one anchor per cohort problem from cached behavior signatures.

    Args:
        cache: task_id -> {"signatures": [...], ...} from compute_signatures().
        cohort: Cohort task_ids (build_cohort() output).
        error_costs: Forwarded to select_anchor().
    Returns:
        (task_id -> anchor dict, count of problems where the all-crash
        fallback fired).
    """
    anchors: dict[str, dict[str, Any]] = {}
    n_fallback = 0
    for task_id in cohort:
        entry = cache.get(task_id)
        if entry is None:
            logger.warning(
                "No cached signatures for %s; excluded from the anchor map", task_id
            )
            continue
        anchor = select_anchor(entry["signatures"], error_costs)
        if anchor["fallback_used"]:
            n_fallback += 1
        anchors[task_id] = anchor
    return anchors, n_fallback


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the anchor-selection entry point."""
    parser = argparse.ArgumentParser(
        description="Select a per-problem repair anchor with the Week 6 "
        "execution-behavior consensus selector"
    )
    parser.add_argument(
        "--source-results",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Uncertainty run holding all N samples per problem.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Behavior-signature cache directory (scripts/diagnose_repairability.py).",
    )
    parser.add_argument("--probe-cap", type=int, default=12)
    parser.add_argument("--execution-timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--legacy-error-costs",
        action="store_true",
        help="Use the (1, 1, 0) abnormal-outcome costs instead of the graded "
        "He et al. default (1, 0.8, 0.6). Continuity arm.",
    )
    parser.add_argument(
        "--refresh-signatures",
        action="store_true",
        help="Recompute behavior signatures even if a cache file exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the summary but skip persisting the anchor map JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    source = json.loads(args.source_results.read_text(encoding="utf-8"))
    cohort, excluded = build_cohort(source)
    n_sample0_failing = sum(
        1
        for p in source["per_problem"]
        if p.get("samples") and not p["samples"][0]["passed"]
    )
    logger.info(
        "sample-0 failing: %d | cohort (no passing sibling): %d | "
        "excluded (has a passing sibling): %d",
        n_sample0_failing,
        len(cohort),
        len(excluded),
    )

    error_costs = LEGACY_ERROR_COSTS if args.legacy_error_costs else DEFAULT_ERROR_COSTS
    cache_path = signature_cache_path(
        source,
        cache_dir=args.cache_dir,
        timeout_s=args.execution_timeout,
        probe_cap=args.probe_cap,
        seed=args.seed,
    )
    cache = compute_signatures(
        source,
        timeout_s=args.execution_timeout,
        probe_cap=args.probe_cap,
        seed=args.seed,
        cache_path=cache_path,
        refresh=args.refresh_signatures,
    )

    anchors, n_fallback = build_anchor_map(cache, cohort, error_costs)
    n_nonzero = sum(1 for a in anchors.values() if a["sample_index"] != 0)

    payload = {
        "run_id": source.get("run_id"),
        "generated_at": datetime.now().isoformat(),
        "config": {
            "source_results": str(args.source_results),
            "probe_cap": args.probe_cap,
            "execution_timeout": args.execution_timeout,
            "seed": args.seed,
            "error_costs": list(error_costs),
            "signature_cache_path": str(cache_path),
        },
        "cohort": {
            "n_sample0_failing": n_sample0_failing,
            "n_cohort": len(cohort),
            "excluded_has_passing_sibling": excluded,
        },
        "anchor_summary": {
            "n_anchors": len(anchors),
            "n_anchor_nonzero": n_nonzero,
            "n_all_crash_fallback": n_fallback,
        },
        "anchors": anchors,
    }
    print(
        f"cohort={len(cohort)}  anchors_selected={len(anchors)}  "
        f"anchors_nonzero={n_nonzero}  all_crash_fallback={n_fallback}"
    )

    if args.dry_run:
        logger.info("[DRY RUN] skipping anchor map JSON persistence")
    else:
        out_path = save_results(payload, prefix="repair_anchor_map_humaneval")
        logger.info("Saved anchor map to %s", out_path)


if __name__ == "__main__":
    main()
