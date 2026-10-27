"""Week 11 field audit, execution family: control C3 on published definitions.

Scores hypothesis H6 of ``reports/week11_field_audit_preregistration.md``:
restricted to samples that actually run, do the two *published* execution
conventions still rank correctness?

Why this is the sharpest part of the audit. Week 7 established that this
project's execution signal survives the runnability/agreement decomposition
(agreement alone ROC 0.77-0.90 among running code, runnability alone
0.53-0.56). The literature check in ``week10_novelty_verification.md`` §1
then found that **none of five execution-based estimators runs that
decomposition** -- but three of them assign *max* distance to crashed pairs,
so they are not naively broken either. That leaves an open question this
script answers: does the missing control actually cost those definitions
anything on a corpus where 13-37% of sample x probe cells crash?

Two published conventions are compared on one replay:

  graded (2605.09023 Eq. 2)   crashed pairs land at 0.6-1.0 depending on the
                              error types -- a *graded* penalty
  isolation (2607.12273 4.3.2) a syntax-error program has zero similarity to
                              everything; a both-timeout probe contributes
                              nothing -- a *hard* penalty

and each is reported over all samples and over the support>0 subpopulation,
which is the honest form of "among code that runs, does agreement predict
correctness?".

Behaviour signatures are cached to disk. The replay is the expensive step
(one subprocess per sample), both definitions read the same signatures, and
the cache makes re-scoring free -- which matters because the scoring
criteria may be applied more than once while the report is written, and
re-replaying would silently re-time-out different samples.

# TODO: SANDBOX -- replays stored generated code in subprocesses (same
# caveat as evaluation.run_code). Research/local use only.

Usage::

    python scripts/audit_published_execution.py --jobs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_published_metrics import CELLS  # noqa: E402
from scripts.diagnose_crash_control import nocrash_support  # noqa: E402
from src import published_metrics as pm  # noqa: E402
from src.evaluation import behavior_signature, extract_probe_inputs  # noqa: E402
from src.uncertainty import (  # noqa: E402
    DEFAULT_ERROR_COSTS,
    execution_behavior_self_consistency,
    execution_crash_rate,
)
from src.utils import (  # noqa: E402
    get_logger,
    load_dataset_split,
    load_evalplus_probe_inputs,
    save_results,
    set_all_seeds,
)

logger = get_logger(__name__)

#: The four execution-capable cells H6 is scored over. These are the same
#: four the Week 6/7 execution results were established on, so the audit's
#: denominator matches the project's own prior numbers rather than being
#: chosen fresh.
EXECUTION_CELLS = ("cg_he164", "ds_he60", "cg_mbppplus", "ds_mbppplus")

#: Pre-registration §7 H6: retained ROC-AUC floor, and how many cells must
#: clear it.
H6_ROC_FLOOR = 0.70
H6_MIN_CELLS = 3


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells", nargs="+", default=list(EXECUTION_CELLS), choices=list(CELLS)
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/.signature_cache"),
        help="Where behaviour signatures are cached between runs.",
    )
    parser.add_argument(
        "--probe-source", default="evalplus", choices=["evalplus", "test"]
    )
    parser.add_argument("--probe-cap", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore any cached signatures and replay from scratch.",
    )
    return parser.parse_args()


def _cache_path(cache_dir: Path, run_file: str, args: argparse.Namespace) -> Path:
    """Cache key covers everything that changes a signature.

    Probe source, cap, seed and timeout all change what is replayed or what
    comes back, so a cache keyed on the run file alone would silently serve
    signatures from a different protocol -- the same class of silent join
    that forced the Week 7 device-matched regeneration.
    """
    key = (
        f"{run_file}|{args.probe_source}|{args.probe_cap}|{args.seed}|{args.timeout_s}"
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir / f"{Path(run_file).stem}_{digest}.json"


def build_signatures(
    results: dict, args: argparse.Namespace
) -> dict[str, list[list[str]]]:
    """Replay every stored sample on its problem's probes; task_id -> signatures."""
    dataset_name = results.get("config", {}).get("dataset", "humaneval")
    problems = {p["task_id"]: p for p in load_dataset_split(dataset_name)}
    evalplus_probes = (
        load_evalplus_probe_inputs(dataset_name, cap=args.probe_cap, seed=args.seed)
        if args.probe_source == "evalplus"
        else {}
    )
    signatures: dict[str, list[list[str]]] = {}
    n_evalplus = 0
    for index, problem in enumerate(results["per_problem"]):
        task_id = problem["task_id"]
        meta = problems.get(task_id)
        samples = problem.get("samples", [])
        if meta is None or not samples:
            continue
        probes = evalplus_probes.get(task_id)
        if probes:
            n_evalplus += 1
        else:
            probes = extract_probe_inputs(meta["test_code"], meta["entry_point"])
        signatures[task_id] = [
            behavior_signature(s["code"], meta["entry_point"], probes, args.timeout_s)
            for s in samples
        ]
        if index % 20 == 19:
            logger.info(
                "  replayed %d/%d problems", index + 1, len(results["per_problem"])
            )
    logger.info(
        "Probe provenance: %d problems on EvalPlus inputs, %d on test-derived fallback",
        n_evalplus,
        len(signatures) - n_evalplus,
    )
    return signatures


def load_or_build_signatures(
    results: dict, run_file: str, args: argparse.Namespace
) -> dict[str, list[list[str]]]:
    """Cached signatures, or a fresh replay written to the cache."""
    path = _cache_path(args.cache_dir, run_file, args)
    if path.exists() and not args.refresh_cache:
        logger.info("Signature cache hit: %s", path.name)
        return json.loads(path.read_text())
    logger.info("Replaying %s (no cache)", run_file)
    signatures = build_signatures(results, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signatures))
    logger.info("Cached %d problems to %s", len(signatures), path.name)
    return signatures


def collect_rows(results: dict, signatures: dict[str, list[list[str]]]) -> list[dict]:
    """One row per sample: both published definitions, the project's, and the label."""
    rows = []
    for problem in results["per_problem"]:
        task_id = problem["task_id"]
        samples = problem.get("samples", [])
        sigs = signatures.get(task_id)
        if not sigs or len(sigs) != len(samples):
            continue
        project = execution_behavior_self_consistency(sigs, DEFAULT_ERROR_COSTS)
        graded = pm.graded_semantic_self_consistency(sigs)
        isolation = pm.codemue_isolation_self_consistency(sigs)
        support = nocrash_support(sigs)
        for sample, proj, grad, iso, sup, sig in zip(
            samples, project, graded, isolation, support, sigs
        ):
            rows.append(
                {
                    "task_id": task_id,
                    "passed": int(bool(sample["passed"])),
                    "crash_rate": execution_crash_rate(sig),
                    "project_graded_self": proj,
                    "published_graded_self": grad,
                    "published_isolation_self": iso,
                    "nocrash_support": sup,
                }
            )
    return rows


def _auc(rows: list[dict], feature: str) -> float | None:
    """ROC-AUC of "sample passes" against a distance feature (higher = worse)."""
    labels = [row["passed"] for row in rows]
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, [-row[feature] for row in rows]))


FEATURES = (
    "crash_rate",
    "project_graded_self",
    "published_graded_self",
    "published_isolation_self",
)


def summarize(rows: list[dict]) -> dict:
    """The decomposition: each feature over all samples and over support>0."""
    supported = [row for row in rows if row["nocrash_support"] > 0.0]
    return {
        "n_samples": len(rows),
        "n_supported": len(supported),
        "frac_supported": len(supported) / len(rows) if rows else 0.0,
        "mean_crash_rate": float(np.mean([r["crash_rate"] for r in rows])),
        "frac_samples_all_crash": float(
            np.mean([1.0 if r["crash_rate"] == 1.0 else 0.0 for r in rows])
        ),
        "pass_rate_all": float(np.mean([r["passed"] for r in rows])) if rows else 0.0,
        "auc_all_samples": {name: _auc(rows, name) for name in FEATURES},
        "auc_supported_only": {name: _auc(supported, name) for name in FEATURES},
    }


def score_h6(per_cell: dict) -> dict:
    """H6: both published definitions retain ROC >= 0.70 on support>0 in >=3/4 cells."""
    verdicts = {}
    for definition in ("published_graded_self", "published_isolation_self"):
        clearing = [
            cell
            for cell, summary in per_cell.items()
            if (summary["auc_supported_only"].get(definition) or 0.0) >= H6_ROC_FLOOR
        ]
        verdicts[definition] = {
            "cells_clearing_floor": clearing,
            "n_clearing": len(clearing),
            "n_cells": len(per_cell),
            "passes": len(clearing) >= H6_MIN_CELLS,
        }
    return {
        "hypothesis": (
            f"each published execution definition retains ROC-AUC >= {H6_ROC_FLOOR} "
            f"on the support>0 subpopulation in >= {H6_MIN_CELLS} of 4 cells"
        ),
        "per_definition": verdicts,
        "verdict": (
            "CONFIRMED" if all(v["passes"] for v in verdicts.values()) else "FALSIFIED"
        ),
    }


def main() -> None:
    """Replay, decompose, score H6."""
    args = parse_args()
    set_all_seeds(args.seed)
    per_cell = {}
    for cell in args.cells:
        path = args.results_dir / CELLS[cell]
        if not path.exists():
            logger.error("Cell %s: missing %s", cell, path)
            continue
        results = json.loads(path.read_text())
        signatures = load_or_build_signatures(results, CELLS[cell], args)
        rows = collect_rows(results, signatures)
        summary = summarize(rows)
        summary["run_file"] = CELLS[cell]
        summary["model"] = results["config"].get("model")
        per_cell[cell] = summary
        logger.info(
            "Cell %s: %d samples, %.1f%% with co-running evidence",
            cell,
            summary["n_samples"],
            100 * summary["frac_supported"],
        )

    analysis = {
        "preregistration": "reports/week11_field_audit_preregistration.md",
        "probe_source": args.probe_source,
        "probe_cap": args.probe_cap,
        "timeout_s": args.timeout_s,
        "seed": args.seed,
        "per_cell": per_cell,
        "H6": score_h6(per_cell),
    }

    print(
        "\n=== C3: the runnability / agreement decomposition, published definitions ==="
    )
    for cell, summary in per_cell.items():
        print(
            f"\n{cell}  n={summary['n_samples']}  "
            f"crash cells={summary['mean_crash_rate']:.3f}  "
            f"all-crash samples={summary['frac_samples_all_crash']:.1%}  "
            f"support>0={summary['frac_supported']:.1%}"
        )
        print(f"    {'feature':<28}{'ROC all':<12}{'ROC support>0':<14}")
        for name in FEATURES:
            a = summary["auc_all_samples"][name]
            s = summary["auc_supported_only"][name]
            print(
                f"    {name:<28}"
                f"{('n/a' if a is None else f'{a:.3f}'):<12}"
                f"{('n/a' if s is None else f'{s:.3f}'):<14}"
            )
    print(f"\nH6: {analysis['H6']['verdict']}")
    print(f"    {analysis['H6']['hypothesis']}")

    path = save_results(analysis, prefix="analysis_week11_field_audit_execution")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
