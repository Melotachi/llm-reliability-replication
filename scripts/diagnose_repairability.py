"""Week 12. Predict repairability, and test whether sibling behaviour carries it.

The paper's own constructive consequence (Section 7.1) was never run: "the
allocation model should be trained on repair outcomes, not on initial failure
... this is a training-target change, not a new experiment." This script is that
change. Target is `final_passed` over the initially-failing coordinates, not
`initial_passed`.

The candidate feature is **sibling behavioural proximity**. Section 6.5 of the
paper controls for problem difficulty with the sibling *pass rate* and notes that
this is difficulty "at a granularity of 1/9, and a finer one could still absorb
more of the effect." Proximity is that finer measure: how close the failing
program's probe *outputs* sit to the outputs of its N-1 siblings. A failure whose
behaviour nearly matches the crowd is one edit from correct and cheap to repair;
one that disagrees everywhere is semantically lost. It reads probe outputs only,
never pass/fail verdicts, so it stays verdict-free like the rest of Objective 1.

**Two bars, fixed before the feature was computed** (Week 12 report Section 8),
because whichever one it clears first would otherwise become the reported bar:

* On the `none` (resample) arm, proximity must beat the **sibling pass rate** --
  0.691 on MBPP, 0.554 on HumanEval -- because it is pitched as a finer version
  of exactly that quantity.
* On the `code_only` and `error` arms it must beat **P(pass)** -- 0.648-0.706 on
  HumanEval, 0.675 on MBPP `code_only`.
* Everywhere it must also beat **error category** and **program length**, both
  already reported as repair-success predictors (arXiv 2604.10508).

**The support hazard, handled rather than imputed.** Week 7's crash-rate control
found that an agreement feature which scores never-running samples as maximally
confident degrades in step with the all-crash fraction. Proximity has the same
shape: it is undefined when the failing program crashes on every probe, and
undefined again when it runs but no sibling runs anywhere it does. Both strata get
an explicit `has_proximity` indicator and their own reported counts; neither gets
an imputed proximity value that would make the crashiest programs look like the
most confident ones.

Usage:
    python scripts/diagnose_repairability.py --cohort both
    python scripts/diagnose_repairability.py --cohort mbpp --timeout 5.0
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_smart_repair import (  # noqa: E402
    _coordinate,
    align_oof_scores,
    load_json,
    select_cohort,
    validate_repair_pool,
)
from scripts.diagnose_arm_allocation import COHORTS  # noqa: E402
from scripts.diagnose_gate_direction import sibling_pass_rates  # noqa: E402
from scripts.diagnose_gate_target import roc_auc  # noqa: E402
from src.evaluation import behavior_signature, extract_probe_inputs  # noqa: E402
from src.uncertainty import evaluate_predictor_cv  # noqa: E402
from src.utils import (  # noqa: E402
    get_logger,
    load_dataset_split,
    load_evalplus_probe_inputs,
    save_results,
    set_all_seeds,
)

LOGGER = get_logger(__name__)

DEFAULT_FIGURE_DIR = Path("reports/figures")
DEFAULT_PROBE_CAP = 12

# Feature arms. The first four are the bars proximity has to clear; the last
# three are the candidate and its combinations.
FEATURE_ARMS: dict[str, list[str]] = {
    "category_only": [
        "category_assertion_error",
        "category_name_error",
        "category_type_error",
        "category_runtime_error",
        "category_syntax_error",
    ],
    # `chars_only` is the honest length arm. `initial_ast_nodes` is -1 on a
    # SyntaxError, and that value coincides *exactly* with the `syntax_error`
    # failure category (3/3 on MBPP, 5/5 on HumanEval) at a 0% repair rate -- so
    # the two-column arm smuggles in a perfectly predictive category indicator.
    # All three arms are reported; only `chars_only` may be quoted as "length".
    "chars_only": ["initial_chars"],
    "ast_nodes_only": ["initial_ast_nodes"],
    "length_only": ["initial_chars", "initial_ast_nodes"],
    "p_pass_only": ["p_pass_oof"],
    "sibling_only": ["sibling_pass_rate"],
    "category_plus_length": [
        "category_assertion_error",
        "category_name_error",
        "category_type_error",
        "category_runtime_error",
        "category_syntax_error",
        "initial_chars",
        "initial_ast_nodes",
    ],
    "proximity_only": ["proximity", "has_proximity"],
    "proximity_plus_crash": [
        "proximity",
        "has_proximity",
        "proximity_support",
        "crash_rate",
    ],
    "proximity_plus_baselines": [
        "proximity",
        "has_proximity",
        "proximity_support",
        "crash_rate",
        "initial_chars",
        "initial_ast_nodes",
        "category_assertion_error",
        "category_name_error",
        "category_type_error",
        "category_runtime_error",
        "category_syntax_error",
    ],
    "everything": [
        "proximity",
        "has_proximity",
        "proximity_support",
        "crash_rate",
        "initial_chars",
        "initial_ast_nodes",
        "category_assertion_error",
        "category_name_error",
        "category_type_error",
        "category_runtime_error",
        "category_syntax_error",
        "p_pass_oof",
        "sibling_pass_rate",
    ],
}

CATEGORIES = (
    "assertion_error",
    "name_error",
    "type_error",
    "runtime_error",
    "syntax_error",
)


def _is_error(outcome: str) -> bool:
    """behavior_signature marks an abnormal probe outcome with an `error:` prefix."""
    return outcome.startswith("error:")


def pair_proximity(
    candidate: list[str], sibling: list[str]
) -> tuple[float | None, int]:
    """Behavioural agreement between two signatures, over co-running probes only.

    Restricting to probes where *both* programs ran is what keeps the feature from
    collapsing into "does it run at all". A pair with no co-running probe is
    undefined and says so, rather than scoring 0 or 1 -- both of which would
    smuggle runnability back in (Week 7's crash-rate control).

    Args:
        candidate: Probe outcomes of the failing program.
        sibling: Probe outcomes of one sibling sample.
    Returns:
        (agreement in [0, 1], number of co-running probes). Agreement is None
        when the two never ran on a common probe.
    """
    agree = 0
    shared = 0
    for left, right in zip(candidate, sibling):
        if _is_error(left) or _is_error(right):
            continue
        shared += 1
        if left == right:
            agree += 1
    if shared == 0:
        return None, 0
    return agree / shared, shared


def proximity_features(
    candidate: list[str], siblings: list[list[str]]
) -> dict[str, float]:
    """Sibling behavioural proximity for one failing candidate.

    Args:
        candidate: Probe outcomes of the failing program (sample 0).
        siblings: Probe outcomes of the other N-1 samples of the same problem.
    Returns:
        proximity (mean agreement over siblings that share a running probe),
        proximity_support (how many siblings that was), has_proximity (the
        indicator that keeps undefined rows in the design matrix without an
        imputed value), crash_rate, and all_crash.
    """
    if not candidate:
        return {
            "proximity": 0.0,
            "proximity_support": 0,
            "has_proximity": 0,
            "crash_rate": 1.0,
            "all_crash": 1,
        }
    crash_rate = sum(1 for outcome in candidate if _is_error(outcome)) / len(candidate)
    agreements = []
    support = 0
    for sibling in siblings:
        agreement, shared = pair_proximity(candidate, sibling)
        if agreement is None:
            continue
        agreements.append(agreement)
        support += 1
    return {
        # 0.0 is a placeholder paired with has_proximity=0, never a measurement:
        # every arm that uses `proximity` also carries `has_proximity`.
        "proximity": float(np.mean(agreements)) if agreements else 0.0,
        "proximity_support": support,
        "has_proximity": int(bool(agreements)),
        "crash_rate": crash_rate,
        "all_crash": int(crash_rate == 1.0),
    }


def signature_cache_path(
    source: dict, *, cache_dir: Path, timeout_s: float, probe_cap: int, seed: int
) -> Path:
    """Where this run's behaviour signatures live.

    Every input that changes a signature is in the filename, so a stale cache is
    not reachable by accident: the run id, the probe cap and seed that choose the
    probes, and the replay timeout that decides what counts as `error:timeout`.
    """
    run_id = source.get("run_id", "unknown_run")
    return cache_dir / (
        f"behavior_signatures_{run_id}_cap{probe_cap}"
        f"_t{timeout_s:g}_seed{seed}.json"
    )


def compute_signatures(
    source: dict,
    *,
    timeout_s: float,
    probe_cap: int,
    seed: int,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """Replay every stored sample on its problem's probes.

    # TODO: SANDBOX -- replays stored generated code (subprocess-isolated, same
    # caveat as evaluation.run_code). Research/local use only.

    EvalPlus probes are preferred and test-derived ones are the fallback, matching
    train_predictor.py; the split is reported so a cell with poor probe coverage
    announces itself instead of being read as a weak signal.

    train_predictor.py recomputes these on every run and stores only the derived
    features, so a re-analysis pays the full replay again -- observed between 2 and
    14 minutes for 600 samples depending on machine state, since crashed and
    timed-out replays dominate the wall clock. They are a pure function of stored
    inputs, so they are cached here, keyed by everything that can change them.
    (scripts/audit_published_execution.py keeps a separate cache under
    results/.signature_cache/ with a different key scheme; the two are not shared.)

    Args:
        source: The uncertainty run holding all N samples per problem.
        timeout_s: Per-sample replay limit.
        probe_cap: Maximum probes per problem.
        seed: Seed for the deterministic probe cap.
        cache_path: Where to read and write the cache; None disables it.
        refresh: Recompute and overwrite even when a cache exists.
    Returns:
        task_id -> {"signatures": list per sample, "n_probes", "probe_source"}.
    """
    if cache_path is not None and cache_path.exists() and not refresh:
        LOGGER.info("reusing cached signatures from %s", cache_path)
        return json.loads(cache_path.read_text())

    dataset_name = source.get("config", {}).get("dataset", "humaneval")
    problems = {p["task_id"]: p for p in load_dataset_split(dataset_name)}
    evalplus = load_evalplus_probe_inputs(dataset_name, cap=probe_cap, seed=seed)

    cache: dict[str, dict[str, Any]] = {}
    n_evalplus = 0
    for problem in source["per_problem"]:
        task_id = problem["task_id"]
        meta = problems.get(task_id)
        if meta is None:
            LOGGER.warning("no dataset entry for %s; skipping", task_id)
            continue
        probes = evalplus.get(task_id) or []
        probe_source = "evalplus"
        if probes:
            n_evalplus += 1
        else:
            probes = extract_probe_inputs(meta["test_code"], meta["entry_point"])
            probe_source = "test"
        signatures = [
            behavior_signature(sample["code"], meta["entry_point"], probes, timeout_s)
            for sample in problem["samples"]
        ]
        # pair_proximity zips a candidate against a sibling, and zip truncates
        # silently -- a ragged signature list would quietly redefine "co-running"
        # to mean "co-running within the shorter prefix". behavior_signature
        # always returns one outcome per probe (verified: 0 ragged problems on
        # the EvalPlus path), but the fixture path in train_predictor.py needs
        # pad_signatures, so the invariant is asserted rather than assumed.
        lengths = {len(signature) for signature in signatures}
        if len(lengths) > 1:
            raise ValueError(
                f"ragged behaviour signatures for {task_id}: lengths {sorted(lengths)}"
            )
        cache[task_id] = {
            "signatures": signatures,
            "n_probes": len(probes),
            "probe_source": probe_source,
        }
    LOGGER.info(
        "signatures for %d problems (%d EvalPlus, %d test-derived)",
        len(cache),
        n_evalplus,
        len(cache) - n_evalplus,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
        LOGGER.info("cached signatures to %s", cache_path)
    return cache


def _ast_nodes(code: str) -> int:
    """Node count of `code`, or -1 when it does not parse.

    -1 rather than 0 keeps unparseable programs separable from empty ones; the
    length arm carries both this and the character count.
    """
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except SyntaxError:
        return -1


def build_rows(
    source: dict,
    cohort: list[dict],
    repair_by_coordinate: dict[tuple[str, int], dict],
    signatures: dict[str, dict[str, Any]],
    sibling_rates: dict[str, float],
    *,
    sample_index: int,
) -> list[dict[str, Any]]:
    """One feature row per initially-failing coordinate, target `final_passed`.

    Args:
        source: The uncertainty run (unused beyond documenting provenance).
        cohort: Scored cohort rows.
        repair_by_coordinate: Frozen repair records keyed by coordinate.
        signatures: Output of compute_signatures().
        sibling_rates: Output of sibling_pass_rates().
        sample_index: Which candidate index the cohort was built from.
    Returns:
        Row dicts carrying every feature named in FEATURE_ARMS plus the target.
    """
    rows = []
    for entry in cohort:
        if entry["initial_passed"]:
            continue
        task_id = entry["task_id"]
        record = repair_by_coordinate[_coordinate(entry)]
        stored = signatures.get(task_id, {})
        all_signatures = stored.get("signatures", [])
        candidate = (
            all_signatures[sample_index] if sample_index < len(all_signatures) else []
        )
        siblings = [
            signature
            for index, signature in enumerate(all_signatures)
            if index != sample_index
        ]
        initial_code = record.get("initial_code", "")
        category = record.get("initial_failure_category", "")
        row: dict[str, Any] = {
            "task_id": task_id,
            "sample_index": entry["sample_index"],
            "final_passed": int(bool(record["final_passed"])),
            "initial_failure_category": category,
            "initial_chars": float(len(initial_code)),
            "initial_ast_nodes": float(_ast_nodes(initial_code)),
            "p_pass_oof": float(entry["p_pass_oof"]),
            "sibling_pass_rate": float(sibling_rates.get(task_id, 0.0)),
            "n_probes": stored.get("n_probes", 0),
            "probe_source": stored.get("probe_source", "none"),
        }
        for name in CATEGORIES:
            row[f"category_{name}"] = float(category == name)
        row.update(proximity_features(candidate, siblings))
        rows.append(row)
    return rows


def score_arms(
    rows: list[dict[str, Any]], *, model_type: str, n_splits: int, seed: int
) -> dict[str, Any]:
    """Cross-validated ROC-AUC per feature arm, plus every single-column AUC.

    Grouping by problem is vacuous on this cohort -- the sample-0 design gives one
    row per problem -- so StratifiedGroupKFold degenerates to StratifiedKFold. It
    is kept anyway so the same code stays correct on a multi-candidate cohort.

    Args:
        rows: Output of build_rows().
        model_type: "logistic" or "random_forest".
        n_splits: Cross-validation folds.
        seed: Seed for the fold split and the model.
    Returns:
        Per-arm cross-validated metrics and a univariate AUC per feature column.
    """
    labels = np.array([row["final_passed"] for row in rows], dtype=int)
    groups = np.array([row["task_id"] for row in rows])

    arms = {}
    for arm, feature_names in FEATURE_ARMS.items():
        matrix = np.array(
            [[float(row[name]) for name in feature_names] for row in rows],
            dtype=float,
        )
        evaluation = evaluate_predictor_cv(
            matrix,
            labels,
            groups,
            model_type=model_type,
            n_splits=n_splits,
            seed=seed,
            feature_names=feature_names,
        )
        summary = evaluation["summary"]
        arms[arm] = {
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "roc_auc_mean": summary["roc_auc_mean"],
            "roc_auc_std": summary["roc_auc_std"],
            "pr_auc_mean": summary["pr_auc_mean"],
            "f1_mean": summary["f1_mean"],
            "oof_roc_auc": roc_auc(
                np.asarray(evaluation["oof_proba"], dtype=float),
                np.asarray(evaluation["oof_y"], dtype=int),
            ),
            "feature_importances": evaluation["feature_importances"],
        }

    univariate = {}
    columns = sorted({name for names in FEATURE_ARMS.values() for name in names})
    for name in columns:
        values = np.array([float(row[name]) for row in rows], dtype=float)
        univariate[name] = roc_auc(values, labels)
    return {"arms": arms, "univariate_roc_auc": univariate}


def strata_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the two strata where proximity is undefined, and their base rates.

    Reported rather than imputed: a stratum whose repair rate differs from the
    rest is exactly what an imputed value would silently encode as a measurement.
    """

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(subset),
            "repaired": sum(row["final_passed"] for row in subset),
            "repair_rate": (
                sum(row["final_passed"] for row in subset) / len(subset)
                if subset
                else None
            ),
        }

    all_crash = [row for row in rows if row["all_crash"]]
    no_support = [
        row for row in rows if not row["all_crash"] and not row["has_proximity"]
    ]
    defined = [row for row in rows if row["has_proximity"]]
    return {
        "all_crash": block(all_crash),
        "runs_but_no_co_running_sibling": block(no_support),
        "proximity_defined": block(defined),
        "mean_support_where_defined": (
            float(np.mean([row["proximity_support"] for row in defined]))
            if defined
            else None
        ),
        "mean_proximity_where_defined": (
            float(np.mean([row["proximity"] for row in defined])) if defined else None
        ),
    }


def build_cohort_analysis(
    cohort_name: str,
    spec: dict[str, Any],
    *,
    sample_index: int = 0,
    timeout_s: float = 5.0,
    probe_cap: int = DEFAULT_PROBE_CAP,
    model_type: str = "logistic",
    n_splits: int = 5,
    seed: int = 42,
    feature_set: str = "self_consistency_plus_execution",
    predictor_model_type: str = "logistic",
    cache_dir: Path | None = None,
    refresh_signatures: bool = False,
) -> dict[str, Any]:
    """Score every repair arm of one cohort against the repairability target."""
    source = load_json(spec["source"])
    predictor = load_json(spec["predictor"])
    scored = align_oof_scores(source, predictor, feature_set, predictor_model_type)
    cohort = select_cohort(scored, sample_index=sample_index)
    sibling_rates = sibling_pass_rates(source, sample_index)

    LOGGER.info("replaying stored samples for %s", cohort_name)
    cache_path = (
        None
        if cache_dir is None
        else signature_cache_path(
            source,
            cache_dir=cache_dir,
            timeout_s=timeout_s,
            probe_cap=probe_cap,
            seed=seed,
        )
    )
    signatures = compute_signatures(
        source,
        timeout_s=timeout_s,
        probe_cap=probe_cap,
        seed=seed,
        cache_path=cache_path,
        refresh=refresh_signatures,
    )

    arms: dict[str, Any] = {}
    for arm_name, pool_path in spec["arms"]:
        pool = load_json(pool_path)
        repair_by_coordinate = validate_repair_pool(
            spec["source"], cohort, pool, sample_index
        )
        rows = build_rows(
            source,
            cohort,
            repair_by_coordinate,
            signatures,
            sibling_rates,
            sample_index=sample_index,
        )
        scores = score_arms(rows, model_type=model_type, n_splits=n_splits, seed=seed)
        arms[arm_name] = {
            "pool_file": str(pool_path),
            "n_rows": len(rows),
            "n_repaired": sum(row["final_passed"] for row in rows),
            "strata": strata_report(rows),
            **scores,
        }

    return {
        "analysis": "week12_repairability",
        "cohort_name": cohort_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "source_results": str(spec["source"]),
            "source_run_id": source.get("run_id"),
            "predictor_results": str(spec["predictor"]),
            "sample_index": sample_index,
            "timeout_s": timeout_s,
            "probe_cap": probe_cap,
            "model_type": model_type,
            "n_splits": n_splits,
            "seed": seed,
            "target": "final_passed over initially-failing coordinates",
            "grouping_note": (
                "one row per problem in the sample-0 design, so grouped CV "
                "degenerates to StratifiedKFold here"
            ),
        },
        "arms": arms,
    }


def plot_repairability(analyses: list[dict[str, Any]], output: Path) -> None:
    """Per-arm cross-validated ROC, one panel per cohort, bars grouped by arm."""
    figure, axes = plt.subplots(
        1, len(analyses), figsize=(6.4 * len(analyses), 4.8), squeeze=False
    )
    order = list(FEATURE_ARMS)
    for index, analysis in enumerate(analyses):
        axis = axes[0][index]
        width = 0.26
        for offset, arm in enumerate(["none", "code_only", "error"]):
            block = analysis["arms"].get(arm)
            if not block:
                continue
            values = [block["arms"][name]["roc_auc_mean"] for name in order]
            axis.bar(
                [position + (offset - 1) * width for position in range(len(order))],
                values,
                width=width,
                label=f"repair `{arm}`",
            )
        axis.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels(order, rotation=35, ha="right", fontsize=7.5)
        axis.set_ylabel("cross-validated ROC-AUC")
        axis.set_title(f"{analysis['cohort_name']}: predicting repair success")
        # Below the axes: the bars run the full height and any in-plot legend
        # covers the column the reader is here to look at.
        axis.legend(
            fontsize=8,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            frameon=False,
        )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _print_summary(analysis: dict[str, Any]) -> None:
    """Print the arm table plus the two undefined strata."""
    print(f"\n=== {analysis['cohort_name']}")
    for arm in ["none", "code_only", "error"]:
        block = analysis["arms"].get(arm)
        if not block:
            continue
        strata = block["strata"]
        print(
            f"\n  repair arm `{arm}`  n={block['n_rows']} repaired={block['n_repaired']}"
            f"  | all-crash {strata['all_crash']['n']}"
            f", no-co-running-sibling {strata['runs_but_no_co_running_sibling']['n']}"
            f", defined {strata['proximity_defined']['n']}"
        )
        print(f"    {'feature arm':26s} {'CV ROC':>8s} {'+-':>6s} {'PR-AUC':>7s}")
        for name in FEATURE_ARMS:
            row = block["arms"][name]
            print(
                f"    {name:26s} {row['roc_auc_mean']:8.3f} "
                f"{row['roc_auc_std']:6.3f} {row['pr_auc_mean']:7.3f}"
            )


def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train and score a repairability predictor on stored repair pools"
    )
    parser.add_argument(
        "--cohort", choices=["mbpp", "humaneval", "both"], default="both"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--probe-cap", type=int, default=DEFAULT_PROBE_CAP)
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/signature_cache"),
        help="Where behaviour signatures are cached; pass 'none' to disable.",
    )
    parser.add_argument("--refresh-signatures", action="store_true")
    parser.add_argument("--prefix", default="analysis_week12_repairability")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    set_all_seeds(args.seed)

    names = ["mbpp", "humaneval"] if args.cohort == "both" else [args.cohort]
    analyses = []
    for name in names:
        analysis = build_cohort_analysis(
            name,
            COHORTS[name],
            sample_index=args.sample_index,
            timeout_s=args.timeout,
            probe_cap=args.probe_cap,
            model_type=args.model_type,
            n_splits=args.n_splits,
            seed=args.seed,
            cache_dir=(
                None if str(args.cache_dir).lower() == "none" else args.cache_dir
            ),
            refresh_signatures=args.refresh_signatures,
        )
        analyses.append(analysis)
        _print_summary(analysis)

    figure = args.figure_dir / "week12_repairability.png"
    plot_repairability(analyses, figure)
    path = save_results(
        {
            "analysis": "week12_repairability",
            "timestamp": datetime.now().isoformat(),
            "figure": str(figure),
            "cohorts": analyses,
        },
        prefix=args.prefix,
    )
    print(f"\nfigure: {figure}")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
