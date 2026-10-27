"""Week 4 predictor training + evaluation: uncertainty features -> pass/fail.

Student A deliverable (CLAUDE.md Section 10.1 Week 4: "Build uncertainty
predictor (ML model for pass/fail prediction)").

Loads a pipeline results JSON (CLAUDE.md Section 8 schema) and trains/
evaluates UncertaintyPredictor (src/uncertainty.py) on two genuinely
different prediction tasks built from the same run, both scored with
leave-problems-out (StratifiedGroupKFold) cross-validation so no problem's
rows ever split across train and test:

  Task A - problem level: "will at least one of this problem's N samples
    pass?", from the problem-level diversity metrics (compute_uncertainty).
    One row per problem.
  Task B - sample level: "will THIS specific sample pass?", from its
    per-sample leave-one-out self-consistency features
    (compute_self_consistency). One row per sample. Problem-level features
    are identical across a problem's N samples and cannot tell which of
    them is the wrong one (see src/uncertainty.py's Week 4 module
    docstring for why) -- Task B exists specifically to use features that
    vary within a problem.

For each task, several feature-set / model-type configurations are
evaluated (a "full" feature set, a reduced set informed by the Week 3
addendum's redundancy heatmap, logistic regression + random forest), and:

  1. reports/figures/week4_predictor_comparison.png -- F1 (positive class)
     by configuration, colored by task, the "which setup actually works"
     view.
  2. A timestamped stats JSON in results/ with every fold's metrics, the
     mean+-std summary, and feature importances per configuration
     (CLAUDE.md Constraint #5: results are always persisted).

Usage:
    python scripts/train_predictor.py                    # latest run in results/
    python scripts/train_predictor.py --results results/humaneval_uncertainty_....json
    python scripts/train_predictor.py --no-codebert       # skip the ~500 MB encoder
    python scripts/train_predictor.py --n-splits 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes PNGs

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_uncertainty import find_latest_results_file, load_run_prompts
from src.evaluation import (
    behavior_signature,
    extract_probe_inputs,
    fixture_behavior_signature,
    pad_signatures,
)
from src.uncertainty import (
    DEFAULT_ERROR_COSTS,
    LEGACY_ERROR_COSTS,
    METRIC_REPRESENTATIONS,
    compute_self_consistency,
    compute_self_consistency_prompt_aware,
    compute_uncertainty,
    compute_uncertainty_prompt_aware,
    evaluate_predictor_cv,
    execution_behavior_diversity,
    execution_behavior_diversity_nocrash,
    execution_behavior_self_consistency,
    execution_behavior_self_consistency_nocrash,
    execution_crash_rate,
    prompt_confound_covariates,
)
from src.utils import (
    get_logger,
    load_dataset_split,
    load_evalplus_probe_inputs,
    save_results,
    set_all_seeds,
)

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "reports" / "figures"

# Same reference palette values as scripts/analyze_uncertainty.py, duplicated
# rather than imported: leading-underscore module constants are file-private
# by convention, and each script should stay independently runnable.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_TASK_A = "#2a78d6"  # categorical slot 1 (blue) - Task A, problem level
_TASK_B = "#1baf7a"  # categorical slot 2 (aqua) - Task B, sample level

# Problem-level feature sets (one row per problem; label = "solved" below).
PROBLEM_LEVEL_FEATURE_SETS = {
    "full": [
        "lexical_diversity",
        "token_jaccard_diversity",
        "token_cosine_diversity",
        "ast_node_diversity",
        "tree_edit_distance_diversity",
        "codebert_cosine_diversity",
        "codebert_bertscore_diversity",
        "pairwise_distance",
    ],
    "reduced": [
        # Week 3 addendum TODO #4: the redundancy heatmap put the real
        # feature count closer to four than nine -- one metric per signal
        # family (lexical, structural, semantic, legacy baseline).
        "token_jaccard_diversity",
        "ast_node_diversity",
        "codebert_cosine_diversity",
        "pairwise_distance",
    ],
    "no_codebert": [
        # Same size as "reduced" (one metric per remaining family) but with
        # token_cosine_diversity standing in for the semantic slot instead
        # of dropping it -- the only arm that still runs under --no-codebert,
        # and a direct answer to "does the semantic channel actually earn
        # its ~500 MB encoder over lexical+structural alone?"
        "token_jaccard_diversity",
        "token_cosine_diversity",
        "ast_node_diversity",
        "pairwise_distance",
    ],
    "full_plus_parse_failure": [
        # Bonus ablation: the Week 2 report explicitly recommended keeping
        # parse_failure_rate OUT of the predictor's primary features ("too
        # rare and too binary... don't feed it to the Week 4 predictor as
        # a primary feature"). This arm checks that call empirically
        # instead of just trusting it.
        "lexical_diversity",
        "token_jaccard_diversity",
        "token_cosine_diversity",
        "ast_node_diversity",
        "tree_edit_distance_diversity",
        "codebert_cosine_diversity",
        "codebert_bertscore_diversity",
        "pairwise_distance",
        "parse_failure_rate",
    ],
    # Week 6 ablation arms: the execution-behavior (functional) axis.
    # execution_only asks "how far does behavior alone get?";
    # reduced_plus_execution asks "does behavior add signal on top of the
    # Week 4 best config?" (see src/uncertainty.py's Week 6 section for the
    # four grounding papers and the verdict-free design constraint).
    "execution_only": [
        "execution_behavior_diversity",
    ],
    "reduced_plus_execution": [
        "token_jaccard_diversity",
        "ast_node_diversity",
        "codebert_cosine_diversity",
        "pairwise_distance",
        "execution_behavior_diversity",
    ],
    # The zero-model-cost all-signal configuration: text + structural +
    # execution, no encoder anywhere ("no embeddings, no LLM-as-judge" —
    # He et al.'s efficiency framing applies to every feature here). The
    # deployment-friendly counterpart of reduced_plus_execution.
    "no_codebert_plus_execution": [
        "token_jaccard_diversity",
        "token_cosine_diversity",
        "ast_node_diversity",
        "pairwise_distance",
        "execution_behavior_diversity",
    ],
    # Week 7 baseline arm: the free scalar. Mean generated-code length needs
    # no sampling, no parsing, no embedding and no execution, yet it
    # recovers 71-96% of ast_node_diversity's correlation magnitude with
    # pass rate in every one of this project's six regime cells. Every
    # arm above has to beat it to have earned its cost, and a reviewer
    # will ask; this makes the comparison a measured result rather than an
    # argument. Available only when the run's prompts were recoverable.
    "length_only": [
        "mean_completion_chars",
    ],
    # Week 7 crash-rate control (problem level). If crash_rate_only rivals
    # execution_only, the Week 6 headline is really about runnability, not
    # behavioral consensus; if execution_nocrash keeps the ROC-AUC, it is
    # about consensus after all and the claim is safe.
    "crash_rate_only": [
        "execution_crash_rate",
    ],
    "execution_nocrash": [
        "execution_behavior_diversity_nocrash",
    ],
}

# Sample-level feature sets (one row per sample; label = "passed").
SAMPLE_LEVEL_FEATURE_SETS = {
    "self_consistency_core": [
        "token_jaccard_self_distance",
        "token_cosine_self_distance",
        "ast_node_self_distance",
    ],
    "self_consistency_with_codebert": [
        "token_jaccard_self_distance",
        "token_cosine_self_distance",
        "ast_node_self_distance",
        "codebert_cosine_self_distance",
    ],
    # Week 6 ablation arms (execution-behavior axis, sample level).
    "execution_self_only": [
        "execution_behavior_self_distance",
    ],
    "self_consistency_plus_execution": [
        "token_jaccard_self_distance",
        "token_cosine_self_distance",
        "ast_node_self_distance",
        "execution_behavior_self_distance",
    ],
    # Week 7 crash-rate control (sample level) — the level where the Week 6
    # execution headline (ROC-AUC 0.89-0.93) actually lives.
    "crash_rate_self_only": [
        "execution_crash_self_rate",
    ],
    "execution_self_nocrash": [
        "execution_behavior_self_distance_nocrash",
    ],
}

MODEL_TYPES = ["logistic", "random_forest"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the predictor training/evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Train and cross-validate the Week 4 uncertainty predictor"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Pipeline results JSON to train on (default: newest run in results/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for the output PNG figure (default: reports/figures/).",
    )
    parser.add_argument(
        "--codebert-model",
        type=str,
        default="microsoft/codebert-base",
        help="HuggingFace encoder for the CodeBERT-based features.",
    )
    parser.add_argument(
        "--no-codebert",
        action="store_true",
        help="Skip CodeBERT-based features (avoids the ~500 MB encoder "
        "download; configurations that need them are skipped).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Requested CV fold count (capped at the number of groups).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--no-tree-edit",
        action="store_true",
        help="Skip tree_edit_distance_diversity. Zhang-Shasha dominates "
        "runtime on library-heavy corpora (minutes per problem on "
        "BigCodeBench, whose runs are generated with --no-tree-edit anyway); "
        "the two feature sets that reference it also require CodeBERT, so "
        "under --no-codebert this drops no arm that would otherwise run.",
    )
    parser.add_argument(
        "--no-execution",
        action="store_true",
        help="Skip the execution-behavior features (avoids re-executing the "
        "stored generated samples; configurations that need them are skipped).",
    )
    parser.add_argument(
        "--execution-timeout",
        type=float,
        default=5.0,
        help="Per-sample wall-clock limit for behavior-signature replay (s).",
    )
    parser.add_argument(
        "--representation",
        choices=METRIC_REPRESENTATIONS,
        default="full",
        help="Which string each text-metric family is computed over (Week 7 "
        "prompt-inclusion confound; see src/uncertainty.py). 'full' is the "
        "Weeks 4-6 behavior and stays the default so every prior predictor "
        "number reproduces; 'corrected' strips the shared prompt prefix per "
        "family. The execution-behavior features are unaffected either way — "
        "they never read the source text.",
    )
    parser.add_argument(
        "--probe-source",
        choices=["test", "evalplus", "fixture"],
        default="test",
        help="Where behavior probes come from: 'test' = argument lists "
        "AST-extracted from the dataset's own test suite (Week 6 original); "
        "'evalplus' = EvalPlus HumanEval+/MBPP+ expanded inputs (Liu et al. "
        "2023) — ~10-80x more inputs, fixes MBPP's ~2.7-probe resolution "
        "problem; falls back to 'test' probes per problem when EvalPlus has "
        "no parseable inputs for it; 'fixture' = run the suite itself and "
        "record what the entry point returns inside its own mocks/setUp "
        "(Week 11 — on BigCodeBench, bare probe inputs crash the benchmark's "
        "own canonical solution 64.9% of the time, which 'fixture' drops to "
        "8.2%).",
    )
    parser.add_argument(
        "--probe-cap",
        type=int,
        default=12,
        help="Max probes per problem for --probe-source evalplus (He et al. "
        "2605.09023: signal is flat from 10 to 1,000 inputs).",
    )
    parser.add_argument(
        "--legacy-error-costs",
        action="store_true",
        help="Use the (1, 1, 0) abnormal-outcome costs (same-error = full "
        "agreement) that the Week 6 overnight ablation used, instead of the "
        "graded He et al. default (1, 0.8, 0.6). Continuity arm.",
    )
    parser.add_argument(
        "--fig-name",
        type=str,
        default="week4_predictor_comparison.png",
        help="Output figure filename inside --out-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the figure but skip persisting the stats JSON to results/.",
    )
    return parser.parse_args()


def add_execution_behavior_features(
    results: dict,
    problem_rows: list[dict],
    sample_rows: list[dict],
    timeout_s: float,
    probe_source: str = "test",
    probe_cap: int = 12,
    error_costs: tuple[float, float, float] | None = None,
    seed: int = 42,
) -> None:
    """Merge execution-behavior features into already-built rows, in place.

    # TODO: SANDBOX — replays stored generated code (subprocess-isolated,
    # same caveat as evaluation.run_code). Research/local use only.

    Joins the dataset split (test_code, entry_point) back onto the stored
    per-problem samples by task_id — the same recompute-from-JSON pattern
    the rest of this script uses, extended with the dataset reload because
    the results JSON does not persist test code. Each problem's samples are
    replayed once on the test suite's extracted probe inputs
    (evaluation.behavior_signature); problem rows gain
    "execution_behavior_diversity", sample rows gain
    "execution_behavior_self_distance". Rows whose task_id is missing from
    the dataset (should not happen) are left untouched and their feature
    sets are skipped by the existing only-what's-present logic.

    Args:
        results: Full pipeline results (CLAUDE.md Section 8 schema).
        problem_rows: Output of build_problem_level_dataset(); mutated.
        sample_rows: Output of build_sample_level_dataset(); mutated.
        timeout_s: Per-sample replay limit, forwarded to behavior_signature().
    """
    dataset_name = results.get("config", {}).get("dataset", "humaneval")
    problems = {p["task_id"]: p for p in load_dataset_split(dataset_name)}
    if error_costs is None:
        error_costs = DEFAULT_ERROR_COSTS
    evalplus_probes: dict[str, list[str]] = {}
    if probe_source == "evalplus":
        evalplus_probes = load_evalplus_probe_inputs(
            dataset_name, cap=probe_cap, seed=seed
        )
        logger.info(
            "EvalPlus probes loaded for %d problems (cap %d)",
            len(evalplus_probes),
            probe_cap,
        )
    sample_rows_by_task: dict[str, list[dict]] = {}
    for row in sample_rows:
        sample_rows_by_task.setdefault(row["task_id"], []).append(row)

    n_evalplus, n_fallback = 0, 0
    for problem in results["per_problem"]:
        task_id = problem["task_id"]
        meta = problems.get(task_id)
        samples = problem.get("samples", [])
        if meta is None or not samples:
            if meta is None:
                logger.warning(
                    "No dataset entry for %s; skipping execution features", task_id
                )
            continue
        if probe_source == "fixture":
            # Week 11: probe *inputs* lifted out of a BigCodeBench suite are
            # invalid without the mocks and setUp that made them valid -- the
            # benchmark's own canonical solution crashes on 64.9% of them
            # (41 of 60 problems on every probe). Running the suite and
            # recording what the entry point returns drops that to 8.2%.
            signatures = pad_signatures(
                [
                    fixture_behavior_signature(
                        s["code"], meta["entry_point"], meta["test_code"], timeout_s
                    )
                    for s in samples
                ]
            )
            n_fallback += 1
        else:
            probes = evalplus_probes.get(task_id, [])
            if probes:
                n_evalplus += 1
            else:
                probes = extract_probe_inputs(meta["test_code"], meta["entry_point"])
                n_fallback += 1
            signatures = [
                behavior_signature(s["code"], meta["entry_point"], probes, timeout_s)
                for s in samples
            ]
        diversity = execution_behavior_diversity(signatures, error_costs)
        self_distances = execution_behavior_self_consistency(signatures, error_costs)
        # Week 7 crash-rate control: the same signatures decomposed into the
        # "does it run at all" half and the "do the runs agree" half (see
        # src/uncertainty.py's crash-rate section).
        diversity_nocrash = execution_behavior_diversity_nocrash(signatures)
        self_nocrash = execution_behavior_self_consistency_nocrash(signatures)
        crash_rates = [execution_crash_rate(s) for s in signatures]
        for row in problem_rows:
            if row["task_id"] == task_id:
                row["execution_behavior_diversity"] = diversity
                row["execution_behavior_diversity_nocrash"] = diversity_nocrash
                row["execution_crash_rate"] = float(np.mean(crash_rates))
        for row, distance, nocrash, crash_rate in zip(
            sample_rows_by_task.get(task_id, []),
            self_distances,
            self_nocrash,
            crash_rates,
        ):
            row["execution_behavior_self_distance"] = distance
            row["execution_behavior_self_distance_nocrash"] = nocrash
            row["execution_crash_self_rate"] = crash_rate
    if probe_source == "evalplus":
        logger.info(
            "Probe provenance: %d problems on EvalPlus inputs, %d on test-derived fallback",
            n_evalplus,
            n_fallback,
        )


def _usable_prompt(problem: dict, prompts: dict[str, str]) -> str | None:
    """The problem's prompt, but only if it actually prefixes its samples.

    A prompt that does not prefix its own stored samples means the run and
    the dataset have diverged; returning None makes the caller fall back to
    the uncorrected computation instead of mis-splitting the strings.
    """
    prompt = prompts.get(problem["task_id"])
    if prompt is None:
        return None
    codes = [s["code"] for s in problem.get("samples", [])]
    if not all(code.startswith(prompt) for code in codes):
        logger.warning(
            "%s: stored samples do not start with the dataset prompt; "
            "computing its features on full strings",
            problem["task_id"],
        )
        return None
    return prompt


def build_problem_level_dataset(
    results: dict,
    include_codebert: bool,
    codebert_model: str,
    representation: str = "full",
    prompts: dict[str, str] | None = None,
    include_tree_edit: bool = True,
) -> list[dict]:
    """One row per problem: task_id, "solved" label, and all diversity features.

    "solved" = 1 iff at least one of the problem's N samples passed --
    Task A's label (CLAUDE.md Section 8's per-problem samples list).

    Args:
        results: Full pipeline results (CLAUDE.md Section 8 schema).
        include_codebert: Forwarded to the metric computation.
        codebert_model: HuggingFace encoder for the CodeBERT features.
        representation: One of src.uncertainty.METRIC_REPRESENTATIONS
            (Week 7 prompt-inclusion confound). "full" reproduces every
            pre-Week-7 predictor number exactly.
        prompts: task_id -> prompt (analyze_uncertainty.load_run_prompts()).
            Required for any representation other than "full" and for the
            Week 7 confound covariates (which carry the length_only arm).
        include_tree_edit: Forwarded to the metric computation. Default True
            (every pre-existing caller keeps its feature columns). Callers
            that never read tree_edit_distance_diversity may pass False:
            Zhang-Shasha is the dominant cost on library-heavy corpora
            (minutes per problem on BigCodeBench) and omitting it changes no
            other column.
    Returns:
        List of per-problem row dicts.
    """
    prompts = prompts or {}
    rows = []
    for problem in results["per_problem"]:
        samples = problem.get("samples", [])
        if not samples:
            logger.warning("Skipping %s: no stored samples", problem.get("task_id"))
            continue
        codes = [s["code"] for s in samples]
        passed = [bool(s["passed"]) for s in samples]
        row = {
            "task_id": problem["task_id"],
            "pass_rate": sum(passed) / len(passed),
            "solved": 1 if any(passed) else 0,
        }
        prompt = _usable_prompt(problem, prompts)
        if prompt is None:
            row.update(
                compute_uncertainty(
                    codes,
                    include_codebert=include_codebert,
                    include_tree_edit=include_tree_edit,
                    codebert_model_name=codebert_model,
                )
            )
        else:
            row.update(
                compute_uncertainty_prompt_aware(
                    codes,
                    prompt,
                    representation=representation,
                    include_codebert=include_codebert,
                    include_tree_edit=include_tree_edit,
                    codebert_model_name=codebert_model,
                )
            )
            row.update(prompt_confound_covariates(codes, prompt))
        rows.append(row)
    return rows


def build_sample_level_dataset(
    results: dict,
    include_codebert: bool,
    codebert_model: str,
    representation: str = "full",
    prompts: dict[str, str] | None = None,
) -> list[dict]:
    """One row per generated sample: task_id, "passed" label, self-consistency features.

    Args:
        results: Full pipeline results (CLAUDE.md Section 8 schema).
        include_codebert: Forwarded to the self-consistency computation.
        codebert_model: HuggingFace encoder for the CodeBERT feature.
        representation: As in build_problem_level_dataset(). The per-sample
            text features inherit the same prompt-inclusion confound as the
            problem-level ones, and Task B's headline numbers are built on
            them.
        prompts: task_id -> prompt (analyze_uncertainty.load_run_prompts()).
    Returns:
        List of per-sample row dicts (task_id repeats across a problem's rows).
    """
    prompts = prompts or {}
    rows = []
    for problem in results["per_problem"]:
        samples = problem.get("samples", [])
        if not samples:
            continue
        codes = [s["code"] for s in samples]
        passed = [bool(s["passed"]) for s in samples]
        prompt = _usable_prompt(problem, prompts)
        if prompt is None:
            self_consistency = compute_self_consistency(
                codes,
                include_codebert=include_codebert,
                codebert_model_name=codebert_model,
            )
        else:
            self_consistency = compute_self_consistency_prompt_aware(
                codes,
                prompt,
                representation=representation,
                include_codebert=include_codebert,
                codebert_model_name=codebert_model,
            )
        for sample_passed, features in zip(passed, self_consistency):
            row = {"task_id": problem["task_id"], "passed": 1 if sample_passed else 0}
            row.update(features)
            rows.append(row)
    return rows


def run_all_configurations(
    problem_rows: list[dict],
    sample_rows: list[dict],
    n_splits: int,
    seed: int,
) -> list[dict]:
    """Evaluate every (task, feature-set, model-type) configuration.

    Feature sets referencing a CodeBERT column are skipped when that column
    is absent from the rows (--no-codebert), the same "only what's present"
    pattern analyze_uncertainty.py uses for METRIC_ORDER.

    Args:
        problem_rows: Output of build_problem_level_dataset().
        sample_rows: Output of build_sample_level_dataset().
        n_splits: Requested CV fold count (forwarded to evaluate_predictor_cv).
        seed: Reproducibility seed.
    Returns:
        List of config result dicts, each evaluate_predictor_cv()'s return
        value plus "task", "feature_set" labels.
    """
    results = []
    problem_keys = set(problem_rows[0].keys()) if problem_rows else set()
    sample_keys = set(sample_rows[0].keys()) if sample_rows else set()

    for feature_set_name, feature_names in PROBLEM_LEVEL_FEATURE_SETS.items():
        available_keys = problem_keys
        if not all(f in problem_keys for f in feature_names):
            logger.info(
                "Skipping problem-level set %r: a feature is unavailable "
                "(--no-codebert, --no-execution, or no recoverable prompts "
                "for the Week 7 length covariates): %s",
                feature_set_name,
                sorted(set(feature_names) - available_keys),
            )
            continue
        X = np.array(
            [[row[f] for f in feature_names] for row in problem_rows], dtype=float
        )
        y = np.array([row["solved"] for row in problem_rows], dtype=int)
        # One row per problem already: no leakage risk, so each row is its
        # own group (StratifiedGroupKFold degenerates to StratifiedKFold).
        groups = np.arange(len(problem_rows))
        for model_type in MODEL_TYPES:
            logger.info(
                "Task A (problem-level) | %s | %s", feature_set_name, model_type
            )
            outcome = evaluate_predictor_cv(
                X,
                y,
                groups,
                model_type=model_type,
                n_splits=n_splits,
                seed=seed,
                feature_names=feature_names,
            )
            outcome["task"] = "problem_level"
            outcome["feature_set"] = feature_set_name
            results.append(outcome)

    for feature_set_name, feature_names in SAMPLE_LEVEL_FEATURE_SETS.items():
        available_keys = sample_keys
        if not all(f in sample_keys for f in feature_names):
            logger.info(
                "Skipping sample-level set %r: a feature is unavailable "
                "(--no-codebert, --no-execution, or no recoverable prompts "
                "for the Week 7 length covariates): %s",
                feature_set_name,
                sorted(set(feature_names) - available_keys),
            )
            continue
        X = np.array(
            [[row[f] for f in feature_names] for row in sample_rows], dtype=float
        )
        y = np.array([row["passed"] for row in sample_rows], dtype=int)
        groups = np.array([row["task_id"] for row in sample_rows])
        for model_type in MODEL_TYPES:
            logger.info("Task B (sample-level) | %s | %s", feature_set_name, model_type)
            outcome = evaluate_predictor_cv(
                X,
                y,
                groups,
                model_type=model_type,
                n_splits=n_splits,
                seed=seed,
                feature_names=feature_names,
            )
            outcome["task"] = "sample_level"
            outcome["feature_set"] = feature_set_name
            results.append(outcome)

    return results


def _style_axes(ax: plt.Axes) -> None:
    """Shared recessive-chrome style, matching analyze_uncertainty.py's figures."""
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=_MUTED, labelsize=8.5, length=0)
    ax.xaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def plot_predictor_comparison(
    configurations: list[dict], out_path: Path, run_label: str
) -> None:
    """F1 (positive class) per configuration, colored by task, with fold std error bars.

    Categorical color by task identity (Task A vs Task B), per the color
    formula: task is what the two bar groups *are*, not a ranked or
    continuous quantity, so it takes fixed categorical slots 1 and 2, never
    a ramp. PR-AUC is in the persisted stats JSON and the printed table
    (this figure's table twin) rather than a second axis on this plot --
    one axis per chart.

    Args:
        configurations: Output of run_all_configurations().
        out_path: PNG destination.
        run_label: Dataset/model/config line for the figure subtitle.
    """
    entries = sorted(
        configurations, key=lambda c: c["summary"]["f1_mean"] or 0.0, reverse=True
    )
    labels = [
        f"{'A' if c['task'] == 'problem_level' else 'B'}: {c['feature_set']} / {c['model_type']}"
        for c in entries
    ]
    values = [c["summary"]["f1_mean"] or 0.0 for c in entries]
    errors = [c["summary"]["f1_std"] or 0.0 for c in entries]
    colors = [_TASK_A if c["task"] == "problem_level" else _TASK_B for c in entries]

    fig, ax = plt.subplots(
        figsize=(9.8, 0.5 * len(entries) + 1.6), constrained_layout=True
    )
    fig.set_facecolor(_SURFACE)
    fig.get_layout_engine().set(rect=(0.0, 0.06, 1.0, 0.94))
    _style_axes(ax)

    positions = np.arange(len(entries))[::-1]
    ax.barh(
        positions,
        values,
        xerr=errors,
        height=0.6,
        color=colors,
        error_kw=dict(ecolor=_MUTED, elinewidth=1.0, capsize=2.5),
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.8, color=_INK)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(
        "F1 (positive class), mean +/- std across CV folds",
        fontsize=9,
        color=_INK_SECONDARY,
    )
    for pos, value, error in zip(positions, values, errors):
        # Clear the error bar's cap, not just the bar itself -- the label
        # must sit past value + error, or it visually collides with the
        # cap for any config whose std pushes the whisker past a fixed
        # bar-relative offset.
        ax.text(
            value + error + 0.015,
            pos,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=8.2,
            color=_INK_SECONDARY,
        )

    # No separate legend box: the y-tick labels already spell out "A:" /
    # "B:" per row, so task identity is never color-alone without one, and
    # a floating legend here would risk colliding with the bottom bars'
    # value labels the way an earlier version of this figure did.
    fig.suptitle(
        "Which predictor configuration works best?",
        fontsize=12.5,
        color=_INK,
        ha="left",
        x=0.012,
    )
    ax.set_title(
        "Task A (blue) = problem level, “will any sample pass?” · "
        "Task B (aqua) = sample level, “will this one?”",
        loc="left",
        fontsize=8.3,
        color=_INK_SECONDARY,
        pad=8,
    )
    fig.text(0.012, 0.01, run_label, fontsize=8, color=_MUTED, ha="left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def print_configuration_table(configurations: list[dict]) -> None:
    """Print the config comparison table to stdout (this figure's table twin)."""
    header = (
        f"\n{'task':<15}{'feature_set':<34}{'model':<15}"
        f"{'F1':<14}{'ROC-AUC':<14}{'PR-AUC':<14}{'n_rows':<8}{'n_groups':<9}"
    )
    print(header)
    for c in sorted(
        configurations, key=lambda c: c["summary"]["f1_mean"] or 0.0, reverse=True
    ):
        s = c["summary"]

        def _fmt(key: str) -> str:
            mean, std = s[f"{key}_mean"], s[f"{key}_std"]
            return f"{mean:.2f}+-{std:.2f}" if mean is not None else "n/a"

        print(
            f"{c['task']:<15}{c['feature_set']:<34}{c['model_type']:<15}"
            f"{_fmt('f1'):<14}{_fmt('roc_auc'):<14}{_fmt('pr_auc'):<14}"
            f"{c['n_rows']:<8}{c['n_groups']:<9}"
        )


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    results_path = args.results if args.results else find_latest_results_file()
    logger.info("Training on %s", results_path)
    with results_path.open() as f:
        results = json.load(f)

    include_codebert = not args.no_codebert
    prompts = load_run_prompts(results)
    problem_rows = build_problem_level_dataset(
        results,
        include_codebert,
        args.codebert_model,
        representation=args.representation,
        prompts=prompts,
        include_tree_edit=not args.no_tree_edit,
    )
    sample_rows = build_sample_level_dataset(
        results,
        include_codebert,
        args.codebert_model,
        representation=args.representation,
        prompts=prompts,
    )
    if not args.no_execution:
        logger.info("Computing execution-behavior features (replaying stored samples)")
        add_execution_behavior_features(
            results,
            problem_rows,
            sample_rows,
            args.execution_timeout,
            probe_source=args.probe_source,
            probe_cap=args.probe_cap,
            error_costs=(
                LEGACY_ERROR_COSTS if args.legacy_error_costs else DEFAULT_ERROR_COSTS
            ),
            seed=args.seed,
        )
    if len(problem_rows) < 4:
        raise ValueError(
            f"Only {len(problem_rows)} problems with samples in {results_path}; "
            "need at least 4 for cross-validated training."
        )

    configurations = run_all_configurations(
        problem_rows, sample_rows, n_splits=args.n_splits, seed=args.seed
    )
    print_configuration_table(configurations)

    config = results.get("config", {})
    run_label = (
        f"{config.get('dataset', '?')} · {config.get('model', '?')} · "
        f"{len(problem_rows)} problems · seed {args.seed} · "
        f"source: {results.get('run_id', results_path.name)}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.fig_name
    plot_predictor_comparison(configurations, out_path, run_label)

    stats = {
        "analysis": "week4_predictor_evaluation",
        "source_results_file": str(results_path),
        "source_run_id": results.get("run_id"),
        "config": {
            "seed": args.seed,
            "n_splits": args.n_splits,
            "codebert_model": args.codebert_model if include_codebert else None,
            "representation": args.representation,
            "include_tree_edit": not args.no_tree_edit,
            "execution_features": not args.no_execution,
            "execution_timeout": args.execution_timeout,
            "probe_source": args.probe_source if not args.no_execution else None,
            "probe_cap": args.probe_cap,
            "error_costs": (
                list(LEGACY_ERROR_COSTS)
                if args.legacy_error_costs
                else list(DEFAULT_ERROR_COSTS)
            ),
        },
        "source_config": config,
        "n_problems": len(problem_rows),
        "n_samples": len(sample_rows),
        "figure": str(out_path),
        "configurations": configurations,
    }
    if args.dry_run:
        logger.info("[DRY RUN] skipping stats JSON persistence")
    else:
        out_json = save_results(
            stats, prefix=f"predictor_{config.get('dataset', 'unknown')}"
        )
        logger.info("Saved predictor evaluation stats to %s", out_json)


if __name__ == "__main__":
    main()
