"""Significance tests for the comparisons the IST manuscript reports without one.

The repair-arm comparisons already carry paired mid-p McNemar tests and the
allocation targets carry permutation and bootstrap intervals. Three groups of
comparisons did not: the sample-level and problem-level ROC-AUC ablations and
the best-of-N selection table. This script adds one test per group, chosen for
how that group's data are structured, and replaces the one-sample t-test on the
200 random allocation replays with the test that design actually supports.

1. **Sample-level ROC-AUC (Table tab:auc).** The ten samples of one problem are
   not independent, so DeLong's test (which assumes independent rows) would be
   anti-conservative. The paired difference in pooled out-of-fold ROC-AUC is
   tested with a problem-cluster bootstrap: resample problems with
   replacement, keep each problem's samples together, recompute both AUCs on
   the same resample. Two-sided percentile p-value.
2. **Problem-level ROC-AUC (Table tab:problemauc).** One row per problem, so
   rows are independent and the paired DeLong test (DeLong et al., 1988;
   fast form of Sun & Xu, 2014) applies directly.
3. **Selection accuracy (Table tab:selection).** The baseline is the expected
   accuracy of a uniform draw, i.e. each problem's own pass fraction, not a
   second binary outcome. Under that null the number of passing picks is
   Poisson-binomial with the problems' pass fractions as probabilities, so the
   selector is tested exactly (two-sided mid-p). A Wilcoxon signed-rank test on
   d_i = 1{pass} - pass_rate_i was considered and rejected: d_i is not
   symmetric about zero under this null, and on the stored CodeGen pass rates
   it rejects about 7.4% of uniform draws at alpha = 0.05.
4. **Random allocation replays.** The 200 replays come from one frozen pool, so
   a one-sample t-test's statistic grows with however many replays are run.
   The gate is instead ranked inside the replay distribution: Monte Carlo
   p = (1 + #{replays at least as extreme}) / (1 + n_replays).

Fold-mean ROC-AUC (what the tables print) and pooled out-of-fold ROC-AUC (what
the tests compare) are not the same number; both are written to the JSON.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_context_mode_ladder import _bh_adjust  # noqa: E402
from src.utils import get_logger, save_results, set_all_seeds  # noqa: E402

LOGGER = get_logger(__name__)

# Column order and source files are those of manuscript_number_audit.md §1/§2.
CELLS: dict[str, dict[str, Path]] = {
    "CG/HE164": {
        "predictor": Path("results/predictor_humaneval_20260805_000941.json"),
        "selection": Path("results/selection_humaneval_20260714_111521.json"),
    },
    "DS/HE60": {
        "predictor": Path("results/predictor_humaneval_20260804_232950.json"),
        "selection": Path("results/selection_humaneval_20260714_105452.json"),
    },
    "CG/MBPP": {
        "predictor": Path("results/predictor_mbpp_20260804_232300.json"),
        "selection": Path("results/selection_mbpp_20260714_110252.json"),
    },
    "DS/MBPP": {
        "predictor": Path("results/predictor_mbpp_20260804_233512.json"),
        "selection": Path("results/selection_mbpp_20260714_110236.json"),
    },
}

# (arm A, arm B): the test asks whether A ranks better than B.
SAMPLE_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("self_consistency_plus_execution", "self_consistency_core"),
    ("self_consistency_plus_execution", "execution_self_only"),
    ("execution_self_only", "crash_rate_self_only"),
)
PROBLEM_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("no_codebert_plus_execution", "length_only"),
    ("no_codebert_plus_execution", "crash_rate_only"),
)
SELECTORS: tuple[str, ...] = (
    "classifier__self_consistency_core",
    "classifier__self_consistency_plus_execution",
    "execution_heuristic",
)

REPLAYS: dict[str, Path] = {
    "mbpp": Path("results/analysis_week8_random_gate_sweep_20260829_211142.json"),
    "humaneval": Path(
        "results/analysis_week8_random_gate_sweep_humaneval_20260830_100905.json"
    ),
}


def _load(path: Path) -> dict:
    """Load one JSON record file, failing loudly if it is missing."""
    import json

    if not path.exists():
        raise FileNotFoundError(f"required input not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based (Mann-Whitney) ROC-AUC with mid-ranks for ties.

    Args:
        labels: Binary labels, 1 = positive.
        scores: Scores; higher means more likely positive.
    Returns:
        ROC-AUC, or NaN when either class is absent.
    """
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = stats.rankdata(np.asarray(scores, dtype=float))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cluster_bootstrap_delta_auc(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    groups: Sequence[int],
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired ROC-AUC difference with a cluster (group) bootstrap.

    Whole groups are resampled with replacement so within-group dependence is
    preserved; both AUCs are computed on the same resample, so the difference
    is paired. Resamples containing one class only are discarded and counted.

    Args:
        labels: Binary labels per row.
        scores_a: Scores of arm A per row.
        scores_b: Scores of arm B per row.
        groups: Group (problem) identifier per row.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling generator.
    Returns:
        auc_a, auc_b, delta (A - B), 95% percentile interval, two-sided
        bootstrap p-value, and the number of valid resamples.
    """
    y = np.asarray(labels, dtype=int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    g = np.asarray(groups)
    unique = np.unique(g)
    members = [np.flatnonzero(g == group) for group in unique]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[i] for i in picked])
        auc_a = roc_auc(y[rows], a[rows])
        if math.isnan(auc_a):
            continue
        deltas.append(auc_a - roc_auc(y[rows], b[rows]))
    boot = np.asarray(deltas)
    # Two-sided percentile p: twice the smaller tail on either side of 0, with
    # the +1 correction so a tail no resample reached reports 2/(B+1), not 0.
    tail = (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (boot.size + 1)
    auc_a, auc_b = roc_auc(y, a), roc_auc(y, b)
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta": auc_a - auc_b,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p": float(min(1.0, 2 * tail)),
        "n_valid_resamples": int(boot.size),
    }


def _midrank_placements(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, ...]:
    # DeLong structural components V10 (per positive) and V01 (per negative).
    combined = np.concatenate([pos, neg])
    ranks_all = stats.rankdata(combined)
    ranks_pos = stats.rankdata(pos)
    ranks_neg = stats.rankdata(neg)
    m, n = pos.size, neg.size
    v10 = (ranks_all[:m] - ranks_pos) / n
    v01 = 1.0 - (ranks_all[m:] - ranks_neg) / m
    return v10, v01


def delong_paired(
    labels: Sequence[int], scores_a: Sequence[float], scores_b: Sequence[float]
) -> dict[str, float]:
    """Paired DeLong test for two correlated ROC-AUCs on independent rows.

    DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44(3); midrank form of
    Sun & Xu (2014), IEEE Signal Processing Letters 21(11).

    Args:
        labels: Binary labels per row (rows must be independent).
        scores_a: Scores of arm A.
        scores_b: Scores of arm B.
    Returns:
        auc_a, auc_b, delta (A - B), standard error, z and two-sided p.
    """
    y = np.asarray(labels, dtype=bool)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    v10_a, v01_a = _midrank_placements(a[y], a[~y])
    v10_b, v01_b = _midrank_placements(b[y], b[~y])
    auc_a, auc_b = float(v10_a.mean()), float(v10_b.mean())
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    cov = s10 / y.sum() + s01 / (~y).sum()
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = auc_a - auc_b
    if var <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "delta": delta, "se": 0.0,
                "z": math.nan, "p": 1.0 if delta == 0 else 0.0}
    se = math.sqrt(var)
    z = delta / se
    return {"auc_a": auc_a, "auc_b": auc_b, "delta": delta, "se": se,
            "z": z, "p": float(2 * stats.norm.sf(abs(z)))}


def poisson_binomial_pmf(probabilities: Sequence[float]) -> np.ndarray:
    """Exact distribution of a sum of independent Bernoulli(p_i) variables.

    Args:
        probabilities: Success probability of each independent trial.
    Returns:
        Array whose k-th entry is P(sum == k), k = 0..len(probabilities).
    """
    pmf = np.zeros(len(probabilities) + 1)
    pmf[0] = 1.0
    for count, prob in enumerate(probabilities, start=1):
        # Convolve with one Bernoulli trial; only the first count+1 cells move.
        shifted = pmf[: count].copy()
        pmf[: count + 1] *= 1.0 - prob
        pmf[1 : count + 1] += prob * shifted
    return pmf


def selection_vs_uniform_draw(
    passed: Sequence[bool], pass_rates: Sequence[float]
) -> dict[str, float]:
    """Exact test of a selector against a uniform draw from each problem's pool.

    Under the null that the selector is no better than picking one of the
    problem's samples uniformly at random, problem i's pick passes with
    probability pass_rate_i, independently across problems, so the number of
    passing picks S follows a Poisson-binomial distribution. The per-problem
    difference 1{pass} - pass_rate_i is not symmetric about zero under this
    null (a Wilcoxon signed-rank test would be anti-conservative when pass
    rates are low), which is why the exact distribution is used. Problems with
    pass_rate 0 or 1 contribute a constant and carry no information.

    Args:
        passed: Per problem, whether the selected sample passed.
        pass_rates: Per problem, the fraction of its samples that pass.
    Returns:
        Selector accuracy, expected random accuracy, mean difference, the
        number of informative problems, the observed count and the
        two-sided mid-p value.
    """
    rates = np.asarray(pass_rates, dtype=float)
    picks = np.asarray(passed, dtype=bool)
    pmf = poisson_binomial_pmf(rates)
    observed = int(picks.sum())
    below = float(pmf[:observed].sum())
    at = float(pmf[observed])
    above = float(pmf[observed + 1 :].sum())
    # Two-sided mid-p: twice the smaller one-sided mid-p tail, as for McNemar.
    p_value = min(1.0, 2 * min(below + 0.5 * at, above + 0.5 * at))
    informative = int(((rates > 1e-12) & (rates < 1 - 1e-12)).sum())
    return {
        "accuracy": float(picks.mean()),
        "expected_random": float(rates.mean()),
        "mean_difference": float(picks.mean() - rates.mean()),
        "n_problems": int(rates.size),
        "n_informative": informative,
        "observed_passes": observed,
        "p": p_value,
    }


def monte_carlo_rank_p(
    observed: float, draws: Sequence[float], *, tail: str
) -> float:
    """One-sided Monte Carlo p-value of an observed value inside a replay set.

    Args:
        observed: The policy's single observed value.
        draws: The replay distribution.
        tail: "lower" asks whether observed is unusually small, "upper"
            whether it is unusually large; ties count as at least as extreme.
    Returns:
        (1 + #{draws at least as extreme}) / (1 + len(draws)).
    """
    values = np.asarray(draws, dtype=float)
    if tail == "lower":
        extreme = int((values <= observed).sum())
    elif tail == "upper":
        extreme = int((values >= observed).sum())
    else:
        raise ValueError(f"tail must be 'lower' or 'upper', got {tail!r}")
    return (1 + extreme) / (1 + values.size)


def _configuration(predictor: dict, task: str, feature_set: str) -> dict:
    # Logistic is the model every manuscript AUC table reports.
    for config in predictor["configurations"]:
        if (
            config["task"] == task
            and config["feature_set"] == feature_set
            and config["model_type"] == "logistic"
        ):
            return config
    raise KeyError(f"no logistic {task}/{feature_set} configuration")


def _source_labels(source: dict) -> tuple[list[int], list[int], list[int]]:
    # Sample labels in row order, their problem index, and problem-level labels.
    sample_y, groups, problem_y = [], [], []
    for index, problem in enumerate(source["per_problem"]):
        flags = [int(bool(sample["passed"])) for sample in problem["samples"]]
        sample_y.extend(flags)
        groups.extend([index] * len(flags))
        problem_y.append(int(any(flags)))
    return sample_y, groups, problem_y


def analyze_cell(name: str, paths: dict[str, Path], *, n_boot: int, seed: int) -> dict:
    """Run every RQ1 test for one model/benchmark cell.

    Args:
        name: Cell label as printed in the manuscript.
        paths: The cell's predictor and selection result files.
        n_boot: Cluster bootstrap resamples for the sample-level tests.
        seed: Bootstrap seed.
    Returns:
        Per-comparison results (unadjusted p-values; adjustment is done
        across cells by the caller).
    """
    predictor = _load(paths["predictor"])
    source = _load(Path(predictor["source_results_file"]))
    sample_y, groups, problem_y = _source_labels(source)

    sample_rows = []
    for arm_a, arm_b in SAMPLE_COMPARISONS:
        conf_a = _configuration(predictor, "sample_level", arm_a)
        conf_b = _configuration(predictor, "sample_level", arm_b)
        # Row order is not recorded explicitly, so prove it: the stored
        # out-of-fold labels must equal the source run's labels in order.
        for conf in (conf_a, conf_b):
            if list(map(int, conf["oof_y"])) != sample_y:
                raise ValueError(f"{name}: oof_y of {conf['feature_set']} misaligned")
        test = cluster_bootstrap_delta_auc(
            sample_y, conf_a["oof_proba"], conf_b["oof_proba"], groups,
            n_boot=n_boot, seed=seed,
        )
        test.update({
            "arm_a": arm_a, "arm_b": arm_b,
            "fold_mean_auc_a": conf_a["summary"]["roc_auc_mean"],
            "fold_mean_auc_b": conf_b["summary"]["roc_auc_mean"],
        })
        sample_rows.append(test)

    problem_rows = []
    for arm_a, arm_b in PROBLEM_COMPARISONS:
        conf_a = _configuration(predictor, "problem_level", arm_a)
        conf_b = _configuration(predictor, "problem_level", arm_b)
        for conf in (conf_a, conf_b):
            if list(map(int, conf["oof_y"])) != problem_y:
                raise ValueError(f"{name}: oof_y of {conf['feature_set']} misaligned")
        test = delong_paired(problem_y, conf_a["oof_proba"], conf_b["oof_proba"])
        test.update({
            "arm_a": arm_a, "arm_b": arm_b,
            "fold_mean_auc_a": conf_a["summary"]["roc_auc_mean"],
            "fold_mean_auc_b": conf_b["summary"]["roc_auc_mean"],
        })
        problem_rows.append(test)

    selection = _load(paths["selection"])
    per_problem = selection["per_problem"]
    rates = [problem["pass_rate"] for problem in per_problem]
    selection_rows = []
    for selector in SELECTORS:
        passed = [bool(problem["picks"][selector]["passed"]) for problem in per_problem]
        test = selection_vs_uniform_draw(passed, rates)
        test["selector"] = selector
        selection_rows.append(test)
    stored = {
        (row["a"], row["b"]): row for row in selection["pairwise_mcnemar"]
    }
    pair = ("classifier__self_consistency_plus_execution", "execution_heuristic")
    heuristic_vs_fused = stored.get(pair) or stored.get(pair[::-1])

    return {
        "cell": name,
        "predictor_file": str(paths["predictor"]),
        "selection_file": str(paths["selection"]),
        "sample_level": sample_rows,
        "problem_level": problem_rows,
        "selection": selection_rows,
        "stored_mcnemar_fused_vs_heuristic": heuristic_vs_fused,
    }


def analyze_replays(path: Path) -> dict[str, Any]:
    """Rank the uncertainty gate inside its 200 coverage-matched random replays.

    Args:
        path: A random-gate sweep result with per-draw values.
    Returns:
        Gate values, replay summaries, and one-sided Monte Carlo p-values for
        "the gate adds fewer passes" and "the gate costs more calls".
    """
    sweep = _load(path)
    added = [draw["added_passes"] for draw in sweep["draws"]]
    calls = [draw["repair_model_calls"] for draw in sweep["draws"]]
    gate = sweep["smart"]
    return {
        "source": str(path),
        "n_replays": len(added),
        "gate_added": gate["added_passes"],
        "gate_calls": gate["repair_model_calls"],
        "random_added_mean": float(np.mean(added)),
        "random_calls_mean": float(np.mean(calls)),
        "p_gate_adds_fewer": monte_carlo_rank_p(gate["added_passes"], added, tail="lower"),
        "p_gate_costs_more": monte_carlo_rank_p(
            gate["repair_model_calls"], calls, tail="upper"
        ),
    }


def _adjust(rows: list[dict]) -> None:
    # Benjamini-Hochberg across every row of one family, in place.
    for row, adjusted in zip(rows, _bh_adjust([row["p"] for row in rows])):
        row["p_bh"] = adjusted


def main() -> None:
    """CLI entry point: run all tests and persist one timestamped JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="analysis_ist_significance")
    args = parser.parse_args()
    set_all_seeds(args.seed)

    cells = [
        analyze_cell(name, paths, n_boot=args.n_boot, seed=args.seed)
        for name, paths in CELLS.items()
    ]
    # One BH family per table, matching how each table is read.
    for family in ("sample_level", "problem_level", "selection"):
        _adjust([row for cell in cells for row in cell[family]])
    replays = {cohort: analyze_replays(path) for cohort, path in REPLAYS.items()}

    payload = {
        "analysis": "ist_significance",
        "config": {"n_boot": args.n_boot, "seed": args.seed},
        "cells": cells,
        "random_replays": replays,
    }
    path = save_results(payload, prefix=args.prefix)
    for cell in cells:
        for family in ("sample_level", "problem_level"):
            for row in cell[family]:
                LOGGER.info(
                    "%s %s %s vs %s: delta=%.3f p=%.4g p_bh=%.4g",
                    cell["cell"], family, row["arm_a"], row["arm_b"],
                    row["delta"], row["p"], row["p_bh"],
                )
        for row in cell["selection"]:
            LOGGER.info(
                "%s selection %s: acc=%.3f exp=%.3f informative=%d p=%.4g p_bh=%.4g",
                cell["cell"], row["selector"], row["accuracy"],
                row["expected_random"], row["n_informative"], row["p"], row["p_bh"],
            )
        LOGGER.info("%s stored McNemar fused vs heuristic: %s",
                    cell["cell"], cell["stored_mcnemar_fused_vs_heuristic"])
    for cohort, row in replays.items():
        LOGGER.info("%s replays: %s", cohort, row)
    LOGGER.info("wrote %s", path)


if __name__ == "__main__":
    main()
