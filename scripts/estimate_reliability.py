"""Week 7: how much of this project's correlation ceiling is just N=10?

Student A deliverable. Every correlation the project has reported since Week 2
has both variables estimated from the *same ten samples*: `pass_rate` is a
Binomial(10, p) proportion (standard error up to 0.158) and the diversity
metric is a statistic of those same ten. Classical test theory says such a
correlation is **attenuated** — the observed coefficient is the true one
multiplied by sqrt(r_xx * r_yy), where r_xx and r_yy are the two variables'
reliabilities. So there is a hard ceiling on |r| that no better metric can
beat, and the project has never measured it.

Method (standard psychometrics, applied to sampled generations):

  1. Randomly split each problem's N samples into two halves.
  2. Recompute every metric, and the pass rate, on each half independently.
  3. Correlate half A against half B across problems -> split-half reliability.
  4. Spearman-Brown correct it back up to full length: r = 2r_half/(1+r_half).
  5. Average over many random splits (Parsons 2021, JOSS: permutation-based
     split-half is the robust practice; Nicewander 2018 and Padilla &
     Veprinsky 2012 for the attenuation equation itself).

The reported "% of ceiling" is |observed r| / sqrt(r_xx * r_yy). It is
strictly more informative than r alone and costs nothing to add to the
existing correlation tables.

Two honest notes, both worth keeping in the write-up:
  - This is attenuation *correction*, not variance *reduction*. Madaan et al.
    (2024) report that item analysis and IRT struggle to reduce benchmark
    variance; that is a different claim from "the measured correlation is
    scaled down by a knowable factor".
  - Spearman-Brown assumes the two halves are parallel tests. It is on solid
    ground for `pass_rate` (a rate estimate) and shakier for the *pairwise*
    diversity statistics, where halving N takes a problem from 45 pairs to 10.
    Metric reliabilities are therefore the softer half of this table.

Tree edit distance and the CodeBERT metrics are excluded by default: they are
the two expensive families, and this script recomputes every metric 2 x
n_permutations times.

Usage:
    python scripts/estimate_reliability.py --results results/<run>.json
    python scripts/estimate_reliability.py --results ... --n-permutations 200
    python scripts/estimate_reliability.py --results ... --representation corrected
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter
from itertools import combinations

from scripts.analyze_uncertainty import load_run_prompts
from src.uncertainty import (
    _WORD_TOKEN_RE,
    METRIC_REPRESENTATIONS,
    _ast_node_sequence,
    _cosine_distance_from_counts,
    _jaccard_distance,
    _lexeme_sequence,
    _normalized_levenshtein,
    compute_uncertainty,
    compute_uncertainty_prompt_aware,
    prepare_metric_inputs,
)
from src.utils import get_logger, save_results, set_all_seeds

logger = get_logger(__name__)

# Cheap metrics only: this script recomputes them 2 x n_permutations times,
# and tree_edit_distance_diversity alone is ~339x the cost of
# ast_node_diversity per problem (Week 3 report, Section 6).
DEFAULT_METRICS = [
    "lexical_diversity",
    "token_jaccard_diversity",
    "token_cosine_diversity",
    "ast_node_diversity",
    "pairwise_distance",
    "parse_failure_rate",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the reliability entry point."""
    parser = argparse.ArgumentParser(
        description="Split-half reliability and the attenuation ceiling on |r|"
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="Pipeline results JSON to analyze."
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=60,
        help="Random half-splits to average over (default: 60; use >=1000 "
        "for publication-grade error bars).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to estimate reliability for (default: the cheap ones).",
    )
    parser.add_argument(
        "--representation",
        choices=METRIC_REPRESENTATIONS,
        default="full",
        help="Metric input representation (Week 7 prompt-inclusion confound). "
        "Note that reliabilities of the uncorrected measurement are "
        "reliabilities of a confounded quantity.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print but do not persist the stats JSON.",
    )
    return parser.parse_args()


def _metrics_for(codes: list[str], prompt: str | None, representation: str) -> dict:
    """compute_uncertainty over one sample subset, honoring the representation."""
    if prompt is None:
        return compute_uncertainty(codes)
    return compute_uncertainty_prompt_aware(
        codes, prompt, representation=representation
    )


# Metrics that are a mean over sample *pairs*. Their value on any subset is
# the mean of that subset's entries in a precomputed N x N matrix, which is
# what makes hundreds of permutations affordable: a problem has only
# C(10, 2) = 45 distinct pairs, but a naive re-computation would evaluate
# 10 pairs per half x 2 halves x n_permutations times. The AST pair term is
# an O(L^2) edit distance over node sequences hundreds of nodes long, so
# this cache is the difference between minutes and hours per permutation.
_PAIRWISE_METRICS = (
    "token_jaccard_diversity",
    "token_cosine_diversity",
    "ast_node_diversity",
    "pairwise_distance",
)


def build_problem_cache(
    codes: list[str], prompt: str | None, representation: str
) -> dict:
    """Precompute everything needed to score any subset of one problem's samples.

    Args:
        codes: The problem's N stored candidate strings.
        prompt: Its prompt, or None to score full strings uncorrected.
        representation: Metric input representation.
    Returns:
        Dict with "n", per-pair distance matrices for the pairwise metrics,
        and the per-sample data the non-pairwise metrics need.
    """
    if prompt is None or representation == "full":
        lexical_inputs = structural_inputs = list(codes)
    elif representation == "corrected":
        lexical_inputs = prepare_metric_inputs(
            codes, prompt, representation="completion"
        )
        structural_inputs = prepare_metric_inputs(
            codes, prompt, representation="signature"
        )
    else:
        lexical_inputs = structural_inputs = prepare_metric_inputs(
            codes, prompt, representation=representation
        )

    n = len(codes)
    token_sets = [set(_WORD_TOKEN_RE.findall(c)) for c in lexical_inputs]
    token_counts = [Counter(_WORD_TOKEN_RE.findall(c)) for c in lexical_inputs]
    node_sequences = [_ast_node_sequence(c) for c in structural_inputs]

    matrices = {name: np.zeros((n, n)) for name in _PAIRWISE_METRICS}
    for i, j in combinations(range(n), 2):
        pairs = {
            "token_jaccard_diversity": _jaccard_distance(token_sets[i], token_sets[j]),
            "token_cosine_diversity": _cosine_distance_from_counts(
                token_counts[i], token_counts[j]
            ),
            "ast_node_diversity": (
                1.0
                if node_sequences[i] is None or node_sequences[j] is None
                else _normalized_levenshtein(node_sequences[i], node_sequences[j])
            ),
            "pairwise_distance": float(
                abs(len(lexical_inputs[i]) - len(lexical_inputs[j]))
            ),
        }
        for name, distance in pairs.items():
            matrices[name][i, j] = matrices[name][j, i] = distance

    return {
        "n": n,
        "matrices": matrices,
        "lexeme_sequences": [_lexeme_sequence(c) for c in lexical_inputs],
        "parse_ok": [sequence is not None for sequence in node_sequences],
    }


def metrics_from_cache(cache: dict, indices) -> dict[str, float]:
    """Score one subset of a problem's samples from its precomputed cache.

    Reproduces compute_uncertainty()'s definitions exactly for the cheap
    metrics (verified against it by a unit test and by a runtime self-check
    on the first problem of every run).

    Args:
        cache: Output of build_problem_cache().
        indices: Sample indices in the subset.
    Returns:
        Metric name -> value, using the same empty/singleton conventions as
        the canonical implementations (pairwise metrics are 0.0 below two
        samples).
    """
    indices = list(indices)
    k = len(indices)
    values: dict[str, float] = {}
    pairs = list(combinations(indices, 2))
    for name, matrix in cache["matrices"].items():
        values[name] = (
            float(np.mean([matrix[i, j] for i, j in pairs])) if pairs else 0.0
        )
    sequences = [cache["lexeme_sequences"][i] for i in indices]
    values["lexical_diversity"] = len(set(sequences)) / k if k else 0.0
    values["parse_failure_rate"] = (
        sum(1 for i in indices if not cache["parse_ok"][i]) / k if k else 0.0
    )
    return values


def _verify_cache_matches_canonical_metrics(
    entry: dict, metrics: list[str], representation: str
) -> None:
    """Assert the cached reconstruction equals the real metric on full N.

    The cache reimplements the cheap metrics as means over a precomputed
    pair matrix, which is a performance shortcut around
    compute_uncertainty(). A shortcut that silently drifts from the
    canonical definition would corrupt every reliability number here, so
    this checks it once per run against the real implementation, on real
    data, and fails loudly rather than warning.

    Raises:
        ValueError: If any metric disagrees beyond floating-point noise.
    """
    canonical = _metrics_for(entry["codes"], entry["prompt"], representation)
    reconstructed = metrics_from_cache(entry["cache"], range(entry["cache"]["n"]))
    for metric in metrics:
        expected, actual = canonical[metric], reconstructed[metric]
        if abs(expected - actual) > 1e-9:
            raise ValueError(
                f"cached reconstruction of {metric} disagrees with "
                f"compute_uncertainty ({actual!r} vs {expected!r})"
            )
    logger.info("Cache self-check passed on %d metrics", len(metrics))


def split_half_reliabilities(
    results: dict,
    metrics: list[str],
    n_permutations: int,
    seed: int,
    representation: str,
    prompts: dict[str, str],
) -> dict:
    """Permutation-based split-half reliability of each metric and of pass rate.

    Args:
        results: Full pipeline results (CLAUDE.md Section 8 schema).
        metrics: Metric keys to estimate.
        n_permutations: Number of random half-splits.
        seed: Reproducibility seed (CLAUDE.md Constraint #4).
        representation: Metric input representation.
        prompts: task_id -> prompt, for the corrected representations.
    Returns:
        Dict with "pass_rate" and one entry per metric, each holding the
        mean Spearman-Brown-corrected reliability and its standard
        deviation across permutations.
    """
    rng = np.random.default_rng(seed)
    problems = [p for p in results["per_problem"] if p.get("samples")]
    if len(problems) < 3:
        raise ValueError("need at least 3 problems with samples")

    # One pass over the data builds every pair distance; the permutation
    # loop below is then pure arithmetic over those caches.
    logger.info("Building pair-distance caches for %d problems", len(problems))
    caches = []
    for problem in problems:
        samples = problem["samples"]
        if len(samples) < 4:  # two halves of >= 2 samples each
            continue
        codes = [s["code"] for s in samples]
        prompt = prompts.get(problem["task_id"])
        if prompt is not None and not all(c.startswith(prompt) for c in codes):
            prompt = None
        caches.append(
            {
                "passed": [bool(s["passed"]) for s in samples],
                "cache": build_problem_cache(codes, prompt, representation),
                "codes": codes,
                "prompt": prompt,
            }
        )
    if not caches:
        raise ValueError("no problem has enough samples for a half-split")

    _verify_cache_matches_canonical_metrics(caches[0], metrics, representation)

    tracked = ["pass_rate", *metrics]
    per_permutation: dict[str, list[float]] = {name: [] for name in tracked}

    for permutation in range(n_permutations):
        halves: dict[str, tuple[list[float], list[float]]] = {
            name: ([], []) for name in tracked
        }
        for entry in caches:
            n = entry["cache"]["n"]
            order = rng.permutation(n)
            groups = (order[: n // 2], order[n // 2 : 2 * (n // 2)])
            for side, index_group in enumerate(groups):
                passed = [entry["passed"][i] for i in index_group]
                halves["pass_rate"][side].append(sum(passed) / len(passed))
                values = metrics_from_cache(entry["cache"], index_group)
                for name in metrics:
                    halves[name][side].append(values[name])

        for name in tracked:
            left, right = (np.asarray(v, dtype=float) for v in halves[name])
            if len(left) < 3 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
                continue
            half_r = float(np.corrcoef(left, right)[0, 1])
            # Spearman-Brown: reliability of a test twice as long. Clamped
            # below at 0 -- a negative split-half correlation means the
            # estimate is noise, not evidence of negative reliability.
            corrected = 2.0 * half_r / (1.0 + half_r) if half_r > -1.0 else 0.0
            per_permutation[name].append(max(0.0, min(1.0, corrected)))
        if (permutation + 1) % 10 == 0:
            logger.info("  %d/%d permutations", permutation + 1, n_permutations)

    return {
        name: {
            "reliability_mean": float(np.mean(values)) if values else None,
            "reliability_std": float(np.std(values)) if values else None,
            "n_permutations_used": len(values),
        }
        for name, values in per_permutation.items()
    }


def observed_correlations(
    results: dict, metrics: list[str], representation: str, prompts: dict[str, str]
) -> dict:
    """Full-N Pearson/Spearman correlation of each metric with pass rate."""
    rows = []
    for problem in results["per_problem"]:
        samples = problem.get("samples", [])
        if not samples:
            continue
        codes = [s["code"] for s in samples]
        passed = [bool(s["passed"]) for s in samples]
        prompt = prompts.get(problem["task_id"])
        if prompt is not None and not all(c.startswith(prompt) for c in codes):
            prompt = None
        row = {"pass_rate": sum(passed) / len(passed)}
        row.update(_metrics_for(codes, prompt, representation))
        rows.append(row)
    pass_rates = np.array([r["pass_rate"] for r in rows], dtype=float)
    observed = {}
    for metric in metrics:
        values = np.array([r[metric] for r in rows], dtype=float)
        if np.ptp(values) == 0.0:
            observed[metric] = {"pearson_r": None, "spearman_rho": None}
            continue
        observed[metric] = {
            "pearson_r": float(scipy_stats.pearsonr(values, pass_rates)[0]),
            "spearman_rho": float(scipy_stats.spearmanr(values, pass_rates).statistic),
        }
    return observed


def build_ceiling_table(reliabilities: dict, observed: dict) -> dict:
    """Combine reliabilities and observed r into the ceiling / %-of-ceiling table.

    ceiling = sqrt(r_metric * r_pass_rate) is the largest |r| attainable
    even by a metric that measures the true construct perfectly; the
    attenuation equation (r_obs = r_true * sqrt(r_xx * r_yy)) rearranged at
    r_true = 1.

    Args:
        reliabilities: Output of split_half_reliabilities().
        observed: Output of observed_correlations().
    Returns:
        One entry per metric with reliability, ceiling, and percent of it.
    """
    pass_reliability = reliabilities["pass_rate"]["reliability_mean"]
    table = {}
    for metric, entry in observed.items():
        metric_reliability = reliabilities.get(metric, {}).get("reliability_mean")
        r_obs = entry["pearson_r"]
        if not metric_reliability or not pass_reliability or r_obs is None:
            table[metric] = {"ceiling": None, "percent_of_ceiling": None}
            continue
        ceiling = float(np.sqrt(metric_reliability * pass_reliability))
        table[metric] = {
            "metric_reliability": metric_reliability,
            "observed_pearson_r": r_obs,
            "observed_spearman_rho": entry["spearman_rho"],
            "ceiling": ceiling,
            "percent_of_ceiling": (
                float(100.0 * abs(r_obs) / ceiling) if ceiling else None
            ),
        }
    return table


def print_table(
    table: dict, pass_reliability: float | None, n_permutations: int
) -> None:
    """Print the reliability / ceiling table to stdout."""
    print(
        f"\nSplit-half reliability over {n_permutations} permutations; "
        f"pass_rate reliability r_yy = "
        f"{'n/a' if pass_reliability is None else f'{pass_reliability:.3f}'}"
    )
    print(
        f"{'metric':<30}{'r_xx':<10}{'observed r':<14}{'ceiling |r|':<14}"
        f"{'% of ceiling':<14}"
    )
    for metric, entry in table.items():
        if entry.get("ceiling") is None:
            print(f"{metric:<30}{'n/a':<10}")
            continue
        print(
            f"{metric:<30}"
            f"{entry['metric_reliability']:<10.3f}"
            f"{entry['observed_pearson_r']:<+14.3f}"
            f"{entry['ceiling']:<14.3f}"
            f"{entry['percent_of_ceiling']:<14.1f}"
        )


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    with args.results.open() as f:
        results = json.load(f)
    prompts = load_run_prompts(results) if args.representation != "full" else {}

    logger.info(
        "Estimating split-half reliabilities (%d permutations)", args.n_permutations
    )
    reliabilities = split_half_reliabilities(
        results,
        args.metrics,
        args.n_permutations,
        args.seed,
        args.representation,
        prompts,
    )
    observed = observed_correlations(
        results, args.metrics, args.representation, prompts
    )
    table = build_ceiling_table(reliabilities, observed)
    print_table(
        table, reliabilities["pass_rate"]["reliability_mean"], args.n_permutations
    )

    stats = {
        "analysis": "week7_reliability_ceiling",
        "source_results_file": str(args.results),
        "source_run_id": results.get("run_id"),
        "config": {
            "seed": args.seed,
            "n_permutations": args.n_permutations,
            "metrics": args.metrics,
            "representation": args.representation,
        },
        "source_config": results.get("config", {}),
        "reliabilities": reliabilities,
        "ceiling_table": table,
    }
    if args.dry_run:
        logger.info("[DRY RUN] skipping stats JSON persistence")
    else:
        out_path = save_results(stats, prefix="analysis_week7_reliability")
        logger.info("Saved reliability stats to %s", out_path)


if __name__ == "__main__":
    main()
