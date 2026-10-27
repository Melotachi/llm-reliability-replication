"""Call-matched comparison of execution-guided repair against base-model resampling.

The context-mode ladder (Week 9) spends exactly three replacement calls on
every initially failing coordinate and never stops early. Revealing the
*stored* samples 1..3 of the same coordinate's problem spends exactly three
base-model calls on the same coordinate and generates nothing new. The two
policies are therefore matched on coordinates and on call count, which makes
the pair a legitimate matched-pairs comparison -- the control the joint
experiment never shipped, and the one a reviewer asks for when the question
is "what is the baseline?".

Two properties of the comparison must travel with every number it produces:

1. The two arms call **different models**. Replacement calls go to
   Qwen2.5-Coder-1.5B-Instruct; resample calls go to the generator that
   produced the failure (CodeGen-350M-mono). A win for repair on one cohort
   is therefore consistent with repairer capability rather than with the
   repair mechanism, and a tie achieved by the smaller model is the more
   informative half.
2. Resampling is replayed from stored samples. No generation happens here.

The script also answers the "report a t-test" request where a continuous
quantity actually exists: the 200 coverage-matched random allocation replays.
That is a one-sample test of the replay distribution against the uncertainty
gate's single observed value, not a population test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_context_mode_ladder import (  # noqa: E402
    _bh_adjust,
    _wilson_interval,
)
from src.evaluation import mcnemar_mid_p  # noqa: E402
from src.utils import get_logger, save_results  # noqa: E402

LOGGER = get_logger(__name__)

# Arms are the three context modes of the Week 9 ladder, in order of how much
# of the failing program the prompt carries. Both cohorts use the Week 9 pools
# so that the call accounting matches the manuscript's fixed-budget table
# (three candidates per failure, no early stop): 444 calls on HumanEval and
# 162 on MBPP. The Week 7/8 pools are deliberately not used here, because they
# stopped on the first passing candidate and their MBPP error arm is a
# different pool from the one the manuscript reports.
COHORTS: dict[str, dict[str, Any]] = {
    "humaneval": {
        "source": Path("results/humaneval_uncertainty_merged164_20260711.json"),
        "expected_sha256": (
            "9332324b2b7ecc9278b4742c66c2716b5bb81b93e5e8040085fe49f330c5af20"
        ),
        "arms": {
            "none": Path("results/humaneval_week9_arm_resample_sample0_qwen15b.json"),
            "code_only": Path(
                "results/humaneval_week9_arm_code_only_sample0_qwen15b.json"
            ),
            "error": Path(
                "results/humaneval_week9_arm_error_nostop_sample0_qwen15b.json"
            ),
        },
        # Reported mean, standard deviation and n of the coverage-matched random
        # allocation replays, plus the uncertainty gate's single observed value.
        "random_replays": {
            "added_mean": 40.5,
            "added_sd": 2.7,
            "n_replays": 200,
            "smart_added": 36,
        },
    },
    "mbpp": {
        "source": Path("results/mbpp_uncertainty_20260712_024909.json"),
        "expected_sha256": (
            "85c09d30cc342f0c8fc54b92a7d6b26563bfebe43b98f9bc92c681064be5b71b"
        ),
        "arms": {
            "none": Path("results/mbpp_week9_arm_resample_sample0_qwen15b.json"),
            "code_only": Path("results/mbpp_week9_arm_code_only_sample0_qwen15b.json"),
            "error": Path("results/mbpp_week9_arm_error_nostop_sample0_qwen15b.json"),
        },
        "random_replays": {
            "added_mean": 7.56,
            "added_sd": 1.19,
            "n_replays": 200,
            "smart_added": 7,
        },
    },
}

MATCHED_K = 3


def _load(path: Path) -> dict:
    """Load one JSON record file, failing loudly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"required input not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _coordinate_hash(records: list[dict]) -> str:
    """Hash the coordinate list of a pool, so two arms can be proven aligned."""
    payload = json.dumps(
        [[record["task_id"], record["sample_index"]] for record in records],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resample_outcomes(
    source: dict, records: list[dict], k: int = MATCHED_K
) -> list[bool]:
    """Per-coordinate outcome of revealing k further stored samples.

    Args:
        source: The uncertainty run that stored all N samples per problem.
        records: One repair pool's coordinate records, in pool order.
        k: Number of further samples revealed; k=3 matches the ladder's
            three replacement candidates per failure.
    Returns:
        One boolean per coordinate, True when any of samples 1..k passed.
    """
    passed_by_task = {
        problem["task_id"]: [bool(sample["passed"]) for sample in problem["samples"]]
        for problem in source["per_problem"]
    }
    outcomes = []
    for record in records:
        flags = passed_by_task[record["task_id"]]
        outcomes.append(any(flags[1 : 1 + k]))
    return outcomes


def one_sample_t(
    mean: float, sd: float, n: int, reference: float
) -> dict[str, float | int]:
    """One-sample t statistic of a replay distribution against one value.

    Computed from the reported mean and standard deviation rather than from
    the individual replays, which is exact for the t statistic. It describes
    the replay distribution over one frozen repair pool; it is not a test
    about a population of cohorts.
    """
    standard_error = sd / math.sqrt(n)
    t_statistic = (mean - reference) / standard_error
    degrees = n - 1
    # Two-sided p from the Student-t survival function via the incomplete beta
    # function, so the script keeps no scipy dependency for one number.
    x = degrees / (degrees + t_statistic**2)
    p_value = _incomplete_beta(degrees / 2.0, 0.5, x)
    return {
        "mean": mean,
        "sd": sd,
        "n": n,
        "reference": reference,
        "standard_error": standard_error,
        "t": t_statistic,
        "df": degrees,
        "p_two_sided": min(1.0, p_value),
    }


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_beta) * _beta_cf(a, b, x) / a
    return 1.0 - math.exp(log_beta) * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float, iterations: int = 300) -> float:
    """Lentz's continued fraction for the incomplete beta function."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def analyze_cohort(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Compare every repair arm of one cohort against the resample control."""
    source = _load(spec["source"])
    arms: dict[str, Any] = {}
    reference_hash: str | None = None
    control: list[bool] | None = None

    for arm_name, path in spec["arms"].items():
        pool = _load(path)
        records = pool["records"]
        sha = pool["config"].get("selection_sha256")
        if sha != spec["expected_sha256"]:
            raise ValueError(
                f"{path}: selection_sha256 {sha} does not match the cohort's "
                f"expected {spec['expected_sha256']}"
            )
        coordinate_hash = _coordinate_hash(records)
        if reference_hash is None:
            reference_hash = coordinate_hash
            control = resample_outcomes(source, records)
        elif coordinate_hash != reference_hash:
            raise ValueError(f"{path}: coordinate list differs from the first arm")

        repaired = [bool(record["final_passed"]) for record in records]
        assert control is not None
        both = sum(1 for r, c in zip(repaired, control) if r and c)
        repair_only = sum(1 for r, c in zip(repaired, control) if r and not c)
        control_only = sum(1 for r, c in zip(repaired, control) if c and not r)
        neither = sum(1 for r, c in zip(repaired, control) if not r and not c)
        arms[arm_name] = {
            "pool_file": str(path),
            "n_coordinates": len(records),
            "repair_successes": sum(repaired),
            "repair_rate": sum(repaired) / len(records),
            "repair_rate_wilson_95": _wilson_interval(sum(repaired), len(records)),
            "repair_model_calls": pool["summary"]["repair_model_calls"],
            "calls_per_success": pool["summary"].get("model_calls_per_success"),
            "duplicate_candidate_rate": pool["summary"].get("duplicate_candidate_rate"),
            "paired_vs_resample": {
                "both": both,
                "repair_only": repair_only,
                "resample_only": control_only,
                "neither": neither,
                "mcnemar_mid_p": mcnemar_mid_p(repaired, control),
            },
        }

    raw = [arms[a]["paired_vs_resample"]["mcnemar_mid_p"] for a in arms]
    for arm_name, adjusted in zip(arms, _bh_adjust(raw)):
        arms[arm_name]["paired_vs_resample"]["mcnemar_mid_p_bh"] = adjusted

    assert control is not None
    n_coordinates = arms[next(iter(arms))]["n_coordinates"]
    successes = sum(control)
    return {
        "cohort_name": name,
        "source_file": str(spec["source"]),
        "selection_sha256": spec["expected_sha256"],
        "coordinate_hash": reference_hash,
        "matched_k": MATCHED_K,
        "resample_control": {
            "model": source.get("config", {}).get("model"),
            "replayed_from_stored_samples": True,
            "n_coordinates": n_coordinates,
            "added_passes": successes,
            "rate": successes / n_coordinates,
            "rate_wilson_95": _wilson_interval(successes, n_coordinates),
            "base_model_calls": MATCHED_K * n_coordinates,
            "calls_per_success": (
                MATCHED_K * n_coordinates / successes if successes else None
            ),
        },
        "arms": arms,
        "random_allocation_t_test": one_sample_t(
            spec["random_replays"]["added_mean"],
            spec["random_replays"]["added_sd"],
            spec["random_replays"]["n_replays"],
            spec["random_replays"]["smart_added"],
        ),
    }


def main() -> None:
    """Run every cohort and persist one timestamped JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default="analysis_regeneration_baseline",
        help="Filename prefix for the saved results JSON",
    )
    args = parser.parse_args()

    payload = {
        "analysis": "regeneration_baseline",
        "matched_k": MATCHED_K,
        "note": (
            "Repair calls go to Qwen2.5-Coder-1.5B-Instruct; resample calls go "
            "to the generator that produced the failure. The arms are matched "
            "on coordinates and on call count, not on model."
        ),
        "cohorts": [analyze_cohort(name, spec) for name, spec in COHORTS.items()],
    }
    path = save_results(payload, prefix=args.prefix)
    LOGGER.info("wrote %s", path)
    for cohort in payload["cohorts"]:
        control = cohort["resample_control"]
        LOGGER.info(
            "%s: resample k=%d adds %d/%d (%d calls)",
            cohort["cohort_name"],
            MATCHED_K,
            control["added_passes"],
            control["n_coordinates"],
            control["base_model_calls"],
        )
        for arm_name, arm in cohort["arms"].items():
            paired = arm["paired_vs_resample"]
            LOGGER.info(
                "  %-10s %3d/%d calls=%d mid-p=%.3g (BH %.3g) disc %d-%d",
                arm_name,
                arm["repair_successes"],
                arm["n_coordinates"],
                arm["repair_model_calls"],
                paired["mcnemar_mid_p"],
                paired["mcnemar_mid_p_bh"],
                paired["repair_only"],
                paired["resample_only"],
            )
        t = cohort["random_allocation_t_test"]
        LOGGER.info(
            "  random replays t=%.2f df=%d p=%.3g",
            t["t"],
            t["df"],
            t["p_two_sided"],
        )


if __name__ == "__main__":
    main()
