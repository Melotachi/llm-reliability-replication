"""Week 13. Is repairability monotone in P(pass), or does it sit in a band?

Every repair gate this project has built is *monotone*: sort the failing
coordinates by a scalar and take a prefix. Week 8 shipped the `risk` direction
(lowest P(pass) first) and lost to a budget-matched random gate on both cohorts;
Week 12 found the sign was simply backwards on HumanEval (allocation ROC 0.706 on
the `error` arm) and discharged the post-hoc sign threat with a random pilot
(`scripts/pilot_gate_direction.py`).

Nobody asked whether the monotone family is the right family. The hypothesis this
script tests is that the payoff sits in a *band* -- repair the near-misses, skip
both the hopeless and the ones that need no help -- and that the deployable rule
is a threshold with a fallback rather than a full ordering. That is a strictly
larger policy family: `band_ordering` at degenerate bounds reproduces
`fixed_ordering` exactly, and a unit test pins the identity so every Week 12
number stays recoverable from this script.

Two traps the design is built around.

1. **Cost coupling.** Under stop-on-pass a coordinate that passes on candidate 1
   costs 1 call and one that never passes costs 3, so `calls` and `repaired` are
   mechanically linked and *calls-per-success* is guaranteed to look better in a
   high-hit-rate band. On HumanEval/`error` the calls-per-success spread across
   P(pass) quintiles is 5.3x but the decoupled calls-per-*attempt* spread is only
   1.4x. Both columns are reported everywhere; the second one is the claimable
   quantity. Same shape as Week 7's crash-support trap and Week 12's attempts
   rule.
2. **Band search is multiple comparisons.** The raw band [0.40, 0.55) has the
   best hit rate on both HumanEval context arms (0.769) on 13 coordinates and 10
   successes. Reporting that is exactly the error the pilot rule exists to
   police. So: the grid is frozen in `POLICY_GRID` before any evaluation, the
   in-sample best is labelled `oracle_band` and is never quotable as achievable,
   significance comes from a label permutation test taken over the *whole grid*
   (which absorbs the multiplicity), and the only deployable number is the
   band chosen out of fold by `pilot_band_ordering`.

Scope rules inherited from the joint track and enforced in the write-up:

* Never compare an MBPP arm rate to a HumanEval one -- different dataset,
  different base model, and the MBPP pool records no resolved device.
* MBPP/`error` carries 10 positives. A null there excludes only a LARGE effect.
* Attempts are counted at the last affordable point (`_at_budget`), and every
  gain is quoted with its attempts column.
* The `none` arm's pool file is `*_arm_resample_*`; the `error` arm's is the
  Week 7/8 pool. There is no file named `*_none_*`.

Usage:
    python scripts/diagnose_band_gates.py --cohort both
    python scripts/diagnose_band_gates.py --cohort humaneval --n-permutations 10000
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_smart_repair import load_json  # noqa: E402
from scripts.diagnose_arm_allocation import (  # noqa: E402
    COHORTS,
    renormalize_stop_on_pass,
)
from scripts.pilot_gate_direction import (  # noqa: E402
    _at_budget,
    _curve,
    build_rows,
    fixed_ordering,
)
from src.utils import get_logger, save_results, set_all_seeds  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_FIGURE_DIR = Path("reports/figures")

#: Fractions of the arm's own Repair-All cost at which every policy is scored.
#: The low end is not decoration. A band whose upper edge sits below the score
#: maximum only *reorders* the top prefix when the fallback is `inverted`, so
#: above the crossover in `prefix_equivalence_attempts` it is provably the same
#: policy as `inverted` on any dataset. Week 12's 25%-and-up grid therefore
#: could not have distinguished them; these fractions can.
DEFAULT_BUDGETS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.40, 0.50, 0.75, 1.00)

#: Fractions of the arm's coordinate count, for the attempt-matched table that
#: controls the cost coupling described in the module docstring.
DEFAULT_ATTEMPT_FRACTIONS: tuple[float, ...] = (
    0.05,
    0.10,
    0.15,
    0.25,
    0.40,
    0.50,
    0.75,
    1.00,
)

#: Pilot fractions for the out-of-fold band choice.
DEFAULT_EPSILONS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50)

#: Pilot sizes (in coordinates) for the band-resolution-power curve.
DEFAULT_PILOT_SIZES: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 50, 75)

#: Marginal calls-per-success thresholds for the stopping rule.
DEFAULT_STOP_THRESHOLDS: tuple[float, ...] = (4.0, 6.0, 8.0, 12.0)

#: Pseudo-attempts used to shrink a band's pilot hit rate towards the pilot's
#: own overall rate before pricing it. Without this the out-of-fold rule picks
#: the narrowest band in the grid every time -- a 1-for-1 band prices better
#: than a 9-for-12 one on raw counts.
SELECTION_PRIOR_ATTEMPTS = 5

#: Scores whose raw units are probabilities, so the raw-band family is meaningful.
#: `initial_chars` is measured in characters, so it gets the quantile family only.
PROBABILITY_SCORES: tuple[str, ...] = ("p_pass",)

# --------------------------------------------------------------------------
# The frozen policy grid
# --------------------------------------------------------------------------
#
# Frozen before any evaluation. A test asserts this literal, so changing it is a
# deliberate act that shows up in a diff rather than a quiet re-search.
#
# `lo`/`hi` are inclusive/exclusive band bounds; None means unbounded on that
# side. `scale="raw"` reads them as score units, `scale="quantile"` as fractions
# of the cohort's own score distribution. `inside`/`outside` name the ordering
# applied within the band and to the fallback remainder -- `outside` is the
# "if the band does not fill the budget, start taking the rest" rule.
POLICY_GRID: tuple[dict[str, Any], ...] = (
    {"name": "risk", "kind": "fixed", "direction": "risk"},
    {"name": "inverted", "kind": "fixed", "direction": "inverted"},
    # Thresholds: attempt everything above the cut, then fall back downwards.
    {"name": "thresh>=0.05", "kind": "band", "scale": "raw", "lo": 0.05, "hi": None},
    {"name": "thresh>=0.10", "kind": "band", "scale": "raw", "lo": 0.10, "hi": None},
    {"name": "thresh>=0.20", "kind": "band", "scale": "raw", "lo": 0.20, "hi": None},
    {"name": "thresh>=0.35", "kind": "band", "scale": "raw", "lo": 0.35, "hi": None},
    # Raw bands, including the band the user proposed ([0.40, 0.55)).
    {"name": "raw[0.10,0.40)", "kind": "band", "scale": "raw", "lo": 0.10, "hi": 0.40},
    {"name": "raw[0.20,0.60)", "kind": "band", "scale": "raw", "lo": 0.20, "hi": 0.60},
    {"name": "raw[0.40,0.55)", "kind": "band", "scale": "raw", "lo": 0.40, "hi": 0.55},
    {"name": "raw[0.30,0.70)", "kind": "band", "scale": "raw", "lo": 0.30, "hi": 0.70},
    # Quantile bands: the percentile reading of the same proposal.
    {
        "name": "q[0.40,0.55)",
        "kind": "band",
        "scale": "quantile",
        "lo": 0.40,
        "hi": 0.55,
    },
    {
        "name": "q[0.50,0.80)",
        "kind": "band",
        "scale": "quantile",
        "lo": 0.50,
        "hi": 0.80,
    },
    {
        "name": "q[0.60,1.00]",
        "kind": "band",
        "scale": "quantile",
        "lo": 0.60,
        "hi": None,
    },
    {
        "name": "q[0.20,0.60)",
        "kind": "band",
        "scale": "quantile",
        "lo": 0.20,
        "hi": 0.60,
    },
)

#: Quintile edges for the descriptive shape table (PR-A reads this).
N_SHAPE_BINS = 5

# --------------------------------------------------------------------------
# Pre-registrations
# --------------------------------------------------------------------------

PRE_REGISTRATIONS: list[dict[str, str]] = [
    {
        "id": "PR-A",
        "statement": (
            "On the one arm with a real allocation signal (HumanEval `error`, "
            "Week 12 ROC 0.706, p=1e-4), repair success is monotone "
            "non-decreasing in P(pass) RANK. There is no interior sweet spot in "
            "the percentile reading."
        ),
        "falsifier": (
            "some interior quintile's hit rate exceeds the top quintile's by "
            "more than that quintile's Wilson half-width"
        ),
    },
    {
        "id": "PR-B",
        "statement": (
            "The user's proposed raw band [0.40, 0.55) beats the `inverted` "
            "ordering at matched ATTEMPTS on both HumanEval context arms, and "
            "survives a label permutation test at alpha=0.05."
        ),
        "falsifier": (
            "the band is <= inverted at every INFORMATIVE attempt-matched point "
            "(every point below its prefix-equivalence crossover, past which the "
            "two orderings cover the same set by construction), OR its ordering "
            "margin fails a label-permutation null at alpha=0.05 at every such "
            "point, OR it does not beat the region ABOVE it (Fisher p >= 0.05) "
            "-- on a monotone gradient any top region beats the cohort average, "
            "so beating the average is not evidence of a band"
        ),
    },
    {
        "id": "PR-C",
        "statement": (
            "Excluding the bottom of the distribution is the robust win: "
            "attempting only p >= 0.10 retains >= 60% of Repair-All's passes for "
            "<= 50% of its calls, on both HumanEval context arms."
        ),
        "falsifier": "either retention < 60% or call share > 50% on either arm",
    },
    {
        "id": "PR-D",
        "statement": (
            "PREDICTED TO FAIL, and pre-registered as such so the null is "
            "publishable: a band chosen OUT OF FOLD by the pilot rule beats the "
            "`inverted` ordering out of sample. Two explanations are not "
            "separated by this test and both must be reported: bands may be "
            "unlearnable at this cohort size, and `select_band_on` prices a band "
            "without regard to its SIZE, so a narrow well-priced band cannot "
            "fill a budget it was selected to spend."
        ),
        "falsifier": (
            "the pilot-selected band is <= inverted at >= 3 of the 4 "
            "sub-full budgets"
        ),
    },
    {
        "id": "PR-E",
        "statement": (
            "Band the ESCALATION, not the first attempt. On the coordinates the "
            "context-free `none` arm fails, escalating only where P(pass) >= 0.20 "
            "captures >= 40% of the context-rescuable coordinates using <= 35% of "
            "the escalation attempts."
        ),
        "falsifier": "capture < 40% of rescues, or attempt share > 35%",
    },
    {
        "id": "PR-F",
        "statement": (
            "Cohort dependence, DESCRIPTIVE ONLY: the band advantage is present "
            "on HumanEval and absent on MBPP. n=2 cohorts cannot identify the "
            "near-miss mechanism the user hypothesises; a third cohort is the "
            "test, and it is the same third cohort Week 12's open item (1) asks "
            "for."
        ),
        "falsifier": (
            "the oracle band beats `inverted` on MBPP by more than it does on "
            "HumanEval (which would make the pattern not cohort-ordered)"
        ),
    },
]


# --------------------------------------------------------------------------
# Small statistics helpers
# --------------------------------------------------------------------------


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because several bands here carry
    fewer than 15 coordinates, where the normal interval leaves the unit
    interval.

    Args:
        successes: Number of successes.
        trials: Number of trials. Zero returns (0.0, 1.0).
        z: Normal quantile; 1.96 is the two-sided 95% interval.
    Returns:
        (lower, upper).
    """
    if trials <= 0:
        return 0.0, 1.0
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order.

    The project reports BH-adjusted p-values wherever a family of tests is run
    together (Weeks 5, 6, 9); the band grid is such a family.

    Args:
        pvalues: Raw p-values.
    Returns:
        Adjusted p-values, same order, each clipped to 1.0.
    """
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 1.0
    for rank, index in enumerate(reversed(order), start=1):
        value = pvalues[index] * n / (n - rank + 1)
        running = min(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------
# Band orderings
# --------------------------------------------------------------------------


def resolve_bounds(
    rows: list[dict[str, Any]],
    *,
    lo: float | None,
    hi: float | None,
    score: str,
    scale: str,
) -> tuple[float, float]:
    """Turn a band specification into concrete score bounds.

    Args:
        rows: Output of build_rows().
        lo: Lower bound, or None for unbounded below.
        hi: Upper bound, or None for unbounded above.
        score: Row key the band is defined over.
        scale: "raw" (bounds are score units) or "quantile" (bounds are
            fractions of this cohort's own score distribution, which is the
            percentile reading of the same proposal and the only one comparable
            across cohorts with different calibration).
    Returns:
        (low, high) in score units, half-open [low, high).
    Raises:
        ValueError: On an unknown scale, or a quantile bound outside [0, 1].
    """
    if scale == "raw":
        return (
            -math.inf if lo is None else float(lo),
            math.inf if hi is None else float(hi),
        )
    if scale != "quantile":
        raise ValueError(f"unknown scale {scale!r}")
    for bound in (lo, hi):
        if bound is not None and not 0.0 <= bound <= 1.0:
            raise ValueError(f"quantile bound {bound} outside [0, 1]")
    values = np.array([row[score] for row in rows], dtype=float)
    low = -math.inf if lo is None else float(np.quantile(values, lo))
    high = math.inf if hi is None else float(np.quantile(values, hi))
    return low, high


def band_ordering(
    rows: list[dict[str, Any]],
    *,
    lo: float | None,
    hi: float | None,
    score: str = "p_pass",
    scale: str = "raw",
    inside: str = "inverted",
    outside: str = "inverted",
) -> list[dict[str, Any]]:
    """Attempt the band [lo, hi) first, then everything outside it.

    This is the generalisation the week is about. `band_ordering(lo=None,
    hi=None, inside=d)` is bit-identical to `fixed_ordering(direction=d)`, so
    the monotone gates of Weeks 8 and 12 are members of this family rather than
    a separate thing being compared against it.

    `outside` is the user's fallback rule: when the band does not fill the
    budget, the remainder is attempted in that direction rather than dropped.
    Dropping is expressible too -- score the ordering at a budget the band
    exhausts -- so no policy has to be re-expressed to answer either question.

    Args:
        rows: Output of build_rows().
        lo: Lower band bound, or None.
        hi: Upper band bound, or None.
        score: Row key the band and the orderings read.
        scale: "raw" or "quantile", passed to resolve_bounds().
        inside: "risk" (lowest first) or "inverted" (highest first), within band.
        outside: Same, for the fallback remainder.
    Returns:
        Rows in attempt order; ties broken by source order for reproducibility.
    """
    low, high = resolve_bounds(rows, lo=lo, hi=hi, score=score, scale=scale)
    inside_rows = [row for row in rows if low <= row[score] < high]
    outside_rows = [row for row in rows if not low <= row[score] < high]
    return fixed_ordering(inside_rows, direction=inside, score=score) + fixed_ordering(
        outside_rows, direction=outside, score=score
    )


def ordering_for(
    rows: list[dict[str, Any]], policy: dict[str, Any], *, score: str
) -> list[dict[str, Any]]:
    """Build the attempt ordering for one member of POLICY_GRID.

    Args:
        rows: Output of build_rows().
        policy: A grid entry.
        score: Row key the policy is applied over.
    Returns:
        Rows in attempt order.
    Raises:
        ValueError: On an unknown policy kind.
    """
    if policy["kind"] == "fixed":
        return fixed_ordering(rows, direction=policy["direction"], score=score)
    if policy["kind"] == "band":
        return band_ordering(
            rows,
            lo=policy["lo"],
            hi=policy["hi"],
            score=score,
            scale=policy["scale"],
            inside=policy.get("inside", "inverted"),
            outside=policy.get("outside", "inverted"),
        )
    raise ValueError(f"unknown policy kind {policy['kind']!r}")


def prefix_equivalence_attempts(
    rows: list[dict[str, Any]], policy: dict[str, Any], *, score: str
) -> int | None:
    """Attempt count above which this policy stops differing from `inverted`.

    A band ordering attempts its members first and the remainder in the fallback
    direction. When the fallback is `inverted` and the band's upper edge sits
    below the cohort's maximum, the two orderings cover the *same set* at every
    prefix at or beyond `len(band) + len(above the band)` -- they differ only in
    the order within that prefix. Any budget past the crossover therefore scores
    the two as identical, for structural reasons that hold on any dataset.

    This is a property of the policy family, not a finding about these cohorts,
    and it is what makes the low end of the budget grid load-bearing: a band can
    only ever pay before its crossover.

    Args:
        rows: Output of build_rows().
        policy: A grid entry.
        score: Row key the policy is applied over.
    Returns:
        The smallest k from which the two prefixes are set-identical, or None if
        they never diverge (the policy IS `inverted`) -- computed by comparison
        rather than by formula, so a change in the fallback rule cannot make the
        reported number quietly wrong.
    """
    ordering = ordering_for(rows, policy, score=score)
    reference = fixed_ordering(rows, direction="inverted", score=score)
    crossover = 0
    for index in range(len(rows), 0, -1):
        left = {row["task_id"] for row in ordering[:index]}
        right = {row["task_id"] for row in reference[:index]}
        if left != right:
            crossover = index + 1
            break
    return None if crossover == 0 else crossover


def grid_for_score(score: str) -> tuple[dict[str, Any], ...]:
    """The frozen grid, minus members whose units do not apply to this score.

    Raw bands are probability bounds. Applying them to `initial_chars` (measured
    in characters, so every coordinate sits above 0.55) would silently degenerate
    every raw member into the same ordering, which reads as agreement when it is
    really vacuity -- so those members are dropped rather than scored.

    Args:
        score: Row key the grid will be applied over.
    Returns:
        The applicable subset of POLICY_GRID, in grid order.
    """
    if score in PROBABILITY_SCORES:
        return POLICY_GRID
    return tuple(
        policy
        for policy in POLICY_GRID
        if policy["kind"] != "band" or policy["scale"] != "raw"
    )


# --------------------------------------------------------------------------
# Curve accounting
# --------------------------------------------------------------------------


def _at_attempts(
    points: list[tuple[int, int, int]], n_attempts: int
) -> tuple[int, int]:
    """Added passes and calls after exactly `n_attempts` attempts.

    The attempt-matched counterpart of `_at_budget`, and the control for the cost
    coupling in the module docstring: at equal *calls* a policy that front-loads
    cheap coordinates has bought more attempts, so a gain measured there is part
    ordering quality and part arithmetic. At equal attempts only the ordering
    differs.

    Args:
        points: Output of `_curve`, i.e. (calls, added, attempts) cumulative.
        n_attempts: Attempt count to read the curve at; clipped to its length.
    Returns:
        (added passes, calls spent).
    """
    index = max(0, min(n_attempts, len(points) - 1))
    calls, added, _ = points[index]
    return added, calls


def marginal_cost_curve(
    ordering: list[dict[str, Any]], *, window: int
) -> list[dict[str, Any]]:
    """Trailing-window calls-per-success along an attempt ordering.

    The quantity a deployment actually feels: not "is this coordinate likely to
    be repaired" but "am I still buying passes at an acceptable price". A window
    is used rather than the cumulative average because the cumulative one is
    dominated by its own prefix and never rises fast enough to stop on.

    Args:
        ordering: Rows in attempt order.
        window: Trailing window length in attempts.
    Returns:
        One entry per attempt: cumulative calls/added/attempts plus the window's
        calls, successes and calls-per-success (None when the window is barren,
        which is an infinite price rather than a missing value).
    """
    out = []
    calls, added = 0, 0
    for index, row in enumerate(ordering):
        calls += row["calls"]
        added += row["repaired"]
        lower = max(0, index - window + 1)
        window_rows = ordering[lower : index + 1]
        window_calls = sum(item["calls"] for item in window_rows)
        window_hits = sum(item["repaired"] for item in window_rows)
        out.append(
            {
                "attempts": index + 1,
                "calls": calls,
                "added": added,
                "window_calls": window_calls,
                "window_successes": window_hits,
                "window_calls_per_success": (
                    None if window_hits == 0 else window_calls / window_hits
                ),
            }
        )
    return out


def stop_at_marginal_cost(
    ordering: list[dict[str, Any]], *, window: int, threshold: float
) -> dict[str, Any]:
    """Stop once the trailing window's price per pass exceeds `threshold`.

    The rule only becomes eligible to fire after a full window has been
    attempted, so a run of early failures inside the first window cannot stop a
    policy before it has been given the window's worth of evidence.

    Args:
        ordering: Rows in attempt order.
        window: Trailing window length in attempts.
        threshold: Marginal calls-per-success above which the rule stops.
    Returns:
        The (attempts, calls, added) at the stop, and whether it fired at all.
    """
    curve = marginal_cost_curve(ordering, window=window)
    for entry in curve:
        if entry["attempts"] < window:
            continue
        price = entry["window_calls_per_success"]
        if price is None or price > threshold:
            return {
                "stopped": True,
                "attempts": entry["attempts"],
                "calls": entry["calls"],
                "added": entry["added"],
            }
    last = curve[-1] if curve else {"attempts": 0, "calls": 0, "added": 0}
    return {
        "stopped": False,
        "attempts": last["attempts"],
        "calls": last["calls"],
        "added": last["added"],
    }


# --------------------------------------------------------------------------
# Descriptive shape
# --------------------------------------------------------------------------


def shape_table(
    rows: list[dict[str, Any]], *, score: str, n_bins: int = N_SHAPE_BINS
) -> dict[str, Any]:
    """Hit rate and both cost columns by equal-count bin of `score`.

    Equal-count (rank) bins, not equal-width, because P(pass) is heavily skewed
    on both cohorts (HumanEval median 0.092) and equal-width bins would put most
    of the cohort in one cell. This is the percentile reading PR-A is scored on.

    Args:
        rows: Output of build_rows().
        score: Row key to bin by.
        n_bins: Number of equal-count bins.
    Returns:
        Per-bin counts, hit rate with its Wilson interval, calls per attempt and
        calls per success, plus whether the sequence of hit rates is monotone
        non-decreasing and by how much any interior bin exceeds the top one.
    """
    n = len(rows)
    order = sorted(range(n), key=lambda i: (rows[i][score], rows[i]["source_order"]))
    bins = []
    for index in range(n_bins):
        chunk = order[index * n // n_bins : (index + 1) * n // n_bins]
        if not chunk:
            continue
        values = [rows[i][score] for i in chunk]
        hits = sum(rows[i]["repaired"] for i in chunk)
        calls = sum(rows[i]["calls"] for i in chunk)
        low, high = wilson_interval(hits, len(chunk))
        bins.append(
            {
                "bin": index + 1,
                "score_min": min(values),
                "score_max": max(values),
                "n": len(chunk),
                "successes": hits,
                "hit_rate": hits / len(chunk),
                "hit_rate_ci95": [low, high],
                "calls": calls,
                "calls_per_attempt": calls / len(chunk),
                "calls_per_success": (calls / hits) if hits else None,
            }
        )

    rates = [entry["hit_rate"] for entry in bins]
    top = rates[-1] if rates else 0.0
    excess = []
    for entry in bins[:-1]:
        low, high = entry["hit_rate_ci95"]
        half_width = (high - low) / 2.0
        excess.append(
            {
                "bin": entry["bin"],
                "hit_rate": entry["hit_rate"],
                "excess_over_top": entry["hit_rate"] - top,
                "wilson_half_width": half_width,
                # PR-A's falsifier: an interior bin beating the top by more than
                # its own noise. A bare "beats the top" would fire on any wiggle.
                "beats_top_beyond_noise": (entry["hit_rate"] - top) > half_width,
            }
        )
    return {
        "score_column": score,
        "n_bins": len(bins),
        "bins": bins,
        "monotone_non_decreasing": all(
            rates[i] <= rates[i + 1] for i in range(len(rates) - 1)
        ),
        "interior_bins_vs_top": excess,
        "any_interior_bin_beats_top_beyond_noise": any(
            entry["beats_top_beyond_noise"] for entry in excess
        ),
        "calls_per_attempt_spread": (
            max(entry["calls_per_attempt"] for entry in bins)
            / min(entry["calls_per_attempt"] for entry in bins)
            if bins
            else None
        ),
        "calls_per_success_spread": _cost_spread(bins),
    }


def _cost_spread(bins: list[dict[str, Any]]) -> float | None:
    """Max/min calls-per-success across bins, ignoring bins with no successes."""
    values = [
        entry["calls_per_success"]
        for entry in bins
        if entry["calls_per_success"] is not None
    ]
    if len(values) < 2:
        return None
    return max(values) / min(values)


def band_summary(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    score: str,
) -> dict[str, Any]:
    """Membership, hit rate and both cost columns for one band policy.

    Args:
        rows: Output of build_rows().
        policy: A grid entry; a "fixed" entry covers the whole cohort.
        score: Row key the band is defined over.
    Returns:
        The band's own statistics (not the ordering's), which is what the
        descriptive table and the permutation test read.
    """
    if policy["kind"] == "fixed":
        inside = list(rows)
        low, high = -math.inf, math.inf
    else:
        low, high = resolve_bounds(
            rows, lo=policy["lo"], hi=policy["hi"], score=score, scale=policy["scale"]
        )
        inside = [row for row in rows if low <= row[score] < high]
    hits = sum(row["repaired"] for row in inside)
    calls = sum(row["calls"] for row in inside)
    ci_low, ci_high = wilson_interval(hits, len(inside))
    total_hits = sum(row["repaired"] for row in rows)
    total_calls = sum(row["calls"] for row in rows)
    return {
        "policy": policy["name"],
        "resolved_low": None if low == -math.inf else low,
        "resolved_high": None if high == math.inf else high,
        "n": len(inside),
        "successes": hits,
        "hit_rate": (hits / len(inside)) if inside else None,
        "hit_rate_ci95": [ci_low, ci_high],
        "calls": calls,
        "calls_per_attempt": (calls / len(inside)) if inside else None,
        "calls_per_success": (calls / hits) if hits else None,
        "pass_retention": (hits / total_hits) if total_hits else None,
        "call_share": (calls / total_calls) if total_calls else None,
    }


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------


def permutation_test(
    rows: list[dict[str, Any]],
    *,
    score: str,
    policies: tuple[dict[str, Any], ...],
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Label-permutation p-values for band hit rates, single and grid-wide.

    Permuting `repaired` across coordinates destroys any relationship between
    the score and repairability while preserving the band memberships and the
    cohort's marginal success count exactly, which is what makes it the right
    null for "could a band this good come from a score that carries nothing".

    Two p-values are produced and they answer different questions:

    * `single`, per band: how often that one band's hit rate is matched under the
      null. Not corrected for having looked at the others.
    * `grid_max`: how often the *best band in the whole grid* is matched under
      the null. This is the multiplicity-absorbing number, and it is the one the
      write-up quotes for any claim of the form "the best band is real".

    Args:
        rows: Output of build_rows().
        score: Row key the bands are defined over.
        policies: Grid members to test; "fixed" members are skipped, since a
            whole-cohort "band" has the observed hit rate under every permutation.
        n_permutations: Permutation count.
        rng: Seeded generator.
    Returns:
        Per-band observed hit rates with raw and BH-adjusted single p-values,
        plus the grid-wide maximum's p-value.
    """
    labels = np.array([row["repaired"] for row in rows], dtype=int)
    band_members = [policy for policy in policies if policy["kind"] == "band"]

    masks, observed = [], []
    for policy in band_members:
        low, high = resolve_bounds(
            rows, lo=policy["lo"], hi=policy["hi"], score=score, scale=policy["scale"]
        )
        mask = np.array([low <= row[score] < high for row in rows], dtype=bool)
        masks.append(mask)
        observed.append(float(labels[mask].mean()) if mask.any() else 0.0)

    observed_max = max(observed) if observed else 0.0
    # Start both counters at 1 for the observed arrangement itself: the
    # permutation p-value is (1 + #{null >= observed}) / (1 + n), which cannot
    # return an impossible zero.
    single_counts = [1] * len(band_members)
    max_count = 1
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        best = 0.0
        for index, mask in enumerate(masks):
            if not mask.any():
                continue
            rate = float(shuffled[mask].mean())
            if rate >= observed[index]:
                single_counts[index] += 1
            best = max(best, rate)
        if best >= observed_max:
            max_count += 1

    raw = [count / (n_permutations + 1) for count in single_counts]
    adjusted = benjamini_hochberg(raw)
    return {
        "n_permutations": n_permutations,
        "bands": [
            {
                "policy": policy["name"],
                "n": int(mask.sum()),
                "observed_hit_rate": rate,
                "p_single": p_raw,
                "p_single_bh": p_adj,
            }
            for policy, mask, rate, p_raw, p_adj in zip(
                band_members, masks, observed, raw, adjusted
            )
        ],
        "grid_max": {
            "best_band": (
                band_members[observed.index(observed_max)]["name"]
                if band_members
                else None
            ),
            "observed_hit_rate": observed_max,
            "p_grid_max": max_count / (n_permutations + 1),
        },
    }


def neighbourhood_test(
    rows: list[dict[str, Any]], policy: dict[str, Any], *, score: str
) -> dict[str, Any]:
    """Does the band beat the region ABOVE it, or only the cohort average?

    The grid-wide permutation in `permutation_test` asks whether a band's hit
    rate could arise from a score carrying nothing. On a monotone gradient that
    question is answered "no" by *any* top region, so passing it is not evidence
    of a peak. The claim "the payoff sits in a band rather than at the top"
    requires the band to beat its own upper neighbourhood, and that is a
    two-by-two comparison on small counts -- Fisher's exact test, not an
    asymptotic one.

    Args:
        rows: Output of build_rows().
        policy: A band grid entry; a "fixed" entry or an unbounded-above band
            has no neighbourhood and returns None fields.
        score: Row key the band is defined over.
    Returns:
        Both cells' counts with Wilson intervals, the odds ratio and the
        two-sided Fisher p-value.
    """
    if policy["kind"] != "band" or policy["hi"] is None:
        return {"applicable": False}
    low, high = resolve_bounds(
        rows, lo=policy["lo"], hi=policy["hi"], score=score, scale=policy["scale"]
    )
    inside = [row for row in rows if low <= row[score] < high]
    above = [row for row in rows if row[score] >= high]
    if not inside or not above:
        return {"applicable": False}
    in_hits = sum(row["repaired"] for row in inside)
    above_hits = sum(row["repaired"] for row in above)
    odds, pvalue = fisher_exact(
        [
            [in_hits, len(inside) - in_hits],
            [above_hits, len(above) - above_hits],
        ]
    )
    return {
        "applicable": True,
        "band_n": len(inside),
        "band_successes": in_hits,
        "band_hit_rate": in_hits / len(inside),
        "band_ci95": list(wilson_interval(in_hits, len(inside))),
        "above_n": len(above),
        "above_successes": above_hits,
        "above_hit_rate": above_hits / len(above),
        "above_ci95": list(wilson_interval(above_hits, len(above))),
        "odds_ratio": float(odds),
        "p_fisher_two_sided": float(pvalue),
    }


def ordering_margin_permutation(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    score: str,
    attempt_counts: list[int],
    n_permutations: int,
    rng: np.random.Generator,
    reference_direction: str = "inverted",
) -> dict[str, Any]:
    """A null for the ORDERING margin, which is not what the hit-rate test tests.

    `permutation_test` nulls a band's hit rate. The quantity actually claimed
    when a band is proposed as a gate is that its *ordering* delivers more passes
    than the monotone one at a matched attempt count, and a margin of one or two
    passes on a few dozen positives needs its own null: two coordinate flips can
    erase it.

    Labels are permuted across coordinates and both orderings are rebuilt from
    the permuted rows, so the null preserves the band memberships, the cost
    structure and the marginal success count while destroying only the
    label-to-score relationship.

    Args:
        rows: Output of build_rows().
        policy: The band grid entry whose ordering is being claimed.
        score: Row key both orderings read.
        attempt_counts: Attempt points to test, normally the informative ones.
        n_permutations: Permutation count.
        rng: Seeded generator.
        reference_direction: The monotone ordering the band must beat.
    Returns:
        Observed margin and one-sided p-value per attempt point.
    """
    reference = fixed_ordering(rows, direction=reference_direction, score=score)
    observed = []
    band_points = _curve(ordering_for(rows, policy, score=score))
    reference_points = _curve(reference)
    for count in attempt_counts:
        observed.append(
            _at_attempts(band_points, count)[0]
            - _at_attempts(reference_points, count)[0]
        )

    shuffled_rows = [dict(row) for row in rows]
    labels = np.array([row["repaired"] for row in rows], dtype=int)
    # Start at 1 for the observed arrangement, as in permutation_test.
    counts_ge = [1] * len(attempt_counts)
    for _ in range(n_permutations):
        permuted = rng.permutation(labels)
        for row, value in zip(shuffled_rows, permuted):
            row["repaired"] = int(value)
        band_null = _curve(ordering_for(shuffled_rows, policy, score=score))
        reference_null = _curve(
            fixed_ordering(shuffled_rows, direction=reference_direction, score=score)
        )
        for index, count in enumerate(attempt_counts):
            margin = (
                _at_attempts(band_null, count)[0]
                - _at_attempts(reference_null, count)[0]
            )
            if margin >= observed[index]:
                counts_ge[index] += 1
    return {
        "reference_direction": reference_direction,
        "attempt_counts": list(attempt_counts),
        "observed_margin": observed,
        "p_margin": [count / (n_permutations + 1) for count in counts_ge],
    }


# --------------------------------------------------------------------------
# Policy evaluation
# --------------------------------------------------------------------------


def evaluate_grid(
    rows: list[dict[str, Any]],
    *,
    score: str,
    budgets: tuple[float, ...],
    attempt_fractions: tuple[float, ...],
    n_seeds: int,
    first_seed: int,
    stop_window: int,
    stop_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    """Score every applicable grid member, plus a random null and the oracle.

    Args:
        rows: Output of build_rows().
        score: Row key every policy is applied over.
        budgets: Fractions of the arm's Repair-All cost.
        attempt_fractions: Fractions of the arm's coordinate count.
        n_seeds: Draws averaged for the random baseline.
        first_seed: Lowest seed.
        stop_window: Trailing window for the marginal-cost stopping rule.
        stop_thresholds: Calls-per-success thresholds for that rule.
    Returns:
        Per-policy tables at matched calls and matched attempts, the band
        statistics, the stopping-rule outcomes, and the in-sample best member
        (`oracle_band`, which is an upper bound and not a deployable number).
    """
    n = len(rows)
    total_calls = sum(row["calls"] for row in rows)
    total_repaired = sum(row["repaired"] for row in rows)
    budget_calls = [int(round(fraction * total_calls)) for fraction in budgets]
    attempt_counts = [int(round(fraction * n)) for fraction in attempt_fractions]
    policies = grid_for_score(score)

    scored: dict[str, Any] = {}
    for policy in policies:
        ordering = ordering_for(rows, policy, score=score)
        points = _curve(ordering)
        at_budget = [_at_budget(points, budget) for budget in budget_calls]
        at_attempts = [_at_attempts(points, count) for count in attempt_counts]
        scored[policy["name"]] = {
            "added_at_budget": [value[0] for value in at_budget],
            "attempts_at_budget": [value[1] for value in at_budget],
            "added_at_attempts": [value[0] for value in at_attempts],
            "calls_at_attempts": [value[1] for value in at_attempts],
            "band": band_summary(rows, policy, score=score),
            "prefix_equivalence_attempts": prefix_equivalence_attempts(
                rows, policy, score=score
            ),
            "stopping_rule": {
                f"tau={threshold:g}": stop_at_marginal_cost(
                    ordering, window=stop_window, threshold=threshold
                )
                for threshold in stop_thresholds
            },
            "stochastic": False,
        }

    random_added: dict[int, list[int]] = {budget: [] for budget in budget_calls}
    random_attempts: dict[int, list[int]] = {budget: [] for budget in budget_calls}
    random_by_attempts: dict[int, list[int]] = {count: [] for count in attempt_counts}
    for seed in range(first_seed, first_seed + n_seeds):
        rng = np.random.default_rng(seed)
        shuffled = [rows[index] for index in rng.permutation(n)]
        points = _curve(shuffled)
        for budget in budget_calls:
            added, attempts = _at_budget(points, budget)
            random_added[budget].append(added)
            random_attempts[budget].append(attempts)
        for count in attempt_counts:
            random_by_attempts[count].append(_at_attempts(points, count)[0])
    scored["random"] = {
        "added_at_budget": [float(np.mean(random_added[b])) for b in budget_calls],
        "added_at_budget_std": [
            float(np.std(random_added[b], ddof=1)) for b in budget_calls
        ],
        "attempts_at_budget": [
            float(np.mean(random_attempts[b])) for b in budget_calls
        ],
        "added_at_attempts": [
            float(np.mean(random_by_attempts[c])) for c in attempt_counts
        ],
        "calls_at_attempts": None,
        "band": None,
        "stopping_rule": None,
        "stochastic": True,
    }

    # In-sample best over the grid, recorded at EVERY budget because the family
    # collapses onto `inverted` above each band's crossover: an oracle read at a
    # large budget is guaranteed to be a tie and says nothing. The headline
    # oracle is therefore taken at the smallest budget, which is the only region
    # where the members are distinguishable at all.
    oracle_by_budget = {
        f"{fraction:.2f}": max(
            (policy["name"] for policy in policies),
            key=lambda name: scored[name]["added_at_budget"][index],
        )
        for index, fraction in enumerate(budgets)
    }
    mid = 0
    oracle_name = oracle_by_budget[f"{budgets[mid]:.2f}"]
    return {
        "score_column": score,
        "n_coordinates": n,
        "total_repair_calls": total_calls,
        "total_repairable": total_repaired,
        "budget_fractions": list(budgets),
        "budget_calls": budget_calls,
        "attempt_fractions": list(attempt_fractions),
        "attempt_counts": attempt_counts,
        "stop_window": stop_window,
        "policies": scored,
        "oracle_band": {
            "policy": oracle_name,
            "selected_at_budget_fraction": budgets[mid],
            "added": scored[oracle_name]["added_at_budget"][mid],
            "by_budget": oracle_by_budget,
            "note": (
                "in-sample maximum over the frozen grid; an upper bound, never "
                "a deployable number -- see pilot_band_ordering"
            ),
        },
    }


# --------------------------------------------------------------------------
# Out-of-fold band selection
# --------------------------------------------------------------------------


def select_band_on(
    pilot: list[dict[str, Any]],
    *,
    score: str,
    policies: tuple[dict[str, Any], ...],
) -> str:
    """Pick the grid member with the best shrunk price per pass on `pilot`.

    Selection reads calls-per-success rather than hit rate because the objective
    is passes per call. The hit rate is first shrunk towards the pilot's own
    overall rate with `SELECTION_PRIOR_ATTEMPTS` pseudo-attempts, which is what
    stops the dominant overfitting mode at these pilot sizes: without it a band
    that went 1-for-1 beats one that went 2-for-2 purely by being smaller, and
    the rule would systematically select the narrowest band in the grid.

    Args:
        pilot: The pilot rows.
        score: Row key the bands are defined over.
        policies: Grid members to choose among.
    Returns:
        The chosen policy's name. Ties break on the wider band and then on the
        policy name, so the choice is deterministic given the pilot.
    """
    total = len(pilot)
    base_rate = sum(row["repaired"] for row in pilot) / total if total else 0.0

    def price(policy: dict[str, Any]) -> tuple[float, int]:
        summary = band_summary(pilot, policy, score=score)
        if summary["n"] == 0:
            return math.inf, 0
        shrunk = (summary["successes"] + SELECTION_PRIOR_ATTEMPTS * base_rate) / (
            summary["n"] + SELECTION_PRIOR_ATTEMPTS
        )
        if shrunk <= 0.0:
            return math.inf, -summary["n"]
        return summary["calls"] / summary["n"] / shrunk, -summary["n"]

    return min(policies, key=lambda policy: (price(policy), policy["name"]))["name"]


def pilot_band_ordering(
    rows: list[dict[str, Any]],
    *,
    epsilon: float,
    rng: np.random.Generator,
    score: str,
    policies: tuple[dict[str, Any], ...],
) -> tuple[list[dict[str, Any]], str]:
    """Spend a random pilot, choose a band on it alone, commit for the rest.

    The generalisation of `pilot_gate_direction.pilot_ordering` from a binary
    sign to a choice among the frozen grid. The pilot's own repairs count towards
    the total, so the rule's cost is not the pilot itself but the fact that those
    coordinates were ordered at random -- and the choice is made on strictly
    fewer coordinates than the full-data best, which is the point.

    Args:
        rows: Output of build_rows().
        epsilon: Fraction of coordinates spent on the pilot.
        rng: Seeded generator for the pilot draw.
        score: Row key the bands are defined over.
        policies: Grid members to choose among.
    Returns:
        (attempt ordering, chosen policy name).
    """
    n_pilot = max(1, math.ceil(epsilon * len(rows)))
    indices = rng.permutation(len(rows))
    pilot = [rows[index] for index in indices[:n_pilot]]
    rest = [rows[index] for index in indices[n_pilot:]]
    chosen = select_band_on(pilot, score=score, policies=policies)
    policy = next(item for item in policies if item["name"] == chosen)
    return pilot + ordering_for(rest, policy, score=score), chosen


def evaluate_pilot_rule(
    rows: list[dict[str, Any]],
    *,
    score: str,
    epsilons: tuple[float, ...],
    budgets: tuple[float, ...],
    attempt_fractions: tuple[float, ...],
    n_seeds: int,
    first_seed: int,
    oracle_policy: str,
) -> dict[str, Any]:
    """Out-of-fold band choice, averaged over seeds, against the oracle.

    Args:
        rows: Output of build_rows().
        score: Row key the bands are defined over.
        epsilons: Pilot fractions to evaluate.
        budgets: Fractions of the arm's Repair-All cost.
        attempt_fractions: Fractions of the arm's coordinate count.
        n_seeds: Draws averaged per pilot fraction.
        first_seed: Lowest seed.
        oracle_policy: Name of the in-sample best, for the agreement rate.
    Returns:
        Per pilot fraction: added passes at both matched calls and matched
        attempts, how often the pilot reproduced the oracle's choice, and the
        distribution of what it chose instead.
    """
    n = len(rows)
    total_calls = sum(row["calls"] for row in rows)
    budget_calls = [int(round(fraction * total_calls)) for fraction in budgets]
    attempt_counts = [int(round(fraction * n)) for fraction in attempt_fractions]
    policies = grid_for_score(score)

    out = {}
    for epsilon in epsilons:
        added: dict[int, list[int]] = {budget: [] for budget in budget_calls}
        attempted: dict[int, list[int]] = {budget: [] for budget in budget_calls}
        by_attempts: dict[int, list[int]] = {count: [] for count in attempt_counts}
        choices: dict[str, int] = {}
        agreed = 0
        for seed in range(first_seed, first_seed + n_seeds):
            rng = np.random.default_rng(seed)
            ordering, chosen = pilot_band_ordering(
                rows, epsilon=epsilon, rng=rng, score=score, policies=policies
            )
            choices[chosen] = choices.get(chosen, 0) + 1
            agreed += int(chosen == oracle_policy)
            points = _curve(ordering)
            for budget in budget_calls:
                gain, attempts = _at_budget(points, budget)
                added[budget].append(gain)
                attempted[budget].append(attempts)
            for count in attempt_counts:
                by_attempts[count].append(_at_attempts(points, count)[0])
        out[f"pilot@{epsilon:.2f}"] = {
            "added_at_budget": [float(np.mean(added[b])) for b in budget_calls],
            "added_at_budget_std": [
                float(np.std(added[b], ddof=1)) for b in budget_calls
            ],
            "attempts_at_budget": [float(np.mean(attempted[b])) for b in budget_calls],
            "added_at_attempts": [
                float(np.mean(by_attempts[c])) for c in attempt_counts
            ],
            "oracle_agreement_rate": agreed / n_seeds,
            "choice_histogram": dict(
                sorted(choices.items(), key=lambda item: -item[1])
            ),
            "stochastic": True,
        }
    return out


def band_resolution_power(
    rows: list[dict[str, Any]],
    *,
    score: str,
    pilot_sizes: tuple[int, ...],
    n_draws: int,
    first_seed: int,
    oracle_policy: str,
) -> dict[str, Any]:
    """How large a pilot must be before it reproduces the full-data band choice.

    The band analogue of `pilot_gate_direction.sign_resolution_power`. A sign is
    one bit; a choice among this grid is closer to four, so the honest
    expectation is that the curve stays far below the sign curve at every pilot
    size this project's cohorts can afford. That comparison is the deliverable.

    Args:
        rows: Output of build_rows().
        score: Row key the bands are defined over.
        pilot_sizes: Pilot sizes in coordinates.
        n_draws: Pilots drawn per size.
        first_seed: Lowest seed.
        oracle_policy: The full-data choice the pilot is scored against.
    Returns:
        Per size: agreement rate and the modal alternative chosen.
    """
    policies = grid_for_score(score)
    out = []
    for size in pilot_sizes:
        if size > len(rows):
            continue
        agreed = 0
        choices: dict[str, int] = {}
        for seed in range(first_seed, first_seed + n_draws):
            rng = np.random.default_rng(seed)
            index = rng.permutation(len(rows))[:size]
            pilot = [rows[i] for i in index]
            chosen = select_band_on(pilot, score=score, policies=policies)
            choices[chosen] = choices.get(chosen, 0) + 1
            agreed += int(chosen == oracle_policy)
        out.append(
            {
                "pilot_size": size,
                "band_agreement_rate": agreed / n_draws,
                "modal_choice": max(choices.items(), key=lambda item: item[1])[0],
            }
        )
    return {
        "score_column": score,
        "oracle_policy": oracle_policy,
        "by_pilot_size": out,
    }


# --------------------------------------------------------------------------
# Escalation banding (PR-E)
# --------------------------------------------------------------------------


def escalation_banding(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    score: str = "p_pass",
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35),
    base_arm: str = "none",
) -> dict[str, Any]:
    """Band the ESCALATION rather than the first repair attempt.

    The context-free `none` arm is both the cheapest and (on HumanEval) the most
    productive first attempt, so the allocation question that actually has money
    in it is not "which coordinate do I repair" but "which of the coordinates
    `none` failed do I pay context for". Week 9's complementarity partition gives
    the ceiling: the union of the arms minus what `none` alone reaches.

    Args:
        rows_by_arm: build_rows() output per arm, all over the same coordinates.
        score: Row key the escalation band is defined over.
        thresholds: Lower cuts to evaluate; each attempts everything at or above.
        base_arm: The arm whose failures are the escalation population.
    Returns:
        The escalation population size, the rescue ceiling, and per threshold the
        rescues captured and the attempt share spent.
    """
    base = {row["task_id"]: row for row in rows_by_arm[base_arm]}
    context_arms = [name for name in rows_by_arm if name != base_arm]
    lookup = {
        name: {row["task_id"]: row for row in rows_by_arm[name]}
        for name in context_arms
    }

    population = [row for row in base.values() if not row["repaired"]]
    rescued = {
        task_id
        for task_id in (row["task_id"] for row in population)
        if any(lookup[name][task_id]["repaired"] for name in context_arms)
    }

    per_threshold = []
    for threshold in thresholds:
        gated = [row for row in population if row[score] >= threshold]
        captured = sum(1 for row in gated if row["task_id"] in rescued)
        per_threshold.append(
            {
                "threshold": threshold,
                "attempted": len(gated),
                "attempt_share": (len(gated) / len(population)) if population else None,
                "rescues_captured": captured,
                "capture_share": (captured / len(rescued)) if rescued else None,
                "escalation_calls": sum(
                    min(lookup[name][row["task_id"]]["calls"] for name in context_arms)
                    for row in gated
                ),
            }
        )
    return {
        "base_arm": base_arm,
        "context_arms": context_arms,
        "score_column": score,
        "escalation_population": len(population),
        "rescue_ceiling": len(rescued),
        "by_threshold": per_threshold,
    }


# --------------------------------------------------------------------------
# Cohort assembly
# --------------------------------------------------------------------------


#: The band the user proposed, and the only one carrying a directed claim. Its
#: two extra tests are run on it alone rather than on the whole grid, because a
#: fifteen-fold Fisher/margin sweep would need its own correction and nothing in
#: the grid but this member was ever proposed as a policy.
USER_BAND_POLICY = "raw[0.40,0.55)"


def _score_user_band(
    rows: list[dict[str, Any]],
    *,
    score: str,
    grid: dict[str, Any],
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Neighbourhood and ordering-margin nulls for the proposed band."""
    policies = {policy["name"]: policy for policy in grid_for_score(score)}
    policy = policies.get(USER_BAND_POLICY)
    if policy is None:
        return {"applicable": False, "reason": f"{USER_BAND_POLICY} not in this grid"}
    crossover = grid["policies"][USER_BAND_POLICY]["prefix_equivalence_attempts"]
    informative = [
        count
        for count in grid["attempt_counts"]
        if crossover is None or count < crossover
    ]
    return {
        "applicable": True,
        "policy": USER_BAND_POLICY,
        "prefix_equivalence_attempts": crossover,
        "informative_attempt_counts": informative,
        "neighbourhood": neighbourhood_test(rows, policy, score=score),
        "ordering_margin": ordering_margin_permutation(
            rows,
            policy,
            score=score,
            attempt_counts=informative,
            n_permutations=n_permutations,
            rng=np.random.default_rng(seed),
        ),
    }


def build_cohort_analysis(
    cohort_name: str,
    spec: dict[str, Any],
    *,
    sample_index: int = 0,
    scores: tuple[str, ...] = ("p_pass", "initial_chars"),
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    attempt_fractions: tuple[float, ...] = DEFAULT_ATTEMPT_FRACTIONS,
    epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
    pilot_sizes: tuple[int, ...] = DEFAULT_PILOT_SIZES,
    stop_thresholds: tuple[float, ...] = DEFAULT_STOP_THRESHOLDS,
    stop_window: int = 10,
    n_seeds: int = 200,
    n_draws: int = 400,
    n_permutations: int = 10_000,
    first_seed: int = 0,
    seed: int = 42,
    feature_set: str = "self_consistency_plus_execution",
    model_type: str = "logistic",
) -> dict[str, Any]:
    """Score the band family on every repair arm of one cohort.

    Both orderings are carried: the shipped `p_pass` gate and the free
    `initial_chars` one Week 12 found outranking it. If the free column's bands
    match or beat the trained column's, that is the seventh free-baseline win in
    this project and the more useful deployment result.

    Args:
        cohort_name: "humaneval" or "mbpp".
        spec: The COHORTS entry.
        sample_index: Candidate index the cohort is built from.
        scores: Row keys to run the whole analysis over.
        budgets: Fractions of each arm's Repair-All cost.
        attempt_fractions: Fractions of each arm's coordinate count.
        epsilons: Pilot fractions.
        pilot_sizes: Pilot sizes for the resolution curve.
        stop_thresholds: Marginal calls-per-success stopping thresholds.
        stop_window: Trailing window for the stopping rule.
        n_seeds: Draws averaged per stochastic policy.
        n_draws: Pilots drawn per resolution-curve size.
        n_permutations: Label permutations for the significance tests.
        first_seed: Lowest seed for the stochastic policies.
        seed: Seed for the permutation generator.
        feature_set: Predictor arm supplying the probabilities.
        model_type: Predictor model supplying the probabilities.
    Returns:
        The cohort block, including the stop-on-pass renormalization guards.
    """
    source = load_json(spec["source"])
    predictor = load_json(spec["predictor"])

    arms: dict[str, Any] = {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    guards: dict[str, Any] = {}
    for arm_name, pool_path in spec["arms"]:
        pool, report = renormalize_stop_on_pass(load_json(pool_path))
        rows = build_rows(
            source,
            predictor,
            pool,
            spec["source"],
            sample_index=sample_index,
            feature_set=feature_set,
            model_type=model_type,
        )
        rows_by_arm[arm_name] = rows
        guards[arm_name] = {
            "pool_file": str(pool_path),
            "stop_on_pass_report": report,
            "total_calls": sum(row["calls"] for row in rows),
            "total_repairable": sum(row["repaired"] for row in rows),
        }

        by_score = {}
        for score in scores:
            grid = evaluate_grid(
                rows,
                score=score,
                budgets=budgets,
                attempt_fractions=attempt_fractions,
                n_seeds=n_seeds,
                first_seed=first_seed,
                stop_window=stop_window,
                stop_thresholds=stop_thresholds,
            )
            oracle = grid["oracle_band"]["policy"]
            by_score[score] = {
                "shape": shape_table(rows, score=score),
                "grid": grid,
                "permutation": permutation_test(
                    rows,
                    score=score,
                    policies=grid_for_score(score),
                    n_permutations=n_permutations,
                    rng=np.random.default_rng(seed),
                ),
                "user_band": _score_user_band(
                    rows,
                    score=score,
                    grid=grid,
                    n_permutations=n_permutations,
                    seed=seed,
                ),
                "pilot": evaluate_pilot_rule(
                    rows,
                    score=score,
                    epsilons=epsilons,
                    budgets=budgets,
                    attempt_fractions=attempt_fractions,
                    n_seeds=n_seeds,
                    first_seed=first_seed,
                    oracle_policy=oracle,
                ),
                "band_resolution": band_resolution_power(
                    rows,
                    score=score,
                    pilot_sizes=pilot_sizes,
                    n_draws=n_draws,
                    first_seed=first_seed,
                    oracle_policy=oracle,
                ),
            }
        arms[arm_name] = {"pool_file": str(pool_path), "by_score": by_score}

    return {
        "analysis": "week13_band_gates",
        "cohort_name": cohort_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "source_results": str(spec["source"]),
            "predictor_results": str(spec["predictor"]),
            "sample_index": sample_index,
            "feature_set": feature_set,
            "model_type": model_type,
            "score_columns": list(scores),
            "budget_fractions": list(budgets),
            "attempt_fractions": list(attempt_fractions),
            "epsilons": list(epsilons),
            "pilot_sizes": list(pilot_sizes),
            "stop_thresholds": list(stop_thresholds),
            "stop_window": stop_window,
            "n_seeds": n_seeds,
            "n_draws": n_draws,
            "n_permutations": n_permutations,
            "first_seed": first_seed,
            "seed": seed,
            "calls_note": "renormalized to stop-on-pass before any comparison",
            "policy_grid": [dict(policy) for policy in POLICY_GRID],
        },
        "guards": guards,
        "arms": arms,
        "escalation_banding": escalation_banding(rows_by_arm),
    }


# --------------------------------------------------------------------------
# Family-wide correction across cells
# --------------------------------------------------------------------------


def family_correction(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """BH-correct the grid-max permutation p-values across every cell.

    `permutation_test` already absorbs the multiplicity of searching the grid
    *within* one (cohort, arm, score) cell. Running that search in twelve cells
    and quoting whichever came out smallest would reintroduce the same error one
    level up, so the cell-level p-values are themselves a family and are
    corrected as one.

    Args:
        analyses: One cohort block per cohort.
    Returns:
        Per cell the raw and BH-adjusted grid-max p-value and the band that
        produced it, sorted by raw p-value.
    """
    cells, pvalues = [], []
    for analysis in analyses:
        for arm, arm_block in analysis["arms"].items():
            for score, block in arm_block["by_score"].items():
                grid_max = block["permutation"]["grid_max"]
                cells.append(
                    {
                        "cell": f"{analysis['cohort_name']}/{arm}/{score}",
                        "best_band": grid_max["best_band"],
                        "observed_hit_rate": grid_max["observed_hit_rate"],
                        "p_grid_max": grid_max["p_grid_max"],
                    }
                )
                pvalues.append(grid_max["p_grid_max"])
    adjusted = benjamini_hochberg(pvalues)
    for cell, value in zip(cells, adjusted):
        cell["p_grid_max_bh"] = value
        cell["significant_bh_0.05"] = value < 0.05
    cells.sort(key=lambda cell: cell["p_grid_max"])
    return {
        "n_cells": len(cells),
        "note": (
            "the within-cell grid search is already absorbed by the permutation "
            "test; this corrects the family of cells on top of it"
        ),
        "cells": cells,
    }


# --------------------------------------------------------------------------
# Pre-registration scoring
# --------------------------------------------------------------------------


def score_pre_registrations(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read already-computed numbers and score each pre-registration.

    Nothing is computed here that the functions above did not already compute --
    this only interprets numbers already in hand, which keeps the scoring
    auditable against the same JSON a reader has.

    Args:
        analyses: One cohort block per cohort.
    Returns:
        The pre-registrations with `observed` and `verdict` filled in.
    """
    by_cohort = {analysis["cohort_name"]: analysis for analysis in analyses}
    scored = [
        {**entry, "cohort_scope": None, "observed": None, "verdict": None}
        for entry in PRE_REGISTRATIONS
    ]

    human = by_cohort.get("humaneval")
    mbpp = by_cohort.get("mbpp")
    context_arms = ("code_only", "error")

    # PR-A: shape on the one arm with a real allocation signal.
    entry = scored[0]
    entry["cohort_scope"] = "humaneval/error"
    if human is None:
        entry["verdict"] = "not scored (needs humaneval)"
    else:
        shape = human["arms"]["error"]["by_score"]["p_pass"]["shape"]
        entry["observed"] = {
            "hit_rates": [row["hit_rate"] for row in shape["bins"]],
            "monotone_non_decreasing": shape["monotone_non_decreasing"],
            "any_interior_bin_beats_top_beyond_noise": shape[
                "any_interior_bin_beats_top_beyond_noise"
            ],
            "calls_per_attempt_spread": shape["calls_per_attempt_spread"],
            "calls_per_success_spread": shape["calls_per_success_spread"],
        }
        entry["verdict"] = (
            "falsified" if shape["any_interior_bin_beats_top_beyond_noise"] else "held"
        )

    # PR-B: the user's raw band, at matched attempts, with a permutation p.
    entry = scored[1]
    entry["cohort_scope"] = "humaneval/{code_only,error}"
    if human is None:
        entry["verdict"] = "not scored (needs humaneval)"
    else:
        observed = {}
        beats_anywhere, significant_everywhere = False, True
        for arm in context_arms:
            block = human["arms"][arm]["by_score"]["p_pass"]
            grid = block["grid"]["policies"]
            band = grid["raw[0.40,0.55)"]["added_at_attempts"]
            reference = grid["inverted"]["added_at_attempts"]
            counts = block["grid"]["attempt_counts"]
            crossover = grid["raw[0.40,0.55)"]["prefix_equivalence_attempts"]
            # Points at or past the crossover compare a policy with itself, so a
            # tie there is arithmetic rather than evidence and must not be
            # allowed to falsify the pre-registration.
            informative = [
                index
                for index, count in enumerate(counts)
                if crossover is None or count < crossover
            ]
            wins = [
                float(band[index]) > float(reference[index]) for index in informative
            ]
            p_single = next(
                row["p_single"]
                for row in block["permutation"]["bands"]
                if row["policy"] == "raw[0.40,0.55)"
            )
            directed = block["user_band"]
            neighbourhood = directed["neighbourhood"]
            margin_p = directed["ordering_margin"]["p_margin"]
            observed[arm] = {
                "attempt_counts": counts,
                "added_at_attempts_band": band,
                "added_at_attempts_inverted": reference,
                "prefix_equivalence_attempts": crossover,
                "informative_attempt_counts": [counts[i] for i in informative],
                "beats_inverted_at_any_informative_point": any(wins),
                "margin_at_informative_points": [
                    float(band[index]) - float(reference[index])
                    for index in informative
                ],
                "band_summary": grid["raw[0.40,0.55)"]["band"],
                "p_single": p_single,
                "p_grid_max": block["permutation"]["grid_max"]["p_grid_max"],
            }
            observed[arm]["neighbourhood"] = neighbourhood
            observed[arm]["ordering_margin"] = directed["ordering_margin"]
            beats_neighbourhood = (
                neighbourhood.get("applicable", False)
                and neighbourhood["band_hit_rate"] > neighbourhood["above_hit_rate"]
                and neighbourhood["p_fisher_two_sided"] < 0.05
            )
            margin_significant = any(value < 0.05 for value in margin_p)
            observed[arm]["beats_upper_neighbourhood"] = beats_neighbourhood
            observed[arm]["ordering_margin_significant"] = margin_significant
            beats_anywhere = beats_anywhere or any(wins)
            significant_everywhere = (
                significant_everywhere
                and p_single < 0.05
                and margin_significant
                and beats_neighbourhood
            )
        entry["observed"] = observed
        entry["verdict"] = (
            "held" if (beats_anywhere and significant_everywhere) else "falsified"
        )

    # PR-C: bottom exclusion.
    entry = scored[2]
    entry["cohort_scope"] = "humaneval/{code_only,error}"
    if human is None:
        entry["verdict"] = "not scored (needs humaneval)"
    else:
        observed, held = {}, True
        for arm in context_arms:
            band = human["arms"][arm]["by_score"]["p_pass"]["grid"]["policies"][
                "thresh>=0.10"
            ]["band"]
            ok = (band["pass_retention"] or 0.0) >= 0.60 and (
                band["call_share"] or 1.0
            ) <= 0.50
            observed[arm] = {
                "pass_retention": band["pass_retention"],
                "call_share": band["call_share"],
                "n": band["n"],
                "meets_bar": ok,
            }
            held = held and ok
        entry["observed"] = observed
        entry["verdict"] = "held" if held else "falsified"

    # PR-D: the out-of-fold band choice, predicted to fail.
    entry = scored[3]
    entry["cohort_scope"] = "humaneval/{code_only,error}"
    if human is None:
        entry["verdict"] = "not scored (needs humaneval)"
    else:
        observed, held = {}, True
        for arm in context_arms:
            block = human["arms"][arm]["by_score"]["p_pass"]
            reference = block["grid"]["policies"]["inverted"]["added_at_budget"]
            # Sub-full budgets only: at 100% every ordering has attempted
            # everything and the comparison is vacuous.
            losses = {}
            for name, values in block["pilot"].items():
                losses[name] = sum(
                    1
                    for pilot_value, inverted_value in zip(
                        values["added_at_budget"][:-1], reference[:-1]
                    )
                    if pilot_value <= inverted_value
                )
            best = min(losses.items(), key=lambda item: item[1])
            observed[arm] = {
                "inverted_added_at_budget": reference,
                "pilot_added_at_budget": {
                    name: values["added_at_budget"]
                    for name, values in block["pilot"].items()
                },
                "oracle_agreement_rate": {
                    name: values["oracle_agreement_rate"]
                    for name, values in block["pilot"].items()
                },
                "best_pilot": best[0],
                "budgets_lost_by_best_pilot": best[1],
            }
            held = held and best[1] < 3
        entry["observed"] = observed
        entry["verdict"] = "held" if held else "falsified"

    # PR-E: escalation banding.
    entry = scored[4]
    entry["cohort_scope"] = "both"
    observed = {}
    held_humaneval = None
    for name, analysis in by_cohort.items():
        block = analysis["escalation_banding"]
        row = next(
            (item for item in block["by_threshold"] if item["threshold"] == 0.20), None
        )
        observed[name] = {
            "escalation_population": block["escalation_population"],
            "rescue_ceiling": block["rescue_ceiling"],
            "at_threshold_0.20": row,
        }
        if name == "humaneval" and row is not None:
            held_humaneval = (row["capture_share"] or 0.0) >= 0.40 and (
                row["attempt_share"] or 1.0
            ) <= 0.35
    entry["observed"] = observed
    entry["verdict"] = (
        "not scored (needs humaneval)"
        if held_humaneval is None
        else ("held" if held_humaneval else "falsified")
    )

    # PR-F: cohort dependence, descriptive.
    entry = scored[5]
    entry["cohort_scope"] = "both"
    if human is None or mbpp is None:
        entry["verdict"] = "not scored (needs both cohorts)"
    else:
        margins = {}
        for name, analysis in (("humaneval", human), ("mbpp", mbpp)):
            block = analysis["arms"]["error"]["by_score"]["p_pass"]["grid"]
            # The smallest budget: above each band's crossover the whole family
            # collapses onto `inverted` and every margin is 0 by construction.
            mid = 0
            oracle = block["oracle_band"]["policy"]
            margins[name] = float(
                block["policies"][oracle]["added_at_budget"][mid]
                - block["policies"]["inverted"]["added_at_budget"][mid]
            )
        entry["observed"] = {
            "budget_fraction": human["arms"]["error"]["by_score"]["p_pass"]["grid"][
                "budget_fractions"
            ][0],
            "oracle_minus_inverted_at_smallest_budget": margins,
            "note": (
                "descriptive only; n=2 cohorts cannot identify the near-miss "
                "mechanism, and the MBPP `error` arm carries 10 positives so its "
                "null excludes only a LARGE effect"
            ),
        }
        # The falsifier only fires if MBPP exceeds HumanEval, and the observed
        # difference is one repair on a statistic whose granularity is one
        # repair. That is the Week 7 P1 situation: not falsified is not the same
        # as confirmed, and reporting it as "held" would overstate a 1-pass gap.
        not_falsified = margins["humaneval"] >= margins["mbpp"]
        entry["observed"]["granularity"] = 1.0
        entry["observed"]["margin_gap"] = margins["humaneval"] - margins["mbpp"]
        entry["verdict"] = (
            "falsified"
            if not not_falsified
            else "not falsified, but UNINFORMATIVE at this granularity "
            "(gap of one repair on a one-repair statistic; do not report as "
            "confirmation of cohort dependence)"
        )

    return scored


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_shape(
    analyses: list[dict[str, Any]], output: Path, *, score_column: str = "p_pass"
) -> None:
    """Hit rate and both cost columns by score bin, one column per cohort."""
    figure, axes = plt.subplots(
        2, len(analyses), figsize=(5.6 * len(analyses), 7.2), squeeze=False
    )
    styles = {"none": ":", "code_only": "--", "error": "-"}
    for column, analysis in enumerate(analyses):
        top, bottom = axes[0][column], axes[1][column]
        for arm, arm_block in analysis["arms"].items():
            shape = arm_block["by_score"][score_column]["shape"]
            bins = [row["bin"] for row in shape["bins"]]
            top.plot(
                bins,
                [row["hit_rate"] for row in shape["bins"]],
                linestyle=styles.get(arm, "-"),
                marker="o",
                markersize=4,
                label=f"`{arm}`",
            )
            bottom.plot(
                bins,
                [row["calls_per_attempt"] for row in shape["bins"]],
                linestyle=styles.get(arm, "-"),
                marker="s",
                markersize=4,
                label=f"`{arm}` calls/attempt",
            )
        top.set_title(f"{analysis['cohort_name']}: repair rate by `{score_column}` bin")
        top.set_ylabel("repair success rate")
        top.set_ylim(0.0, 1.0)
        top.legend(fontsize=7)
        bottom.set_title("cost, decoupled from the hit rate")
        bottom.set_xlabel(f"`{score_column}` bin (equal count, 1 = lowest)")
        bottom.set_ylabel("calls per attempt")
        bottom.legend(fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_band_pilot(
    analyses: list[dict[str, Any]], output: Path, *, score_column: str = "p_pass"
) -> None:
    """Band agreement against pilot size, one line per (cohort, arm)."""
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    styles = {"none": ":", "code_only": "--", "error": "-"}
    for analysis in analyses:
        for arm, arm_block in analysis["arms"].items():
            resolution = arm_block["by_score"][score_column]["band_resolution"]
            sizes = [row["pilot_size"] for row in resolution["by_pilot_size"]]
            rates = [row["band_agreement_rate"] for row in resolution["by_pilot_size"]]
            axis.plot(
                sizes,
                rates,
                linestyle=styles.get(arm, "-"),
                marker="o",
                markersize=3.5,
                label=(
                    f"{analysis['cohort_name']} `{arm}` "
                    f"(full-data pick: {resolution['oracle_policy']})"
                ),
            )
    axis.axhline(1.0 / len(POLICY_GRID), color="grey", linestyle="--", linewidth=1)
    axis.set_xlabel("pilot size (coordinates repaired before committing)")
    axis.set_ylabel("P(pilot reproduces the full-data band choice)")
    axis.set_title(f"Can a band be bought out of fold? (`{score_column}`)")
    axis.set_ylim(0.0, 1.05)
    axis.legend(fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


def _print_summary(analysis: dict[str, Any], *, score_column: str = "p_pass") -> None:
    """Print, per arm, the shape table, the band grid and the pilot rule."""
    print(f"\n=== {analysis['cohort_name']} (ordering by `{score_column}`)")
    for arm, arm_block in analysis["arms"].items():
        block = arm_block["by_score"][score_column]
        grid = block["grid"]
        print(
            f"\n  arm `{arm}`  n={grid['n_coordinates']} "
            f"repairable={grid['total_repairable']} "
            f"budget={grid['total_repair_calls']} calls"
        )
        print("    shape (equal-count bins):")
        for row in block["shape"]["bins"]:
            cps = row["calls_per_success"]
            print(
                f"      bin {row['bin']} [{row['score_min']:.3f},{row['score_max']:.3f}] "
                f"n={row['n']:3d} hit={row['hit_rate']:.3f} "
                f"CI[{row['hit_rate_ci95'][0]:.2f},{row['hit_rate_ci95'][1]:.2f}] "
                f"c/att={row['calls_per_attempt']:.2f} "
                f"c/succ={'n/a' if cps is None else f'{cps:5.1f}'}"
            )
        print(
            f"      monotone={block['shape']['monotone_non_decreasing']} "
            f"c/att spread={block['shape']['calls_per_attempt_spread']:.2f}x "
            f"c/succ spread="
            f"{_format_optional(block['shape']['calls_per_success_spread'])}"
        )
        header = "  ".join(f"{f:>9.0%}" for f in grid["budget_fractions"])
        print(f"\n    {'policy (added/attempts @ calls)':32s} {header}")
        for name, values in grid["policies"].items():
            row = "  ".join(
                f"{v:4.1f}/{a:<4.0f}"
                for v, a in zip(values["added_at_budget"], values["attempts_at_budget"])
            )
            print(f"    {name:32s} {row}")
        header = "  ".join(f"{f:>6.0%}" for f in grid["attempt_fractions"])
        print(f"\n    {'policy (added @ matched attempts)':32s} {header}")
        for name, values in grid["policies"].items():
            row = "  ".join(f"{v:6.1f}" for v in values["added_at_attempts"])
            print(f"    {name:32s} {row}")
        print(
            f"\n    oracle (in-sample, NOT deployable): "
            f"{grid['oracle_band']['policy']} -> {grid['oracle_band']['added']}"
        )
        stopping = grid["policies"]["inverted"]["stopping_rule"]
        print(
            f"\n    marginal-cost stopping on `inverted` "
            f"(window {grid['stop_window']}), full run = "
            f"{grid['total_repairable']} passes / {grid['total_repair_calls']} calls:"
        )
        for name, stop in stopping.items():
            share_passes = stop["added"] / max(grid["total_repairable"], 1)
            share_calls = stop["calls"] / max(grid["total_repair_calls"], 1)
            price = stop["calls"] / max(stop["added"], 1)
            print(
                f"      {name:8s} fired={str(stop['stopped']):5s} "
                f"att={stop['attempts']:3d} calls={stop['calls']:3d} "
                f"added={stop['added']:3d} "
                f"({share_passes:.0%} of passes for {share_calls:.0%} of calls, "
                f"price {price:.2f})"
            )

        permutation = block["permutation"]
        print(
            f"    grid-max permutation p = "
            f"{permutation['grid_max']['p_grid_max']:.4f} "
            f"(best band {permutation['grid_max']['best_band']}, "
            f"hit {permutation['grid_max']['observed_hit_rate']:.3f})"
        )
        for name, values in block["pilot"].items():
            print(
                f"    {name}: added@budget="
                f"{[round(v, 1) for v in values['added_at_budget']]} "
                f"oracle-agreement={values['oracle_agreement_rate']:.3f}"
            )


def _format_optional(value: float | None) -> str:
    """Format a possibly-absent ratio for the console table."""
    return "n/a" if value is None else f"{value:.2f}x"


def _print_escalation(analysis: dict[str, Any]) -> None:
    """Print the escalation-banding table (PR-E)."""
    block = analysis["escalation_banding"]
    print(
        f"\n  escalation banding ({analysis['cohort_name']}): "
        f"`{block['base_arm']}` fails {block['escalation_population']}, "
        f"context rescues {block['rescue_ceiling']}"
    )
    for row in block["by_threshold"]:
        print(
            f"    p>={row['threshold']:.2f}  attempted={row['attempted']:3d} "
            f"({_format_share(row['attempt_share'])})  "
            f"rescues={row['rescues_captured']:2d} "
            f"({_format_share(row['capture_share'])})  "
            f"calls={row['escalation_calls']}"
        )


def _format_share(value: float | None) -> str:
    """Format a possibly-absent share for the console table."""
    return "n/a" if value is None else f"{value:.0%}"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Band gates: is repairability monotone in P(pass) or banded?"
    )
    parser.add_argument(
        "--cohort", choices=["mbpp", "humaneval", "both"], default="both"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--n-draws", type=int, default=400)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--stop-window", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-set", default="self_consistency_plus_execution")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--prefix", default="analysis_week13_band_gates")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    set_all_seeds(args.seed)

    names = ["mbpp", "humaneval"] if args.cohort == "both" else [args.cohort]
    analyses = []
    for name in names:
        LOGGER.info("scoring cohort %s", name)
        analysis = build_cohort_analysis(
            name,
            COHORTS[name],
            sample_index=args.sample_index,
            n_seeds=args.n_seeds,
            n_draws=args.n_draws,
            n_permutations=args.n_permutations,
            stop_window=args.stop_window,
            first_seed=args.first_seed,
            seed=args.seed,
            feature_set=args.feature_set,
            model_type=args.model_type,
        )
        analyses.append(analysis)
        for score_column in ("p_pass", "initial_chars"):
            _print_summary(analysis, score_column=score_column)
        _print_escalation(analysis)

    correction = family_correction(analyses)
    print("\n=== grid-max permutation p, BH-corrected across cells")
    for cell in correction["cells"]:
        marker = " *" if cell["significant_bh_0.05"] else ""
        print(
            f"  {cell['cell']:38s} p={cell['p_grid_max']:.4f} "
            f"BH={cell['p_grid_max_bh']:.4f} "
            f"({cell['best_band']}, hit {cell['observed_hit_rate']:.3f}){marker}"
        )

    pre_registrations = score_pre_registrations(analyses)
    print("\n=== pre-registrations")
    for entry in pre_registrations:
        print(f"  {entry['id']} [{entry['cohort_scope']}]: {entry['verdict']}")

    figures = []
    for score_column in ("p_pass", "initial_chars"):
        shape_figure = args.figure_dir / f"week13_band_gates_{score_column}.png"
        plot_shape(analyses, shape_figure, score_column=score_column)
        figures.append(str(shape_figure))
        pilot_figure = args.figure_dir / f"week13_band_pilot_{score_column}.png"
        plot_band_pilot(analyses, pilot_figure, score_column=score_column)
        figures.append(str(pilot_figure))

    path = save_results(
        {
            "analysis": "week13_band_gates",
            "timestamp": datetime.now().isoformat(),
            "figures": figures,
            "cohorts": analyses,
            "family_correction": correction,
            "pre_registrations": pre_registrations,
        },
        prefix=args.prefix,
    )
    print(f"\nfigures: {figures}")
    print(f"results: {path}")


if __name__ == "__main__":
    main()
