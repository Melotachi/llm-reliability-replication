"""Week 7: pre-register the N-sweep's predicted correlations before generating.

Student A deliverable. The Week 7 report measures a hard attenuation ceiling on
|r| of ~0.83-0.90 at N = 10 samples per problem and reports each metric's
"% of ceiling". That is currently a *caveat*. This script turns it into a
*testable measurement model* by deriving, from the already-stored reliability
estimates alone, what each correlation must become at a larger N — and writing
those numbers down before the larger-N runs exist.

The model (classical test theory, same basis as scripts/estimate_reliability.py):

  1. Invert Spearman-Brown to get the single-sample reliability implied by the
     measured N = 10 reliability:      r_1 = r_N / (N - (N - 1) * r_N)
  2. Project it forward to the target N:
                                       r_M = M * r_1 / (1 + (M - 1) * r_1)
  3. Ceiling at M:                     c_M = sqrt(r_xx(M) * r_yy(M))
  4. Treat "% of ceiling" as the estimate of the *true* (disattenuated)
     correlation, which is by construction independent of N, and predict:
                                       |r|_M = (|r|_10 / c_10) * c_M

Step 4 is the falsifiable part. If the ceiling story is right, observed |r|
rises along the predicted curve as N grows. If observed |r| stays flat while
the predicted ceiling climbs, the model is wrong and the project's correlations
are limited by something other than the sample budget — which would itself be
a reportable finding, and is exactly why the prediction is committed first.

Honest scope: the same caveats as estimate_reliability.py apply. Spearman-Brown
assumes parallel halves, which is solid for `pass_rate` and shakier for the
pairwise diversity statistics; and these reliabilities were measured on the
`full` (prompt-confounded) representation. The prediction is therefore about
the *shape* of the curve, not about three decimal places.

Usage:
    python scripts/preregister_n_sweep.py
    python scripts/preregister_n_sweep.py --target-n 50 100 --out reports/x.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)

#: Every stored run in this project samples N = 10 per problem, so every
#: reliability in results/analysis_week7_reliability_*.json is an r_10.
SOURCE_N = 10

#: The cells the sweep can actually afford to re-generate: both are 60
#: HumanEval problems, the two models the project already uses.
DEFAULT_CELLS = {
    "cg/HE60": "results/analysis_week7_reliability_20260804_232252.json",
    "ds/HE60": "results/analysis_week7_reliability_20260804_231809.json",
}

#: Reported for the metrics whose reliability the Week 7 table already carries.
#: pairwise_distance and parse_failure_rate are excluded: the former is the
#: legacy starter-pack metric, the latter is representation-dependent from
#: Week 7 onward (report Section 2.4) and so is not a stable prediction target.
PREDICTED_METRICS = (
    "lexical_diversity",
    "token_jaccard_diversity",
    "token_cosine_diversity",
    "ast_node_diversity",
)


def single_sample_reliability(reliability: float, n: int = SOURCE_N) -> float:
    """Invert the Spearman-Brown formula to the implied one-sample reliability.

    Args:
        reliability: Reliability measured with `n` samples per problem.
        n: The sample budget that reliability was measured at.
    Returns:
        The implied reliability of a single sample.
    Raises:
        ValueError: If the inversion is undefined or leaves [0, 1] — which
            happens for reliabilities at or above 1.0 and signals that the
            split-half estimate, not this formula, is the thing to distrust.
    """
    denominator = n - (n - 1) * reliability
    if denominator <= 0:
        raise ValueError(
            f"reliability={reliability} at n={n} inverts to a non-positive "
            "denominator; the split-half estimate is degenerate"
        )
    r_1 = reliability / denominator
    if not 0.0 < r_1 < 1.0:
        raise ValueError(f"implied single-sample reliability {r_1} outside (0, 1)")
    return r_1


def project_reliability(r_1: float, n: int) -> float:
    """Spearman-Brown: reliability of an `n`-sample aggregate given r_1."""
    return n * r_1 / (1.0 + (n - 1) * r_1)


def ceiling(r_xx: float, r_yy: float) -> float:
    """Attenuation ceiling on |r| for two variables of the given reliabilities."""
    return math.sqrt(r_xx * r_yy)


def predict_cell(
    reliabilities: dict, ceiling_table: dict, target_ns: list[int]
) -> dict:
    """Build the predicted-|r| table for one cell.

    Args:
        reliabilities: The "reliabilities" block of a reliability analysis JSON.
        ceiling_table: The "ceiling_table" block of the same JSON.
        target_ns: Sample budgets to predict for.
    Returns:
        Dict keyed by metric name, each holding observed values at SOURCE_N and
        the predicted ceiling and |r| at every target N.
    """
    pass_rate_r1 = single_sample_reliability(
        reliabilities["pass_rate"]["reliability_mean"]
    )

    predictions = {}
    for metric in PREDICTED_METRICS:
        if metric not in ceiling_table:
            logger.warning("metric %s absent from ceiling table; skipping", metric)
            continue
        entry = ceiling_table[metric]
        metric_r1 = single_sample_reliability(entry["metric_reliability"])
        # The disattenuated (true) correlation the model holds constant in N.
        true_r = abs(entry["observed_pearson_r"]) / entry["ceiling"]

        by_n = {}
        for n in target_ns:
            c_n = ceiling(
                project_reliability(metric_r1, n), project_reliability(pass_rate_r1, n)
            )
            by_n[n] = {
                "ceiling": c_n,
                "predicted_abs_pearson_r": min(true_r * c_n, 1.0),
            }

        predictions[metric] = {
            "observed_abs_pearson_r_at_10": abs(entry["observed_pearson_r"]),
            "ceiling_at_10": entry["ceiling"],
            "percent_of_ceiling_at_10": entry["percent_of_ceiling"],
            "implied_true_r": true_r,
            "by_target_n": by_n,
        }
    return predictions


def _render_markdown(cells: dict, target_ns: list[int], generated_at: str) -> str:
    """Render the committed pre-registration document."""
    lines = [
        "# Week 7 N-sweep — pre-registration",
        "",
        f"**Generated:** {generated_at} by `scripts/preregister_n_sweep.py`, from the",
        "stored Week 7 reliability estimates only. **Committed before the larger-N runs",
        "were generated** — that is the entire point of the file.",
        "",
        "## What is being predicted",
        "",
        "Every correlation this project reports has both variables estimated from the",
        "same ten samples, so each is attenuated by `sqrt(r_xx * r_yy)`. Week 7 measured",
        "those reliabilities. Projecting them forward with Spearman-Brown gives a ceiling",
        "at any larger N, and holding the disattenuated correlation constant gives a",
        "point prediction for what the larger-N run must observe.",
        "",
        "**Falsification condition.** If observed `|r|` at the target N stays flat while",
        "the predicted ceiling climbs, the measurement model is wrong: the project's",
        "correlations are bounded by something other than the sample budget. That",
        "outcome is reportable and is why this file exists.",
        "",
        "Tolerance stated in advance: the prediction concerns the *shape and direction*",
        "of the curve. Spearman-Brown assumes parallel halves — solid for `pass_rate`,",
        "shakier for pairwise diversity statistics, where halving N takes a problem from",
        "45 pairs to 10. Treat a deviation under ~0.05 as agreement.",
        "",
        "**Additional known difference:** the sweep runs on a Kaggle GPU while every",
        "stored run was generated on CPU, so the N=10 point of the new pool will land",
        "*near* the stored value, not exactly on it. The N=10 column below is the",
        "reference for that check.",
        "",
    ]

    for cell_name, cell in cells.items():
        lines += [
            f"## {cell_name}",
            "",
            f"Source: `{cell['source_results_file']}` "
            f"(model `{cell['model']}`, {cell['n_problems']} problems, N={SOURCE_N}).",
            "",
        ]
        header = (
            "| metric | observed \\|r\\| @10 | ceiling @10 | % of ceiling | implied true r | "
            + " | ".join(f"**pred \\|r\\| @{n}** (ceiling)" for n in target_ns)
            + " |"
        )
        lines.append(header)
        lines.append("|" + "---|" * (5 + len(target_ns)))
        for metric, pred in cell["predictions"].items():
            cols = [
                f"`{metric}`",
                f"{pred['observed_abs_pearson_r_at_10']:.3f}",
                f"{pred['ceiling_at_10']:.3f}",
                f"{pred['percent_of_ceiling_at_10']:.1f}%",
                f"{pred['implied_true_r']:.3f}",
            ]
            for n in target_ns:
                block = pred["by_target_n"][n]
                cols.append(
                    f"**{block['predicted_abs_pearson_r']:.3f}** "
                    f"({block['ceiling']:.3f})"
                )
            lines.append("| " + " | ".join(cols) + " |")
        lines.append("")

    lines += [
        "## How this gets checked",
        "",
        "1. Generate the pools (`notebooks/kaggle_n_sweep.py`, Kaggle GPU).",
        "2. `python scripts/analyze_n_sweep.py --results results/<new-run>.json`, which",
        "   subsamples each pool at every N and plots observed vs. predicted.",
        "3. Report the comparison against this table **unchanged**, whichever way it goes.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target-n",
        type=int,
        nargs="+",
        default=[20, 30, 50, 100],
        help="Sample budgets to predict for (default: 20 30 50 100).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/week7_n_sweep_preregistration.md"),
        help="Markdown pre-registration to write.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/prereg_n_sweep.json"),
        help="Machine-readable copy, consumed by analyze_n_sweep.py.",
    )
    return parser.parse_args()


def main() -> None:
    """Emit the pre-registration document and its machine-readable twin."""
    args = parse_args()
    target_ns = sorted(set(args.target_n))

    cells = {}
    for cell_name, path_str in DEFAULT_CELLS.items():
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"missing reliability analysis: {path}")
        data = json.loads(path.read_text())
        cells[cell_name] = {
            "source_results_file": data["source_results_file"],
            "model": data["source_config"]["model"],
            "n_problems": data["source_config"]["n_problems"],
            "representation": data["config"].get("representation", "full"),
            "predictions": predict_cell(
                data["reliabilities"], data["ceiling_table"], target_ns
            ),
        }
        logger.info(
            "predicted %d metrics for %s",
            len(cells[cell_name]["predictions"]),
            cell_name,
        )

    generated_at = datetime.now().isoformat(timespec="seconds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_render_markdown(cells, target_ns, generated_at))
    logger.info("wrote %s", args.out)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {
                "analysis": "week7_n_sweep_preregistration",
                "generated_at": generated_at,
                "source_n": SOURCE_N,
                "target_n": target_ns,
                "cells": cells,
            },
            indent=2,
        )
    )
    logger.info("wrote %s", args.json_out)


if __name__ == "__main__":
    main()
