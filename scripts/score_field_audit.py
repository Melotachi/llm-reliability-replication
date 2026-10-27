"""Score the Week 11 field audit against its frozen pre-registration.

Reads the measurement JSON written by ``scripts/audit_published_metrics.py``
and applies the criteria fixed in
``reports/week11_field_audit_preregistration.md`` §6-§7. It measures
nothing itself, so it can be re-run freely without touching the evidence.

The separation is deliberate: the measuring script must not know what
counts as a hit, and the scoring script must not be able to change a
number. Anything this script reports that the pre-registration did not
ask for is printed under an explicit POST-HOC heading.

Usage::

    python scripts/score_field_audit.py                     # latest audit JSON
    python scripts/score_field_audit.py --audit results/analysis_week11_...json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this figure is written, never displayed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, save_results  # noqa: E402

logger = get_logger(__name__)

#: Pre-registration §6: the project's own Week 7 P3 tolerance, reused
#: rather than freshly chosen.
DELTA_TOLERANCE = 0.12

#: Pre-registration §6 criterion (ii) uses this alpha for the FDR family.
ALPHA = 0.05

#: The anchor configuration of each audited definition. arXiv 2508.14288
#: proposes sigma_struct and sigma_value as co-equal variants, so both are
#: anchors of their definition; d=2 and eps=1e-6 are the pre-registered
#: anchors (§5) -- the paper supplies no value for either.
ANCHORS: dict[str, list[str]] = {
    "tsed": ["tsed_paper"],
    "sent_jsd": ["sent_jsd_d2_struct_eps6", "sent_jsd_d2_value_eps6"],
    "sent_sce": ["sent_sce_d2_struct_eps6", "sent_sce_d2_value_eps6"],
}

#: Stored key of this project's own tree edit distance, for H3.
PROJECT_TED = "tree_edit_distance_diversity"


# Figure palette. The chart's job is CHANGE WITH POLARITY (a coefficient moves
# from `full` to `corrected`, and the direction is the point), so this is a
# diverging assignment: two opposed hues plus a neutral for "no change".
# Validated rather than eyeballed -- blue/red clear every check on the light
# surface (CVD delta-E 21.6 protan, normal-vision 32.3, both >= 3:1 contrast).
# Direction is never carried by hue alone: the marker fill distinguishes the
# two conditions, the segment's direction along the axis shows the sign, and
# every row that trips the pre-registered tolerance is labelled outright.
COLOR_STRONGER = "#e34948"
COLOR_WEAKER = "#2a78d6"
COLOR_FLAT = "#52514e"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

SHORT_LABEL = {
    "tsed_paper": "TSED",
    "sent_jsd_d2_struct_eps6": "JSD $\\sigma_{struct}$",
    "sent_jsd_d2_value_eps6": "JSD $\\sigma_{value}$",
    "sent_sce_d2_struct_eps6": "SCE $\\sigma_{struct}$",
    "sent_sce_d2_value_eps6": "SCE $\\sigma_{value}$",
}


def plot_c1(c1: dict, out_path: Path) -> None:
    """Dumbbell chart: |rho| under `full` -> `corrected`, one row per anchor.

    One axis by construction (|Spearman rho| against pass rate); the pair of
    markers on each row is the same quantity measured two ways, not two
    scales. Rows are grouped by cell so a reader can see whether a definition
    moves everywhere or only in one regime -- which is the difference between
    a property of the metric and a property of a corpus.
    """
    rows = []
    for cell, entries in c1.items():
        for name in SHORT_LABEL:
            row = entries.get(name)
            if row:
                rows.append((cell, name, row))
    if not rows:
        logger.warning("Nothing to plot")
        return

    height = max(4.0, 0.30 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(9.0, height))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    labels, positions = [], []
    y = 0.0
    last_cell = None
    for cell, name, row in rows:
        if last_cell is not None and cell != last_cell:
            y += 0.9  # a gap between cells, not a rule -- the grid stays quiet
        last_cell = cell
        before, after = abs(row["rho_full"]), abs(row["rho_corrected"])
        color = {
            "stronger": COLOR_STRONGER,
            "weaker": COLOR_WEAKER,
            "flat": COLOR_FLAT,
        }[row["direction"]]
        ax.plot([before, after], [y, y], color=color, linewidth=2.0, zorder=2)
        ax.plot(
            before,
            y,
            "o",
            markersize=8,
            markerfacecolor=SURFACE,
            markeredgecolor=color,
            markeredgewidth=2.0,
            zorder=3,
        )
        ax.plot(after, y, "o", markersize=8, color=color, zorder=3)
        if row["conclusion_change"]:
            ax.annotate(
                f"{after - before:+.2f}",
                (max(before, after), y),
                textcoords="offset points",
                xytext=(9, -3),
                fontsize=7.5,
                color=INK_PRIMARY,
            )
        labels.append(f"{cell}  {SHORT_LABEL[name]}")
        positions.append(y)
        y += 1.0

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.5, color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.set_xlabel("|Spearman $\\rho$| against pass rate", fontsize=9, color=INK_PRIMARY)
    ax.set_xlim(left=0.0)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=8)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D(
            [],
            [],
            color=COLOR_STRONGER,
            linewidth=2.0,
            marker="o",
            markersize=8,
            label="stronger when corrected",
        ),
        plt.Line2D(
            [],
            [],
            color=COLOR_WEAKER,
            linewidth=2.0,
            marker="o",
            markersize=8,
            label="weaker when corrected",
        ),
        plt.Line2D(
            [],
            [],
            color=COLOR_FLAT,
            linewidth=2.0,
            marker="o",
            markersize=8,
            label="unchanged",
        ),
        plt.Line2D(
            [],
            [],
            color=INK_SECONDARY,
            linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor=SURFACE,
            markeredgecolor=INK_SECONDARY,
            markeredgewidth=2.0,
            label="hollow = full (prompt included)",
        ),
        plt.Line2D(
            [],
            [],
            color=INK_SECONDARY,
            linestyle="none",
            marker="o",
            markersize=8,
            label="solid = corrected (docstring stripped)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=7.5,
        frameon=False,
        labelcolor=INK_SECONDARY,
    )
    ax.set_title(
        "Published metric definitions under the prompt-representation control",
        fontsize=10.5,
        color=INK_PRIMARY,
        loc="left",
        pad=12,
    )
    fig.text(
        0.01,
        0.005,
        f"Labelled rows cross the pre-registered tolerance |delta rho| > {DELTA_TOLERANCE} "
        "or flip FDR significance.",
        fontsize=7,
        color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Audit JSON to score (default: newest analysis_week11_field_audit_*).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Results directory."
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/week11_field_audit_c1.png"),
        help="Where to write the C1 dumbbell chart.",
    )
    return parser.parse_args()


def _latest_audit(results_dir: Path) -> Path:
    """Newest audit JSON, so the common case needs no argument."""
    candidates = sorted(results_dir.glob("analysis_week11_field_audit_*.json"))
    if not candidates:
        raise SystemExit("No analysis_week11_field_audit_*.json found in results/")
    return candidates[-1]


def _anchor_family_fdr(cell_entry: dict, representation: str) -> dict[str, float]:
    """BH-FDR over the ANCHOR configurations only, not the whole sweep.

    The pre-registration says "BH-FDR at 0.05 within cell" without naming
    the family, so it is named here and the choice is declared: the family
    is the set of *primary claims* -- one entry per anchor configuration --
    because the ~50 grid configurations are sensitivity analyses of the same
    few hypotheses, not fifty independent tests. Correcting across the full
    grid would be conservative in a way that manufactures significance
    flips, which is precisely the criterion being scored.
    """
    correlations = cell_entry["representations"][representation]["correlations"]
    names = [name for anchors in ANCHORS.values() for name in anchors]
    present = [
        n for n in names if correlations.get(n, {}).get("spearman_p") is not None
    ]
    if not present:
        return {}
    raw = [correlations[n]["spearman_p"] for n in present]
    adjusted = scipy_stats.false_discovery_control(raw, method="bh")
    return dict(zip(present, (float(a) for a in adjusted)))


def score_representation_control(audit: dict) -> dict:
    """Control C1: does `full` -> `corrected` change a definition's conclusion?"""
    findings: dict = {}
    for cell, entry in audit["cells"].items():
        reps = entry["representations"]
        if "full" not in reps or "corrected" not in reps:
            continue
        fdr_full = _anchor_family_fdr(entry, "full")
        fdr_corr = _anchor_family_fdr(entry, "corrected")
        cell_rows = {}
        for definition, anchors in ANCHORS.items():
            for name in anchors:
                before = reps["full"]["correlations"].get(name, {})
                after = reps["corrected"]["correlations"].get(name, {})
                if (
                    before.get("spearman_rho") is None
                    or after.get("spearman_rho") is None
                ):
                    continue
                rho_before = before["spearman_rho"]
                rho_after = after["spearman_rho"]
                sig_before = fdr_full.get(name, 1.0) < ALPHA
                sig_after = fdr_corr.get(name, 1.0) < ALPHA
                delta = abs(rho_after) - abs(rho_before)
                cell_rows[name] = {
                    "definition": definition,
                    "rho_full": rho_before,
                    "rho_corrected": rho_after,
                    "abs_delta": abs(rho_after - rho_before),
                    "signed_strength_delta": delta,
                    "fdr_p_full": fdr_full.get(name),
                    "fdr_p_corrected": fdr_corr.get(name),
                    "significance_flip": sig_before != sig_after,
                    "exceeds_tolerance": abs(rho_after - rho_before) > DELTA_TOLERANCE,
                    "direction": (
                        "stronger" if delta > 0 else "weaker" if delta < 0 else "flat"
                    ),
                }
                cell_rows[name]["conclusion_change"] = (
                    cell_rows[name]["significance_flip"]
                    or cell_rows[name]["exceeds_tolerance"]
                )
        findings[cell] = cell_rows
    return findings


def score_length_control(audit: dict) -> dict:
    """Control C2: does partialling completion length change the conclusion?"""
    findings: dict = {}
    for cell, entry in audit["cells"].items():
        rows = {}
        for representation, block in entry["representations"].items():
            for definition, anchors in ANCHORS.items():
                for name in anchors:
                    stats = block["correlations"].get(name, {})
                    raw = stats.get("spearman_rho")
                    partial = stats.get("partial_rho_length")
                    if raw is None or partial is None:
                        continue
                    rows[f"{name}@{representation}"] = {
                        "definition": definition,
                        "representation": representation,
                        "raw_rho": raw,
                        "partial_rho": partial,
                        "abs_delta": abs(abs(partial) - abs(raw)),
                        "exceeds_tolerance": abs(abs(partial) - abs(raw))
                        > DELTA_TOLERANCE,
                        "retained_fraction": (
                            abs(partial) / abs(raw) if abs(raw) > 0 else None
                        ),
                    }
        findings[cell] = rows
    return findings


def score_h1(c1: dict) -> dict:
    """H1: sigma_value moves in >=4/7 cells, sigma_struct does not in >=5/7."""
    value_hits = sum(
        1
        for rows in c1.values()
        if rows.get("sent_jsd_d2_value_eps6", {}).get("abs_delta", 0.0)
        > DELTA_TOLERANCE
    )
    struct_stable = sum(
        1
        for rows in c1.values()
        if rows.get("sent_jsd_d2_struct_eps6", {}).get("abs_delta", 1.0)
        <= DELTA_TOLERANCE
    )
    n_cells = len(c1)
    return {
        "hypothesis": "sigma_value |delta rho| > 0.12 in >=4/7; sigma_struct <= 0.12 in >=5/7",
        "value_cells_moving": value_hits,
        "struct_cells_stable": struct_stable,
        "n_cells": n_cells,
        "verdict": (
            "CONFIRMED" if value_hits >= 4 and struct_stable >= 5 else "FALSIFIED"
        ),
    }


def score_h2(audit: dict) -> dict:
    """H2: the unspecified depth bound d moves |rho| by > 0.12 somewhere."""
    worst = {"range": 0.0}
    per_cell = {}
    for cell, entry in audit["cells"].items():
        for representation, block in entry["representations"].items():
            correlations = block["correlations"]
            for metric in ("sent_jsd", "sent_sce"):
                for encoding in ("struct", "value"):
                    values = []
                    for depth in (1, 2, 3, 4):
                        stats = correlations.get(
                            f"{metric}_d{depth}_{encoding}_eps6", {}
                        )
                        if stats.get("spearman_rho") is not None:
                            values.append(abs(stats["spearman_rho"]))
                    if len(values) < 2:
                        continue
                    spread = max(values) - min(values)
                    key = f"{cell}/{representation}/{metric}_{encoding}"
                    per_cell[key] = spread
                    if spread > worst["range"]:
                        worst = {"range": spread, "where": key}
    return {
        "hypothesis": "max_d |rho| - min_d |rho| > 0.12 in at least one cell",
        "largest_depth_spread": worst.get("range"),
        "where": worst.get("where"),
        "all_spreads": per_cell,
        "verdict": (
            "CONFIRMED" if worst.get("range", 0.0) > DELTA_TOLERANCE else "FALSIFIED"
        ),
    }


def score_h3(audit: dict, results_dir: Path) -> dict:
    """H3: the normalizer objection does not matter empirically."""
    rows = {}
    for cell, entry in audit["cells"].items():
        block = entry["representations"].get("full")
        if not block or "tsed_unit" not in block["correlations"]:
            continue
        run = json.loads((results_dir / entry["run_file"]).read_text())
        stored = {
            p["task_id"]: p["uncertainty"].get(PROJECT_TED)
            for p in run["per_problem"]
            if p.get("samples")
        }
        per_problem = block["per_problem"]
        ours = [row["tsed_unit"] for row in per_problem]
        theirs = [stored.get(row["task_id"]) for row in per_problem]
        pairs = [(a, b) for a, b in zip(ours, theirs) if b is not None]
        if len(pairs) < 3:
            continue
        rho_between = scipy_stats.spearmanr(
            [a for a, _ in pairs], [b for _, b in pairs]
        ).statistic
        # A problem with no stored TED must be dropped, never coerced to 0.0 --
        # substituting a value would move the very coefficient being compared.
        aligned = [
            (stored[row["task_id"]], row["pass_rate"])
            for row in per_problem
            if stored.get(row["task_id"]) is not None
        ]
        if len(aligned) < 3:
            continue
        rho_tsed = abs(block["correlations"]["tsed_unit"]["spearman_rho"])
        rho_project = abs(
            scipy_stats.spearmanr(
                [v for v, _ in aligned], [p for _, p in aligned]
            ).statistic
        )
        rows[cell] = {
            "spearman_between_normalizers": float(rho_between),
            "abs_rho_tsed_maxnode": rho_tsed,
            "abs_rho_project_sumnode": float(rho_project),
            "abs_delta": abs(rho_tsed - rho_project),
            "within_tolerance": abs(rho_tsed - rho_project) <= DELTA_TOLERANCE,
            "gate_band_pass": 0.85 <= rho_between <= 0.99,
        }
    all_within = all(row["within_tolerance"] for row in rows.values()) if rows else None
    return {
        "hypothesis": "max-node vs sum-node normalizer: |delta rho| <= 0.12 in EVERY cell",
        "per_cell": rows,
        "verdict": (
            "NOT SCORED (no TSED in audit)"
            if not rows
            else "CONFIRMED" if all_within else "FALSIFIED"
        ),
    }


def score_h4(audit: dict) -> dict:
    """H4: the Java-tuned insert weight transfers to Python (<= 0.05 everywhere)."""
    rows = {}
    for cell, entry in audit["cells"].items():
        for representation, block in entry["representations"].items():
            paper = block["correlations"].get("tsed_paper", {}).get("spearman_rho")
            unit = block["correlations"].get("tsed_unit", {}).get("spearman_rho")
            if paper is None or unit is None:
                continue
            rows[f"{cell}/{representation}"] = {
                "rho_paper_weights": paper,
                "rho_unit_weights": unit,
                "abs_delta": abs(abs(paper) - abs(unit)),
                "within_tolerance": abs(abs(paper) - abs(unit)) <= 0.05,
            }
    all_within = all(row["within_tolerance"] for row in rows.values()) if rows else None
    return {
        "hypothesis": "(1.0, 0.8, 1.0) vs (1.0, 1.0, 1.0): |delta rho| <= 0.05 in every cell",
        "per_cell": rows,
        "verdict": (
            "NOT SCORED (no TSED in audit)"
            if not rows
            else "CONFIRMED" if all_within else "FALSIFIED"
        ),
    }


def score_h5(audit: dict) -> dict:
    """H5: the correctness validation the ASE 2025 paper never runs succeeds."""
    rows = {}
    hits = 0
    for cell, entry in audit["cells"].items():
        fdr = _anchor_family_fdr(entry, "full")
        correlations = entry["representations"]["full"]["correlations"]
        stats = correlations.get("sent_jsd_d2_struct_eps6", {})
        rho = stats.get("spearman_rho")
        if rho is None:
            continue
        significant = fdr.get("sent_jsd_d2_struct_eps6", 1.0) < ALPHA
        negative_and_significant = bool(rho < 0 and significant)
        hits += int(negative_and_significant)
        # The verdict is scored on sigma_struct (declared), but the paper
        # proposes both encodings, so sigma_value is reported beside it
        # rather than left out of the record.
        value_stats = correlations.get("sent_jsd_d2_value_eps6", {})
        rows[cell] = {
            "spearman_rho": rho,
            "fdr_p": fdr.get("sent_jsd_d2_struct_eps6"),
            "negative_and_fdr_significant": negative_and_significant,
            "value_encoding_rho": value_stats.get("spearman_rho"),
            "value_encoding_fdr_p": fdr.get("sent_jsd_d2_value_eps6"),
        }
    return {
        "hypothesis": "sent_jsd (anchor) negative and FDR-significant in >=5/7 cells",
        "per_cell": rows,
        "cells_passing": hits,
        "n_cells": len(rows),
        "verdict": "CONFIRMED" if hits >= 5 else "FALSIFIED",
    }


def summarize_k(c1: dict) -> dict:
    """K = definitions with >=1 conclusion change, broken down by direction."""
    changed: dict[str, dict] = {}
    for cell, rows in c1.items():
        for name, row in rows.items():
            if not row["conclusion_change"]:
                continue
            entry = changed.setdefault(
                row["definition"],
                {"cells": [], "stronger": 0, "weaker": 0, "flat": 0},
            )
            entry["cells"].append(f"{cell}:{name}")
            entry[row["direction"]] += 1
    return {
        "K": len(changed),
        "of": len(ANCHORS),
        "per_definition": changed,
        "note": (
            "Direction is reported because a published metric that gets STRONGER "
            "under correction says the field is underselling its own metrics, "
            "which is a different and better claim than 'your numbers are inflated'."
        ),
    }


def main() -> None:
    """Apply the frozen criteria and print the scorecard."""
    args = parse_args()
    audit_path = args.audit or _latest_audit(args.results_dir)
    audit = json.loads(audit_path.read_text())
    logger.info("Scoring %s", audit_path.name)

    c1 = score_representation_control(audit)
    c2 = score_length_control(audit)
    scorecard = {
        "audit_file": audit_path.name,
        "preregistration": "reports/week11_field_audit_preregistration.md",
        "tolerance": DELTA_TOLERANCE,
        "alpha": ALPHA,
        "K": summarize_k(c1),
        "control_c1_representation": c1,
        "control_c2_length": c2,
        "H1": score_h1(c1),
        "H2": score_h2(audit),
        "H3": score_h3(audit, args.results_dir),
        "H4": score_h4(audit),
        "H5": score_h5(audit),
    }

    print("\n=== Week 11 field audit scorecard ===")
    k = scorecard["K"]
    print(f"\nK = {k['K']} of {k['of']} definitions change conclusion under C1")
    for definition, entry in k["per_definition"].items():
        directions = ", ".join(
            f"{v} {d}" for d, v in entry.items() if d in ("stronger", "weaker", "flat")
        )
        print(f"  {definition:10s} {len(entry['cells'])} cell-configs  ({directions})")

    print("\nC1 -- representation (full -> corrected), Spearman rho vs pass rate")
    anchor_names = [n for anchors in ANCHORS.values() for n in anchors]
    for cell, rows in c1.items():
        print(f"\n  {cell}")
        for name in anchor_names:
            row = rows.get(name)
            if not row:
                continue
            flag = "CHANGE" if row["conclusion_change"] else "      "
            print(
                f"    {name:28s} {row['rho_full']:+.3f} -> {row['rho_corrected']:+.3f}"
                f"  |d|={row['abs_delta']:.3f}  {row['direction']:8s} {flag}"
            )

    for key in ("H1", "H2", "H3", "H4", "H5"):
        entry = scorecard[key]
        print(f"\n{key}: {entry['verdict']}")
        print(f"    {entry['hypothesis']}")

    plot_c1(c1, args.figure)

    path = save_results(scorecard, prefix="analysis_week11_field_audit_scorecard")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
