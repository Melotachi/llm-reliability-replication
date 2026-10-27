"""Tests for the matched-call regeneration baseline used in the IST manuscript.

The comparison the script makes is only meaningful if three things hold: the
resample control reads the right samples, the pairing is coordinate-by-
coordinate, and a pool whose coordinates do not line up is rejected rather than
silently compared. Each is tested here on synthetic pools.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.compare_regeneration_baseline import (
    MATCHED_K,
    _coordinate_hash,
    analyze_cohort,
    one_sample_t,
    resample_outcomes,
)

SHA = "a" * 64


def _source(n_problems: int, passing_after_first: set[int]) -> dict:
    """Generation run with four samples per problem.

    Sample 0 always fails, so every problem is a coordinate. Problems in
    `passing_after_first` have sample 2 passing, which the k=3 control sees.
    """
    return {
        "config": {"model": "unit/generator"},
        "per_problem": [
            {
                "task_id": f"UNIT/{index}",
                "samples": [
                    {"passed": False},
                    {"passed": False},
                    {"passed": index in passing_after_first},
                    {"passed": False},
                ],
            }
            for index in range(n_problems)
        ],
    }


def _pool(n_problems: int, repaired: set[int], *, sha: str = SHA) -> dict:
    """Repair pool over every problem's sample 0, `repaired` ones succeeding."""
    records = [
        {
            "task_id": f"UNIT/{index}",
            "sample_index": 0,
            "final_passed": index in repaired,
        }
        for index in range(n_problems)
    ]
    successes = len(repaired)
    return {
        "config": {"selection_sha256": sha},
        "summary": {
            "repair_model_calls": 3 * n_problems,
            "model_calls_per_success": (
                3 * n_problems / successes if successes else None
            ),
            "duplicate_candidate_rate": 0.0,
        },
        "records": records,
    }


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_resample_control_reads_samples_one_through_k():
    """Sample 0 is the coordinate itself and must never count as a rescue."""
    source = _source(4, passing_after_first={1, 3})
    pool = _pool(4, repaired=set())
    outcomes = resample_outcomes(source, pool["records"], k=MATCHED_K)
    assert outcomes == [False, True, False, True]


def test_resample_control_respects_k():
    """The rescue at sample index 2 is invisible to a k=1 replay."""
    source = _source(2, passing_after_first={0})
    pool = _pool(2, repaired=set())
    assert resample_outcomes(source, pool["records"], k=1) == [False, False]
    assert resample_outcomes(source, pool["records"], k=3) == [True, False]


def test_coordinate_hash_separates_different_coordinate_lists():
    """Two pools over different coordinates must not hash alike."""
    assert _coordinate_hash(_pool(3, set())["records"]) != _coordinate_hash(
        _pool(4, set())["records"]
    )


def test_analyze_cohort_pairs_outcomes_and_counts_discordances(tmp_path):
    """The 2x2 table is built per coordinate, not from the two totals."""
    source = _source(4, passing_after_first={0, 1})
    spec = {
        "source": _write(tmp_path, "source.json", source),
        "expected_sha256": SHA,
        "arms": {"only": _write(tmp_path, "arm.json", _pool(4, repaired={1, 2}))},
        "random_replays": {
            "added_mean": 2.0,
            "added_sd": 1.0,
            "n_replays": 10,
            "smart_added": 1,
        },
    }
    result = analyze_cohort("unit", spec)
    control = result["resample_control"]
    assert control["added_passes"] == 2
    assert control["base_model_calls"] == MATCHED_K * 4
    paired = result["arms"]["only"]["paired_vs_resample"]
    # coordinate 1 passes under both, 2 only under repair, 0 only under
    # resampling, 3 under neither.
    assert (paired["both"], paired["repair_only"]) == (1, 1)
    assert (paired["resample_only"], paired["neither"]) == (1, 1)
    assert paired["mcnemar_mid_p"] == pytest.approx(1.0)


def test_analyze_cohort_rejects_a_pool_from_another_cohort(tmp_path):
    """A mismatched selection hash is an error, not a silent comparison."""
    spec = {
        "source": _write(tmp_path, "source.json", _source(3, set())),
        "expected_sha256": SHA,
        "arms": {
            "only": _write(tmp_path, "arm.json", _pool(3, repaired=set(), sha="b" * 64))
        },
        "random_replays": {
            "added_mean": 1.0,
            "added_sd": 1.0,
            "n_replays": 10,
            "smart_added": 1,
        },
    }
    with pytest.raises(ValueError, match="selection_sha256"):
        analyze_cohort("unit", spec)


def test_analyze_cohort_rejects_misaligned_arms(tmp_path):
    """Two arms that do not cover the same coordinates cannot be paired."""
    spec = {
        "source": _write(tmp_path, "source.json", _source(4, set())),
        "expected_sha256": SHA,
        "arms": {
            "a": _write(tmp_path, "a.json", _pool(4, repaired=set())),
            "b": _write(tmp_path, "b.json", _pool(3, repaired=set())),
        },
        "random_replays": {
            "added_mean": 1.0,
            "added_sd": 1.0,
            "n_replays": 10,
            "smart_added": 1,
        },
    }
    with pytest.raises(ValueError, match="coordinate list differs"):
        analyze_cohort("unit", spec)


def test_one_sample_t_matches_the_reported_replay_statistics():
    """The two manuscript t-tests, to the precision the paper prints."""
    humaneval = one_sample_t(40.5, 2.7, 200, 36)
    mbpp = one_sample_t(7.56, 1.19, 200, 7)
    assert humaneval["t"] == pytest.approx(23.57, abs=0.005)
    assert humaneval["df"] == 199
    assert humaneval["p_two_sided"] < 1e-50
    assert mbpp["t"] == pytest.approx(6.66, abs=0.005)
    assert mbpp["p_two_sided"] == pytest.approx(2.69e-10, rel=0.01)


def test_one_sample_t_is_two_sided():
    """A mean below the reference gives the same p as one equally far above."""
    below = one_sample_t(5.0, 1.0, 50, 6.0)
    above = one_sample_t(7.0, 1.0, 50, 6.0)
    assert below["t"] == pytest.approx(-above["t"])
    assert below["p_two_sided"] == pytest.approx(above["p_two_sided"])
    assert not math.isnan(below["p_two_sided"])
