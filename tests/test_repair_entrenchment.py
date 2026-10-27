"""Tests for scripts/analyze_repair_entrenchment.py (Week 9, Student B)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_repair_entrenchment",
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "analyze_repair_entrenchment.py",
)
entrench = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(entrench)


def _record(task_id: str, candidates: list[tuple[str, bool]]) -> dict:
    return {
        "task_id": task_id,
        "sample_index": 0,
        "initial_code": "def f():\n    return 0\n",
        "initial_failure_category": "assertion_error",
        "trace": [
            {"iteration": 0, "candidates": []},
            {
                "iteration": 1,
                "candidates": [
                    {"candidate_index": i, "code": code, "passed": passed}
                    for i, (code, passed) in enumerate(candidates)
                ],
            },
        ],
    }


def _stub_signatures(monkeypatch, mapping: dict[str, list[str]]):
    monkeypatch.setattr(
        entrench,
        "behavior_signature",
        lambda code, entry_point, probes, timeout_s=5.0: mapping[code],
    )


def test_shared_probe_support_ignores_positions_where_a_program_never_ran():
    signatures = [
        ["1", "error:unrunnable", "3"],
        ["1", "2", "error:TypeError"],
    ]

    assert entrench.shared_probe_support(signatures) == 2


def test_shared_probe_support_is_zero_without_probes():
    assert entrench.shared_probe_support([]) == 0
    assert entrench.shared_probe_support([[], []]) == 0


def test_pair_features_flags_an_all_crash_pair_as_carrying_no_behavior():
    """Week 7 section (h): a pair that never ran carries no behavioral evidence.

    Under DEFAULT_ERROR_COSTS an identical-error pair scores the constant 0.6
    rather than the legacy corner's 0.0, so the trap is softened but not
    removed: the value is a constant, not a measurement. The support count is
    what tells the two apart.
    """
    monkey = pytest.MonkeyPatch()
    _stub_signatures(
        monkey,
        {"a": ["error:unrunnable"] * 3, "b": ["error:unrunnable"] * 3},
    )
    out = entrench.pair_features(["a", "b"], "f", ["1", "2", "3"])
    monkey.undo()

    assert out["behavior_disagreement"] == pytest.approx(0.6)
    assert out["all_crash"] is True
    assert out["probe_support"] == 0


def test_pair_features_reports_support_and_identity_for_a_running_pair():
    monkey = pytest.MonkeyPatch()
    _stub_signatures(monkey, {"def f():\n    return 1\n": ["1", "2"]})
    out = entrench.pair_features(
        ["def f():\n    return 1\n", "def f():\n    return 1\n"], "f", ["a", "b"]
    )
    monkey.undo()

    assert out["identical"] is True
    assert out["probe_support"] == 2
    assert out["ast_distance"] == 0.0
    assert out["token_jaccard"] == 0.0


def test_build_rows_keeps_one_level1_row_per_coordinate(monkeypatch):
    pool = {
        "records": [
            _record("T/1", [("def f():\n    return 1\n", True)]),
            _record("T/2", [("def f():\n    return 2\n", False)]),
        ]
    }
    problems = {
        "T/1": {"entry_point": "f", "test_code": "assert f() == 1"},
        "T/2": {"entry_point": "f", "test_code": "assert f() == 1"},
    }
    monkeypatch.setattr(entrench, "behavior_signature", lambda *a, **k: ["1"])

    level1, level2 = entrench.build_rows(pool, problems, {"T/1": ["()"], "T/2": ["()"]})

    assert [row["task_id"] for row in level1] == ["T/1", "T/2"]
    assert [row["candidate_passed"] for row in level1] == [True, False]
    # T/2's first candidate failed but no second candidate exists.
    assert level2 == []


def test_build_rows_adds_a_level2_row_only_when_the_first_candidate_failed(monkeypatch):
    pool = {
        "records": [
            _record(
                "T/1",
                [
                    ("def f():\n    return 2\n", False),
                    ("def f():\n    return 1\n", True),
                ],
            ),
            _record(
                "T/2",
                [
                    ("def f():\n    return 1\n", True),
                    ("def f():\n    return 3\n", False),
                ],
            ),
        ]
    }
    problems = {
        "T/1": {"entry_point": "f", "test_code": "assert f() == 1"},
        "T/2": {"entry_point": "f", "test_code": "assert f() == 1"},
    }
    monkeypatch.setattr(entrench, "behavior_signature", lambda *a, **k: ["1"])

    _, level2 = entrench.build_rows(pool, problems, {"T/1": ["()"], "T/2": ["()"]})

    assert [row["task_id"] for row in level2] == ["T/1"]
    assert level2[0]["later_candidate_passed"] is True


def test_build_rows_skips_a_task_missing_from_the_dataset(monkeypatch):
    pool = {"records": [_record("T/404", [("def f():\n    return 1\n", True)])]}
    monkeypatch.setattr(entrench, "behavior_signature", lambda *a, **k: ["1"])

    level1, level2 = entrench.build_rows(pool, {}, {})

    assert level1 == [] and level2 == []


def test_score_features_scores_every_feature_and_counts_positives():
    rows = [
        {"passed": True, "ast_distance": 0.9, "token_jaccard": 0.8},
        {"passed": False, "ast_distance": 0.1, "token_jaccard": 0.2},
        {"passed": True, "ast_distance": 0.7, "token_jaccard": 0.6},
        {"passed": False, "ast_distance": 0.2, "token_jaccard": 0.3},
    ]

    out = entrench.score_features(
        rows,
        "passed",
        ("ast_distance", "token_jaccard"),
        seed=1,
        n_permutations=200,
        n_bootstrap=200,
    )

    assert out["n"] == 4 and out["n_positive"] == 2
    assert out["features"]["ast_distance"]["roc_auc"] == pytest.approx(1.0)
    assert set(out["features"]) == {"ast_distance", "token_jaccard"}


def test_score_features_reports_a_single_class_target_as_undefined():
    rows = [{"passed": False, "ast_distance": 0.5}] * 3

    out = entrench.score_features(
        rows, "passed", ("ast_distance",), seed=1, n_permutations=10, n_bootstrap=10
    )

    assert out["n_positive"] == 0
    assert out["features"]["ast_distance"]["roc_auc"] is None


def test_crash_label_crosstab_shows_that_restricting_drops_only_negatives():
    """A passing candidate must run, so all-crash rows can only be negatives."""
    rows = [
        {"all_crash": True, "passed": False},
        {"all_crash": True, "passed": False},
        {"all_crash": False, "passed": True},
        {"all_crash": False, "passed": False},
    ]

    out = entrench.crash_label_crosstab(rows, "passed")

    assert out["all_crash_positive"] == 0
    assert out["all_crash_negative"] == 2
    assert out["base_rate_all_rows"] == pytest.approx(0.25)
    assert out["base_rate_supported_rows"] == pytest.approx(0.5)


def test_crash_label_crosstab_handles_an_empty_supported_population():
    rows = [{"all_crash": True, "passed": False}]

    out = entrench.crash_label_crosstab(rows, "passed")

    assert out["base_rate_supported_rows"] is None
