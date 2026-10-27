"""Tests for scripts/diagnose_crash_control.py (Week 7 runnability control).

The point of this diagnostic is to separate "the code runs" from "the runs
agree", so the tests pin the exact ambiguity it exists to resolve: a sample
with no co-running probe scores 0.0 on the nocrash distance, which looks
identical to perfect agreement unless the support value is read alongside it.
"""

from __future__ import annotations

import pytest

from scripts.diagnose_crash_control import _auc, nocrash_support, summarize
from src.uncertainty import execution_behavior_self_consistency_nocrash


def test_support_is_one_when_every_pair_has_a_co_running_probe():
    signatures = [["1", "2"], ["1", "3"], ["4", "2"]]
    assert nocrash_support(signatures) == pytest.approx([1.0, 1.0, 1.0])


def test_support_is_zero_for_a_sample_that_never_runs():
    # Sample 2 errors on every probe, so it shares no co-running probe with
    # anyone; samples 0 and 1 still have each other.
    signatures = [["1", "2"], ["1", "3"], ["error:X", "error:Y"]]
    support = nocrash_support(signatures)
    assert support[2] == 0.0
    assert support[0] == pytest.approx(0.5)
    assert support[1] == pytest.approx(0.5)


def test_the_trap_this_diagnostic_exists_for():
    # The all-crashing sample gets the SAME nocrash score (0.0) as a sample
    # that genuinely agreed with everyone — "no evidence" is indistinguishable
    # from "perfect agreement" unless support is consulted.
    signatures = [["1", "1"], ["1", "1"], ["error:X", "error:Y"]]
    scores = execution_behavior_self_consistency_nocrash(signatures)
    support = nocrash_support(signatures)
    assert scores[0] == scores[2] == 0.0
    assert support[0] > 0.0 and support[2] == 0.0


def test_support_with_fewer_than_two_samples():
    assert nocrash_support([["1"]]) == [0.0]
    assert nocrash_support([]) == []


def test_auc_orients_higher_diversity_as_lower_confidence():
    # Passing samples have the lower distance, so the AUC should be 1.0
    # after the inversion the function applies.
    rows = [
        {"passed": 1, "execution_self": 0.1},
        {"passed": 1, "execution_self": 0.2},
        {"passed": 0, "execution_self": 0.8},
        {"passed": 0, "execution_self": 0.9},
    ]
    assert _auc(rows, "execution_self") == pytest.approx(1.0)


def test_auc_is_none_for_a_single_class():
    rows = [{"passed": 1, "execution_self": 0.1}, {"passed": 1, "execution_self": 0.2}]
    assert _auc(rows, "execution_self") is None


def test_summarize_splits_the_supported_subpopulation():
    rows = [
        {
            "passed": 1,
            "crash_rate": 0.0,
            "execution_self": 0.1,
            "nocrash_self": 0.1,
            "nocrash_support": 1.0,
        },
        {
            "passed": 0,
            "crash_rate": 0.5,
            "execution_self": 0.6,
            "nocrash_self": 0.4,
            "nocrash_support": 0.5,
        },
        {
            "passed": 0,
            "crash_rate": 1.0,
            "execution_self": 0.9,
            "nocrash_self": 0.0,
            "nocrash_support": 0.0,
        },
    ]
    summary = summarize(rows)
    assert summary["n_samples"] == 3
    assert summary["n_supported"] == 2
    assert summary["frac_samples_all_crash"] == pytest.approx(1 / 3)
    # Over all three samples the nocrash feature ranks the all-crashing
    # sample as the most confident, which is exactly the artefact.
    assert summary["auc_all_samples"]["nocrash_self"] is not None
    assert summary["auc_supported_only"]["nocrash_self"] == pytest.approx(1.0)
