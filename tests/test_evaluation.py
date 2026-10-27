"""Tests for src/evaluation.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.evaluation import (
    aggregate_results,
    behavior_signature,
    classify_failure,
    evaluate_outputs,
    extract_probe_inputs,
    mcnemar_mid_p,
    pass_at_k,
    run_code,
)


def test_run_code_success():
    success, error = run_code("def f(x):\n    return x + 1\n", "assert f(1) == 2")
    assert success is True
    assert error == ""


def test_run_code_assertion_failure():
    success, error = run_code("def f(x):\n    return x\n", "assert f(1) == 2")
    assert success is False
    assert "AssertionError" in error


def test_run_code_syntax_error_is_caught_not_raised():
    success, error = run_code("def f(x)\n    return x\n", "")
    assert success is False
    assert "SyntaxError" in error


def test_run_code_infinite_loop_hits_timeout():
    success, error = run_code("while True:\n    pass\n", "", timeout_s=0.2)
    assert success is False
    assert "TimeoutError" in error
    assert classify_failure(error) == "timeout"


def test_run_code_timer_is_disarmed_after_timeout():
    # A timed-out call must not leave a pending alarm that kills a later call.
    run_code("while True:\n    pass\n", "", timeout_s=0.2)
    success, error = run_code("x = 1\n", "assert x == 1", timeout_s=5.0)
    assert success is True
    assert error == ""


def test_run_code_timeout_disabled_when_nonpositive():
    success, _ = run_code("x = 1\n", "assert x == 1", timeout_s=0)
    assert success is True


def test_evaluate_outputs_matches_run_code_per_sample():
    outputs = ["def f():\n    return 1\n", "def f():\n    return 2\n"]
    results = evaluate_outputs(outputs, "assert f() == 1")
    assert [success for success, _ in results] == [True, False]


@pytest.mark.parametrize(
    "message,expected",
    [
        ("", ""),
        ("Traceback...\nAssertionError\n", "assertion_error"),
        ("Traceback...\nTypeError: bad type\n", "type_error"),
        ("Traceback...\nNameError: name 'x' is not defined\n", "name_error"),
        ("Traceback...\nSyntaxError: invalid syntax\n", "syntax_error"),
        ("Traceback...\nModuleNotFoundError: No module named 'foo'\n", "import_error"),
        ("Traceback...\nTimeoutError: execution exceeded 5.0s time limit\n", "timeout"),
        ("Traceback...\nZeroDivisionError: division by zero\n", "runtime_error"),
    ],
)
def test_classify_failure(message, expected):
    assert classify_failure(message) == expected


def test_pass_at_k_k_equals_1_is_empirical_pass_rate():
    assert pass_at_k(n=5, c=2, k=1) == pytest.approx(2 / 5)


def test_pass_at_k_worked_example_n5_c2():
    # Worked example used in the Week 1 report: n=5, c=2.
    assert pass_at_k(n=5, c=2, k=1) == pytest.approx(0.4)
    assert pass_at_k(n=5, c=2, k=2) == pytest.approx(0.7)
    assert pass_at_k(n=5, c=2, k=5) == pytest.approx(1.0)


def test_pass_at_k_guaranteed_pass_when_not_enough_failures():
    # n=10, c=8: only 2 failures, can't fill a k=5 subset with all failures.
    assert pass_at_k(n=10, c=8, k=5) == 1.0


def test_pass_at_k_zero_passes_is_zero_for_k_equal_n():
    assert pass_at_k(n=5, c=0, k=5) == pytest.approx(0.0)


def test_pass_at_k_rejects_c_greater_than_n():
    with pytest.raises(AssertionError):
        pass_at_k(n=5, c=6, k=1)


def test_aggregate_results_means_and_pooled_repair_rate():
    per_problem = [
        {
            "pass_at_1": 0.4,
            "pass_at_5": 1.0,
            "initial_pass_rate": 0.4,
            "post_repair_pass_rate": 0.6,
            "n_samples": 5,
            "n_initial_pass": 2,
            "n_post_repair_pass": 3,
        },
        {
            "pass_at_1": 0.0,
            "pass_at_5": 0.0,
            "initial_pass_rate": 0.0,
            "post_repair_pass_rate": 0.2,
            "n_samples": 5,
            "n_initial_pass": 0,
            "n_post_repair_pass": 1,
        },
    ]
    summary = aggregate_results(per_problem)
    assert summary["mean_pass_at_1"] == pytest.approx(0.2)
    assert summary["mean_pass_at_5"] == pytest.approx(0.5)
    assert summary["mean_initial_pass_rate"] == pytest.approx(0.2)
    assert summary["mean_post_repair_pass_rate"] == pytest.approx(0.4)
    # pooled: initially failing = (5-2)+(5-0)=8, newly passing=(3-2)+(1-0)=2 -> 2/8
    assert summary["repair_success_rate"] == pytest.approx(2 / 8)


def test_aggregate_results_handles_missing_repair_data():
    per_problem = [{"pass_at_1": 1.0, "initial_pass_rate": 1.0}]
    summary = aggregate_results(per_problem)
    assert summary["repair_success_rate"] is None
    assert summary["mean_post_repair_pass_rate"] is None


# ---------------------------------------------------------------------------
# mcnemar_mid_p (selection experiment + Week 7 three-way comparison)
# ---------------------------------------------------------------------------


def test_mcnemar_mid_p_no_discordant_pairs_is_one():
    outcomes = [True, False, True, True]
    assert mcnemar_mid_p(outcomes, outcomes) == 1.0


def test_mcnemar_mid_p_worked_example_n01_8_n10_1():
    # n=9 discordant, k=1: mid-p = 2*CDF(1) - pmf(1)
    #   = 2*(1+9)/512 - 9/512 = 11/512
    a = [True] * 8 + [False] * 1 + [True, True]  # 2 concordant-correct pairs
    b = [False] * 8 + [True] * 1 + [True, True]
    assert mcnemar_mid_p(a, b) == pytest.approx(11 / 512)


def test_mcnemar_mid_p_is_symmetric():
    a = [True, True, True, False, False, True, False, True]
    b = [False, True, False, True, False, False, True, True]
    assert mcnemar_mid_p(a, b) == pytest.approx(mcnemar_mid_p(b, a))


def test_mcnemar_mid_p_balanced_discordance_capped_at_one():
    # n01 == n10: maximally null-consistent; mid-p must stay <= 1.0
    a = [True, False] * 5
    b = [False, True] * 5
    assert mcnemar_mid_p(a, b) <= 1.0


def test_mcnemar_mid_p_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_mid_p([True, False], [True])


# ---------------------------------------------------------------------------
# Week 6: extract_probe_inputs (execution-behavior feature, probe side)
# ---------------------------------------------------------------------------


HUMANEVAL_STYLE_TEST = """
def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0], 0.3) == True
    assert candidate([1.0, 2.0, 3.0], 0.5) == False  # duplicate call

check(has_close_elements)
"""


def test_extract_probe_inputs_humaneval_candidate_calls_deduplicated():
    probes = extract_probe_inputs(HUMANEVAL_STYLE_TEST, "has_close_elements")
    assert probes == ["[1.0, 2.0, 3.0], 0.5", "[1.0, 2.8, 3.0], 0.3"]


def test_extract_probe_inputs_mbpp_direct_entry_point_calls():
    test = "assert min_cost(5) == 8\nassert min_cost(2) == 3\n"
    assert extract_probe_inputs(test, "min_cost") == ["5", "2"]


def test_extract_probe_inputs_keyword_arguments_roundtrip():
    test = "def check(candidate):\n    assert candidate(1, flag=True) == 2\n"
    assert extract_probe_inputs(test, "f") == ["1, flag=True"]


def test_extract_probe_inputs_unparseable_test_returns_empty():
    assert extract_probe_inputs("def broken(:\n", "f") == []


def test_extract_probe_inputs_ignores_unrelated_calls():
    test = "assert helper(1) == 2\nassert target(3) == 4\n"
    assert extract_probe_inputs(test, "target") == ["3"]


# ---------------------------------------------------------------------------
# Week 6: behavior_signature (execution-behavior feature, replay side)
# ---------------------------------------------------------------------------


def test_behavior_signature_records_return_values():
    code = "def add(a, b):\n    return a + b\n"
    assert behavior_signature(code, "add", ["1, 2", "3, 4"]) == ["3", "7"]


def test_behavior_signature_records_exception_type_per_probe():
    code = "def div(a, b):\n    return a / b\n"
    assert behavior_signature(code, "div", ["6, 2", "1, 0"]) == [
        "3.0",
        "error:ZeroDivisionError",
    ]


def test_behavior_signature_unrunnable_code_marks_every_probe():
    assert behavior_signature("def broken(:\n", "broken", ["1", "2"]) == [
        "error:unrunnable",
        "error:unrunnable",
    ]


def test_behavior_signature_missing_entry_point_is_unrunnable():
    assert behavior_signature("x = 1\n", "f", ["1"]) == ["error:unrunnable"]


def test_behavior_signature_timeout_marks_every_probe():
    code = "def hang(x):\n    while True:\n        pass\n"
    assert behavior_signature(code, "hang", ["1", "2"], timeout_s=0.5) == [
        "error:timeout",
        "error:timeout",
    ]


def test_behavior_signature_empty_probes_returns_empty():
    assert behavior_signature("def f():\n    return 1\n", "f", []) == []


def test_behavior_signature_identical_behavior_different_source():
    # Two implementations that differ textually but behave identically must
    # produce identical signatures — the property the diversity metric needs.
    a = "def f(xs):\n    return sorted(xs)\n"
    b = "def f(xs):\n    out = list(xs)\n    out.sort()\n    return out\n"
    probes = ["[3, 1, 2]", "[]"]
    assert behavior_signature(a, "f", probes) == behavior_signature(b, "f", probes)


def test_run_code_cannot_write_into_the_callers_directory(tmp_path, monkeypatch):
    """Week 11: BigCodeBench code writes real files; it must not write here.

    The first canonical-gate pass ran in the repository root and left
    df_contents.txt, empty_test.xlsx, data.json, filename and mock_output.json
    behind -- and deleted src/, which had to be restored from git.
    """
    monkeypatch.chdir(tmp_path)
    code = "open('side_effect.txt', 'w').write('x')\n"
    passed, error = run_code(code, "assert True\n", timeout_s=15.0)
    assert passed, error
    assert not (tmp_path / "side_effect.txt").exists()


def test_subprocess_paths_run_matplotlib_headless(monkeypatch):
    """Week 11: a plotting sample must not open a GUI window on the operator.

    271 of BigCodeBench's suites plot. Under an interactive backend plt.show()
    blocks until a human closes the window, so the sample is scored as a
    timeout instead of on its behavior. Forced here even when the caller's own
    environment asks for an interactive backend, since the child inherits it.
    """
    monkeypatch.setenv("MPLBACKEND", "MacOSX")
    code = (
        "import matplotlib\n"
        "import matplotlib.pyplot as plt\n"
        "def f():\n"
        "    plt.plot([1, 2, 3])\n"
        "    plt.show()\n"
        "    return matplotlib.get_backend().lower()\n"
    )
    passed, error = run_code(code, "assert f() == 'agg'\n", timeout_s=10.0)
    assert passed, error
    assert behavior_signature(code, "f", ["", ""], timeout_s=10.0) == ["'agg'"] * 2


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="RLIMIT_AS is not reliably enforced on macOS; the parent-side "
    "capture cap (next test) is what covers the observed OOM path there",
)
def test_runaway_allocation_in_the_child_is_bounded_not_an_OOM(monkeypatch):
    """A child that allocates without bound is killed, not left to exhaust RAM.

    Regression: the subprocess call sites used capture_output=True with no
    ceiling, so a wrong sample that allocates (or prints) without bound grew
    the parent's buffer until the host OOM-ed. The child now runs under
    RLIMIT_AS; the address-space cap is lowered here so the test costs
    milliseconds instead of gigabytes.
    """
    monkeypatch.setattr("src.evaluation._CHILD_ADDRESS_SPACE_BYTES", 256 * 1024**2)
    code = "def f():\n    return [bytearray(1024 * 1024) for _ in range(4096)]\n"
    passed, error = run_code(code, "assert f()\n", timeout_s=10.0)
    assert passed is False
    assert "MemoryError" in error or "code" in error


def test_child_stdout_flood_does_not_reach_the_parent_unbounded(monkeypatch):
    """An infinite print loop is capped on disk and truncated on read."""
    monkeypatch.setattr("src.evaluation._CHILD_FILE_SIZE_BYTES", 4 * 1024**2)
    code = "def f():\n    while True:\n        print('x' * 4096)\n"
    passed, _ = run_code(code, "assert f()\n", timeout_s=10.0)
    assert passed is False


def test_run_code_without_timeout_also_isolates_the_working_directory(
    tmp_path, monkeypatch
):
    """The in-process path (timeout_s <= 0) is isolated by chdir, not cwd=."""
    monkeypatch.chdir(tmp_path)
    code = "open('side_effect.txt', 'w').write('x')\n"
    passed, error = run_code(code, "assert True\n", timeout_s=0)
    assert passed, error
    assert not (tmp_path / "side_effect.txt").exists()
    assert Path.cwd() == tmp_path  # the cwd is restored even on the happy path


def test_run_code_restores_the_working_directory_after_a_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    passed, _ = run_code("raise ValueError('boom')\n", "assert True\n", timeout_s=0)
    assert not passed
    assert Path.cwd() == tmp_path


def test_behavior_signature_cannot_write_into_the_callers_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    code = (
        "def task_func(x):\n"
        "    open('probe_side_effect.txt', 'w').write('x')\n"
        "    return x\n"
    )
    outcomes = behavior_signature(code, "task_func", ["1"], timeout_s=15.0)
    assert outcomes == ["1"]
    assert not (tmp_path / "probe_side_effect.txt").exists()
