"""Tests for the Week 11 BigCodeBench cell: the unittest driver and the gate."""

from __future__ import annotations

import pytest

from scripts.validate_bigcodebench_cell import check_problem, mutate
from src.evaluation import classify_failure, run_code
from src.uncertainty import split_prompt_completion
from src.utils import BIGCODEBENCH_TEST_DRIVER

PROMPT = "def task_func(x):\n"
TEST_BODY = """import unittest


class TestCases(unittest.TestCase):
    def test_doubles(self):
        self.assertEqual(task_func(2), 4)

    def test_zero(self):
        self.assertEqual(task_func(0), 0)
"""
TEST_CODE = TEST_BODY + BIGCODEBENCH_TEST_DRIVER


def test_driver_passes_a_correct_solution():
    passed, error = run_code(PROMPT + "    return x * 2\n", TEST_CODE)
    assert passed, error


def test_driver_fails_a_wrong_solution():
    """The defect this driver exists for: unittest collects, it does not raise."""
    passed, error = run_code(PROMPT + "    return x * 3\n", TEST_CODE)
    assert not passed
    assert classify_failure(error) == "assertion_error"


def test_driver_preserves_the_exception_type_for_the_taxonomy():
    """A raising solution must not be reported as an assertion failure."""
    passed, error = run_code(PROMPT + "    raise TypeError('boom')\n", TEST_CODE)
    assert not passed
    assert classify_failure(error) == "type_error"


def test_driver_reports_a_missing_entry_point():
    passed, error = run_code("def other():\n    return 1\n", TEST_CODE)
    assert not passed
    assert classify_failure(error) in {"name_error", "runtime_error"}


@pytest.mark.parametrize("kind", ["none", "raise"])
def test_mutants_cannot_pass_their_own_tests(kind):
    """Shadowing the entry point is behavior-changing for every problem.

    The first version of this control edited the body (`return X` ->
    `return None or X`), which is a no-op for truthy X and let every mutant
    survive -- a false alarm about the driver.
    """
    mutated = mutate("    return x * 2\n", "task_func", kind)
    passed, _ = run_code(PROMPT + mutated, TEST_CODE)
    assert not passed


def test_mutate_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown mutation kind"):
        mutate("    return x\n", "task_func", "scramble")


def test_check_problem_reports_the_prompt_split_and_the_verdict():
    problem = {
        "task_id": "BigCodeBench/unit",
        "prompt": PROMPT,
        "test_code": TEST_CODE,
        "entry_point": "task_func",
        "canonical_solution": "    return x * 2\n",
    }
    record = check_problem(problem, timeout_s=15.0)
    assert record["canonical_passed"] is True
    assert record["prompt_split_ok"] is True
    assert record["failure_category"] == ""
    assert record["elapsed_s"] >= 0.0


def test_prompt_split_holds_for_the_bigcodebench_assembly():
    """The Week 7 representation apparatus rests on this split holding.

    `check_problem` assembles `prompt + canonical_solution`, so the split can
    only fail if the dataset's prompt field is not a literal prefix of the
    candidate -- which is exactly what the gate reports per problem.
    """
    completion = "    return x * 2\n"
    prompt, body = split_prompt_completion(PROMPT + completion, PROMPT)
    assert prompt == PROMPT
    assert body == completion
    with pytest.raises(ValueError):
        split_prompt_completion(PROMPT + completion, "def other(x):\n")
