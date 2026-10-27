"""Tests for src/utils.py. Dataset loaders mock `datasets.load_dataset`
(no network in unit tests, per CLAUDE.md FAQ on mocking slow dependencies).
"""

from __future__ import annotations

import json
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src import utils
from src.utils import (
    _entry_point_from_code,
    get_logger,
    load_dataset_split,
    save_results,
    set_all_seeds,
)


def test_set_all_seeds_makes_all_three_rngs_reproducible():
    set_all_seeds(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_all_seeds(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_get_logger_is_idempotent_and_does_not_propagate():
    logger_a = get_logger("test_utils_logger")
    logger_b = get_logger("test_utils_logger")
    assert logger_a is logger_b
    assert len(logger_a.handlers) == 1  # no duplicate handlers on repeat calls
    assert logger_a.propagate is False


def test_save_results_writes_timestamped_json(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "RESULTS_DIR", tmp_path)
    payload = {"summary": {"mean_pass_at_1": 0.5}}

    out_path = save_results(payload, prefix="unittest_run")

    assert out_path.parent == tmp_path
    assert out_path.name.startswith("unittest_run_")
    assert out_path.suffix == ".json"
    with out_path.open() as f:
        assert json.load(f) == payload


def test_entry_point_from_code_finds_top_level_def():
    code = "import math\ndef solve(x):\n    return x\n"
    assert _entry_point_from_code(code) == "solve"


def test_entry_point_from_code_raises_without_def():
    with pytest.raises(ValueError):
        _entry_point_from_code("x = 1\n")


def test_load_dataset_split_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset_name"):
        load_dataset_split("defects4j")


def test_load_dataset_split_humaneval_appends_check_call():
    fake_rows = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "test": "def check(candidate):\n    assert candidate(1, 1) == 2\n",
            "entry_point": "add",
        }
    ]
    with patch("datasets.load_dataset", return_value={"test": fake_rows}):
        problems = load_dataset_split("humaneval")

    assert len(problems) == 1
    problem = problems[0]
    assert set(problem.keys()) == {"task_id", "prompt", "test_code", "entry_point"}
    assert problem["test_code"].rstrip().endswith("check(add)")


def test_load_dataset_split_mbpp_builds_prompt_and_prepends_setup_code():
    fake_rows = [
        {
            "task_id": 11,
            "text": "Write a function to add two numbers.",
            "code": "def add(a, b):\n    return a + b\n",
            "test_list": ["assert add(1, 1) == 2"],
            "test_setup_code": "import math",
        },
        {
            "task_id": 12,
            "text": "Write a function to double a number.",
            "code": "def double(x):\n    return 2 * x\n",
            "test_list": ["assert double(2) == 4"],
            "test_setup_code": "",
        },
    ]
    with patch("datasets.load_dataset", return_value={"test": fake_rows}):
        problems = load_dataset_split("mbpp")

    with_setup, without_setup = problems
    assert with_setup["task_id"] == "MBPP/11"
    assert with_setup["entry_point"] == "add"
    # NL text plus the reference signature, so the model has an entry-point contract.
    assert with_setup["prompt"].startswith('"""Write a function to add two numbers.')
    assert "def add(a, b):" in with_setup["prompt"]
    # Setup code must execute before the assertions that rely on it.
    assert with_setup["test_code"] == "import math\nassert add(1, 1) == 2"
    assert without_setup["test_code"] == "assert double(2) == 4"


# ---------------------------------------------------------------------------
# Week 6: EvalPlus probe-input parsing (load_evalplus_probe_inputs helper)
# ---------------------------------------------------------------------------


def test_probe_inputs_from_evalplus_test_parses_literal_lists():
    from src.utils import _probe_inputs_from_evalplus_test

    test_src = "def helper():\n    pass\ninputs = [[1, 2], ['a', [3.0]], []]\n"
    assert _probe_inputs_from_evalplus_test(test_src) == [
        "1, 2",
        "'a', [3.0]",
        "",
    ]


def test_probe_inputs_from_evalplus_test_ignores_other_assignments():
    from src.utils import _probe_inputs_from_evalplus_test

    assert _probe_inputs_from_evalplus_test("results = [[1]]\nx = 2\n") == []


def test_probe_inputs_from_evalplus_test_nonliteral_returns_empty():
    from src.utils import _probe_inputs_from_evalplus_test

    # A call inside the list makes it non-literal -> unparseable, skip.
    assert _probe_inputs_from_evalplus_test("inputs = [[foo()]]\n") == []


def test_probe_inputs_from_evalplus_test_unparseable_source():
    from src.utils import _probe_inputs_from_evalplus_test

    assert _probe_inputs_from_evalplus_test("def broken(:\n") == []


# ---------------------------------------------------------------------------
# Week 7: EvalPlus label loaders (load_evalplus_test_code / _canonical_solutions)
# ---------------------------------------------------------------------------


_FAKE_PLUS_ROWS = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def f(x):\n",
        "canonical_solution": "    return x + 1\n",
        "entry_point": "f",
        "test": "def check(candidate):\n    assert candidate(1) == 2\n",
    }
]


def _patch_plus_dataset(monkeypatch):
    """Route datasets.load_dataset to a one-row HumanEval+ stand-in."""
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: {"test": _FAKE_PLUS_ROWS}
    )


def test_load_evalplus_test_code_appends_entry_point_call(monkeypatch):
    from src.utils import load_evalplus_test_code

    _patch_plus_dataset(monkeypatch)
    tests = load_evalplus_test_code("humaneval")
    assert set(tests) == {"HumanEval/0"}
    # Same convention as _load_humaneval_split(), so run_code() consumes it.
    assert tests["HumanEval/0"].endswith("\ncheck(f)\n")
    assert "assert candidate(1) == 2" in tests["HumanEval/0"]


def test_load_evalplus_canonical_solutions_concatenates_prompt_and_body(monkeypatch):
    from src.utils import load_evalplus_canonical_solutions

    _patch_plus_dataset(monkeypatch)
    canon = load_evalplus_canonical_solutions("humaneval")
    assert canon["HumanEval/0"] == "def f(x):\n    return x + 1\n"


def test_evalplus_label_loaders_reject_unsupported_dataset():
    import pytest

    from src.utils import load_evalplus_canonical_solutions, load_evalplus_test_code

    for loader in (load_evalplus_test_code, load_evalplus_canonical_solutions):
        with pytest.raises(ValueError):
            loader("defects4j")


_FAKE_MBPP_PLUS_ROWS = [
    {
        "task_id": 11,
        "code": "def f(x):\n    return x + 1\n",
        "test": "assert f(1) == 2\n",
    }
]


def _patch_mbpp_plus_dataset(monkeypatch):
    """Route datasets.load_dataset to a one-row MBPP+ stand-in."""
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: {"test": _FAKE_MBPP_PLUS_ROWS}
    )


def test_load_evalplus_test_code_keeps_the_mbpp_harness_verbatim(monkeypatch):
    """MBPP+ calls its entry point at module level: nothing may be appended."""
    from src.utils import load_evalplus_test_code

    _patch_mbpp_plus_dataset(monkeypatch)
    tests = load_evalplus_test_code("mbpp")

    assert set(tests) == {"MBPP/11"}
    assert tests["MBPP/11"] == "assert f(1) == 2\n"
    assert "check(" not in tests["MBPP/11"]


def test_load_evalplus_canonical_solutions_uses_the_mbpp_code_column(monkeypatch):
    from src.utils import load_evalplus_canonical_solutions

    _patch_mbpp_plus_dataset(monkeypatch)
    canon = load_evalplus_canonical_solutions("mbpp")

    assert canon["MBPP/11"] == "def f(x):\n    return x + 1\n"
