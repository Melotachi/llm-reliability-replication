"""Shared utilities: seeding, logging, result persistence, dataset loading.

Jointly owned by Student A and Student B (see CLAUDE.md Section 6.5).
"""

from __future__ import annotations

import ast
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_HUMANEVAL_HF_NAME = "openai_humaneval"
_HUMANEVALPLUS_HF_NAME = "evalplus/humanevalplus"
_MBPP_HF_NAME = "mbpp"
_BIGCODEBENCH_HF_NAME = "bigcode/bigcodebench"
_BIGCODEBENCH_SPLIT = "v0.1.4"

#: Appended to every BigCodeBench `test` field. The dataset ships a
#: `unittest.TestCase` subclass and no runner, and unittest *collects*
#: failures into a result object instead of raising -- so without an explicit
#: `wasSuccessful()` check every candidate would score as passing. Errors are
#: re-raised with their original type so `classify_failure` keeps working
#: (a bare `assert result.wasSuccessful()` would collapse the whole Section 9
#: taxonomy into `assertion_error`).
BIGCODEBENCH_TEST_DRIVER = '''

import io as _bcb_io
import unittest as _bcb_unittest


class _BcbResult(_bcb_unittest.TextTestResult):
    """Keeps the first exception object, which the default result discards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_exception = None

    def addError(self, test, err):
        super().addError(test, err)
        if self.first_exception is None:
            self.first_exception = err[1]

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.first_exception is None:
            self.first_exception = err[1]


def _bcb_run_tests():
    suite = _bcb_unittest.TestLoader().loadTestsFromTestCase(TestCases)
    runner = _bcb_unittest.TextTestRunner(
        stream=_bcb_io.StringIO(), verbosity=0, resultclass=_BcbResult
    )
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise result.first_exception or AssertionError("BigCodeBench tests failed")


_bcb_run_tests()
'''


def set_all_seeds(seed: int) -> None:
    """Set random, numpy, and torch seeds for full reproducibility.

    Args:
        seed: Seed value applied to all three RNG sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger. Use this everywhere instead of print().

    Args:
        name: Logger name, typically `__name__` of the calling module.
    Returns:
        A logger with a single stream handler attached (idempotent across calls).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def save_results(results: dict, prefix: str = "run") -> Path:
    """Save results dict as a timestamped JSON file in results/.

    Args:
        results: The results dictionary (see CLAUDE.md Section 8 schema).
        prefix: Filename prefix, e.g. "humaneval_full".
    Returns:
        The path of the saved file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{prefix}_{timestamp}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    return out_path


def _entry_point_from_code(code: str) -> str:
    """Extract the top-level function name from a reference solution."""
    match = re.search(r"^def\s+(\w+)\s*\(", code, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Could not find a top-level function definition in code")
    return match.group(1)


def _load_humaneval_split() -> list[dict]:
    """Load and normalize the HumanEval test split."""
    from datasets import load_dataset

    raw = load_dataset(_HUMANEVAL_HF_NAME, cache_dir=str(DATA_DIR / "humaneval"))[
        "test"
    ]
    normalized = []
    for row in raw:
        test_code = row["test"] + f"\ncheck({row['entry_point']})\n"
        normalized.append(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "test_code": test_code,
                "entry_point": row["entry_point"],
            }
        )
    return normalized


def _load_mbpp_split() -> list[dict]:
    """Load and normalize the MBPP test split.

    MBPP has no separate function-signature field, so the prompt is built
    from the natural-language task text plus the reference solution's
    signature line (giving the model the same entry-point contract that
    HumanEval prompts provide explicitly). Some MBPP rows carry a
    test_setup_code block (imports/helpers the assertions need); it must
    run before the test_list asserts or those problems fail spuriously.
    """
    from datasets import load_dataset

    raw = load_dataset(_MBPP_HF_NAME, cache_dir=str(DATA_DIR / "mbpp"))["test"]
    normalized = []
    for row in raw:
        entry_point = _entry_point_from_code(row["code"])
        signature_line = re.search(
            r"^def\s+\w+\([^)]*\):", row["code"], flags=re.MULTILINE
        )
        signature = (
            signature_line.group(0) if signature_line else f"def {entry_point}(...):"
        )
        prompt = f'"""{row["text"]}"""\n{signature}\n'
        setup = (row.get("test_setup_code") or "").strip()
        test_lines = ([setup] if setup else []) + list(row["test_list"])
        test_code = "\n".join(test_lines)
        normalized.append(
            {
                "task_id": f"MBPP/{row['task_id']}",
                "prompt": prompt,
                "test_code": test_code,
                "entry_point": entry_point,
            }
        )
    return normalized


def _load_bigcodebench_split() -> list[dict]:
    """Load and normalize BigCodeBench (Complete split, v0.1.4).

    The harder, library-heavy cell added in Week 11. Two differences from
    HumanEval/MBPP that every downstream consumer must know about:

    * `complete_prompt` is the completion-model prompt (imports, signature,
      docstring), so `prompt + sample` assembles the same way as HumanEval and
      the Week 7 representation split still applies.
    * `test` is a bare `unittest.TestCase` subclass with no runner;
      `BIGCODEBENCH_TEST_DRIVER` is appended to execute it and re-raise the
      first real exception, preserving the Section 9 failure taxonomy.

    Returns:
        List of normalized problem dicts, entry point `task_func` throughout.
    """
    from datasets import load_dataset

    raw = load_dataset(
        _BIGCODEBENCH_HF_NAME,
        split=_BIGCODEBENCH_SPLIT,
        cache_dir=str(DATA_DIR / "bigcodebench"),
    )
    normalized = []
    for row in raw:
        normalized.append(
            {
                "task_id": row["task_id"],
                "prompt": row["complete_prompt"],
                "test_code": row["test"] + BIGCODEBENCH_TEST_DRIVER,
                "entry_point": row["entry_point"],
                "canonical_solution": row["canonical_solution"],
            }
        )
    return normalized


def _probe_inputs_from_evalplus_test(test_source: str) -> list[str]:
    """Parse EvalPlus's `inputs = [[...], ...]` literal into probe strings.

    EvalPlus's HuggingFace mirrors embed the raw test inputs as a literal
    Python assignment inside the `test` field, decoupled from expected
    outputs. Only the inputs are read — expected values are deliberately
    never extracted (the execution-behavior metrics must stay verdict-free,
    see src/uncertainty.py's Week 6 section).
    """
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "inputs" for t in node.targets):
            continue
        try:
            arg_lists = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError, MemoryError):
            return []
        if not isinstance(arg_lists, list):
            return []
        return [
            ", ".join(repr(arg) for arg in args)
            for args in arg_lists
            if isinstance(args, (list, tuple))
        ]
    return []


def load_evalplus_probe_inputs(
    dataset_name: str, cap: int = 12, seed: int = 42
) -> dict[str, list[str]]:
    """Load EvalPlus's expanded test INPUTS as behavior-probe strings.

    Sources (Liu et al., 2023, NeurIPS — EvalPlus/HumanEval+/MBPP+):
    HuggingFace `evalplus/humanevalplus` (all 164 HumanEval problems, ~80x
    the original test volume) and `evalplus/mbppplus` (the 378-problem
    MBPP-sanitized subset, ~35x). Task ids are normalized to this project's
    format ("HumanEval/0", "MBPP/11"). Problems whose inputs cannot be
    parsed as literals are omitted — callers fall back to the test-derived
    probes from evaluation.extract_probe_inputs() for those.

    cap bounds replay cost: He et al. (arXiv 2605.09023, Fig. 1) show the
    behavioral signal is flat from 10 to 1,000 inputs, so a small cap
    loses nothing. When a problem has more inputs than cap, a
    seed-deterministic sample is taken (CLAUDE.md Constraint #4).

    Args:
        dataset_name: "humaneval" or "mbpp" (case-insensitive).
        cap: Maximum probes per problem (<= 0 disables capping).
        seed: Seed for the deterministic subsample.
    Returns:
        Dict mapping task_id to a list of probe argument-list strings
        (same format behavior_signature() consumes).
    """
    from datasets import load_dataset

    key = dataset_name.strip().lower()
    if key == "humaneval":
        raw = load_dataset(
            "evalplus/humanevalplus", cache_dir=str(DATA_DIR / "humanevalplus")
        )["test"]
        normalize = lambda row: str(
            row["task_id"]
        )  # noqa: E731 — already "HumanEval/k"
    elif key == "mbpp":
        raw = load_dataset("evalplus/mbppplus", cache_dir=str(DATA_DIR / "mbppplus"))[
            "test"
        ]
        normalize = lambda row: f"MBPP/{row['task_id']}"  # noqa: E731
    else:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name!r}. Expected 'humaneval' or 'mbpp'."
        )

    rng = random.Random(seed)
    probes_by_task: dict[str, list[str]] = {}
    for row in raw:
        probes = _probe_inputs_from_evalplus_test(row["test"])
        if not probes:
            continue
        if cap > 0 and len(probes) > cap:
            probes = rng.sample(probes, cap)
        probes_by_task[normalize(row)] = probes
    return probes_by_task


def load_evalplus_test_code(dataset_name: str) -> dict[str, str]:
    """Load EvalPlus's expanded test suites as drop-in `test_code` strings.

    Sibling of :func:`load_evalplus_probe_inputs`, but for *labels* rather
    than probes: the HumanEval+ / MBPP+ `test` field is a full
    ``check(candidate)`` harness, so appending the entry-point call yields
    exactly the shape :func:`src.evaluation.run_code` already consumes from
    :func:`load_dataset_split`.

    Two differences from the base suites matter for interpretation
    (Liu et al., 2023, NeurIPS — EvalPlus): the input set is ~80x (HumanEval+)
    / ~35x (MBPP+) larger, and comparisons go through EvalPlus's
    ``assertion(out, exp, atol)`` with an ``np.allclose`` tolerance path
    instead of base HumanEval's exact ``==``. The relabel is therefore
    *two-directional* — a float-precision sample can fail the base suite and
    pass the plus one — so callers must not assume plus-failure is a subset
    of base-failure.

    No cap is applied (unlike the probe loader's ``cap=12``): a subsampled
    suite is a different, weaker label.

    The two mirrors assemble differently (Week 9): HumanEval+'s `test` defines
    `check(candidate)` and needs that call appended, while MBPP+'s calls its
    entry point directly at module level and is runnable as it stands. MBPP+
    ships no `entry_point` column, which is why this loader was HumanEval-only
    until Week 9; nothing in the harness actually needs one.

    Args:
        dataset_name: "humaneval" or "mbpp" (case-insensitive).
    Returns:
        Dict mapping task_id ("HumanEval/0", "MBPP/11") to a ready-to-execute
        test_code string.
    """
    from datasets import load_dataset

    key = dataset_name.strip().lower()
    if key == "humaneval":
        raw = load_dataset(
            _HUMANEVALPLUS_HF_NAME, cache_dir=str(DATA_DIR / "humanevalplus")
        )["test"]
        normalize = lambda row: str(row["task_id"])  # noqa: E731 — "HumanEval/k"
        assemble = (  # noqa: E731
            lambda row: row["test"] + f"\ncheck({row['entry_point']})\n"
        )
    elif key == "mbpp":
        raw = load_dataset("evalplus/mbppplus", cache_dir=str(DATA_DIR / "mbppplus"))[
            "test"
        ]
        normalize = lambda row: f"MBPP/{row['task_id']}"  # noqa: E731
        assemble = lambda row: row["test"]  # noqa: E731 — self-contained harness
    else:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name!r}. Expected 'humaneval' or 'mbpp'."
        )

    return {normalize(row): assemble(row) for row in raw}


def load_evalplus_canonical_solutions(dataset_name: str) -> dict[str, str]:
    """Load EvalPlus's reference solutions, as executable `prompt + body`.

    Used to validate a relabelling run before trusting it: the reference
    solution must pass its own plus suite for every problem, and its
    wall-clock time is what a timeout budget should be set from.

    Args:
        dataset_name: "humaneval" or "mbpp" (case-insensitive). MBPP+ stores
            its reference solution complete in `code`, so nothing is
            concatenated for it.
    Returns:
        Dict mapping task_id to a complete, runnable solution string.
    """
    from datasets import load_dataset

    key = dataset_name.strip().lower()
    if key == "humaneval":
        raw = load_dataset(
            _HUMANEVALPLUS_HF_NAME, cache_dir=str(DATA_DIR / "humanevalplus")
        )["test"]
        normalize = lambda row: str(row["task_id"])  # noqa: E731
        assemble = lambda row: row["prompt"] + row["canonical_solution"]  # noqa: E731
    elif key == "mbpp":
        raw = load_dataset("evalplus/mbppplus", cache_dir=str(DATA_DIR / "mbppplus"))[
            "test"
        ]
        normalize = lambda row: f"MBPP/{row['task_id']}"  # noqa: E731
        assemble = lambda row: row["code"]  # noqa: E731 — already complete
    else:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name!r}. Expected 'humaneval' or 'mbpp'."
        )

    return {normalize(row): assemble(row) for row in raw}


def load_dataset_split(dataset_name: str) -> list[dict]:
    """Load HumanEval or MBPP using the `datasets` library.

    Normalizes field names so downstream code is dataset-agnostic.
    Normalized fields: "task_id", "prompt", "test_code", "entry_point".

    Args:
        dataset_name: "humaneval", "mbpp" or "bigcodebench" (case-insensitive).
    Returns:
        List of normalized problem dicts.
    """
    key = dataset_name.strip().lower()
    if key == "humaneval":
        return _load_humaneval_split()
    if key == "mbpp":
        return _load_mbpp_split()
    if key == "bigcodebench":
        return _load_bigcodebench_split()
    raise ValueError(
        f"Unknown dataset_name: {dataset_name!r}. "
        "Expected 'humaneval', 'mbpp' or 'bigcodebench'."
    )
