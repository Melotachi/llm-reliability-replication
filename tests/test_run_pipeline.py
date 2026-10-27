"""CLI configuration tests for the integrated pipeline."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from scripts.run_pipeline import parse_args, run_pipeline


def _fake_loaded_model() -> MagicMock:
    """Stand-in model whose weights carry a real dtype.

    run_pipeline records the loaded weights' dtype in the results config
    (``dtype_actual``), so the fake must expose ``.parameters()`` the way a
    real HF model does.
    """
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])
    return model


def test_error_context_is_opt_in_for_week2_compatibility():
    with patch.object(sys, "argv", ["run_pipeline.py"]):
        args = parse_args()

    assert args.use_error_context is False
    assert args.repair_candidates == 1


def test_cli_enables_week3_error_context():
    with patch.object(
        sys,
        "argv",
        ["run_pipeline.py", "--mode", "repair", "--use-error-context"],
    ):
        args = parse_args()

    assert args.mode == "repair"
    assert args.use_error_context is True


def test_cli_enables_week5_candidate_ranking():
    with patch.object(
        sys,
        "argv",
        ["run_pipeline.py", "--mode", "repair", "--repair-candidates", "4"],
    ):
        args = parse_args()

    assert args.mode == "repair"
    assert args.repair_candidates == 4


def test_cli_rejects_nonpositive_week5_candidate_count():
    with (
        patch.object(
            sys,
            "argv",
            ["run_pipeline.py", "--repair-candidates", "0"],
        ),
        pytest.raises(SystemExit),
    ):
        parse_args()


def _fake_tokenizer():
    """Minimal tokenizer stand-in: the pipeline calls it for the truncation proxy."""

    def tokenize(text: str):
        return SimpleNamespace(input_ids=text.split())

    return tokenize


def test_pipeline_reuses_ranked_trace_result_without_reexecution():
    args = SimpleNamespace(
        dataset="humaneval",
        n_problems=1,
        problem_offset=0,
        task_ids_file=None,
        n_samples=1,
        max_new_tokens=256,
        exec_timeout=5.0,
        model="model",
        seed=42,
        mode="repair",
        temperature=0.2,
        device="auto",
        dtype="auto",
        device_map="none",
        no_tree_edit=False,
        use_error_context=True,
        max_repair_iterations=1,
        repair_candidates=3,
        codebert_model="microsoft/codebert-base",
        no_codebert=True,
        dry_run=False,
    )
    problem = {
        "task_id": "HumanEval/0",
        "prompt": "def f():\n",
        "test_code": "assert f() == 1",
    }

    def fake_repair_pipeline(*args, traces, **kwargs):
        traces.append(
            [
                {
                    "iteration": 0,
                    "passed": False,
                    "error_message": "AssertionError",
                },
                {
                    "iteration": 1,
                    "passed": True,
                    "error_message": "",
                    "selected_candidate_index": 1,
                },
            ]
        )
        return ["def f():\n    return 1\n"]

    with (
        patch(
            "scripts.run_pipeline.load_dataset_split",
            return_value=[problem],
        ),
        patch(
            "scripts.run_pipeline.load_model",
            return_value=(_fake_tokenizer(), _fake_loaded_model()),
        ),
        patch(
            "scripts.run_pipeline.multi_sample_generation",
            return_value=["    return 0\n"],
        ),
        patch(
            "scripts.run_pipeline.evaluate_outputs",
            return_value=[(False, "AssertionError")],
        ) as fake_evaluate,
        patch(
            "scripts.run_pipeline.repair_pipeline",
            side_effect=fake_repair_pipeline,
        ),
    ):
        results = run_pipeline(args)

    fake_evaluate.assert_called_once()
    sample = results["per_problem"][0]["samples"][0]
    assert sample["repaired_passed"] is True
    assert sample["repair_trace"][-1]["selected_candidate_index"] == 1


def _uncertainty_args(tmp_path, task_ids_file):
    """Minimal uncertainty-mode args for the protocol-hash tests."""
    return SimpleNamespace(
        dataset="bigcodebench",
        n_problems=2,
        problem_offset=0,
        task_ids_file=task_ids_file,
        n_samples=1,
        max_new_tokens=256,
        exec_timeout=5.0,
        model="model",
        seed=42,
        mode="uncertainty",
        temperature=0.2,
        device="auto",
        dtype="auto",
        device_map="none",
        no_tree_edit=True,
        use_error_context=False,
        max_repair_iterations=1,
        repair_candidates=3,
        codebert_model="microsoft/codebert-base",
        no_codebert=True,
        dry_run=False,
    )


def _run_with_allowlist(tmp_path, ids):
    """Run the pipeline over `ids` via an allowlist file of a FIXED name."""
    allowlist = tmp_path / "cell_task_ids.txt"
    allowlist.write_text("\n".join(ids) + "\n")
    problems = [
        {
            "task_id": task_id,
            "prompt": "def f():\n",
            "test_code": "assert f() == 1",
            "entry_point": "f",
        }
        for task_id in ("BigCodeBench/1", "BigCodeBench/2", "BigCodeBench/3")
    ]
    with (
        patch("scripts.run_pipeline.load_dataset_split", return_value=problems),
        patch(
            "scripts.run_pipeline.load_model",
            return_value=(_fake_tokenizer(), _fake_loaded_model()),
        ),
        patch(
            "scripts.run_pipeline.multi_sample_generation",
            return_value=["    return 1\n"],
        ),
        patch(
            "scripts.run_pipeline.evaluate_outputs",
            return_value=[(True, "")],
        ),
    ):
        return run_pipeline(_uncertainty_args(tmp_path, allowlist))


def test_task_ids_sha256_distinguishes_runs_that_share_an_allowlist_path(tmp_path):
    """Week 11: the path does not identify the problem set; the hash does.

    A capability ladder joins its rungs on the protocol keys. Two rungs can
    name the same --task-ids-file and evaluate different problems -- which is
    exactly what happens when the file is rewritten by a gate whose result
    depends on the executing image -- and would otherwise join silently.
    """
    first = _run_with_allowlist(tmp_path, ["BigCodeBench/1", "BigCodeBench/2"])
    second = _run_with_allowlist(tmp_path, ["BigCodeBench/2", "BigCodeBench/3"])

    assert first["config"]["task_ids_file"] == second["config"]["task_ids_file"]
    assert first["config"]["task_ids_sha256"] != second["config"]["task_ids_sha256"]


def test_task_ids_sha256_is_stable_across_runs_over_the_same_problems(tmp_path):
    first = _run_with_allowlist(tmp_path, ["BigCodeBench/1", "BigCodeBench/3"])
    second = _run_with_allowlist(tmp_path, ["BigCodeBench/1", "BigCodeBench/3"])
    assert first["config"]["task_ids_sha256"] == second["config"]["task_ids_sha256"]
