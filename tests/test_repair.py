"""Tests for src/repair.py. generate_code is mocked per CLAUDE.md FAQ
("mock generate_code() in unit tests; reserve real model calls for
integration tests tagged @pytest.mark.slow").
"""

from __future__ import annotations

from unittest.mock import patch

from src import repair


class ChatTokenizer:
    chat_template = "template"


def _ranking_candidate(
    index, code, *, passed=False, category="assertion_error", duplicate=False
):
    return {
        "candidate_index": index,
        "code": code,
        "passed": passed,
        "failure_category": "" if passed else category,
        "error_message": "" if passed else category,
        "duplicate": duplicate,
    }


def test_rank_repair_candidates_prioritizes_pass_then_execution_progress():
    candidates = [
        _ranking_candidate(-1, "current", category="syntax_error"),
        _ranking_candidate(0, "runtime", category="runtime_error"),
        _ranking_candidate(1, "passing", passed=True),
        _ranking_candidate(2, "assertion", category="assertion_error"),
    ]

    ranked = repair.rank_repair_candidates(candidates, current_code="current")

    assert [candidate["code"] for candidate in ranked] == [
        "passing",
        "assertion",
        "runtime",
        "current",
    ]
    assert all("ranking_score" in candidate for candidate in ranked)


def test_rank_repair_candidates_keeps_current_over_a_regression():
    candidates = [
        _ranking_candidate(-1, "current", category="assertion_error"),
        _ranking_candidate(0, "worse", category="syntax_error"),
    ]

    ranked = repair.rank_repair_candidates(candidates, current_code="current")

    assert ranked[0]["candidate_index"] == -1


def test_rank_repair_candidates_prefers_changed_candidate_on_equal_progress():
    candidates = [
        _ranking_candidate(-1, "current", category="assertion_error"),
        _ranking_candidate(0, "changed", category="assertion_error"),
    ]

    ranked = repair.rank_repair_candidates(candidates, current_code="current")

    assert ranked[0]["candidate_index"] == 0


def test_repair_code_builds_fix_prompt_and_delegates_to_generate_code():
    with patch.object(
        repair, "generate_code", return_value="    return 1\n"
    ) as fake_generate:
        fixed = repair.repair_code(
            prompt="def f(x):\n",
            failed_code="def f(x):\n    return x + \n",
            tokenizer=object(),
            model=object(),
            temperature=0.5,
        )

    assert fixed == "def f(x):\n    return 1\n"
    args, kwargs = fake_generate.call_args
    assert "def f(x):\n    return x +" in args[0]
    assert "Corrected implementation" in args[0]
    assert args[0].endswith("def f(x):\n")
    assert kwargs["temperature"] == 0.5


def test_repair_code_uses_baseline_prompt_without_error_message():
    with patch.object(
        repair, "generate_code", return_value="    return 1\n"
    ) as fake_generate:
        repair.repair_code(
            prompt="def f():\n",
            failed_code="def f():\n    return 2\n",
            tokenizer=object(),
            model=object(),
        )

    args, _ = fake_generate.call_args
    assert "Traceback" not in args[0]
    assert "AssertionError" not in args[0]


def test_repair_code_includes_error_message_for_week3_context():
    with patch.object(
        repair, "generate_code", return_value="    return 1\n"
    ) as fake_generate:
        repair.repair_code(
            prompt="def f():\n",
            failed_code="def f():\n    return 2\n",
            tokenizer=object(),
            model=object(),
            error_message="Traceback...\nAssertionError\n",
        )

    args, _ = fake_generate.call_args
    assert "# Traceback:" in args[0]
    assert "AssertionError" in args[0]
    assert args[0].endswith("def f():\n")


def test_repair_code_preserves_full_code_completion():
    with patch.object(repair, "generate_code", return_value="def f():\n    return 1\n"):
        fixed = repair.repair_code(
            prompt="def f():\n",
            failed_code="def f():\n    return 2\n",
            tokenizer=object(),
            model=object(),
        )

    assert fixed == "def f():\n    return 1\n"


def test_repair_code_uses_instruction_prompt_and_extracts_fenced_code():
    generated = "Here is the fix:\n```python\ndef f():\n    return 1\n```"
    with patch.object(repair, "generate_code", return_value=generated) as fake_generate:
        fixed = repair.repair_code(
            prompt="def f():\n",
            failed_code="def f():\n    return 2\n",
            tokenizer=ChatTokenizer(),
            model=object(),
            error_message="AssertionError",
        )

    repair_prompt = fake_generate.call_args.args[0]
    assert "Failing solution:" in repair_prompt
    assert "Execution feedback:" in repair_prompt
    assert fixed == "def f():\n    return 1\n"
    assert fake_generate.call_args.kwargs["use_chat_template"] is True


def test_repair_pipeline_only_repairs_failing_samples():
    passing_code = "def f():\n    return 1\n"
    failing_code = "def f():\n    return 2\n"
    test_code = "assert f() == 1"
    outputs = [passing_code, failing_code]

    with patch.object(
        repair, "generate_code", return_value="    return 1\n"
    ) as fake_generate:
        repaired = repair.repair_pipeline(
            prompt="def f():\n",
            outputs=outputs,
            test_code=test_code,
            tokenizer=object(),
            model=object(),
            max_iterations=1,
        )

    assert repaired[0] == passing_code  # unchanged, already passing
    assert repaired[1] == "def f():\n    return 1\n"  # repaired once
    fake_generate.assert_called_once()


def test_repair_pipeline_passes_error_message_only_when_use_error_context():
    failing_code = "def f():\n    return 2\n"
    test_code = "assert f() == 1"

    with patch.object(repair, "repair_code", return_value="fixed") as fake_repair_code:
        repair.repair_pipeline(
            "def f():\n",
            [failing_code],
            test_code,
            object(),
            object(),
            use_error_context=True,
            max_iterations=1,
        )
    _, kwargs = fake_repair_code.call_args
    assert kwargs["error_message"] != ""

    with patch.object(repair, "repair_code", return_value="fixed") as fake_repair_code:
        repair.repair_pipeline(
            "def f():\n",
            [failing_code],
            test_code,
            object(),
            object(),
            use_error_context=False,
            max_iterations=1,
        )
    _, kwargs = fake_repair_code.call_args
    assert kwargs["error_message"] == ""


def test_iterative_repair_stops_after_first_success():
    execution_results = [
        (False, "first error"),
        (False, "second error"),
        (True, ""),
    ]

    with (
        patch.object(repair, "run_code", side_effect=execution_results),
        patch.object(
            repair, "repair_code", side_effect=["first fix", "second fix"]
        ) as fake_repair,
    ):
        result = repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_iterations=5,
        )

    assert result == "second fix"
    assert fake_repair.call_count == 2
    assert fake_repair.call_args_list[0].kwargs["error_message"] == "first error"
    assert fake_repair.call_args_list[1].kwargs["error_message"] == "second error"


def test_iterative_repair_stops_at_iteration_limit():
    with (
        patch.object(repair, "run_code", return_value=(False, "still broken")),
        patch.object(
            repair, "repair_code", side_effect=["fix 1", "fix 2", "fix 3"]
        ) as fake_repair,
    ):
        result = repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_iterations=3,
        )

    assert result == "fix 3"
    assert fake_repair.call_count == 3


def test_iterative_repair_ranks_multiple_candidates_and_records_seeds():
    trace = []
    with (
        patch.object(
            repair,
            "run_code",
            side_effect=[
                (False, "SyntaxError"),
                (False, "AssertionError"),
                (False, "TypeError"),
                (True, ""),
            ],
        ) as fake_run,
        patch.object(
            repair,
            "repair_code",
            side_effect=["assertion fix", "type fix", "passing fix"],
        ) as fake_repair,
        patch.object(repair, "set_all_seeds") as fake_seed,
    ):
        result = repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_iterations=1,
            n_candidates=3,
            seed=100,
            trace=trace,
        )

    assert result == "passing fix"
    assert fake_repair.call_count == 3
    assert fake_run.call_count == 4
    assert [call.args[0] for call in fake_seed.call_args_list] == [100, 101, 102]
    assert trace[-1]["selected_candidate_index"] == 2
    assert len(trace[-1]["candidates"]) == 3
    assert trace[-1]["passed"] is True


def test_iterative_repair_executes_each_unique_candidate_once():
    trace = []
    with (
        patch.object(
            repair,
            "run_code",
            side_effect=[
                (False, "AssertionError"),
                (False, "AssertionError"),
            ],
        ) as fake_run,
        patch.object(
            repair,
            "repair_code",
            side_effect=["broken", "alternative", "alternative"],
        ),
    ):
        result = repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_iterations=1,
            n_candidates=3,
            trace=trace,
        )

    assert result == "alternative"
    assert fake_run.call_count == 2
    assert [candidate["duplicate"] for candidate in trace[-1]["candidates"]] == [
        True,
        False,
        True,
    ]
    assert trace[-1]["selected_candidate_index"] == 1


def test_iterative_repair_resamples_duplicates_until_unique_target():
    trace = []
    with (
        patch.object(
            repair,
            "run_code",
            side_effect=[
                (False, "AssertionError"),
                (False, "AssertionError"),
                (False, "AssertionError"),
                (False, "AssertionError"),
            ],
        ) as fake_run,
        patch.object(
            repair,
            "repair_code",
            side_effect=[
                "broken",
                "alternative 1",
                "alternative 1",
                "alternative 2",
                "alternative 3",
            ],
        ) as fake_repair,
        patch.object(repair, "set_all_seeds") as fake_seed,
    ):
        result = repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_iterations=1,
            n_candidates=3,
            duplicate_policy="resample",
            max_candidate_calls=6,
            seed=100,
            trace=trace,
        )

    assert result == "alternative 1"
    assert fake_repair.call_count == 5
    assert fake_run.call_count == 4  # initial + three unique generated programs
    assert [call.args[0] for call in fake_seed.call_args_list] == [
        100,
        101,
        102,
        103,
        104,
    ]
    assert trace[-1]["model_calls"] == 5
    assert trace[-1]["unique_candidates_executed"] == 3
    assert trace[-1]["duplicate_candidates"] == 2
    assert trace[-1]["candidate_generation_stop_reason"] == "target_unique_candidates"


def test_iterative_repair_stops_candidate_batch_on_first_pass():
    trace = []
    with (
        patch.object(
            repair,
            "run_code",
            side_effect=[(False, "AssertionError"), (True, "")],
        ) as fake_run,
        patch.object(repair, "repair_code", return_value="fixed") as fake_repair,
    ):
        result = repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_iterations=1,
            n_candidates=3,
            stop_on_pass=True,
            trace=trace,
        )

    assert result == "fixed"
    assert fake_repair.call_count == 1
    assert fake_run.call_count == 2
    assert trace[-1]["model_calls"] == 1
    assert trace[-1]["candidate_generation_stop_reason"] == "passing_candidate"
    assert trace[-1]["repair_stop_reason"] == "success"


def test_iterative_repair_stops_after_duplicate_streak():
    trace = []
    with (
        patch.object(
            repair, "run_code", return_value=(False, "AssertionError")
        ) as fake_run,
        patch.object(repair, "repair_code", return_value="broken") as fake_repair,
    ):
        result = repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_iterations=1,
            n_candidates=3,
            duplicate_policy="resample",
            max_candidate_calls=6,
            max_consecutive_duplicates=2,
            trace=trace,
        )

    assert result == "broken"
    assert fake_run.call_count == 1
    assert fake_repair.call_count == 2
    assert trace[-1]["duplicate_candidates"] == 2
    assert (
        trace[-1]["candidate_generation_stop_reason"] == "consecutive_duplicate_limit"
    )
    assert trace[-1]["repair_stop_reason"] == "no_improvement"


def test_iterative_repair_stops_when_model_returns_unchanged_code():
    trace = []
    with (
        patch.object(
            repair, "run_code", return_value=(False, "still broken")
        ) as fake_run,
        patch.object(repair, "repair_code", return_value="broken") as fake_repair,
    ):
        result = repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_iterations=3,
            trace=trace,
        )

    assert result == "broken"
    fake_run.assert_called_once_with("broken", "tests")
    fake_repair.assert_called_once()
    assert [step["iteration"] for step in trace] == [0, 1]
    assert trace[-1]["duplicate"] is True


def test_repair_pipeline_executes_initial_candidate_only_once():
    traces = []
    with (
        patch.object(
            repair,
            "run_code",
            side_effect=[(False, "AssertionError"), (True, "")],
        ) as fake_run,
        patch.object(repair, "repair_code", return_value="fixed"),
    ):
        result = repair.repair_pipeline(
            "task",
            ["broken"],
            "tests",
            object(),
            object(),
            max_iterations=1,
            traces=traces,
        )

    assert result == ["fixed"]
    assert traces[0][-1]["selected_candidate_index"] == 0
    assert fake_run.call_count == 2
    assert fake_run.call_args_list[0].args == ("broken", "tests")
    assert fake_run.call_args_list[1].args == ("fixed", "tests")


def test_repair_pipeline_reserves_nonoverlapping_seed_ranges_for_resampling():
    with patch.object(
        repair, "iterative_repair", side_effect=lambda *args, **kwargs: args[1]
    ) as fake_iterative:
        repair.repair_pipeline(
            "task",
            ["broken 1", "broken 2"],
            "tests",
            object(),
            object(),
            max_iterations=2,
            n_candidates=3,
            duplicate_policy="resample",
            max_candidate_calls=6,
            seed=100,
        )

    assert [call.kwargs["seed"] for call in fake_iterative.call_args_list] == [100, 112]


def test_iterative_repair_rejects_negative_iteration_limit():
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_iterations=-1,
        )


def test_iterative_repair_rejects_nonpositive_candidate_count():
    import pytest

    with pytest.raises(ValueError, match="n_candidates must be positive"):
        repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            n_candidates=0,
        )


def test_iterative_repair_rejects_invalid_week6_candidate_policy():
    import pytest

    with pytest.raises(ValueError, match="duplicate_policy"):
        repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            duplicate_policy="unknown",
        )

    with pytest.raises(ValueError, match="at least n_candidates"):
        repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            n_candidates=3,
            max_candidate_calls=2,
        )

    with pytest.raises(ValueError, match="must be positive"):
        repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            max_consecutive_duplicates=0,
        )


def test_iterative_repair_records_initial_and_repaired_candidates():
    trace = []
    with (
        patch.object(
            repair,
            "run_code",
            side_effect=[(False, "AssertionError"), (True, "")],
        ),
        patch.object(repair, "repair_code", return_value="fixed"),
    ):
        result = repair.iterative_repair(
            "task",
            "broken",
            "tests",
            object(),
            object(),
            trace=trace,
        )

    assert result == "fixed"
    assert [step["iteration"] for step in trace] == [0, 1]
    assert [step["passed"] for step in trace] == [False, True]
    assert trace[-1]["selected_candidate_index"] == 0
    assert trace[-1]["candidates"][0]["duplicate"] is False


def _prompt_for_mode(mode, tokenizer):
    """Return the prompt repair_code() would send under one context mode."""
    with patch.object(
        repair, "generate_code", return_value="def f():\n    return 1\n"
    ) as fake_generate:
        repair.repair_code(
            prompt="def f():\n",
            failed_code="def f():\n    return 2\n",
            tokenizer=tokenizer,
            model=object(),
            error_message="AssertionError: boom",
            context_mode=mode,
        )
    return fake_generate.call_args.args[0]


def test_repair_code_context_mode_none_withholds_code_and_error_on_chat_model():
    repair_prompt = _prompt_for_mode("none", ChatTokenizer())

    assert "return 2" not in repair_prompt
    assert "AssertionError" not in repair_prompt
    assert "Task and function contract:" in repair_prompt


def test_repair_code_context_mode_code_only_keeps_code_and_drops_the_traceback():
    repair_prompt = _prompt_for_mode("code_only", ChatTokenizer())

    assert "return 2" in repair_prompt
    assert "AssertionError" not in repair_prompt


def test_repair_code_context_mode_error_is_the_unchanged_default():
    """The default must stay bit-identical to the pre-Week-9 prompt."""
    tokenizer = ChatTokenizer()
    with patch.object(
        repair, "generate_code", return_value="def f():\n    return 1\n"
    ) as fake_generate:
        repair.repair_code(
            prompt="def f():\n",
            failed_code="def f():\n    return 2\n",
            tokenizer=tokenizer,
            model=object(),
            error_message="AssertionError: boom",
        )
    default_prompt = fake_generate.call_args.args[0]

    assert default_prompt == _prompt_for_mode("error", tokenizer)
    assert "AssertionError: boom" in default_prompt


def test_repair_code_context_mode_none_sends_the_bare_prompt_to_a_causal_model():
    class PlainTokenizer:
        chat_template = None

    assert _prompt_for_mode("none", PlainTokenizer()) == "def f():\n"


def test_repair_code_rejects_an_unknown_context_mode():
    try:
        repair.repair_code(
            prompt="def f():\n",
            failed_code="",
            tokenizer=ChatTokenizer(),
            model=object(),
            context_mode="traceback_only",
        )
    except ValueError as exc:
        assert "context_mode" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError")


def test_iterative_repair_forwards_context_mode_to_every_candidate_call():
    calls = []

    def fake_repair_code(prompt, failed_code, tokenizer, model, **kwargs):
        calls.append(kwargs["context_mode"])
        return f"def f():\n    return {len(calls)}\n"

    with patch.object(repair, "repair_code", side_effect=fake_repair_code):
        repair.iterative_repair(
            prompt="def f():\n",
            failed_code="def f():\n    return 2\n",
            test_code="assert f() == 99",
            tokenizer=ChatTokenizer(),
            model=object(),
            max_iterations=2,
            n_candidates=2,
            seed=42,
            context_mode="none",
        )

    assert calls == ["none"] * 4


def test_iterative_repair_rejects_an_unknown_context_mode():
    try:
        repair.iterative_repair(
            prompt="def f():\n",
            failed_code="def f():\n    return 2\n",
            test_code="assert f() == 1",
            tokenizer=ChatTokenizer(),
            model=object(),
            context_mode="oracle",
        )
    except ValueError as exc:
        assert "context_mode" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError")
