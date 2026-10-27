"""Execution-guided repair pipeline owned by Student B.

The module retains the Week 2 one-step baseline, adds optional execution-error
context from Week 3, provides Week 4 bounded iterative repair, and implements
Week 5 execution-guided ranking across multiple repair candidates. Week 6 adds
duplicate-aware resampling, explicit model-call budgets, and observable early
stopping without changing the earlier defaults.
"""

from __future__ import annotations

import re

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.evaluation import classify_failure, run_code
from src.generation import generate_code
from src.utils import get_logger, set_all_seeds

logger = get_logger(__name__)

# A deterministic fallback for candidates that all fail the tests. Passing
# candidates always rank first. Among failures, reaching an assertion means the
# program parsed and ran far enough for the test oracle to reject its behavior.
# Other runtime failures rank above timeouts/import failures, while syntax
# errors are least useful as the next iterative-repair state.
_FAILURE_PROGRESS = {
    "syntax_error": 0,
    "import_error": 1,
    "timeout": 1,
    "name_error": 2,
    "type_error": 2,
    "runtime_error": 2,
    "assertion_error": 3,
    "": 4,
}


def _supports_chat_template(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Return whether the tokenizer can format instruction/chat messages."""
    return bool(getattr(tokenizer, "chat_template", None))


def _build_instruction_repair_prompt(
    prompt: str, failed_code: str, error_message: str
) -> str:
    """Build a direct debugging request for an instruction-tuned code model."""
    feedback = error_message.strip() or "The candidate failed its tests."
    return (
        "Repair the Python solution using the execution feedback.\n\n"
        "Original task and function contract:\n"
        f"{prompt.rstrip()}\n\n"
        "Failing solution:\n"
        "```python\n"
        f"{failed_code.rstrip()}\n"
        "```\n\n"
        "Execution feedback:\n"
        f"{feedback}\n\n"
        "Return only the complete corrected Python program. "
        "Do not use Markdown fences and do not explain the answer."
    )


def _extract_python_program(text: str) -> str:
    """Remove common chat-model Markdown wrapping from generated code."""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    program = fenced.group(1) if fenced else text
    return program.strip() + "\n"


#: Week 9. How much context a candidate-generation call is given. The ladder
#: exists to decompose what execution-guided repair's gain is actually made
#: of: "none" regenerates from the task prompt alone (an i.i.d. resample at
#: the same model and budget), "code_only" adds the failing program, and
#: "error" adds the traceback on top — the shipped Week 3+ behavior and the
#: default, so every earlier call site is unchanged.
REPAIR_CONTEXT_MODES = ("error", "code_only", "none")


def _build_instruction_generation_prompt(prompt: str) -> str:
    """Build a from-scratch solve request for an instruction-tuned code model.

    The counterpart of _build_instruction_repair_prompt with both execution
    signals removed: no failing program, no feedback. Kept in the same house
    style (identical framing, identical output instruction) so the arms differ
    in context and nothing else.
    """
    return (
        "Write a Python solution for the task below.\n\n"
        "Task and function contract:\n"
        f"{prompt.rstrip()}\n\n"
        "Return only the complete Python program. "
        "Do not use Markdown fences and do not explain the answer."
    )


def _build_instruction_code_only_repair_prompt(prompt: str, failed_code: str) -> str:
    """Build a repair request that shows the failing program but no traceback."""
    return (
        "Repair the Python solution below. It fails its tests.\n\n"
        "Original task and function contract:\n"
        f"{prompt.rstrip()}\n\n"
        "Failing solution:\n"
        "```python\n"
        f"{failed_code.rstrip()}\n"
        "```\n\n"
        "Return only the complete corrected Python program. "
        "Do not use Markdown fences and do not explain the answer."
    )


def _build_baseline_repair_prompt(prompt: str, failed_code: str) -> str:
    """Build the Week 2 repair prompt without using execution errors.

    The original dataset prompt is repeated at the end so a causal code model
    continues from the same function-signature anchor used during generation.
    """
    return (
        "# The following Python solution failed the tests.\n"
        "# Write a corrected implementation for the same task.\n\n"
        f"{failed_code.rstrip()}\n\n"
        "# Corrected implementation:\n"
        f"{prompt}"
    )


def _build_error_context_repair_prompt(
    prompt: str, failed_code: str, error_message: str
) -> str:
    """Build the Week 3 repair prompt using the execution traceback."""
    return (
        "# The following Python solution failed the tests.\n"
        "# Use the traceback to write a corrected implementation for the same task.\n\n"
        "# Failed solution:\n"
        f"{failed_code.rstrip()}\n\n"
        "# Traceback:\n"
        f"{error_message.strip()}\n\n"
        "# Corrected implementation:\n"
        f"{prompt}"
    )


def _assemble_repaired_code(prompt: str, completion: str) -> str:
    """Convert a repair completion into complete runnable candidate code."""
    stripped = completion.lstrip()
    if stripped.startswith(("def ", "from ", "import ", "class ")):
        return completion
    return prompt + completion


def repair_code(
    prompt: str,
    failed_code: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    error_message: str = "",
    temperature: float = 0.7,
    *,
    context_mode: str = "error",
) -> str:
    """Attempt to repair a failing code sample using the LLM.

    Week 2 behavior: error_message empty (baseline, no error context).
    Week 3 behavior: error_message included in the repair prompt when present.

    Args:
        prompt: Original problem prompt.
        failed_code: The code that failed execution.
        tokenizer, model: From generation.load_model().
        error_message: Exception traceback from evaluation.run_code().
                       Empty string in Week 2 baseline. Ignored unless
                       context_mode is "error".
        temperature: Sampling temperature for repair generation.
        context_mode: Week 9, keyword-only. One of REPAIR_CONTEXT_MODES.
            "error" (default) is the shipped behavior, so every pre-Week-9
            call site is bit-identical. "code_only" withholds the traceback
            and "none" withholds the failing program as well, turning the
            call into a same-model, same-budget resample of the task —
            the control that separates execution guidance from the repairer's
            own capability.
    Returns:
        Complete repaired code string ready for execution.
    """
    if context_mode not in REPAIR_CONTEXT_MODES:
        raise ValueError(
            f"context_mode must be one of {REPAIR_CONTEXT_MODES}, got {context_mode!r}"
        )
    use_chat_template = _supports_chat_template(tokenizer)
    if context_mode == "none":
        repair_prompt = (
            _build_instruction_generation_prompt(prompt)
            if use_chat_template
            else prompt
        )
    elif context_mode == "code_only":
        repair_prompt = (
            _build_instruction_code_only_repair_prompt(prompt, failed_code)
            if use_chat_template
            else _build_baseline_repair_prompt(prompt, failed_code)
        )
    elif use_chat_template:
        repair_prompt = _build_instruction_repair_prompt(
            prompt, failed_code, error_message
        )
    else:
        repair_prompt = (
            _build_error_context_repair_prompt(prompt, failed_code, error_message)
            if error_message.strip()
            else _build_baseline_repair_prompt(prompt, failed_code)
        )
    completion = generate_code(
        repair_prompt,
        tokenizer,
        model,
        temperature=temperature,
        use_chat_template=use_chat_template,
    )
    if use_chat_template:
        completion = _extract_python_program(completion)
    return _assemble_repaired_code(prompt, completion)


def _candidate_ranking_score(candidate: dict, current_code: str) -> tuple[int, ...]:
    """Build the deterministic Week 5 ranking tuple for one candidate."""
    passed = bool(candidate["passed"])
    category = candidate["failure_category"]
    is_unique = not bool(candidate.get("duplicate", False))
    changed = candidate["code"] != current_code
    candidate_index = int(candidate["candidate_index"])
    return (
        int(passed),
        _FAILURE_PROGRESS.get(category, -1),
        int(is_unique),
        int(changed),
        -candidate_index,
    )


def rank_repair_candidates(candidates: list[dict], current_code: str) -> list[dict]:
    """Rank evaluated repair candidates from best to worst.

    The ranker is deliberately execution-guided and deterministic:

    1. candidates passing all tests;
    2. failing candidates that progressed furthest through execution;
    3. unique candidates over exact duplicates;
    4. changed candidates over the unchanged current solution; and
    5. the lower generation index as a stable tie-break.

    Each returned record includes its JSON-serializable ``ranking_score``.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    scored = [
        {
            **candidate,
            "ranking_score": list(
                _candidate_ranking_score(candidate, current_code=current_code)
            ),
        }
        for candidate in candidates
    ]
    return sorted(
        scored,
        key=lambda candidate: tuple(candidate["ranking_score"]),
        reverse=True,
    )


def iterative_repair(
    prompt: str,
    failed_code: str,
    test_code: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    max_iterations: int = 3,
    use_error_context: bool = True,
    temperature: float = 0.7,
    n_candidates: int = 1,
    duplicate_policy: str = "accept",
    max_candidate_calls: int | None = None,
    stop_on_pass: bool = False,
    max_consecutive_duplicates: int | None = None,
    seed: int | None = None,
    trace: list[dict] | None = None,
    context_mode: str = "error",
) -> str:
    """Generate, rank, and iterate repairs until success, stagnation, or the limit.

    When ``trace`` is provided, one record is appended for the initial
    candidate (iteration 0) and for every ranking round. Generated candidates,
    scores, and the selected candidate index are retained for Week 5 analysis.
    Exact duplicates reuse their cached execution result. The unchanged current
    candidate participates in ranking so a worse repair cannot regress the
    iterative state.

    Week 9: ``context_mode`` is forwarded to every candidate call (see
    REPAIR_CONTEXT_MODES). It does not change the loop — execution, ranking
    and stopping are identical — only how much of the failure the generator is
    shown, so the arms remain budget-matched by construction.

    ``duplicate_policy="accept"`` preserves the Week 5 behavior: exactly
    ``n_candidates`` model calls are made per non-terminal round and duplicate
    outputs count toward that total. ``duplicate_policy="resample"`` instead
    treats ``n_candidates`` as the target number of unique new programs and
    keeps sampling, up to ``max_candidate_calls``. Optional pass and duplicate-
    streak early stops make the Week 6 efficiency policy explicit while the
    defaults remain backward compatible.
    """
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if n_candidates < 1:
        raise ValueError("n_candidates must be positive")
    if duplicate_policy not in {"accept", "resample"}:
        raise ValueError("duplicate_policy must be 'accept' or 'resample'")
    if context_mode not in REPAIR_CONTEXT_MODES:
        raise ValueError(
            f"context_mode must be one of {REPAIR_CONTEXT_MODES}, got {context_mode!r}"
        )
    candidate_call_limit = (
        n_candidates if max_candidate_calls is None else max_candidate_calls
    )
    if candidate_call_limit < n_candidates:
        raise ValueError("max_candidate_calls must be at least n_candidates")
    if max_consecutive_duplicates is not None and max_consecutive_duplicates < 1:
        raise ValueError("max_consecutive_duplicates must be positive")

    current_code = failed_code
    success, error_message = run_code(current_code, test_code)
    execution_cache = {current_code: (success, error_message)}
    total_model_calls = 0
    if trace is not None:
        trace.append(
            {
                "iteration": 0,
                "passed": success,
                "error_message": error_message,
                "code": current_code,
                "duplicate": False,
                "failure_category": classify_failure(error_message),
                "selected_candidate_index": None,
                "selected_current": True,
                "model_calls": 0,
                "unique_candidates_executed": 0,
                "duplicate_candidates": 0,
                "candidate_generation_stop_reason": "not_started",
                "repair_stop_reason": "initial_success" if success else None,
                "candidates": [],
            }
        )

    for iteration in range(1, max_iterations + 1):
        if success:
            logger.info("Repair succeeded before iteration %d", iteration)
            break

        logger.info(
            "Repair iteration %d/%d: targeting %d candidate(s), call limit %d",
            iteration,
            max_iterations,
            n_candidates,
            candidate_call_limit,
        )
        generated_candidates = []
        round_unique_candidates = 0
        consecutive_duplicates = 0
        candidate_generation_stop_reason = "candidate_call_budget"
        for candidate_index in range(candidate_call_limit):
            if seed is not None:
                candidate_seed = seed + total_model_calls
                set_all_seeds(candidate_seed)
            else:
                candidate_seed = None

            candidate_code = repair_code(
                prompt,
                current_code,
                tokenizer,
                model,
                error_message=error_message if use_error_context else "",
                temperature=temperature,
                context_mode=context_mode,
            )
            model_call_index = total_model_calls
            total_model_calls += 1
            duplicate = candidate_code in execution_cache
            if duplicate:
                candidate_success, candidate_error = execution_cache[candidate_code]
                consecutive_duplicates += 1
            else:
                candidate_success, candidate_error = run_code(candidate_code, test_code)
                execution_cache[candidate_code] = (
                    candidate_success,
                    candidate_error,
                )
                round_unique_candidates += 1
                consecutive_duplicates = 0
            generated_candidates.append(
                {
                    "candidate_index": candidate_index,
                    "model_call_index": model_call_index,
                    "seed": candidate_seed,
                    "passed": candidate_success,
                    "error_message": candidate_error,
                    "failure_category": classify_failure(candidate_error),
                    "code": candidate_code,
                    "duplicate": duplicate,
                }
            )

            if stop_on_pass and candidate_success:
                candidate_generation_stop_reason = "passing_candidate"
                break
            if (
                max_consecutive_duplicates is not None
                and consecutive_duplicates >= max_consecutive_duplicates
            ):
                candidate_generation_stop_reason = "consecutive_duplicate_limit"
                break
            if (
                duplicate_policy == "resample"
                and round_unique_candidates >= n_candidates
            ):
                candidate_generation_stop_reason = "target_unique_candidates"
                break
            if (
                duplicate_policy == "accept"
                and len(generated_candidates) >= n_candidates
            ):
                candidate_generation_stop_reason = "candidate_count_reached"
                break

        current_candidate = {
            "candidate_index": -1,
            "seed": None,
            "passed": success,
            "error_message": error_message,
            "failure_category": classify_failure(error_message),
            "code": current_code,
            "duplicate": False,
        }
        ranked = rank_repair_candidates(
            [current_candidate, *generated_candidates],
            current_code=current_code,
        )
        selected = ranked[0]
        score_by_index = {
            candidate["candidate_index"]: candidate["ranking_score"]
            for candidate in ranked
        }
        scored_generated = [
            {
                **candidate,
                "ranking_score": score_by_index[candidate["candidate_index"]],
            }
            for candidate in generated_candidates
        ]
        selected_current = selected["candidate_index"] == -1
        round_duplicate_candidates = sum(
            bool(candidate["duplicate"]) for candidate in generated_candidates
        )
        if selected["passed"]:
            repair_stop_reason = "success"
        elif selected_current:
            repair_stop_reason = "no_improvement"
        elif iteration == max_iterations:
            repair_stop_reason = "max_iterations"
        else:
            repair_stop_reason = "continue"

        if trace is not None:
            trace.append(
                {
                    "iteration": iteration,
                    "passed": selected["passed"],
                    "error_message": selected["error_message"],
                    "failure_category": selected["failure_category"],
                    "code": selected["code"],
                    "duplicate": selected_current,
                    "selected_current": selected_current,
                    "selected_candidate_index": (
                        None if selected_current else selected["candidate_index"]
                    ),
                    "ranking_score": selected["ranking_score"],
                    "model_calls": len(generated_candidates),
                    "unique_candidates_executed": (
                        len(generated_candidates) - round_duplicate_candidates
                    ),
                    "duplicate_candidates": round_duplicate_candidates,
                    "candidate_generation_stop_reason": (
                        candidate_generation_stop_reason
                    ),
                    "repair_stop_reason": repair_stop_reason,
                    "candidates": scored_generated,
                }
            )

        if selected_current:
            logger.info(
                "No generated candidate outranked the current solution at "
                "iteration %d; stopping",
                iteration,
            )
            break

        current_code = selected["code"]
        success = selected["passed"]
        error_message = selected["error_message"]
        if success:
            logger.info("Repair succeeded at iteration %d", iteration)
            break

    return current_code


def repair_pipeline(
    prompt: str,
    outputs: list[str],
    test_code: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    use_error_context: bool = False,
    max_iterations: int = 3,
    temperature: float = 0.7,
    n_candidates: int = 1,
    duplicate_policy: str = "accept",
    max_candidate_calls: int | None = None,
    stop_on_pass: bool = False,
    max_consecutive_duplicates: int | None = None,
    seed: int | None = None,
    traces: list[list[dict]] | None = None,
) -> list[str]:
    """Apply iterative repair to all failing samples in outputs.

    Passing samples are returned unchanged.

    Args:
        prompt: Original problem prompt.
        outputs: Fully-assembled candidate code strings (prompt + sample).
        test_code: Test assertions to check pass/fail before repairing.
        tokenizer, model: From generation.load_model().
        use_error_context: If True, passes error message into repair_code().
                           Set False for Week 2 baseline, True from Week 3 on.
        n_candidates: Number of independently sampled repairs ranked per
            iteration. One preserves the Week 2-4 behavior.
        duplicate_policy: ``"accept"`` preserves the fixed-call Week 5
            behavior; ``"resample"`` targets ``n_candidates`` unique outputs.
        max_candidate_calls: Per-iteration model-call cap. Defaults to
            ``n_candidates`` for backward compatibility.
        stop_on_pass: Stop generating the current candidate batch as soon as
            one candidate passes all tests.
        max_consecutive_duplicates: Optional duplicate-streak early-stop cap.
        seed: Optional deterministic base seed for repair candidate generation.
        traces: Optional output collector receiving one complete trace per
            input sample, in the same order as ``outputs``.
    Returns:
        List of (repaired or original) code strings, same length as outputs.
    """
    repaired = []
    candidate_call_limit = (
        n_candidates if max_candidate_calls is None else max_candidate_calls
    )
    seed_stride = max(1, max_iterations * candidate_call_limit)
    for sample_index, code in enumerate(outputs):
        sample_seed = None if seed is None else seed + sample_index * seed_stride
        sample_trace = [] if traces is not None else None
        repaired.append(
            iterative_repair(
                prompt,
                code,
                test_code,
                tokenizer,
                model,
                max_iterations=max_iterations,
                use_error_context=use_error_context,
                temperature=temperature,
                n_candidates=n_candidates,
                duplicate_policy=duplicate_policy,
                max_candidate_calls=max_candidate_calls,
                stop_on_pass=stop_on_pass,
                max_consecutive_duplicates=max_consecutive_duplicates,
                seed=sample_seed,
                trace=sample_trace,
            )
        )
        if traces is not None:
            traces.append(sample_trace)
    return repaired
