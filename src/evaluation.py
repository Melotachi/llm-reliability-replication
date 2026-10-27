"""Execution engine, failure classification, and pass@k aggregation.

Jointly owned by Student A and Student B (see CLAUDE.md Section 6.4).

Week 6 (Student A, disclosed shared-file addition): probe-input extraction +
behavior signatures (extract_probe_inputs, behavior_signature) — the
execution side of the execution-behavior diversity metric in
src/uncertainty.py. Lives here rather than in uncertainty.py so that all
code execution stays in this module (same # TODO: SANDBOX caveat and the
same subprocess isolation pattern as run_code).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import traceback
from math import comb
from statistics import mean

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

# Section 9 failure taxonomy: map an exception class name (as it appears on
# the final line of a traceback) to the exact category labels the project
# uses everywhere else (results JSON, repair prioritization, plots).
_FAILURE_TAXONOMY: dict[str, str] = {
    "SyntaxError": "syntax_error",
    "IndentationError": "syntax_error",
    "TabError": "syntax_error",
    "NameError": "name_error",
    "AttributeError": "name_error",
    "UnboundLocalError": "name_error",
    "TypeError": "type_error",
    "AssertionError": "assertion_error",
    "ImportError": "import_error",
    "ModuleNotFoundError": "import_error",
    "TimeoutError": "timeout",
}


def _run_code_in_subprocess(
    code: str, test_code: str, timeout_s: float
) -> tuple[bool, str]:
    """Execute generated code in a fresh Python process, in a scratch directory.

    The scratch cwd is not a convenience. BigCodeBench solutions read and write
    real files (271 of its 1,140 test suites touch the filesystem), and running
    them in the repository root is destructive: the first local pass of the
    canonical gate wrote `df_contents.txt`, `empty_test.xlsx`, `data.json`,
    `filename` and `mock_output.json` into the working tree and **deleted
    `src/`**, which had to be restored from git. Generated code, which is wrong
    by definition, is strictly worse than canonical code here. A per-execution
    temporary directory bounds the blast radius to files the candidate itself
    created; it is still not a sandbox (the # TODO: SANDBOX above stands --
    nothing stops an absolute path, a network call, or `rm -rf ~`).
    """
    runner = """
import json
import sys
import traceback

payload = json.load(sys.stdin)
try:
    exec_globals = {}
    exec(payload["code"], exec_globals)
    exec(payload["test_code"], exec_globals)
    result = {"success": True, "error": ""}
except Exception:
    result = {"success": False, "error": traceback.format_exc()}
print(json.dumps(result))
"""
    payload = json.dumps({"code": code, "test_code": test_code})
    try:
        with tempfile.TemporaryDirectory(prefix="rcg_exec_") as scratch:
            stdout, returncode = _run_child(
                [sys.executable, "-c", runner], payload, timeout_s, scratch
            )
    except subprocess.TimeoutExpired:
        return (
            False,
            "Traceback (most recent call last):\n"
            f"TimeoutError: execution exceeded {timeout_s}s time limit\n",
        )

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return (
            False,
            "Traceback (most recent call last):\n"
            f"RuntimeError: execution subprocess exited with code {returncode}\n",
        )
    return bool(result["success"]), str(result["error"])


def run_code(code: str, test_code: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Execute generated code against test cases.

    # TODO: SANDBOX — currently uses exec() with no isolation.
    # Risk: arbitrary code execution. Acceptable for research/local use only.
    # Future: replace with subprocess + RestrictedPython or Docker sandbox.

    A child process guards against infinite loops in generated code
    (Section 9 "timeout" failure category). This is still not a full
    security sandbox, but it gives the repair pipeline a reproducible
    timeout signal on Windows, macOS, and Linux.

    Args:
        code: The generated (or repaired) function code.
        test_code: The test assertions from the dataset.
        timeout_s: Wall-clock execution limit in seconds; <= 0 disables the
            guard. On expiry, returns (False, <TimeoutError traceback>),
            which classify_failure() maps to "timeout".
    Returns:
        (success, error_message)
        success: True if all assertions pass with no exception.
        error_message: Full traceback string on failure; empty string on success.
    """
    if timeout_s > 0:
        return _run_code_in_subprocess(code, test_code, timeout_s)

    # The in-process path still runs in a scratch directory: relative-path file
    # writes are the common case in BigCodeBench, and one of them deleted this
    # repository's src/ during the Week 11 canonical gate.
    origin = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="rcg_exec_") as scratch:
            os.chdir(scratch)
            exec_globals: dict = {}
            exec(code, exec_globals)  # TODO: SANDBOX
            exec(test_code, exec_globals)  # TODO: SANDBOX
            return True, ""
    except Exception:
        return False, traceback.format_exc()
    finally:
        os.chdir(origin)


def evaluate_outputs(
    outputs: list[str], test_code: str, timeout_s: float = 5.0
) -> list[tuple[bool, str]]:
    """Run run_code() for each sample in outputs.

    Args:
        outputs: Candidate code strings to test.
        test_code: The test assertions shared by all candidates.
        timeout_s: Per-sample execution limit forwarded to run_code().
    Returns:
        List of (success, error_message) tuples, same length as outputs.
    """
    return [run_code(code, test_code, timeout_s=timeout_s) for code in outputs]


def extract_probe_inputs(test_code: str, entry_point: str) -> list[str]:
    """Extract the argument lists of every entry-point call in a test suite.

    Statically walks test_code's AST for calls to `candidate(...)` (the
    HumanEval check(candidate) convention) or to the entry-point name
    directly (the MBPP convention) and returns each call's argument list as
    a source string ("1, 2", "[1.0, 2.8], 0.3", ...), deduplicated in
    order of first appearance. These are the shared "probe inputs" that
    behavior_signature() replays every sample on — the test suite's *inputs*
    only; the expected values the asserts compare against are deliberately
    not extracted (see src/uncertainty.py's execution-behavior metrics for
    why behavior must stay verdict-free).

    An argument that references a test-local variable still round-trips
    fine: it will fail identically (NameError) for every sample at replay
    time, contributing a constant — never a misleading — probe outcome.

    Args:
        test_code: The dataset's test source (HumanEval or MBPP style).
        entry_point: The function name under test.
    Returns:
        List of argument-list source strings; empty if test_code does not
        parse or contains no matching calls.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []
    probes: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id in ("candidate", entry_point)):
            continue
        parts = [ast.unparse(arg) for arg in node.args]
        parts += [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]
        source = ", ".join(parts)
        if source not in seen:
            seen.add(source)
            probes.append(source)
    return probes


# Replays one sample on all probe inputs inside a fresh child process (same
# isolation pattern as _run_code_in_subprocess): exec the sample once, then
# eval "<entry_point>(<probe>)" per probe, recording repr(result) or
# "error:<ExceptionType>". If the sample itself fails to exec (or lacks the
# entry point), every probe reports "error:unrunnable".
def _headless_env() -> dict[str, str]:
    """The parent environment, forced non-interactive for the child process.

    Generated BigCodeBench solutions import matplotlib and call plt.show().
    Under an interactive backend that opens a GUI window on the operator's
    screen and blocks until it is closed, so the sample is scored as a
    timeout rather than on its behavior. The stored runs were generated
    headless (Kaggle), so pinning Agg here makes local replay agree with the
    environment the labels came from. Both subprocess call sites run with
    cwd=<scratch>, which is why the repository's matplotlibrc does not reach
    them and the setting has to travel in the environment.
    """
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    return env


# Per-child resource ceilings for the execution subprocesses. The code these
# children run is wrong by definition; a runaway allocation or an infinite
# print loop in one of them, captured into an unbounded parent buffer, is
# enough to exhaust machine memory (observed: a 30s window filling tens of
# GB and OOM-ing the host). Two independent caps close that: the child's own
# address space / file size / CPU here, and the parent-side read cap
# (_MAX_CAPTURE_BYTES) in _run_child(). The address-space ceiling is
# deliberately generous (real BigCodeBench solutions import pandas, sympy,
# and occasionally tensorflow) and overridable for constrained hosts.
_CHILD_ADDRESS_SPACE_BYTES = int(os.environ.get("RCG_CHILD_MEM_GIB", "8")) * 1024**3
_CHILD_FILE_SIZE_BYTES = 128 * 1024**2
_MAX_CAPTURE_BYTES = 1024**2


def _apply_child_limits(timeout_s: float):
    """Build a POSIX preexec_fn that caps the execution child, or None.

    Sets RLIMIT_AS (memory), RLIMIT_FSIZE (any single file the child writes,
    which now includes its redirected stdout) and RLIMIT_CPU. Never raises a
    limit above the inherited hard limit. Returns None where `resource` is
    unavailable, where subprocess ignores preexec_fn anyway.
    """
    try:
        import resource
    except ImportError:  # non-POSIX
        return None

    ceilings = (
        (resource.RLIMIT_AS, _CHILD_ADDRESS_SPACE_BYTES),
        (resource.RLIMIT_FSIZE, _CHILD_FILE_SIZE_BYTES),
        (resource.RLIMIT_CPU, max(1, int(timeout_s) + 5)),
    )

    def _limit() -> None:
        for res, want in ceilings:
            try:
                hard = resource.getrlimit(res)[1]
                cap = want if hard == resource.RLIM_INFINITY else min(want, hard)
                resource.setrlimit(res, (cap, hard))
            except (ValueError, OSError):
                pass

    return _limit


def _run_child(
    argv: list[str], payload: str, timeout_s: float, scratch: str
) -> tuple[str, int]:
    """Run one execution subprocess with bounded capture.

    Child stdout/stderr are written to files under `scratch` rather than
    streamed into a parent pipe, so a child that prints without bound cannot
    grow the parent's memory; at most _MAX_CAPTURE_BYTES of stdout is read
    back. RLIMIT_FSIZE (see _apply_child_limits) bounds the on-disk size in
    turn. Raises subprocess.TimeoutExpired on expiry, like subprocess.run.
    """
    out_path = os.path.join(scratch, "__child_stdout")
    err_path = os.path.join(scratch, "__child_stderr")
    preexec = _apply_child_limits(timeout_s)
    with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
        completed = subprocess.run(
            argv,
            input=payload.encode("utf-8"),
            stdout=out_f,
            stderr=err_f,
            timeout=timeout_s,
            check=False,
            cwd=scratch,
            env=_headless_env(),
            preexec_fn=preexec if preexec and os.name == "posix" else None,
        )
    with open(out_path, "rb") as out_f:
        stdout = out_f.read(_MAX_CAPTURE_BYTES).decode("utf-8", "replace")
    return stdout, completed.returncode


_BEHAVIOR_RUNNER = """
import json
import sys

payload = json.load(sys.stdin)
outcomes = []
ns = {}
try:
    exec(payload["code"], ns)
    if payload["entry_point"] not in ns:
        raise NameError(payload["entry_point"])
except BaseException:
    outcomes = ["error:unrunnable"] * len(payload["probes"])
else:
    for probe in payload["probes"]:
        try:
            result = eval(payload["entry_point"] + "(" + probe + ")", ns)
            outcomes.append(repr(result))
        except BaseException as exc:
            outcomes.append("error:" + type(exc).__name__)
print(json.dumps(outcomes))
"""


_FIXTURE_RUNNER = """
import json
import sys
import unittest
from contextlib import nullcontext

payload = json.load(sys.stdin)
entry_point = payload["entry_point"]
records = []

# Assertions are neutralised BEFORE the suite runs, so control flow inside a
# test method cannot depend on whether the candidate is correct. Without this,
# a wrong sample stops at its first failing assert and never reaches the later
# task_func() calls -- which would make the signature's *length* a function of
# the label. Nothing here reads an assertion outcome.
def _noop(*a, **k):
    return None


def _null_ctx(*a, **k):
    return nullcontext()


for _name in dir(unittest.TestCase):
    if _name.startswith(("assert", "failUnless", "failIf")):
        setattr(unittest.TestCase, _name, _null_ctx if "Raises" in _name
                or "Warns" in _name or "Logs" in _name else _noop)

ns = {}
try:
    exec(payload["code"], ns)
    if entry_point not in ns:
        raise NameError(entry_point)
except BaseException:
    print(json.dumps(["error:unrunnable"]))
    sys.exit(0)

_target = ns[entry_point]


def _recorder(*args, **kwargs):
    try:
        out = _target(*args, **kwargs)
    except BaseException as exc:
        records.append("error:" + type(exc).__name__)
        raise
    try:
        records.append(repr(out)[:300])
    except BaseException:
        records.append("error:unrepresentable")
    return out


ns[entry_point] = _recorder
# BigCodeBench suites carry utils.BIGCODEBENCH_TEST_DRIVER, which runs every
# test method at module level and then re-raises the first failure. The raise
# happens AFTER the whole suite has run, so the recording is already complete
# and the exception must be swallowed, not reported -- it is a verdict.
try:
    exec(payload["test_code"], ns)
except BaseException:
    pass

if "_bcb_run_tests" not in ns:
    # A bare TestCase with no runner (HumanEval/MBPP-shaped suites): drive it.
    cases = sorted(
        (name, obj)
        for name, obj in ns.items()
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase)
    )
    for case_name, case in cases:
        for method in sorted(m for m in dir(case) if m.startswith("test")):
            records.append("|" + case_name + "." + method)
            try:
                case(method).run(unittest.TestResult())
            except BaseException as exc:
                records.append("error:harness:" + type(exc).__name__)

print(json.dumps(records if records else ["error:testload"]))
"""


def fixture_behavior_signature(
    code: str, entry_point: str, test_code: str, timeout_s: float = 15.0
) -> list[str]:
    """Record what a sample *returns* inside the test suite's own fixtures.

    # TODO: SANDBOX — executes generated code (subprocess-isolated like
    # run_code, but not a security sandbox). Research/local use only.

    Week 11. The alternative to behavior_signature() for benchmarks whose
    callable cannot be invoked bare. On BigCodeBench, probe inputs extracted
    from the test suite are argument lists lifted out of the mock/`setUp`
    context that makes them valid, so the benchmark's *own canonical
    solution* crashes on 64.9% of them (41 of 60 problems crash on every
    probe) — an instrumentation floor, not a property of the samples. Here
    the suite runs unmodified: `setUp`, `@patch` decorators and temporary
    fixtures all apply, and only the entry point is wrapped, to record the
    value it returns on each call.

    Verdict-free by construction: assertions are replaced with no-ops before
    the suite runs (see the runner), so no recorded element is derived from
    a pass/fail outcome and the call sequence does not branch on correctness.
    A sample that raises still stops early — that is a behavioural
    difference, recorded as "error:<Type>", not a leaked label.

    Signatures within a problem may differ in length (a sample that raises
    records fewer calls); callers compare them after padding to the longest,
    for which pad_signatures() is provided.

    Args:
        code: Full runnable candidate (prompt + completion).
        entry_point: Function the suite calls.
        test_code: The benchmark's test suite, run as-is.
        timeout_s: Wall-clock limit for the whole suite.
    Returns:
        Outcome strings in call order, interleaved with "|<Case>.<method>"
        separators so a missing call cannot silently shift the alignment.
        ["error:unrunnable"] if the candidate does not exec or lacks the
        entry point, ["error:testload"] if the suite itself fails to load,
        ["error:timeout"] on expiry.
    """
    payload = json.dumps(
        {"code": code, "entry_point": entry_point, "test_code": test_code}
    )
    try:
        with tempfile.TemporaryDirectory(prefix="rcg_fixture_") as scratch:
            stdout, _ = _run_child(
                [sys.executable, "-c", _FIXTURE_RUNNER], payload, timeout_s, scratch
            )
    except subprocess.TimeoutExpired:
        return ["error:timeout"]
    try:
        records = json.loads(stdout)
    except json.JSONDecodeError:
        return ["error:unrunnable"]
    if not isinstance(records, list):
        return ["error:unrunnable"]
    return [str(r) for r in records]


def pad_signatures(signatures: list[list[str]]) -> list[list[str]]:
    """Pad fixture signatures to a common length with "error:aborted".

    fixture_behavior_signature() returns one element per *executed* call, so
    a sample that raises early yields a shorter list. Padding makes the set
    rectangular for the pairwise metrics in src/uncertainty.py; the filler is
    an error outcome because "the call never happened" is behaviourally an
    abnormal outcome, not agreement (the Week 7 §(h) crash-support trap in
    its original form).
    """
    if not signatures:
        return []
    width = max(len(s) for s in signatures)
    return [s + ["error:aborted"] * (width - len(s)) for s in signatures]


def behavior_signature(
    code: str, entry_point: str, probe_inputs: list[str], timeout_s: float = 5.0
) -> list[str]:
    """Run one sample on every probe input and record what it *does*.

    # TODO: SANDBOX — executes generated code (subprocess-isolated like
    # run_code, but not a security sandbox). Research/local use only.

    The returned signature is a list of outcome strings, one per probe:
    repr() of the returned value, or "error:<ExceptionType>" if the call
    raised, or "error:unrunnable" for every probe if the sample itself
    fails to exec / lacks the entry point, or "error:timeout" for every
    probe if the whole replay exceeds timeout_s (one hung probe poisons
    the batch — coarse, but an infinite-looping sample is behaviorally
    degenerate on every probe anyway, and a single outer timeout keeps
    the subprocess pattern portable).

    Two samples with identical signatures behave identically on the probe
    set regardless of whether either is *correct* — the point of the
    execution-behavior metrics (src/uncertainty.py) that consume these.

    Args:
        code: Full runnable candidate (prompt + completion, as stored in
            results JSON samples).
        entry_point: Function name to call.
        probe_inputs: Argument-list source strings from extract_probe_inputs().
        timeout_s: Wall-clock limit for the whole replay.
    Returns:
        List of outcome strings, same length/order as probe_inputs.
    """
    if not probe_inputs:
        return []
    payload = json.dumps(
        {"code": code, "entry_point": entry_point, "probes": probe_inputs}
    )
    try:
        # Same scratch-directory isolation as run_code: a probe replay executes
        # the candidate, and a candidate that writes or deletes relative paths
        # would otherwise do it in the caller's working directory.
        with tempfile.TemporaryDirectory(prefix="rcg_probe_") as scratch:
            stdout, _ = _run_child(
                [sys.executable, "-c", _BEHAVIOR_RUNNER], payload, timeout_s, scratch
            )
    except subprocess.TimeoutExpired:
        return ["error:timeout"] * len(probe_inputs)
    try:
        outcomes = json.loads(stdout)
    except json.JSONDecodeError:
        return ["error:unrunnable"] * len(probe_inputs)
    if not isinstance(outcomes, list) or len(outcomes) != len(probe_inputs):
        return ["error:unrunnable"] * len(probe_inputs)
    return [str(o) for o in outcomes]


def classify_failure(error_message: str) -> str:
    """Map a run_code() traceback to a Section 9 failure-taxonomy label.

    Args:
        error_message: The traceback string returned by run_code() on
            failure. Empty string (success) maps to "".
    Returns:
        One of the labels in CLAUDE.md Section 9, or "" if error_message
        is empty.
    """
    stripped = error_message.strip()
    if not stripped:
        return ""
    last_line = stripped.splitlines()[-1]
    exception_name = last_line.split(":", 1)[0].strip()
    return _FAILURE_TAXONOMY.get(exception_name, "runtime_error")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. (HumanEval/Codex paper, Eq. 1).

    pass@k is the probability that at least one of k samples drawn (without
    replacement) from a pool of n generated samples passes all tests. Given
    c of the n samples pass, the probability that a random k-subset misses
    every passing sample is C(n-c, k) / C(n, k), so pass@k is one minus that.

    This is preferred over the naive plug-in estimator 1-(1-c/n)^k, which
    is a biased *underestimate* that does not vanish even as n grows past
    5k (see Chen et al., Appendix A / Figure 13) — plugging a noisy
    empirical rate into a nonlinear formula does not preserve unbiasedness.

    The direct combinatorial formula overflows for large n via raw
    factorials, so it is expanded as a product of k terms close to 1:
        C(n-c,k)/C(n,k) = prod_{i=0}^{k-1} (n-c-i)/(n-i)

    Args:
        n: Total number of samples generated per problem.
        c: Number of samples that pass all tests.
        k: The k in pass@k (e.g. 1, 5, 10).
    Returns:
        Estimated probability that at least one of k samples passes.
    """
    assert 0 <= c <= n, f"c={c} must satisfy 0 <= c <= n={n}"
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def mcnemar_mid_p(outcomes_a: list[bool], outcomes_b: list[bool]) -> float:
    """Two-sided mid-p McNemar test for paired binary outcomes.

    Compares two methods evaluated on the same problems using only the
    discordant pairs (problems where exactly one of the two is correct):
    under the null of equal accuracy, the count of "A right, B wrong"
    among the n discordant pairs is Binomial(n, 1/2). The mid-p variant
    (exact conditional p minus the point probability at the observed
    count) is recommended over both the asymptotic and the exact
    conditional test for matched binary pairs (Fagerland, Lydersen &
    Laake, 2013, BMC Medical Research Methodology). McNemar on a shared
    test set is the standard test for comparing two classifiers
    (Dietterich, 1998, Neural Computation). Shared by the selection
    experiment (scripts/evaluate_selection.py) and the planned Week 7
    three-way comparison.

    Args:
        outcomes_a: Per-problem correctness of method A.
        outcomes_b: Per-problem correctness of method B, same order/length.
    Returns:
        Two-sided mid-p value in [0, 1]. Returns 1.0 when there are no
        discordant pairs (the test is undefined; no evidence of any
        difference).
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError(
            f"Paired outcomes must have equal length; got "
            f"{len(outcomes_a)} and {len(outcomes_b)}"
        )
    n01 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a and not b)
    n10 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if b and not a)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    pmf = [comb(n, i) * 0.5**n for i in range(k + 1)]
    # mid-p = 2 * [P(X < k) + 0.5 * P(X = k)] = 2 * CDF(k) - pmf(k)
    return float(min(1.0, 2.0 * sum(pmf) - pmf[k]))


def _mean_ignoring_none(values: list[float | None]) -> float | None:
    """Average a list of optional floats, skipping None entries."""
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def aggregate_results(per_problem_results: list[dict]) -> dict:
    """Compute dataset-level summary statistics from per-problem result dicts.

    Expects each dict to (optionally) contain: "pass_at_1", "pass_at_5",
    "initial_pass_rate", "post_repair_pass_rate", "n_samples",
    "n_initial_pass", "n_post_repair_pass". Missing keys are treated as
    None/absent so this works for uncertainty-only and repair-only runs.

    repair_success_rate is computed *pooled* across all samples (total
    newly-passing / total initially-failing) rather than averaged per
    problem, since a per-problem ratio is undefined whenever a problem has
    zero initial failures.

    Args:
        per_problem_results: One dict per problem (see CLAUDE.md Section 8).
    Returns:
        Dict with mean pass@1, pass@5, initial/post-repair pass rate, and
        repair success rate.
    """
    total_initially_failing = 0
    total_newly_passing = 0
    has_repair_data = False
    for problem in per_problem_results:
        n_samples = problem.get("n_samples")
        n_initial_pass = problem.get("n_initial_pass")
        n_post_repair_pass = problem.get("n_post_repair_pass")
        if n_samples is None or n_initial_pass is None or n_post_repair_pass is None:
            continue
        has_repair_data = True
        initially_failing = n_samples - n_initial_pass
        newly_passing = max(0, n_post_repair_pass - n_initial_pass)
        total_initially_failing += initially_failing
        total_newly_passing += newly_passing

    repair_success_rate = (
        total_newly_passing / total_initially_failing
        if has_repair_data and total_initially_failing > 0
        else (0.0 if has_repair_data else None)
    )

    return {
        "mean_pass_at_1": _mean_ignoring_none(
            [p.get("pass_at_1") for p in per_problem_results]
        ),
        "mean_pass_at_5": _mean_ignoring_none(
            [p.get("pass_at_5") for p in per_problem_results]
        ),
        "mean_initial_pass_rate": _mean_ignoring_none(
            [p.get("initial_pass_rate") for p in per_problem_results]
        ),
        "mean_post_repair_pass_rate": _mean_ignoring_none(
            [p.get("post_repair_pass_rate") for p in per_problem_results]
        ),
        "repair_success_rate": repair_success_rate,
        # Week 11: the fraction of samples that spent their whole generation
        # budget. On HumanEval/MBPP this is near zero and uninteresting; on a
        # longer-solution cell it is a first-class diagnostic, because a
        # truncated sample does not parse and an unparseable sample scores as
        # maximally distant in every diversity metric.
        "mean_truncation_rate": _mean_ignoring_none(
            [p.get("truncation_rate") for p in per_problem_results]
        ),
    }
