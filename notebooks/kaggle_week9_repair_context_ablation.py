"""Kaggle driver for the Week 9 repair-context ablation (Student B).

NOT production pipeline code (CLAUDE.md FAQ: notebooks/ holds drivers). Paste
one `# %%` block per Kaggle cell. All experiment logic lives in
`scripts/run_iterative_repair_experiment.py`; every analysis afterwards runs
locally on the JSONs this brings back.

WHY THIS EXISTS
---------------
Three separate Week 8/9 findings converge on one missing arm.

1. `Repair-All` on the HumanEval cohort repaired 54 of 148 failures (36.5%)
   with 368 Qwen2.5-Coder-1.5B-Instruct calls. Nothing in the project shows
   how many of those 54 the SAME REPAIRER would have fixed by simply being
   asked the question again at the same budget. Olausson et al. (2024) found
   that GPT-3.5/4 self-repair gains largely vanish under a budget-matched
   resampling control; at 1.5B, with execution feedback, that has not been
   tested. Note the construct is not identical: this cohort is CROSS-model
   (the failing programs are codegen-350M's, the repairer is Qwen-1.5B), so
   the control asks whether editing a WEAKER model's broken program beats
   solving the task afresh, not whether a model can fix its own output.

2. `scripts/evaluate_patch_overfitting.py` measured patch-level false
   positives against HumanEval+ at 9.3% [4.0, 19.9] for repaired patches
   against 20.0% [11.2, 33.0] for unrepaired base-passing samples
   (Fisher p = 0.164). The reference population is CodeGen-350M and the
   patches are Qwen-1.5B, so the contrast is confounded by capability in
   exactly the direction of the observed result. A same-model reference arm
   removes that confound.

3. The same script found the *selection* channel structurally unmeasurable:
   the Week 8 pool ran with `--stop-on-pass`, so generation halted at the
   first base-passing candidate and ranking never had to choose between two
   base-passers (0 decidable coordinates). Dropping the flag makes it
   measurable.

One set of runs answers all three, because the three arms differ ONLY in how
much of the failure the generator is shown:

    --context-mode none        task prompt only        (i.i.d. resample)
    --context-mode code_only   + the failing program   (no traceback)
    --context-mode error       + the traceback         (shipped behavior)

Everything else is held fixed: same 148 coordinates, same model, same
temperature, same seed policy, 3 candidates per coordinate, no early stop. So
the arms are budget-matched by construction and the ladder decomposes the gain
into *model capability*, *code context* and *error context*.

WHAT TO UPLOAD
--------------
One Kaggle Dataset containing the repository subset (same shape as the Week 8
upload; add the new files):

    src/                          (MUST be the Week 9 copy: repair.py needs
                                   REPAIR_CONTEXT_MODES)
    scripts/                      (MUST carry --context-mode)
    tests/                        (optional, to run the suite as a smoke check)
    requirements.txt
    results/humaneval_uncertainty_merged164_20260711.json

Upload it as a NEW VERSION of the Week 8 dataset, not a new dataset: Cell 1's
`DATASET_DIR` is the Week 8 slug and a fresh upload would change it. Cell 1
asserts the Week 9 code is present either way.

WHAT TO BRING BACK
------------------
    results/humaneval_week9_arm_resample_sample0_qwen15b.json
    results/humaneval_week9_arm_error_nostop_sample0_qwen15b.json
    results/humaneval_week9_arm_code_only_sample0_qwen15b.json

RUNTIME
-------
Each arm is 148 coordinates x exactly 3 candidate calls = 444 calls, with no
early stop. The Week 8 pool ran 368 calls in 60-100 min on a T4, so budget
75-120 min PER ARM. Each arm checkpoints after every coordinate and resumes
with `--resume`, so arms can be split across sessions; run them in the cell
order below if you have to stop early — the first two carry the headline.

DEVICE PARITY -- READ BEFORE COMPARING ANYTHING
-----------------------------------------------
The Week 8 HumanEval pool was generated on CUDA/float16, and these arms must
match it (`--device auto --dtype auto` resolves to exactly that on a Kaggle
GPU; the resolved values are persisted into every output config). Do NOT
compare any number here against the CPU/float32 MBPP pools: Week 7 section 9
measured a device swap alone moving pass@1 0.230 -> 0.2167.

The `error` arm here is NOT the Week 8 pool re-run: it drops `--stop-on-pass`,
so it spends 3 calls on every coordinate instead of stopping at the first
pass. That is the point (it is what makes the ladder budget-matched and the
selection channel measurable), but it means the two are compared on outcomes,
never on call counts.
"""

# %% [markdown]
# ## Cell 1 — set up the working copy

# %%
import shutil
import subprocess
import sys
from pathlib import Path

# Adjust to the dataset slug you uploaded under.
DATASET_DIR = Path("/kaggle/input/humaneval-repair-pool-project")
WORK = Path("/kaggle/working/project")


def _find_repo_root(base: Path) -> Path:
    """Locate the directory holding src/ and scripts/, however Kaggle nested it.

    Kaggle keeps whatever folder structure the uploaded archive carried, so the
    repository subset can land either directly under the dataset mount or one
    level down inside an archive-named folder. Week 8 lost a session to this.
    """
    candidates = [base, *(p for p in base.rglob("*") if p.is_dir())]
    for candidate in candidates:
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError(f"no src/ + scripts/ pair found under {base}")


SOURCE = _find_repo_root(DATASET_DIR)
print("repository root in dataset:", SOURCE)

if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(SOURCE, WORK)
(WORK / "results").mkdir(exist_ok=True)
print(sorted(p.name for p in (WORK / "results").glob("*.json")))

# The Week 8 dataset's copy of the repo predates --context-mode. Fail here,
# in two seconds, rather than 75 minutes into an arm.
experiment = (WORK / "scripts/run_iterative_repair_experiment.py").read_text()
assert "--context-mode" in experiment, (
    "This dataset holds a pre-Week-9 copy of scripts/. Upload the Week 9 "
    "package (a NEW VERSION of the same dataset keeps this slug valid)."
)
print("context-mode plumbing present")

# %% [markdown]
# ## Cell 2 — dependencies

# %%
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "datasets", "zss"], check=True
)

# %% [markdown]
# ## Cell 3 — confirm the GPU and the resolved runtime
#
# This must print cuda / float16 to match the Week 8 cohort.

# %%
sys.path.insert(0, str(WORK))
import torch  # noqa: E402

from src.generation import resolve_runtime  # noqa: E402

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("resolved runtime:", resolve_runtime("auto", "auto"))

# %% [markdown]
# ## Cell 4 — confirm the cohort is the same 148 coordinates
#
# Cheap guard: the arms are only comparable to each other and to the Week 8
# pool if the failing set is identical.

# %%
import json  # noqa: E402

source = json.loads(
    (WORK / "results/humaneval_uncertainty_merged164_20260711.json").read_text()
)
failing = [
    (problem["task_id"], 0)
    for problem in source["per_problem"]
    if not problem["samples"][0]["passed"]
]
print(f"{len(failing)} failing sample-0 coordinates (expected 148)")
print(f"calls per arm: {len(failing)} x 3 = {len(failing) * 3}")

# %% [markdown]
# ## Cell 4b — render one prompt per arm through the real Qwen tokenizer
#
# The `none` and `code_only` prompts for a chat model have only ever been
# exercised against mocked tokenizers. This renders each arm's actual prompt
# through `apply_chat_template` and prints its length, which is the whole
# untested path — ~30 seconds against 3 x 75 minutes of generation.

# %%
from transformers import AutoTokenizer  # noqa: E402

from src.repair import (  # noqa: E402
    _build_instruction_code_only_repair_prompt,
    _build_instruction_generation_prompt,
    _build_instruction_repair_prompt,
)
from src.utils import load_dataset_split  # noqa: E402

_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
# The run JSON stores prompt_length, not the prompt, so the text comes from
# the dataset — the same source run_iterative_repair_experiment.py reads.
_problems = {row["task_id"]: row for row in load_dataset_split("humaneval")}
_task = next(
    problem for problem in source["per_problem"] if not problem["samples"][0]["passed"]
)
_prompt = _problems[_task["task_id"]]["prompt"]
_failing = _task["samples"][0]
_built = {
    "none": _build_instruction_generation_prompt(_prompt),
    "code_only": _build_instruction_code_only_repair_prompt(_prompt, _failing["code"]),
    "error": _build_instruction_repair_prompt(
        _prompt, _failing["code"], _failing["error_message"]
    ),
}
for _mode, _text in _built.items():
    _ids = _tokenizer.apply_chat_template(
        [{"role": "user", "content": _text}], add_generation_prompt=True, tokenize=True
    )
    print(f"{_mode:>10}: prompt {len(_text)} chars -> {len(_ids)} tokens")
    assert len(_ids) > 0

# The arms must differ in exactly the way they claim to.
assert "```python" not in _built["none"]
assert _failing["code"].strip() not in _built["none"]
assert _failing["code"].strip() in _built["code_only"]
assert "Execution feedback" not in _built["code_only"]
assert "Execution feedback" in _built["error"]
print("prompt ladder verified")

# %% [markdown]
# ## Cell 5 — shared arm runner
#
# One helper so the three arms cannot drift apart in anything but
# `--context-mode`. Append `--resume` by passing `resume=True` after a timeout.

# %%
ARM_OUTPUTS = {
    "none": "results/humaneval_week9_arm_resample_sample0_qwen15b.json",
    "error": "results/humaneval_week9_arm_error_nostop_sample0_qwen15b.json",
    "code_only": "results/humaneval_week9_arm_code_only_sample0_qwen15b.json",
}


def run_arm(context_mode: str, *, resume: bool = False) -> None:
    """Generate one ablation arm over the frozen 148-coordinate cohort."""
    command = [
        sys.executable,
        "scripts/run_iterative_repair_experiment.py",
        "--source-results",
        "results/humaneval_uncertainty_merged164_20260711.json",
        "--dataset",
        "humaneval",
        "--n-failures",
        "164",
        "--sample-index",
        "0",
        "--selection-mode",
        "source_order",
        "--max-iterations",
        "1",
        "--repair-candidates",
        "3",
        "--temperature",
        "0.8",
        "--duplicate-policy",
        "accept",
        "--context-mode",
        context_mode,
        "--seed",
        "42",
        "--model",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "--device",
        "auto",
        "--dtype",
        "auto",
        "--output",
        ARM_OUTPUTS[context_mode],
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=WORK, check=True)


# %% [markdown]
# ## Cell 6 — arm 1 of 3: `none` (the resampling control, ~75-120 min)
#
# The headline arm. Same model, same 3-call budget, no failing program and no
# traceback: whatever it fixes, execution guidance did not fix.

# %%
run_arm("none")

# %% [markdown]
# ## Cell 7 — arm 2 of 3: `error` without early stop (~75-120 min)
#
# The shipped repair prompt at the same fixed budget as arm 1. Also the arm
# that makes the selection channel measurable, because all three candidates
# are generated even after one passes.

# %%
run_arm("error")

# %% [markdown]
# ## Cell 8 — arm 3 of 3: `code_only` (~75-120 min, run last)
#
# Splits arm 2's gain into "seeing the failing program" and "seeing the
# traceback". Drop this arm first if the session budget is tight.

# %%
run_arm("code_only")

# %% [markdown]
# ## Cell 9 — in-session sanity read
#
# No GPU needed. Prints the one number each arm exists to produce.

# %%
for mode, path in ARM_OUTPUTS.items():
    full = WORK / path
    if not full.exists():
        print(f"{mode:>10}: not run")
        continue
    payload = json.loads(full.read_text())
    summary = payload["summary"]
    config = payload["config"]
    print(
        f"{mode:>10}: {summary['successful_repairs']}/"
        f"{summary['completed_failures']} coordinates solved, "
        f"{summary['repair_model_calls']} calls, "
        f"device {config.get('device_resolved')}/{config.get('dtype_resolved')}"
    )

# %% [markdown]
# ## Cell 10 — collect everything for download

# %%
OUT = Path("/kaggle/working/week9_repair_context_ablation")
OUT.mkdir(exist_ok=True)
for path in sorted((WORK / "results").glob("*week9*.json")):
    shutil.copy2(path, OUT / path.name)
print(sorted(p.name for p in OUT.iterdir()))

# %% [markdown]
# ## After the download — the local analyses these arms unlock
#
# ```bash
# # 1. The resampling control: repair vs. i.i.d. resampling at 3 calls each.
# #    (Both arms carry per-candidate pass flags, so pass@k for k = 1..3 is
# #     computed from the stored traces; no GPU.)
#
# # 2. Patch overfitting, de-confounded: the `none` arm is the same-model
# #    reference population the CodeGen siblings could not be.
# python scripts/evaluate_patch_overfitting.py \
#   --repair-pool results/humaneval_week9_arm_error_nostop_sample0_qwen15b.json \
#   --source-results results/humaneval_uncertainty_merged164_20260711.json
#
# # 3. The selection channel, now decidable: coordinates whose trace holds two
# #    or more base-passing candidates that disagree under HumanEval+.
# #    Same script, `selection_channel` block of the output JSON.
# ```
