"""Kaggle driver: MBPP+ replication of the Week 9 repair-context ladder (Student B).

NOT production pipeline code (CLAUDE.md FAQ: notebooks/ holds drivers). Paste
one `# %%` block per Kaggle cell. All experiment logic lives in
`scripts/run_iterative_repair_experiment.py`; every analysis afterwards runs
locally on the JSONs this brings back.

WHY THIS EXISTS
---------------
The HumanEval ladder is already measured and written up
(`reports/week9_joint_report.md`): over 148 base-failing coordinates at a fixed
three-call budget, repair context is *net harmful* —

    none (resample only)   91/148   61.5%
    code_only              65/148   43.9%
    error (shipped)        54/148   36.5%

with every rung significant on McNemar's mid-p under both the base and the
HumanEval+ labels, and the full-context arm solving ZERO coordinates that no
other arm also solves.

That is one cohort. Week 8's pre-registration split 1-1 across cohorts (the
domination result replicated, the mechanism reversed), which is exactly why a
second cohort is worth its GPU time here. This run decides one thing:

    PRE-REGISTERED READING. If MBPP+ reproduces the ordering with `none` on
    top, the claim is a property of the capability gap and is stated for both
    cohorts. If the arms tie or invert, the claim is scoped to HumanEval in the
    paper and the difference between the cohorts becomes the finding.

Write the outcome down before looking: the HumanEval effect is large
(none - error = +37 coordinates of 148), so even MBPP's smaller cohort has
power against an effect of that size — but not against a small one. See
"POWER" below; that limit is the reason this is a replication and not a
generalization.

WHAT IS HELD FIXED
------------------
The cohort is bit-identical to the Week 7/8 MBPP repair cohort: 54 base-failing
sample-0 coordinates from `results/mbpp_uncertainty_20260712_024909.json` (all
of them MBPP+ members, so every arm is EvalPlus-relabellable afterwards).
Verified locally before writing this file:

    selection_sha256 = 85c09d30cc342f0c8fc54b92a7d6b26563bfebe43b98f9bc92c681064be5b71b

which equals `results/mbpp_week7_repair_all_sample0_qwen15b.json`'s. Cell 4
re-asserts it on Kaggle: if it differs, the arms are not the Week 7 cohort and
nothing downstream is comparable. `scripts/rescore_evalplus_labels.py
--repair-results` also hard-fails if the three arms do not share one initial
pool, so a drift between arms cannot pass silently either.

Same model, temperature, seed policy, three candidates, one repair round, no
early stop. Only `--context-mode` changes.

As on HumanEval, this is CROSS-model repair: the failing programs were generated
by `Salesforce/codegen-350M-mono` and the repairer is
`Qwen2.5-Coder-1.5B-Instruct`. The `code_only`/`error` arms therefore measure
anchoring to a weaker model's broken program, not self-repair, and the write-up
must keep saying so.

POWER -- READ BEFORE INTERPRETING A NULL
----------------------------------------
54 coordinates, not 148. Week 7 repaired 10 of these 54 with the `error` prompt
(18.5%). A null here is therefore weak evidence: the cohort can resolve a
HumanEval-sized gap but not a modest one. Report the discordant counts and the
Wilson intervals, never "no difference".

DEVICE PARITY -- READ BEFORE COMPARING ANYTHING
-----------------------------------------------
`--device auto --dtype auto` resolves to cuda/float16 on a Kaggle GPU and the
resolved values are persisted into every output config (Cell 3 prints them).

The Week 7/8 MBPP pool records NO resolved device — it predates that field and
was generated on CPU/float32. So:

  * these three arms are comparable to EACH OTHER (same session, same device);
  * they are NOT comparable to Week 7's 18.5%, and that number must not appear
    in the same sentence as an arm rate here;
  * they are NOT comparable to the HumanEval arms' rates either — different
    dataset, different base model behind the failure pool. Only the ORDERING
    and the paired tests transfer across cohorts.

Week 7 section 9 measured a device swap alone moving pass@1 0.230 -> 0.2167,
which is the whole reason this paragraph exists.

Which GPU is a separate axis from device/dtype, and the run configs do NOT
record it (they store only `device_resolved` / `dtype_resolved`). Kaggle's UI
lets you pick T4 x2; a kernel pushed through the API cannot -- `enable_gpu` is
the only knob in the push request -- and gets a Tesla P100 (sm_60), which the
image's torch 2.10+cu128 cannot execute at all. See
`kaggle_week9_mbpp_arms_batch.py` for the batch path and its torch pin. Either
accelerator is fine for THIS replication, because the three arms share one
session; it is one more reason the arms are not rate-comparable across cohorts.

WHAT TO UPLOAD
--------------
The Kaggle Dataset `melihelibol/week9-mbpp-context-ablation-code` (Cell 1's
slug; a fresh dataset would change it), containing:

    src/                       (Week 9 copy: repair.py needs REPAIR_CONTEXT_MODES)
    scripts/                   (must carry --context-mode)
    requirements.txt
    results/mbpp_uncertainty_20260712_024909.json     <-- ADD THIS FILE
    results/humaneval_uncertainty_merged164_20260711.json  (already there)

WHAT TO BRING BACK
------------------
    results/mbpp_week9_arm_resample_sample0_qwen15b.json
    results/mbpp_week9_arm_error_nostop_sample0_qwen15b.json
    results/mbpp_week9_arm_code_only_sample0_qwen15b.json

RUNTIME
-------
54 coordinates x exactly 3 calls = 162 calls per arm, roughly 40% of the
HumanEval arms' 444. Budget 30-50 min per arm on a T4, 1.5-2.5 h for all three.
Each arm checkpoints after every coordinate and resumes with `--resume`, so an
interrupted session loses at most one coordinate. Run the cells in order; the
first two carry the headline if the session budget runs out.
"""

# %% [markdown]
# ## Cell 1 — set up the working copy

# %%
import shutil
import subprocess
import sys
from pathlib import Path

# Kaggle nests the mount differently depending on how the kernel was created
# (the UI mounts at /kaggle/input/<slug>/, an API-pushed kernel at
# /kaggle/input/datasets/<owner>/<slug>/<version>/), so search rather than
# hardcode. The dataset is melihelibol/week9-mbpp-context-ablation-code.
INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/project")

SOURCE_RESULTS = "results/mbpp_uncertainty_20260712_024909.json"
EXPECTED_SELECTION_SHA = (
    "85c09d30cc342f0c8fc54b92a7d6b26563bfebe43b98f9bc92c681064be5b71b"
)


def _walk_dirs(base: Path, max_depth: int = 5):
    """Yield directories under base, breadth-first, bounded in depth."""
    frontier = [(base, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        yield current
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except (PermissionError, FileNotFoundError):
            continue
        frontier.extend((child, depth + 1) for child in children)


def _find_repo_root(base: Path) -> Path:
    """Locate the directory holding src/ and scripts/, however Kaggle nested it.

    Both mount layouts are covered by searching from /kaggle/input rather than
    from a fixed slug directory: an API-pushed kernel puts the dataset under
    /kaggle/input/datasets/<owner>/<slug>/<version>/, which a slug-rooted
    lookup misses entirely (that is the FileNotFoundError that killed the
    2026-08-31 sessions).
    """
    for candidate in _walk_dirs(base):
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError(f"no src/ + scripts/ pair found under {base}")


SOURCE = _find_repo_root(INPUT_ROOT)
print("repository root in dataset:", SOURCE)

if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(SOURCE, WORK)
(WORK / "results").mkdir(exist_ok=True)
print(sorted(p.name for p in (WORK / "results").glob("*.json")))

# Fail here, in two seconds, rather than 40 minutes into an arm.
experiment = (WORK / "scripts/run_iterative_repair_experiment.py").read_text()
assert "--context-mode" in experiment, (
    "This dataset holds a pre-Week-9 copy of scripts/. Upload the Week 9 "
    "package (a NEW VERSION of the same dataset keeps this slug valid)."
)
assert (WORK / SOURCE_RESULTS).exists(), (
    f"{SOURCE_RESULTS} is missing from the dataset. It is the MBPP failure "
    "pool this replication is defined over; add it and re-upload."
)
print("context-mode plumbing and MBPP source pool present")

# %% [markdown]
# ## Cell 2 — dependencies

# %%
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "datasets", "zss"], check=True
)

# %% [markdown]
# ## Cell 3 — confirm the GPU and the resolved runtime
#
# Must print cuda / float16. The resolved values are persisted into every arm's
# config; the Week 7/8 MBPP pool has none, which is why its 18.5% is not
# comparable to anything produced here.

# %%
sys.path.insert(0, str(WORK))
import torch  # noqa: E402

from src.generation import resolve_runtime  # noqa: E402

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("resolved runtime:", resolve_runtime("auto", "auto"))

# %% [markdown]
# ## Cell 4 — confirm the cohort is the Week 7/8 MBPP cohort, coordinate for coordinate
#
# The arms are only comparable to each other, and the replication only speaks to
# the Week 7/8 cohort, if this hash matches. Cheap; no model loaded.

# %%
import json  # noqa: E402

from scripts.run_iterative_repair_experiment import (  # noqa: E402
    _selection_sha256,
    collect_failures,
)

source = json.loads((WORK / SOURCE_RESULTS).read_text())
failures = collect_failures(source, 0)
coordinates = [
    {"task_id": f["task_id"], "sample_index": f["sample_index"]} for f in failures[:60]
]
sha = _selection_sha256(coordinates)
print(f"{len(failures)} failing sample-0 coordinates (expected 54)")
print(f"calls per arm: {len(coordinates)} x 3 = {len(coordinates) * 3}")
print("selection sha256:", sha)
assert sha == EXPECTED_SELECTION_SHA, (
    "Selection differs from the Week 7/8 MBPP cohort. Do not run the arms: "
    "nothing downstream would be comparable."
)
print("cohort verified")

# %% [markdown]
# ## Cell 4b — render one prompt per arm through the real Qwen tokenizer
#
# MBPP prompts are shaped differently from HumanEval's (a natural-language
# description plus a signature line, no docstring body), and the `none` /
# `code_only` builders have only ever been exercised on HumanEval prompts.
# ~30 seconds against 3 x 40 minutes of generation.

# %%
from transformers import AutoTokenizer  # noqa: E402

from src.repair import (  # noqa: E402
    _build_instruction_code_only_repair_prompt,
    _build_instruction_generation_prompt,
    _build_instruction_repair_prompt,
)
from src.utils import load_dataset_split  # noqa: E402

_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
_problems = {row["task_id"]: row for row in load_dataset_split("mbpp")}
_failing = failures[0]
_prompt = _problems[_failing["task_id"]]["prompt"]
_built = {
    "none": _build_instruction_generation_prompt(_prompt),
    "code_only": _build_instruction_code_only_repair_prompt(
        _prompt, _failing["initial_code"]
    ),
    "error": _build_instruction_repair_prompt(
        _prompt, _failing["initial_code"], _failing["initial_error_message"]
    ),
}
for _mode, _text in _built.items():
    _ids = _tokenizer.apply_chat_template(
        [{"role": "user", "content": _text}], add_generation_prompt=True, tokenize=True
    )
    print(f"{_mode:>10}: prompt {len(_text)} chars -> {len(_ids)} tokens")
    assert len(_ids) > 0

# The arms must differ in exactly the way they claim to.
assert _failing["initial_code"].strip() not in _built["none"]
assert _failing["initial_code"].strip() in _built["code_only"]
assert "Execution feedback" not in _built["code_only"]
assert "Execution feedback" in _built["error"]
print("prompt ladder verified on an MBPP prompt")

# %% [markdown]
# ## Cell 5 — shared arm runner
#
# One helper so the three arms cannot drift apart in anything but
# `--context-mode`. Note what is NOT here: `--stop-on-pass`. Leaving it off is
# what makes the arms budget-matched (three calls on every coordinate, 162 per
# arm) and matches the HumanEval arms' protocol. Append `--resume` by passing
# `resume=True` after a timeout.

# %%
ARM_OUTPUTS = {
    "none": "results/mbpp_week9_arm_resample_sample0_qwen15b.json",
    "error": "results/mbpp_week9_arm_error_nostop_sample0_qwen15b.json",
    "code_only": "results/mbpp_week9_arm_code_only_sample0_qwen15b.json",
}


def run_arm(context_mode: str, *, resume: bool = False) -> None:
    """Generate one ablation arm over the frozen 54-coordinate MBPP cohort."""
    command = [
        sys.executable,
        "scripts/run_iterative_repair_experiment.py",
        "--source-results",
        SOURCE_RESULTS,
        "--dataset",
        "mbpp",
        "--n-failures",
        "60",
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
# ## Cell 6 — arm 1 of 3: `none` (the resampling control, ~30-50 min)
#
# The arm that carries the HumanEval headline. Same model, same 3-call budget,
# no failing program and no traceback.

# %%
run_arm("none")

# %% [markdown]
# ## Cell 7 — arm 2 of 3: `error` without early stop (~30-50 min)
#
# The shipped repair prompt at the same fixed budget as arm 1. Outcome-
# comparable to Week 7's MBPP pool coordinate by coordinate; NOT rate-comparable
# to it (that pool records no resolved device).

# %%
run_arm("error")

# %% [markdown]
# ## Cell 8 — arm 3 of 3: `code_only` (~30-50 min, run last)
#
# Splits arm 2's result into "seeing the failing program" and "seeing the
# traceback". Drop this arm first if the session budget is tight — the
# `none` vs `error` contrast is the pre-registered one.

# %%
run_arm("code_only")

# %% [markdown]
# ## Cell 9 — in-session sanity read
#
# No GPU needed. Prints the one number each arm exists to produce, plus the
# resolved device that makes them comparable.

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
        f"stop_on_pass={config.get('stop_on_pass')}, "
        f"device {config.get('device_resolved')}/{config.get('dtype_resolved')}"
    )

# %% [markdown]
# ## Cell 10 — collect everything for download

# %%
OUT = Path("/kaggle/working/week9_mbpp_context_ablation")
OUT.mkdir(exist_ok=True)
for path in sorted((WORK / "results").glob("mbpp_week9*.json")):
    shutil.copy2(path, OUT / path.name)
print(sorted(p.name for p in OUT.iterdir()))

# %% [markdown]
# ## After the download — the local analyses, in order
#
# ```bash
# # 1. MBPP+ relabel (CPU, ~5 min). Scores the shared initial pool once and
# #    every arm's patches against the MBPP+ suite, and reports the per-arm
# #    false-positive rate the ladder has to survive.
# python scripts/rescore_evalplus_labels.py \
#   --dataset mbpp --sensitivity-timeout 300 \
#   --repair-results \
#     results/mbpp_week9_arm_resample_sample0_qwen15b.json \
#     results/mbpp_week9_arm_code_only_sample0_qwen15b.json \
#     results/mbpp_week9_arm_error_nostop_sample0_qwen15b.json
#
# # 2. The ladder itself, base label (CPU, seconds). No --week8-results: the
# #    Week 7 MBPP pool ran WITH --stop-on-pass and on an unrecorded device, so
# #    the bit-for-bit reproduction check does not apply to this cohort.
# python scripts/analyze_context_mode_ladder.py --dataset mbpp --label base \
#   --arms \
#     results/mbpp_week9_arm_resample_sample0_qwen15b.json \
#     results/mbpp_week9_arm_code_only_sample0_qwen15b.json \
#     results/mbpp_week9_arm_error_nostop_sample0_qwen15b.json
#
# # 3. The ladder under MBPP+ labels, over the *_evalpluslabel_*.json files
# #    step 1 wrote (substitute the real timestamps).
# python scripts/analyze_context_mode_ladder.py --dataset mbpp --label plus \
#   --arms results/mbpp_week9_arm_*_evalpluslabel_*.json
# ```
#
# Then update `reports/week9_joint_report.md` section 11 item 1 and
# `reports/overleaf/main_joint.tex` section `sec:contextladder` with whichever
# branch of the pre-registration fired.
