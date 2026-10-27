"""Kaggle driver for the Week 7 EvalPlus relabel (Student A).

NOT production pipeline code (CLAUDE.md FAQ: notebooks/ holds drivers). Paste
one `# %%` block per Kaggle cell. All experiment logic lives in
`scripts/rescore_evalplus_labels.py` and `scripts/analyze_capability_sweep.py`.

WHY THIS EXISTS
---------------
Week 7 §9's capability ladder is scored on the *base* HumanEval suite. Stroebl
et al. (2024) show weak verifiers admit false positives — samples that pass the
shipped tests while being wrong — and nothing guarantees that rate is constant
across models. If it grows with capability, the ladder is stretched
non-uniformly and P1/P3's trend claims are contaminated.

This is a RE-ANALYSIS, not a re-run: the five rungs' samples are already
stored, so no model is loaded and no GPU is needed. What it costs is CPU: 5
rungs x 59 problems x 10 samples = ~2,950 executions of HumanEval+ suites
(~80x the base input volume), plus a long-timeout retry pass.

RUNTIME / HARDWARE
------------------
CPU-only session. No GPU, no accelerator. Internet must be ON — the plus
suites are pulled from the `evalplus/humanevalplus` HuggingFace mirror.
Budget ~45-70 min end to end on a Kaggle 4-core CPU box: ~15 min for the
relabel pass, the rest for the long-timeout sensitivity retries and the
capability re-analysis.

WHAT TO UPLOAD
--------------
A single Kaggle Dataset containing the repository subset:

    src/                        (utils.py carries the new plus-label loaders)
    scripts/                    (rescore_evalplus_labels.py + the sweep chain)
    tests/                      (optional, to run the suite as a smoke check)
    requirements.txt
    results/humaneval_uncertainty_20260827_194450.json   codegen-350M
    results/humaneval_uncertainty_20260822_071639.json   Qwen-0.5B
    results/humaneval_uncertainty_20260827_200356.json   deepseek-1.3b
    results/humaneval_uncertainty_20260822_073415.json   Qwen-1.5B
    results/humaneval_uncertainty_20260827_192505.json   Qwen-3B

WHAT TO BRING BACK
------------------
Everything the run writes into `results/`:

    humaneval_evalpluslabel_*.json          (5 re-labelled runs)
    analysis_week7_evalplus_relabel_*.json  (2x2, FP rates, trend)
    analysis_week7_capability_*.json        (the ladder re-scored on plus labels)
    reports/figures/week7_capability_sweep_evalpluslabel.png

READING THE RESULT
------------------
The headline is NOT the pass@1 drop — an absolute drop is bounded by base
pass@1 and is therefore mechanically capability-correlated. The decisive
quantity is `false_positive_rate` = P(plus-fail | base-pass) per rung, read in
capability order. Flat across the ladder => §9's verdicts survive and this is a
footnote. Rising with capability => P1 and P3 must be re-scored on the new
label.
"""

# %% [markdown]
# ## Cell 1 — set up the working copy

# %%
import shutil
import subprocess
import sys
from pathlib import Path

# Adjust to the dataset slug you uploaded under.
DATASET_DIR = Path("/kaggle/input/evalplus-relabel-project")
WORK = Path("/kaggle/working/project")

if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(DATASET_DIR, WORK)
(WORK / "results").mkdir(exist_ok=True)
(WORK / "reports" / "figures").mkdir(parents=True, exist_ok=True)
print(sorted(p.name for p in (WORK / "results").glob("*.json")))

# %% [markdown]
# ## Cell 2 — dependencies (Kaggle already ships numpy/scipy/matplotlib/sklearn)

# %%
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "datasets", "zss"], check=True
)

# %% [markdown]
# ## Cell 3 — smoke check: EvalPlus's own reference solutions must pass
#
# 59/60 is the expected result. HumanEval/32's plus harness in the mirror
# asserts `_poly(*candidate(*inp), inp)`, which unpacks a float and raises
# TypeError even for the canonical solution; the script excludes that task from
# BOTH labels so the comparison stays like-for-like.

# %%
sys.path.insert(0, str(WORK))
from src.evaluation import run_code  # noqa: E402
from src.utils import (  # noqa: E402
    load_evalplus_canonical_solutions,
    load_evalplus_test_code,
)

tests = load_evalplus_test_code("humaneval")
canon = load_evalplus_canonical_solutions("humaneval")
ok_count = 0
for i in range(60):
    task_id = f"HumanEval/{i}"
    ok, err = run_code(canon[task_id], tests[task_id], timeout_s=120.0)
    ok_count += ok
    if not ok:
        print(task_id, err.strip().splitlines()[-1])
print("canonical passing:", ok_count, "/60")

# %% [markdown]
# ## Cell 4 — the relabel pass (the long one)
#
# `--sensitivity-timeout 300` retries every timed-out base-passing sample at
# 300 s. A rescue there would mean the 30 s pass produced a false negative, so
# the false-positive rate would have to be reported at the longer limit.

# %%
RUNS = [
    "results/humaneval_uncertainty_20260827_194450.json",
    "results/humaneval_uncertainty_20260822_071639.json",
    "results/humaneval_uncertainty_20260827_200356.json",
    "results/humaneval_uncertainty_20260822_073415.json",
    "results/humaneval_uncertainty_20260827_192505.json",
]
subprocess.run(
    [
        sys.executable,
        "scripts/rescore_evalplus_labels.py",
        "--results",
        *RUNS,
        "--timeout",
        "30",
        "--workers",
        "8",
        "--sensitivity-timeout",
        "300",
    ],
    cwd=WORK,
    check=True,
)

# %% [markdown]
# ## Cell 5 — re-score the capability ladder on the plus labels
#
# The pre-registered fits stay frozen; only the observed values move. Base and
# plus tables are reported side by side, never merged.

# %%
relabelled = sorted(
    str(p.relative_to(WORK))
    for p in (WORK / "results").glob("humaneval_evalpluslabel_*.json")
)
subprocess.run(
    [
        sys.executable,
        "scripts/analyze_capability_sweep.py",
        "--results",
        *relabelled,
        "--figure",
        "reports/figures/week7_capability_sweep_evalpluslabel.png",
    ],
    cwd=WORK,
    check=True,
)

# %% [markdown]
# ## Cell 6 — collect the outputs

# %%
OUT = Path("/kaggle/working/outputs")
OUT.mkdir(exist_ok=True)
for pattern in (
    "results/humaneval_evalpluslabel_*.json",
    "results/analysis_week7_evalplus_relabel_*.json",
    "results/analysis_week7_capability_*.json",
    "reports/figures/week7_capability_sweep_evalpluslabel.png",
):
    for path in WORK.glob(pattern):
        shutil.copy2(path, OUT / path.name)
print(sorted(p.name for p in OUT.iterdir()))
