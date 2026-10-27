"""Kaggle driver for the Week 8 HumanEval repair pool (joint Objective 3).

NOT production pipeline code (CLAUDE.md FAQ: notebooks/ holds drivers). Paste
one `# %%` block per Kaggle cell. All experiment logic lives in
`scripts/run_iterative_repair_experiment.py`; the analysis afterwards runs
locally on the JSON this brings back.

WHY THIS EXISTS
---------------
Week 7's Smart Repair gate lost to a seeded random control on one 60-candidate
MBPP cohort. Week 8's local re-analysis explained why and hardened the claim:

    scripts/diagnose_gate_target.py   P(pass) ranks initial correctness at
                                      ROC-AUC 0.944 but repairability at 0.518
                                      (p = 0.87, 95% CI [0.298, 0.743])
    scripts/sweep_random_gate.py      82.5% of 200 random draws match or beat
                                      Smart@75 on BOTH added passes and calls

Both rest on ONE cohort, and the allocation AUC's interval is wide because it
has 10 positives. A second, larger cohort is the cheapest thing that narrows
it, and HumanEval is the natural choice: Student A's side is already built.

    source     results/humaneval_uncertainty_merged164_20260711.json
    predictor  results/predictor_humaneval_20260712_215312.json
               (task=sample_level, self_consistency_plus_execution, logistic,
                1640 out-of-fold rows, EvalPlus probes)

Verified: `analyze_smart_repair.py` derives its row count from the source run
rather than hardcoding MBPP's 600, so the whole chain runs on 164 problems
unchanged. What is missing is only the Repair-All trace pool.

WHAT IS BEING GENERATED
-----------------------
164 problems, sample_index=0, pass@1 = 0.0915 on this run => roughly 149
failing coordinates, each repaired once with up to 3 ranked candidates and
stop-on-pass, so about 350-450 Qwen-1.5B calls. That is the only GPU work here.

RUNTIME / HARDWARE
------------------
GPU session (T4 is enough for 1.5B at float16). Internet ON for the first run
so the checkpoint downloads; drop `--offline` accordingly. Budget ~60-100 min.
The run checkpoints after every coordinate, so a session timeout is recoverable
with `--resume` in a second session.

DEVICE PARITY -- READ BEFORE COMPARING ANYTHING
-----------------------------------------------
Every stored Week 4-7 repair trace was generated on CPU/float32. This run is
CUDA/float16. Those are not numerically comparable: Week 7 section 9 measured a
device swap moving pass@1 0.230 -> 0.2167 and 0.495 -> 0.4467 on the capability
ladder, which is why that ladder had to be regenerated device-matched.

Consequence for this cohort: the HumanEval numbers form their OWN cohort and
are compared internally (smart vs random vs repair-all, all generated here).
Do NOT put a number from this run in the same table row as the stored MBPP
CPU numbers without labelling the device.

`--device` / `--dtype` now exist on the experiment script and their resolved
values are persisted into the output config, so a cross-device comparison is
detectable rather than silent.

WHAT TO UPLOAD
--------------
A single Kaggle Dataset containing the repository subset:

    src/
    scripts/
    tests/                        (optional, to run the suite as a smoke check)
    requirements.txt
    results/humaneval_uncertainty_merged164_20260711.json
    results/predictor_humaneval_20260712_215312.json

WHAT TO BRING BACK
------------------
    results/humaneval_week8_repair_all_sample0_qwen15b.json

Then run locally, no GPU needed:

    python scripts/diagnose_gate_target.py \
      --source-results results/humaneval_uncertainty_merged164_20260711.json \
      --predictor-results results/predictor_humaneval_20260712_215312.json \
      --repair-results results/humaneval_week8_repair_all_sample0_qwen15b.json

    python scripts/sweep_random_gate.py \
      --source-results results/humaneval_uncertainty_merged164_20260711.json \
      --predictor-results results/predictor_humaneval_20260712_215312.json \
      --repair-results results/humaneval_week8_repair_all_sample0_qwen15b.json \
      --n-seeds 200

READING THE RESULT
------------------
The pre-registered question is whether MBPP's two findings replicate:

1. allocation ROC-AUC near 0.5 with a NARROWER interval (more positives), and
2. a large fraction of random draws dominating Smart on both axes.

Both replicating => the negative result is a property of the signal, not of one
cohort, and the joint paper can state it as such. Either reversing => the MBPP
result is cohort-specific and the claim must be scoped to it. A reversal is a
real possibility worth naming in advance: HumanEval's cohort has ~149 failures
against MBPP's 54, and its problems are structurally different.
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

if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(DATASET_DIR, WORK)
(WORK / "results").mkdir(exist_ok=True)
print(sorted(p.name for p in (WORK / "results").glob("*.json")))

# %% [markdown]
# ## Cell 2 — dependencies

# %%
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "datasets", "zss"], check=True
)

# %% [markdown]
# ## Cell 3 — confirm the GPU and the resolved runtime
#
# `auto` resolves to cuda/float16 here. Print it, because this value is the one
# that goes into the output config and defines what this cohort may be compared
# against.

# %%
sys.path.insert(0, str(WORK))
import torch  # noqa: E402

from src.generation import resolve_runtime  # noqa: E402

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("resolved runtime:", resolve_runtime("auto", "auto"))

# %% [markdown]
# ## Cell 4 — verify the Student A inputs align before spending any GPU time
#
# `align_oof_scores` raises on a run-id mismatch, a row-count mismatch, or a
# label-order mismatch. Running it first turns a two-hour mistake into a
# two-second one.

# %%
from pathlib import Path as _Path  # noqa: E402

from scripts.analyze_smart_repair import (  # noqa: E402
    align_oof_scores,
    load_json,
    select_cohort,
)

SOURCE = _Path(WORK / "results/humaneval_uncertainty_merged164_20260711.json")
PREDICTOR = _Path(WORK / "results/predictor_humaneval_20260712_215312.json")

scored = align_oof_scores(load_json(SOURCE), load_json(PREDICTOR))
cohort = select_cohort(scored, sample_index=0)
failing = [row for row in cohort if not row["initial_passed"]]
print(f"cohort {len(cohort)} candidates, {len(failing)} failing")
print(f"expected repair calls: {len(failing)} x up to 3 = <= {len(failing) * 3}")

# %% [markdown]
# ## Cell 5 — generate the Repair-All trace pool (the long one)
#
# The profile is frozen to match Week 7's MBPP pool exactly, so the two cohorts
# differ in dataset and device only: temperature 0.8, 1 iteration, 3 ranked
# candidates, `accept` duplicates, stop-on-pass, seed 42.
#
# Re-run this same cell with `--resume` appended if the session times out.

# %%
subprocess.run(
    [
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
        "--stop-on-pass",
        "--seed",
        "42",
        "--model",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "--device",
        "auto",
        "--dtype",
        "auto",
        "--output",
        "results/humaneval_week8_repair_all_sample0_qwen15b.json",
    ],
    cwd=WORK,
    check=True,
)

# %% [markdown]
# ## Cell 6 — run the two analyses in-session as a smoke check
#
# Neither needs a GPU; running them here confirms the pool validates against
# the source before you download it.

# %%
for script, extra in [
    ("scripts/diagnose_gate_target.py", []),
    ("scripts/sweep_random_gate.py", ["--n-seeds", "200"]),
]:
    subprocess.run(
        [
            sys.executable,
            script,
            "--source-results",
            "results/humaneval_uncertainty_merged164_20260711.json",
            "--predictor-results",
            "results/predictor_humaneval_20260712_215312.json",
            "--repair-results",
            "results/humaneval_week8_repair_all_sample0_qwen15b.json",
            *extra,
        ],
        cwd=WORK,
        check=True,
    )

# %% [markdown]
# ## Cell 7 — collect everything for download

# %%
OUT = Path("/kaggle/working/week8_humaneval_repair")
OUT.mkdir(exist_ok=True)
for path in sorted((WORK / "results").glob("*week8*.json")):
    shutil.copy2(path, OUT / path.name)
for path in sorted((WORK / "reports" / "figures").glob("week8_*.png")):
    shutil.copy2(path, OUT / path.name)
print(sorted(p.name for p in OUT.iterdir()))
