"""Kaggle driver for the Week 7 capability sweep (Student A).

NOT production pipeline code (CLAUDE.md FAQ: notebooks/ is for exploration and
drivers). Paste one `# %%` block per Kaggle cell. It only drives
`scripts/run_pipeline.py`; no experiment logic lives here.

WHY THIS EXISTS
---------------
Every finding in Weeks 2-7 rests on two models, a 350M and a 1.3B. The Week 7
report names this first in its own limitations: "Seven cells, one metric suite,
two models. [...] The capability sweep that would turn this into a study is not
done." Two points cannot separate a trend from a coincidence, and several of the
project's most interesting claims are explicitly about *where on the capability
scale* something happens:

  * the Semantic Consensus Hypothesis holds at 350M, with a 23.5%
    confidently-wrong residual;
  * the crash-support method trap "is invisible at the >= 7B scale where the
    concurrent literature works";
  * the execution channel's advantage is "about ranking quality where the model
    is competent".

All three are capability claims stated from two data points. This sweep adds two
more.

THE LADDER
----------
Four models on the identical cell (HumanEval 0..59, N=10, temperature 0.7,
seed 42), two of which already exist and are NOT re-generated here:

  | model                          | params | status                       |
  |--------------------------------|--------|------------------------------|
  | Salesforce/codegen-350M-mono   | 0.35B  | run 2026-08-22, pass@1 0.230 |
  | Qwen/Qwen2.5-Coder-0.5B        | 0.5B   | run 2026-08-22, pass@1 0.433 |
  | deepseek-ai/deepseek-coder-1.3b-base | 1.3B | run 2026-08-22, 0.495   |
  | Qwen/Qwen2.5-Coder-1.5B        | 1.5B   | run 2026-08-22, pass@1 0.595 |
  | Qwen/Qwen2.5-Coder-3B          | 3.1B   | CELL 6 (second pass)         |
  | Qwen/Qwen2.5-Coder-7B          | 7.6B   | CELL 8, needs sharding       |

SECOND PASS (2026-08-22, after the first four rungs were scored)
----------------------------------------------------------------
Two additions, for two different reasons:

  * **Cell 6 — the 3B rung.** Four points fitted two pre-registered lines and
    falsified two other predictions (P2's band, P4's parse-failure threshold);
    the post-hoc reading of P2's failure is a *ceiling* effect, visible as two
    quartiles tied at frac_solved = 1.0 on the 1.5B rung. A ceiling claim made
    from one rung is an anecdote. 3B is the cheapest rung that tests it, and it
    is the largest Qwen that still fits a single T4 **in float32**, so it joins
    the ladder without a dtype deviation.
  * **Cell 7 — device parity.** The two stored rungs (codegen, deepseek) were
    generated on CPU before `--device` existed; both Qwen rungs ran on CUDA.
    `generation._resolve_device` documents that the same seed draws different
    tokens per device, so the ladder currently carries a device confound *inside
    the comparison*. Re-generating those two rungs here, under the identical
    `COMMON` block, removes it. Cheap (~1 h for both) and it deletes a
    limitations bullet rather than merely footnoting it.

Do NOT re-run Cells 4-5: those two JSONs exist and are already scored. Run
Cells 1-3, then 6 and 7.

Two deliberate choices:

1. **Base models only.** Student B's work uses Qwen2.5-Coder-*-Instruct; those
   are the wrong comparison here, because instruction tuning would be confounded
   with capability. Every rung of this ladder is a base completion model, as the
   two stored cells are.
2. **The x-axis is measured pass@1, not parameter count.** Qwen2.5-Coder-0.5B is
   three years newer than codegen-350M and will very likely outrank it despite
   being a similar size. That is a feature: it decouples capability from scale,
   so the resulting claims are about how *good* the model is, not how big.

The pre-registered predictions are in
`reports/week7_capability_sweep_preregistration.md`, committed before this
notebook was run. Read them first; do not re-derive them afterwards.

COST
----
N = 10 (not the N-sweep's 50), so each run is a fifth of an N-sweep cell:
roughly 20-40 minutes per model on a T4 at ~2-4 s/sample. Measured on the first
pass: 0.5B and 1.5B together finished well inside one session. The second pass
(Cells 6 + 7) is ~1.5-2 h for 3B plus ~1 h for the two parity re-runs, so budget
a ~3 h commit run.
"""

# %% [markdown]
# # Week 7 capability sweep - generation runs
#
# Settings -> **Accelerator: GPU T4 x2**, **Internet: On**.
# Run with **Save Version -> Save & Run All (Commit)**.
#
# Cells 1-3 are identical to `kaggle_n_sweep.py`; if that notebook is already
# set up, copy those cells across unchanged.

# %%
# --- Cell 1: environment check -------------------------------------------
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then re-run.")
print("device:", torch.cuda.get_device_name(0))
print(
    "vram:",
    round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    "GB",
)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zss"], check=True)
print("zss installed")

# %%
# --- Cell 2: stage the project code --------------------------------------
# Path is discovered, not hardcoded: Kaggle slugifies dataset titles and nests
# wrapped zips, so guessing it is what breaks.
INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/project")
MARKERS = ("src/generation.py", "scripts/run_pipeline.py")


def find_project_root(root: Path) -> Path | None:
    """Return the first directory under `root` holding every MARKERS path."""
    if not root.exists():
        return None
    for candidate in root.rglob("generation.py"):
        base = candidate.parent.parent
        if all((base / marker).is_file() for marker in MARKERS):
            return base
    return None


SRC = find_project_root(INPUT_ROOT)
if SRC is None:
    print("Could not locate the project under /kaggle/input.\n")
    if not INPUT_ROOT.exists() or not any(INPUT_ROOT.iterdir()):
        print(
            "/kaggle/input is EMPTY -> no dataset is attached to this notebook.\n"
            "Fix: right-hand panel -> '+ Add Input' -> Datasets -> Your Datasets\n"
            "     -> click '+' next to the uploaded dataset. Then re-run."
        )
    else:
        print("What IS mounted (first 40 entries):")
        for i, path in enumerate(sorted(INPUT_ROOT.rglob("*"))):
            if i >= 40:
                print("  ...")
                break
            print("  ", path)
    raise SystemExit("project not found under /kaggle/input — see the listing above")

print("found project at:", SRC)
if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(SRC, WORK)
(WORK / "results").mkdir(exist_ok=True)
os.chdir(WORK)
print("cwd:", Path.cwd())

# %%
# --- Cell 3: the streaming runner ----------------------------------------
# run_pipeline.py reports progress with tqdm, which redraws using "\r"; a
# notebook rendering a piped subprocess never shows those redraws, so a long
# run looks frozen at the last "\n"-terminated line. Streaming the child's
# output with an explicit flush is what makes its per-problem
# "[i/N] ... eta H:MM:SS" heartbeats visible as they happen.


def run_pipeline_streaming(*extra_args: str) -> None:
    """Run run_pipeline.py, echoing its output line by line as it arrives."""
    command = [sys.executable, "-u", "scripts/run_pipeline.py", *extra_args]
    print(">>", " ".join(command), flush=True)
    started = time.time()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in process.stdout:
        print(line.rstrip("\n").split("\r")[-1], flush=True)
    process.wait()
    print(
        f"<< exit {process.returncode} after {time.time() - started:.0f}s", flush=True
    )
    if process.returncode != 0:
        raise SystemExit(f"run_pipeline.py failed with exit {process.returncode}")


# Identical to the two stored cells in every respect except --model. Do not
# change any of these: the whole point is that the four cells differ only in
# the model, so any difference in the results is attributable to capability.
COMMON = [
    "--mode",
    "uncertainty",
    "--dataset",
    "humaneval",
    "--n-problems",
    "60",
    "--problem-offset",
    "0",
    "--n-samples",
    "10",
    "--temperature",
    "0.7",
    "--seed",
    "42",
    "--device",
    "cuda",
    "--dtype",
    "float32",
    "--no-codebert",
    # Zhang-Shasha over O(N^2) pairs costs ~99 s/problem at N=50 against
    # ~0.3 s for every other metric combined, and no analysis downstream of
    # these runs reads it (estimate_reliability.py and analyze_n_sweep.py
    # exclude it by default).
    "--no-tree-edit",
]
print("runner ready")

# %%
# --- Cell 4: rung 2 - Qwen2.5-Coder-0.5B (base) --------------------------
run_pipeline_streaming(*COMMON, "--model", "Qwen/Qwen2.5-Coder-0.5B")

# %%
# --- Cell 5: rung 4 - Qwen2.5-Coder-1.5B (base) --------------------------
run_pipeline_streaming(*COMMON, "--model", "Qwen/Qwen2.5-Coder-1.5B")

# %%
# --- Cell 6: rung 5 - Qwen2.5-Coder-3B (base) ----------------------------
# 3.09B params x 4 bytes = ~12.4 GB in float32, against a T4's ~15 GB usable.
# It fits, and fitting *in float32* is the whole point: every other rung on
# this ladder is float32, so this one joins without a dtype deviation to
# caveat. Do not "optimise" it to float16 — that would trade the rung's
# comparability for runtime this notebook does not need.
run_pipeline_streaming(*COMMON, "--model", "Qwen/Qwen2.5-Coder-3B")

# %%
# --- Cell 7: device parity for the two stored rungs ----------------------
# Same COMMON block, so these differ from the stored CPU-era runs in exactly
# one respect: the device. That is the point — the ladder should not mix
# devices across rungs when the project's own code documents that device
# changes the sampled tokens at a fixed seed.
#
# Keep the stored CPU JSONs. They are what every Week 2-7 number was computed
# from, and the paper's reproduction claims point at them; these two are the
# device-matched ladder inputs, not replacements. Scoring should use one set or
# the other consistently, never a mix.
for parity_model in (
    "Salesforce/codegen-350M-mono",
    "deepseek-ai/deepseek-coder-1.3b-base",
):
    run_pipeline_streaming(*COMMON, "--model", parity_model)

# %%
# --- Cell 8 (OPTIONAL): rung 6 - Qwen2.5-Coder-7B (base) -----------------
# Only worth attempting after Cell 6 succeeds, and NOT with this notebook's
# loader as written: load_model() ends in a plain `model.to(device)`, which
# is single-device. 7B is ~30.4 GB in float32 and ~15.2 GB in float16, so
# neither fits one T4; the 2x T4 allocation only helps once load_model()
# accepts device_map="auto". Until that flag exists, this cell will OOM.
# Historical note, kept because it was wrong: the original comment read 7B in
# float32 is ~28 GB and cannot fit, while float16 is ~14.2 GB against a T4's
# ~15 GB usable. It may still OOM once activations are allocated.
#
# The scientific value is high enough to try, because >= 7B is exactly the
# scale the concurrent literature works at. Note that the first pass already
# falsified P4 — parse-failure did NOT fall with capability; every rung of the
# device-matched ladder sits at 0.015-0.033 against a pre-registered < 0.005.
# Cell 7's parity runs did NOT establish a mechanism: deepseek-1.3b's CPU
# baseline is a single unparseable problem, so its 0.0017 -> 0.0217 move is a
# count of one, and the codegen control moves only 0.0133 -> 0.0150.
# A 7B rung tests whether that instability persists at the scale the concurrent
# literature works at.
#
# Note the dtype difference from the other rungs and record it: float16 is a
# real deviation from the ladder's protocol, not a free choice, and any
# conclusion drawn from this rung must say so.
RUN_7B = False  # flip to True to attempt it

if RUN_7B:
    run_pipeline_streaming(
        *[a if a != "float32" else "float16" for a in COMMON],
        "--model",
        "Qwen/Qwen2.5-Coder-7B",
    )
else:
    print("skipped (set RUN_7B = True to attempt the 7B rung)")

# %%
# --- Cell 9: collect outputs for download --------------------------------
import json

out = Path("/kaggle/working")
produced = sorted((WORK / "results").glob("humaneval_uncertainty_*.json"))
for f in produced:
    shutil.copy(f, out / f.name)
    d = json.loads(f.read_text())
    cfg, summ = d["config"], d["summary"]
    print(
        f"{f.name}  model={cfg['model'].split('/')[-1]:<26} "
        f"N={cfg['n_samples']}  problems={cfg['n_problems']}  "
        f"dtype={cfg['dtype_resolved']}  pass@1={summ['mean_pass_at_1']:.4f}"
    )

shutil.rmtree(WORK)
print("\nDownload these from the Output tab into results/, then locally run:")
print("  python scripts/analyze_capability_sweep.py --results results/*.json")
