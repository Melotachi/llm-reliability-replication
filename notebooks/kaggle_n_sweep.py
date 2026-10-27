"""Kaggle driver for the Week 7 N-sweep generation runs (Student A).

NOT production pipeline code (CLAUDE.md FAQ: notebooks/ is for exploration and
drivers). This file is a cell-by-cell script meant to be pasted into a Kaggle
notebook, one `# %%` block per cell. It only *drives* `scripts/run_pipeline.py`;
no experiment logic lives here.

WHY THIS EXISTS
---------------
The Week 7 report establishes a hard attenuation ceiling on |r| of ~0.83-0.90
at N = 10 samples per problem, and that several metrics already sit at 85-94%
of it. The sweep tests the ceiling *as a theory* rather than as a caveat: it
makes a numeric prediction before the data exists.

  Reliability at N samples follows Spearman-Brown from the single-sample
  reliability r_1:            r_N = N * r_1 / (1 + (N - 1) * r_1)

  Measured at N = 10 on cg/HE60:  r_yy(pass_rate) = 0.902  ->  r_1 = 0.532
                                  r_xx(ast_node)  = 0.823  ->  r_1 = 0.343

  Predicted at N = 50:            r_yy = 0.983,  r_xx = 0.963
                                  ceiling |r| = sqrt(r_xx * r_yy) = 0.973

  cg/HE60 ast_node currently observes |r| = 0.624 = 72.5% of its 0.861 ceiling.
  If "% of ceiling" is the stable quantity (i.e. it estimates the true
  correlation), the N = 50 run must observe |r| ~= 0.725 * 0.973 = 0.705.
  For ds/HE60 ast_node (85.0% of ceiling) the prediction is ~0.827.

  PRE-REGISTER THESE TWO NUMBERS BEFORE RUNNING. A prediction written after
  seeing the result is worth nothing; written before, it converts Week 7's
  ceiling section from a limitation into a validated measurement model.

WHAT THIS NOTEBOOK PRODUCES
---------------------------
Two results JSONs, same schema as every other run in results/ (CLAUDE.md
Section 8), differing from the existing HE60 cells only in --n-samples:

  cg/HE60  N=50  Salesforce/codegen-350M-mono
  ds/HE60  N=50  deepseek-ai/deepseek-coder-1.3b-base

Analysis happens locally afterwards (scripts/analyze_n_sweep.py subsamples
these pools at N = 5, 10, 20, 30, 40, 50). Nothing here needs a GPU except the
generation itself.

REPRODUCIBILITY NOTE (CLAUDE.md Constraint #4)
----------------------------------------------
`multi_sample_generation` seeds sample i with `seed + i`, so an N=50 run with
seed 42 is a strict superset of an N=10 run with seed 42 *on the same device
and dtype*. Kaggle's GPU is neither, so these runs will not be bit-identical to
the stored CPU runs. That is fine and expected: the sweep is internally
consistent (one pool of 50, subsampled), and the N=10 point serves as its own
check — it should land near the stored -0.624 / -0.740, not exactly on it.
`--dtype float32` is pinned rather than accepting the CUDA float16 default so
that dtype is not an extra difference on top of the device change.
"""

# %% [markdown]
# # Week 7 N-sweep - generation runs
#
# Before running: notebook settings -> **Accelerator: GPU T4 x2** (or P100) and
# **Internet: On** (needed for the HuggingFace model/dataset downloads).
#
# Use **Save Version -> Save & Run All (Commit)** so the job survives closing
# the browser tab.

# %%
# --- Cell 1: environment check -------------------------------------------
# Fail loudly and early if the accelerator or internet is not actually on.
import subprocess
import sys

import torch

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then re-run.")
print("device:", torch.cuda.get_device_name(0))

# zss (Zhang-Shasha tree edit distance) is the only project dependency Kaggle
# does not ship. transformers/datasets/torch/sklearn are preinstalled.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zss"], check=True)
print("zss installed")

# %%
# --- Cell 2: stage the project code --------------------------------------
# The repo is private, so it is uploaded as a private Kaggle Dataset rather
# than cloned. Copy it out of the read-only /kaggle/input mount so that
# run_pipeline.py can write results/ next to it.
#
# The mount path is DISCOVERED, not hardcoded: Kaggle slugifies the dataset
# title (and nests the contents when the uploaded zip had a top-level folder),
# so guessing the path is what breaks. Search for the marker files instead and
# print the actual tree when the search fails.
import os
import shutil
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/project")
# A directory is the project root iff it contains both of these.
MARKERS = ("src/generation.py", "scripts/run_pipeline.py")


def find_project_root(root: Path) -> Path | None:
    """Return the first directory under `root` holding every MARKERS path."""
    if not root.exists():
        return None
    # rglob on the marker keeps this cheap regardless of how deep Kaggle nested
    # the upload; datasets here are a few dozen files.
    for candidate in root.rglob("generation.py"):
        base = candidate.parent.parent  # .../<root>/src/generation.py -> <root>
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
            "     -> click '+' next to llm-reliability-pipeline. Then re-run."
        )
    else:
        print("What IS mounted (first 40 entries):")
        for i, path in enumerate(sorted(INPUT_ROOT.rglob("*"))):
            if i >= 40:
                print("  ...")
                break
            print("  ", path)
        print(
            "\nThe dataset is attached but does not contain "
            "src/generation.py + scripts/run_pipeline.py.\n"
            "Most likely the zip was uploaded with an extra wrapping folder, or "
            "a different zip was uploaded."
        )
    raise SystemExit("project not found under /kaggle/input — see the listing above")

print("found project at:", SRC)

if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(SRC, WORK)
(WORK / "results").mkdir(exist_ok=True)
os.chdir(WORK)
print("cwd:", Path.cwd())
print("staged:", sorted(p.name for p in WORK.iterdir()))

# %%
# --- Cell 3: timing probe ------------------------------------------------
# Decide N from a measurement, not a guess. Kaggle GPU sessions cap at ~9h;
# this prints the projected wall-clock for both planned runs.
import time

from src.generation import generate_code, load_model
from src.utils import load_dataset_split, set_all_seeds

problems = load_dataset_split("humaneval")[:3]

for model_name in [
    "Salesforce/codegen-350M-mono",
    "deepseek-ai/deepseek-coder-1.3b-base",
]:
    tok, mdl = load_model(model_name, device="cuda", dtype="float32")
    times = []
    for p in problems:
        set_all_seeds(42)
        t0 = time.time()
        generate_code(p["prompt"], tok, mdl, temperature=0.7)
        times.append(time.time() - t0)
    per_sample = sum(times) / len(times)
    print(
        f"{model_name.split('/')[-1]:38s} {per_sample:5.2f}s/sample "
        f"| 60 problems x 50 = {per_sample * 60 * 50 / 3600:4.1f}h"
        f" | x100 = {per_sample * 60 * 100 / 3600:4.1f}h"
    )
    del mdl
    torch.cuda.empty_cache()

# %%
# --- Cell 3b: the streaming runner ---------------------------------------
# Why this exists rather than a plain subprocess.run: run_pipeline.py reports
# progress with tqdm, which redraws using "\r". A notebook rendering a piped
# subprocess never shows those redraws, so an hours-long run appears frozen at
# whatever line last ended in "\n" -- which is alarming and indistinguishable
# from an actual hang. Two fixes, both needed:
#   * PYTHONUNBUFFERED, so the child does not sit on a 4-8 KB stdout buffer;
#   * read the child's output and echo it with an explicit flush, so each of
#     run_pipeline's per-problem "[i/N] ... eta H:MM:SS" heartbeats appears as
#     it happens.
import time


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
        # tqdm's carriage returns would otherwise glue several updates onto one
        # unreadable line; keep only the last segment of each.
        print(line.rstrip("\n").split("\r")[-1], flush=True)
    process.wait()
    print(
        f"<< exit {process.returncode} after {time.time() - started:.0f}s", flush=True
    )
    if process.returncode != 0:
        raise SystemExit(f"run_pipeline.py failed with exit {process.returncode}")


COMMON = [
    "--mode",
    "uncertainty",
    "--dataset",
    "humaneval",
    "--n-problems",
    "60",
    "--problem-offset",
    "0",
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
# --- Cell 4: run A - codegen-350M, HumanEval 0..59, N=50 -----------------
# Same cell as the stored `humaneval_uncertainty_20260710_014920` in every
# respect except --n-samples (10 -> 50). --no-codebert because the reliability
# analysis excludes the embedding families by default (they are the two
# expensive ones and get recomputed 2 x n_permutations times).
N_SAMPLES = 50  # raise to 100 only if Cell 3 says both runs still fit in ~8h

run_pipeline_streaming(
    *COMMON,
    "--n-samples",
    str(N_SAMPLES),
    "--model",
    "Salesforce/codegen-350M-mono",
)

# %%
# --- Cell 5: run B - deepseek-coder-1.3b, HumanEval 0..59, N=50 ----------
run_pipeline_streaming(
    *COMMON,
    "--n-samples",
    str(N_SAMPLES),
    "--model",
    "deepseek-ai/deepseek-coder-1.3b-base",
)

# %%
# --- Cell 6: collect outputs for download --------------------------------
# Anything left in /kaggle/working appears under the notebook's Output tab.
import json

out = Path("/kaggle/working")
produced = sorted((WORK / "results").glob("humaneval_uncertainty_*.json"))
for f in produced:
    shutil.copy(f, out / f.name)
    d = json.loads(f.read_text())
    cfg, summ = d["config"], d["summary"]
    print(
        f"{f.name}  model={cfg['model'].split('/')[-1]}  "
        f"n_samples={cfg['n_samples']}  n_problems={cfg['n_problems']}  "
        f"dtype={cfg['dtype_resolved']}  pass@1={summ['mean_pass_at_1']:.4f}"
    )

# Drop the staged copy so the Output tab contains only the results JSONs.
shutil.rmtree(WORK)
print("\nDownload these from the notebook's Output tab, then locally run:")
print("  python scripts/analyze_n_sweep.py --results results/<file>.json")
