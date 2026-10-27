"""Kaggle batch driver for the Week 11 BigCodeBench cell.

HEADLESS driver, pushed as a Kaggle *script* kernel with the API:

    kaggle kernels push -p <dir>          # dir holds this file + kernel-metadata.json
    kaggle kernels status melihelibol/bigcodebench-cell-a
    kaggle kernels output melihelibol/bigcodebench-cell-a -p <dir>

It drives `scripts/validate_bigcodebench_cell.py` and `scripts/run_pipeline.py`
only; no experiment logic lives here (CLAUDE.md FAQ: notebooks/ is drivers).

WHY THIS EXISTS
---------------
HumanEval-60 is saturated: the 7B rung solves 59/60, the confidently-wrong
cell is empty at every rung above 350M, and the AST-quartile spread has
collapsed to one or two granules. A further capability rung there measures
nothing, and the project's weakest column is external validity -- two
benchmarks, both small, both function-level, one of them (MBPP) with
three-line canonical solutions.

BigCodeBench is the harder, different-distribution cell: 1,140 problems whose
solutions compose real library calls (pandas 426, numpy 334, matplotlib 309,
sklearn 152), with canonical bodies ~2x longer than HumanEval's. What it buys,
stated as the claim set it can actually support:

  (a) does the correlation structure (AST / token-Jaccard / execution ranking)
      replicate in a harder cell with a different distribution;
  (b) a NON-SATURATED cell for the attenuation ceiling and the N-sweep, whose
      current versions run on cells where pass rate is floor- or
      ceiling-compressed;
  (c) the crash-support decomposition in a much higher-crash regime.

WHAT IT DOES *NOT* BUY, AND THE DRIVER MUST NOT PRETEND OTHERWISE
------------------------------------------------------------------
This is a TRANSFER cell, not a sixth rung of the pre-registered ladder. The
P1-P5 fit is anchored on HumanEval cells and the pre-registration names
HumanEval; feeding BigCodeBench pass@1 into that frozen fit is a
cross-benchmark extrapolation, and re-fitting after the fact is exactly the
move this project has refused four times. Score the correlations, not the
predictions.

THREE PROTOCOL DEVIATIONS, ALL DELIBERATE AND ALL RECORDED
------------------------------------------------------------
1. **dtype is float16 for every rung of this cell.** 7B cannot be float32
   (30.4 GB), and a cell whose top rung differs in dtype from its bottom rung
   repeats the confound that forced the ladder's device-matched regeneration.
   So the BigCodeBench cell is internally fp16-matched and is never compared
   cell-to-cell against the fp32 HumanEval ladder on absolute pass@1.
2. **`--max-new-tokens 512`, not 256.** BigCodeBench canonical bodies are mean
   466 / p90 787 characters. At 256 tokens the top decile is clipped, and a
   truncated sample does not parse -- and an unparseable sample counts as
   *maximally distant* in every diversity metric, which would inflate the very
   correlations this cell exists to test. `truncation_rate` is persisted per
   problem and `mean_truncation_rate` per run; it is a first-class diagnostic
   here, not a footnote.
3. **`--exec-timeout 30`, not 5.** A pandas/sklearn cold import can exceed 5 s
   by itself; the canonical gate measured p95 ~6 s locally.

THE CANONICAL GATE RUNS *IN THIS KERNEL*, AND THAT IS THE POINT
-----------------------------------------------------------------
271 of the 1,140 tests touch the filesystem, 198 use `unittest.mock`, 40 pull
`requests`. Whether a given problem is runnable is a property of the
*environment*, and the environment of record is the one that executes the
samples -- this kernel, not the laptop. Stage A therefore re-runs the whole
gate here and draws the 60-problem subsample from what THIS image can run,
writing the ids it chose. Without that, the cell's pass@1 carries an unknown
floor contribution from environment breakage and "harder cell" quietly becomes
"partly broken cell".

The laptop pass (936/1,140 eligible, 82.1%) is a driver validation, not a
filter: its misses are a long tail of absent packages (faker 39, statsmodels 7,
flask 5, openpyxl 7, tensorflow 3, ...), and intersecting with it would skew
the cell away from precisely the library-heavy problems that make this the
harder distribution.

The gate also proves the unittest driver discriminates: `unittest` collects
failures into a result object instead of raising, so a naive runner reports
success for broken code. Mutants (entry point shadowed to return None, and to
raise) must all fail. Same discipline as the Week 6 pass@1 sanity check, which
found nothing wrong -- and this one already earned itself locally, catching a
no-op mutation (`return X` -> `return None or X`) that had made all six
mutants look like survivors.

STAGE ORDER
-----------
Cheapest-first; every stage publishes the moment it finishes, so a failure at
the end never costs what already succeeded.

One driver, two kernels, selected by the BCB_RUNGS prelude the packaging
script injects -- so the two sessions run byte-identical experiment code and
cannot drift apart in protocol.

  BCB_RUNGS=small (kernel A)
    A. canonical gate + subsample selection (CPU, ~25-35 min). This kernel
       chooses the 60 problems; the ids it writes are shipped in the code
       dataset for the large kernel to reuse.
    B. smoke -- codegen-350M, 3 problems x 2 samples, --dry-run.
    C. codegen-350M, the ladder's floor model.
    D. Qwen2.5-Coder-0.5B.
    E. Qwen2.5-Coder-1.5B -- the mid rung, and the one that says whether the
       cell has usable pass-rate variance at all.

  BCB_RUNGS=large (kernel B)
    F. Qwen2.5-Coder-3B.
    G. Qwen2.5-Coder-7B, device_map=auto.
    It re-uses the small kernel's task ids rather than re-selecting: the rungs
    are only comparable on an identical problem set.

The split is a session-length constraint, not a preference. BigCodeBench
prompts are ~1,112 chars against HumanEval's ~450 and the token budget is
doubled, so the 7B rung's 12,105 s on HumanEval is a floor, not an estimate.

THREE KAGGLE FACTS INHERITED FROM WEEKS 8-9 (each cost a session)
-------------------------------------------------------------------
1. **Mount path.** A UI-attached dataset lands at `/kaggle/input/<slug>/`; an
   API-pushed kernel's at `/kaggle/input/datasets/<owner>/<slug>/<version>/`.
   `find_dataset_root()` searches from `/kaggle/input` rather than guessing.
2. **Accelerator.** The push request carries only `enable_gpu`, so an
   API-pushed GPU kernel gets a **Tesla P100 (sm_60)**, not the UI's T4. The
   image's torch ships no sm_60 kernels and the first `generate()` dies
   asynchronously with "no kernel image is available", presenting as a hang.
   Pin torch 2.6.0+cu124 on capability (6, 0), after removing
   torchvision/torchaudio.
3. **No periodic faulthandler dump.** `dump_traceback_later(..., repeat=True)`
   segfaulted all three Week 9 arms inside Qwen's forward pass.
   `faulthandler.enable()` alone is safe.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

T0 = time.time()


def stamp(message: str) -> None:
    """Print with a monotonic elapsed-time prefix, flushed for Kaggle's log."""
    print(f"[T+{time.time() - T0:7.1f}s] {message}", flush=True)


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/project")
PUBLISH = Path("/kaggle/working")

#: Where this kernel writes the ids it actually selected.
CELL_IDS = "data/bigcodebench_cell_task_ids.txt"

SUBSAMPLE = 60
SEED = 42

#: Which rungs this kernel runs. "small" selects the cell (gate + subsample)
#: and runs the three cheap rungs; "large" reuses the ids the small kernel
#: already chose -- shipped in the code dataset -- and runs 3B and 7B, which do
#: not fit in one session alongside the others. Set by the one-line prelude the
#: packaging script injects; the default keeps this file runnable as-is.
RUNGS = os.environ.get("BCB_RUNGS", "small")
assert RUNGS in {"small", "large"}, f"unknown BCB_RUNGS={RUNGS!r}"

MODELS = {
    "small": [
        ("C-codegen-350M", "Salesforce/codegen-350M-mono", "none"),
        ("D-qwen-0.5B", "Qwen/Qwen2.5-Coder-0.5B", "none"),
        ("E-qwen-1.5B", "Qwen/Qwen2.5-Coder-1.5B", "none"),
    ],
    "large": [
        ("F-qwen-3B", "Qwen/Qwen2.5-Coder-3B", "none"),
        # 7B in float16 is ~15.2 GB against a P100's ~15.9 GB usable. The
        # single-device path materializes the weights in CPU RAM and allocates
        # the same bytes again on the GPU while the CPU copy is live, so
        # fragmentation sinks it before the arithmetic says it should fit.
        # device_map="auto" streams shard-by-shard and spills to CPU instead of
        # raising. Offloading is numerically neutral -- accelerate stages each
        # weight onto the GPU before its matmul -- so it costs wall clock only.
        ("G-qwen-7B", "Qwen/Qwen2.5-Coder-7B", "auto"),
    ],
}[RUNGS]


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


def find_dataset_root() -> Path:
    """Locate the mounted repository subset, however Kaggle nested it."""
    for candidate in _walk_dirs(INPUT_ROOT):
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
        if (candidate / "project.zip").is_file():
            return candidate
    raise FileNotFoundError(
        "no src/+scripts/ pair and no project.zip under /kaggle/input; "
        "attach the code dataset to this kernel"
    )


def free_gb(path: str = "/") -> float:
    """Free space in GB."""
    return shutil.disk_usage(path).free / 1e9


# ---------------------------------------------------------------- setup ----
DATASET = find_dataset_root()
stamp(f"dataset root: {DATASET}")
stamp("dataset contents: " + str(sorted(p.name for p in DATASET.iterdir())))

if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

archive = DATASET / "project.zip"
if archive.exists():
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(WORK)
else:
    shutil.copytree(DATASET, WORK, dirs_exist_ok=True)
(WORK / "results").mkdir(exist_ok=True)
(WORK / "data").mkdir(exist_ok=True)
stamp("working copy: " + str(sorted(p.name for p in WORK.iterdir())))

# Fail now, not an hour in, if the dataset holds a pre-Week-11 copy.
pipeline_source = (WORK / "scripts/run_pipeline.py").read_text()
assert "--max-new-tokens" in pipeline_source, "dataset predates --max-new-tokens"
assert "--exec-timeout" in pipeline_source, "dataset predates --exec-timeout"
assert "bigcodebench" in pipeline_source, "dataset predates the bigcodebench choice"
assert (
    "BIGCODEBENCH_TEST_DRIVER" in (WORK / "src/utils.py").read_text()
), "dataset holds a pre-Week-11 src/ copy: the unittest driver is missing"
stamp("code version verified (bigcodebench cell present)")

stamp("pip install datasets zss accelerate")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "datasets", "zss", "accelerate"],
    check=True,
)
stamp("pip done")

SM60_TORCH = ["torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"]
probe = subprocess.run(
    [
        sys.executable,
        "-c",
        "import torch;"
        "print(torch.cuda.is_available(), torch.cuda.get_device_capability(0)"
        " if torch.cuda.is_available() else None)",
    ],
    capture_output=True,
    text=True,
)
stamp("gpu probe: " + probe.stdout.strip())
if "(6, 0)" in probe.stdout:
    stamp("sm_60 device: reinstalling torch 2.6.0+cu124 (a few minutes)")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "-q",
            "torchvision",
            "torchaudio",
        ],
        check=False,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *SM60_TORCH], check=True
    )
    stamp("torch reinstall done")

sys.path.insert(0, str(WORK))
import torch  # noqa: E402

stamp(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "GPU accelerator is off - cells non-comparable"
stamp("device: " + torch.cuda.get_device_name(0))
stamp(f"capability: {torch.cuda.get_device_capability(0)}")
stamp(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
stamp(f"free disk: {free_gb():.1f} GB")

_a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
stamp(f"fp16 matmul on device ok: {bool(torch.isfinite(_a @ _a).all())}")
del _a
torch.cuda.empty_cache()

# ------------------------------------------------------------- runner ------
# fmt: off
COMMON = [
    "--mode", "uncertainty",
    "--dataset", "bigcodebench",
    "--task-ids-file", CELL_IDS,
    "--n-problems", str(SUBSAMPLE),
    "--n-samples", "10",
    "--temperature", "0.7",
    "--seed", str(SEED),
    "--device", "cuda",
    "--dtype", "float16",
    "--max-new-tokens", "512",
    "--exec-timeout", "30",
    "--no-codebert",
    "--no-tree-edit",
]
# fmt: on

WRAPPER_TEMPLATE = (
    "import faulthandler, runpy; faulthandler.enable(); "
    "runpy.run_path('{script}', run_name='__main__')"
)


def run_script(stage: str, script: str, *extra_args: str) -> None:
    """Run one project script, streaming its output; raise on a non-zero exit."""
    command = [
        sys.executable,
        "-u",
        "-c",
        WRAPPER_TEMPLATE.format(script=script),
        *extra_args,
    ]
    stamp(f"=== stage {stage} ===")
    stamp(f"cmd: {script} " + " ".join(extra_args))
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=WORK,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in process.stdout:
        print(line.rstrip("\n").split("\r")[-1], flush=True)
    process.wait()
    stamp(
        f"=== stage {stage} exit {process.returncode} "
        f"after {time.time() - started:.0f}s ==="
    )
    publish_new_results(stage)
    if process.returncode != 0:
        raise SystemExit(f"stage {stage} failed with exit {process.returncode}")


def publish_new_results(stage: str) -> list[Path]:
    """Copy any results JSON not yet published to /kaggle/working."""
    published = []
    patterns = ("bigcodebench_uncertainty_*.json", "analysis_week11_bcb_gate_*.json")
    for pattern in patterns:
        for path in sorted((WORK / "results").glob(pattern)):
            target = PUBLISH / path.name
            if target.exists():
                continue
            shutil.copy(path, target)
            data = json.loads(path.read_text())
            if "summary" in data and "config" in data:
                config, summary = data["config"], data["summary"]
                stamp(
                    f"[{stage}] published {path.name}: "
                    f"model={config.get('model')} "
                    f"dtype={config.get('dtype_actual')} "
                    f"pass@1={summary.get('mean_pass_at_1')} "
                    f"truncation={summary.get('mean_truncation_rate')}"
                )
            else:
                stamp(f"[{stage}] published {path.name}")
            published.append(target)
    return published


# ------------------------------------------------- stage A: canonical gate --
# Over all 1,140, NOT restricted to the laptop's eligible list. Runnability is
# a property of the environment and the environment of record is the one that
# executes the samples -- this kernel. Intersecting with the laptop's list
# would permanently exclude problems this image can run: locally the misses are
# a long tail of absent packages (faker 39, statsmodels 7, flask 5, openpyxl 7,
# tensorflow 3, ...), several of which ship in the Kaggle image, and dropping
# them would skew the cell away from exactly the library-heavy problems that
# make BigCodeBench the harder distribution. The laptop's run (936/1,140
# eligible, 0/40 mutants surviving, 0 prompt-split failures) validated the
# driver; this one selects.
if RUNGS == "small":
    run_script(
        "A-gate",
        "scripts/validate_bigcodebench_cell.py",
        "--timeout",
        "30",
        "--subsample",
        str(SUBSAMPLE),
        "--n-mutants",
        "20",
        "--seed",
        str(SEED),
        "--task-ids-out",
        CELL_IDS,
    )
else:
    # The large kernel must NOT re-select: the rungs are only comparable on an
    # identical problem set, and re-running the gate here could return a
    # different subsample if this image differs even slightly from the one that
    # ran the small rungs. The ids come from the code dataset, written by the
    # small kernel's gate.
    stamp(f"reusing the small kernel's selection from {CELL_IDS}")

selected = (WORK / CELL_IDS).read_text().split()
stamp(f"cell task ids: {len(selected)} selected, first 5 {selected[:5]}")
assert len(selected) == SUBSAMPLE, f"expected {SUBSAMPLE} ids, got {len(selected)}"

# Provenance, not a formality. The large kernel reads its problem set out of
# the code dataset, and the dataset also carries the laptop gate's ids -- a
# DIFFERENT 60, because the laptop's 204 exclusions were mostly absent packages
# that the Kaggle image has. If the small kernel's ids were never uploaded, the
# large kernel would silently run a different cell and the rungs would not be
# comparable: exactly the silent protocol join that forced the Week 7 section 9
# device-matched regeneration. The packaging step injects the sha256 of the ids
# it shipped; nothing runs until they are byte-identical. (Same pattern as
# Week 9's selection_sha256.)
CELL_SHA256 = os.environ.get("BCB_CELL_SHA256", "")
observed_sha256 = hashlib.sha256((WORK / CELL_IDS).read_bytes()).hexdigest()
stamp(f"cell ids sha256: {observed_sha256}")
if RUNGS == "large":
    assert CELL_SHA256, "BCB_CELL_SHA256 unset; the large kernel cannot verify its cell"
    assert observed_sha256 == CELL_SHA256, (
        f"cell ids sha256 {observed_sha256} != expected {CELL_SHA256}; the dataset "
        "does not carry the small kernel's selection -- re-upload it before "
        "spending GPU time"
    )
for path in sorted((WORK / "data").glob("bigcodebench_*.txt")):
    shutil.copy(path, PUBLISH / path.name)

# --------------------------------------------------------- stage B: smoke --
# fmt: off
run_script(
    "B-smoke",
    "scripts/run_pipeline.py",
    "--mode", "uncertainty",
    "--dataset", "bigcodebench",
    "--task-ids-file", CELL_IDS,
    "--n-problems", "3",
    "--n-samples", "2",
    "--temperature", "0.7",
    "--seed", str(SEED),
    "--device", "cuda",
    "--dtype", "float16",
    "--max-new-tokens", "512",
    "--exec-timeout", "30",
    "--no-codebert",
    "--no-tree-edit",
    "--model", "Salesforce/codegen-350M-mono",
    "--dry-run",
)
# fmt: on

# ----------------------------------------------------------------- rungs ---
for stage, model, device_map in MODELS:
    run_script(
        stage,
        "scripts/run_pipeline.py",
        *COMMON,
        "--device-map",
        device_map,
        "--model",
        model,
    )
    stamp(f"free disk after {stage}: {free_gb():.1f} GB")

stamp("all stages complete")
