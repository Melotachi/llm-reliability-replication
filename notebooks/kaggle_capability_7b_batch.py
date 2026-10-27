"""Kaggle batch driver for the >=7B rung of the Week 7 capability ladder.

HEADLESS driver, pushed as a Kaggle *script* kernel with the API:

    kaggle kernels push -p <dir>          # dir holds this file + kernel-metadata.json
    kaggle kernels status melihelibol/capability-7b-rung
    kaggle kernels output melihelibol/capability-7b-rung -p <dir>

It only drives `scripts/run_pipeline.py`; no experiment logic lives here
(CLAUDE.md FAQ: notebooks/ is for exploration and drivers).

WHY THIS EXISTS
---------------
`notebooks/kaggle_capability_sweep.py` Cell 8 is the 7B rung, and it has sat
disabled (`RUN_7B = False`) since it was written, for a stated reason:

    "NOT with this notebook's loader as written: load_model() ends in a plain
    `model.to(device)`, which is single-device. [...] Until that flag exists,
    this cell will OOM."

That flag now exists — `load_model(device_map=...)` / `run_pipeline.py
--device-map` — so the rung is runnable. It matters because three of the
project's live claims are explicitly about the >=7B scale the concurrent
literature works at, and all three are currently stated from below it:

  * Week 7 (h): the crash-support method trap "is invisible at the >= 7B scale
    the concurrent literature works at";
  * Week 7 §9: P3 is FALSIFIED at the 3B rung (dev -0.188 against a +-0.12
    tolerance) and AST |rho| is non-monotone (0.716 at 1.5B -> 0.625 at 3B,
    "unexplained, inside n=60 noise"). A sixth rung is the cheapest way to
    tell a turn from noise;
  * the surviving §9 claim — the two-point fit systematically over-predicts,
    increasingly at the top — is a claim about the top, evidenced by three
    held-out cells whose largest deviation is the highest rung.

THE DTYPE DEVIATION, AND WHY THIS DRIVER RUNS TWO MODELS
---------------------------------------------------------
Every rung of the device-matched ladder is cuda/float32. Qwen2.5-Coder-7B is
~30.4 GB in float32 and cannot be run that way on any accelerator Kaggle
offers, so this rung is float16. That is a real protocol deviation, and this
project has already paid once for treating an environment difference as a
footnote: §9's ladder had to be regenerated device-matched when the CPU-era
rungs moved (pass@1 0.230 -> 0.2167, 0.495 -> 0.4467), which retired a
forensic argument built on the old numbers.

So the deviation is *measured*, not caveated. Stage B re-runs the existing 3B
rung at float16 and nothing else — same model, device, seed, temperature, N,
problems, and the same single-device load path — so it differs from the stored
`humaneval_uncertainty_20260827_192505.json` in exactly one respect. Whatever
float16 does to pass@1 and to AST |rho| is then a measured quantity on this
ladder rather than an assumption, and the 7B rung can be read with that
correction in hand.

`scripts/analyze_capability_sweep.py` now carries `device_resolved` and
`dtype_resolved` in PROTOCOL_KEYS (they were missing, which is why a float16
rung would previously have joined the ladder silently), so both new cells
announce themselves in `protocol_conflicts` when scored against the float32
five.

WHY device_map="auto" IS NOT A PERFORMANCE TWEAK
-------------------------------------------------
7B in float16 is ~15.2 GB against a P100's ~15.9 GB usable. The single-device
path materializes the weights in CPU RAM and then allocates the same bytes
again on the GPU while the CPU copy is still live, so allocator fragmentation
sinks it well before the arithmetic says it should fit. `device_map="auto"`
streams the checkpoint shard-by-shard straight to the target and spills
whatever does not fit to CPU instead of raising.

Offloading is numerically neutral — accelerate moves each weight onto the GPU
before its matmul — so an offloaded run produces the same values as a fully
resident one, only slower. Quantization would NOT be neutral, which is why it
is deliberately not used here even though it would fit more comfortably.
Stage B therefore keeps `--device-map none`: it fits easily at 6.2 GB, and
holding the load path fixed against the stored 3B rung is what makes it a
clean dtype control.

STAGE ORDER
-----------
Cheapest-first, and every stage publishes the moment it finishes, so a failure
at the end never costs what already succeeded:

  A. smoke — codegen-350M, 3 problems x 2 samples. Proves mount path, torch
     pin, dataset download and the pipeline end to end in ~2 minutes.
  B. 3B float16, single-device — the dtype control. A scientific deliverable
     on its own, so it runs before the risky stage rather than after it.
  C. 7B float16, device_map=auto — the new rung.

THREE KAGGLE FACTS INHERITED FROM WEEK 9 (each cost a session)
---------------------------------------------------------------
1. **Mount path.** A UI-attached dataset lands at `/kaggle/input/<slug>/`; an
   API-pushed kernel's lands at `/kaggle/input/datasets/<owner>/<slug>/<version>/`.
   `find_dataset_root()` searches from `/kaggle/input` rather than guessing.
2. **Accelerator.** The push request carries only `enable_gpu`; there is no
   accelerator field, so an API-pushed GPU kernel gets a **Tesla P100 (sm_60)**,
   not the T4 the UI offers. The image's torch ships no sm_60 kernels, so the
   first `model.generate()` dies with "CUDA error: no kernel image is available
   for execution on the device" — asynchronously, so it presents as a hang.
   Pin torch 2.6.0+cu124 on capability (6, 0), after removing
   torchvision/torchaudio (compiled against the image's torch, they break
   `import transformers` under 2.6).
3. **No periodic faulthandler dump.** `dump_traceback_later(..., repeat=True)`
   segfaulted all three Week 9 arms inside Qwen's forward pass.
   `faulthandler.enable()` alone is safe.

COST
----
Stage A ~2 min. Stage B ~25-40 min (3B fp16 on a P100). Stage C dominates:
~40 min if the weights stay resident, up to ~2 h if a few layers offload. The
kernel's own log prints `hf_device_map`, which is what tells the two apart
after the fact.
"""

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
        "attach the capability-sweep code dataset to this kernel"
    )


def free_gb(path: str = "/") -> float:
    """Free space in GB, for the checkpoint-cache decisions below."""
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


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
stamp("working copy: " + str(sorted(p.name for p in WORK.iterdir())))

# Fail now, not 40 minutes in, if the dataset holds a pre-device_map copy.
assert (
    "--device-map" in (WORK / "scripts/run_pipeline.py").read_text()
), "dataset holds a pre-2026-08-31 scripts/ copy: --device-map is missing"
assert (
    "DEVICE_MAP_CHOICES" in (WORK / "src/generation.py").read_text()
), "dataset holds a pre-2026-08-31 src/ copy: load_model has no device_map"
stamp("code version verified (device_map present)")

stamp("pip install datasets zss accelerate")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "datasets", "zss", "accelerate"],
    check=True,
)
stamp("pip done")

# See module docstring, Kaggle fact 2.
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
# --device auto silently resolves to cpu/float32 without a GPU, which would run
# for hours and produce cells comparable to nothing. Fail in 30 seconds.
assert (
    torch.cuda.is_available()
), "GPU accelerator is off - cells would be non-comparable"
stamp("device: " + torch.cuda.get_device_name(0))
stamp(f"capability: {torch.cuda.get_device_capability(0)}")
stamp(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
stamp(f"free disk: {free_gb():.1f} GB")

# Prove the assigned architecture can execute a float16 matmul before spending
# a session on generation; the sm_60 failure surfaces asynchronously otherwise
# and looks exactly like a hang.
_a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
stamp(f"fp16 matmul on device ok: {bool(torch.isfinite(_a @ _a).all())}")
del _a
torch.cuda.empty_cache()

import accelerate  # noqa: E402

stamp(f"accelerate {accelerate.__version__}")

# ------------------------------------------------------------- runner ------
# Every flag below is copied verbatim from kaggle_capability_sweep.py's COMMON
# block. The ladder's whole premise is that cells differ only in the model, so
# --dtype (and, for stage C only, --device-map) are the ONLY deviations, and
# both are recorded in the results config.
# fmt: off
COMMON = [
    "--mode", "uncertainty",
    "--dataset", "humaneval",
    "--n-problems", "60",
    "--problem-offset", "0",
    "--n-samples", "10",
    "--temperature", "0.7",
    "--seed", "42",
    "--device", "cuda",
    "--no-codebert",
    "--no-tree-edit",
]
# fmt: on

# faulthandler.enable() covers fatal signals only. Do NOT add
# dump_traceback_later(): see Kaggle fact 3.
WRAPPER = (
    "import faulthandler, runpy; faulthandler.enable(); "
    "runpy.run_path('scripts/run_pipeline.py', run_name='__main__')"
)


def publish_new_results(stage: str) -> list[Path]:
    """Copy any results JSON not yet published to /kaggle/working.

    Called immediately after each stage so a later failure cannot cost an
    earlier stage's data: /kaggle/working is only persisted when the kernel
    exits, but a file written there early still survives a crash at the end.
    """
    published = []
    for path in sorted((WORK / "results").glob("humaneval_uncertainty_*.json")):
        target = PUBLISH / path.name
        if target.exists():
            continue
        shutil.copy(path, target)
        data = json.loads(path.read_text())
        config, summary = data["config"], data["summary"]
        stamp(
            f"[{stage}] published {path.name}: "
            f"model={config['model']} "
            f"dtype={config['dtype_actual']} "
            f"device_map={config.get('device_map_requested')} "
            f"pass@1={summary['mean_pass_at_1']:.4f}"
        )
        if config.get("device_map_resolved"):
            placements = sorted(set(config["device_map_resolved"].values()))
            stamp(f"[{stage}] placement devices in use: {placements}")
        published.append(target)
    return published


def run_stage(stage: str, *extra_args: str) -> None:
    """Run one pipeline invocation, streaming its output, then publish."""
    command = [sys.executable, "-u", "-c", WRAPPER, *extra_args]
    stamp(f"=== stage {stage} ===")
    stamp("cmd: run_pipeline.py " + " ".join(extra_args))
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=WORK,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # Inherit rather than replace: the HF cache location, CUDA paths and
        # proxy settings all live in the kernel's environment, and a stripped
        # env would re-download every checkpoint into a different place.
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in process.stdout:
        # run_pipeline reports progress with tqdm, which redraws using "\r";
        # a captured log never renders those, so keep only the last segment.
        print(line.rstrip("\n").split("\r")[-1], flush=True)
    process.wait()
    elapsed = time.time() - started
    stamp(f"=== stage {stage} exit {process.returncode} after {elapsed:.0f}s ===")
    publish_new_results(stage)
    if process.returncode != 0:
        raise SystemExit(f"stage {stage} failed with exit {process.returncode}")


# --------------------------------------------------------- stage A: smoke --
# Deliberately the ladder's own smallest model rather than a toy: it exercises
# the identical code path the scored stages use, and its weights are already
# the ones every Week 2-7 number came from.
# fmt: off
run_stage(
    "A-smoke",
    "--mode", "uncertainty",
    "--dataset", "humaneval",
    "--n-problems", "3",
    "--n-samples", "2",
    "--temperature", "0.7",
    "--seed", "42",
    "--device", "cuda",
    "--dtype", "float16",
    "--device-map", "auto",
    "--no-codebert",
    "--no-tree-edit",
    "--model", "Salesforce/codegen-350M-mono",
    "--dry-run",
)
# fmt: on

# ------------------------------------------------- stage B: dtype control --
# Differs from the stored humaneval_uncertainty_20260827_192505.json (the 3B
# rung) in exactly one respect: float16 instead of float32. --device-map is
# left at "none" precisely so the load path is held fixed too.
# fmt: off
run_stage(
    "B-3b-fp16",
    *COMMON,
    "--dtype", "float16",
    "--device-map", "none",
    "--model", "Qwen/Qwen2.5-Coder-3B",
)
# fmt: on

stamp(f"free disk after stage B: {free_gb():.1f} GB")

# --------------------------------------------------------- stage C: 7B -----
# Base completion model, not -Instruct: instruction tuning would be confounded
# with capability, and every other rung of this ladder is a base model.
# fmt: off
run_stage(
    "C-7b-fp16",
    *COMMON,
    "--dtype", "float16",
    "--device-map", "auto",
    "--model", "Qwen/Qwen2.5-Coder-7B",
)
# fmt: on

# ------------------------------------------------------------- wrap up -----
stamp("all stages complete")
for path in sorted(PUBLISH.glob("humaneval_uncertainty_*.json")):
    data = json.loads(path.read_text())
    config, summary = data["config"], data["summary"]
    stamp(
        f"OUTPUT {path.name}  {config['model']:<28} "
        f"dtype={config['dtype_actual']} pass@1={summary['mean_pass_at_1']:.4f}"
    )
shutil.rmtree(WORK, ignore_errors=True)
stamp("done — download the Output tab into results/, then score locally with")
stamp("  python scripts/analyze_capability_sweep.py --results results/<the six cells>")
