"""Week 9 MBPP+ replication of the repair-context ladder — all three arms.

HEADLESS twin of `kaggle_week9_mbpp_context_ablation.py`. That file is the
cell-by-cell interactive driver; this one is a single script pushed as a Kaggle
*batch* kernel with the API:

    kaggle kernels push -p <dir>          # dir holds this file + kernel-metadata.json
    kaggle kernels status melihelibol/week9-mbpp-arms
    kaggle kernels output melihelibol/week9-mbpp-arms -p <dir>

kernel-metadata.json:

    {"id": "melihelibol/week9-mbpp-arms", "title": "week9-mbpp-arms",
     "code_file": "kaggle_week9_mbpp_arms_batch.py", "language": "python",
     "kernel_type": "script", "is_private": true, "enable_gpu": true,
     "enable_internet": true,
     "dataset_sources": ["melihelibol/week9-mbpp-context-ablation-code"]}

THREE FACTS THIS FILE ENCODES, each of which cost a session
-----------------------------------------------------------
1. **Mount path.** A UI-attached dataset lands at `/kaggle/input/<slug>/`; an
   API-pushed kernel's lands at `/kaggle/input/datasets/<owner>/<slug>/<version>/`.
   A slug-rooted lookup finds nothing and dies with FileNotFoundError, so
   `find_dataset_root()` searches from `/kaggle/input` instead.
2. **Accelerator.** The push request carries only `enable_gpu`; there is no
   field for the accelerator type, and an API-pushed GPU kernel is scheduled on
   a **Tesla P100 (sm_60)**, not the T4 the UI offers. The image's torch
   2.10+cu128 ships no sm_60 kernels, so the first `model.generate()` raises
   `CUDA error: no kernel image is available for execution on the device` —
   asynchronously, which is why it can present as a hang rather than an error.
   The workaround below pins torch 2.6.0+cu124 (the last build with sm_60) and
   removes torchvision/torchaudio first, since the image's copies are compiled
   against 2.10 and break `import transformers` outright under 2.6.
3. **No periodic faulthandler dump.** `faulthandler.dump_traceback_later(...,
   repeat=True)` around the generation subprocess segfaulted all three arms
   (rc=-11) inside Qwen's forward pass, at exactly 1x and 4x the dump interval:
   the dump walks every thread's frames while CUDA kernels are in flight. A
   3-coordinate smoke never reaches the first tick, so it cannot catch this.
   `faulthandler.enable()` alone (fatal signals only) is safe and is kept.

RECOVERY
--------
Every arm checkpoints after each coordinate, so a crash is resumable: an arm
that exits non-zero is retried in-process with `--resume`, up to ARM_ATTEMPTS,
and a retry that advances zero coordinates stops rather than looping on the
same crash. Candidate seeds are index-deterministic, so a resumed arm is
bit-identical to an uninterrupted one.


Batch driver for a single Kaggle GPU kernel. Kaggle's `--resume` story does
not survive a kernel restart (the checkpoint lives in /kaggle/working, which
is only published when the kernel exits), so the three arms run in ONE kernel,
in order, and each arm's JSON is copied to /kaggle/working the moment that arm
finishes. An arm-3 hang therefore still publishes arms 1 and 2.

Order is `none` -> `error` -> `code_only`: the pre-registered contrast first,
the decomposition arm last.

Only `--context-mode` differs between the arms. Every other flag is copied
verbatim from notebooks/kaggle_week9_mbpp_context_ablation.py's `run_arm`,
including the deliberate ABSENCE of --stop-on-pass (that is what makes the
arms budget-matched at 3 calls per coordinate, 162 per arm).
"""

import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

T0 = time.time()


def stamp(message: str) -> None:
    print(f"[T+{time.time() - T0:7.1f}s] {message}", flush=True)


INPUT_ROOT = Path("/kaggle/input")


def _walk_dirs(base: Path, max_depth: int = 5):
    """Yield directories under base, breadth-first, bounded in depth.

    Kaggle mounts an API-pushed kernel's datasets under
    /kaggle/input/datasets/<owner>/<slug>/<version>/, not under
    /kaggle/input/<slug>/ as the UI does, so the path cannot be hardcoded.
    """
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
    """Locate the mounted copy of the repository subset, however Kaggle nested it."""
    for candidate in _walk_dirs(INPUT_ROOT):
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
        if (candidate / "project.zip").is_file():
            return candidate
    raise FileNotFoundError(
        "no src/+scripts/ pair and no project.zip under /kaggle/input; "
        "attach melihelibol/week9-mbpp-context-ablation-code to this kernel"
    )


WORK = Path("/kaggle/working/project")
PUBLISH = Path("/kaggle/working")
SOURCE_RESULTS = "results/mbpp_uncertainty_20260712_024909.json"
EXPECTED_SELECTION_SHA = (
    "85c09d30cc342f0c8fc54b92a7d6b26563bfebe43b98f9bc92c681064be5b71b"
)
N_FAILURES = 60  # the cohort has 54; 60 is the notebook's cap, kept verbatim
ARM_TIMEOUT_S = 2700  # ~2.5x the smoke run's 54 x 21.4s = 19 min projection

ARM_ORDER = ["none", "error", "code_only"]
ARM_OUTPUTS = {
    "none": "results/mbpp_week9_arm_resample_sample0_qwen15b.json",
    "error": "results/mbpp_week9_arm_error_nostop_sample0_qwen15b.json",
    "code_only": "results/mbpp_week9_arm_code_only_sample0_qwen15b.json",
}

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
stamp("working copy: " + str(sorted(p.name for p in WORK.iterdir())))

assert (WORK / SOURCE_RESULTS).exists(), "MBPP source pool missing from the dataset"
assert (
    "--context-mode"
    in (WORK / "scripts/run_iterative_repair_experiment.py").read_text()
), "dataset holds a pre-Week-9 copy of scripts/"

stamp("pip install datasets zss")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "datasets", "zss"], check=True
)
stamp("pip done")

# An API-pushed Kaggle kernel is scheduled on a Tesla P100 (sm_60); the T4 the
# UI offers cannot be requested through the API (the push request carries only
# enable_gpu). The image's torch 2.10+cu128 ships no sm_60 kernels, so every
# generation call dies with
#   CUDA error: no kernel image is available for execution on the device
# Pin the last CUDA build that still compiles sm_60 when the assigned GPU
# needs it, and leave the image alone otherwise. Recorded in the run configs
# only as device=cuda/dtype=float16; the three arms share whatever this
# resolves to, which is what the comparison requires.
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
    # torchvision/torchaudio on the image are compiled against torch 2.10 and
    # their C++ ops fail to register under 2.6, which breaks  itself (transformers imports torchvision whenever it is
    # merely *installed*). Neither is used anywhere in this project, so remove
    # them rather than downloading a second matched pair.
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
# Device parity is the whole method here: --device auto silently resolves to
# cpu/float32 without a GPU and would run for hours producing arms that are not
# comparable to anything. Fail in 30 seconds instead.
assert (
    torch.cuda.is_available()
), "GPU accelerator is off - arms would be non-comparable"
stamp("device: " + torch.cuda.get_device_name(0))
stamp(f"capability: {torch.cuda.get_device_capability(0)}")

# Prove the assigned architecture can execute a float16 matmul before spending
# a session on generation; the sm_60 failure surfaces asynchronously otherwise
# and looks exactly like a hang.
_a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
stamp(f"fp16 matmul on device ok: {bool(torch.isfinite(_a @ _a).all())}")

from src.generation import resolve_runtime  # noqa: E402

resolved = resolve_runtime("auto", "auto")
stamp(f"resolved runtime: {resolved}")
assert str(resolved[0]) == "cuda" and "float16" in str(resolved[1]), resolved

# ------------------------------------------------------ cohort identity ----
sys.argv = ["x"]
from scripts.run_iterative_repair_experiment import (  # noqa: E402
    _selection_sha256,
    collect_failures,
)

source = json.loads((WORK / SOURCE_RESULTS).read_text())
failures = collect_failures(source, 0)
coordinates = [
    {"task_id": f["task_id"], "sample_index": f["sample_index"]}
    for f in failures[:N_FAILURES]
]
sha = _selection_sha256(coordinates)
stamp(f"{len(failures)} failing sample-0 coordinates (expected 54), sha={sha}")
stamp(f"calls per arm: {len(coordinates)} x 3 = {len(coordinates) * 3}")
assert sha == EXPECTED_SELECTION_SHA, (
    "Selection differs from the Week 7/8 MBPP cohort. Refusing to run the arms: "
    "nothing downstream would be comparable."
)
stamp("cohort verified")

# ----------------------------------------------------------- prompt ladder --
from transformers import AutoTokenizer  # noqa: E402

from src.repair import (  # noqa: E402
    _build_instruction_code_only_repair_prompt,
    _build_instruction_generation_prompt,
    _build_instruction_repair_prompt,
)
from src.utils import load_dataset_split  # noqa: E402

stamp("tokenizer + mbpp download ...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
problems = {row["task_id"]: row for row in load_dataset_split("mbpp")}
stamp(f"tokenizer ready, mbpp loaded: {len(problems)} problems")

_failing = failures[0]
_prompt = problems[_failing["task_id"]]["prompt"]
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
    _ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": _text}], add_generation_prompt=True, tokenize=True
    )
    stamp(f"{_mode:>10}: prompt {len(_text)} chars -> {len(_ids)} tokens")
assert _failing["initial_code"].strip() not in _built["none"]
assert _failing["initial_code"].strip() in _built["code_only"]
assert "Execution feedback" not in _built["code_only"]
assert "Execution feedback" in _built["error"]
stamp("prompt ladder verified on an MBPP prompt")

# ------------------------------------------------------------- the arms ----
# faulthandler.enable() installs a handler for fatal signals only, so a real
# crash prints a C-level traceback. Do NOT add dump_traceback_later() here:
# the periodic dump walks every thread's frames while CUDA kernels are in
# flight, and the 2026-08-31 run segfaulted (rc=-11) in all three arms at
# exactly 1x and 4x the dump interval, inside Qwen's forward pass. The 3-
# coordinate smoke never reached the first tick, which is why it looked clean.
WRAPPER = (
    "import faulthandler, runpy; faulthandler.enable(); "
    "runpy.run_path('scripts/run_iterative_repair_experiment.py', "
    "run_name='__main__')"
)

#: A crashed arm is resumable: the runner checkpoints after every coordinate
#: and --resume validates the stored config before continuing, so a retry costs
#: only the coordinates that were actually lost, and the candidate seeds are
#: index-deterministic (record_seed = seed + index) — a resumed arm is
#: bit-identical to an uninterrupted one.
ARM_ATTEMPTS = 3


def arm_command(context_mode: str, *, resume: bool) -> list[str]:
    """Build one arm's command line. Only --context-mode varies."""
    command = [
        sys.executable,
        "-u",
        "-c",
        WRAPPER,
        "--source-results",
        SOURCE_RESULTS,
        "--dataset",
        "mbpp",
        "--n-failures",
        str(N_FAILURES),
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
    return command


def completed_coordinates(mode: str) -> int:
    """How many coordinates the arm's checkpoint already holds."""
    produced = WORK / ARM_OUTPUTS[mode]
    if not produced.exists():
        return 0
    try:
        return len(json.loads(produced.read_text())["records"])
    except (json.JSONDecodeError, KeyError):
        return 0


outcomes: dict[str, str] = {}
for mode in ARM_ORDER:
    stamp(f"=== arm {mode} ({ARM_ORDER.index(mode) + 1}/3) ===")
    started = time.time()
    for attempt in range(1, ARM_ATTEMPTS + 1):
        done_before = completed_coordinates(mode)
        try:
            completed = subprocess.run(
                arm_command(mode, resume=attempt > 1),
                cwd=WORK,
                timeout=ARM_TIMEOUT_S,
                check=False,
            )
            outcomes[mode] = f"rc={completed.returncode}"
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            outcomes[mode] = f"TIMEOUT after {ARM_TIMEOUT_S}s"
            returncode = None
        except Exception as error:  # noqa: BLE001 - never lose a finished arm
            outcomes[mode] = f"{type(error).__name__}: {error}"
            returncode = None
        done_after = completed_coordinates(mode)
        stamp(
            f"arm {mode} attempt {attempt}/{ARM_ATTEMPTS}: {outcomes[mode]}, "
            f"coordinates {done_before} -> {done_after}"
        )
        if returncode == 0:
            break
        # A retry that advanced nothing would just repeat the same crash.
        if done_after == done_before:
            stamp(f"arm {mode}: attempt made no progress, giving up")
            break
    stamp(f"arm {mode}: {outcomes[mode]} after {time.time() - started:.1f}s")

    # Publish immediately: a later arm crashing must not cost an earlier one.
    produced = WORK / ARM_OUTPUTS[mode]
    if produced.exists():
        shutil.copy2(produced, PUBLISH / produced.name)
        stamp(f"published {produced.name} ({completed_coordinates(mode)} coordinates)")
    else:
        stamp(f"no output file for arm {mode}")

# ------------------------------------------------------- in-session read ----
stamp("=== summary ===")
for mode in ARM_ORDER:
    produced = PUBLISH / Path(ARM_OUTPUTS[mode]).name
    if not produced.exists():
        print(f"{mode:>10}: not produced ({outcomes.get(mode, 'not run')})", flush=True)
        continue
    payload = json.loads(produced.read_text())
    summary = payload["summary"]
    config = payload["config"]
    print(
        f"{mode:>10}: {summary['successful_repairs']}/"
        f"{summary['completed_failures']} coordinates solved, "
        f"{summary['repair_model_calls']} calls, "
        f"stop_on_pass={config.get('stop_on_pass')}, "
        f"device {config.get('device_resolved')}/{config.get('dtype_resolved')}",
        flush=True,
    )
stamp("all arms complete")
