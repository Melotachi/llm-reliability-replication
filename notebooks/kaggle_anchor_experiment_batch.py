"""Objective 3 follow-up: does the repair ANCHOR matter? Two arms, one kernel.

HEADLESS batch driver, pushed with the Kaggle API:

    kaggle kernels push -p <dir>
    kaggle kernels status melihelibol/humaneval-anchor-experiment
    kaggle kernels output melihelibol/humaneval-anchor-experiment -p <dir>

kernel-metadata.json (written by notebooks/package_anchor_experiment_kaggle.py):

    {"id": "melihelibol/humaneval-anchor-experiment",
     "title": "humaneval-anchor-experiment",
     "code_file": "kaggle_anchor_experiment_batch.py", "language": "python",
     "kernel_type": "script", "is_private": true, "enable_gpu": true,
     "enable_internet": true,
     "dataset_sources": ["melihelibol/anchor-experiment-code"]}

THE EXPERIMENT
---------------
Objective 3 has always repaired `sample_index = 0` -- an arbitrary draw from
the 10 stored samples per problem. Week 6's execution-behavior consensus
selector (argmin `execution_behavior_self_distance`, verdict-free) lifts
pass@1 0.495 -> 0.700 on one cell and was never combined with repair. Week 6's
own lesson was "prediction != selection: this apparatus works as a SELECTOR
and fails as an ALLOCATOR" -- so use it where it works, to choose which of a
problem's N failing samples the repairer sees, instead of always sample 0.

Cohort: 128 HumanEval problems (of 148 sample-0 failures) where sample 0
fails AND no sibling passes -- `scripts/select_repair_anchors.py`'s output,
persisted as `results/repair_anchor_map_humaneval_20260907_230255.json`.
87 of the 128 anchors are != 0 (the selector disagrees with sample 0 on 68%
of the cohort); one problem (HumanEval/162) hit the all-crash fallback and
kept its anchor at sample 0, flagged `all_crash: true` in the map.

TWO ARMS, NOT THREE
--------------------
  code_only (treatment): repair the selector-chosen anchor with the failing
      program shown, no traceback.
  none (placebo): resample from the problem prompt at the SAME anchors,
      never revealing which sample was picked.

A KNOWN CONFOUND THIS EXPERIMENT MUST NOT BE SCORED WITHOUT
-------------------------------------------------------------
`iterative_repair`'s candidate seed is `record_seed + call_index`, where
`record_seed = seed + record_index` (record_index = position in the run's own
coordinate list). This run has 128 records; the existing Week 9 sample-0
baseline (`results/humaneval_week9_arm_code_only_sample0_qwen15b.json`) has
148. For a given task_id, record_index therefore differs between the two
runs -- so candidate seeds differ EVEN on the 41 problems whose anchor IS
sample 0. Treatment-vs-baseline is consequently never a clean anchor
contrast: on those 41 problems it is pure re-indexing noise, and on the other
87 it is anchor effect and re-indexing noise mixed together.

This is exactly what the `none` placebo measures, since it never sees the
anchor at all: any treatment-vs-baseline delta on the placebo arm IS the
re-indexing noise floor, not an anchor effect. Score
`scripts/score_anchor_experiment.py`'s output accordingly -- report the
placebo delta first as the noise floor, and only credit the code_only
treatment with an anchor effect to the extent it exceeds that floor, split by
whether each problem's anchor changed (87) or stayed at 0 (41, with
HumanEval/162's all-crash fallback case kept visible as its own row).

Order is `none` -> `code_only`: the diagnostic that scores the confound comes
first.

Every documented Kaggle trap from Week 9/11 applies verbatim: the API mount
path (/kaggle/input/datasets/<owner>/<slug>/<version>/, not
/kaggle/input/<slug>/), the Tesla P100/sm_60 torch pin, no periodic
faulthandler dump, and the ingest-window dataset-files poll (handled by the
packaging script, not this driver). This driver additionally gates on BOTH
sha256 hashes -- the coordinate selection AND the anchor map itself -- before
any generation, so a stale or mismatched dataset upload fails loudly instead
of producing a comparison of nothing.
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
    """Locate the mounted copy of the repository subset, however Kaggle nested it.

    Two shapes must both be accepted, because which one is served is Kaggle's
    choice and not the uploader's: the archive as pushed (`project.zip`), or the
    archive already expanded into `project/` during ingest. A run that only looks
    for the zip dies at startup with a message pointing at the kernel's dataset
    attachment, which is the wrong place to look -- the dataset was attached and
    was complete.
    """
    for candidate in _walk_dirs(INPUT_ROOT):
        if (candidate / "project.zip").is_file():
            return candidate
        expanded = candidate / "project"
        if (expanded / "src").is_dir() and (expanded / "scripts").is_dir():
            return candidate
    raise FileNotFoundError(
        "no project.zip and no expanded project/src+project/scripts under "
        "/kaggle/input; attach melihelibol/anchor-experiment-code to this kernel"
    )


WORK = Path("/kaggle/working/project")
PUBLISH = Path("/kaggle/working")
SOURCE_RESULTS = "results/humaneval_uncertainty_merged164_20260711.json"
ANCHOR_MAP = "results/repair_anchor_map_humaneval_20260907_230255.json"
EXPECTED_SELECTION_SHA256 = (
    "51af53ac935d7e222921d1590a69fa4ee76e8f1f3379109b16bdaf8fe5494e28"
)
EXPECTED_ANCHOR_MAP_SHA256 = (
    "4252f20869446e314945612732f49c7a247bada045e2f5419220a08c274d2544"
)
EXPECTED_COHORT_SIZE = 128
ARM_TIMEOUT_S = 7200  # ~2h per arm; 128 coordinates x 3 candidates = 384 calls

ARM_ORDER = ["none", "code_only"]
ARM_OUTPUTS = {
    "none": "results/humaneval_anchor_arm_none_qwen15b.json",
    "code_only": "results/humaneval_anchor_arm_code_only_qwen15b.json",
}

# ---------------------------------------------------------------- setup ----
DATASET = find_dataset_root()
stamp(f"dataset root: {DATASET}")
stamp("dataset contents: " + str(sorted(p.name for p in DATASET.iterdir())))

if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

archive = DATASET / "project.zip"
if archive.is_file():
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(WORK)
    stamp("materialised the working copy from project.zip")
else:
    shutil.copytree(DATASET / "project", WORK, dirs_exist_ok=True)
    stamp("materialised the working copy from the expanded project/ tree")
(WORK / "results").mkdir(parents=True, exist_ok=True)
shutil.copy2(DATASET / Path(SOURCE_RESULTS).name, WORK / SOURCE_RESULTS)
shutil.copy2(DATASET / Path(ANCHOR_MAP).name, WORK / ANCHOR_MAP)
stamp("working copy: " + str(sorted(p.name for p in WORK.iterdir())))

assert (WORK / SOURCE_RESULTS).exists(), "HumanEval source pool missing"
assert (WORK / ANCHOR_MAP).exists(), "anchor map missing"
assert (
    "--anchor-map" in (WORK / "scripts/run_iterative_repair_experiment.py").read_text()
), "dataset holds a pre-anchor-map copy of scripts/"

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
# needs it, and leave the image alone otherwise.
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
    # their C++ ops fail to register under 2.6, which breaks `import
    # transformers` itself (it imports torchvision whenever it is merely
    # installed). Neither is used anywhere in this project, so remove them
    # rather than downloading a second matched pair.
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
# cpu/float32 without a GPU and would run for hours producing arms that are
# not comparable to anything. Fail in 30 seconds instead.
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
    _anchor_map_sha256,
    _load_anchor_map,
    _selection_coordinates,
    _selection_sha256,
    collect_failures,
)

anchor_map = _load_anchor_map(WORK / ANCHOR_MAP)
anchor_sha = _anchor_map_sha256(anchor_map)
stamp(f"anchor map: {len(anchor_map)} task_ids, sha={anchor_sha}")
assert anchor_sha == EXPECTED_ANCHOR_MAP_SHA256, (
    "Anchor map differs from the one this run was gated on. Refusing to run "
    "the arms: nothing downstream would be comparable."
)

source = json.loads((WORK / SOURCE_RESULTS).read_text())
failures = collect_failures(source, anchor_map=anchor_map)
selected = sorted(failures, key=lambda f: f["task_id"])
coordinates = _selection_coordinates(selected)
selection_sha = _selection_sha256(coordinates)
stamp(
    f"{len(selected)} anchor coordinates (expected {EXPECTED_COHORT_SIZE}), "
    f"sha={selection_sha}"
)
stamp(f"calls per arm: {len(coordinates)} x 3 = {len(coordinates) * 3}")
assert (
    len(selected) == EXPECTED_COHORT_SIZE
), f"Cohort size {len(selected)} != expected {EXPECTED_COHORT_SIZE}"
assert selection_sha == EXPECTED_SELECTION_SHA256, (
    "Selection differs from the anchor-experiment cohort. Refusing to run "
    "the arms: nothing downstream would be comparable."
)
stamp("cohort + anchor map both verified")

# ----------------------------------------------------------- prompt ladder --
from transformers import AutoTokenizer  # noqa: E402

from src.repair import (  # noqa: E402
    _build_instruction_code_only_repair_prompt,
    _build_instruction_generation_prompt,
    _build_instruction_repair_prompt,
)
from src.utils import load_dataset_split  # noqa: E402

stamp("tokenizer + humaneval download ...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
problems = {row["task_id"]: row for row in load_dataset_split("humaneval")}
stamp(f"tokenizer ready, humaneval loaded: {len(problems)} problems")

_failing = selected[0]
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
stamp("prompt ladder verified on a HumanEval anchor")

# ------------------------------------------------------------- the arms ----
# faulthandler.enable() installs a handler for fatal signals only, so a real
# crash prints a C-level traceback. Do NOT add dump_traceback_later() here:
# the periodic dump walks every thread's frames while CUDA kernels are in
# flight, and the 2026-08-31 Week 9 MBPP run segfaulted (rc=-11) in all three
# arms at exactly 1x and 4x the dump interval, inside Qwen's forward pass.
WRAPPER = (
    "import faulthandler, runpy; faulthandler.enable(); "
    "runpy.run_path('scripts/run_iterative_repair_experiment.py', "
    "run_name='__main__')"
)

#: A crashed arm is resumable: the runner checkpoints after every coordinate
#: and --resume validates the stored config before continuing, so a retry
#: costs only the coordinates that were actually lost, and the candidate
#: seeds are index-deterministic -- a resumed arm is bit-identical to an
#: uninterrupted one.
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
        "humaneval",
        "--anchor-map",
        ANCHOR_MAP,
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
    stamp(f"=== arm {mode} ({ARM_ORDER.index(mode) + 1}/{len(ARM_ORDER)}) ===")
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

# Publish the exact anchor map and source-pool inputs this run consumed, so
# scoring can verify it read the same coordinates the kernel gated on without
# re-deriving them from the repository state at scoring time.
shutil.copy2(WORK / ANCHOR_MAP, PUBLISH / Path(ANCHOR_MAP).name)

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
