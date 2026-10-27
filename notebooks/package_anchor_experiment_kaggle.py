"""Package the anchor-selection experiment for Kaggle and push dataset + kernel.

Run from the repository root:

    python notebooks/package_anchor_experiment_kaggle.py --push

Objective 3 follow-up: does the repair ANCHOR matter? Every prior Objective 3
pool repaired sample_index=0 (an arbitrary draw); this experiment repairs the
Week 6 execution-behavior consensus selector's per-problem choice instead
(`scripts/select_repair_anchors.py`), over the 128-problem cohort where
sample 0 fails and no sibling passes.

Two arms, one kernel (same pattern as notebooks/kaggle_week9_mbpp_arms_batch.py):
  - `code_only` (treatment): the selector-chosen anchor, shown to the repairer
    without a traceback.
  - `none` (placebo): the SAME anchors, but the repair prompt never reveals
    which sample was picked (a same-model, same-budget resample). Any
    treatment-vs-baseline delta this arm shows is re-indexing noise (see the
    driver's module docstring), not an anchor effect, since it never sees the
    anchor at all.

Builds `project.zip` (src/, scripts/, requirements.txt/pyproject.toml) plus
the two JSON inputs the run needs at the top level of the dataset (not zipped,
so the driver can read them directly): the merged-164 source pool and the
anchor map. One artefact set, one code version; the kernel asserts on both
sha256 gates before spending a GPU hour.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = "melihelibol"
DATASET_SLUG = "anchor-experiment-code"
KERNEL_SLUG = "humaneval-anchor-experiment"
DRIVER_SCRIPT = "notebooks/kaggle_anchor_experiment_batch.py"

#: Inputs the driver reads directly (not inside project.zip) so it never has
#: to unzip them separately from the code archive.
SOURCE_RESULTS = "results/humaneval_uncertainty_merged164_20260711.json"
ANCHOR_MAP = "results/repair_anchor_map_humaneval_20260907_230255.json"

INCLUDE_DIRS = ("src", "scripts")
INCLUDE_FILES = ("requirements.txt", "pyproject.toml")


def build_zip(staging: Path) -> Path:
    """Write project.zip (code only) into `staging` and return its path."""
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "project.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for directory in INCLUDE_DIRS:
            for path in sorted((ROOT / directory).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                handle.write(path, path.relative_to(ROOT))
        for name in INCLUDE_FILES:
            path = ROOT / name
            if path.exists():
                handle.write(path, path.relative_to(ROOT))
    return archive


def stage_inputs(staging: Path) -> None:
    """Copy the source pool and anchor map alongside project.zip, unzipped."""
    for relative in (SOURCE_RESULTS, ANCHOR_MAP):
        source_path = ROOT / relative
        destination = staging / Path(relative).name
        shutil.copy2(source_path, destination)


def write_dataset_metadata(staging: Path) -> None:
    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Anchor experiment code and inputs",
                "id": f"{OWNER}/{DATASET_SLUG}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_kernel_metadata(staging: Path) -> Path:
    kernel_dir = staging.parent / f"{staging.name}_kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / DRIVER_SCRIPT, kernel_dir / Path(DRIVER_SCRIPT).name)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{OWNER}/{KERNEL_SLUG}",
                "title": KERNEL_SLUG,
                "code_file": Path(DRIVER_SCRIPT).name,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{OWNER}/{DATASET_SLUG}"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return kernel_dir


def expected_sizes(staging: Path) -> dict[str, int]:
    """Uncompressed size of every top-level file the dataset should serve."""
    sizes = {}
    for path in staging.iterdir():
        if path.is_file() and path.name not in ("dataset-metadata.json",):
            sizes[path.name] = path.stat().st_size
    return sizes


def wait_for_dataset(staging: Path, timeout_s: float = 900.0, poll_s: float = 20.0):
    """Block until the version just uploaded is the one Kaggle serves.

    `kaggle datasets version` returns as soon as the upload lands, and a
    kernel pushed into that window attaches an EMPTY input (Week 9/11
    lesson). `kaggle datasets status` is NOT a usable gate -- it reports
    "ready" for the PREVIOUS version throughout the ingest window. Poll
    `kaggle datasets files` until the anchor-map size matches what was just
    pushed, which is version-specific by construction.
    """
    probe = Path(ANCHOR_MAP).name
    want = expected_sizes(staging)[probe]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        completed = subprocess.run(
            # --page-size is not optional. The CLI defaults to 20 rows and
            # pages the rest behind a token, so a probe file that sorts late
            # (any `results/` JSON, once `project/...` is in the archive) is
            # simply absent from the default listing and this loop spins until
            # its timeout while the dataset has been ready for minutes. Max is
            # 200; the archives here are well under that.
            [
                "kaggle",
                "datasets",
                "files",
                f"{OWNER}/{DATASET_SLUG}",
                "--page-size",
                "200",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == probe:
                have = int(fields[1])
                print(f"dataset {probe}: {have} bytes (want {want})", flush=True)
                if have == want:
                    return
                break
        else:
            print(f"dataset files: {completed.stdout.strip()[:200]}", flush=True)
        time.sleep(poll_s)
    raise TimeoutError(f"{OWNER}/{DATASET_SLUG} still stale after {timeout_s:.0f}s")


def run(command: list[str], cwd: Path) -> int:
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--staging", type=Path, default=ROOT / "build" / "kaggle_anchor_experiment"
    )
    parser.add_argument("--push", action="store_true", help="Actually push.")
    parser.add_argument(
        "--new-dataset",
        action="store_true",
        help="Create the dataset instead of pushing a new version.",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Push the kernel against the dataset version already on Kaggle.",
    )
    args = parser.parse_args()

    staging = args.staging
    for path in (staging, staging.parent / f"{staging.name}_kernel"):
        if path.exists():
            shutil.rmtree(path)
    archive = build_zip(staging)
    stage_inputs(staging)
    write_dataset_metadata(staging)
    print(f"zip: {archive} ({archive.stat().st_size / 1e6:.2f} MB)")
    for name, size in expected_sizes(staging).items():
        print(f"dataset file: {name} ({size / 1e3:.1f} KB)")

    kernel_dir = write_kernel_metadata(staging)
    print(f"kernel staged: {KERNEL_SLUG} -> {kernel_dir}")

    if not args.push:
        print("dry run: pass --push to upload")
        return

    if args.skip_dataset:
        print("skipping the dataset push; kernel will attach the current version")
    elif args.new_dataset:
        run(["kaggle", "datasets", "create", "-p", str(staging), "-r", "zip"], ROOT)
        wait_for_dataset(staging)
    else:
        run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(staging),
                "-m",
                "anchor experiment code and inputs",
                "-r",
                "zip",
            ],
            ROOT,
        )
        wait_for_dataset(staging)

    run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], ROOT)
    print(f"status: kaggle kernels status {OWNER}/{KERNEL_SLUG}")


if __name__ == "__main__":
    main()
