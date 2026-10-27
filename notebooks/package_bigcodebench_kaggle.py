"""Package the BigCodeBench cell for Kaggle and push dataset + kernel.

Run from the repository root:

    python notebooks/package_bigcodebench_kaggle.py --push

Builds `project.zip` (src/, scripts/, data/*.txt, requirements.txt) into a
staging directory with a `dataset-metadata.json`, pushes it as a Kaggle
dataset, then writes each kernel's `kernel-metadata.json` and pushes it.

Why a zip rather than a directory tree: Kaggle's dataset uploader flattens and
re-nests directories unpredictably, and the driver's `find_dataset_root()`
already knows how to unpack a `project.zip`. One artefact, one code version,
and the kernel asserts on the presence of the Week 11 flags before spending a
GPU hour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = "melihelibol"
DATASET_SLUG = "bcb-cell-code"
#: The problem set the rungs share. The small kernel writes it; it must be
#: re-uploaded before the large kernel is pushed.
CELL_IDS = "data/bigcodebench_cell_task_ids.txt"
#: slug -> (driver script, BCB_RUNGS value). Both kernels run the SAME driver;
#: the packaging step injects a one-line prelude selecting the rung set, so the
#: two sessions cannot drift apart in protocol.
KERNELS = {
    "bigcodebench-cell-a": ("notebooks/kaggle_bigcodebench_cell.py", "small"),
    "bigcodebench-cell-b": ("notebooks/kaggle_bigcodebench_cell.py", "large"),
}

#: Everything the driver needs and nothing else -- no results/, no reports/.
INCLUDE_DIRS = ("src", "scripts")
INCLUDE_FILES = ("requirements.txt", "pyproject.toml")
INCLUDE_GLOBS = ("data/*.txt",)


def build_zip(staging: Path) -> Path:
    """Write project.zip into `staging` and return its path."""
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
        for pattern in INCLUDE_GLOBS:
            for path in sorted(ROOT.glob(pattern)):
                handle.write(path, path.relative_to(ROOT))
    return archive


def write_dataset_metadata(staging: Path) -> None:
    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "BigCodeBench cell code",
                "id": f"{OWNER}/{DATASET_SLUG}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_kernel_metadata(staging: Path, slug: str, script: str, rungs: str) -> Path:
    """Stage one kernel script plus its metadata; return the staging dir.

    The driver is copied verbatim with a one-line prelude setting BCB_RUNGS, so
    both kernels run byte-identical experiment code and differ only in which
    rungs they execute.
    """
    # NOT under the dataset staging dir: `kaggle datasets create -p` uploads
    # every file it finds there, so a kernel folder nested inside would ship
    # as dataset content.
    kernel_dir = staging.parent / f"{staging.name}_kernels" / slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    source = (ROOT / script).read_text(encoding="utf-8")
    # The sha256 of the ids this push is shipping. The large kernel refuses to
    # run unless the dataset it mounts carries exactly these -- see the driver's
    # BCB_CELL_SHA256 check for why a silent mismatch would void the ladder.
    cell_sha256 = hashlib.sha256((ROOT / CELL_IDS).read_bytes()).hexdigest()
    prelude = (
        "import os\n\n"
        f'os.environ["BCB_RUNGS"] = "{rungs}"\n'
        f'os.environ["BCB_CELL_SHA256"] = "{cell_sha256}"\n\n'
    )
    (kernel_dir / Path(script).name).write_text(prelude + source, encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{OWNER}/{slug}",
                "title": slug,
                "code_file": Path(script).name,
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


def expected_sizes(archive: Path) -> dict[str, int]:
    """Uncompressed size of every entry in project.zip, keyed by archive path.

    Kaggle expands a zip uploaded with `-r zip`, so these are exactly the
    file names and sizes `kaggle datasets files` will report once the new
    version is live.
    """
    with zipfile.ZipFile(archive) as handle:
        return {info.filename: info.file_size for info in handle.infolist()}


def wait_for_dataset(archive: Path, timeout_s: float = 900.0, poll_s: float = 20.0):
    """Block until the version just uploaded is the one Kaggle serves.

    `kaggle datasets version` returns as soon as the upload lands, and a kernel
    pushed into the ingest window attaches an EMPTY input -- the run then dies
    at `find_dataset_root()` with "no src/+scripts/ pair ... under
    /kaggle/input", minutes of queue time later and with nothing in the message
    pointing at the dataset. Both failed bigcodebench-cell-a runs died this way.

    `kaggle datasets status` is NOT a usable gate: it reports "ready" for the
    previously ingested version, so it answers "ready" throughout the window it
    is supposed to detect. This polls `kaggle datasets files` instead and waits
    for the sizes to match the archive that was just pushed -- version-specific
    by construction.
    """
    probe = "data/bigcodebench_eligible_task_ids.txt"
    want = expected_sizes(archive)[probe]
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
    parser.add_argument("--staging", type=Path, default=ROOT / "build" / "kaggle_bcb")
    parser.add_argument("--push", action="store_true", help="Actually push.")
    parser.add_argument(
        "--new-dataset",
        action="store_true",
        help="Create the dataset instead of pushing a new version.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(KERNELS),
        action="append",
        help=(
            "Push only these kernels (repeatable). The large kernel reuses the "
            "task ids the small one selected, so it must not be pushed until "
            "the small kernel has finished and its ids are back in the dataset."
        ),
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Push kernels against the dataset version already on Kaggle.",
    )
    args = parser.parse_args()

    staging = args.staging
    for path in (staging, staging.parent / f"{staging.name}_kernels"):
        if path.exists():
            shutil.rmtree(path)
    archive = build_zip(staging)
    write_dataset_metadata(staging)
    print(f"zip: {archive} ({archive.stat().st_size / 1e6:.2f} MB)")

    kernel_dirs = {
        slug: write_kernel_metadata(staging, slug, script, rungs)
        for slug, (script, rungs) in KERNELS.items()
    }
    for slug, kernel_dir in kernel_dirs.items():
        print(f"kernel staged: {slug} -> {kernel_dir}")

    if not args.push:
        print("dry run: pass --push to upload")
        return

    if args.skip_dataset:
        print("skipping the dataset push; kernels will attach the current version")
    elif args.new_dataset:
        run(["kaggle", "datasets", "create", "-p", str(staging), "-r", "zip"], ROOT)
        wait_for_dataset(archive)
    else:
        run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(staging),
                "-m",
                "week11 bigcodebench cell",
                "-r",
                "zip",
            ],
            ROOT,
        )
        wait_for_dataset(archive)

    selected = args.only or sorted(KERNELS)
    for slug in selected:
        run(["kaggle", "kernels", "push", "-p", str(kernel_dirs[slug])], ROOT)
        print(f"status: kaggle kernels status {OWNER}/{slug}")


if __name__ == "__main__":
    main()
