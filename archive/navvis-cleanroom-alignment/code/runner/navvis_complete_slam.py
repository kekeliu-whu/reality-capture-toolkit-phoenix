#!/usr/bin/env python3
"""Run raw laser bags -> native frontend -> loops -> Stage1 -> Stage2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.surveyor_slam import load_imu_rosbag  # noqa: E402


RAW_IMU_HEADER = struct.Struct("<16sQ")
RAW_IMU_SAMPLE = struct.Struct("<q10d")
RAW_IMU_MAGIC = b"NVCRRAWIMU01\0\0\0\0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete autonomous clean-room SLAM")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build-release",
        help="C++ worker build (default: code/build-release)",
    )
    parser.add_argument(
        "--stage-build-dir",
        type=Path,
        help="Optional build containing the Stage1/Stage2 Ceres workers",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an existing clean-room NVSLAM6 archive instead of rebuilding it",
    )
    parser.add_argument("--imu-bag", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--solver-threads", type=int, default=7)
    parser.add_argument("--batch-limit", type=int)
    parser.add_argument("--reference-state", type=Path)
    parser.add_argument("--reference-loops", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def _worker(directory: Path, name: str) -> Path:
    candidates = [directory / name]
    if os.name == "nt":
        candidates.extend((directory / f"{name}.exe", directory / "Release" / f"{name}.exe"))
    result = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()), None
    )
    if result is None or not os.access(result, os.X_OK):
        raise FileNotFoundError(
            "missing executable; checked " + ", ".join(str(path) for path in candidates)
        )
    return result


def _native_library(directory: Path) -> Path:
    names = (
        ("navvis_recon_slam_frontend_native.dll",)
        if os.name == "nt"
        else ("libnavvis_recon_slam_frontend_native.so",)
    )
    candidates = [directory / name for name in names]
    if os.name == "nt":
        candidates.extend(directory / "Release" / name for name in names)
    result = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()), None
    )
    if result is None:
        raise FileNotFoundError(
            "missing SLAM native library; checked "
            + ", ".join(str(path) for path in candidates)
        )
    return result


def _export_raw_imu(bag: Path, output: Path) -> int:
    samples = load_imu_rosbag(bag)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(RAW_IMU_HEADER.pack(RAW_IMU_MAGIC, len(samples)))
        for sample in samples:
            stream.write(
                RAW_IMU_SAMPLE.pack(
                    sample.timestamp_ns,
                    *sample.linear_acceleration,
                    *sample.angular_velocity,
                    *sample.orientation_xyzw,
                )
            )
    return len(samples)


def _default_imu_bag(dataset: Path) -> Path:
    candidates = sorted(
        path
        for path in (dataset / "internal" / "bags").glob("bag_*.bag")
        if not path.name.startswith("bag_laser_")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one non-laser bag, found {len(candidates)}; pass --imu-bag"
        )
    return candidates[0]


def main() -> int:
    args = build_parser().parse_args()
    if args.threads < 1 or args.solver_threads < 1:
        raise ValueError("thread counts must be positive")
    if args.batch_limit is not None and args.batch_limit < 2:
        raise ValueError("--batch-limit must be at least 2")

    dataset = args.dataset.resolve()
    work = args.work_dir.resolve()
    build = args.build_dir.resolve()
    stage_build = (args.stage_build_dir or build).resolve()
    imu_bag = (
        args.imu_bag.resolve() if args.imu_bag else _default_imu_bag(dataset)
    )
    if not imu_bag.is_file():
        raise FileNotFoundError(imu_bag)

    pandar = _worker(build, "navvis_recon_pandar")
    frontend = _worker(build, "navvis_recon_slam")
    stage1 = _worker(stage_build, "navvis_recon_stage1_imu_ceres_solver")
    stage2 = _worker(stage_build, "navvis_recon_stage2_imu_ceres_solver")
    native = _native_library(build)

    generated_archive = work / "raw_scans.nvslam6"
    local_trajectory = work / "frontend_trajectory.csv"
    frontend_state = work / "frontend_state.bin"
    optimized_trajectory = work / "optimized_trajectory.csv"
    report = work / "slam_alignment_report.json"
    timing_report = work / "complete_slam_timing.json"
    raw_imu = work / "raw_imu.bin"
    backend_work = work / "backend"
    owned_outputs = (local_trajectory, frontend_state, optimized_trajectory, report)
    if any(path.exists() for path in owned_outputs) or backend_work.exists():
        if not args.force:
            raise FileExistsError(f"{work} contains prior SLAM output; pass --force")
        for path in owned_outputs:
            if path.is_file():
                path.unlink()
        if backend_work.is_dir():
            shutil.rmtree(backend_work)
    work.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float | int | str] = {}
    total_started = time.perf_counter()

    imu_started = time.perf_counter()
    imu_sample_count = _export_raw_imu(imu_bag, raw_imu)
    timing["imu_export_seconds"] = time.perf_counter() - imu_started
    timing["imu_samples"] = imu_sample_count

    if args.archive:
        archive = args.archive.resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
    else:
        archive = generated_archive
        if archive.exists() and not args.force:
            raise FileExistsError(f"{archive} exists; pass --archive to reuse it")
        archive_started = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "runner" / "navvis_slam_archive.py"),
                str(dataset),
                str(archive),
                "--pandar-worker",
                str(pandar),
                *(["--force"] if args.force else []),
            ],
            check=True,
        )
        timing["archive_seconds"] = time.perf_counter() - archive_started

    frontend_command = [
        str(frontend),
        "--archive",
        str(archive),
        "--imu-file",
        str(raw_imu),
        "--output",
        str(local_trajectory),
        "--state-output",
        str(frontend_state),
        "--threads",
        str(args.threads),
        "--progress-every",
        "500",
    ]
    if args.batch_limit is not None:
        frontend_command.extend(("--batch-limit", str(args.batch_limit)))
    frontend_started = time.perf_counter()
    subprocess.run(frontend_command, check=True)
    timing["frontend_seconds"] = time.perf_counter() - frontend_started

    backend_command = [
        sys.executable,
        str(ROOT / "runner" / "navvis_slam_recon.py"),
        "--frontend-state",
        str(frontend_state),
        "--imu-bag",
        str(imu_bag),
        "--stage1-solver",
        str(stage1),
        "--stage2-solver",
        str(stage2),
        "--work-dir",
        str(backend_work),
        "--output-trajectory",
        str(optimized_trajectory),
        "--output-report",
        str(report),
        "--solver-threads",
        str(args.solver_threads),
    ]
    if args.reference_state:
        backend_command.extend(
            ("--reference-state", str(args.reference_state.resolve()))
        )
    if args.reference_loops:
        backend_command.extend(
            ("--reference-loops", str(args.reference_loops.resolve()))
        )
    environment = os.environ.copy()
    environment["NAVVIS_RECON_SLAM_NATIVE"] = str(native)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{ROOT / 'src'}{os.pathsep}{existing_python_path}"
        if existing_python_path
        else str(ROOT / "src")
    )
    backend_started = time.perf_counter()
    subprocess.run(backend_command, check=True, env=environment)
    timing["backend_seconds"] = time.perf_counter() - backend_started
    timing["total_seconds"] = time.perf_counter() - total_started
    timing["dataset"] = str(dataset)
    timing_report.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")

    print(f"optimized trajectory: {optimized_trajectory}")
    print(f"SLAM report: {report}")
    print(f"complete timing: {timing_report}")
    print(
        "post-processing input: "
        f"./run_navvis_recon.sh DATASET --proc-base-dir DIR --trajectory-csv {optimized_trajectory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
