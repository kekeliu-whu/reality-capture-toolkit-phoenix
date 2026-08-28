#!/usr/bin/env python3
"""Run raw laser bags -> native frontend -> loops -> Stage1 -> Stage2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


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
    result = (directory / name).resolve()
    if not result.is_file() or not os.access(result, os.X_OK):
        raise FileNotFoundError(f"missing executable {result}")
    return result


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
    native = (build / "libnavvis_recon_slam_frontend_native.so").resolve()
    if not native.is_file():
        raise FileNotFoundError(native)

    generated_archive = work / "raw_scans.nvslam6"
    local_trajectory = work / "frontend_trajectory.csv"
    frontend_state = work / "frontend_state.bin"
    optimized_trajectory = work / "optimized_trajectory.csv"
    report = work / "slam_alignment_report.json"
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

    if args.archive:
        archive = args.archive.resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
    else:
        archive = generated_archive
        if archive.exists() and not args.force:
            raise FileExistsError(f"{archive} exists; pass --archive to reuse it")
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

    frontend_command = [
        str(frontend),
        "--archive",
        str(archive),
        "--imu-bag",
        str(imu_bag),
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
    subprocess.run(frontend_command, check=True)

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
        f"{ROOT / 'src'}:{existing_python_path}"
        if existing_python_path
        else str(ROOT / "src")
    )
    subprocess.run(backend_command, check=True, env=environment)

    print(f"optimized trajectory: {optimized_trajectory}")
    print(f"SLAM report: {report}")
    print(
        "post-processing input: "
        f"./run_navvis_recon.sh DATASET --proc-base-dir DIR --trajectory-csv {optimized_trajectory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
