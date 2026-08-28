#!/usr/bin/env python3
"""Benchmark image post-processing and panorama rendering as separate stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--info-dir", type=Path, required=True)
    parser.add_argument("--surface-cloud", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--captures", type=int, default=6)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    args.recording = args.recording.resolve()
    args.info_dir = args.info_dir.resolve()
    args.surface_cloud = args.surface_cloud.resolve()
    args.worker = args.worker.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {args.output_dir}")
    camera_dir = args.output_dir / "cam"
    panorama_dir = args.output_dir / "panoramas"
    camera_dir.mkdir(parents=True, exist_ok=True)
    panorama_dir.mkdir(parents=True, exist_ok=True)

    sensor_frame = args.recording / "sensor_frame.xml"
    camera_masks = Path("/opt/NavVis/panorama-rendering/res/g8")
    operator_mask = Path("/opt/NavVis/panorama-rendering/res/g8_operator_mask.png")
    capture_count = args.captures
    worker_count = max(1, min(args.workers, capture_count))

    def postprocess_capture(index: int) -> None:
        capture = f"{index:05d}"
        for camera in range(4):
            source = args.recording / "cam" / f"{capture}-cam{camera}.dng"
            if not source.is_file():
                raise FileNotFoundError(source)
        subprocess.run(
            [
                str(args.worker),
                "--sensor-frame",
                str(sensor_frame),
                "--input-dir",
                str(args.recording / "cam"),
                "--metadata-dir",
                str(args.recording / "cam"),
                "--capture",
                capture,
                "--width",
                str(args.width),
                "--output",
                str(args.output_dir / f"{capture}-unused.jpg"),
                "--decoded-dir",
                str(camera_dir),
                "--camera-only",
                "true",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    image_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        list(pool.map(postprocess_capture, range(capture_count)))
    image_seconds = time.perf_counter() - image_started

    def render_capture(index: int) -> None:
        capture = f"{index:05d}"
        command = [
            str(args.worker),
            "--sensor-frame",
            str(sensor_frame),
            "--processed-camera-dir",
            str(camera_dir),
            "--capture",
            capture,
            "--width",
            str(args.width),
            "--output",
            str(panorama_dir / f"{capture}.jpg"),
            "--surface-cloud",
            str(args.surface_cloud),
            "--panorama-info",
            str(args.info_dir / f"{capture}-info.json"),
        ]
        if camera_masks.is_dir():
            command.extend(["--camera-mask-dir", str(camera_masks)])
        if operator_mask.is_file():
            command.extend(["--operator-mask", str(operator_mask)])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    panorama_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        list(pool.map(render_capture, range(capture_count)))
    panorama_seconds = time.perf_counter() - panorama_started

    report = {
        "schema": "navvis-cleanroom-staged-image-panorama-benchmark-v1",
        "recording": str(args.recording),
        "captures": capture_count,
        "camera_images": capture_count * 4,
        "width": args.width,
        "workers": worker_count,
        "dng_decode": "direct LibRaw/DNG-SDK; no converter subprocess",
        "worker": str(args.worker),
        "image_postprocessing_seconds": image_seconds,
        "panorama_rendering_seconds": panorama_seconds,
        "total_seconds": image_seconds + panorama_seconds,
        "camera_dir": str(camera_dir),
        "panorama_dir": str(panorama_dir),
    }
    report_path = args.output_dir / "timings.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
