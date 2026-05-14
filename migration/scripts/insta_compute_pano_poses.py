#!/usr/bin/env python3
"""Export pano poses from xsfm sparse image poses.

This script reads cam0 poses from image-poses.txt, maps each image to its
0-based index in the extracted video frame list under images/cam0, and writes
the result to pano-poses.txt.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ============================================================================
# [CONFIG] 命令行参数默认值 - 修改此处便于手动运行
# ============================================================================

DEFAULT_OUTPUT_ROOT = R"Z:\rick\dataset\q9000\MT20260430-112900-collect-by-app-only\output"
DEFAULT_IMAGE_POSES = (
		R"Z:\rick\dataset\q9000\MT20260430-112900-collect-by-app-only\output\images\ImgPose.txt"
)
DEFAULT_IMAGES_DIR = (
		R"Z:\rick\dataset\q9000\MT20260430-112900-collect-by-app-only\output\images"
)
DEFAULT_OUTPUT = (
		R"Z:\rick\dataset\q9000\MT20260430-112900-collect-by-app-only\output\xsfm\sparse\pano-poses.txt"
)
DEFAULT_CAMERA_PREFIX = "cam0/"
DEFAULT_STRICT = False
DEFAULT_INCLUDE_RPY = False


def quaternion_to_rpy_degrees(
		rw: float,
		rx: float,
		ry: float,
		rz: float,
) -> tuple[float, float, float]:
	"""Convert quaternion to roll, pitch, yaw in degrees."""
	sinr_cosp = 2.0 * (rw * rx + ry * rz)
	cosr_cosp = 1.0 - 2.0 * (rx * rx + ry * ry)
	roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

	sinp = 2.0 * (rw * ry - rz * rx)
	sinp = max(-1.0, min(1.0, sinp))
	pitch = math.degrees(math.asin(sinp))

	siny_cosp = 2.0 * (rw * rz + rx * ry)
	cosy_cosp = 1.0 - 2.0 * (ry * ry + rz * rz)
	yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

	return roll, pitch, yaw


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
			description=(
					"Export pano-poses.txt by matching cam0 images in ImgPose.txt "
					"against the 0-based frame order under output/images."
			)
	)
	parser.add_argument(
			"--output-root",
			type=Path,
			default=DEFAULT_OUTPUT_ROOT,
			help=(
					"Output root containing images/ and xsfm/sparse/. When set, "
					"default input/output paths are derived from it."
			),
	)
	parser.add_argument(
			"--image-poses",
			type=Path,
			default=DEFAULT_IMAGE_POSES,
			help="Path to output/images/ImgPose.txt.",
	)
	parser.add_argument(
			"--images-dir",
			type=Path,
			default=DEFAULT_IMAGES_DIR,
			help=(
					"Path to output/images or output/images/cam0. If output/images is "
					"given, the camera subfolder is resolved automatically."
			),
	)
	parser.add_argument(
			"--output",
			type=Path,
			default=DEFAULT_OUTPUT,
			help="Path to output pano-poses.txt.",
	)
	parser.add_argument(
			"--camera-prefix",
			default=DEFAULT_CAMERA_PREFIX,
			help="Image prefix to filter from image-poses.txt. Default: cam0/",
	)
	parser.add_argument(
			"--strict",
			action="store_true",
			default=DEFAULT_STRICT,
			help="Fail when an image in image-poses.txt is missing from images dir.",
	)
	parser.add_argument(
			"--include-rpy",
			action="store_true",
			default=DEFAULT_INCLUDE_RPY,
			help="Also output raw roll/pitch/yaw columns in addition to heading.",
	)
	return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
	camera_prefix = args.camera_prefix.strip()
	camera_dir_name = camera_prefix.rstrip("/").split("/")[-1]

	image_poses = args.image_poses
	images_dir = args.images_dir
	output_path = args.output

	if args.output_root is not None:
		output_root = args.output_root
		image_poses = image_poses or output_root / "images" / "ImgPose.txt"
		images_dir = images_dir or output_root / "images"
		output_path = output_path or output_root / "xsfm" / "sparse" / "pano-poses.txt"

	if image_poses is None or images_dir is None or output_path is None:
		raise ValueError(
				"Provide --output-root or all of --image-poses, --images-dir, and --output."
		)

	camera_dir = images_dir / camera_dir_name if (images_dir / camera_dir_name).is_dir() else images_dir
	return image_poses, camera_dir, output_path, camera_prefix


def build_image_index(camera_dir: Path) -> dict[str, int]:
	if not camera_dir.is_dir():
		raise FileNotFoundError(f"Camera image directory does not exist: {camera_dir}")

	image_names = sorted(
			path.name
			for path in camera_dir.iterdir()
			if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
	)
	if not image_names:
		raise ValueError(f"No image files found under {camera_dir}")

	return {image_name: index for index, image_name in enumerate(image_names)}


def load_pose_rows(
		image_poses_path: Path,
		camera_prefix: str,
		image_index: dict[str, int],
		strict: bool,
) -> tuple[list[dict[str, object]], list[str]]:
	if not image_poses_path.is_file():
		raise FileNotFoundError(f"ImgPose.txt does not exist: {image_poses_path}")

	rows: list[dict[str, object]] = []
	missing_names: list[str] = []
	seen_names: set[str] = set()

	with image_poses_path.open("r", encoding="utf-8") as handle:
		header = handle.readline().strip().split()
		expected_header = [
				"image",
				"x",
				"y",
				"z",
				"roll",
				"pitch",
				"yaw",
				"qx",
				"qy",
				"qz",
				"qw",
				"timestamp",
		]
		if header[: len(expected_header)] != expected_header:
			raise ValueError(
					f"Unexpected ImgPose header in {image_poses_path}: {' '.join(header)}"
			)

		for line_number, line in enumerate(handle, start=2):
			stripped = line.strip()
			if not stripped:
				continue

			parts = stripped.split()
			if len(parts) < 12:
				raise ValueError(
						f"Malformed line {line_number} in {image_poses_path}: {stripped}"
				)

			image_name = parts[0]
			if not image_name.startswith(camera_prefix):
				continue

			raw_image_name = Path(image_name).name
			if raw_image_name in seen_names:
				raise ValueError(f"Duplicate image pose entry found for {raw_image_name}")
			seen_names.add(raw_image_name)

			idx_in_video = image_index.get(raw_image_name)
			if idx_in_video is None:
				missing_names.append(raw_image_name)
				continue

			x, y, z = parts[1:4]
			rx, ry, rz, rw = parts[7:11]
			roll, pitch, yaw = quaternion_to_rpy_degrees(
					float(rw),
					float(rx),
					float(ry),
					float(rz),
			)
			rows.append(
					{
							"idx_in_video": idx_in_video,
							"x": x,
							"y": y,
							"z": z,
							"roll": roll,
							"pitch": pitch,
							"yaw": yaw,
							"rw": rw,
							"rx": rx,
							"ry": ry,
							"rz": rz,
							"raw_image_name": raw_image_name,
					}
			)

	if strict and missing_names:
		missing_preview = ", ".join(missing_names[:5])
		raise FileNotFoundError(
				f"{len(missing_names)} images from ImgPose.txt were not found in the "
				f"images dir. First missing entries: {missing_preview}"
		)

	rows.sort(key=lambda row: int(row["idx_in_video"]))
	return rows, missing_names


def build_output_rows(
		rows: list[dict[str, object]],
		include_rpy: bool,
) -> tuple[list[str], list[list[object]]]:
	first_yaw = float(rows[0]["yaw"])
	header = [
			"idx_in_video",
			"x",
			"y",
			"z",
			"heading",
	]
	if include_rpy:
		header.extend(["roll", "pitch", "yaw"])
	header.extend(["rw", "rx", "ry", "rz", "raw_image_name"])

	output_rows: list[list[object]] = []
	for row in rows:
		heading = float(row["yaw"]) - first_yaw
		output_row: list[object] = [
				row["idx_in_video"],
				row["x"],
				row["y"],
				row["z"],
				f"{heading:.6f}",
		]
		if include_rpy:
			output_row.extend(
					[
							f"{float(row['roll']):.6f}",
							f"{float(row['pitch']):.6f}",
							f"{float(row['yaw']):.6f}",
					]
			)
		output_row.extend(
				[
						row["rw"],
						row["rx"],
						row["ry"],
						row["rz"],
						row["raw_image_name"],
				]
		)
		output_rows.append(output_row)

	return header, output_rows


def write_pano_poses(
		output_path: Path,
		header: list[str],
		rows: list[list[object]],
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(header)
		writer.writerows(rows)


def main() -> int:
	args = parse_args()
	image_poses_path, camera_dir, output_path, camera_prefix = resolve_paths(args)
	image_index = build_image_index(camera_dir)
	rows, missing_names = load_pose_rows(
			image_poses_path=image_poses_path,
			camera_prefix=camera_prefix,
			image_index=image_index,
			strict=args.strict,
	)

	if not rows:
		raise ValueError(
				f"No pose rows with prefix {camera_prefix!r} were found in {image_poses_path}"
		)

	header, output_rows = build_output_rows(rows, include_rpy=args.include_rpy)
	write_pano_poses(output_path, header, output_rows)

	print(f"[OK] Wrote {len(rows)} rows to {output_path}")
	print(f"[INFO] Indexed {len(image_index)} frames from {camera_dir}")
	if missing_names:
		preview = ", ".join(missing_names[:5])
		print(
				f"[WARN] Skipped {len(missing_names)} images missing from the images dir: "
				f"{preview}"
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
