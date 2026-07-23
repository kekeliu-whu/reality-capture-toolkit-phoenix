"""Create a legacy COLMAP text model directly from ImgPose-style camera poses.

The camera and image IDs are copied from an existing COLMAP database so that
``constrained_point_refiner`` can transcribe keypoints and matches without an
intermediate mapper. Input poses are camera-to-world and are converted to
COLMAP's camera-from-world convention.
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
from pathlib import Path


CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=Path, required=True)
    parser.add_argument("--pose_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def read_poses(path: Path) -> dict[str, tuple[float, ...]]:
    poses: dict[str, tuple[float, ...]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        next(handle, None)
        for line in handle:
            fields = line.split()
            if len(fields) < 11:
                continue
            poses[fields[0].replace("\\", "/")] = tuple(
                float(fields[index]) for index in (1, 2, 3, 7, 8, 9, 10)
            )
    return poses


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> tuple[tuple[float, ...], ...]:
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm == 0.0:
        raise ValueError("zero-length quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )


def main() -> None:
    args = parse_args()
    poses = read_poses(args.pose_path)
    args.output_path.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(args.database_path))
    cameras = connection.execute(
        "SELECT camera_id, model, width, height, params FROM cameras ORDER BY camera_id"
    ).fetchall()
    images = connection.execute(
        "SELECT image_id, camera_id, name FROM images ORDER BY image_id"
    ).fetchall()
    connection.close()

    with (args.output_path / "cameras.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id, model_id, width, height, params_blob in cameras:
            model_name = CAMERA_MODELS.get(model_id)
            if model_name is None:
                raise ValueError(f"unsupported COLMAP camera model ID: {model_id}")
            params = struct.unpack(f"<{len(params_blob) // 8}d", params_blob)
            handle.write(
                f"{camera_id} {model_name} {width} {height} "
                + " ".join(f"{value:.17g}" for value in params)
                + "\n"
            )

    written = 0
    missing: list[str] = []
    with (args.output_path / "images.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, camera_id, name in images:
            normalized_name = name.replace("\\", "/")
            pose = poses.get(normalized_name)
            if pose is None:
                missing.append(normalized_name)
                continue
            cx, cy, cz, qx, qy, qz, qw = pose
            rotation_c2w = quat_to_matrix(qx, qy, qz, qw)
            # R_cw = R_c2w^T and t_cw = -R_cw * C_world.
            tx = -(rotation_c2w[0][0] * cx + rotation_c2w[1][0] * cy + rotation_c2w[2][0] * cz)
            ty = -(rotation_c2w[0][1] * cx + rotation_c2w[1][1] * cy + rotation_c2w[2][1] * cz)
            tz = -(rotation_c2w[0][2] * cx + rotation_c2w[1][2] * cy + rotation_c2w[2][2] * cz)
            handle.write(
                f"{image_id} {qw:.17g} {-qx:.17g} {-qy:.17g} {-qz:.17g} "
                f"{tx:.17g} {ty:.17g} {tz:.17g} {camera_id} {normalized_name}\n\n"
            )
            written += 1

    (args.output_path / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n",
        encoding="utf-8",
    )
    if missing:
        preview = ", ".join(missing[:5])
        print(
            f"Warning: skipped {len(missing)} database images without an initial pose: {preview}"
        )
    if written < 2:
        raise RuntimeError(f"only {written} posed images were written")
    print(f"Wrote initial COLMAP model with {written} posed images and {len(cameras)} cameras")


if __name__ == "__main__":
    main()
