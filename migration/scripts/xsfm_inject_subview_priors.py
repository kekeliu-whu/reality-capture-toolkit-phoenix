"""Inject trajectory priors into a multi-camera subview database.

Given a trajectory file with panorama-level positions (one per timestamp),
writes pose_priors for ALL subview images in the database (same position for
each camera face of the same timestamp).

With --ref_camera_only, only the reference camera face (cam_idx=0) receives a
prior per timestamp. This avoids duplicate prior residuals on the same frame
center when using the multi-camera pipeline.

Usage:
    python inject_subview_priors.py \
        --database_path sub_database.db \
        --trajectory_path trajectory.txt \
        --num_cameras 4 \
        --camera_prefix pano_camera \
        --stddev 0.05 \
        --coordinate_system 1 \
        --ref_camera_only
"""

import argparse
import sqlite3
import struct
from pathlib import Path


CREATE_POSE_PRIORS_SQL = """\
CREATE TABLE IF NOT EXISTS pose_priors (
    pose_prior_id    INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    corr_data_id     INTEGER NOT NULL,
    corr_sensor_id   INTEGER NOT NULL,
    corr_sensor_type INTEGER NOT NULL,
    position         BLOB,
    position_covariance BLOB,
    gravity          BLOB,
    coordinate_system INTEGER NOT NULL
);
"""


def get_pose_priors_columns(db: sqlite3.Connection) -> set[str]:
    return {
        row[1] for row in db.execute("PRAGMA table_info(pose_priors)").fetchall()
    }


def parse_trajectory(path: Path) -> dict[str, tuple[float, float, float]]:
    """Parse trajectory file.

    Supports two formats:
      Legacy:  ``name x y z [...]``  (whitespace-separated, name first)
      CSV:     header line containing ``raw_image_name``; columns
               ``idx_in_video,x,y,z,heading,rw,rx,ry,rz,raw_image_name``

    Returns dict mapping timestamp stem -> (x, y, z).
    """
    import csv as _csv

    traj: dict[str, tuple[float, float, float]] = {}
    with open(path, newline="") as f:
        first = f.readline()
        f.seek(0)

        if "raw_image_name" in first:
            # CSV format with header — pano files are named by idx_in_video
            # (NOT raw_image_name), so key by idx_in_video.
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    name = str(int(row["idx_in_video"]))
                    traj[name] = (float(row["x"]), float(row["y"]), float(row["z"]))
                except (KeyError, ValueError):
                    continue
        else:
            # Legacy whitespace format: name x y z [...]
            # If multiple lines share a stem (e.g. cam0/<ts>.jpg and cam1/<ts>.jpg),
            # keep the first one (ref camera order is usually first in file).
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                name = Path(parts[0]).stem
                if name in traj:
                    continue
                try:
                    traj[name] = (float(parts[1]), float(parts[2]), float(parts[3]))
                except ValueError:
                    continue
    return traj


def main():
    parser = argparse.ArgumentParser(
        description="Inject trajectory priors into subview database."
    )
    parser.add_argument("--database_path", type=Path, required=True)
    parser.add_argument("--trajectory_path", type=Path, required=True)
    parser.add_argument("--num_cameras", type=int, default=3,
                        help="Number of camera faces per rig (default: 3)")
    parser.add_argument("--camera_prefix", default="pano_camera",
                        help="Folder prefix for subview images (default: pano_camera)")
    parser.add_argument("--stddev", type=float, default=0.05,
                        help="Position standard deviation in meters (default: 0.05)")
    parser.add_argument("--coordinate_system", type=int, default=1,
                        help="0=WGS84, 1=CARTESIAN (default: 1)")
    parser.add_argument("--ref_camera_only", action="store_true",
                        help="Only write priors for the reference camera "
                             "(cam_idx=0) to avoid duplicate residuals on the "
                             "same frame center in the multi-camera pipeline")
    args = parser.parse_args()

    assert args.num_cameras >= 1, (
        f"--num_cameras must be >= 1, got {args.num_cameras}"
    )

    trajectory = parse_trajectory(args.trajectory_path)
    print(f"  Loaded {len(trajectory)} trajectory entries")

    db = sqlite3.connect(str(args.database_path))
    db.executescript(CREATE_POSE_PRIORS_SQL)
    pose_priors_columns = get_pose_priors_columns(db)

    # Clear existing priors.
    db.execute("DELETE FROM pose_priors")

    # Build lookup: image name -> (image_id, camera_id).
    rows = db.execute("SELECT image_id, camera_id, name FROM images").fetchall()
    name_to_info: dict[str, tuple[int, int]] = {}
    for image_id, camera_id, name in rows:
        name_to_info[name] = (image_id, camera_id)

    # Covariance and gravity blobs.
    var = args.stddev ** 2
    cov_blob = struct.pack("<9d", var, 0, 0, 0, var, 0, 0, 0, var)
    gravity_blob = struct.pack("<3d", 0.0, 0.0, 0.0)

    cam_indices = range(1) if args.ref_camera_only else range(args.num_cameras)

    matched = 0
    for timestamp, (x, y, z) in trajectory.items():
        position_blob = struct.pack("<3d", x, y, z)
        for cam_idx in cam_indices:
            img_name = f"{args.camera_prefix}{cam_idx}/{timestamp}.jpg"
            info = name_to_info.get(img_name)
            if info is None:
                continue
            image_id, camera_id = info
            if "corr_data_id" in pose_priors_columns:
                db.execute(
                    "INSERT INTO pose_priors "
                    "(corr_data_id, corr_sensor_id, corr_sensor_type, "
                    " position, position_covariance, gravity, coordinate_system) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (image_id, camera_id, 0, position_blob, cov_blob,
                     gravity_blob, args.coordinate_system),
                )
            else:
                db.execute(
                    "INSERT INTO pose_priors "
                    "(image_id, position, coordinate_system, position_covariance) "
                    "VALUES (?, ?, ?, ?)",
                    (image_id, position_blob, args.coordinate_system, cov_blob),
                )
            matched += 1

    db.commit()
    db.close()
    num_cams_written = 1 if args.ref_camera_only else args.num_cameras
    print(f"  Wrote {matched} pose priors ({matched // max(num_cams_written, 1)} timestamps "
          f"x {num_cams_written} camera(s), stddev={args.stddev}m, "
          f"coord_sys={args.coordinate_system}"
          f"{', ref_camera_only' if args.ref_camera_only else ''})")
    unmatched = len(trajectory) * num_cams_written - matched
    if unmatched > 0:
        print(f"  Warning: {unmatched} images not found in database")


if __name__ == "__main__":
    main()
