"""Fix multi-camera rig structure in COLMAP database.

The feature_extractor with single_camera_per_folder creates one rig per camera.
This script restructures to: 1 rig with N sensors, images grouped by timestamp.

Usage:
    python fix_rig_database.py --database_path DB --rig_config RIG_CONFIG
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import struct
from collections import defaultdict


CAMERA_MODEL_NAME_TO_ID = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}


def pack_rigid3d(qw, qx, qy, qz, tx, ty, tz):
    """Pack Rigid3d as 7 little-endian doubles (COLMAP BLOB format)."""
    return struct.pack("<7d", qw, qx, qy, qz, tx, ty, tz)


def pack_camera_params(params):
    return struct.pack("<" + "d" * len(params), *[float(v) for v in params])


def update_camera_params_from_rig(cursor, cameras_cfg, prefix_to_cam_id):
    updated = 0
    for cam_cfg in cameras_cfg:
        prefix = cam_cfg["image_prefix"].rstrip("/")
        model_name = cam_cfg.get("camera_model_name")
        camera_params = cam_cfg.get("camera_params")
        if model_name is None or camera_params is None:
            print(f"  [WARN] camera params missing for prefix={prefix}; keeping existing camera row")
            continue

        model_key = str(model_name).upper()
        if model_key not in CAMERA_MODEL_NAME_TO_ID:
            raise ValueError(f"Unsupported camera model in rig config: {model_name}")
        if prefix not in prefix_to_cam_id:
            raise KeyError(f"Image prefix from rig config not found in database: {prefix}")

        cam_id = prefix_to_cam_id[prefix]
        cursor.execute(
            "UPDATE cameras SET model = ?, params = ? WHERE camera_id = ?",
            (CAMERA_MODEL_NAME_TO_ID[model_key], pack_camera_params(camera_params), cam_id),
        )
        updated += cursor.rowcount
        print(f"  camera_id={cam_id} prefix={prefix} model={model_key} params={camera_params}")
    print(f"Updated {updated} camera parameter row(s) from rig config")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--rig_config", required=True,
                        help="rig_config.json with cam_from_rig transforms")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    args = parser.parse_args()

    # Backup
    if args.backup:
        backup = args.database_path + ".bak"
        shutil.copy2(args.database_path, backup)
        print(f"Backup: {backup}")

    # Load rig config
    with open(args.rig_config) as f:
        rig_configs = json.load(f)
    assert len(rig_configs) == 1, "Expected exactly 1 rig config"
    cameras_cfg = rig_configs[0]["cameras"]

    n_sensors = len(cameras_cfg)
    actual_prefixes = [c["image_prefix"].rstrip("/") for c in cameras_cfg]
    print(f"Rig sensors ({n_sensors}): {actual_prefixes}")

    # Determine ref sensor and non-ref sensor transforms
    ref_prefix = None
    cam_prefix_to_idx = {}
    sensor_transforms = {}  # camera_id -> (qw, qx, qy, qz, tx, ty, tz)

    for i, cam_cfg in enumerate(cameras_cfg):
        prefix = cam_cfg["image_prefix"].rstrip("/")
        cam_prefix_to_idx[prefix] = i
        if cam_cfg.get("ref_sensor"):
            ref_prefix = prefix

    db = sqlite3.connect(args.database_path)
    c = db.cursor()

    # Map image prefix to camera_id
    c.execute("SELECT DISTINCT camera_id, name FROM images")
    prefix_to_cam_id = {}
    for cam_id, name in c.fetchall():
        prefix = name.split("/")[0]
        prefix_to_cam_id[prefix] = cam_id

    ref_cam_id = prefix_to_cam_id[ref_prefix]
    print(f"Reference camera: prefix={ref_prefix} camera_id={ref_cam_id}")

    update_camera_params_from_rig(c, cameras_cfg, prefix_to_cam_id)

    # Build sensor_from_rig for non-ref cameras
    for cam_cfg in cameras_cfg:
        prefix = cam_cfg["image_prefix"].rstrip("/")
        cam_id = prefix_to_cam_id[prefix]
        if cam_cfg.get("ref_sensor"):
            continue
        rot = cam_cfg["cam_from_rig_rotation"]  # [qw, qx, qy, qz]
        trans = cam_cfg.get("cam_from_rig_translation", [0, 0, 0])
        sensor_transforms[cam_id] = (rot[0], rot[1], rot[2], rot[3],
                                     trans[0], trans[1], trans[2])
        print(f"  camera_id={cam_id} prefix={prefix} "
              f"q=[{rot[0]:.4f},{rot[1]:.4f},{rot[2]:.4f},{rot[3]:.4f}] "
              f"t=[{trans[0]:.4f},{trans[1]:.4f},{trans[2]:.4f}]")

    # Group images by timestamp (filename without prefix)
    c.execute("SELECT image_id, name, camera_id FROM images")
    timestamp_groups = defaultdict(list)
    for img_id, name, cam_id in c.fetchall():
        # name = "pano_camera0/1749887003_138947.jpg"
        filename = name.split("/")[-1]
        timestamp = os.path.splitext(filename)[0]
        timestamp_groups[timestamp].append((img_id, cam_id))

    # Filter: only keep groups with all cameras present
    all_cam_ids = set(prefix_to_cam_id.values())
    valid_groups = {}
    for ts, imgs in sorted(timestamp_groups.items()):
        group_cam_ids = {cam_id for _, cam_id in imgs}
        if group_cam_ids == all_cam_ids:
            valid_groups[ts] = imgs
    print(f"\nValid frame groups: {len(valid_groups)} / {len(timestamp_groups)}")

    # Clear old rig/frame data
    c.execute("DELETE FROM rig_sensors")
    c.execute("DELETE FROM rigs")
    c.execute("DELETE FROM frame_data")
    c.execute("DELETE FROM frames")
    c.execute("DELETE FROM sqlite_sequence WHERE name IN ('rigs', 'frames')")
    db.commit()

    # Create 1 rig with ref_sensor
    SENSOR_TYPE_CAMERA = 0
    c.execute("INSERT INTO rigs (rig_id, ref_sensor_id, ref_sensor_type) VALUES (1, ?, ?)",
              (ref_cam_id, SENSOR_TYPE_CAMERA))

    # Add non-ref sensors to rig_sensors
    for cam_id, transform in sensor_transforms.items():
        blob = pack_rigid3d(*transform)
        c.execute("INSERT INTO rig_sensors (rig_id, sensor_id, sensor_type, sensor_from_rig) "
                  "VALUES (1, ?, ?, ?)",
                  (cam_id, SENSOR_TYPE_CAMERA, blob))
    print(f"Created rig with {1 + len(sensor_transforms)} sensors "
          f"(ref={ref_cam_id})")

    # Create frames and frame_data
    frame_id = 1
    for ts in sorted(valid_groups.keys()):
        imgs = valid_groups[ts]
        c.execute("INSERT INTO frames (frame_id, rig_id) VALUES (?, 1)",
                  (frame_id,))
        for img_id, cam_id in imgs:
            c.execute("INSERT INTO frame_data (frame_id, data_id, sensor_id, sensor_type) "
                      "VALUES (?, ?, ?, ?)",
                      (frame_id, img_id, cam_id, SENSOR_TYPE_CAMERA))
        frame_id += 1

    db.commit()
    print(f"Created {frame_id - 1} frames")

    # Verify
    c.execute("SELECT frame_id, COUNT(*) FROM frame_data GROUP BY frame_id LIMIT 5")
    print(f"\nVerification - images per frame: {c.fetchall()}")
    c.execute("SELECT COUNT(*) FROM rig_sensors")
    print(f"rig_sensors count: {c.fetchone()[0]}")

    db.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
