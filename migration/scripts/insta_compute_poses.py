#!/usr/bin/env python3
"""
Compute camera poses from IMU trajectory and camera-IMU extrinsic parameters.
Handles multiple cameras organized in cam0, cam1, ... folders.
Generates ImgPose.txt file with positions, quaternions, and camera IDs.

Usage:
    python insta_compute_poses.py --poses-file <traj.txt> --calib-file <calib.dat>
                                   --image-folder <parent_folder> --output <output.txt>

Expected folder structure:
    parent_folder/
        cam0/
            *.jpg
        cam1/
            *.jpg
        ...
"""

import argparse
import numpy as np
import csv
from pathlib import Path
from proto.calib_pb2 import SensorCalib
import re
from scipy.spatial.transform import Rotation, Slerp


def extract_timestamp_from_filename(filename):
    """Extract timestamp from image filename (format: 1735732912_646576.jpg -> 1735732912.646576)"""
    name = Path(filename).stem
    match = re.match(r"(\d+)_(\d+)", name)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None


def interpolate_pose(timestamp, imu_poses):
    """
    Interpolate camera pose at given timestamp from IMU trajectory.
    Uses linear interpolation for position and SLERP for rotation.

    Args:
        timestamp: Target timestamp
        imu_poses: List of pose dicts with timestamp, tx, ty, tz, rx, ry, rz, rw

    Returns:
        Interpolated pose dict, or None if timestamp is out of range
    """
    # Find surrounding poses
    before = None
    after = None

    for pose in imu_poses:
        if pose["timestamp"] <= timestamp:
            before = pose
        if pose["timestamp"] >= timestamp and after is None:
            after = pose

    if before is None or after is None:
        return None

    # If exact match or very close
    if abs(before["timestamp"] - timestamp) < 1e-6:
        return before.copy()
    if abs(after["timestamp"] - timestamp) < 1e-6:
        return after.copy()

    # Interpolation factor
    t = (timestamp - before["timestamp"]) / (after["timestamp"] - before["timestamp"])

    # Linear interpolation for position
    tx = before["tx"] + t * (after["tx"] - before["tx"])
    ty = before["ty"] + t * (after["ty"] - before["ty"])
    tz = before["tz"] + t * (after["tz"] - before["tz"])

    # SLERP for rotation
    q_before = Rotation.from_quat(
        [before["rx"], before["ry"], before["rz"], before["rw"]]
    )
    q_after = Rotation.from_quat([after["rx"], after["ry"], after["rz"], after["rw"]])

    # Create interpolator
    rotvecs = np.array([q_before.as_rotvec(), q_after.as_rotvec()])
    times = np.array([0, 1])
    slerp = Slerp(times, Rotation.from_rotvec(rotvecs))

    q_interp = slerp(t)
    qx, qy, qz, qw = q_interp.as_quat()

    return {
        "timestamp": timestamp,
        "tx": tx,
        "ty": ty,
        "tz": tz,
        "rx": qx,
        "ry": qy,
        "rz": qz,
        "rw": qw,
    }


def apply_extrinsic_transform(pose, extrinsic):
    """
    Apply camera-IMU extrinsic transformation.
    pose_cam = extrinsic * pose_imu

    Args:
        pose: IMU pose dict
        extrinsic: Dict with 'position' and 'quaternion' keys

    Returns:
        Transformed pose dict
    """
    # IMU rotation and position
    R_imu = Rotation.from_quat([pose["rx"], pose["ry"], pose["rz"], pose["rw"]])
    t_imu = np.array([pose["tx"], pose["ty"], pose["tz"]])

    # Extrinsic rotation and position
    R_ext = Rotation.from_quat(extrinsic["quaternion"])
    t_ext = np.array(extrinsic["position"])

    # Combined transform
    R_cam = R_imu * R_ext
    t_cam = R_imu.apply(t_ext) + t_imu

    qx, qy, qz, qw = R_cam.as_quat()

    return {
        "timestamp": pose["timestamp"],
        "tx": t_cam[0],
        "ty": t_cam[1],
        "tz": t_cam[2],
        "rx": qx,
        "ry": qy,
        "rz": qz,
        "rw": qw,
    }


def load_calibration_from_pb(calib_file):
    """Load camera calibration from protobuf file."""
    calib = SensorCalib()
    with open(calib_file, "rb") as f:
        calib.ParseFromString(f.read())
    return calib


def load_poses_from_txt(poses_file):
    """Load poses from text file format.
    Expected format: timestamp tx ty tz rx ry rz rw
    """
    poses = []
    with open(poses_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                pose_dict = {
                    "timestamp": float(parts[0]),
                    "tx": float(parts[1]),
                    "ty": float(parts[2]),
                    "tz": float(parts[3]),
                    "rx": float(parts[4]),
                    "ry": float(parts[5]),
                    "rz": float(parts[6]),
                    "rw": float(parts[7]),
                }
                poses.append(pose_dict)
    return poses


def process_poses(poses_file, calib_file, image_folder, output_file, image_list=None):
    """
    Process image timestamps with IMU trajectory to compute camera poses.
    Handles multiple cameras in cam0, cam1, ... folders.

    Args:
        poses_file: Path to IMU trajectory file (text format)
        calib_file: Path to calibration file (protobuf with camera-IMU extrinsic)
        image_folder: Path to parent folder containing cam0, cam1, ... subdirectories
        output_file: Path to output ImgPose.txt file
        image_list: Optional list of image filenames (in order)
    """

    # Load IMU poses
    imu_poses = load_poses_from_txt(poses_file)

    if not imu_poses:
        print(f"[ERROR] No IMU poses found in {poses_file}")
        return False

    print(f"[OK] Loaded {len(imu_poses)} IMU poses from {poses_file}")

    # Load calibration
    calib = load_calibration_from_pb(calib_file)

    print(f"[OK] Loaded calibration from {calib_file}")

    # Get camera-IMU extrinsic parameters
    if not hasattr(calib, "camera_param") or len(calib.camera_param) == 0:
        print("[ERROR] No camera_param extrinsic found in calibration file")
        return False

    num_cameras = len(calib.camera_param)
    print(f"[OK] Found {num_cameras} camera(s) in calibration")

    # Find all camera folders (cam0, cam1, ...)
    image_folder = Path(image_folder)
    camera_folders = {}

    for cam_idx in range(num_cameras):
        cam_folder = image_folder / f"cam{cam_idx}"
        if cam_folder.exists() and cam_folder.is_dir():
            camera_folders[cam_idx] = cam_folder

    if not camera_folders:
        print(f"[ERROR] No camera folders (cam0, cam1, ...) found in {image_folder}")
        return False

    print(
        f"[OK] Found {len(camera_folders)} camera folder(s): {sorted(camera_folders.keys())}"
    )

    # Write output
    processed_files = set()  # Track processed files to avoid duplicates

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")

        # Write header
        writer.writerow(
            [
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
        )

        # Process each camera
        valid_count = 0
        for cam_idx in sorted(camera_folders.keys()):
            cam_folder = camera_folders[cam_idx]

            # Get camera extrinsic for this camera
            cam_imu_ext = calib.camera_param[cam_idx]
            ext = cam_imu_ext.extrinsic
            extrinsic = {
                "position": np.array([ext.tx, ext.ty, ext.tz]),
                "quaternion": np.array([ext.rx, ext.ry, ext.rz, ext.rw]),
                "time_offset": cam_imu_ext.extrinsic.time_offset,
            }

            print(f"\n  Processing cam{cam_idx}...")

            # Recursively find images in this camera folder
            image_files = []
            for img_ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                image_files.extend(cam_folder.rglob(img_ext))

            image_files.sort()
            print(f"  Found {len(image_files)} images in cam{cam_idx}")

            # Process each image
            cam_valid_count = 0
            for img_path in image_files:
                img_name = img_path.name
                # Get relative path from image_folder (e.g., cam0/image.jpg)
                img_relative_path = img_path.relative_to(image_folder)

                # Skip if already processed (avoid duplicates)
                path_str = str(img_relative_path.as_posix())
                if path_str in processed_files:
                    continue
                processed_files.add(path_str)

                # Extract timestamp from filename
                timestamp = extract_timestamp_from_filename(img_name)
                if timestamp is None:
                    continue

                # Interpolate IMU pose at this timestamp
                imu_pose = interpolate_pose(
                    timestamp + extrinsic["time_offset"], imu_poses
                )
                if imu_pose is None:
                    continue

                # Apply extrinsic transformation to get camera pose
                cam_pose = apply_extrinsic_transform(imu_pose, extrinsic)

                # Write row with relative path
                writer.writerow(
                    [
                        path_str,
                        cam_pose["tx"],
                        cam_pose["ty"],
                        cam_pose["tz"],
                        0.0,
                        0.0,
                        0.0,
                        cam_pose["rx"],
                        cam_pose["ry"],
                        cam_pose["rz"],
                        cam_pose["rw"],
                        timestamp,
                    ]
                )

                cam_valid_count += 1
                valid_count += 1

            print(f"  [OK] Processed {cam_valid_count} images in cam{cam_idx}")

    print(f"\n[OK] Successfully written {valid_count} total poses to {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute image poses from IMU trajectory and camera-IMU extrinsics"
    )
    parser.add_argument(
        "--poses-file",
        "-p",
        default=r"D:/slam/output/traj.txt",
        help="Path to IMU trajectory file (text format) (default: D:\\slam\\output\\traj.txt)",
    )
    parser.add_argument(
        "--calib-file",
        "-c",
        default=r"D:/slam/calibration.dat",
        help="Path to calibration file with camera extrinsics (protobuf) (default: D:\\slam\\calibration.dat)",
    )
    parser.add_argument(
        "--image-folder",
        "-i",
        default=r"D:/slam/camera",
        help="Path to parent folder containing cam0, cam1, ... subdirectories (default: D:\\slam\\camera)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=r"D:/slam/camera/ImgPose.txt",
        help="Output file path (default: D:\\slam\\camera\\ImgPose.txt)",
    )
    parser.add_argument(
        "--image-list",
        help="Path to file containing list of image names (one per line)",
    )

    args = parser.parse_args()

    # Load image list if provided
    image_list = None
    if args.image_list:
        with open(args.image_list, "r") as f:
            image_list = [line.strip() for line in f if line.strip()]

    # Process poses
    success = process_poses(
        args.poses_file, args.calib_file, args.image_folder, args.output, image_list
    )

    return 0 if success else 1


if __name__ == "__main__":
    main()
