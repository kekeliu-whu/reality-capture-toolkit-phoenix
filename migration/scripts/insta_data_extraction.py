import logging
import os
import telemetry_parser
from proto.sensors_pb2 import ImuMsgList
import math
import csv
from pathlib import Path
import subprocess
import json
import argparse

from spdlog_compat import init_spdlog_like_logger


LOGGER = init_spdlog_like_logger()

# ============================================================
# 尝试导入ROS相关库（可选）
# ============================================================
ROSBAG_AVAILABLE = False
try:
    import rosbag
    from sensor_msgs.msg import Imu, Image, CameraInfo
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Vector3

    ROSBAG_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Failed to import ROS-related libraries: {e}")
    print("  rosbag export will be disabled, but other features remain available")

def parse_args() -> argparse.Namespace:
    # ============================================================
    # 配置参数和路径 (所有设置放在这里)
    # ============================================================

    # [CONFIG] 命令行参数默认值 - 修改此处便于手动运行
    parser = argparse.ArgumentParser(
        description="Scientific camera data extraction tool"
    )
    parser.add_argument(
        "--input-video-filename",
        type=str,
        default=R"Z:\rick\dataset\q9000\MT20260430-112900-collect-by-app-only-test\insta\1749886847469595719_00.insv",
        help="Input video file path",
    )
    parser.add_argument(
        "--imu-file",
        type=str,
        default=R"Z:\rick\dataset\q9000\MT20260430-112900-collect-by-app-only-test\output\imu.dat",
        help="Path to device imu.dat.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=R"Z:\rick\dataset\q9000\MT20260430-112900-collect-by-app-only-test\output\images",
        help="Output directory",
    )
    parser.add_argument(
        "--frame-sample-rate",
        type=int,
        default=1,
        help="Keep exported frames at indices 0, N, 2N... from telemetry-aligned frames",
    )
    return parser.parse_args()

# --------- 视频提取参数 ---------
QUALITY = 2  # 0-31, 越低质量越好
NUM_STREAMS = 2  # 摄像头流数量 (cam0, cam1等)

# --------- ROS/Rosbag 参数 ---------
SAVE_TO_ROSBAG = False  # 是否保存数据到rosbag文件

# ============================================================
# 函数定义
# ============================================================


def format_timestamp_filename(timestamp, image_type="jpg"):
    timestamp_str = f"{timestamp:.6f}".replace(".", "_")
    return f"{timestamp_str}.{image_type}"


def clean_camera_output_dir(cam_dir, extensions=(".jpg",)):
    os.makedirs(cam_dir, exist_ok=True)
    extensions = tuple(ext.lower() for ext in extensions)
    for old_file in os.listdir(cam_dir):
        old_path = os.path.join(cam_dir, old_file)
        if os.path.isdir(old_path):
            continue
        if old_file.startswith("temp_") or old_file.lower().endswith(extensions):
            try:
                os.remove(old_path)
            except Exception as e:
                print(f"  [WARN] Failed to delete old file {old_file}: {e}")


def natural_sort_key(path_obj):
    stem = path_obj.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return (0, int(digits), stem)
    return (1, stem)


def parse_exported_frame_index(path_obj):
    stem = path_obj.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame index from exported file: {path_obj.name}")
    return int(digits)


def build_imu_msg_list_from_telemetry(telemetry_record):
    imu_msg_list = ImuMsgList()

    meta = telemetry_record["Default"]["Metadata"]
    first_frame_ts_us = meta["first_frame_timestamp"]
    gyro_base_ts_us = first_frame_ts_us + round(meta["gyro_timestamp"] * 1000)

    gyro_scale = telemetry_record["Gyroscope"]["Scale"]
    accel_scale = telemetry_record["Accelerometer"]["Scale"]
    for gyro_sample, accel_sample in zip(
        telemetry_record["Gyroscope"]["Data"],
        telemetry_record["Accelerometer"]["Data"],
    ):
        imu_msg = imu_msg_list.imu_msgs.add()
        imu_msg.timestamp = (
            round(gyro_base_ts_us + gyro_sample["t"] * 1_000_000) / 1_000_000.0
        )
        imu_msg.gx = gyro_sample["y"] / gyro_scale * math.pi / 180.0
        imu_msg.gy = -gyro_sample["x"] / gyro_scale * math.pi / 180.0
        imu_msg.gz = gyro_sample["z"] / gyro_scale * math.pi / 180.0
        imu_msg.ax = accel_sample["y"] / accel_scale
        imu_msg.ay = -accel_sample["x"] / accel_scale
        imu_msg.az = accel_sample["z"] / accel_scale

    return imu_msg_list


def build_frame_timestamps_from_telemetry(telemetry_record, time_offset_secs):
    return [timestamp_ms / 1000.0 + time_offset_secs for timestamp_ms in telemetry_record["Default"]["Timestamps"]]


def compute_time_offset_from_device_imu(device_imu_file, insta_imu_msg_list):
    device_imu_file = str(device_imu_file)

    if not os.path.exists(device_imu_file):
        raise FileNotFoundError(f"Device IMU file not found: {device_imu_file}")

    print("\n" + "=" * 60)
    print("Running IMU time synchronization...")
    print("=" * 60)
    print(f"  Device IMU: {device_imu_file}")
    print("  Insta IMU: telemetry-derived IMU (in-memory)")

    device_imu_msg_list = ImuMsgList()
    with open(device_imu_file, "rb") as f:
        device_imu_msg_list.ParseFromString(f.read())

    from insta_time_sync import compute_final_time_delay_from_imu_msg_lists

    time_offset_secs = compute_final_time_delay_from_imu_msg_lists(
        device_imu_msg_list,
        insta_imu_msg_list,
    )
    print(f"[OK] Auto-computed time offset: {time_offset_secs:.6f} seconds")
    return time_offset_secs


def extract_frames_from_video(
    input_video_path,
    output_base_dir,
    quality,
    num_streams,
    frame_sample_rate,
    all_timestamps,
):
    """
    从视频文件提取帧到多个摄像头目录，并按显式时间戳重命名

    参数:
        input_video_path: 输入视频文件路径 (.insv 或 .mp4)
        output_base_dir: 输出基础目录
        quality: 视频质量 (0-31, 越低越好)
        num_streams: 流的数量 (默认2个: cam0, cam1)
        frame_sample_rate: 仅保留索引 0, N, 2N... 的对齐帧
        all_timestamps: 导出图片使用的显式时间戳列表

    返回:
        包含导出后的时间戳和每个摄像头的图片列表的字典
        {'timestamps': [...], 'cam0': [...], 'cam1': [...], ...}
    """
    input_video_path = str(input_video_path)
    output_base_dir = str(output_base_dir)
    if not all_timestamps:
        raise ValueError("all_timestamps is required for fisheye export")
    if frame_sample_rate <= 0:
        raise ValueError("frame_sample_rate must be greater than 0")

    exported_timestamps = []

    # 创建输出目录，并清空旧的帧文件
    for i in range(num_streams):
        cam_dir = os.path.join(output_base_dir, f"cam{i}")
        clean_camera_output_dir(cam_dir, extensions=(".jpg",))

    # 为每个摄像头流提取帧
    image_lists = {}

    for stream_idx in range(num_streams):
        cam_dir = os.path.join(output_base_dir, f"cam{stream_idx}")

        print(f"[OK] Processing cam{stream_idx}...")

        try:
            # 使用ffmpeg提取stream到临时文件
            temp_output = os.path.join(cam_dir, "temp_%05d.jpg")

            # 用ffmpeg导出所有帧
            cmd = [
                "ffmpeg",
                "-threads",
                "8",
                "-hwaccel",
                "auto",
                "-i",
                input_video_path,
                "-map",
                f"0:{stream_idx}",
                "-q:v",
                str(quality),
                "-threads",
                "8",
                temp_output,
            ]

            print("  Extracting raw frames...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # 获取所有临时帧文件
            temp_frames = sorted(
                [
                    f
                    for f in os.listdir(cam_dir)
                    if f.startswith("temp_") and f.endswith(".jpg")
                ],
                key=lambda name: natural_sort_key(Path(name)),
            )
            print(f"  [OK] Extracted {len(temp_frames)} frames for cam{stream_idx}")

            if len(temp_frames) != len(all_timestamps):
                print(
                    "  [WARN] Extracted frame count "
                    f"({len(temp_frames)}) differs from timestamp count "
                    f"({len(all_timestamps)}); exporting first "
                    f"{min(len(temp_frames), len(all_timestamps))} aligned frames"
                )

            aligned_frame_count = min(len(temp_frames), len(all_timestamps))
            sampled_indices = range(0, aligned_frame_count, frame_sample_rate)
            print(
                f"  Exporting sampled frames: {len(range(0, aligned_frame_count, frame_sample_rate))} "
                f"from {aligned_frame_count} aligned frames"
            )

            exported_images = []
            current_stream_timestamps = []
            for exported_idx in sampled_indices:
                temp_frame = temp_frames[exported_idx]
                temp_path = os.path.join(cam_dir, temp_frame)

                timestamp = all_timestamps[exported_idx]
                new_name = format_timestamp_filename(timestamp, image_type="jpg")

                new_path = os.path.join(cam_dir, new_name)

                # 重命名文件
                try:
                    os.rename(temp_path, new_path)
                    exported_images.append(new_name)
                    current_stream_timestamps.append(timestamp)
                except Exception as e:
                    print(f"  [WARN] Failed to rename file {temp_frame}: {e}")
                    continue

            for temp_frame in temp_frames:
                temp_path = os.path.join(cam_dir, temp_frame)
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        print(f"  [WARN] Failed to delete unsampled file {temp_frame}: {e}")

            if not exported_timestamps:
                exported_timestamps = current_stream_timestamps

            print(
                f"  [OK] cam{stream_idx} export complete: {len(exported_images)} frames"
                " (explicit timestamps)"
            )

            # 保存图片列表
            image_lists[f"cam{stream_idx}"] = exported_images

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to extract frames for cam{stream_idx}")
            print(f"  stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            print("[ERROR] ffmpeg was not found. Make sure it is installed and available in PATH")
            raise

    # 将时间戳添加到返回值
    result = {"timestamps": exported_timestamps}
    result.update(image_lists)

    return result


def count_frames_in_output(output_dir):
    """计算输出目录中的帧数"""
    cam_dirs = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("cam") and os.path.isdir(os.path.join(output_dir, d))
    ]
    frame_counts = {}
    for cam_dir in sorted(cam_dirs):
        cam_path = os.path.join(output_dir, cam_dir)
        frame_count = len([f for f in os.listdir(cam_path) if f.endswith(".jpg")])
        frame_counts[cam_dir] = frame_count
    return frame_counts


def save_to_rosbag(
    output_dir,
    imu_msg_list,
    timestamps,
    num_streams=2,
    timeoffset_secs=0.0,
):
    """
    将IMU和相机数据保存到ROS bag文件

    参数:
        output_dir: 输出基础目录
        imu_msg_list: Protobuf格式的IMU消息列表
        timestamps: 相机时间戳列表
        num_streams: 流的数量 (cam0, cam1等)
        timeoffset_secs: 时间偏移量 (秒)
    """
    if not ROSBAG_AVAILABLE:
        print("[ERROR] rosbag is unavailable or the ROS environment is missing, skipping rosbag export")
        print("  Install rosbag support if you need rosbag export")
        return False

    try:
        import rosbag
        from sensor_msgs.msg import Imu, Image
        from std_msgs.msg import Header
        from cv_bridge import CvBridge
        import cv2
        from geometry_msgs.msg import Vector3

        bag_path = os.path.join(output_dir, "images.bag")
        bridge = CvBridge()

        print(f"[OK] Creating rosbag file: {bag_path}")

        with rosbag.Bag(bag_path, "w") as bag:
            # ==================== 保存 IMU 数据 ====================
            print("  Writing IMU data...")
            imu_count = 0
            for imu_msg in imu_msg_list.imu_msgs:
                # 创建ROS Imu消息
                timestamp = imu_msg.timestamp + timeoffset_secs

                imu_ros = Imu()
                imu_ros.header.seq = imu_count
                imu_ros.header.stamp.secs = int(timestamp)
                imu_ros.header.stamp.nsecs = int((timestamp % 1.0) * 1e9)
                imu_ros.header.frame_id = "imu"

                # 设置角速度
                imu_ros.angular_velocity.x = imu_msg.gx
                imu_ros.angular_velocity.y = imu_msg.gy
                imu_ros.angular_velocity.z = imu_msg.gz

                # 设置线加速度
                imu_ros.linear_acceleration.x = imu_msg.ax
                imu_ros.linear_acceleration.y = imu_msg.ay
                imu_ros.linear_acceleration.z = imu_msg.az

                # 写入到bag
                bag.write("/imu0", imu_ros, t=imu_ros.header.stamp)
                imu_count += 1

            print(f"    [OK] Wrote {imu_count} IMU messages")

            # ==================== 保存 相机 数据 ====================
            for cam_idx in range(num_streams):
                cam_dir = Path(output_dir) / f"cam{cam_idx}"
                if not cam_dir.exists():
                    continue

                print(f"  Writing cam{cam_idx} images...")
                image_files = sorted(
                    cam_dir.glob("*.jpg"), key=lambda x: int(x.stem.split("_")[0])
                )

                img_count = 0
                for img_idx, img_file in enumerate(image_files):
                    if img_idx >= len(timestamps):
                        break

                    try:
                        # 读取图片
                        cv_image = cv2.imread(str(img_file))
                        if cv_image is None:
                            print(f"    [WARN] Failed to read image: {img_file}")
                            continue

                        # resize to 1/2
                        cv_image = cv2.resize(
                            cv_image, (cv_image.shape[1] // 2, cv_image.shape[0] // 2)
                        )

                        # 转换为ROS Image消息
                        ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

                        # 设置时间戳
                        timestamp = timestamps[img_idx] + timeoffset_secs
                        ros_image.header.seq = img_count
                        ros_image.header.stamp.secs = int(timestamp)
                        ros_image.header.stamp.nsecs = int((timestamp % 1.0) * 1e9)
                        ros_image.header.frame_id = f"cam{cam_idx}"

                        # 写入到bag
                        topic = f"/cam{cam_idx}/image_raw"
                        bag.write(topic, ros_image, t=ros_image.header.stamp)
                        img_count += 1
                    except Exception as e:
                        print(f"    [WARN] Failed to process image {img_file}: {str(e)}")
                        continue

                print(f"    [OK] Wrote {img_count} images for cam{cam_idx}")

        file_size = os.path.getsize(bag_path)
        print("\n[OK] rosbag file created successfully")
        print(f"  Path: {bag_path}")
        print(f"  Size: {file_size / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print("[ERROR] Failed to create rosbag file")
        print(f"  {str(e)}")
        return False


# ============================================================
# 初始化和执行逻辑
# ============================================================

def main() -> int:
    args = parse_args()
    input_video = args.input_video_filename
    output_directory = args.output_dir
    imu_file = args.imu_file
    frame_sample_rate = args.frame_sample_rate

    print("=" * 60)
    print("Starting data extraction and processing pipeline...")
    print("=" * 60)

    tp = telemetry_parser.Parser(input_video)
    os.makedirs(output_directory, exist_ok=True)
    telemetry_record = tp.telemetry()[0]

    # save tp.telemetry() to JSON for debugging
    telemetry_json_path = os.path.join(output_directory, "telemetry.json")
    with open(telemetry_json_path, "w", encoding="utf-8") as f:
        json.dump(tp.telemetry(), f, indent=2)
    print(f"[OK] Telemetry data saved to {telemetry_json_path}")

    imu_msg_list = build_imu_msg_list_from_telemetry(telemetry_record)

    print("[OK] Built telemetry-derived IMU message list")
    print(f"  IMU sample count: {len(imu_msg_list.imu_msgs)}")
    print(f"  Camera: {tp.camera}")
    print(f"  Model: {tp.model}")

    time_offset_secs = compute_time_offset_from_device_imu(
        imu_file,
        imu_msg_list,
    )

    print("\n" + "=" * 60)
    print("Preparing timestamp data...")
    print("=" * 60)
    all_timestamps = build_frame_timestamps_from_telemetry(telemetry_record, time_offset_secs)
    print(
        f"[OK] Loaded {len(all_timestamps)} timestamps with time offset {time_offset_secs:.6f}s"
    )

    print("\n" + "=" * 60)
    print("Starting fisheye frame extraction...")
    print("=" * 60)
    result = extract_frames_from_video(
        input_video_path=input_video,
        output_base_dir=output_directory,
        quality=QUALITY,
        num_streams=NUM_STREAMS,
        frame_sample_rate=frame_sample_rate,
        all_timestamps=all_timestamps,
    )

    exported_timestamps = result["timestamps"]
    image_lists = {k: v for k, v in result.items() if k != "timestamps"}

    print("\n" + "=" * 60)
    print("Frame extraction summary:")
    print("=" * 60)
    for cam, images in image_lists.items():
        print(f"  {cam}: {len(images)} frames")

    print(f"\n[OK] Original timestamp count: {len(all_timestamps)}")
    print(f"  Exported timestamp count: {len(exported_timestamps)}")
    print("  Timestamp rule: each exported frame uses the supplied telemetry timestamp")
    if len(exported_timestamps) > 0:
        print(
            f"  Timestamp range: {exported_timestamps[0]:.6f} to {exported_timestamps[-1]:.6f}"
        )

    print("\n" + "=" * 60)
    print("Frame and timestamp validation:")
    print("=" * 60)
    frame_counts = {cam: len(images) for cam, images in image_lists.items()}
    for cam, frame_count in frame_counts.items():
        match_status = (
            "[OK] match"
            if frame_count == len(exported_timestamps)
            else "[ERROR] mismatch"
        )
        print(
            f"  {cam}: {frame_count} frames vs {len(exported_timestamps)} timestamps {match_status}"
        )

    if SAVE_TO_ROSBAG:
        print("\n" + "=" * 60)
        print("Saving data to rosbag...")
        print("=" * 60)
        save_to_rosbag(
            output_dir=output_directory,
            imu_msg_list=imu_msg_list,
            timestamps=exported_timestamps,
            num_streams=NUM_STREAMS,
            timeoffset_secs=time_offset_secs,
        )
    else:
        print("\n" + "=" * 60)
        print("[WARN] Skipping rosbag export (SAVE_TO_ROSBAG = False)")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("[OK] Data processing completed")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
