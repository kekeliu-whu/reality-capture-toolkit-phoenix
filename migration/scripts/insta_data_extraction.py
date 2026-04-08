import logging
import os
import telemetry_parser
from proto.sensors_pb2 import ImuMsgList
import math
import csv
from pathlib import Path
import subprocess
import json
import cv2
import argparse
import sys

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

# ============================================================
# 配置参数和路径 (所有设置放在这里)
# ============================================================

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Scientific camera data extraction tool")

# --------- 输入文件路径 ---------
parser.add_argument(
    "--input-video-filename",
    type=str,
    default=R"D:\Users\rick\Downloads\2026-03-25_09-41-05-1\VID_20260325_094100_00_226.insv",
    help="Input video file path",
)

# --------- 输出目录 ---------
parser.add_argument(
    "--output-dir",
    type=str,
    default="D:/output/images",
    help="Output directory",
)

# --------- 时间偏移参数 ---------
parser.add_argument(
    "--time-offset",
    type=float,
    default=1735726109.5283203,
    help="Time offset in seconds",
)

# --------- 数据导出参数 ---------
parser.add_argument(
    "--export-frames",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to export camera frames (default: enabled; use --no-export-frames to disable)",
)
parser.add_argument(
    "--frame-sample-rate",
    type=int,
    default=12,
    help="Export one frame every N frames (default: 12)",
)

# 解析命令行参数
args = parser.parse_args()

# 从参数中获取配置值
INPUT_VIDEO = args.input_video_filename
OUTPUT_DIRECTORY = args.output_dir
TIME_OFFSET_SECS = args.time_offset
EXPORT_FRAMES = args.export_frames
FRAME_SAMPLE_RATE = args.frame_sample_rate

# IMU输出文件路径（保存在输出目录中）
IMU_OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, "insv.dat")

# --------- 视频提取参数 ---------
QUALITY = 2  # 0-31, 越低质量越好
NUM_STREAMS = 2  # 摄像头流数量 (cam0, cam1等)

# --------- ROS/Rosbag 参数 ---------
SAVE_TO_ROSBAG = False  # 是否保存数据到rosbag文件

# ============================================================
# 函数定义
# ============================================================


def extract_frames_from_video(
    input_video_path,
    output_base_dir=".",
    quality=2,
    num_streams=2,
    frame_sample_rate=1,
    all_timestamps=None,
):
    """
    从视频文件提取帧到多个摄像头目录，支持采样和时间戳对应

    参数:
        input_video_path: 输入视频文件路径 (.insv 或 .mp4)
        output_base_dir: 输出基础目录
        quality: 视频质量 (0-31, 越低越好)
        num_streams: 流的数量 (默认2个: cam0, cam1)
        frame_sample_rate: 采样率，每隔N帧取1帧 (1表示取所有帧, 2表示每2帧取1帧)
        all_timestamps: 所有帧的时间戳列表，用于重命名文件

    返回:
        包含采样后的时间戳和每个摄像头的图片列表的字典
        {'timestamps': [...], 'cam0': [...], 'cam1': [...], ...}
    """
    input_video_path = str(input_video_path)
    output_base_dir = str(output_base_dir)

    # 创建输出目录，并清空旧的帧文件
    for i in range(num_streams):
        cam_dir = os.path.join(output_base_dir, f"cam{i}")
        os.makedirs(cam_dir, exist_ok=True)

        # 清空目录中的旧帧文件（.jpg, temp_*.jpg 及时间戳格式的文件）
        for old_file in os.listdir(cam_dir):
            if old_file.endswith(".jpg") or old_file.startswith("temp_"):
                old_path = os.path.join(cam_dir, old_file)
                try:
                    os.remove(old_path)

                except Exception as e:
                    print(f"  [WARN] Failed to delete old file {old_file}: {e}")

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
                ]
            )
            print(f"  [OK] Extracted {len(temp_frames)} frames for cam{stream_idx}")

            # 根据采样率进行采样并重命名文件
            sampled_images = []
            sampled_timestamps = []
            for idx in range(0, len(temp_frames), frame_sample_rate):
                temp_frame = temp_frames[idx]
                temp_path = os.path.join(cam_dir, temp_frame)

                # 确定新文件名
                if all_timestamps and idx < len(all_timestamps):
                    timestamp = all_timestamps[idx]
                    # 文件名格式: timestamp.jpg (例如: 1735732913.646514.jpg，6位小数)
                    timestamp_str = f"{timestamp:.6f}".replace(".", "_")
                    new_name = f"{timestamp_str}.jpg"
                else:
                    continue  # 跳过没有时间戳的帧，确保文件名与时间戳对应

                new_path = os.path.join(cam_dir, new_name)

                # 重命名文件
                try:
                    os.rename(temp_path, new_path)
                    sampled_images.append(new_name)
                    sampled_timestamps.append(all_timestamps[idx])
                except Exception as e:
                    print(f"  [WARN] Failed to rename file {temp_frame}: {e}")
                    continue

            # 删除未采样的临时文件
            remaining_temps = [
                f
                for f in os.listdir(cam_dir)
                if f.startswith("temp_") and f.endswith(".jpg")
            ]
            for temp_file in remaining_temps:
                try:
                    os.remove(os.path.join(cam_dir, temp_file))
                except Exception as e:
                    print(f"  [WARN] Failed to delete unsampled file {temp_file}: {e}")

            print(
                f"  [OK] cam{stream_idx} sampling complete: {len(sampled_images)} frames (sample rate: {frame_sample_rate})"
            )

            # 保存图片列表
            image_lists[f"cam{stream_idx}"] = sampled_images

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to extract frames for cam{stream_idx}")
            print(f"  stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            print("[ERROR] ffmpeg was not found. Make sure it is installed and available in PATH")
            raise

    # 将时间戳添加到返回值
    result = {"timestamps": sampled_timestamps}
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

# 开始处理
print("=" * 60)
print("Starting data extraction and processing pipeline...")
print("=" * 60)

# 初始化解析器
tp = telemetry_parser.Parser(INPUT_VIDEO)

# 确保输出目录存在
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# 获取归一化的IMU数据
imu_data = tp.normalized_imu()

# 将IMU数据转换为protobuf格式
imu_msg_list = ImuMsgList()

for imu_sample in imu_data:
    imu_msg = imu_msg_list.imu_msgs.add()
    imu_msg.timestamp = imu_sample["timestamp_ms"] / 1000.0  # 转换为秒并加上时间偏移
    imu_msg.gx = imu_sample["gyro"][0] / 180.0 * math.pi  # 转换为弧度
    imu_msg.gy = imu_sample["gyro"][1] / 180.0 * math.pi
    imu_msg.gz = imu_sample["gyro"][2] / 180.0 * math.pi
    imu_msg.ax = imu_sample["accl"][0]
    imu_msg.ay = imu_sample["accl"][1]
    imu_msg.az = imu_sample["accl"][2]

# 将protobuf消息写入二进制文件
with open(IMU_OUTPUT_FILE, "wb") as f:
    f.write(imu_msg_list.SerializeToString())

# 验证写入成功
file_size = os.path.getsize(IMU_OUTPUT_FILE)
print(f"[OK] Wrote output to: {IMU_OUTPUT_FILE}")
print(f"  File size: {file_size} bytes")
print(f"  IMU sample count: {len(imu_msg_list.imu_msgs)}")
print(f"  Camera: {tp.camera}")
print(f"  Model: {tp.model}")

if not EXPORT_FRAMES:
    print("\n" + "=" * 60)
    print("[WARN] Camera frames were not exported, timestamps were not consumed")
    print("=" * 60)
    sys.exit(0)

# 获取时间戳数据（在提取帧之前）
print("\n" + "=" * 60)
print("Preparing timestamp data...")
print("=" * 60)
exposure_data = tp.telemetry()[0]["Exposure"]["Data"]
all_timestamps = [e["t"] + TIME_OFFSET_SECS for e in exposure_data]  # 加上时间偏移
print(f"[OK] Loaded {len(all_timestamps)} timestamps")

# ============================================================
# 执行逻辑
# ============================================================

# 调用函数提取帧
print("\n" + "=" * 60)
print(f"Starting frame extraction (sample rate: {FRAME_SAMPLE_RATE})...")
print("=" * 60)
result = extract_frames_from_video(
    input_video_path=INPUT_VIDEO,
    output_base_dir=OUTPUT_DIRECTORY,
    quality=QUALITY,
    num_streams=NUM_STREAMS,
    frame_sample_rate=FRAME_SAMPLE_RATE,
    all_timestamps=all_timestamps,
)

# 从返回结果中分离时间戳和图片列表
sampled_timestamps = result["timestamps"]
image_lists = {k: v for k, v in result.items() if k != "timestamps"}

# 统计提取的帧数
print("\n" + "=" * 60)
print("Frame extraction summary:")
print("=" * 60)
for cam, images in image_lists.items():
    print(f"  {cam}: {len(images)} frames")

print(f"\n[OK] Original timestamp count: {len(all_timestamps)}")
print(f"  Sampled timestamp count: {len(sampled_timestamps)}")
print(f"  Sample rate: {FRAME_SAMPLE_RATE}")
if len(sampled_timestamps) > 0:
    print(f"  Timestamp range: {sampled_timestamps[0]:.6f} to {sampled_timestamps[-1]:.6f}")

# 验证帧数和时间戳是否匹配
print("\n" + "=" * 60)
print("Frame and timestamp validation:")
print("=" * 60)
frame_counts = {cam: len(images) for cam, images in image_lists.items()}
for cam, frame_count in frame_counts.items():
    match_status = "[OK] match" if frame_count == len(sampled_timestamps) else "[ERROR] mismatch"
    print(
        f"  {cam}: {frame_count} frames vs {len(sampled_timestamps)} timestamps {match_status}"
    )

# 保存到 rosbag（可选）
if SAVE_TO_ROSBAG:
    print("\n" + "=" * 60)
    print("Saving data to rosbag...")
    print("=" * 60)
    save_to_rosbag(
        output_dir=OUTPUT_DIRECTORY,
        imu_msg_list=imu_msg_list,
        timestamps=sampled_timestamps,
        num_streams=NUM_STREAMS,
        timeoffset_secs=TIME_OFFSET_SECS,
    )
else:
    print("\n" + "=" * 60)
    print("[WARN] Skipping rosbag export (SAVE_TO_ROSBAG = False)")
    print("=" * 60)

print("\n" + "=" * 60)
print("[OK] Data processing completed")
print("=" * 60)
