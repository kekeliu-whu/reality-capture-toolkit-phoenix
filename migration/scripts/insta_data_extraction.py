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


# LOGGER = init_spdlog_like_logger()
SCRIPT_PATH = Path(__file__).resolve()
# When running as a PyInstaller .exe, sys.executable is the .exe itself (inside build-pack/).
# Use the exe's directory as the build-pack root so MediaSDK can be found at build-pack/MediaSDK/.
# When running as a plain script, derive paths from the repo root via __file__.
if getattr(sys, 'frozen', False):
    _BUILD_PACK = Path(sys.executable).parent
    REPO_ROOT = _BUILD_PACK.parent
    DEFAULT_MEDIASDK_EXE = _BUILD_PACK / "MediaSDK" / "MediaSDKTest.exe"
else:
    REPO_ROOT = SCRIPT_PATH.parents[2]
    DEFAULT_MEDIASDK_EXE = REPO_ROOT / "build-pack" / "MediaSDK" / "MediaSDKTest.exe"

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
    default=R"Z:\rick\dataset\q9000\MT20260424-112349-fisheye\VID_20260424_112349_00_189.insv",
    help="Input video file path",
)

# --------- 输出目录 ---------
parser.add_argument(
    "--output-dir",
    type=str,
    default=R"Z:\rick\dataset\q9000\MT20260424-112349-fisheye\output\images",
    help="Output directory",
)

# --------- 时间偏移参数 ---------
parser.add_argument(
    "--time-offset",
    type=float,
    default=1749886801.779175,
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
MEDIASDK_EXE = str(DEFAULT_MEDIASDK_EXE)
PANORAMA_CAMERA_INDEX = 0
PANORAMA_STITCH_TYPE = "optflow"
PANORAMA_IMAGE_TYPE = "jpg"

# --------- ROS/Rosbag 参数 ---------
SAVE_TO_ROSBAG = False  # 是否保存数据到rosbag文件

# ============================================================
# 函数定义
# ============================================================


def format_timestamp_filename(timestamp, image_type="jpg"):
    timestamp_str = f"{timestamp:.6f}".replace(".", "_")
    return f"{timestamp_str}.{image_type}"


def build_sampled_frame_indices(total_count, frame_sample_rate):
    if frame_sample_rate <= 0:
        raise ValueError("frame_sample_rate must be greater than 0")
    return list(range(0, total_count, frame_sample_rate))


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


def detect_panorama_output_size(video_path):
    """Detect insv video resolution (n x n) and return panorama output size '2n x n'."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    n = height
    output_size = f"{2 * n}x{n}"
    print(f"[OK] Detected video resolution: {width}x{height} -> panorama output: {output_size}")
    return output_size


def extract_panorama_frames_from_mediasdk(
    input_video_path,
    output_base_dir=".",
    media_sdk_exe=DEFAULT_MEDIASDK_EXE,
    camera_index=0,
    frame_sample_rate=12,
    all_timestamps=None,
    stitch_type="optflow",
    image_type="jpg",
    output_size="7680x3840",
):
    """
    使用 MediaSDK 导出全景图，并按时间戳重命名到 cam{N} 目录。

    延时摄影视频的实际帧数 = len(all_timestamps) / frame_sample_rate，
    因此导出全部视频帧，第 i 帧对应 all_timestamps[i * frame_sample_rate]。
    """
    if not all_timestamps:
        raise ValueError("all_timestamps is required for panorama export")

    media_sdk_exe = Path(media_sdk_exe)
    if not media_sdk_exe.exists():
        raise FileNotFoundError(f"MediaSDK executable not found: {media_sdk_exe}")

    build_pack_dir = media_sdk_exe.parent.parent  # build-pack/ is parent of MediaSDK/
    media_sdk_dir = media_sdk_exe.parent

    # 第一帧对应 all_timestamps[0]（即 exposure_data[0]），后续每帧 +0.5s
    base_timestamp = all_timestamps[0]

    cam_dir = Path(output_base_dir) / f"cam{camera_index}"
    clean_camera_output_dir(str(cam_dir), extensions=(f".{image_type}",))

    temp_export_dir = cam_dir / "_mediasdk_tmp"
    if temp_export_dir.exists():
        for old_path in temp_export_dir.iterdir():
            if old_path.is_file():
                old_path.unlink()
    else:
        temp_export_dir.mkdir(parents=True, exist_ok=True)

    # 导出全部视频帧（不传 -export_frame_index）
    cmd = [
        str(media_sdk_exe),
        "-inputs",
        str(input_video_path),
        "-image_sequence_dir",
        str(temp_export_dir),
        "-output_size",
        output_size,
        "-stitch_type",
        stitch_type,
        "-image_type",
        image_type,
        "-disable_cuda",
        "false",
    ]

    print(f"[OK] Processing panorama export into cam{camera_index}...")
    print(f"  MediaSDK executable: {media_sdk_exe}")
    print(f"  Exporting all video frames...")

    env = os.environ.copy()
    path_entries = [str(media_sdk_dir), str(build_pack_dir)]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_entries + ([existing_path] if existing_path else []))

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(media_sdk_dir),
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to export panorama frames for cam{camera_index}")
        if e.returncode == 3221225781:
            print("  MediaSDKTest.exe is missing one or more dependent DLLs in its process PATH")
            print(f"  PATH included: {media_sdk_dir}; {build_pack_dir}")
        print(f"  stderr: {e.stderr}")
        raise

    exported_images = sorted(
        temp_export_dir.glob(f"*.{image_type}"),
        key=natural_sort_key,
    )
    if not exported_images:
        raise RuntimeError(f"MediaSDK produced no .{image_type} files in {temp_export_dir}")

    print(f"  MediaSDK exported {len(exported_images)} images")

    # 排序后按顺序分配时间戳：第 i 帧 = exposure_data[0] + i * 0.5
    sampled_images = []
    finalized_timestamps = []
    for video_idx, img_path in enumerate(exported_images):
        timestamp = base_timestamp + video_idx * 0.5
        new_name = format_timestamp_filename(timestamp, image_type=image_type)
        new_path = cam_dir / new_name
        img_path.replace(new_path)
        sampled_images.append(new_name)
        finalized_timestamps.append(timestamp)

    for remaining_file in temp_export_dir.iterdir():
        if remaining_file.is_file():
            remaining_file.unlink()
    temp_export_dir.rmdir()

    print(
        f"  [OK] cam{camera_index} panorama export complete: {len(sampled_images)} frames"
        f" (base_ts={base_timestamp:.6f}, step=0.5s)"
    )

    return {
        "timestamps": finalized_timestamps,
        f"cam{camera_index}": sampled_images,
    }


def extract_frames_from_video(
    input_video_path,
    output_base_dir,
    quality,
    num_streams,
    frame_sample_rate,
    all_timestamps,
):
    """
    从视频文件提取帧到多个摄像头目录，并按固定 0.5s 步长重命名

    参数:
        input_video_path: 输入视频文件路径 (.insv 或 .mp4)
        output_base_dir: 输出基础目录
        quality: 视频质量 (0-31, 越低越好)
        num_streams: 流的数量 (默认2个: cam0, cam1)
        frame_sample_rate: 保留原接口参数，当前鱼眼导出不再做采样删除
        all_timestamps: 时间戳列表，仅使用首个时间戳作为起点

    返回:
        包含导出后的时间戳和每个摄像头的图片列表的字典
        {'timestamps': [...], 'cam0': [...], 'cam1': [...], ...}
    """
    input_video_path = str(input_video_path)
    output_base_dir = str(output_base_dir)
    if not all_timestamps:
        raise ValueError("all_timestamps is required for fisheye export")

    base_timestamp = all_timestamps[0]
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
                ]
            )
            print(f"  [OK] Extracted {len(temp_frames)} frames for cam{stream_idx}")

            # 与全景导出保持一致：首帧对应首个 timestamp，后续每帧 +0.5s
            exported_images = []
            current_stream_timestamps = []
            for frame_idx, temp_frame in enumerate(temp_frames):
                temp_path = os.path.join(cam_dir, temp_frame)

                timestamp = base_timestamp + frame_idx * 0.5
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

            if not exported_timestamps:
                exported_timestamps = current_stream_timestamps

            print(
                f"  [OK] cam{stream_idx} export complete: {len(exported_images)} frames"
                f" (base_ts={base_timestamp:.6f}, step=0.5s)"
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

# 自动检测视频分辨率，计算全景输出尺寸
panorama_output_size = detect_panorama_output_size(INPUT_VIDEO)

# 调用函数提取帧
print("\n" + "=" * 60)
print("Starting fisheye frame extraction...")
print("=" * 60)
# result = extract_panorama_frames_from_mediasdk(
#     input_video_path=INPUT_VIDEO,
#     output_base_dir=OUTPUT_DIRECTORY,
#     media_sdk_exe=MEDIASDK_EXE,
#     camera_index=PANORAMA_CAMERA_INDEX,
#     frame_sample_rate=FRAME_SAMPLE_RATE,
#     all_timestamps=all_timestamps,
#     stitch_type=PANORAMA_STITCH_TYPE,
#     image_type=PANORAMA_IMAGE_TYPE,
#     output_size=panorama_output_size,
# )
result = extract_frames_from_video(
    input_video_path=INPUT_VIDEO,
    output_base_dir=OUTPUT_DIRECTORY,
    quality=QUALITY,
    num_streams=NUM_STREAMS,
    frame_sample_rate=FRAME_SAMPLE_RATE,
    all_timestamps=all_timestamps,
)

# 从返回结果中分离时间戳和图片列表
exported_timestamps = result["timestamps"]
image_lists = {k: v for k, v in result.items() if k != "timestamps"}

# 统计提取的帧数
print("\n" + "=" * 60)
print("Frame extraction summary:")
print("=" * 60)
for cam, images in image_lists.items():
    print(f"  {cam}: {len(images)} frames")

print(f"\n[OK] Original timestamp count: {len(all_timestamps)}")
print(f"  Exported timestamp count: {len(exported_timestamps)}")
print("  Timestamp rule: base timestamp + 0.5s per frame")
if len(exported_timestamps) > 0:
    print(f"  Timestamp range: {exported_timestamps[0]:.6f} to {exported_timestamps[-1]:.6f}")

# 验证帧数和时间戳是否匹配
print("\n" + "=" * 60)
print("Frame and timestamp validation:")
print("=" * 60)
frame_counts = {cam: len(images) for cam, images in image_lists.items()}
for cam, frame_count in frame_counts.items():
    match_status = "[OK] match" if frame_count == len(exported_timestamps) else "[ERROR] mismatch"
    print(
        f"  {cam}: {frame_count} frames vs {len(exported_timestamps)} timestamps {match_status}"
    )

# 保存到 rosbag（可选）
if SAVE_TO_ROSBAG:
    print("\n" + "=" * 60)
    print("Saving data to rosbag...")
    print("=" * 60)
    save_to_rosbag(
        output_dir=OUTPUT_DIRECTORY,
        imu_msg_list=imu_msg_list,
        timestamps=exported_timestamps,
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
