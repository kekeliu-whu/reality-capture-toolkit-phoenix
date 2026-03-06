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
    print(f"⚠ 警告: 未能导入ROS相关库 - {e}")
    print("  rosbag功能将被禁用，但其他功能可正常运行")

# ============================================================
# 配置参数和路径 (所有设置放在这里)
# ============================================================

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="科研相机数据提取工具")

# --------- 输入文件路径 ---------
parser.add_argument(
    "--input-video-filename",
    type=str,
    default="\\\\wsl.localhost\\Ubuntu-24.04\\home\\rick\\iKalibr\\src\\iKalibr\\2026-02-06_11-34-29-s20\\VID_20260206_033425_00_011.insv",
    help="输入视频文件路径",
)

# --------- 输出目录 ---------
parser.add_argument(
    "--output-dir",
    type=str,
    default="D:/slam/cameras/",
    help="输出目录",
)

# --------- 时间偏移参数 ---------
parser.add_argument(
    "--time-offset",
    type=float,
    default=1735725830.203135,
    help="时间偏移量（秒）",
)

# --------- 数据导出参数 ---------
parser.add_argument(
    "--export-frames",
    action="store_true",
    default=True,
    help="是否导出相机帧（默认: 导出）",
)

# 解析命令行参数
args = parser.parse_args()

# 从参数中获取配置值
INPUT_VIDEO = args.input_video_filename
OUTPUT_DIRECTORY = args.output_dir
TIME_OFFSET_SECS = args.time_offset
EXPORT_FRAMES = args.export_frames

# IMU输出文件路径（保存在输出目录中）
IMU_OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, "insv.dat")

# --------- 视频提取参数 ---------
FRAME_SAMPLE_RATE = 12  # 采样率：每隔N帧取1帧 (1表示取所有帧, 2表示每2帧取1帧)
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
                    print(f"  ⚠ 无法删除旧文件 {old_file}: {e}")

    # 为每个摄像头流提取帧
    image_lists = {}

    for stream_idx in range(num_streams):
        cam_dir = os.path.join(output_base_dir, f"cam{stream_idx}")

        print(f"✓ 正在处理 cam{stream_idx}...")

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

            print(f"  提取原始帧...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # 获取所有临时帧文件
            temp_frames = sorted(
                [
                    f
                    for f in os.listdir(cam_dir)
                    if f.startswith("temp_") and f.endswith(".jpg")
                ]
            )
            print(f"  ✓ cam{stream_idx} 已提取 {len(temp_frames)} 帧")

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
                    print(f"  ⚠ 无法重命名文件 {temp_frame}: {e}")
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
                    print(f"  ⚠ 无法删除未采样的文件 {temp_file}: {e}")

            print(
                f"  ✓ cam{stream_idx} 采样完成: {len(sampled_images)} 帧 (采样率: {frame_sample_rate})"
            )

            # 保存图片列表
            image_lists[f"cam{stream_idx}"] = sampled_images

        except subprocess.CalledProcessError as e:
            print(f"✗ 错误: 提取 cam{stream_idx} 帧失败")
            print(f"  stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            print("✗ 错误: 未找到 ffmpeg，请确保已安装并在 PATH 中")
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
        print("✗ 错误: rosbag 库未安装或ROS环境不可用，跳过 rosbag 保存")
        print("  提示: 若要使用rosbag功能，请安装 rosbag 库")
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

        print(f"✓ 开始创建 rosbag 文件: {bag_path}")

        with rosbag.Bag(bag_path, "w") as bag:
            # ==================== 保存 IMU 数据 ====================
            print("  正在写入 IMU 数据...")
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

            print(f"    ✓ 已写入 {imu_count} 条 IMU 消息")

            # ==================== 保存 相机 数据 ====================
            for cam_idx in range(num_streams):
                cam_dir = Path(output_dir) / f"cam{cam_idx}"
                if not cam_dir.exists():
                    continue

                print(f"  正在写入 cam{cam_idx} 图片...")
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
                            print(f"    ⚠ 无法读取图片: {img_file}")
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
                        print(f"    ⚠ 处理图片失败 {img_file}: {str(e)}")
                        continue

                print(f"    ✓ 已写入 {img_count} 张 cam{cam_idx} 图片")

        file_size = os.path.getsize(bag_path)
        print(f"\n✓ rosbag 文件已生成完成")
        print(f"  文件路径: {bag_path}")
        print(f"  文件大小: {file_size / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print(f"✗ 错误: 创建 rosbag 文件失败")
        print(f"  {str(e)}")
        return False


# ============================================================
# 初始化和执行逻辑
# ============================================================

# 开始处理
print("=" * 60)
print("开始数据提取和处理流程...")
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
print(f"✓ 已输出到: {IMU_OUTPUT_FILE}")
print(f"  文件大小: {file_size} 字节")
print(f"  IMU数据点数: {len(imu_msg_list.imu_msgs)}")
print(f"  Camera: {tp.camera}")
print(f"  Model: {tp.model}")

if not EXPORT_FRAMES:
    print("\n" + "=" * 60)
    print("⚠ 注意: 未导出相机帧，时间戳数据未被使用")
    print("=" * 60)
    exit(0)

# 获取时间戳数据（在提取帧之前）
print("\n" + "=" * 60)
print("准备时间戳数据...")
print("=" * 60)
exposure_data = tp.telemetry()[0]["Exposure"]["Data"]
all_timestamps = [e["t"] + TIME_OFFSET_SECS for e in exposure_data]  # 加上时间偏移
print(f"✓ 已读取 {len(all_timestamps)} 个时间戳")

# ============================================================
# 执行逻辑
# ============================================================

# 调用函数提取帧
print("\n" + "=" * 60)
print(f"开始提取视频帧 (采样率: {FRAME_SAMPLE_RATE})...")
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
print("帧提取统计:")
print("=" * 60)
for cam, images in image_lists.items():
    print(f"  {cam}: {len(images)} 帧")

print(f"\n✓ 原始时间戳数: {len(all_timestamps)}")
print(f"  采样后时间戳数: {len(sampled_timestamps)}")
print(f"  采样率: {FRAME_SAMPLE_RATE}")
if len(sampled_timestamps) > 0:
    print(f"  时间戳范围: {sampled_timestamps[0]:.6f} 到 {sampled_timestamps[-1]:.6f}")

# 验证帧数和时间戳是否匹配
print("\n" + "=" * 60)
print("帧与时间戳验证:")
print("=" * 60)
frame_counts = {cam: len(images) for cam, images in image_lists.items()}
for cam, frame_count in frame_counts.items():
    match_status = "✓ 匹配" if frame_count == len(sampled_timestamps) else "✗ 不匹配"
    print(
        f"  {cam}: {frame_count} 帧 vs {len(sampled_timestamps)} 时间戳 {match_status}"
    )

# 保存到 rosbag（可选）
if SAVE_TO_ROSBAG:
    print("\n" + "=" * 60)
    print("保存数据到 rosbag...")
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
    print("⚠ 跳过 Rosbag 保存 (SAVE_TO_ROSBAG = False)")
    print("=" * 60)

print("\n" + "=" * 60)
print("✓ 所有数据处理完成！")
print("=" * 60)
