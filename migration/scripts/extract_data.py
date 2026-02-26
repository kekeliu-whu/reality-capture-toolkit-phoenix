import logging
import os
import telemetry_parser
from proto.sensors_pb2 import ImuMsgList
import math
import csv
from pathlib import Path
import subprocess
import json

# ============================================================
# 配置参数和路径
# ============================================================

# 输入文件路径
input_video = R"\\wsl.localhost\Ubuntu-24.04\home\rick\iKalibr\2026-02-06_11-34-29\VID_20260206_033425_00_011.insv"
imu_output_file = R"D:\insv_gyro.dat"

# 输出目录
output_directory = "."

# 视频提取参数
frame_rate = 2  # fps (仅在 export_all=False 时使用)
quality = 2     # 0-31, 越低越好
export_all_frames = True  # 导出所有帧，确保与时间戳一一对应
num_streams = 2  # cam0 和 cam1

# ============================================================
# 初始化
# ============================================================

tp = telemetry_parser.Parser(input_video)

# 获取归一化的IMU数据
imu_data = tp.normalized_imu()

# 将IMU数据转换为protobuf格式
imu_msg_list = ImuMsgList()

for imu_sample in imu_data:
    imu_msg = imu_msg_list.imu_msgs.add()
    imu_msg.timestamp = imu_sample["timestamp_ms"] / 1000.0  # 转换为秒
    imu_msg.gx = imu_sample["gyro"][0] / 180.0 * math.pi  # 转换为弧度
    imu_msg.gy = imu_sample["gyro"][1] / 180.0 * math.pi
    imu_msg.gz = imu_sample["gyro"][2] / 180.0 * math.pi
    imu_msg.ax = imu_sample["accl"][0]
    imu_msg.ay = imu_sample["accl"][1]
    imu_msg.az = imu_sample["accl"][2]
    # print(
    #     f"Processed IMU sample at {imu_msg.timestamp} s accl: {imu_msg.ax}, {imu_msg.ay}, {imu_msg.az} gyro: {imu_msg.gx}, {imu_msg.gy}, {imu_msg.gz}"
    # )

# 将protobuf消息写入二进制文件
with open(imu_output_file, "wb") as f:
    f.write(imu_msg_list.SerializeToString())

# 验证写入成功
file_size = os.path.getsize(imu_output_file)
print(f"✓ 已输出到: {imu_output_file}")
print(f"  文件大小: {file_size} 字节")
print(f"  IMU数据点数: {len(imu_msg_list.imu_msgs)}")
print(f"  Camera: {tp.camera}")
print(f"  Model: {tp.model}")

# print("telemetry_parser version:", tp.telemetry())
## save telemetry to text file
print(tp.telemetry()[0]['Exposure']['Data'].__len__())
# with open("telemetry_info.txt", "w") as f:
#     if isinstance(telemetry_data, list):
#         f.write('\n'.join(str(item) for item in telemetry_data))
#     else:
#         f.write(str(telemetry_data))
print([item['t'] for item in tp.telemetry()[0]['Exposure']['Data']])

# ============================================================
# 函数定义
# ============================================================

def extract_frames_from_video(input_video_path, output_base_dir=".", frame_rate=None, quality=2, num_streams=2, export_all=True):
    """
    从视频文件提取帧到多个摄像头目录
    
    参数:
        input_video_path: 输入视频文件路径 (.insv 或 .mp4)
        output_base_dir: 输出基础目录
        frame_rate: 帧率 (fps)，如果 export_all=True 则忽略此参数
        quality: 视频质量 (0-31, 越低越好)
        num_streams: 流的数量 (默认2个: cam0, cam1)
        export_all: 是否导出所有帧（True时忽略frame_rate）
    """
    input_video_path = str(input_video_path)
    output_base_dir = str(output_base_dir)
    
    # 创建输出目录
    for i in range(num_streams):
        cam_dir = os.path.join(output_base_dir, f"cam{i}")
        os.makedirs(cam_dir, exist_ok=True)
    
    # 为每个摄像头流提取帧
    for stream_idx in range(num_streams):
        cam_dir = os.path.join(output_base_dir, f"cam{stream_idx}")
        output_pattern = os.path.join(cam_dir, "%05d.jpg")
        
        # 构建 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-threads", "8",
            "-hwaccel", "auto",
            "-i", input_video_path,
            "-map", f"0:{stream_idx}",
            "-q:v", str(quality),
            "-threads", "8",
        ]
        
        # 只在不导出全部帧时添加 -r 参数
        if not export_all and frame_rate:
            cmd.extend(["-r", str(frame_rate)])
        
        cmd.append(output_pattern)
        
        if export_all:
            print(f"✓ 正在导出 {f'cam{stream_idx}'} 的全部帧...")
        else:
            print(f"✓ 正在提取 {f'cam{stream_idx}'} 的帧 ({frame_rate} fps)...")
        print(f"  命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if export_all:
                print(f"✓ {f'cam{stream_idx}'} 全部帧导出完成，输出到: {cam_dir}")
            else:
                print(f"✓ {f'cam{stream_idx}'} 帧提取完成，输出到: {cam_dir}")
        except subprocess.CalledProcessError as e:
            print(f"✗ 错误: 提取 {f'cam{stream_idx}'} 帧失败")
            print(f"  stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            print("✗ 错误: 未找到 ffmpeg，请确保已安装并在 PATH 中")
            raise

def count_frames_in_output(output_dir):
    """计算输出目录中的帧数"""
    cam_dirs = [d for d in os.listdir(output_dir) if d.startswith('cam')]
    frame_counts = {}
    for cam_dir in sorted(cam_dirs):
        cam_path = os.path.join(output_dir, cam_dir)
        frame_count = len([f for f in os.listdir(cam_path) if f.endswith('.jpg')])
        frame_counts[cam_dir] = frame_count
    return frame_counts

def rename_images_with_timestamps(output_dir, timestamps, num_streams=2):
    """
    根据时间戳重命名提取的图片
    
    参数:
        output_dir: 输出基础目录
        timestamps: 时间戳列表 [t1, t2, t3, ...]
        num_streams: 流的数量
    """
    import shutil
    
    for cam_idx in range(num_streams):
        cam_dir = Path(output_dir) / f"cam{cam_idx}"
        if not cam_dir.exists():
            continue
        
        # 获取编号的图片列表
        image_files = sorted(cam_dir.glob("*.jpg"), key=lambda x: int(x.stem))
        
        # 为每个图片添加时间戳
        timestamp_map = []
        for img_idx, img_file in enumerate(image_files):
            if img_idx < len(timestamps):
                timestamp = timestamps[img_idx]
                # 格式: 00001_t1234567890.123.jpg (原始编号_时间戳)
                timestamp_str = str(timestamp).replace('.', '_')
                new_name = f"{img_file.stem}_{timestamp_str}.jpg"
                new_path = img_file.parent / new_name
                
                # 重命名文件
                img_file.rename(new_path)
                timestamp_map.append({
                    'original': img_file.name,
                    'renamed': new_name,
                    'timestamp': timestamp,
                    'index': img_idx
                })
        
        print(f"✓ {f'cam{cam_idx}'} 图片已重命名并添加时间戳")
        
        # 保存映射信息为 JSON
        json_path = cam_dir / "image_metadata.json"
        with open(json_path, "w") as f:
            json.dump(timestamp_map, f, indent=2)
        print(f"  元数据已保存到: {json_path}")

def count_frames_precise(cap):
    """精确计算视频帧数"""
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    return count

# ============================================================
# 执行逻辑
# ============================================================

# 调用函数提取帧
print("=" * 60)
print("开始提取视频帧...")
print("=" * 60)
extract_frames_from_video(
    input_video_path=input_video,
    output_base_dir=output_directory,
    frame_rate=frame_rate if not export_all_frames else None,
    quality=quality,
    num_streams=2,  # cam0 和 cam1
    export_all=export_all_frames
)

# 统计提取的帧数
print("\n" + "=" * 60)
print("帧提取统计:")
print("=" * 60)
frame_counts = count_frames_in_output(output_directory)
for cam, count in frame_counts.items():
    print(f"  {cam}: {count} 帧")

# 获取时间戳数据
exposure_data = tp.telemetry()[0]['Exposure']['Data']
timestamps = [e['t'] for e in exposure_data]

print(f"\n✓ 总时间戳数: {len(timestamps)}")
print(f"  时间戳范围: {timestamps[0]:.6f} 到 {timestamps[-1]:.6f}")

# 验证帧数和时间戳是否匹配
print("\n" + "=" * 60)
print("帧与时间戳验证:")
print("=" * 60)
for cam, frame_count in frame_counts.items():
    match_status = "✓ 匹配" if frame_count == len(timestamps) else "✗ 不匹配"
    print(f"  {cam}: {frame_count} 帧 vs {len(timestamps)} 时间戳 {match_status}")

# 重命名图片并添加时间戳
print("\n" + "=" * 60)
print("重命名图片并添加时间戳...")
print("=" * 60)
rename_images_with_timestamps(output_directory, timestamps, num_streams=2)

# 为每个摄像头目录生成详细的 CSV 映射
print("\n" + "=" * 60)
print("生成时间戳映射文件...")
print("=" * 60)
for cam_idx in range(2):  # cam0 和 cam1
    cam_dir = Path(output_directory) / f"cam{cam_idx}"
    if cam_dir.exists():
        # 获取重命名后的图片文件列表
        image_files = sorted(cam_dir.glob("*.jpg"), key=lambda x: int(x.stem.split('_')[0]))
        
        # 导出为 CSV：图片名, 时间戳
        csv_path = cam_dir / "image_timestamps.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Timestamp", "Index"])
            for img_idx, img_file in enumerate(image_files):
                if img_idx < len(timestamps):
                    writer.writerow([img_file.name, timestamps[img_idx], img_idx])
        
        print(f"✓ {f'cam{cam_idx}'} 图片时间戳映射已保存到: {csv_path}")

