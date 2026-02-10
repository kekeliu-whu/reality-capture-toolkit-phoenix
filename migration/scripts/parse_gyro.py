import os
import telemetry_parser
from proto.sensors_pb2 import ImuMsgList
import math

telemetry_file = R"\\wsl.localhost\Ubuntu-24.04\home\rick\iKalibr\2026-02-06_11-34-29\VID_20260206_033425_00_011.insv"
output_file = R"D:\insv_gyro.dat"

tp = telemetry_parser.Parser(telemetry_file)

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
with open(output_file, "wb") as f:
    f.write(imu_msg_list.SerializeToString())

# 验证写入成功
file_size = os.path.getsize(output_file)
print(f"✓ 已输出到: {output_file}")
print(f"  文件大小: {file_size} 字节")
print(f"  IMU数据点数: {len(imu_msg_list.imu_msgs)}")
print(f"  Camera: {tp.camera}")
print(f"  Model: {tp.model}")
