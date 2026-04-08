# 使用说明

## convert_s20 - 将 ROS Bag 转换为 protobuf 格式

```bash
convert_s20 \
  --bag_filename="sensor_data.bag" \
  --output_dir="D:/slam_data"
```

**说明**：
- `--bag_filename`: 输入的 ROS Bag 文件路径
- `--output_dir`: 输出目录，会生成 `calibration.dat`、`imu.dat`、`encoder.dat`、`lidar.dat`

---

## slam - LiDAR SLAM 处理

```bash
slam \
  --project_dirname="D:/slam_data" \
  --output_dir="D:/slam_output" \
  --indoor=true
```

**说明**：
- `--project_dirname`: 包含 `calibration.dat`、`imu.dat`、`encoder.dat`、`lidar.dat` 的目录
- `--output_dir`: 输出目录，生成 `trajectory.txt`（轨迹）和 `map.las`（地图）
- `--indoor`: `true` 使用室内模式，`false` 使用室外模式

