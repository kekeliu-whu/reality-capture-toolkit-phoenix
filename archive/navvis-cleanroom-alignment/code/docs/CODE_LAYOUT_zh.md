# 代码与脚本目录索引

本文件说明当前代码树中哪些文件属于生产链、共享算法、评测工具或历史证据。正常运行只应从
两个顶层 shell 入口开始，不应直接调用 `slam_probes/` 中的原版动态探针。

## 顶层生产入口

| 入口 | 用途 | 实际调度 |
|---|---|---|
| `run_navvis_slam_recon.sh` | raw 双 Pandar + raw IMU 到自主优化轨迹 | 归档、C++ Frontend、自主回环、Stage1、Stage2 |
| `run_navvis_recon.sh` | 优化轨迹到点云、Quality、Surface、相机、全景和着色 | Python runner 调度五个 C++ worker |
| `scripts/run_all_tests.sh` | 统一构建、语法和单元回归 | CMake/CTest、Python compile、四组 Python tests |

两套生产入口默认共用 `code/build-release/`。可用 `NAVVIS_RECON_BUILD_DIR` 覆盖；SLAM 的
Stage1/Stage2 需要单独 Ceres 构建时，再设置 `NAVVIS_RECON_SLAM_STAGE_BUILD_DIR`。

## SLAM 生产链

| 层次 | 文件 | 职责 |
|---|---|---|
| 统一调度 | `runner/navvis_complete_slam.py` | 校验输入/worker，依次运行 archive、Frontend 和 backend |
| raw 归档 | `runner/navvis_slam_archive.py` | 两路 laser bag 按纳秒顺序送入 Pandar worker，生成 NVSLAM6 |
| Frontend ELF | `cpp/apps/slam_pipeline.cpp` | collator、raw IMU 去畸变、ICP、Submap、HybridGrid、MotionFilter、状态输出 |
| Frontend 核心 | `cpp/src/slam_{archive,batch_collator,imu,rosbag,frontend}.cpp` | NVSLAM6 mmap、批次、IMU、ROS bag、局部建图 |
| Frontend 接口 | `cpp/include/navvis_recon/slam_*.hpp` | C++ 类型和运行时接口 |
| Python 状态桥 | `src/navvis_recon/autonomous_slam.py` | 读取自产 native state，构造回环和后端对象 |
| 回环/后端调度 | `runner/navvis_slam_recon.py` | 自产回环、Stage1/Stage2 调度、轨迹输出和只读评测 |
| 图优化模型 | `src/navvis_recon/surveyor_frontend.py`、`surveyor_slam.py` | 回环匹配、图结构、IMU 因子和 native worker 协议 |
| Stage1 ELF | `cpp/apps/stage1_imu_ceres_solver.cpp` | pose-only fast-IMU Ceres 图 |
| Stage2 ELF | `cpp/apps/stage2_imu_ceres_solver.cpp` | pose/velocity/gravity/IMU 标定联合 Ceres 图 |

`navvis_recon_slam` 单独运行只覆盖 Frontend；完整 SLAM 的稳定入口是
`run_navvis_slam_recon.sh`。

默认 SLAM 工作目录结构：

```text
slam-work/
├── raw_scans.nvslam6          # 未传 --archive 时生成
├── frontend_trajectory.csv    # 局部 Frontend 轨迹
├── frontend_state.bin         # 自产 node/Submap/surfel/HybridGrid 状态
├── backend/                   # Stage1/Stage2 交换文件
├── optimized_trajectory.csv   # 后处理的 --trajectory-csv 输入
└── slam_alignment_report.json # 可选官方只读评测与求解统计
```

## 后处理生产链

| 模块 | C++ 入口 | C++ 核心/共享接口 |
|---|---|---|
| Pandar/CloudBuilder | `cpp/apps/pandar_cloud_pipeline.cpp` | `cloud_builder.cpp/.hpp` |
| Mapped-space Quality | `cpp/apps/mapped_space_quality.cpp` | `mapped_space_quality.cpp/.hpp` |
| Free-space/Surface | `cpp/apps/pandar_shard_surface_filter.cpp` | `cloud_surface_filter`、`binary_surface_pipeline` |
| Image/Panorama | `cpp/apps/ocam_panorama_pipeline.cpp` | `image_postprocessing`、`panorama_rendering` |
| Point coloring | `cpp/apps/surface_panorama_colorizer.cpp` | `pointcloud_coloring.cpp/.hpp` |

`runner/navvis_postprocessing_recon.py` 是唯一生产编排器。它接受自主
`--trajectory-csv`、既有 `--trajectory-bag`，或录制的 global/local 轨迹。共享 rec-v4
只读输入位于 `src/navvis_recon/recording_io.py`，统一负责：

- `sensor_frame.xml` 激光位姿；
- split laser bag 的数字顺序；
- 水平/垂直 topic 的全局纳秒顺序归并；
- `world_builder/add_scans` 控制窗口。

## Python 算法包

`src/navvis_recon/` 只放可复用算法和格式，不放命令行调度：

- `recording_io.py`：rec-v4 标定和 bag 输入；
- `autonomous_slam.py`、`surveyor_frontend.py`、`surveyor_slam.py`、
  `slam_reconstruction.py`：SLAM；
- `cloud_builder.py`、`cloud_surface_filter.py`、`quality_map.py`：点云/表面/Quality 参考实现；
- `image_postprocessing.py`、`panorama_rendering.py`、`pointcloud_coloring.py`：图像与颜色参考实现；
- `floor_estimator.py`：Floor 生产算法；
- `models.py`、`pipeline.py`：共享数据类型和模块说明。

Python 中标为参考实现的高开销路径不替代已验收的生产 C++ worker。

## Runner 辅助脚本

以下文件用于冻结小输入或制作诊断数据，不属于正常全量生产入口：

- `capture_laser_bag_window.py`：截取完整扫描窗口；
- `capture_laser_frames.py`：冻结 Pandar framed stream；
- `extract_pandar_scan_set.py`：提取指定扫描集合；
- `make_pandar_motion_state_probe.py`、`split_pandar_motion_probe.py`：运动状态探针；
- `merge_pandar_slam_frames.py`：合并 warm-up/main framed stream。

这些脚本现在直接使用 `navvis_recon.recording_io`，不再导入完整后处理 runner。

## 评测、回归和证据

- `tools/evaluate_*.py`：只读评测器；可读取冻结官方结果，但不得用于初始化生产求解。
- `tools/slam_state_io.py`：评测状态格式转换，不是生产状态入口。
- `scripts/run_stage1_slam_acceptance.sh`：冻结 Stage1 专项验收。
- `tests/*.py`：Python 单元测试；可直接执行，不依赖 pytest。
- `cpp/tests/*`：C++ 单元/捕获验收。
- `slam_probes/`：逆向研究的 GDB/Lua/C++/Python 探针和证据。正常 clean-room 生产不得运行
  `run_vendor_slam_probe.sh`，也不得链接该目录中的 capture `.so`。
- 相邻 `../regression/`：按数据集保存的不可变结果报告和机器指标。

## 构建和验证

```bash
./scripts/run_all_tests.sh
```

该命令是整理后的统一守门入口。需要全量数据回归时，再分别运行顶层 SLAM 和后处理命令；
单元守门不会自动启动耗时 10–60 分钟、占用数十 GB 的全量任务。
