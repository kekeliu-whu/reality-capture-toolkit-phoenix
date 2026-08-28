# 可迁移源码与回归包

该目录已经按“换机后从源码重建”的方式整理。包内保留实现、脚本、冻结测试输入和对齐报告；
不保留当前机器的 CMake 缓存、预编译 ELF/共享库、全量流水线输出和临时实验目录。

## 目录内容

- `code/`：C++17/Python 源码、生产 runner、测试和 SLAM 研究探针。
- `scripts/`：对齐研究、冻结验收和诊断脚本；其中很多研究脚本需要自行指定外部数据。
- `test_resources/`：可随包复制的模块测试输入，包括 3MB 的 Stage-1 冻结夹具。
- `regression/`：已完成测试的 JSON/Markdown/CSV/NPZ 证据，以及 457MB 的完整 SLAM 帧归档。
- `EXTERNAL_DATASETS.json`：未打包的原始录制、官方结果和原版程序路径清单。
- `MANIFEST.sha256`：迁移包所有文件的校验值；`SHA256SUMS.core` 是关键大夹具的短清单。

原始录制和官方全量输出没有复制进来。完整流程换机复测时，需要按
`EXTERNAL_DATASETS.json` 重新挂载或修改路径，并把生成结果写到本目录之外。

## 新机器依赖

基础 C++ 依赖：CMake 3.16+、C++17、pkg-config、Eigen3、OpenCV 4、OpenMP、Brotli、PCL
头文件和 Ceres。Ubuntu 类系统对应的常见包为：

```bash
sudo apt install build-essential cmake pkg-config \
  libeigen3-dev libopencv-dev libceres-dev libpcl-dev libbrotli-dev libraw-dev
```

完整 SLAM/录制读取还需要 ROS1 的 `rosbag`、`sensor_msgs` C++ pkg-config 模块和 Python
`rosbag`。运行 CMake 前先加载对应 ROS 环境。全景 worker 要求 LibRaw 0.22 ABI
（`liblibraw.so.24`）；若系统版本不同，CMake 会明确跳过该 target。

Python 要求 3.10+。普通 Python 依赖可在 `code/` 中执行：

```bash
python3 -m pip install -e .
```

标准发行版的 `libceres-dev` 可以构建算法；要复现已记录的 32 线程颜色收敛分布，推荐
Ceres 2.2.x。旧机器的 Ceres/PCL 临时解包目录不属于迁移包，也不应写进 CMake 缓存。

## 构建与测试

```bash
code/scripts/run_all_tests.sh
code/scripts/run_stage1_slam_acceptance.sh
scripts/run_frozen_acceptance.sh
```

第一个命令会新建 `code/build-release/` 并执行 C++/Python 单元测试。Stage-1 验收只读取
`test_resources/slam_stage1_20260827/`，结果写入构建目录，不修改冻结报告。完整 SLAM 和后处理
的生产入口分别是：

```bash
code/run_navvis_slam_recon.sh --help
code/run_navvis_recon.sh --help
```

## 复制与校验

复制时保留权限和时间戳：

```bash
rsync -a --info=progress2 navvis-cleanroom-alignment/ TARGET/navvis-cleanroom-alignment/
cd TARGET/navvis-cleanroom-alignment
sha256sum -c MANIFEST.sha256
```

源码或冻结资源有意修改后，用 `python3 scripts/update_manifest.py` 重新生成完整清单。
