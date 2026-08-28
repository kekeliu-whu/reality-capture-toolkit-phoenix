# NavVis 后处理与轨迹侧 SLAM 净室重构：代码与对齐状态

## 生产代码与脚本布局基线（2026-08-28）

正常生产只有两个顶层入口：`run_navvis_slam_recon.sh` 负责 raw Pandar/raw IMU 到优化轨迹，
`run_navvis_recon.sh` 负责优化轨迹到全部后处理产物；二者默认共用 `build-release/`。统一守门
入口为 `scripts/run_all_tests.sh`。rec-v4 激光标定、numeric split-bag、跨 topic 纳秒归并和
`world_builder/add_scans` 窗口集中在 `src/navvis_recon/recording_io.py`，runner/probe 不得再
反向导入 `navvis_postprocessing_recon.py` 取得这些函数。`slam_probes/` 只保留逆向证据，正常
生产不得运行其中 vendor probe 或加载 capture `.so`。完整索引见 `docs/CODE_LAYOUT_zh.md`。

## SurveyorSLAM 全自主链工程对齐基线（2026-08-28）

`2026-07-21_11.07.05` 已完成 raw 双 Pandar + raw IMU → C++ Frontend → 自产回环 →
Ceres Stage1 → Ceres Stage2 的全量链。自产 10,933 nodes、11 Submaps、789 条有效回环；求解
不读取官方状态，官方只在 Stage2 完成后评测。单个全局 SE(3) gauge 对齐后，最终 ATE
translation mean/p95/max=`0.654/1.343/1.938 mm`，rotation mean/p95=
`0.01066/0.01950°`，按用户结果接近口径为 `NEAR / 工程已对齐`。

回环 pair 集对官方 TP/FP/FN=`493/296/369`，因此不得称 topology-exact。统一入口为
`run_navvis_slam_recon.sh`；自产 `optimized_trajectory.csv` 可通过后处理 runner 的
`--trajectory-csv` 继续进入点云/表面/着色链。完整报告见
`../regression/2026-07-21_11.07.05/SLAM_FULL_AUTONOMOUS_ALIGNMENT_20260828.md`。

## 全局自由空间 intersection 完全对齐基线（2026-08-28）

`2026-07-21_11.41.12` chronological 同输入的 2,667,377 个 2 cm 叶已完成逐 key 全量验收：
occupancy、hit count 和 intersection count 均为 2,667,377/2,667,377 exact，删除叶
852,434，delete mask FP/FN=`0/0`。此前 12 个 intersection count 各差 1 的根因是
Spherical Fibonacci 解码把原版 FMA 写成普通乘减，现已恢复；不得回退为经验阈值或 key
特判。5,242 个质心 float32 XYZ 最大差 8.596 µm，按当前结果口径视为无差异，但不应称其
文件位模式完全一致。报告见
`../regression/2026-07-21_11.41.12/FREESPACE_INTERSECTION_EXACT_ALIGNMENT_20260828.md`。

## Multi-scale normal 完全对齐基线（2026-08-28）

冻结 G11 同输入 `multi_scale_normal_estimation` 已从当前源码新鲜构建并完成逐位验收：
109,322/109,322 个 normal XYZ、XYZ、intensity、curvature 和 weight 可观察语义字段逐记录
bit-exact；双方零法向均为同一 6 点，角误差 P50/P95/max 均为 `0°`。单线程与 32 线程结果
一致。旧 `P95 0.002241° / 11 个异常点 / max 1.11017°` 是 NumPy 独立复算残差，不代表当前
生产 C++。报告见
`../regression/2026-02-08_07.33.20/MULTISCALE_NORMAL_EXACT_ALIGNMENT_20260828.md`。

本文件适用于本目录及全部子目录。它汇总共享工作区中两个相关 Codex 任务已经落盘的代码，给后续开发者提供唯一的模块地图、标准评测口径和剩余差距。

## Floor 完全对齐基线（2026-08-28）

Floor 当前在 `/media/cybergeo/12T/DT` 的 220 组同输入参考上为 220/220 exact：共
1,796,815 条 trace、615 个 floor，最终 `z_min/z_max` 和全部纳秒 time range 逐字段一致。
runner 已改为从 `/imu/magnetic_field` 与轨迹生成官方格式 `artifacts/trace.csv`，并写官方顶层
数组 schema 的 `artifacts/floors.json`；两个 G11 单层/多层真实数据的 trace 均 byte-exact，
floors 均 exact。完整报告和守门命令见 `FLOOR_FULL_ALIGNMENT_20260828.md`；缺少 trace 时钟或
轨迹时间不合法必须直接失败，不得恢复旧 `floor_summary` 或静默 fallback。

## Mapped-space quality 语义完全对齐基线（2026-08-28）

标准生产路径现为 `navvis_recon_mapped_space_quality`：读取 exact per-ray `.raytile`，恢复
`resolution/3` 候选格采样与投影、255 个 Spherical Fibonacci 方向桶、厘米 range LUT、方向
距离权重、ray count rescale/filter、Morton key 和 v2 13-byte `<QHHB>` Brotli 输出。64-ray
合成探针的 1,739/1,739 voxel 和 5,147 条真实 G11 ray 的 9,429/9,429（min=1）、
1,092/1,092（标准 min=36）在 key/diversity/ray count/min range 上全部逐 voxel exact；runner
已接线并对残缺/旧聚合 shard fail-closed。135,658,017 条 G11 全量对官方 aggregator 核心的
59,713 条 `<QHHB>` 四字段、record 多重集合和顺序全部 Exact。官方命令行 estimator 外层实际
多调用 3 次 `addAlongRay`，冻结产物因此有 21 个 ray-count 和 5 个 diversity 残差；相同官方
32-thread 命令重跑还会有 3 个 diversity 随 merge 顺序漂移。生产不复刻重复提交和非确定调度，
所以对核心是 `QUALITY_CORE_RECORD_EXACT`，对 estimator 文件不是 Byte Exact。报告见
`../regression/2026-07-21_11.41.12/MAPPED_SPACE_QUALITY_ALIGNMENT_20260828.md`。

## 全景渲染完全对齐基线（2026-08-27）

标准 G8 路径从 processed camera JPEG 和 `pano_depth_sparse.png` 到 8192×4096 filled JPEG
已经完全对齐。`2026-07-21_11.41.12` 全部 34 个 capture 最终文件与冻结 NavVis 结果
byte-exact（34/34，mismatch 0）；2K/8K depth effect、projection、曝光、GraphCut seam、
十层 multiband、BinaryMask、JPEG round-trip 和 floor fill 也已分阶段 exact。最新报告为
`../regression/2026-07-21_11.41.12/PANORAMA_FULL_EXACT_ALIGNMENT_20260827.md`，守门命令为
`../scripts/verify_panorama_full_exact.sh`。

`--surface-cloud` 上游现也已有独立 exact 验收：34/34 个 capture 的 1024×512 sparse depth
逐像素一致，`00000` 的正式 C++ Surface→PCG→最终 8K JPEG 与官方文件 byte-exact。报告为
`../regression/2026-07-21_11.41.12/PANORAMA_SURFACE_TO_FINAL_BYTE_EXACT_20260827.md`。
`00017` clean 17.88 s 对 NavVis 12.31 s，结果 exact，但性能仍慢约 1.45×。

## 0. 已对齐 C++ 路径的代码导航（2026-08-26）

本轮只整理已有冻结结果守门的生产 C++，算法常量、浮点顺序、容器遍历和有序归并不得因
后续“美化”改变。统一格式文件为 `cpp/.clang-format`。

- `cpp/apps/pandar_shard_surface_filter.cpp`
  - `prepareInputShards`：识别 `.raytile/.tile`、校验并准备稳定输入；
  - `loadRuntimeOptions`：读取 CompactOctree anchor；
  - `preprocessShard`：单 shard 读取、自由空间证据和 Surface 输入构建；
  - `processSurfaceTile`：遮挡清理、helper 聚合、Binary Surface 和 5 m core crop；
  - `main`：只负责阶段调度、线程队列、确定性归并和 PLY 写出。
- `cpp/apps/surface_panorama_colorizer.cpp`
  - `ColoringScene/loadColoringScene`：PLY、capture/camera pose、OCam 和 mask；
  - `DepthMaps`：原子深度、回归读写、nearest/linear 可见性规则；
  - `prepareDepthAndCloudBounds`：PCT/全景深度和曝光 bounds；
  - `buildVoxelViewRankings`：可选 voxel OVS；
  - `writeColoredCloud`：逐点 Top-5、颜色融合、fallback、直方图和 KNN；
  - `main`：场景 → 深度 → 曝光 → OVS → 最终着色。
- `cpp/src/cloud_surface_filter.cpp` 和 `cpp/src/binary_surface_pipeline.cpp` 是对应核心算法；对应
  `.hpp` 共同使用同一格式。不要把算法重新塞回入口 lambda，也不要把确定性归并改为完成顺序。

冻结回归：Surface PLY SHA
`5d24e311ee3a939bd84220ffa6eec8fbb248a99b4958259c0f950ab9011b93c5`，固定输入着色 PLY
SHA `5cccc3c6e781a3efda1fba421b4cde40ac82250b291fddf9693e2e52f46a32bb`，direct mask SHA
`376850d6a11a779cf0b372fca98f8a63601dbbbd783af257da9e40008196cbbd`。完整命令、时间和产物见
`../regression/2026-07-21_11.41.12/ALIGNED_CPP_READABILITY_REFACTOR_20260826.md` 与
`../work/readability_refactor_20260826/`。

## 1. 范围与真实性

- 本项目是 C++17/Python 的独立净室实现，不是 NavVis 厂商逐字源码。
- 项目现已包含从双 Pandar 原始扫描和 raw IMU 到 1,617-node 离线轨迹的完整 SurveyorSLAM 净室链：去畸变、三级 surfel ICP、双活动 Submap、HybridGrid、候选采样、回环和 9D IMU 稀疏图优化均已实现。拓扑/回环集合 exact，最终轨迹达到亚毫米结果级对齐；不是厂商逐字源码，也不是 protobuf/浮点逐位复刻。
- `2026-07-21_11.41.12` 的首 5 秒 98-node raw-IMU/局部 ICP 前端已达到舍入误差级；原生 `cpp/apps/stage1_imu_ceres_solver.cpp` 在完整 2,660-node 冻结 Stage 1 上与官方 final cost 差 `6.17e-9`、ATE translation max `1.96e-8 mm`，可在声明容差/Build ID 下称为结果 exact。大数据整条自主链现已工程对齐，但全量自产 loop pair 集和 Stage 2 数值仍不是 protobuf/topology/byte exact。
- 外部提供的 `trajectory.bag` 和 `sensor_frame.xml` 始终是只读输入。`--slam-reference-bag` 也只参与评测，不得用于修改生成结果。
- 默认生产计算链是 `runner/navvis_postprocessing_recon.py` 调度四个 C++ worker。`src/navvis_recon/*.py` 是轻量参考/测试实现，不是 G11 standard 的数值对齐主路径。
- 正常净室运行不得启动 `nv_cloud-surface-filter` 或 `nv_colorcloud`。安装二进制仅用于离线反汇编、只读动态探针和冻结 capture 验收。
- 不得根据最终点数反向调阈值。参数必须有反汇编、运行时对象、黑盒消融或冻结中间态支持。

## 2. 多会话合并状态

当前代码树已经包含两个相关任务的改动：

1. 主任务完成了着色、图像、全景、自由空间、遮挡分类、表面点选择、输出体素、density 和 Adaptive SOR 等实现及验收。
2. 任务“定位navvis-postprocessing源码 (2)”补充了 G11 Pandar 点云构建：`world_builder/add_scans` 控制窗口、820-byte XTM 扫描组织、逐点轨迹插值、垂直脚部 PlaneFilter/RANSAC、扫描统计及 runner 接线。

后续改动先把多层边缘过滤从该任务结束时的全量差 38,216 点改善到 V11 的 15,705 点；
2026-08-26 又按运行时标定位模式、谓词控制流以及端点/origin 两条不同的微秒取整路径，消除了
精确时间 3-scan/10 秒回归的全部分类和官方七字段 `PointRayIntensity` 记录差异。旧全量
`float64` 帧流的诊断残差为基础 `-5`、fringe `-67`，原始全量 bag 清理后尚未
恢复，故全量精确输入验收待办。以当前工作树和最新报告为准，不以任一旧任务的最终消息作为
代码真值。此目录没有可用 Git 历史来可靠标记逐行会话来源。

## 3. 标准与证据优先级

发生冲突时按以下顺序判断：

1. 同一输入、同一阶段的二进制冻结记录与当前验收程序输出；
2. 安装 ELF 的静态控制流和只读 GDB 运行时捕获；
3. 原版同数据最终产物和日志；
4. 当前完整/裁剪回归 JSON；
5. README 中的历史基线或经验实现。

主要标准数据：

- G11 模块回归：`2026-02-08_07.33.20`；具有原版点云、着色裁剪和二进制中间态。
- G11 完整历史参考：`2026-08-10_20.13.41`；原版 standard/0.01 m 为 159,147,139 点、299 张 8192×4096 全景。旧净室全量结果已经被后续算法修改淘汰，只能作为历史趋势。
- G10/VLP16 功能回归：`2023-05-15_10.18.42`；原厂许可过期，没有同数据 GT，不能声称原版数值对齐。

最新报告位于相邻目录 `../navvis_alignment_reference/`：

- `SURFACE_SAME_INPUT_ACCEPTANCE_V1.md`：自由空间与表面最新同输入验收；
- `../regression/2026-07-21_11.41.12/FREESPACE_INTERSECTION_EXACT_ALIGNMENT_20260828.md`：
  2,667,377 个全局自由空间叶的 occupancy/hit/intersection/delete mask 当前全量 Exact 验收；
  优先于旧 787,895 叶和 12 个 intersection Near 的结论。
- `occlusion_octree_query_binary_report_v1.md`：遮挡 octree 查询状态机；
- `surface_backhalf_binary_report_v2.md`：表面后半链和输出体素；
- `multiscale_normal_binary_report_v1.md`、`surface_selection_binary_report_v1.md`：法向与表面位置；
- `../regression/2026-02-08_07.33.20/MULTISCALE_NORMAL_EXACT_ALIGNMENT_20260828.md`：
  Multi-scale normal 当前 109,322/109,322 逐位 Exact 验收；优先于旧 NumPy 独立复算角差。
- `CLEANROOM_COLOR_ALIGNMENT_V10.md`：当前着色裁剪验收；
- `ALIGNMENT_STATUS_V11.md`：Pandar、图像和全景的历史综合状态；
- `ALIGNMENT_1_2_3_20260824.md`：本轮完整 SLAM、全量着色和全景的基线结果；
- `ALIGNMENT_FOLLOWUP_20260824.md`：样本级 IMU 结构、全量最终着色 OVS 和四层全景深度优化的最新复核，优先于前述报告的冲突结论。
- `COLOR_ALIGNMENT_DYNAMIC_CAUCHY_HSV_20260824.md`：全量曝光 dynamic Cauchy、float32 归一化和 HSV255 应用路径修复后的最新着色验收；优先于旧的全量自动 Gamma 数字。
- `../regression/2026-02-08_07.33.20/slam_raw/COMPLETE_SLAM_ALIGNMENT_20260825.md`：完整 raw-Pandar/raw-IMU SurveyorSLAM 的标准链、全量误差、证据边界和复现命令。
- `../regression/2026-02-08_07.33.20/PANDAR_EXACT_ALIGNMENT_20260826.md`：Pandar 运行时
  标定位模式、精确纳秒输入和 multilayer predicate 的最新结果验收；优先于旧
  `-978/-15,705` 数字。
- `../regression/2026-07-21_11.41.12/SLAM_IMU_BACKEND_ALIGNMENT_20260827.md`：首 5 秒前端、Stage 1 原生 Ceres exact 结果和 Stage 2 联合标定的最新隔离验收。
- `../regression/2026-07-21_11.41.12/PANORAMA_FULL_EXACT_ALIGNMENT_20260827.md`：标准 G8 sparse-depth 到 8K filled JPEG 的 34/34 byte-exact 全量验收；优先于旧全景 `NEAR` 结论。
- `../regression/2026-07-21_11.41.12/PANORAMA_SURFACE_TO_FINAL_BYTE_EXACT_20260827.md`：
  Surface PLY→sparse depth→PCG→最终 8K 的最新验收；34/34 个低分辨率 sparse exact，
  `00000` 正式端到端 final byte-exact。

## 4. 代码模块地图

| 模块 | 生产代码 | 入口/辅助代码 | 标准状态 |
|---|---|---|---|
| 调度、轨迹与标定 | `runner/navvis_postprocessing_recon.py` | `run_navvis_recon.sh`、`src/navvis_recon/pipeline.py`、`models.py` | 控制窗口和相机/雷达外参已接线 |
| SurveyorSLAM 前端/后端 | `cpp/apps/slam_pipeline.cpp`、`runner/navvis_slam_recon.py`、`src/navvis_recon/surveyor_{frontend,slam}.py`、两个 Ceres solver | `run_navvis_slam_recon.sh`、`runner/navvis_slam_archive.py`、完整 SLAM evaluator | 10,933-node 全自主链已串联；最终 ATE mean/p95 `0.654/1.343 mm`，工程已对齐；回环 pair topology 非 exact |
| 激光解码与点云构建 | `cpp/apps/pandar_cloud_pipeline.cpp`、`cpp/src/cloud_builder.cpp` | `cpp/include/navvis_recon/cloud_builder.hpp`、`src/navvis_recon/cloud_builder.py` | 精确 ns 的 3-scan/10 秒基础点数、fringe 点数、删除 mask 及 `origin/XYZ/intensity` 七字段记录多重集合完全一致；旧全量有损时间流仅作诊断（基础 `-5`、fringe `-67`），全量 raw 验收待恢复 |
| 全局自由空间 | `cpp/apps/pandar_shard_surface_filter.cpp`、`cpp/src/cloud_surface_filter.cpp` | `cpp/include/navvis_recon/cloud_surface_filter.hpp` | G11 全量 2,667,377 叶同输入 occupancy/hit/intersection/delete mask 完全一致 |
| G11 遮挡与表面点集 | `cpp/src/binary_surface_pipeline.cpp` | `cpp/include/navvis_recon/binary_surface_pipeline.hpp`、`cpp/tests/surface_capture_acceptance.cpp` | 遮挡、输入聚合、Multi-scale normal 109,322/109,322 语义记录 bit-exact、输出 key、density、SOR 的冻结验收已精确 |
| 点云着色 | `cpp/apps/surface_panorama_colorizer.cpp`、`cpp/src/pointcloud_coloring.cpp` | `cpp/include/navvis_recon/pointcloud_coloring.hpp`、`src/navvis_recon/pointcloud_coloring.py` | 裁剪自动 Gamma MAE 0.057943；全量自动 Gamma MAE 0.748406，原版 Gamma 隔离 MAE 0.322122；结果基本对齐但尚非逐点完全一致 |
| 图像后处理 | `cpp/apps/ocam_panorama_pipeline.cpp`、`cpp/src/image_postprocessing.cpp` | `cpp/include/navvis_recon/image_postprocessing.hpp`、`src/navvis_recon/image_postprocessing.py` | 直接读取原始 DNG；24/24 JPEG 对官方整文件 byte-exact、解码像素 MAE 0；标准 high-quality 路径 `EXACT` |
| 全景与点云渲染 | `cpp/apps/ocam_panorama_pipeline.cpp`、`cpp/src/panorama_rendering.cpp` | `cpp/include/navvis_recon/panorama_rendering.hpp`、`src/navvis_recon/panorama_rendering.py` | 标准 G8 sparse-depth→8K filled JPEG 为 34/34 byte-exact；Surface→sparse 为 34/34 exact，`00000` Surface→最终 8K byte-exact；`00017` 17.88 s vs NVS 12.31 s |
| Floor estimator | `src/navvis_recon/floor_estimator.py` | `tools/evaluate_floor_alignment.py`、runner、Python 测试 | 220/220 组同输入、615 floors 全字段 exact；runner 两组 G11 trace byte-exact/floors exact，已接线并 fail-closed |
| Mapped-space quality | `cpp/apps/mapped_space_quality.cpp`、`cpp/src/mapped_space_quality.cpp` | `cpp/include/navvis_recon/mapped_space_quality.hpp`、`src/navvis_recon/quality_map.py`、`tools/evaluate_mapped_space_quality.py`、runner | G11 全量 135,658,017 ray / 59,713 voxel 对官方核心四字段及 record 顺序 `EXACT`；官方 estimator 多提交 3 ray，冻结文件余 21 count/5 diversity，且并行 diversity 非确定；runner 已接线/fail-closed |

## 5. 各模块的标准评测差距

### 5.1 SLAM、轨迹和标定

最新优先结论：首 5 秒 98-node 前端 ATE translation mean/max 为 `1.594e-11/5.304e-11 mm`。Stage 1 原生 Ceres 的 2,658 个 acceleration 和 2,659 个 rotation measurement 最大差 `4.44e-16/5.55e-16`，完整后端 final cost 与官方精确值差 `6.17e-9`，ATE translation mean/max `9.89e-9/1.96e-8 mm`。该 exact 只覆盖冻结输入、指定 Build ID 和声明浮点容差。大数据全自主链的最终 ATE translation mean/p95 为 `0.654/1.343 mm`，按结果接近口径已对齐；严格 loop topology/serialization 尚未 exact。

当前完整链：Pandar scan archive → 双雷达 50 ms/58,000-ray 合批 → raw-IMU 旋转与常速平移去畸变 → 0.04 m scan centroid → 三级 float surfel ICP → 两个 active Submap → 0.2 m HybridGrid → 401 eligible / FixedRatioSampler 41 searches → FCS/ICP → 1,616 个相邻节点 9D IMU 因子 → 稀疏图优化。

`2026-02-08_07.33.20` 全量回归真值：

| 项目 | 当前结果 | 对官方差距 |
|---|---:|---:|
| 原始前端节点 | 1,617 | exact |
| Submap / memberships | 6 / 2,581 | 生命周期、时间戳、成员逐项 exact |
| 局部前端 pose | translation mean/p95/max `0.497/0.732/4.123 mm`；rotation mean/p95 `0.00444/0.01399 deg` | 亚毫米主体；node 1150 为单个 4.12 mm 切图边界离群点 |
| sampled / accepted loops | 41 / 15 | pair 集 TP/FP/FN=`15/0/0` |
| 冻结点云回环测量 | translation mean/p95/max `0.318/0.637/0.688 mm`；rotation mean/p95 `0.00181/0.00418 deg` | 结果级对齐 |
| 生成点云回环测量 | translation mean/p95/max `0.817/1.218/1.243 mm`；rotation mean/p95 `0.00653/0.01348 deg` | 集合 exact，测量亚毫米到 1.24 mm |
| 完整生成链最终 node ATE | translation mean/p95/max `0.421/0.802/0.869 mm`；rotation mean/p95 `0.00213/0.00391 deg` | 亚毫米结果级对齐 |
| 完整生成链 1 s RPE | translation mean/p95 `0.072/0.191 mm`；rotation mean/p95 `0.00104/0.00247 deg` | 亚毫米/毫度级 |
| 冻结 loop 隔离 ATE | translation mean/p95 `0.398/0.755 mm` | 说明剩余误差主要在 node↔Submap 局部测量，不在 loop pair |

9D backend 已按二进制探针实现 15 参数块 Exact 因子结构，8,480 条 IMU 样本组成 1,616 个因子。使用官方 topology/loop 隔离时 node ATE translation mean/p95 为 `0.139/0.337 mm`、rotation mean/p95 `0.000775/0.001348 deg`。完整生成链求解 42 次迭代成功，最终 cost `47,446.21`。

HybridGrid 高分在 `(2,1120)` 官方 FCS pose 上精确得到 `0.5435789227485657`，标准 `0.55` 门限拒绝。官方 Z1 `SAVE_ALL` 全重放确认 41 个候选的失败分布为 NONE/THRESH_ROT/HIGH_RES/ICP_STABILITY=`15/16/6/4`；FCS 以 `max(search_region.scaling)<0.1 m` 跳过 17 个。详见 `slam_probes/HYBRID_GRID_LOOP_EVIDENCE.md` 与 `slam_probes/ICP_STABILITY_EVIDENCE.md`。

标准产物位于 `../regression/2026-02-08_07.33.20/slam_raw/`，完整结论和复现命令见 `COMPLETE_SLAM_ALIGNMENT_20260825.md`。不得把当前结果称为 byte-exact：生成 HybridGrid 六图总体素规模差约 0.038%，局部 surfel/5,000 点 adaptive filter 仍有细微集合差，完整前端耗时 `752.39 s`；完整后端 warm run `148.94 s`，其中因子构建 `140.60 s`。

### 5.2 G11 Pandar 点云构建

`cpp/apps/pandar_cloud_pipeline.cpp` 是真实数据主实现，包含：

- 820-byte Pandar XTM 与 1206-byte VLP16 自动识别；
- block/beam 时间、ring、方位角、距离、强度解码；
- 逐点轨迹插值去畸变、设备区域/距离/强度/no-motion 过滤；
- 多层边缘候选和有序邻域；
- 垂直激光脚部 0.5 m 圆柱候选、固定种子 12345 的平面 RANSAC、1 cm 拟合和 2 cm 剔除阈值；
- 1 cm endpoint 体素、真实 ray origin 和 10 m `.raytile` 分片。

标准差距：

| 项目 | 原版 | 当前 C++ | 差距 |
|---|---:|---:|---:|
| 权威短窗脚部过滤输入 | 51,480 | 51,480 | 0 |
| 短窗脚部候选/剔除 | 4,238 / 1,808 | 4,238 / 1,808 | 标签逐点一致 |
| 短窗保留点位置 | — | — | 均值 1.53 µm，最大 0.030 mm；强度一致 |
| 精确 ns 10 秒设备/距离/强度/运动门限后 | 10,077,921 | 10,077,921 | 0 |
| 精确 ns 10 秒多层边缘过滤后 | 9,982,810 | 9,982,810 | 删除 TP/FP/FN=`95,111/0/0` |
| 精确 ns 10 秒 `origin/XYZ/intensity` 七字段记录 | 10,077,921 | 10,077,921 | 忽略并行顺序后位模式多重集合完全一致 |
| 旧全量 `float64` 流门限后（诊断） | 83,822,437 | 83,822,432 | -5，输入已丢低位纳秒 |
| 旧全量 `float64` 流 fringe 后（诊断） | 82,199,764 | 82,199,697 | -67，输入已丢低位纳秒 |

当前全量端点交付为约 24.893M 个 1 cm endpoint voxel；这是表面过滤前阶段，不能与原版最终 12.061M 或 159.147M 表面点数直接比较。

二进制确认端点轨迹查询使用 `trunc(double(scan_us)+double(relative)*1e6)`，ray origin 使用
`scan_us+trunc(float(relative*1e6F))`。两条路径必须分别查询，不能复用同一 pose。完整原始
bag 不在便携包内，所以当前状态为
`EXACT_FIELDS_AND_DECISIONS_10S / FULL_RAW_ACCEPTANCE_PENDING`。

### 5.3 全局自由空间叶集合

`pandar_shard_surface_filter.cpp` 保留跨分片原始射线历史，按 2 cm 紧凑占据叶统计 hit 和 traversal。最新同输入验收不使用最近邻伪配对，直接按同一世界锚点整数 key 比较：

| 项目 | 原版 | 当前 C++ | 差距 |
|---|---:|---:|---:|
| occupied keys | 2,667,377 | 2,667,377 | 完全一致 |
| hit counts | 2,667,377 | 2,667,377 | 全部一致 |
| intersection counts | 2,667,377 | 2,667,377 | 全部一致，差异 0 |
| deleted leaves | 852,434 | 852,434 | TP 852,434，FP/FN 0/0 |
| float32 centroid 逐位一致 | 2,667,377 | 2,662,135 | 5,242 个 µm 级算术顺序差异 |

质心距离 mean/max 为 `1.2833e-9/8.5963e-6 m`，不改变任何 hit、intersection 或 delete
决策；按当前结果验收口径计为无差异，单独的 bitwise 审计仍保留 5,242 这一事实。完整报告见
`../regression/2026-07-21_11.41.12/FREESPACE_INTERSECTION_EXACT_ALIGNMENT_20260828.md`。

### 5.4 遮挡清理与主表面输入

`binary_surface_pipeline.cpp` 已实现二进制的 helper surface、稀疏 Revelles octree、首个非空叶叶内原序、primitive early stop、endpoint 全局 kNN1 和单次 pair predicate。

冻结 raw-status capture：

| 项目 | 原版 | 当前 C++ | 差距 |
|---|---:|---:|---:|
| raw rays | 146,045 | 146,045 | 0 |
| retained rays | 138,708 | 138,708 | 0 |
| status exact | 146,045 | 146,045 | 逐条一致 |
| keep-mask exact | 146,045 | 146,045 | 逐条一致 |
| 遮挡后 1 cm 主输入 | 109,829 | 109,829 | 0 |
| 主输入字段 | — | — | XYZ/origin/intensity/weight 全部逐位一致 |

### 5.5 表面法向、位置与输出点集

主链顺序为 multi-scale normal → surface selection → invalid-normal removal → 1 cm output aggregation → density → Adaptive SOR → support/post normal。

| 阶段 | 原版 | 当前 C++ | 当前差距 |
|---|---:|---:|---|
| multi-scale normal | 109,322 | 109,322 | normal XYZ 与全部可观察语义字段 109,322/109,322 bit-exact；零法向 6/6 exact；角误差 p50/p95/max 均 0° |
| surface selection | 109,322 | 109,322 | weight 全部逐位一致；XYZ 109,003/109,322 逐位一致，最大差 0.954 µm |
| invalid normal removal | 109,322→109,316 | 109,322→109,316 | 数量/mask 一致 |
| output voxel aggregation | 109,316→87,255 | 109,316→87,255 | 87,255 个 1 cm key 全部一致；86,712 XYZ 逐位一致，平均误差 2.39 nm |
| density | 87,255→86,408 | 87,255→86,408 | 冻结输入上记录/mask 一致 |
| Adaptive SOR | 86,408→86,241 | 86,408→86,241 | 冻结输入上 mask 一致 |

完整 217-tile 同输入最终 PLY：

| 指标 | 原版 | 当前 C++/差距 |
|---|---:|---:|
| 最终点数 | 707,777 | 708,039；+262，+0.0370% |
| 同一世界 1 cm voxel 匹配 | — | 690,231；覆盖当前点的 97.4849% |
| 同 voxel 位置误差 | — | mean 0.0901 mm，p95 0.4354 mm，p99 2.337 mm |
| 同 voxel 无向法向角 | — | median 0.0567°，p95 2.7276° |
| 最近邻 clean→original | — | mean 0.4255 mm，p95 0.9683 mm，p99 9.344 mm |

剩余 262 点来自完整 tile 输入/顺序对 normal、selection 和阈值边界的影响。不得把完整 PLY 描述为 bytewise 完全一致；自由空间、遮挡状态和输出 voxel 拓扑可以在各自冻结验收范围内描述为完全一致。

2026-08-24 的性能对齐保留上述数值链和输出顺序，并逐步加入：83 个 10 m shard 并行预处理后按原文件顺序归并；所有 5 m tile 的 halo 单次有序装载；表面 kernel 复用 Adaptive SOR 首轮 KNN、支持邻域只做存在性查询、ray octree 返回只读叶引用；每个 OpenMP 线程复用 normal/selection/density/SOR/post/occlusion 的邻域 scratch；自由空间叶使用连续 candidate→leaf 映射并直接生成 1-byte 删除 mask。随后按安装二进制中的 CompactOctree、PCL FLANN 和 NanoFLANN 证据，把 normal 半径查询、density KNN 和法向空间 Adaptive SOR 改成连续平衡三维树；KNN 始终按 `(float32 squared_distance, original_index)` 保持精确顺序，normal 查询仍按 point index 排序后累加。完整 312-tile 的 8×4 阶段计时由最初 `1,949.74 s`、第一轮 `587.113 s`、哈希/KD-tree 优化后 `443.771/220.491/119.330 s`，继续降至 `79.960 s`，wall `80.42 s`。输出仍为 12,076,978 点、386,463,639 bytes，净室冻结 SHA-256 始终为 `0dcdc2729230d0ec770fb8ccbb6d16dbee1a71e8c156c5fed847d672ae3f4207`。最新峰值 RSS 为 11,859,244 KiB。

最新自由空间优化按 `CompactOctree::collectIntersected*` 反汇编，把每个 octree 节点重复 slab 除法改成根节点一次参数化、子节点中点传递和负方向 octant 反射；`+/-2` PCA 邻域改成连续二维列索引；`.raytile` 直接进入处理，取消 `.raymerged` 整盘复制。free-space/input 从 `51.989→26.624 s`，merge/input 从 `14.490→0.0003 s`；tile kernel `43.592 s`。原版同数据 `CloudProcessor` 为 `72.297 s`，当前完整 surface 为 `79.960 s`，约慢 `1.106×`，旧为 `1.65×`。10 秒和完整 PLY 均 byte-exact 于修改前净室冻结输出，自由空间删除数仍为 1,582,625；这不代表与 vendor 最终 PLY byte-exact，vendor 点数仍为 12,061,091。Release 使用 `-O3 -DNDEBUG`，生产线程继续使用 `8 tile × 4 inner`。最新完整报告见 `regression/2026-02-08_07.33.20/SURFACE_FREESPACE_ACCELERATION_20260824.md`。

该完整输出相对原版 12,061,091 点多 15,887（+0.1317%）。五个轨迹区域双向最近邻的平均 p50 约 `0.251 mm`、平均 p95 约 `4.80 mm`，约 98.3%/99.64% 在 1/2 cm 内。详细报告位于 `/media/cybergeo/12T/CSSJ/datasets_proc_regression_20260823/2026-02-08_07.33.20/data_regression_report.json`。

### 5.6 点云着色

`surface_panorama_colorizer.cpp` 的生产链包含 24 路 684×456 PCT 深度图、G11 mask/Vignetting4、二进制顺序权重图、逐点 Top-5、GammaModel 和精确五近邻几何加权外推。

122,701 点 V10 同序裁剪：

| 项目 | 结果/差距 |
|---|---|
| 几何、强度、法线、曲率、alpha | 全部逐位一致 |
| 直接/外推 mask | 122,701/122,701 一致；119,714 直接、2,987 外推 |
| 六张权重中间图 | float32 全部逐位一致 |
| 使用同次原版 GammaModel | RGB MAE 0.002700，RMSE 0.051965，99.40098% 点 RGB 完全一致，最大通道误差 1 |
| 净室自动 GammaModel | RGB MAE 0.057943，RMSE 0.240714，84.17780% 点 RGB 完全一致，94.20570% 通道一致，最大通道误差 1 |

早期全量曝光实现误把每个相机限制为 16 个一米空间体素；小裁剪只有 12 个体素，所以该 bug 没有被 V10 暴露。删除截断后，全量曝光点/体素为 `416,789/5,450`，与原版 exact；有颜色曝光点 `261,408` 对原版 `261,325`，差 +83。使用 GT 同序几何的净室自动 Gamma，RGB MAE 已由 `26.8549` 降至 `6.1138797/255`，PSNR `28.21 dB`；同一净室深度、视图、融合和 KNN 链加载原版 Gamma 时 MAE 为 `0.3221219/255`，98.98% 点最大通道差不超过 5。因此剩余差距定位在全场景自动曝光样本/目标，不在最终融合链。

2026-08-24 后续全量最终 OVS 动态捕获确认原版直接点为 `9,463,771`；净室自建深度为 `9,464,532`（+761），直接 mask IoU `99.9911%`。两者同点同 view 的 `27,749,713` 条观测中 RGB `99.9999928%` 完全一致、最大差 1；原版 Gamma 下净室直接点 RGB MAE 为 `0.22983/255`。此前把 `/tmp/nv_pct_captured_current.EA8nlh/depth` 当作 view 0..23 回放是错误的：旧 GDB 脚本按并行访问顺序编号，其中 12 张是相同的全 -1 图；该错误输入只得到 `318,626` 个直接点、错误 MAE 约 72 且 KNN 极慢。标准回归必须让净室自建深度，或使用由净室 `--depth-map-output-dir` 按真实 view ID 写出的文件；不得再使用该旧目录。

二进制证据确认 `Histogram8U::addData` 排除 HSV value=0，且每个 view 的三维 dynamic-range 残差块必须使用 `ScaledLoss(CauchyLoss(0.1), view_scale)`。此前只实现 scale、漏掉 Cauchy，导致全量 dynamic 约为裁剪 `107.06x` 时 Gamma 严重偏移。现在默认保留零值排除并补齐 robust loss，同时按原版先做 float32 `/255`、`/65535` 再提升 double，Gamma 应用也改为显式 `RGB -> HSV255 -> 修改 V -> RGB`。

同次原版 OVS 隔离求解的 joint/dynamic/scene 块数与初始代价 `780.9323` exact，净室最终代价 `0.1068101` 对原版 `0.1066163`；24 路 gain/exponent MAE 分别为 `0.00116786/0.00104385`。正确自建深度的全量自动 Gamma RGB MAE 已由旧零值 histogram 基线 `7.3246277/255` 降到 `0.7484060/255`，PSNR `37.9361 dB`，`98.9848%` 点最大通道差不超过 5；原版 Gamma 隔离仍为 `0.3221219/255`。wall `95.13 s`、RSS `3,544,752 KiB`，对原版 standalone `18.90 s` 仍约慢 `5.03x`。剩余差距是极少量 OVS/质量边界和未收敛平坦目标下的模型轨迹，不得宣称完全一致。最新路径和全部数字见 `../navvis_alignment_reference/COLOR_ALIGNMENT_DYNAMIC_CAUCHY_HSV_20260824.md`。

### 5.7 图像后处理

生产链实现 LibRaw 解码、白平衡/暗角、`0/1.5/3 EV` 三曝光、Mertens `1/1/1`、自适应 NLM、`sigma=3` 非锐化和 JPEG 95。二进制动态捕获和反汇编恢复了原版 NLM 公式：`gain=max(0,4.5+6*log2(ISO/100))`，`noise=gain+6*1.5`，`h=1.75*(7.5/1.75)^(noise/27)`，模板/搜索窗口为 `7/17`。程序保留未旋转的传感器栅格并按原版布局写 TIFF/EXIF APP1。

2026-08-27 修正 contrast 的全通道 `cv::Scalar::all(0.5)` 和 DNG 元数据 APP1 序列化后，
三曝光、Mertens、normalize/threshold、tone、NLM、Gaussian、unsharp 与 JPEG encoder 输入均
逐位一致。`2026-07-21_11.07.05` 前 6 个 capture 的 24 张 JPEG 对官方 `24/24` 整文件
byte-exact，解码像素 MAE `0/255`。原始 DNG 由 LibRaw/DNG SDK 直接读取，runner 不再调用
`nv_dng-converter`。该 `EXACT` 结论只覆盖冻结 Build ID 的标准 high-quality、无
`--blur-regions` 路径；fast/plain、隐私模糊和全景另行验收。完整报告见
`../regression/2026-07-21_11.07.05/IMAGE_POSTPROCESSING_EXACT_ALIGNMENT_20260827.md`。

### 5.8 全景与点云渲染

标准路径在 `2026-07-21_11.41.12` 全部 `00000..00033` 上完成 34/34 最终 JPEG
byte-exact 验收。旧六张 `NEAR`、7-band、mask IoU 和 1024 MAE 数据已经失效。

- 四层深度 PCG 的停止条件使用未预条件化 `r.dot(r) < 1e-15`；raw double 只宣称机器精度
  数值等价，float/毫米量化效果和全部下游结果 exact。
- 2K 连续 float world map 求 soft BGR gain 与 GraphCut；8K 使用 native dense depth 的
  float→毫米 uint16 截断→double `0.001`→float→线性 resize，并复用 2K gain/seam。
- 像素中心球面射线、OCam 投影、四路 2K/8K projection mask 和 exposure 图像均 exact。
- GraphCut pair 为 `0-1、1-2、2-3、0-3`，移位 1228/819，camera 2 MIRA 窗口从列 1025
  开始；seam mask 及 seam-prepared 图像 exact。
- 主混合为十层 float32 circular multiband。floor mask 是四路原生 8K projection mask 的
  OR；no-floor JPEG 写出/读回、wrapped PyramidInpainting、最终 q95 optimize=1 的顺序不可省略。
- operator-mask 标准链 34/34 exact，no-mask 受控链也 exact。`--surface-cloud` 路径使用
  PCL 0.05 m octree、double 像素中心射线、float 距离归约和 float 毫米截断；34/34 个
  1024×512 sparse depth exact，`00000` Surface→最终 8K JPEG byte-exact。尚未宣称
  34 帧 Surface→8K final 全量实跑。
- `00017` clean 17.88 s / 3,713,696 KiB，NavVis 12.31 s / 2,504,544 KiB；结果 exact，
  性能慢约 1.45×、内存约 1.48×。

完整报告：`../regression/2026-07-21_11.41.12/PANORAMA_FULL_EXACT_ALIGNMENT_20260827.md`；
机器结果：同目录 `panorama_full_exact_alignment_20260827.json`；守门：
`../scripts/verify_panorama_full_exact.sh`，预期 `captures=34 byte_exact=34 mismatches=0`。
Surface 上游报告：
`../regression/2026-07-21_11.41.12/PANORAMA_SURFACE_TO_FINAL_BYTE_EXACT_20260827.md`。

### 5.9 Floor estimator 与 mapped-space quality

- `floor_estimator.py` 已恢复顺序 floor 状态机、负高度向零截断 histogram、tiny/adjacent merge、double-floor split 和全 trace 种子重放。2026-08-28 扫描 220 组原版 `artifacts/trace.csv`，共 1,796,815 条样本和 615 个参考 floor，最终顶层 JSON 数组、z 边界与全部纳秒 time range 220/220 逐字段一致。runner 已移除简化 `floor_summary`，从 raw `/imu/magnetic_field` 与轨迹生成 6 有效数字的官方 trace，再写 `artifacts/floors.json`；`2026-07-21_11.07.05`（5,374 条/1 层）和 `2026-01-19_19.04.51`（17,878 条/5 层）的 trace byte-exact、floors exact。守门见 `FLOOR_FULL_ALIGNMENT_20260828.md`。
- Mapped-space quality 的生产 C++ 已实现完整四字段聚合和 v2 序列化，runner 在 Surface 前从
  exact per-ray shard 生成 `mapped_space/`。64-ray、5,147 条真实 G11 ray、every-nth/min-rays
  分支以及 135,658,017-ray 全量对官方核心均 Exact；方向最近邻必须用 float32 squared distance，
  `max-ray-length=50 m` 只影响官方分区、不裁短射线。官方 estimator 外层会额外提交 3 条 ray，
  并存在 parallel merge diversity 漂移，所以冻结命令行文件仍有 21 count/5 diversity 残差且
  不 byte-exact；不得用 key 特判复刻某次调度。

2026-08-24 的逐模块结果/效率统一验收见
`../regression/2026-02-08_07.33.20/MODULE_ACCEPTANCE_20260824.md` 和
`../regression/2026-02-08_07.33.20/module_acceptance_20260824.json`。

### 5.10 二进制性能分析与小数据回归

逐模块原版/净室时间、原版二进制数据结构证据、无损优化顺序和 24/52/132-view 小数据回归见
`../regression/2026-02-08_07.33.20/MODULE_PERFORMANCE_BINARY_ANALYSIS_20260824.md` 与同名 JSON。
当前结论是编译选项不是主因：surface 受控仍慢 `6.14×`，核心差异是原版全局 CompactOctree、
连续 Trace 与 NanoFLANN/PCL KD-tree，而当前仍有 shard-local hash/均匀网格；着色受控旧路径慢
`5.03×`，本轮已修正曝光 cache 反复解码和完整外部 depth 下的重复 PCT 渲染。小数据 OVS、
Gamma、mask、最终 PLY 验收保持 byte-exact，但修改后的完整 color 模块尚未重跑，不得把小样本
`4.69×/7.29×` 直接宣称为全量收益。同步源码 SHA-256 为
`0d9f4936b116ab1464d4203fe4a90bf1717bcabcb4332104e17efa80a71951a4`。

### 5.11 2026-08-24 重测与无损加速

最新性能真值见
`../regression/2026-02-08_07.33.20/PERFORMANCE_REBENCHMARK_ACCELERATION_20260824.md` 和同名 JSON。
Adaptive SOR 的法向空间索引 cell 从 `0.05` 改为 `0.01`，该调用显式扩展到 `3.2` 以保持同一
distance/index Top-11；完整 surface `443.771→220.491 s`，输出 SHA 仍为
`0dcdc2729230d0ec770fb8ccbb6d16dbee1a71e8c156c5fed847d672ae3f4207`，相对 NVS
`72.297 s` 仍慢 `3.05×`。完整冻结-Gamma 着色因跳过已有 depth 下的重复 PCT 渲染，从
`95.13→56.06 s`，输出 SHA 仍为
`ac1930ae74fc5e4b45c747c14ca9422b6336bb5b7a0582ea8475cde018b20057`，相对 NVS
仍慢 `2.97×`。连续 flat index、单次最大 normal 查询、增量 shell 和 SOR cell `0.005/0.020`
均已实测否决，不要重复采用。资源 `build-release/navvis_recon_shard_surface_filter` SHA 为
`6e63d499c3242f22aa75932cc1405c073502960afe559ce404e2475710b5b2a8`。

## 6. 构建与测试

从本目录执行：

```bash
cmake -S cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp -j8
ctest --test-dir build-cpp --output-on-failure
PYTHONPATH=src python3 tests/test_reconstruction.py
python3 tests/test_complete_slam_evaluator.py
python3 -m py_compile runner/navvis_postprocessing_recon.py
```

当前预期：C++ CTest `2/2` 通过；Python reconstruction `28/28`、complete-SLAM
evaluator `4/4` 通过。

表面冻结验收从工作区根目录执行：

```bash
outputs/navvis_non_slam_reconstruction/build-cpp/navvis_recon_surface_capture_acceptance \
  --occlusion outputs/navvis_alignment_reference/g11_pre_surface_filter_status_mask_probe_v1_raw_status52.bin

outputs/navvis_non_slam_reconstruction/build-cpp/navvis_recon_surface_capture_acceptance \
  --occlusion-main-input \
  outputs/navvis_alignment_reference/g11_pre_surface_filter_status_mask_probe_v1_raw_status52.bin \
  outputs/navvis_alignment_reference/surface_main_input_clean_occlusion_probe_v1.bin

outputs/navvis_non_slam_reconstruction/build-cpp/navvis_recon_surface_capture_acceptance \
  --voxel-compare \
  outputs/navvis_alignment_reference/surface_intermediates_probe_v1_03_output_voxel_aggregation_before_input.bin \
  outputs/navvis_alignment_reference/surface_intermediates_probe_v1_03_output_voxel_aggregation_after_output.bin
```

完整替代入口：

```bash
./run_navvis_recon.sh \
  --bagplayer-args=--quiet \
  --proc-base-dir=/path/to/datasets_proc \
  --caller=sitemaker \
  /path/to/datasets_rec/DATASET_ID \
  --trajectory-bag=/path/to/reference/artifacts/trajectory.bag \
  --slam-reference-bag=/path/to/reference/artifacts/trajectory.bag \
  --aligned-standard --force --res=0.01 --cloud-format=ply \
  --preset=standard --num-threads-panos=32
```

## 7. 修改规则

- 修改 C++ 算法后至少运行 CMake build、CTest 和受影响的冻结 capture 验收。
- 修改 runner 后运行 `py_compile`，并至少做一次短时 `--max-duration` 端到端测试。
- 修改 SLAM 轨迹适配后必须同时报告绝对轨迹误差和固定时间间隔 RPE；不得用刚性配准后的误差代替原坐标系误差，也不得把轨迹侧融合称为完整离线 SLAM。
- 修改着色时必须分别报告“原版 Gamma 隔离”和“净室自动 Gamma”，不得混用。
- 着色回归若使用外部 `--depth-map-input-dir`，必须证明 24 个文件按真实 view ID 编号并逐图记录 SHA；旧 `nv_pct_captured_current.EA8nlh/depth` 是并行访问顺序，禁止作为标准输入。
- 修改图像/全景时必须标明是当前探针还是旧 299 张全量，不能用过时全量掩盖当前回归。
- 对 G10/VLP16 只能声称功能兼容和内部一致性；没有原版 GT。
- 新评测必须记录输入路径、二进制 SHA/Build ID、命令、点数、误差方向和机器可读 JSON。
- “完全一致”必须限定字段、阶段和输入；最终完整 PLY 当前仍差 262 点。

## 8. 会话清理后的着色继续入口

资源包根目录为 `/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment`。继续 2026-08-24 的全量着色对齐时，先读资源包根 `AGENTS.md`，再使用：

- 代码：`code/cpp/apps/surface_panorama_colorizer.cpp`；
- 最新结论：`regression/2026-02-08_07.33.20/color_exposure_dynamic_cauchy_20260824/COLOR_ALIGNMENT_DYNAMIC_CAUCHY_HSV_20260824.md`；
- 指标 JSON：同目录 `color_dynamic_cauchy_hsv_full_20260824.json`；
- 曝光 OVS、GammaModel、日志和探针：`test_resources/color_exposure_dynamic_cauchy_20260824/`；
- 24 张真实 view ID 深度：`test_resources/color_final_ovs_20260824/clean_depth/`。

先运行 `scripts/build_and_test.sh` 或第 6 节命令。求解器隔离必须解压 `original_exposure_ovs.bin.zst` 到新建临时目录，使用 Ceres worker 的 `--exposure-ovs-binary-input`，并检查 `168827/516152`、初始代价 `780.9323` 和最终代价约 `0.1068101`。随后才允许运行全量最终着色；完整可复现命令保存在资源目录的 `run.time`。原版 Gamma 隔离与净室自动 Gamma 必须分表报告。资源包修改后重新生成并以 `sha256sum -c MANIFEST.sha256` 验证清单。
