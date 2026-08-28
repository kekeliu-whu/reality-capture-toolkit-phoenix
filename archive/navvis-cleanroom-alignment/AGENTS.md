# NavVis 后处理与轨迹侧 SLAM 净室重构：代码与对齐状态

## SurveyorSLAM 全自主链工程对齐基线（2026-08-28）

`2026-07-21_11.07.05` 已完成 raw 双 Pandar + raw IMU → C++ Frontend → 自产回环 →
Ceres Stage1 → Ceres Stage2 的 10,933-node 全量运行。求解过程不读取官方 pose/Submap/loop/
optimization state；官方文件只在 Stage2 完成后参与评测。单个全局 SE(3) gauge 对齐后，最终
ATE translation mean/p95/max 为 `0.654/1.343/1.938 mm`，rotation mean/p95 为
`0.01066/0.01950°`；1 s RPE translation mean/p95 为 `0.181/0.443 mm`。按用户要求的结果
接近口径判定为 `NEAR / 工程已对齐`，不得称为 byte-exact 或 topology-exact。

自产有效回环 789 条；与官方 862 条 pair 集的 TP/FP/FN 为 `493/296/369`，所以严格回环拓扑
仍为 MISMATCH。生产入口为 `code/run_navvis_slam_recon.sh`，输出的
`optimized_trajectory.csv` 已由后处理 runner 的 `--trajectory-csv` 直接接收。详细边界、命令和
机器结果见
`regression/2026-07-21_11.07.05/SLAM_FULL_AUTONOMOUS_ALIGNMENT_20260828.md`。

## SurveyorSLAM C++ 前端基线（2026-08-28）

生产前端现已由单一 ELF `navvis_recon_slam` 直接执行，Python 只保留为离线 oracle。C++17
链路包含 NVSLAM6 mmap/packet 索引、双 Pandar 50 ms/58,000 raw-slot collator、ROS1 raw IMU
读取、逐射线去畸变、0.04 m centroid、三级 split-surfel/UniBN octree ICP、双活动 Submap、
HybridGrid 和 MotionFilter；入口为 `code/cpp/apps/slam_pipeline.cpp`，核心位于
`code/cpp/src/slam_{archive,imu,batch_collator,rosbag,frontend}.cpp`。首个 Submap 必须保持单位
旋转，后续 Submap 才调用完整重力对齐四元数链；MotionFilter 角阈值为 `0.02°`。

冻结 `2026-07-21_11.07.05` 回归中，100 retained-node C++ wall/core 为 `4.83/4.745 s`，
Python oracle core 为 `12.421 s`，核心加速 `2.62×`。相对官方 100 节点，translation
mean/p95/max 为 `5.34/13.00/17.23 µm`，rotation mean/p95/max 为
`0.000148/0.000468/0.000749°`；Surfel 三层为 `15438/2621/528`，官方
`15437/2620/528`，HybridGrid 为 `32941`，官方 `32942`。503 个自主 batch 产生 500 节点，
core/wall 为 `27.993/28.11 s`，对 Python 500-node `77.948 s` 加速 `2.78×`；500 个官方
时间戳中 498 个 exact，另两个 packet boundary 分别早约 3.01/1.93 ms，轨迹 translation
mean/p95/max 为 `9.14/19.42/41.05 µm`。这些属于用户认可的 µm 级结果相同，不得表述为
byte-exact SLAM。

结果保持型优化包括持久 cell-label 哈希、工作缓冲复用和独立 surfel PCA 并行；优化前后
100/500 节点 CSV 均逐字节相同。全 CMake 树已同时通过 Ceres 2.0 legacy
LocalParameterization 与 Ceres 2.2 Manifold 构建，CTest `4/4`。当前 C++ 可执行程序完成的是
自主局部前端；该报告记录的是后端接线前的阶段性状态，现已由上面的全自主链基线取代。详细报告见
`regression/2026-07-21_11.07.05/SLAM_CPP_FRONTEND_20260828.md`。

## Multi-scale normal 完全对齐基线（2026-08-28）

冻结 G11 同输入 `multi_scale_normal_estimation` 已从当前源码新鲜构建并完成逐位验收：
109,322/109,322 个 normal XYZ、XYZ、intensity、curvature 和 weight 可观察语义字段逐记录
bit-exact；双方零法向均为同一 6 点，角误差 P50/P95/max 均为 `0°`。`OMP_NUM_THREADS=1/32`
结果一致，CTest `2/2` 通过。旧 `P95 0.002241° / 11 个异常点 / max 1.11017°` 是未复刻
CompactOctree 枚举与 float32 eigensolver 路径的 NumPy 独立复算结果，不代表当前生产 C++。
完整证据见
`regression/2026-02-08_07.33.20/MULTISCALE_NORMAL_EXACT_ALIGNMENT_20260828.md` 和同目录 JSON。

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
`regression/2026-07-21_11.41.12/MAPPED_SPACE_QUALITY_ALIGNMENT_20260828.md`。

## 点云全自动曝光/着色完全对齐基线（2026-08-27）

`2026-07-21_11.41.12` 的冻结同几何回归现已从 0.1 m 曝光采样云开始，不注入官方 OVS、
GammaModel、direct mask 或颜色，完成自动曝光、最终 Top-5、直接融合和 KNN 补色。44,544 个
曝光质心的 XYZ/法向/Morton 顺序 exact；168,393 条有效 exposure OVS 的 view/RGB/quality
逐字节 exact；136/136 个 GammaModel 参数及文本文件 byte-exact；12,504,451 条有效 final OVS、
direct mask 和 KNN 两个有序分区 exact。最终 2,857,623 个 PLY 的 36 字节点记录 body
byte-exact，RGB changed points `0`、MAE `0/255`、max delta `0`。完整文件只因 321 字节 PLY
header 元数据不同而 SHA 不同。

最终守门还把官方带色 PLY 转成仅含八个非颜色 float 字段的 Surface PLY，再从零生成曝光、
Gamma、最终 OVS、融合和 KNN。无颜色输入运行的五个结果文件与带色输入运行整文件一致，最终
PLY body 仍与官方 byte-exact，排除了继承输入 RGB 的假对齐。转换工具为
`scripts/strip_binary_ply_color.py`。

最后四个根因是：曝光聚合前漏做输入法向单位化、曝光云漏做 PCL Morton 输出、把仅属于最终
14 字节适配器的 Fibonacci 法向量化错误用于 48 字节曝光 PCL 云，以及图像金字塔只建 6 层
而官方保留到 level 7。新鲜最终路径还必须以 packed OVS 的 `quality>0` 为融合边界。完整证据、
命令和机器指标见
`regression/2026-07-21_11.41.12/COLOR_FULL_AUTOMATIC_EXACT_ALIGNMENT_20260827.md` 与
`regression/2026-07-21_11.41.12/color_full_automatic_exact_alignment_20260827.json`；当前产物在
`work/color_alignment/official_single_exact_20260827/clean_full_exact_uncolored_input/`。

两份 OVS 整文件 SHA 仍会因 `quality=0` 的哨兵/未定义 padding payload 不同；有效 mask 和所有
`quality>0` payload 均 exact，且该 padding 不进入任何目标、融合或最终结果。该 EXACT 结论限定
于冻结 Surface、相机、标定和 136 张已验收 depth；不得外推为自主 Surface 或完整 SLAM 已 exact。

## 全景渲染完全对齐基线（2026-08-27）

标准 G8 全景路径现已从处理后的四路相机 JPEG 和 `pano_depth_sparse.png` 开始，到最终
8192×4096 filled JPEG 完全对齐。在 `2026-07-21_11.41.12` 的全部 34 个 capture 上，净室
C++ 与冻结 NavVis 结果逐文件 `byte-exact`（34/34，mismatch 0）。2K/8K projection mask、
曝光输入、GraphCut seam、seam preparation、十层 multiband、BinaryMask、no-floor JPEG、
JPEG round-trip 和 wrapped nadir/floor fill 均已分阶段 exact。完整报告和机器指标见
`regression/2026-07-21_11.41.12/PANORAMA_FULL_EXACT_ALIGNMENT_20260827.md` 与
`regression/2026-07-21_11.41.12/panorama_full_exact_alignment_20260827.json`。

上游 `--surface-cloud` 旁路也已在同一冻结数据上完成独立验收：34/34 个 capture 的
1024×512 稀疏毫米深度逐像素 exact，`00000` 的正式 C++ 路径从 Surface PLY 到最终 8K JPEG
与官方文件 byte-exact。PCG 四层 raw double 在该路径上也已逐位一致。报告见
`regression/2026-07-21_11.41.12/PANORAMA_SURFACE_TO_FINAL_BYTE_EXACT_20260827.md`。单张
`00017` clean 为 17.88 s，NavVis 为 12.31 s，结果已对齐但性能仍慢约 `1.45×`。点云着色
现在另有上述完整自动曝光 Exact 验收；两者仍是独立模块结论，不得把任一结论外推到自主
Surface 或完整 SLAM。

## 已对齐 C++ 路径的可读性基线（2026-08-26）

已在原文件中整理全局自由空间、Binary Surface 和固定输入最终着色路径。统一格式由
`code/cpp/.clang-format` 定义；不要复制一套“clean/readable”旁路实现。主要入口现在按阶段阅读：

- Surface：`prepareInputShards` → `preprocessShard` → `processSurfaceTile` → 有序归并；
- 着色：`loadColoringScene` → `DepthMaps`/`prepareDepthAndCloudBounds` →
  `buildVoxelViewRankings` → `writeColoredCloud`。

`2026-07-21_11.41.12` 回归中，2,812,840 点 Surface PLY、2,857,623 点固定
depth/Gamma 着色 PLY 和 direct mask 均与重构前 byte-exact；CTest `2/2` 通过。完整记录见
`regression/2026-07-21_11.41.12/ALIGNED_CPP_READABILITY_REFACTOR_20260826.md`。这个 exact
结论只覆盖上述 Surface/着色冻结输入，不能外推到 Pandar、SLAM、图像后处理或全景模块；
这些模块必须分别采用各自最新验收，其中图像后处理和标准全景现已有独立 `EXACT` 结论。

## 0. 便携包状态（2026-08-25）

本目录已整理为约 1.9 GB 的便携源码/测试包。历史全量点云、raw shards、panorama 输出、
性能实验工作目录及可由 raw frame 重建的 SLAM scan cache 已移出代码树；报告中的部分历史
产物路径因此仅作为结果记录，不能假定文件仍存在。先阅读 `PACKAGING.md`，按其中的标准顺序
构建和回归，所有新生成成果必须写到代码根目录之外。

最新累计链回归已在较大的 `2026-07-21_11.07.05` 完成 `4/34/234/1234` 四组实跑。模块 4
使用官方 Surface 时 XYZ/强度/法线/曲率 bit-exact，正式完整运行 RGB MAE
`0.758742/255`；模块 3 固定同一 Gamma 后对最终 PLY 的增量仅 `0.000442/255`。完整
1+2+3+4 为 11,167,627 点，对官方多 4,880 点（+0.04372%），五 ROI clean→official
最近邻均值/p50 约 `1.405/0.589 mm`，1 cm 覆盖 `98.438%`。模块 1 为 `168.36 s`，比官方
快 15.4%；模块 2 热链路慢 `1.32×`，模块 3 慢 `0.86%`，模块 4 慢 `7.79×`，故当前首要
性能差距是深度/曝光/颜色融合。官方仍未保留 CloudBuilder ray-history，`234` 使用净室模块 1
冻结输入，不能称为严格 vendor-1 隔离。详见
`regression/2026-07-21_11.07.05/PIPELINE_MATRIX_4_34_234_1234_20260825.md`。此前小数据
`2026-07-21_11.41.12` 回归继续保留，用于快速迭代，不覆盖这次大数据结论。

2026-08-26 又在空工作目录对 `2026-07-21_11.07.05` 完整重跑 clean `1→2→3→4`，没有
复用 ray-history、Surface、相机 JPEG、深度或 Gamma。28 GB raw shards、Surface、360 张
相机图、360 张深度图、direct mask 和 exposure OVS 均与 2026-08-25 运行一致；自动 Gamma
轻微漂移只造成最终 RGB MAE `0.028822/255`，几何 bit-exact。冷启动四模块合计
`1,913.58 s`，官方对应 `1,445.46 s`，慢 `1.324×`；模块 4 仍是主要性能瓶颈。详见
`regression/2026-07-21_11.07.05/FULL_RERUN_20260826.md`。

同日完成模块 4 的结果保持型加速：一米格保守相机剔除、复用 OCam 投影、固定输入跳过冗余
bounds 扫描、四 worker 并行相机图预处理、曝光/最终着色共享缓存，以及 PCT 视图外层并行。
小数据固定 depth/Gamma 从 `31.80 s` 降到 `11.47 s`，PLY 和 direct mask byte-exact，已快于
NVS 的 `12.24 s`。`2026-07-21_11.07.05` 完整自动模块 4 从 `302.28 s` 降到
`188.98 s`，但仍慢于官方 `44.107 s` 的 `4.28×`；固定旧 Gamma 的全量最终 PLY byte-exact，
360 张 depth、direct mask、exposure OVS 和 final OVS 也全部 exact。剩余性能瓶颈已收敛到
PCT ray-depth，详见
`regression/2026-07-21_11.07.05/COLORIZER_ACCELERATION_20260826.md`。

同日又从空目录完成该数据的 clean `1→2→3→4` 全量 5 mm 重跑，`--res 0.005` 和
`--output-cell 0.005` 均已进入 C++ 核心链。Surface/最终 PLY 均为 40,094,930 点，是同代码
1 cm 结果的 `3.59028×`；最终 PLY 对 Surface 的 XYZ、强度、法向和曲率全部 bit-exact。
39,583,244 点直接相机着色，511,686 点经五近邻补色。四模块合计 `3,343.15 s`
（55:43.15），约为 1 cm 冷重跑的 `1.747×`。本轮未生成官方 NVS 5 mm GT，结论只覆盖 clean
全链成功、5 mm 参数生效与内部完整性，不得称为官方 5 mm 最终 PLY 对齐。详见
`regression/2026-07-21_11.07.05/FULL_RERUN_5MM_20260826.md`。

其他未完成事项：5 mm Surface 已修复 `--output-cell` 未传入核心流水线的问题；1 cm 冻结小样
SHA 保持完全一致，5 mm 小样点数按预期增加。全量 5 mm 验证因打包清理中止，迁移后应优先
复跑。着色已在 `2026-07-21_11.41.12` 上完成确定性单线程官方分支的完整自动曝光 Exact
验收：32,274 个 joint block、138,899 个标量 residual、136 个 GammaModel、最终有效 OVS
和 PLY 点记录均 exact。旧 `RGB MAE 0.172450/255`、少 7 个 block/13 个 residual 的结论已
失效。固定 depth 的完整净室链为 14.79 s；完整自动链的主要剩余速度差距仍是未包含在该计时
中的 PCT ray-depth。原版 32 线程 Gamma 本身非确定；严格逐位回归必须使用同一官方单线程
分支，并同时守门 OVS、目标结构、模型和最终结果。

本文件适用于本目录及全部子目录。它汇总共享工作区中两个相关 Codex 任务已经落盘的代码，给后续开发者提供唯一的模块地图、标准评测口径和剩余差距。

## 1. 范围与真实性

- 本项目是 C++17/Python 的独立净室实现，不是 NavVis 厂商逐字源码。
- 项目现已包含从双 Pandar 原始扫描和 raw IMU 到 1,617-node 离线轨迹的完整 SurveyorSLAM 净室链：去畸变、三级 surfel ICP、双活动 Submap、HybridGrid、候选采样、回环和 9D IMU 稀疏图优化均已实现。拓扑/回环集合 exact，最终轨迹达到亚毫米结果级对齐；不是厂商逐字源码，也不是 protobuf/浮点逐位复刻。
- 2026-08-27 的 `2026-07-21_11.41.12` 更新把首 5 秒 98-node raw-IMU/局部 ICP 前端推进到浮点舍入误差级，并完成原生 C++/Ceres Stage 1 `ImuCostFunctionFast`。完整 2,660-node Stage 1 后端 final cost 与官方精确重算值仅差 `6.17e-9`，ATE translation max `1.96e-8 mm`，可在声明容差和冻结 Build ID 下称为结果 exact，但不是 protobuf byte-exact。大数据全自主链现已工程对齐；全量 loop pair topology 仍不 exact，所以不得称为严格完全一致。
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
精确时间 3-scan/10 秒回归的全部分类和官方七字段 `PointRayIntensity` 记录差异。2026-08-27
恢复 `2026-07-21_11.07.05` 全量原始 bag 后，进一步按官方 Matrix4f/四元数双坐标路径和
PlaneFilter 边界控制流修复 Fringe/Foot：21,609 个接受扫描的输入、20,414 个有效 Fringe
结果及 10,207 个有效组合 Foot 结果均逐扫描 exact。旧 `float64` 流的 `-5/-67` 只保留为
有损输入诊断，不再代表当前全量 raw 状态。以当前工作树和最新报告为准，不以任一旧任务的
最终消息作为代码真值。此目录没有可用 Git 历史来可靠标记逐行会话来源。

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
- `occlusion_octree_query_binary_report_v1.md`：遮挡 octree 查询状态机；
- `surface_backhalf_binary_report_v2.md`：表面后半链和输出体素；
- `multiscale_normal_binary_report_v1.md`、`surface_selection_binary_report_v1.md`：法向与表面位置；
- `regression/2026-02-08_07.33.20/MULTISCALE_NORMAL_EXACT_ALIGNMENT_20260828.md`：
  Multi-scale normal 当前 109,322/109,322 逐位 Exact 验收；优先于旧 NumPy 独立复算角差。
- `CLEANROOM_COLOR_ALIGNMENT_V10.md`：当前着色裁剪验收；
- `ALIGNMENT_STATUS_V11.md`：Pandar、图像和全景的历史综合状态；
- `ALIGNMENT_1_2_3_20260824.md`：本轮完整 SLAM、全量着色和全景的基线结果；
- `ALIGNMENT_FOLLOWUP_20260824.md`：样本级 IMU 结构、全量最终着色 OVS 和四层全景深度优化的最新复核，优先于前述报告的冲突结论。
- `COLOR_ALIGNMENT_DYNAMIC_CAUCHY_HSV_20260824.md`：全量曝光 dynamic Cauchy、float32 归一化和 HSV255 应用路径修复后的最新着色验收；优先于旧的全量自动 Gamma 数字。
- `regression/2026-02-08_07.33.20/CLOUDBUILDER_ACCELERATION_20260824.md`：Pandar/CloudBuilder 批量并行、完整速度和 byte-exact 守门；优先于旧性能报告中的 139/186 秒数字。
- `regression/2026-02-08_07.33.20/PANDAR_EXACT_ALIGNMENT_20260826.md`：Pandar 运行时标定
  位模式、精确纳秒输入和 multilayer predicate 的最新结果验收；其分类结论优先于全部旧
  Pandar `-978/-15,705` 数字。
- `regression/2026-07-21_11.07.05/PANDAR_CLOUDBUILDER_FULL_EXACT_ALIGNMENT_20260827.md`：
  全量 raw 双雷达状态机、Fringe、独立/组合 Foot 和七字段的最新验收；判断 G11
  Pandar/CloudBuilder 结果时优先于 2026-08-26 的 10 秒报告。
- `regression/2026-02-08_07.33.20/MODULE_REBENCHMARK_20260825.md`：2026-08-25 当前 Release 的全模块实跑时间与结果一致性；其时间和完整输出指标优先于所有 2026-08-24 性能汇总。
- `regression/2026-02-08_07.33.20/slam_raw/COMPLETE_SLAM_ALIGNMENT_20260825.md`：完整 raw-Pandar/raw-IMU SurveyorSLAM 的标准链、全量误差、证据边界和复现命令。
- `regression/2026-07-21_11.41.12/SLAM_IMU_BACKEND_ALIGNMENT_20260827.md`：首 5 秒前端、Stage 1 原生 Ceres 浮点精度 exact 结果和 Stage 2 IMU 联合标定的最新隔离验收；判断后端时优先于 2026-08-25 的固定标定结论。
- `regression/2026-07-21_11.41.12/SLAM_STRICT_SUBMAP_ACCEPTANCE_20260827.md`：Submap schema v2、HybridGrid 和约束图严格验收语义；缺 generated 字段时必须报告 unavailable，禁止复制官方值形成伪 exact。
- `regression/2026-07-21_11.41.12/COLOR_ALIGNMENT_20260825.md`：小数据集着色的历史基线及
  Ceres 并行分支非确定性证据；结果结论已被下面的 2026-08-27 Exact 报告取代。
- `regression/2026-07-21_11.41.12/COLOR_FULL_AUTOMATIC_EXACT_ALIGNMENT_20260827.md`：
  0.1 m 曝光云、有效 exposure/final OVS、GammaModel、direct/KNN 分区和最终 PLY 点记录的
  最新 Exact 验收；判断冻结同几何着色结果时优先于全部旧 `NEAR`/MAE 报告。
- `regression/2026-07-21_11.41.12/PANORAMA_FULL_EXACT_ALIGNMENT_20260827.md`：标准 G8 sparse-depth 到 8K filled JPEG 的 34/34 byte-exact 全量验收；其全景结果结论优先于全部旧 `NEAR`、六张和 7-band 报告。
- `regression/2026-07-21_11.41.12/PANORAMA_SURFACE_TO_FINAL_BYTE_EXACT_20260827.md`：
  Surface PLY→PCL octree sparse depth→四层 PCG→官方 8K final 的最新验收；34/34 个
  1024×512 sparse depth exact，`00000` 最终 JPEG byte-exact。
- `regression/2026-07-21_11.41.12/PIPELINE_MATRIX_4_34_234_1234_20260825.md`：同一数据上的 `4/34/234/1234` 累计模块实跑、最终 PLY 对官方比较、固定 Gamma 隔离和累计性能；判断模块间最终影响时优先使用。
- `regression/2026-07-21_11.07.05/PIPELINE_MATRIX_4_34_234_1234_20260825.md`：较大数据集上的最新 `4/34/234/1234` 累计链回归；包含 5.46 亿原始回波、11.16M 最终点、逐模块时间、五 ROI 几何/颜色及曝光非确定性隔离，优先于 `11.41.12` 的累计链数字。
- `regression/2026-07-21_11.07.05/FULL_RERUN_20260826.md`：从空目录完整重跑 `1→2→3→4` 的最新确定性和冷启动性能复核；判断跨运行稳定性时优先使用。
- `regression/2026-07-21_11.07.05/COLORIZER_ACCELERATION_20260826.md`：模块 4 的保守视图剔除、并行相机预处理、共享缓存和 PCT 外层并行回归；包含小/大数据 byte-exact 门槛与最新速度，性能结论优先于旧着色报告。
- `regression/2026-07-21_11.07.05/IMAGE_POSTPROCESSING_EXACT_ALIGNMENT_20260827.md`：模块 3 high-quality 图像后处理的最新逐阶段、JPEG 文件和纯净室直读 DNG 验收；结果状态优先于旧 `NEAR_HYBRID` 报告。

## 4. 代码模块地图

| 模块 | 生产代码 | 入口/辅助代码 | 标准状态 |
|---|---|---|---|
| 调度、轨迹与标定 | `runner/navvis_postprocessing_recon.py` | `run_navvis_recon.sh`、`src/navvis_recon/pipeline.py`、`models.py` | 控制窗口和相机/雷达外参已接线 |
| SurveyorSLAM 前端/后端 | `code/cpp/apps/slam_pipeline.cpp`、`code/runner/navvis_slam_recon.py`、`code/src/navvis_recon/surveyor_{frontend,slam}.py`、两个 Ceres solver | `code/run_navvis_slam_recon.sh`、`code/runner/navvis_slam_archive.py`、完整 SLAM evaluator | 10,933-node 全自主链已串联；最终 ATE mean/p95 `0.654/1.343 mm`，工程已对齐；回环 pair topology 非 exact |
| 激光解码与点云构建 | `cpp/apps/pandar_cloud_pipeline.cpp`、`cpp/src/cloud_builder.cpp` | `runner/capture_laser_frames.py`、`cpp/include/navvis_recon/cloud_builder.hpp`、`src/navvis_recon/cloud_builder.py` | G11 精确 ns 全量 raw：21,609 个接受扫描输入 exact；20,414 个有效 Fringe 和 10,207 个有效组合 Foot 逐扫描 exact；`origin/XYZ/intensity` 七字段位模式多重集合 exact。结果状态 `FULL_RAW_DECISIONS_EXACT / OBSERVABLE_FIELDS_EXACT`；完整输出 wall 待同口径复测 |
| 全局自由空间 | `cpp/apps/pandar_shard_surface_filter.cpp`、`cpp/src/cloud_surface_filter.cpp` | `cpp/include/navvis_recon/cloud_surface_filter.hpp` | 同输入 occupancy/hit/intersection/delete mask 完全一致 |
| G11 遮挡与表面点集 | `cpp/src/binary_surface_pipeline.cpp` | `cpp/include/navvis_recon/binary_surface_pipeline.hpp`、`cpp/tests/surface_capture_acceptance.cpp` | 遮挡、输入聚合、Multi-scale normal 109,322/109,322 语义记录 bit-exact、输出 key、density、SOR 的冻结验收已精确 |
| 点云着色 | `cpp/apps/surface_panorama_colorizer.cpp`、`cpp/src/pointcloud_coloring.cpp` | `cpp/include/navvis_recon/pointcloud_coloring.hpp`、`src/navvis_recon/pointcloud_coloring.py` | `11.41.12` 冻结同几何完整自动曝光：有效 OVS、136 Gamma 和 2,857,623 点 PLY body `EXACT`；固定 depth wall 14.79 s。大数据完整自动历史性能仍慢于 NVS，瓶颈为 PCT ray-depth |
| 图像后处理 | `cpp/apps/ocam_panorama_pipeline.cpp`、`cpp/src/image_postprocessing.cpp` | `cpp/include/navvis_recon/image_postprocessing.hpp`、`src/navvis_recon/image_postprocessing.py` | 直接读取原始 DNG；24/24 JPEG 对官方整文件 byte-exact、解码像素 MAE 0；不再调用 NVS converter，标准 high-quality 路径状态 `EXACT` |
| 全景与点云渲染 | `cpp/apps/ocam_panorama_pipeline.cpp`、`cpp/src/panorama_rendering.cpp` | `cpp/include/navvis_recon/panorama_rendering.hpp`、`src/navvis_recon/panorama_rendering.py` | 标准 G8 sparse-depth→8K filled JPEG 在 34/34 capture 上 byte-exact；Surface→sparse 为 34/34 exact，`00000` Surface→最终 8K JPEG byte-exact；`00017` 17.88 s vs NVS 12.31 s，结果 `EXACT`、性能慢约 1.45× |
| Floor estimator | `src/navvis_recon/floor_estimator.py` | Python 测试 | 原版 802 条 trace 的单层 z/时间范围精确；runner 尚未接线，未验证多层 |
| Mapped-space quality | `cpp/apps/mapped_space_quality.cpp`、`cpp/src/mapped_space_quality.cpp` | `cpp/include/navvis_recon/mapped_space_quality.hpp`、`src/navvis_recon/quality_map.py`、`tools/evaluate_mapped_space_quality.py`、runner | G11 全量 135,658,017 ray / 59,713 voxel 对官方核心四字段及 record 顺序 `EXACT`；官方 estimator 多提交 3 ray，冻结文件余 21 count/5 diversity，且并行 diversity 非确定；runner 已接线/fail-closed |

## 5. 各模块的标准评测差距

### 5.1 SLAM、轨迹和标定

`2026-07-21_11.41.12` 的 2026-08-27 最新优先结论：raw-IMU tracker 已对齐整数纳秒转 double、区间端点插值和 4 秒 gravity fade 的外层 request-time 语义。首 5 秒 98-node 前端 ATE translation mean/max 为 `1.594e-11/5.304e-11 mm`，rotation mean/max 为 `1.558e-13/3.576e-13°`；97/97 局部 ICP 的 source/origin、三级 target 和 normals 均逐位一致。

Stage 1 已按二进制积分顺序完成原生 C++/Ceres 求解。2,658 个 acceleration 和 2,659 个 rotation measurement 的最大绝对差为 `4.44e-16/5.55e-16`；完整后端 final cost 与官方精确值差 `6.17e-9`，gauge-aligned ATE translation mean/max 为 `9.89e-9/1.96e-8 mm`，rotation mean/max 为 `3.34e-10/1.11e-8°`。这是冻结输入/Build ID 下的结果 exact，不是 protobuf byte-exact。

Stage 2 已按安装二进制实际 Lua 配置联合优化重力、IMU 姿态、加速度偏置/尺度和陀螺偏置/尺度。完整 2,660-node 后端隔离 final cost 为 `25,715.8574`，官方 `25,715.00`，差 `0.003334%`；gauge-aligned ATE translation mean/p95/max 为 `0.02574/0.02788/0.03108 mm`，1 s RPE mean/p95 为 `0.003023/0.006671 mm`。该冻结状态隔离 exact 不能外推为自主链 exact；自主 runner 现已串联并达到工程 NEAR，当前严格缺口是自产 loop pair topology 与序列化。详见 `regression/2026-07-21_11.41.12/SLAM_IMU_BACKEND_ALIGNMENT_20260827.md` 和 `regression/2026-07-21_11.07.05/SLAM_FULL_AUTONOMOUS_ALIGNMENT_20260828.md`。

当前完整链：Pandar scan archive → 双雷达 50 ms/58,000-ray 合批 → raw-IMU 旋转与常速平移去畸变 → 0.04 m scan centroid → 三级 float surfel ICP → 两个 active Submap → 0.2 m HybridGrid → 401 eligible / FixedRatioSampler 41 searches → FCS/ICP → 1,616 个相邻节点 9D IMU 因子 → 稀疏图优化。

`2026-02-08_07.33.20` 全量回归真值：

| 项目 | 当前结果 | 对官方差距 |
|---|---:|---:|
| 原始前端节点 | 1,617 | exact |
| Submap / memberships | 6 / 2,581 | 生命周期、时间戳、成员逐项 exact |
| 局部前端 pose | translation mean/p95/max `0.497/0.732/4.123 mm`；rotation mean/p95 `0.00444/0.01399°` | 亚毫米主体；node 1150 为单个 4.12 mm 切图边界离群点 |
| sampled / accepted loops | 41 / 15 | pair 集 TP/FP/FN=`15/0/0` |
| 冻结点云回环测量 | translation mean/p95/max `0.318/0.637/0.688 mm`；rotation mean/p95 `0.00181/0.00418°` | 结果级对齐 |
| 生成点云回环测量 | translation mean/p95/max `0.817/1.218/1.243 mm`；rotation mean/p95 `0.00653/0.01348°` | 集合 exact，测量亚毫米到 1.24 mm |
| 完整生成链最终 node ATE | translation mean/p95/max `0.421/0.802/0.869 mm`；rotation mean/p95 `0.00213/0.00391°` | 亚毫米结果级对齐 |
| 完整生成链 1 s RPE | translation mean/p95 `0.072/0.191 mm`；rotation mean/p95 `0.00104/0.00247°` | 亚毫米/毫度级 |
| 冻结 loop 隔离 ATE | translation mean/p95 `0.398/0.755 mm` | 说明剩余误差主要在 node↔Submap 局部测量，不在 loop pair |

9D backend 已按二进制探针实现 15 参数块 Exact 因子结构，8,480 条 IMU 样本组成 1,616 个因子。使用官方 topology/loop 隔离时 node ATE translation mean/p95 为 `0.139/0.337 mm`、rotation mean/p95 `0.000775/0.001348°`。完整生成链求解 42 次迭代成功，最终 cost `47,446.21`。

HybridGrid 高分在 `(2,1120)` 官方 FCS pose 上精确得到 `0.5435789227485657`，标准 `0.55` 门限拒绝。官方 Z1 `SAVE_ALL` 全重放确认 41 个候选的失败分布为 NONE/THRESH_ROT/HIGH_RES/ICP_STABILITY=`15/16/6/4`；FCS 以 `max(search_region.scaling)<0.1 m` 跳过 17 个。详见 `code/slam_probes/HYBRID_GRID_LOOP_EVIDENCE.md` 与 `ICP_STABILITY_EVIDENCE.md`。

标准产物：`regression/2026-02-08_07.33.20/slam_raw/frontend_1617_complete_hybrid.json`、`frontend_1617_complete_hybrid_state.npz`、`loop_frontend_frozen_exact.json`、`complete_slam_generated_loops.json`、`complete_slam_frozen_loops_isolation.json`。不得把当前结果称为 byte-exact：生成 HybridGrid 六图总体素规模差约 0.038%，局部 surfel/5,000 点 adaptive filter 仍有细微集合差，完整前端耗时 `752.39 s`；完整后端 warm run `148.94 s`，其中因子构建 `140.60 s`。

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
| 全量 raw 接受扫描输入 | 21,609 scans | 21,609 scans | 逐扫描点数差异 0 |
| 全量 raw 有效 Fringe | 10,956,106 removed | 10,956,106 removed | 20,414 scans 逐扫描差异 0 |
| 全量 raw 独立 Foot | 4,275,867 removed | 4,275,867 removed | 10,207 scans 逐扫描差异 0 |
| 全量 raw `Fringe→Foot` | 3,901,617 removed | 3,901,617 removed | 输入点数和删除数逐扫描差异 0 |

当前全量端点交付为约 24.893M 个 1 cm endpoint voxel；这是表面过滤前阶段，不能与原版最终 12.061M 或 159.147M 表面点数直接比较。

2026-08-24 的无损性能修改按安装二进制的 scan 批量连续数组/GOMP 证据，把 packet decode、
逐点轨迹投影、有序法向和 multilayer fringe 改成固定索引并行；方位角每点只算一次；法向小邻域
使用栈上定长缓冲；最终发射复用第一次 world/origin 投影；raytile 用整数 tile key 分桶。完整
runner 从旧净室 `139.2566 s` 降到 `27.4193 s`（`5.08×`），NVS CloudBuilder 为
`32.6554 s`，当前快约 `16.0%`。完整 528,916 包、3,174 scans、83 个 raytile 对修改前净室
逐文件 byte-exact，下游完整 surface SHA 仍为
`0dcdc2729230d0ec770fb8ccbb6d16dbee1a71e8c156c5fed847d672ae3f4207`。2026-08-26 的
精确时间回归已把上述历史 `-978/-15,705` 分类差消除；二进制确认端点使用
`trunc(double(scan_us)+double(relative)*1e6)`，origin 使用
`scan_us+trunc(float(relative*1e6F))`，分别查询后官方七字段记录也完全同集。完整原始 bag
恢复后，2026-08-27 又确认 Fringe 有序点使用逆位姿 Matrix4f 路径，而 PlaneFilter Region
使用逆位姿四元数直接乘点；两条数学等价的路径不能合并。删除 Plane PCA 特征向量的冗余
归一化，并按运行时 RegionTransformed 矩阵和包含边界的圆柱比较实现后，全量逐扫描结果归零
差异。当前状态为 `FULL_RAW_DECISIONS_EXACT / OBSERVABLE_FIELDS_EXACT`。这不是不同 PLY
header、扩展字段和并行顺序下的整文件 byte-exact。最新结果和复现命令见
`regression/2026-07-21_11.07.05/PANDAR_CLOUDBUILDER_FULL_EXACT_ALIGNMENT_20260827.md`；
旧时间路径证据仍见 `regression/2026-02-08_07.33.20/PANDAR_EXACT_ALIGNMENT_20260826.md`，
性能细节见 `CLOUDBUILDER_ACCELERATION_20260824.md`。

### 5.3 全局自由空间叶集合

`pandar_shard_surface_filter.cpp` 保留跨分片原始射线历史，按 2 cm 紧凑占据叶统计 hit 和 traversal。最新同输入验收不使用最近邻伪配对，直接按同一世界锚点整数 key 比较：

| 项目 | 原版 | 当前 C++ | 差距 |
|---|---:|---:|---:|
| occupied keys | 787,895 | 787,895 | 完全一致 |
| hit counts | 787,895 | 787,895 | 全部一致 |
| intersection counts | 787,895 | 787,895 | 全部一致 |
| deleted leaves | 3,893 | 3,893 | TP 3,893，FP/FN 0/0 |
| float32 centroid 逐位一致 | 787,895 | 741,503 | 46,392 个有算术顺序差异 |

质心差异均值 `2.107e-8 m`、p95 `5.96e-8 m`、最大 `2.739e-6 m`，不改变任何 hit、intersection 或 delete 决策。

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

最新优先结论：`2026-07-21_11.41.12` 的确定性单线程官方分支已完成完整自动曝光和最终着色
Exact 对齐。44,544 个曝光点 XYZ/法向/顺序 exact，168,393 条有效曝光观测、32,274 个 joint
block、138,899 个 residual、136 个 GammaModel、12,504,451 条有效最终观测、direct mask、
KNN 分区及 2,857,623 个最终 PLY 点记录均 exact。RGB changed points/MAE/max delta 为
`0 / 0 / 0`。OVS 整文件仅 `quality=0` padding 不同，PLY 整文件仅 321 字节 header 不同；
可观察计算结果不存在差距。见
`regression/2026-07-21_11.41.12/COLOR_FULL_AUTOMATIC_EXACT_ALIGNMENT_20260827.md`。

以下 V10、2026-08-24 和 2026-08-25 数字只保留为历史收敛记录；凡与上段冻结同几何结果冲突，
以上段及 2026-08-27 报告为准。

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

生产链实现 LibRaw 解码、白平衡/暗角、`0/1.5/3 EV` 三曝光、Mertens `1/1/1`、自适应 NLM、`sigma=3` 非锐化和 JPEG 95。二进制动态捕获和反汇编恢复了原版 NLM 公式：`gain=max(0,4.5+6*log2(ISO/100))`，`noise=gain+6*1.5`，`h=1.75*(7.5/1.75)^(noise/27)`，模板/搜索窗口为 `7/17`。程序保留未旋转的传感器栅格并写原版 TIFF/EXIF APP1。

2026-08-27 的最新回归修正了两个最终舍入/封装差异：contrast 中点必须使用全通道
`cv::Scalar::all(0.5)`，不能使用隐式单值 Scalar；APP1 必须从原始 DNG 复制 Make、Model、
Orientation、DateTime 和 10 个 Photo EXIF 标签并按原版 little-endian TIFF 布局序列化。修复后
三曝光、Mertens、normalize/threshold、tone、NLM、Gaussian、unsharp 和 JPEG encoder 输入均
逐位一致，DQT/DHT/SOS entropy 也完全一致。

`2026-07-21_11.07.05` 前 6 个 capture 的 24 张 JPEG 对官方 `24/24` 整文件 SHA-256
一致，解码像素 MAE `0/255`、通道逐值一致 `100%`、最大差 `0`。净室已直接读取原始 DNG，
runner 和 benchmark 不再调用 `nv_dng-converter`；冻结 worker wall `63.68 s`、峰值 RSS
`6,611,428 KiB`，旧桥接同机为 `64.01 s`。该 `EXACT` 结论限定于冻结 Build ID 的标准
high-quality 路径且未提供 `--blur-regions`；fast/plain、隐私区域模糊和全景不包含在此结论。
详见 `regression/2026-07-21_11.07.05/IMAGE_POSTPROCESSING_EXACT_ALIGNMENT_20260827.md`。

### 5.8 全景与点云渲染

`ocam_panorama_pipeline.cpp` 和 `panorama_rendering.cpp` 的标准 G8 路径已经从冻结的
processed camera JPEG 与 `pano_depth_sparse.png` 一直对齐到 8192×4096 filled JPEG。
`2026-07-21_11.41.12` 的 `00000..00033` 全部 34 张最终文件 byte-exact，解码像素 MAE 0、
最大差 0、SSIM 1。旧六张 `NEAR`、7-band 和 1024 MAE 数据均已被此次全量验收取代。

- 深度 objective 为四层 2×2 valid average 金字塔；PCG 的 `1e-15` 阈值作用于未预条件化
  `r.dot(r)`。raw double 只达到机器精度数值等价，不称 bit-exact；float/毫米量化及所有可观察
  下游输出 exact。
- 标准 8K 是双分辨率链：2K 连续 float depth 负责曝光和 GraphCut；8K 使用 native dense
  float→`1000.0F`→uint16 毫米截断→double `0.001`→float→线性 resize，并复用 2K gain/seam。
- 世界射线使用像素中心双精度三角函数并一次转换到 float；2K/8K 四路 projection mask 与
  exposure-compensated 图像均 exact。
- GraphCut 顺序为 `0-1、1-2、2-3、0-3`，第三/第四对移位为 1228/819；camera 2 的 MIRA
  窗口从列 1025 开始。四个 seam mask、seam-prepared 图像均 exact。
- 主混合为十层 float32 circular multiband，不是旧文档中的 7-band。floor mask 来自四个原生
  8K projection mask 的 OR；no-floor q95 JPEG 必须写出再读回，随后运行 wrapped
  PyramidInpainting，最终 q95 optimize=1。BinaryMask、no-floor 和 final JPEG 均 byte-exact。
- operator mask 标准链 34/34 exact；`--no-mask-pano` 受控 `00000` 也 exact。生产程序不调用
  NavVis renderer，安装二进制只用于离线反汇编和只读动态探针。
- `00017` clean wall 17.88 s、峰值 3,713,696 KiB；NavVis 为 12.31 s、2,504,544 KiB。
  结果已完全对齐，当前性能仍慢约 `1.45×`、内存约 `1.48×`。
- `--surface-cloud` 上游使用 PCL `OctreePointCloudSearch<PointXYZ>`、0.05 m voxel、double
  像素中心射线、float endpoint/origin 距离和 float 毫米截断。34/34 个 capture 的 1024×512
  sparse depth exact；`00000` 的 Surface→PCG→最终 8K JPEG byte-exact。该结论尚未宣称
  34 个 capture 的 Surface→8K final 全量实跑，只宣称 34 个 sparse 和一个正式端到端样本。

完整证据、命令和逐阶段表见
`regression/2026-07-21_11.41.12/PANORAMA_FULL_EXACT_ALIGNMENT_20260827.md`；机器结果见同目录
`panorama_full_exact_alignment_20260827.json`。守门命令为
`scripts/verify_panorama_full_exact.sh`，预期 `captures=34 byte_exact=34 mismatches=0`。
Surface 上游证据和精确浮点语义见
`regression/2026-07-21_11.41.12/PANORAMA_SURFACE_TO_FINAL_BYTE_EXACT_20260827.md`。

### 5.9 Floor estimator 与 mapped-space quality

- `floor_estimator.py` 已按二进制证据实现 0.1 m 高度 bin、2.1–4.0 m 楼层范围、3.0 m 标准层高、0.03 m 容差，以及 merge/split/refine 流程。2026-08-24 用原版 802 条 `artifacts/trace.csv` 同输入验收时，双方均为 1 层，z_min/z_max 和纳秒时间范围全部精确一致；这只覆盖单层数据。runner 当前仍调用简化 `floor_summary`，其 z_min/z_max 差 -0.123/+0.205 mm，时间边界差 -9.87/+54.67 ms，且序列化不同；必须接通 `refined_floor_estimator` 并验证多层数据后才能称生产链完全对齐。
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
`regression/2026-02-08_07.33.20/MODULE_PERFORMANCE_BINARY_ANALYSIS_20260824.md` 与同名 JSON。
历史性能结论是编译选项不是主因：surface 受控仍慢 `6.14×`，核心差异是原版全局 CompactOctree、
连续 Trace 与 NanoFLANN/PCL KD-tree，而当前仍有 shard-local hash/均匀网格；着色受控旧路径慢
`5.03×`，本轮已修正曝光 cache 反复解码和完整外部 depth 下的重复 PCT 渲染。小数据 OVS、
Gamma、mask、最终 PLY 验收保持 byte-exact；完整 color 模块现已有 2026-08-27 Exact 重跑，
但其 14.79 s 固定-depth 计时仍不得直接外推为包含 PCT 的全量收益。旧同步源码 SHA-256 为
`0d9f4936b116ab1464d4203fe4a90bf1717bcabcb4332104e17efa80a71951a4`。

### 5.11 2026-08-24 重测与无损加速

最新性能真值见
`regression/2026-02-08_07.33.20/PERFORMANCE_REBENCHMARK_ACCELERATION_20260824.md` 和同名 JSON。
Adaptive SOR 的法向空间索引 cell 从 `0.05` 改为 `0.01`，该调用显式扩展到 `3.2` 以保持同一
distance/index Top-11；完整 surface `443.771→220.491 s`，输出 SHA 仍为
`0dcdc2729230d0ec770fb8ccbb6d16dbee1a71e8c156c5fed847d672ae3f4207`，相对 NVS
`72.297 s` 仍慢 `3.05×`。完整冻结-Gamma 着色因跳过已有 depth 下的重复 PCT 渲染，从
`95.13→56.06 s`，输出 SHA 仍为
`ac1930ae74fc5e4b45c747c14ca9422b6336bb5b7a0582ea8475cde018b20057`，相对 NVS
仍慢 `2.97×`。连续 flat index、单次最大 normal 查询、增量 shell 和 SOR cell `0.005/0.020`
均已实测否决，不要重复采用。资源 `build-release/navvis_recon_shard_surface_filter` SHA 为
`6e63d499c3242f22aa75932cc1405c073502960afe559ce404e2475710b5b2a8`。

### 5.12 2026-08-25 全模块当前真值

最新完整实跑见 `regression/2026-02-08_07.33.20/MODULE_REBENCHMARK_20260825.md` 和
`module_rebenchmark_20260825.json`。后续任务必须优先使用这些数字：

| 模块 | 当前 | NVS | 状态 |
|---|---:|---:|---|
| CloudBuilder runner | 24.324 s | 32.655 s | 结果已升级为全量 raw Fringe/Foot 逐扫描 exact 和七字段 exact；该旧输出速度早于 origin/双坐标修复，当前完整输出 wall 仍待同口径复测；2026-08-27 全量统计模式为 77.09–89.86 s，不与输出 wall 横比 |
| 完整 surface | 75.85 s | 72.297 s | `NEAR`，慢 4.9%；12,076,978 点，SHA 仍为 `0dcdc2…4207` |
| 同 GT 几何点云着色 | 最新固定-depth 14.79 s；旧完整 49.39 s | 旧完整 18.903 s | 最新冻结同几何结果 `EXACT`、RGB MAE 0；14.79 s 不含 PCT，不能与旧完整时间横比 |
| 图像后处理 | 57.260 s | 54.895 s | 历史同数据性能基线；结果状态已由 2026-08-27 的另一组 24 张 byte-exact 回归升级为 `EXACT`，当前纯净室 wall 63.68 s，跨数据集不计算比率 |
| 全景渲染（标准单张 `00017`） | 17.88 s | 12.31 s | `EXACT`，最终 JPEG byte-exact；当前慢 1.45×、峰值内存约 1.48× |
| Floor 算法 | 0.03722 s median | 0.03392 s | 单层结果 `EXACT`，速度慢 9.7% |

旧六阶段合计 `279.99 s` 对 NVS `212.20 s`、慢 `1.319×` 的数字包含已经失效的旧全景链，
不再是当前可公平相加的总时间；全景最新公平单张数据以上表 `00017` 为准。完整
SurveyorSLAM 的 Stage 1 冻结后端已达到浮点精度结果 exact，旧完整自主链达到拓扑/回环 exact、最终轨迹亚毫米结果级对齐，但当前净室前端与后端的
分段计时条件不同于 NVS 日志中的 `99.055 s`，因此仍不并入这组六阶段公平合计；录制轨迹融合
的 `0.64 s` 也绝不能冒充完整 SurveyorSLAM。mapped-space quality 现为官方核心 record Exact；
官方 estimator 外层仍有 3 次重复 ray 提交和 parallel diversity 漂移，命令行文件不 byte-exact。
2026-08-27 起
图像链直接读取原始 DNG，`nv_dng-converter` 已从生产 runner 和阶段 benchmark 中移除。

`test_resources/compare_camera_jpegs.py` 按解码像素比较 camera JPEG；
`scripts/benchmark_image_panorama_staged.py` 按 NVS 阶段边界计时。旧六张 8K
`130.61 s`、MAE/SSIM 数字仅保留在历史报告，不得覆盖 2026-08-27 的 34/34 byte-exact 结论。

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

当前预期：C++ CTest `4/4` 通过；Python reconstruction `38/38`、complete-SLAM
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
- 修改着色时必须同时守门 exposure cloud、有效 exposure/final OVS、自动 Gamma、direct/KNN
  分区和最终 PLY；固定官方中间态只能作为根因隔离，不能代替净室自动链验收。
- 着色回归若使用外部 `--depth-map-input-dir`，必须证明全部文件按真实 view ID 编号并逐图记录
  SHA；旧 `nv_pct_captured_current.EA8nlh/depth` 是并行访问顺序，禁止作为标准输入。
- 修改图像/全景时必须标明是当前探针还是旧 299 张全量，不能用过时全量掩盖当前回归。
- 对 G10/VLP16 只能声称功能兼容和内部一致性；没有原版 GT。
- 新评测必须记录输入路径、二进制 SHA/Build ID、命令、点数、误差方向和机器可读 JSON。
- “完全一致”必须限定字段、阶段和输入；冻结同几何着色 PLY body 已 exact，但自主 Surface
  完整链的历史验收仍差 262 点，二者不得混称。

## 8. 会话清理后的着色继续入口

资源包根目录为 `/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment`。`code/` 只放源码；所有新 build、临时 OVS、Gamma、深度、日志和点云必须写入 `work/color_alignment/`，正式指标与结论写入 `regression/<dataset>/`。不得在 `code/` 新建 build 或结果目录，也不要删除其中迁移前遗留的 `build-release/bin/lib`，除非用户明确要求清理。

当前优先入口：

- 源码：`code/cpp/apps/surface_panorama_colorizer.cpp` 和 `code/cpp/CMakeLists.txt`；
- 最新报告：`regression/2026-07-21_11.41.12/COLOR_FULL_AUTOMATIC_EXACT_ALIGNMENT_20260827.md`；
- 机器指标：`regression/2026-07-21_11.41.12/color_full_automatic_exact_alignment_20260827.json`；
- 最新净室无颜色输入完整结果：
  `work/color_alignment/official_single_exact_20260827/clean_full_exact_uncolored_input/`；
- 无颜色 Surface 输入与转换工具：
  `work/color_alignment/official_single_exact_20260827/uncolored_surface_input.ply`、
  `scripts/strip_binary_ply_color.py`；
- 同次官方 OVS/Gamma/PLY：`work/color_alignment/official_single_exact_20260827/`；
- 累计链报告：`regression/2026-07-21_11.41.12/PIPELINE_MATRIX_4_34_234_1234_20260825.md`；
- 累计链机器指标：`regression/2026-07-21_11.41.12/pipeline_matrix_4_34_234_1234_20260825.json`；
- 累计链全部 PLY、深度、OVS、Gamma、全景和计时：`work/pipeline_matrix_2026-07-21_11.41.12/`；
- 官方小集结果：`work/color_alignment/nvs_1cm/2026-07-21_11.41.12/`；
- 原版 OVS/Gamma、二进制探针与反汇编：`work/color_alignment/original_capture_1cm/`；
- 净室端到端结果：`work/color_alignment/clean_1cm/end_to_end_ceres22/`；
- Ceres 2.2 构建：`work/color_alignment/build-ceres22/`；
- Ceres 2.2 安装前缀：`work/color_alignment/toolchain/ceres-2.2.0-install/`。

标准着色处理顺序：

1. 用 Ceres 2.2.x 配置 Release build，构建输出只能放 `work/color_alignment/build-*`；运行 CTest。
2. 固定官方同几何 PLY、按真实 view ID 的深度、相机 mask、相机信息和全景输入，并记录路径、点数、SHA 和二进制 Build ID。最终验收必须使用去掉 RGB/alpha 的 8-float Surface PLY 再跑一次，确认 exposure OVS、Gamma、final OVS、mask 和最终 PLY 均不依赖输入颜色。
3. 分别生成曝光 OVS 和最终 OVS；要求 `quality>0` 的 validity mask 和完整 8 字节 payload
   exact。`quality=0` padding 只记录为非语义字节差，不得伪装成观测差距。
4. 曝光求解使用 LM/CGNR/JACOBI、50 iteration、初始 radius `1e4`、tolerance
   `1e-6/1e-10/1e-8`；严格 exact 使用 1 thread，性能运行可用 32 threads。参数按 residual
   first-use 惰性注册。
5. 确定性 exact 守门使用官方/净室单线程求解；要求 136/136 参数 exact。另行测试并行性能时，
   原版并行求解非确定，不得把不同合法模型误判为回归。
6. 最终结果分三层报告：原版 Gamma oracle、原版 OVS + 净室求解、净室 OVS + 净室求解。每层至少报告 XYZ 是否 bit-exact、RGB MAE/PSNR/exact、max-channel ≤1/≤5/≤20、direct/fallback 数和 wall time。
7. 当前基线为 32,274 joint block、138,899 residual、最终 PLY body exact。后续优化必须保持
   这些守门不退化；优先优化 PCT ray-depth 时间和内存，不得用最终 RGB 反调阈值。

标准累计链回归顺序：

1. `4`：冻结官方 Surface、相机、info 和 panorama，只运行净室着色；要求所有非颜色字段 bit-exact，并报告完整深度+曝光+着色时间。
2. `34`：冻结官方 Surface，运行净室图像/全景和着色；除自动曝光结果外，再使用 `4` 的同一 GammaModel 做隔离，避免把 Ceres 并行非确定误判为模块 3 差距。
3. `234`：只能在记录模块 1 输入来源后运行。若官方 CloudBuilder ray-history 不存在，必须标为“净室模块 1 冻结输入”，不得声称严格隔离模块 2+3+4。
4. `1234`：从 raw Pandar 开始，固定同一官方轨迹，运行完整非 SLAM 链；比较最终点数、五个轨迹 ROI 双向最近邻、1 cm 匹配颜色和各阶段 wall/RSS。
5. 若 `234` 与 `1234` 的 Surface byte-exact，可内容共享模块 3 结果，但模块 4 仍应独立运行；独立曝光结果用分布门限判断，不要求 Gamma 文本或最终 RGB byte-exact。

历史 `2026-02-08_07.33.20` 全量入口仍保留在 `regression/2026-02-08_07.33.20/color_exposure_dynamic_cauchy_20260824/` 和 `test_resources/color_exposure_dynamic_cauchy_20260824/`，用于跨数据集复验，不覆盖本节小集的最新求解器结论。资源包准备迁移时再重新生成并以 `sha256sum -c MANIFEST.sha256` 验证清单；日常中间实验不必反复重写清单。
