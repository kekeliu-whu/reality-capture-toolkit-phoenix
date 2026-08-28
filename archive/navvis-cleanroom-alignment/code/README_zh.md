# NavVis 后处理 C++ 与轨迹侧 SLAM 净室重构

代码、生产入口、评测工具和历史探针的完整目录说明见
[`docs/CODE_LAYOUT_zh.md`](docs/CODE_LAYOUT_zh.md)。正常运行使用
`run_navvis_slam_recon.sh` 和 `run_navvis_recon.sh`，统一回归使用
`scripts/run_all_tests.sh`。

## 结论

这是一个可编译、可运行的 C++17 净室重构，不是 NavVis 厂商仓库中的逐字源码。
`/usr/bin/navvis-postprocessing` 本身是 Nuitka 编译的 Python 调度器，点云、图像和全景的
主要计算实际由 C++ ELF/Cython/OpenCV/PCL 组件完成。本项目保持相同的分层：Python 只负责
ROS bag、参数和流程调度，耗时算法由四个 C++ worker 及 C++ 算法库完成。

项目现已补入独立的 SurveyorSLAM 完整链：双 Pandar 原始扫描、raw IMU 去畸变、三级
point-to-plane scan matching、两个 active 的 0.2 m Submap、HybridGrid/FastCorrelative/ICP
回环，以及 15 参数块 Exact 9D IMU 稀疏图优化。`2026-02-08_07.33.20` 全量回归的
1,617 nodes、6 submaps、2,581 memberships 和 15 loops 与冻结官方结果完全一致；最终节点
ATE translation mean/p95/max 为 `0.421/0.802/0.869 mm`，rotation mean/p95 为
`0.00213/0.00391 deg`。这是亚毫米结果级对齐，不是点集顺序、protobuf 或浮点逐位复刻。
完整报告和复现命令见
`../regression/2026-02-08_07.33.20/slam_raw/COMPLETE_SLAM_ALIGNMENT_20260825.md`。

## G11-0109 点云构建逐点对齐（2026-02-08）

`navvis_recon_pandar` 现在直接复现安装版 `cloud_builder` 的 G11 扫描级路径：自动识别
820 字节 Pandar XTM 包，解码 6×32 个距离/强度/ring/方位角和逐束时间，使用指定
`trajectory.bag` 插值每个回波时刻，再合成 `sensor_frame.xml` 中的水平/垂直雷达外参。
轨迹只读，整个过程不运行 SLAM。

垂直脚部阶段不是简单圆柱裁剪。二进制 RTTI 和运行时断点确认它是
`PlaneFilter<PointXYZNormalITR>` 包含 `RegionFilter`：内层在统一扫描时刻的垂直雷达
坐标中执行“旋转变换、半径 0.5 m 圆柱、z≤0”；外层在逐点世界坐标中运行 PCL 1.12
平面 RANSAC（固定种子 12345、最多 50 次、1 cm 内点阈值），并剔除候选区内偏离主平面
超过 2 cm 的点。短窗 `1770532442.4–1770532442.5` 的原版与重构结果如下：

| 指标 | 原版 | C++ 重构 |
|---|---:|---:|
| 脚部过滤前回波 | 51,480 | 51,480 |
| 脚部候选 | 4,238 | 4,238 |
| 脚部剔除 | 1,808 | 1,808 |
| 脚部过滤后回波 | 49,672 | 49,672 |

剔除/保留标签逐点完全相同；保留点世界坐标平均差 1.53 µm、最大差 0.030 mm，强度逐点
完全相同。完整 `2026-02-08_07.33.20` 数据处理 528,916 个 XTM 包、3,174 个完整扫描，
最新回归耗时 186 秒：

| 扫描级阶段 | 原版 | C++ 重构 | 差异 |
|---|---:|---:|---:|
| 设备/距离/强度/运动门限后 | 83,822,437 | 83,821,459 | -978（-0.0012%） |
| 多层边缘过滤后 | 82,199,764 | 82,184,059 | -15,705（-0.0191%） |
| C++ 脚部剔除 | — | 397,747 | 每扫描执行同一 RANSAC |
| 1 cm 端点体素 | — | 24,893,349 | 本节交付阶段 |

全量 1 cm 端点输出位于
`/media/cybergeo/12T/CSSJ/datasets_proc_recon_cpp_aligned/2026-02-08_07.33.20/pointcloud.ply`。
该文件是本节要求的扫描过滤和端点体素阶段，不应和原版经过自由空间表面过滤、法线/着色后的
12,061,091 点最终 PLY 直接比较点数；后者对应 `--aligned-standard` 的后续阶段。

## 2026-02-08 完整数据回归与表面并行优化

表面 worker 现支持三级确定性并行：10 m shard 预处理、不同 5 m tile，以及 tile 内的 OpenMP
点循环。shard/tile 任务可以按负载排序，但结果始终写回固定槽并按原输入顺序归并；halo 也只装载
一次且保留旧有序语义。kernel 另外消除了 ray leaf 拷贝、重复支持邻域收集和 Adaptive SOR 的第二次
KNN。冻结验收全部保持一致；小规模测试的每轮优化和完整回归都比较输出 SHA，而不只比较点数。

`2026-02-08_07.33.20` 的完整 81,785,598-ray、312-tile 运行使用 8 个预处理线程、8 tile ×
4 点线程。单次 halo、连续 candidate→leaf 映射、1-byte 删除 mask 和每线程邻域 scratch 把阶段耗时
由 1,949.74 s 先降至 587.113 s，再降至 443.771 s（累计 4.39×）；wall 444.52 s，峰值 RSS
11,762,644 KiB。输出仍为 12,076,978 点、386,463,639 bytes，SHA-256 精确保持
`0dcdc2729230d0ec770fb8ccbb6d16dbee1a71e8c156c5fed847d672ae3f4207`。相对 NavVis 的
12,061,091 点多 15,887（+0.1317%）；五个轨迹区域双向最近邻平均 p50 约 0.251 mm、平均
p95 约 4.80 mm，约 98.3%/99.64% 在 1/2 cm 内。

原版同数据 CloudProcessor 为 72.297 s，当前仍约慢 6.14×。Release 已启用 `-O3 -DNDEBUG`；
`-march=native -flto` 隔离实验仅快约 6–9%，主要差距是原版全局 CompactOctree 与当前自定义
空间索引/数据布局；二进制符号还确认表面邻域使用 NanoFLANN/PCL KD-tree，而不是当前均匀哈希
网格。差距不是漏开优化开关。完整性能证据位于
`/media/cybergeo/12T/CSSJ/datasets_proc_performance_alignment_20260823/PERFORMANCE_ALIGNMENT_20260824.md`。

同次完整着色输出 12,076,978 点，1 cm 匹配点的五区平均 RGB MAE 为 26.85/255，候选偏暗；
6 张 8K 全景在 1024×512 评测的平均 MAE 为 15.46/255、平均 SSIM 为 0.698。这两个模块仍
没有全量结果级完全对齐。结果和机器可读报告位于
`/media/cybergeo/12T/CSSJ/datasets_proc_regression_20260823/2026-02-08_07.33.20/`。

8K capture 在 125 GiB 回归主机上最多六路并发，单 capture 内 GraphCut 顺序不变。worker 取消
只读 8K 深拷贝并并行独立 nadir 循环后，同一数值链的六张 8K 回归由串行 560.691 s、四路
197.073 s 降至六路 130.722 s（相对串行 4.29×），6 张全景和 24 张相机 JPEG 全部逐文件 SHA
一致。原版图像后处理加全景渲染为 88.307 s，当前仍约慢 1.48×，因此这里只能称无损加速，不能
称性能或像素结果已经与原版完全对齐。限制每 worker 的 TBB 线程数没有收益，生产配置保持 OpenCV 默认。

## 完整数据的既有基线（表面参数修复前）

原命令已经在独立目录完整执行成功，耗时 3:16:25，作为参考真值；本项目此前也处理完整的
33+33 个激光 bag 和 299 个采集点。下表的净室结果来自旧 0.0131 m 输出格、旧射线阈值，
本轮已删除这些点数拟合参数，因此只保留作历史基线，不能代表当前代码的新全量结果。

| 指标 | 原版 | C++ 重构 | 差异 |
|---|---:|---:|---:|
| 最终表面点 | 159,147,139 | 151,301,323 | -4.93% |
| 包围盒 X | -42.590…82.880 m | -42.596…82.871 m | 边界最大约 9 mm |
| 包围盒 Y | -42.707…56.434 m | -42.662…56.417 m | 最小 Y 尚差 4.5 cm |
| 包围盒 Z | -8.228…14.102 m | -8.228…14.093 m | 边界最大约 8 mm |
| 单位法线比例 | 100% | 100% | 一致 |
| 全景数量/尺寸 | 299 / 8192×4096 | 299 / 8192×4096 | 一致 |

五个沿轨迹分布的局部区域中，原版表面到重构表面的 2 cm 覆盖率为
99.29%、99.21%、99.65%、99.00%、96.00%，中位最近距离为 0.58–0.81 cm。
反向中位距离为 0.39、0.42、0.44、0.41、1.01 cm，反向 2 cm 命中率为
94.31%、93.78%、91.73%、86.68%、76.31%。相对没有射线历史的 v4，五区平均反向
2 cm 命中率提高 4.60 个百分点，平均距离、p90 和 p99 分别下降 42.45%、23.26% 和
59.61%；原版表面覆盖率平均只下降 0.13 个百分点。

全部 299 张全景在 1024×512 比较尺度上的平均 MAE 为 22.14/255，中位 MAE 21.77，
灰度 SSIM 平均 0.534；相位相关的中位绝对偏移为水平 0.47 px、垂直 0.74 px。
所有原生输出尺寸都相同。首路自有 DNG 图像后处理与原版相机 JPEG 的 MAE 为 12.65/255，
相位偏移约 0.004 px。

旧版最终 151,301,323 点全部完成着色，其中 89.09% 使用深度检验后的四路相机直接融合，
10.91% 使用全景回退。RGB 的 11 个全局分位点逐通道与原版最大差 1 个灰阶；五个区域内
1 cm 几何匹配点的 RGB MAE 为 22.41–46.86/255。因此颜色的总体分布已高度接近，局部
对应颜色仍不是逐点相同。

### 24 鱼眼净室着色回归

在较小且可逐点回归的 `2026-02-08_07.33.20`（6 个采集点、24 个鱼眼视图）上，着色后端
已改为 684×456 PCT 深度图、G11 掩膜、二进制一致的点权重图、逐点 Top-5 排名、
GammaModel 响应和精确五近邻几何加权外推。它在运行时
不执行 `nv_colorcloud`；安装的二进制只用于离线静态分析和黑盒消融。

在 122,701 点裁剪回归中，直接/外推掩膜 122,701/122,701 一致（119,714 个直接点、
2,987 个外推点），几何、强度、法线和曲率全部逐位一致。高斯、边界、mask 和最终点权重
六个中间图也全部逐 float 位一致。加载同次原版 GammaModel 时，RGB MAE 为 0.002700，
99.40098% 的点三通道完全一致，最大通道误差为 1；净室自行求解 GammaModel 的完整路径
RGB MAE 为 0.05794，84.1778% 的点完全一致，所有通道误差仍不超过 1。最新 12,076,978 点
全量回归在非同序几何的 1 cm 最近邻匹配上五区平均 RGB MAE 为 26.85/255；这说明同输入裁剪
已经高度对齐，但完整数据的自动曝光/图像输入链仍有明显偏暗问题。

量化数据见同级交付中的 `navvis_alignment_reference/` 与最终对齐报告。上述结果说明主表面、
坐标链、全景方向和输出规模已接近，但不代表像素级或二进制级复刻。

## G10/VLP16 实测

`2023-05-15_10.18.42`（G10-512、C7.0）也已完成全量测试。程序自动识别 1206 字节
Velodyne VLP16 包，处理 104,340 个激光包、69.25 秒已有 SLAM 轨迹和 4 个采集点。推荐的
保守射线阈值为至少 6 次穿越且穿越/端点命中比达到 3.0；输出 3,538,026 个全彩表面点，
四张全景均为 8192×4096。默认值会依据 `dataset.json` 中的 G10/G11 设备序列号自动选择，
显式 CLI 参数仍具有最高优先级。

这台 G10-512 的原厂处理许可已经过期，因此原程序在 LicenseCheck 阶段退出，无法生成同数据
原版点云作为真值。这里报告的是解码完整性、射线历史 A/B、几何合法性和视觉检查结果，不能把
它表述成与原厂结果完成了逐点对齐。

## C++ 实现

| 程序/模块 | 功能 |
|---|---|
| `navvis_recon_pandar` | 自动识别 PandarXTM 820 字节包或 VLP16 1206 字节包、逐点时间插值去畸变、设备过滤、1 cm 端点体素、真实射线原点与 10 m 分片 |
| `navvis_recon_shard_surface_filter` | 端点/原点簇合并、稀疏 3D DDA 自由空间射线雕刻、双偏移密度网格、10 cm 多尺度 PCA 法线/曲率、切平面投影、自适应高密度支持度、输出体素化 |
| `navvis_recon_ocam_panorama` | 全分辨率 DNG/LibRaw、OCam 投影、曝光补偿、GraphCut 接缝、7 层多频段融合、色调与盲区处理、8K JPEG |
| `navvis_recon_surface_colorizer` | 24 张 684×456 PCT 鱼眼深度图、G11 掩膜/暗角、二进制一致权重、逐点 Top-5 鲁棒融合、GammaModel 响应、精确五近邻几何加权外推 |
| `cpp/src/*.cpp` | 点云构建、表面过滤、点云着色、图像后处理、全景/点云渲染的独立算法库 |
| `src/navvis_recon/surveyor_frontend.py` | raw-IMU 去畸变、point-to-plane 雷达前端、0.2 m 重叠 Submap、HybridGrid/FastCorrelative/ICP 回环 |
| `src/navvis_recon/surveyor_slam.py` | 原版 protobuf、中间图拓扑、15 参数块 Exact 9D IMU 稀疏图优化 |
| `src/navvis_recon/slam_reconstruction.py` | 录制全局 SLAM/局部里程计融合、轨迹插值、ATE 与固定间隔 RPE 评测 |
| `src/navvis_recon/floor_estimator.py` | 顺序楼层状态机、tiny/adjacent merge、double-floor split 与官方 JSON 序列化 |
| `tools/evaluate_floor_alignment.py` | 扫描 reference trace/floors 对并执行逐字段 fail-closed Floor 守门 |
| `tools/evaluate_slam_trajectory.py` | 与原版 `trajectory.bag` 做只读轨迹误差报告 |

最终 PLY 使用与参考结果相同的 36 字节记录顺序：
`x y z, red green blue alpha, intensity, nx ny nz, curvature`。

## 当前 Floor、图像、全景和 SLAM 对齐状态

- Floor 已在 220 组同输入、1,796,815 条 trace、615 个 floor 上达到 220/220 全字段 exact；
  runner 的两组 G11 trace 分别 5,374/17,878 行 byte-exact，floors 为 1/5 层 exact，输出位于
  `artifacts/trace.csv` 和 `artifacts/floors.json`。完整报告见
  `FLOOR_FULL_ALIGNMENT_20260828.md`。
- 标准 high-quality 相机后处理已直接读取原始 DNG；三曝光 Mertens、自适应 NLM、锐化、
  JPEG entropy 和 TIFF/EXIF APP1 均按二进制路径复现。冻结数据前 6 个 capture 的 24 张
  JPEG 对官方 `24/24` 整文件 byte-exact，解码像素 MAE 为 `0/255`。该结论不包含
  `--blur-regions`、fast/plain preset 或全景拼接。
- 原版顺序 GraphCut 的四张最终 mask IoU 为 `99.10%/99.84%/99.52%/98.41%`；同一原版
  projected input 的 2K MAE 为 `4.4414/255`。8K nadir 缺口 MAE 为 `2.885/255`、整图
  `0.4631/255`。剩余差距集中在 depth/operator mask 生成、精确 multiband 舍入和少量
  nadir 纹理。
- `2026-02-08_07.33.20` 的完整原始前端得到 1,617 nodes、6 submaps、2,581 memberships，
  lifecycle、成员与时间戳逐项 exact；401 eligible pair 产生 41 次搜索，15 条 accepted pair
  集 TP/FP/FN=`15/0/0`。完整生成链最终节点 ATE translation mean/p95/max 为
  `0.421/0.802/0.869 mm`，rotation mean/p95 为 `0.00213/0.00391 deg`。这是结果级对齐，
  生成 HybridGrid 和 adaptive-filter 点集仍非逐位一致。

## 构建与运行

依赖 CMake、C++17、Eigen3、OpenCV 4、OpenMP、LibRaw，以及 Python 的 ROS `rosbag`、
NumPy/SciPy。录制中的 tiled lossless-JPEG DNG 由 LibRaw 0.22 和随安装环境提供的 DNG SDK
直接解码，不再调用 `nv_dng-converter`；RAW 解码、图像处理、EXIF 序列化和拼接都在本项目
C++ 中完成。
为兼容只有 LibRaw 运行库、没有 `libraw-dev` 的部署机，`cpp/third_party/libraw` 随附了
LibRaw 0.20 的公共头文件及其 LGPL/CDDL 许可证；CMake 会优先使用系统开发包。

```bash
cmake -S cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp -j
ctest --test-dir build-cpp --output-on-failure
```

对应当前数据的完整替代命令：

```bash
./run_navvis_slam_recon.sh \
  /media/cybergeo/12T/CSSJ/datasets_rec/2026-07-21_11.07.05 \
  --work-dir=/path/to/slam-work

./run_navvis_recon.sh \
  --bagplayer-args=--quiet \
  --proc-base-dir=/media/cybergeo/12T/CSSJ/datasets_proc \
  --caller=sitemaker \
  /media/cybergeo/12T/CSSJ/datasets_rec/2026-08-10_20.13.41 \
  --trajectory-csv=/path/to/slam-work/optimized_trajectory.csv \
  --aligned-standard \
  --force --res=0.01 --cloud-format=ply --preset=standard \
  --num-threads-panos=32 \
  --log-file=/tmp/navvis_sitemaker_postproc_recon.log
```

着色默认使用 `--color-backend recon`，因此即使本机安装了 `nv_colorcloud` 也仍运行净室 C++。
只有显式指定 `--color-backend original` 才会执行原始程序；`auto` 保留用于受控对比。
在 `2026-02-08_07.33.20` 的 122,701 点同序裁剪上，净室后端自行完成 global exposure、
逐点视图排名和 KNN fill 后，相对原版 RGB MAE 为 0.05794/255，且最大通道误差为 1；使用
同次原版 GammaModel 做隔离回归时 MAE 为 0.002700/255。

`run_navvis_slam_recon.sh` 从 raw 双 Pandar 和 raw IMU 生成优化轨迹；`--trajectory-csv`
把该结果直接交给后处理。也可用 `--trajectory-bag` 显式指定既有优化结果；两者都不提供时，
运行器使用录制的在线全局 SLAM/局部里程计。`--slam-reference-bag` 只做评测，不参与输出求解。
默认全景宽度为原版的 8192；短链路回归使用 `--max-duration=10 --max-panos=1
--pano-width=1024`。

## 真实性和适用边界

- 类结构、算法族、处理顺序和大量常量来自 RTTI、动态符号、字符串、反汇编、调试器和完整
  参考执行；私有自由空间判定、曝光目标函数和部分权重曲线是实测校准后的独立等价实现。
- 当前 `.raytile` 原始分片为每个端点/原点簇保存真实激光原点，表面过滤使用裁剪后的稀疏
  3D DDA 累计穿越与端点命中证据。全量运行剔除 24,933,821 个自由空间冲突候选；阈值是
  依据局部 A/B 实测校准的独立实现，不是厂商私有 octree 的逐指令复刻，仍不能保证逐点一致。
- 全景已按原版四对顺序 GraphCut；mask IoU 为 98.41%–99.84%，同投影输入 2K MAE
  `4.4414/255`。8K nadir 缺口 MAE 为 `2.885/255`，但 depth/operator mask 生成和精确
  multiband 舍入仍有差距。
- SLAM 已包含 raw 双 Pandar 归档、raw-IMU 去畸变、C++ Frontend、HybridGrid、
  FastCorrelative/ICP 自主回环、Stage1 和 Stage2。`2026-07-21_11.07.05` 全量自产 10,933
  nodes/11 Submaps/789 loops，最终 ATE translation mean/p95 为 `0.654/1.343 mm`；回环 pair
  topology 与官方不完全一致，不能称为 byte/topology exact。
- `standard/G11/0.01 m` 有原厂完整结果对齐；`standard/G10/VLP16/0.01 m` 已完成全量功能和
  射线历史 A/B 测试，但没有可用原厂真值；其他设备与 preset 不能套用这里的量化结论。

二进制证据和完整参考执行信息见 `EVIDENCE.md`，运行参数与输出目录见
`runner/README_zh.md`。
