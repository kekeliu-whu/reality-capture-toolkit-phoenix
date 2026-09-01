# rec-v5 射线历史对齐运行器

生产入口、内部 runner 和诊断脚本的边界见 `../docs/CODE_LAYOUT_zh.md`。正常任务不要直接运行
本目录的 capture/probe 脚本。

`navvis_postprocessing_recon.py` 接受原命令的主要参数，并增加 `--trajectory-bag`、
`--trajectory-csv` 和
`--aligned-standard`。点云着色后端由 `--color-backend recon|original|auto` 控制，默认
`recon`，因此不会静默调用 `/opt/NavVis/pointcloud-coloring/bin/nv_colorcloud`。只有显式选择
`original`（或用于受控对比的 `auto`）才会调用原始组件。Python 负责读取 ROS bag、调用 DNG
无损转封装和调度；激光解码、mapped-space quality、表面重建、图像后处理、全景及默认点云
着色均由自有 C++ 执行。

运行器同时支持 G11 的 820 字节 Pandar XTM 包和 G10 的 1206 字节 Velodyne VLP16 包；
包格式由载荷长度和包内标志自动识别。

## 完整命令

```bash
./run_navvis_recon.sh \
  --bagplayer-args=--quiet \
  --proc-base-dir=/media/cybergeo/12T/CSSJ/datasets_proc \
  --caller=sitemaker \
  /media/cybergeo/12T/CSSJ/datasets_rec/2026-08-10_20.13.41 \
  --trajectory-bag=/media/cybergeo/12T/CSSJ/datasets_proc_reference/2026-08-10_20.13.41/artifacts/trajectory.bag \
  --aligned-standard \
  --force --res=0.01 --cloud-format=ply --preset=standard \
  --num-threads-panos=32 --surface-tile-threads=8 \
  --log-file=/tmp/navvis_sitemaker_postproc_recon.log
```

运行器优先读取显式提供的优化后 `trajectory.bag` 或自主 SLAM 生成的
`optimized_trajectory.csv`；否则可融合录制的在线全局 SLAM 与局部里程计生成高频轨迹。
`--slam-reference-bag` 只计算 ATE/RPE，不参与或修改输出。完整自主 SLAM 使用：

```bash
./run_navvis_slam_recon.sh /path/to/rec-v4 \
  --work-dir /path/to/slam-work

./run_navvis_recon.sh /path/to/rec-v4 \
  --proc-base-dir=/path/to/proc \
  --trajectory-csv=/path/to/slam-work/optimized_trajectory.csv \
  --aligned-standard
```

前一条命令执行 raw Pandar 归档、C++ Frontend、自主回环、Stage1 和 Stage2；后一条命令把
结果直接交给点云、Surface、图像、全景和着色链。

默认输出为 `PROC_BASE/DATASET_ID/recon/`：

```text
recon/
├── pointcloud.ply             # 36 字节/点的最终 RGBA/I/normal/curvature PLY
├── pointcloud_surface.ply     # 着色前的 32 字节/点表面 PLY
├── raw_shards/*.raytile       # 56 字节端点/原点记录，可重跑射线与表面算法
├── mapped_space/              # v2 quality_voxels.bin/sidecar 和 mapped_space.pcd
├── panoramas/00000.jpg ...    # 默认 8192×4096
├── pano/00000-pano.jpg ...    # 与原版同名的硬链接及 pano-poses.csv
├── cam/00000-cam0.jpg ...     # 自有 DNG 图像后处理结果
├── info/00000-info.json ...   # 由轨迹和标定合成的优化相机位姿
├── artifacts/trace.csv        # magnetic-field 纳秒时钟上的官方格式轨迹 trace
├── artifacts/floors.json      # 官方顶层数组 schema 的楼层与纳秒时间范围
├── trajectory.csv
├── processing_report.json
├── dataset.json
└── logs/proc/postprocessing-recon.log
```

## 快速端到端验证

```bash
./run_navvis_recon.sh \
  --proc-base-dir=/tmp/navvis-recon-proc \
  --output-dir=/tmp/navvis-recon-smoke \
  --caller=sitemaker \
  /media/cybergeo/12T/CSSJ/datasets_rec/2026-08-10_20.13.41 \
  --trajectory-bag=/path/to/trajectory.bag \
  --aligned-standard --force --res=0.01 --preset=standard \
  --max-duration=0.5 --max-panos=1 --pano-width=512
```

本机真实数据测试耗时 7.6 秒：读取两路 3,000 个包，保留 64 个原始分片，输出 120,315 个
表面着色点，并完成相机 JPEG、全景、深度遮挡着色和处理报告。

## G10/VLP16 全量验证

```bash
./run_navvis_recon.sh \
  --proc-base-dir=/media/cybergeo/12T/CSSJ/datasets_proc \
  --output-dir=/media/cybergeo/12T/CSSJ/datasets_proc/2023-05-15_10.18.42/recon \
  --caller=sitemaker \
  /media/cybergeo/12T/CSSJ/datasets_rec/2023-05-15_10.18.42 \
  --trajectory-bag=/media/cybergeo/12T/CSSJ/datasets_rec/2023-05-15_10.18.42/internal/trajectory_slam.bag \
  --aligned-standard --force --res=0.01 --cloud-format=ply --preset=standard \
  --num-threads-panos=32
```

G10 会自动选用较保守的自由空间参数 `minimum_intersections=6`、
`intersection_hit_ratio=3.0`。全量实测解码 104,340 个 VLP16 包，输出 3,538,026 个着色点和
4 张 8192×4096 全景。原版程序因该 G10-512 的许可过期而在 LicenseCheck 阶段停止，所以这组
数据没有可用于逐点几何误差评估的原版点云。

## 禾赛（HS/Pandar）默认自由空间参数

禾赛 HS/Pandar 数据的生产处理默认使用保守 v2 参数，避免自由空间删除过强造成墙面、地面或
远距离结构空洞：

| 参数 | 默认值 |
|---|---:|
| `minimum_intersections` | `3` |
| `intersection_hit_ratio` | `2.0` |
| `endpoint_margin` | `0.08 m`（8 cm） |
| 自由空间占据/穿越网格 | `0.02 m`（2 cm） |
| 射线半径 | `0.006 m`（6 mm） |
| 射线抽样步长 | `1`（不抽样） |

直接重跑 `navvis_recon_shard_surface_filter` 时必须显式传入以下参数；
`--free-space-nonstandard` 用于允许覆盖厂商对齐回归的冻结阈值：

```text
--free-space-carving
--free-space-mode sparse
--free-space-nonstandard
--free-space-traversal-resolution 0.02
--free-space-ray-radius 0.006
--free-space-ray-stride 1
--free-space-min-intersections 3
--free-space-intersection-hit-ratio 2.0
--free-space-endpoint-margin 0.08
```

自由空间的 2 cm 网格只用于射线穿越和占据判定，不是最终点云分辨率。最终 5 mm 点云仍需同时
传入 `--resolution 0.005 --output-cell 0.005`。只有复现厂商 G11 standard 或执行冻结对齐回归时，
才使用捕获参数 `1 / 1.0 / 0.05 m`；不要把该对齐口径作为 HS 数据的生产默认值。

## 完整阶段

1. 优先只读已有优化轨迹；缺失时融合录制的在线全局 SLAM 与局部里程计，并可对只读参考轨迹计算 ATE/RPE。
2. 从 raw `/imu/magnetic_field` 取得纳秒采样时钟，在轨迹上插值并按原版 6 有效数字写
   `artifacts/trace.csv`；顺序 Floor 状态机及 tiny/adjacent/split refiner 写官方
   `artifacts/floors.json`。缺少时钟、无时间重叠或时间非递增会直接报错，不使用旧高度聚类。
3. 读取水平和垂直 laser bag，按 Pandar 或 VLP16 block/beam 时间插值位姿。
4. 应用对应激光的逐束标定、设备反射区域和强度/距离规则；把每束激光的真实世界原点与回波端点
   一起写入 10 m `.raytile` 分片，并按端点体素与 0.5 m 原点单元保留独立视点。
5. 从 exact per-ray `.raytile` 生成 mapped-space quality：1/6 m voxel、每格至少 36 次贡献、
   255 个 Spherical Fibonacci 方向桶、厘米 range LUT、方向距离权重，以及 Brotli level 5 的
   13-byte v2 `<Morton key, diversity, ray count, min range>` 记录。`max-ray-length=50 m` 只用于
   原版空间分区，不裁短射线。
6. 禾赛 HS/Pandar 生产处理使用上节的保守 v2 自由空间参数 `3 / 2.0 / 0.08 m`，并保留跨
   10 m 分片射线历史；随后执行法线/曲率和与 `--res` 一致的输出体素化。只有厂商 G11
   standard 冻结对齐回归使用捕获规则：2 cm 占据体素、5 cm 端点截短、6 mm 射线到质心、
   0.5–15 m、85° 入射上限、至少 1 次穿越/比值 1.0。两套口径不得混用。
7. 每个采集点把四路 JPEG-XL DNG 无损转封装，C++ 解码并输出相机 JPEG；同时执行 OCam、
   GraphCut、曝光补偿和多频带融合生成 8K 全景。
8. 默认 `recon` 后端按 24 个独立鱼眼视图建立 684×456 PCT 深度图，应用 G11 相机 mask、
   二进制一致的高斯/边界权重和四次径向暗角模型，逐点保留最多 5 个视图并稳健融合，再执行
   自动 GammaModel 曝光求解和精确五近邻几何加权外推。`original` 仅供安装了原组件时做受控基准；`auto` 也只应在
   明确需要原版优先的对比实验中使用。
9. 写出最终 PLY，并在 `processing_report.json` 的 `cloud_coloring.backend` 中记录实际后端。

## 注意

- 当前全量 `.raytile` 约 63 GB，合并工作区峰值约 39 GB，表面/最终 PLY 分别约 4.6/5.1 GB，
  另需相机 JPEG 和全景空间；建议至少预留 140 GB。
- `--force` 只清理运行器在目标 `recon/` 下拥有的 `raw_shards`、`mapped_space`、`surface_work`、`panoramas`、
  `cam`、`info` 和三个点云文件，不删除目录中的其他文件。
- 8K 全景在 125 GiB 回归主机上最多并发 6 个采集点，并受 `--num-threads-panos` 总预算限制；
  `/usr/bin/time` 报告的最大单进程 RSS 约 6.4 GiB，但它不是六个子进程的聚合内存。原版同数据
  ImagePostprocessing 的聚合 PSS 峰值约 42.7 GiB，因此六路是证据支持的标准配置。单采集点
  内部 GraphCut 顺序不变。内存不足时降低 `--num-threads-panos`：20–24 对应最多 4 路，
  10–14 对应最多 2 路。
- 表面阶段支持 shard 预处理、tile 和 tile 内点循环三级并行。`--surface-preprocess-threads`
  控制同时读取/构建的 10 m shard，默认 8；`--surface-tile-threads` 控制同时计算的 5 m tile，运行器把
  `--num-threads-panos` CPU 预算平均分配为每个 tile 的 OpenMP 点线程。32 核、内存充足的
  本机完整 G11 回归使用 8 个预处理线程和 8 tile × 4 点线程；当前默认即为该配置。完整
  表面峰值 RSS 约 12.24 GiB，内存较小时应同时降低两个外层线程数。
- worker 自动发现优先 `build-release/`，然后是 `build-cpp/` 和 `cpp/build/`。正式回归仍建议
  显式传入 worker 路径并记录 SHA，避免不同构建污染对照。迁移包不携带旧机器预编译文件。
- 全景 worker 需要兼容的 LibRaw 0.22 开发/运行库；缺少依赖时应先安装再从源码构建。
- 默认 `--color-backend recon` 是源码独立路径，不调用 NavVis SDK 或原二进制。显式选择
  `original` 要求本机已安装原组件，并只用于离线对照。2026-02-08 的 122,701 点裁剪回归中，
  净室自动曝光完整路径保持全部非颜色字段及直接/外推 mask 一致，RGB MAE 为 0.05794，
  最大通道误差为 1；旧全量 MAE 22.49 是早期实现结果，当前版本尚未重新跑完整 12,061,091 点。
