# C++17 核心算法与真实数据 worker

## 代码布局

六组非 SLAM 核心算法位于 `include/navvis_recon/` 和 `src/`：

- `cloud_builder`：轨迹插值、逐点去畸变、过滤、体素化、规则扫描法线和 RANSAC；
- `cloud_surface_filter`：自由空间/占用体素、多尺度 PCA 法线、密度与离群点、切向平滑；
- `mapped_space_quality`：精确 ray traversal、方向/range 聚合、13-byte v2 记录和 Brotli 输出；
- `pointcloud_coloring`：深度遮挡、patch 投影、视图排名、曝光和多视图颜色融合；
- `image_postprocessing`：白平衡、暗角、HDR、去噪、锐化、隐私模糊和 JPEG；
- `panorama_rendering`：OCam/等距柱状投影、多频带融合、盲区修补与 surfel 渲染。

真实 G10/G11 rec-v4 的流式入口位于 `apps/`：

| target | 输入/输出 |
|---|---|
| `navvis_recon_pandar` | 自动识别 stdin 中的 820 字节 Pandar XTM 或 1206 字节 Velodyne VLP16 包帧；输出 1 cm 端点体素与真实世界射线原点，可用 `--retain-shards DIR --shards-only` 保留 56 字节 `.raytile` 中间数据 |
| `navvis_recon_shard_surface_filter` | 合并端点/原点簇并执行稀疏自由空间射线雕刻；输出 8×float、32 字节表面 PLY；`--adaptive-density` 是完整 standard 对齐分支 |
| `navvis_recon_mapped_space_quality` | 读取 exact per-ray 56 字节 `.raytile`，输出 `mapped_space/quality_voxels.bin`、sidecar 和 PCD；旧聚合 shard 会直接失败 |
| `navvis_recon_ocam_panorama` | 四路 lossless-JPEG DNG；输出全景 JPEG，并可用 `--decoded-dir` 输出后处理相机 JPEG |
| `navvis_recon_surface_colorizer` | 8-float 表面 PLY 或已有 36 字节彩色 PLY、优化相机位姿、24 路相机 JPEG 和全景；输出与参考相同字段顺序的 36 字节点云。`--direct-mask FILE.u8` 可额外输出每点 1 字节的直接相机着色诊断掩码 |

## 已验证的 standard 参数

- Pandar 距离包单位：包头值 5 对应 0.005 m；
- 32 束水平/垂直激光分别使用逐束 elevation、azimuth 和旋转光源偏置；
- 原始体素 0.01 m、空间分片 10 m；
- G11 standard 稀疏自由空间严格使用二进制捕获参数：2 cm 占据体素、射线区间
  `[0.5 m, min(range-0.05 m, 15 m)]`、射线到占据质心 6 mm、最大入射角 85°、
  每条射线参与统计，至少 1 次穿越且穿越/命中比达到 1.0；交叉 10 m 分片的射线历史会保留；
- 表面密度格 0.025 m，PCA 法线格 0.10 m，最终输出格跟随 `--res`（standard 为 0.01 m）；
- 已删除没有二进制依据的“占据体素达到 8,000,000 就切换密度门槛”和 0.0131 m 点数拟合；
- G11 着色按 24 个独立鱼眼视图生成二进制一致的 684×456 PCT 深度图，使用原始
  5472×3648 掩膜、一像素表面 splat 和原版 `DepthMap::isVisibleInCamera` 分支；最大视距 30 m；
- OCam 轴约定为 `(Y,X,-Z)`，忽略 JPEG EXIF 旋转；径向暗角采用二进制中确认的正向/逆向
  三项多项式组合；每点保留 Top-5 视图并执行自适应鲁棒融合；
- 默认曝光从 0.1 m 曝光云求解 GammaModel；未直接着色点由独立 C++ KD 树按原版精确的
  五近邻、双精度距离平方和几何权重外推；`--voxel-view-selection` 仅保留为诊断实验，
  原版最终五视图选择是逐点路径；
- 全景 8192×4096，GraphCut `COLOR` 在 2048×1024 尺度估计接缝，7 层多频带融合；
- high 图像路径的已确认常量为 Mertens 权重 1/1/1、NLM `h=4.6785717`/模板 7/搜索 17、
  `sigma=3` 非锐化、JPEG 质量 95；
- 8K 使用逐相机 gain，避免 OpenCV 4.5 的 block-gain 在原生尺寸上产生超大内存分配。

G10/VLP16 使用标准 16 束仰角表、2 mm 距离单位、块间方位角插值和逐通道发射时间；水平/垂直
激光分别使用原始强度下限 2/1，该分支已在 104,340 个真实 VLP16 包上完成全量运行。但 G10
原厂许可已过期，没有同数据原版表面结果，因此 6 次穿越/比值 3.0 只保留在 diagnostic
directional 路径，不再冒充 standard 二进制参数；当前 sparse standard 明确对齐的是 G11。

## 构建

需要 CMake 3.16+、C++17、Eigen3、OpenCV 4 和 OpenMP。全景 worker 还需要 LibRaw 头文件和
库；没有 LibRaw 时，其余库和三个点云 worker 仍可构建。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

本机验证环境为 GCC 11.4、Eigen 3.4、OpenCV 4.5.4、OpenMP 4.5。当前 smoke test 覆盖
轨迹去畸变、表面法线、方向与稀疏自由空间雕刻、深度着色、图像处理、全景和 surfel 渲染；统一 runner 的 0.5 秒
真实数据端到端测试还覆盖了原始分片、adaptive filter、相机位姿合成和最终 PLY。

## 与原发布物的对应

代码保留了发布物中可由 RTTI/符号确认的结构，例如 `FreespaceOctree`、
`AdaptiveStatisticalOutlierRemoval`、`MultiScaleNormalEstimator`、`DirectPatchColorExtractor`、
`PatchProjector`、`VoxelRanking`、`DepthMap`、`GlobalExposureOptimizer`、
`MultiViewColorBlending`、`GaussNewtonDepthMapOptimizer`、`SeamMaskPreparer`、
`MultiBandBlender`、`PyramidInpainting` 和 `FloorFiller`。

这些源码是依据发布二进制和可控实验写出的独立实现。剥离后的局部变量名、模板源文件、
专有自由空间阈值和优化器内部权重无法无损恢复；相关等价实现及剩余误差记录在上级
`EVIDENCE.md` 和对齐报告中。
