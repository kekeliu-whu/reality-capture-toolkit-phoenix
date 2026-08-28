# G11 标准流程分模块结果与效率验收（2026-08-24）

## 1. 结论

本轮在 `2026-02-08_07.33.20` 上按“先结果、后性能”重新验收当前 Release worker。构建、
单元测试、冻结表面验收、三个端到端比较器、录制 SLAM 评测、当前着色 worker 全量重跑和
10 秒 raw-bag 集成 smoke 均通过。当前优化后的 surface、color 和 panorama 输出分别与冻结
净室基线逐字节一致，说明性能修改没有改变净室结果。

但整条流程尚未达到“结果基本相同且效率接近原版”的总目标：Pandar/CloudBuilder 已在保持
冻结净室 shards byte-exact 的前提下快于原版约 16%，surface 已缩小到慢约 1.11×；全量颜色、
全景结果、raw-scan-to-final 离线 SLAM 和 quality map 仍有差距。只有图像后处理像素、冻结
自由空间/遮挡/表面决策，以及若干隔离的着色/全景阶段可以称为 exact 或基本相同。

状态定义：

- `EXACT`：同一冻结输入、同一字段或 mask 逐位/逐条一致。
- `NEAR`：误差很小或端到端几何达到当前基本相同门限，但不是逐位一致。
- `NOT_ALIGNED`：完整结果有明显差距、缺少同输入验收，或尚未接入标准生产链。

## 2. 本轮实际执行

| 验收 | 本轮结果 |
|---|---|
| Release 构建 | `-O3 -DNDEBUG -fopenmp -std=c++17`，成功 |
| C++ / Python | CTest 1/1；Python 13/13；runner `py_compile` 通过 |
| 冻结表面 acceptance | 通过；wall 0.36 s，峰值 RSS 32.5 MiB |
| 统一比较器复跑 | geometry/color/panorama JSON 与冻结报告字段完全一致 |
| 录制 SLAM 复跑 | 指标与冻结 JSON 完全一致；wall 0.57 s（仅评测/融合，不是离线 SLAM） |
| 当前着色 worker 全量复跑 | 107.74 s；12,076,978 点；输出与冻结净室 PLY byte-exact |
| 当前 10 s raw-bag smoke | CloudBuilder 66,322 个 Pandar 包、398 scans、51 raytiles，约 3.101 s；修改前为 14.147 s；shards byte-exact |
| 当前全景性能产物 | 6 张 panorama + 24 张 camera JPEG 与冻结净室输出 byte-exact |
| 资源完整性 | `MANIFEST.sha256` 2,879/2,879 通过 |

当前 worker SHA-256：

| Worker | SHA-256 |
|---|---|
| Pandar | `44a5e3f3cddd1bb22de51095bd227b9fd5d1a062a9e2522324ee4aaa1429cbf6` |
| Surface | `1794b9756f3fda20d1630b326223cf977a1c77b033089a2ba95debfae352be49` |
| Colorizer | `d0b14127246ff094dd7a3e29567fa39aa65f75115d8e6582d0406a88bd5c55ec` |
| Panorama | `24f7e8f95641efc34ed094817ecb21138fe405893eb480371c7b5d36e4c1885b` |
| Surface acceptance | `eb914912570d3e1b8ccecc8e502cf1432379f4fa96faba5f3a7fadf22f55c34a` |

## 3. 标准流程分模块验收矩阵

| 标准模块 | 同输入/端到端结果 | 判定 | 当前效率与原版比较 |
|---|---|---|---|
| 轨迹输入、标定与调度 | 提供原版 `trajectory.bag` 时 8,081 poses 与参考一致；10 s smoke 可从 bag 接通 Pandar→surface | `EXACT`（仅后处理输入模式） | 调度开销不是主瓶颈 |
| Raw-to-final Surveyor SLAM | 前端、双 active submap、回环和 IMU pose/velocity graph 有实现，但未接入 runner 的 raw Pandar 离线全链 | `NOT_ALIGNED` | 无可比完整耗时；不能与原版 99.055 s 比 |
| Submap 生命周期 | 原版 6 submaps/2,581 memberships；净室 6/2,572，少 9（0.35%） | `NEAR` | 无独立生产计时 |
| 回环候选/约束 | 401 eligible、41 searches 与原版一致；15 个原版 pair 的 ICP 平移 mean/p95 6.51/16.31 mm，旋转 mean/p95 0.109/0.449°；独立 accepted pair 集仍不同 | `NEAR` | 无完整可比计时 |
| IMU 图后端 | 历史隔离实验使用原版 topology 时，对 final state 平移 mean/p95 5.30/9.94 mm，旋转 0.119/0.279° | `NEAR`（后端隔离） | 本轮重算 197.65 s 尚未完成即终止；原版两次优化约 0.34/0.55 s，性能明显未对齐 |
| 录制在线 SLAM 回退 | ATE 位置 mean/p95 22.88/34.93 mm；旋转 0.238/0.434°；1 s RPE 平移 mean 2.83 mm | `NOT_ALIGNED` | 本轮 0.57 s 只是融合与评测 |
| Pandar 解码/点云构建 | 528,916 包、3,174 scans；门限后少 978（-0.00117%）；边缘后少 15,705（-0.01911%） | `NEAR`；性能修改前后净室 shards `EXACT` | 27.419 s vs 32.655 s，净室快约 16.0% |
| 垂直脚部过滤 | 51,480 输入，4,238 候选、1,808 删除标签逐点一致；坐标 mean 1.53 µm、max 0.030 mm | `EXACT`（决策）/`NEAR`（坐标） | 包含在 CloudBuilder |
| 自由空间 | 2,667,377 occupied key、hit、intersection 逐叶全一致；删除 852,434，FP/FN 0/0 | `EXACT`（内部计数/决策） | 最终源码全量 free-space/input 49.176 s（后续 Surface 仅跑 1 个诊断 tile） |
| 自由空间质心 | 2,662,135/2,667,377 float32 逐位一致；mean/max 误差 1.28 nm/8.596 µm | `EXACT`（µm 差异按当前口径忽略） | 未改变 occupancy/hit/intersection/delete mask；非 bitwise 的 5,242 项继续披露 |
| 遮挡与主输入 | 146,045 status/keep 全一致；保留 138,708；109,829 主输入全部字段逐位一致 | `EXACT` | 冻结探针约 0.29 s |
| Multi-scale normal | 零法向 mask 全一致；角误差 p50/p95 0.000625°/0.002242°，max 1.110° | `NEAR` | 冻结探针约 0.09 s |
| Surface selection | weight 全逐位一致；109,003/109,322 XYZ 逐位一致，最大差 0.954 µm | `NEAR` | 冻结探针约 0.02 s |
| Invalid normal / output voxel | 数量/mask 一致；87,255 个 output key 全一致，86,712 XYZ 逐位一致 | `EXACT`（决策/key）/`NEAR`（坐标） | 冻结探针约 0.07 s |
| Density / Adaptive SOR | 87,255→86,408 与 86,408→86,241 的 mask/记录一致 | `EXACT` | 冻结探针各约 0.02/0.09 s |
| 完整 surface 点集 | 12,076,978 vs 原版 12,061,091，+0.13172%；五区域双向 p95 平均约 4.80 mm | `NEAR` | 79.960 s vs 72.297 s，慢 1.106×；冻结净室 SHA 不变 |
| 点云着色权重与分类 | 122,701 点裁剪属性、直接/外推 mask、六张 float 权重图一致 | `EXACT` | 无独立裁剪生产计时 |
| 自动 Gamma 裁剪着色 | RGB MAE 0.057943/255，最大通道差 1 | `NEAR` | 当前全量着色 107.74 s |
| 全量点云颜色 | 五区域 1 cm 匹配 RGB MAE 26.8549/255，系统性偏暗 | `NOT_ALIGNED` | 107.74 s vs 18.9025 s，慢 5.70× |
| 图像曝光/Mertens | 三曝光和 Mertens 冻结输出逐位一致；NLM 输入 99.999557% 通道一致 | `EXACT`（中间态） | 与 panorama 在同一 capture worker，无法可信拆时 |
| 图像最终像素 | cam0–3 MAE 0.000670/0.000610/0.000499/0.000712（8-bit）；JPEG 私有元数据不同 | `NEAR` | 图像+全景合计见下 |
| GraphCut seam | 四 mask IoU 99.10/99.84/99.52/98.41% | `NEAR` | 包含在全景总时间 |
| Projected-input stitch | 同一原版 projected input 的 2K RGB MAE 4.4414/255 | `NEAR` | 包含在全景总时间 |
| Depth/operator mask 与 multiband | depth/operator mask 未完成同输入验收；7-band 流程已知但精确浮点/舍入未验收 | `NOT_ALIGNED` | 当前主要结果缺口之一 |
| Nadir 填充 | 缺口 MAE 2.885/255；整图 0.4631；底部 20% 2.293 | `NEAR` | 8K 隔离 55.67 s，约 1.93 GiB |
| 六张全链全景 | 平均 MAE 15.4569/255，PSNR 21.125 dB，SSIM 0.6980 | `NOT_ALIGNED` | 130.722 s vs 图像+全景 88.307 s，慢 1.48×；比旧净室快 4.29× |
| Floor 算法（2026-08-28 重验） | 220 组、1,796,815 条 trace、615 floors；顶层 schema、z 边界和全部纳秒 ranges 220/220 逐字段一致 | `EXACT` | 当前守门 wall 5.335 s；含单层、多层、回访、tiny/adjacent merge、double split |
| Floor runner 接线（2026-08-28 重验） | 已生成官方 `artifacts/trace.csv`/`floors.json`；两组 G11 的 5,374/17,878 行 trace byte-exact，1/5 floors exact | `EXACT`（已验 G11） | 缺失 magnetic clock/无时间重叠直接失败；G10 smoke 670 条/1 floor，无官方真值 |
| Mapped-space quality | ray-count 合成测试通过；runner 未生成原版 1,320,880 voxel 的 `quality_voxels.bin` | `NOT_ALIGNED` | 1.27 ms 合成测试；已验证去掉 78-byte 私有头后的 Brotli payload 为 17,171,440 bytes、推断 13 bytes/voxel，但不能与原版 13.024 s 比 |

## 4. 确定性与输出

- Surface 当前完整性能输出：
  `/media/cybergeo/12T/CSSJ/datasets_proc_performance_alignment_20260823/surface_scratch_freespace_8pre_8x4/pointcloud_surface.ply`
  ，SHA-256 `0dcdc2729230d0ec770fb8ccbb6d16dbee1a71e8c156c5fed847d672ae3f4207`。
- 当前 colorizer 独立全量输出：
  `/media/cybergeo/12T/CSSJ/datasets_proc_performance_alignment_20260823/color_current_worker_module_verify_20260824/pointcloud.ply`
  ，SHA-256 `c2ce9bd767f9d1754b60d37b60fa2a10d8f3e895165e37762f21dd032f6787b6`。
- 当前 10 秒集成 smoke：
  `/media/cybergeo/12T/CSSJ/datasets_proc_performance_alignment_20260823/module_smoke_current_10s_20260824/pointcloud_surface.ply`
  ，2,634,023 点、SHA-256 `5ecf65ebf22daaa2cea0ab5f6e9c5384e37481cd1b6b73c25867a798c1998f0b`。
- 六路全景性能输出：
  `/media/cybergeo/12T/CSSJ/datasets_proc_performance_alignment_20260823/pano_parallel_8k_6capture/`；
  6 张 panorama 和 24 张 camera JPEG 与冻结净室输出逐文件一致。

以上 byte-exact 只表示“当前优化前后净室结果不变”，不表示与 NavVis 原版结果 byte-exact。

## 5. 性能判断与后续顺序

当前 Release 选项正确，`-march=native -flto` 的历史隔离实验只改善约 6–9%。性能缺口主要是
算法数据结构和内存布局，不是漏开 `-O3`：

1. Surface：用一棵全局 CompactOctree 替换 shard-local/均匀哈希路径，并按原版
   NanoFLANN/PCL KD-tree 的数据布局重做邻域查询；必须保持完整 surface SHA 不变。
2. 全量着色：先修自动 GammaModel 的采样、权重和求解，再优化 9.47M 直接点和 2.61M
   五近邻填色；不能通过最终 RGB 经验增亮。
3. Panorama：先冻结 depth/operator mask，再定位 multiband 浮点/舍入；保持 30 个 JPEG
   对当前 clean baseline 的确定性。
4. SLAM：把 raw Pandar scan→双 active submap→candidate/loop→IMU graph 接入 runner，
   另行报告 raw-to-final ATE/RPE 和完整耗时；当前 Python/SciPy 后端 197.65 s 未跑完，
   需要在保持历史隔离误差的前提下改为稀疏/解析 Jacobian 和高效线性求解。
5. Quality：实现原版序列化和逐 voxel 验收。Floor 的 runner 接线与多层回归已在
   2026-08-28 完成，见 `FLOOR_FULL_ALIGNMENT_20260828.md`，不再属于待办。

## 6. 证据

- `test_resources/SURFACE_SAME_INPUT_ACCEPTANCE_V1.md`
- `test_resources/CLEANROOM_COLOR_ALIGNMENT_V10.md`
- `code/EVIDENCE.md`
- `regression/2026-02-08_07.33.20/data_regression_report.json`
- `regression/2026-02-08_07.33.20/geometry_regression_vs_navvis.json`
- `regression/2026-02-08_07.33.20/color_regression_vs_navvis.json`
- `regression/2026-02-08_07.33.20/panorama_regression_vs_navvis.json`
- `regression/2026-02-08_07.33.20/recorded_slam_regression_vs_navvis.json`
- `regression/2026-02-08_07.33.20/topology_acceptance_20260824.json`
- `regression/2026-02-08_07.33.20/floor_trace_acceptance_20260824.json`
- `regression/2026-02-08_07.33.20/quality_acceptance_20260824.json`
- `regression/2026-02-08_07.33.20/runner_calibration_acceptance_20260824.json`
- `regression/2026-02-08_07.33.20/PERFORMANCE_ALIGNMENT_20260824.md`
- 原版计时：`/media/cybergeo/12T/CSSJ/datasets_proc_reference_g11_0109/2026-02-08_07.33.20/logs/proc/processing_step_timings.json`
