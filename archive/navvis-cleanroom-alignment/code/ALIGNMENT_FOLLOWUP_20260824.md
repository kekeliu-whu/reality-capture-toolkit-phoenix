# 二进制对齐复核（2026-08-24）

权威数据集：
`/media/cybergeo/12T/CSSJ/datasets_rec/2026-02-08_07.33.20`

只读 GT：
`/media/cybergeo/12T/CSSJ/datasets_proc_reference_g11_0109/2026-02-08_07.33.20`

本报告优先于 `ALIGNMENT_1_2_3_20260824.md` 中冲突的全量着色、全景和 IMU 结构结论。

## 1. SLAM

- 原版 `optimization_data.pb` 的 8,480 条 IMU 样本组成 1,616 个相邻节点的 9 维因子，不是 8,480 个独立 pose 因子。
- `ImuCostFunctionExact` 有 15 个参数块：两端 pose/velocity、IMU 外参、加速度/角速度内参和 gravity。
- `surveyor_slam.py` 现可直接读取优化器 protobuf，并保留 acceleration、angular velocity、delta velocity、delta rotation、orientation 与 calibration。
- 只替换积分顺序的候选使 ATE 平移 mean/p95 恶化到 20.971/39.661 mm，旋转 mean/p95 为 0.290/0.431°，1 s RPE 为 4.443 mm/0.152°，200 次评估未收敛；该数值改动已撤回。
- 当前 topology 读取仍 exact；独立 Submap 少 9 memberships，loop TP/FP/FN 仍为 7/11/8。完整 SLAM 未对齐。

下一探针应固定单线程、同一个 Exact factor，捕获样本区间、15 个参数块和 9 维输出残差，再实现完整坐标与权重约定。

## 2. 点云着色

### 2.1 发现并排除错误回归输入

旧目录 `/tmp/nv_pct_captured_current.EA8nlh/depth` 由 GDB 按并行访问顺序编号，并非 view 0..23；其中 12 张是相同的全 -1 图。把它作为 `--depth-map-input-dir` 会把直接点从约 946 万降到 318,626，错误 PLY 的 RGB MAE 约 72/255，KNN wall 达 9–13 分钟。这些结果无效，不能作为回归基线。

标准回归必须：

1. 不传 `--depth-map-input-dir`，让净室自建按 view 索引的 PCT 深度；或
2. 只使用净室 `--depth-map-output-dir` 写出的 24 张文件，并逐图记录 SHA。

### 2.2 正确深度下的最终 OVS

原版最终 OVS 由只读探针
`../navvis_alignment_reference/dump_original_final_ovs_colored_full_20260824.gdb`
捕获；生产实现不调用原版。

| 指标 | 原版 | 净室/差距 |
|---|---:|---:|
| 最终直接点 | 9,463,771 | 9,464,532；+761 |
| 直接 mask IoU | — | 99.991093% |
| 同点同 view 观测 | — | 27,749,713 |
| 观测 RGB exact | — | 99.9999928% |
| 观测 RGB 最大差 | — | 1 |
| 原版 Gamma 下直接点 MAE | — | 0.229833/255 |

此前的原版 Gamma 全点云隔离结果 MAE 0.322122/255 仍有效；正确深度证明最终视图、图像采样、融合和 KNN 不是自动曝光 6.11 的主差距。

### 2.3 自动曝光

二进制证据确认 `Histogram8U::addData` 排除 HSV value=0；启用后 24/24 个 dynamic low/high 与原版一致。但在剩余曝光目标未对齐时，结果发生误差抵消：

| 自动曝光口径 | RGB MAE/255 | PSNR | 最大通道差≤20 |
|---|---:|---:|---:|
| 旧默认（保留零值） | 6.113880 | 28.213 dB | 93.7046% |
| 排除零值实验 | 7.324628 | 26.928 dB | 87.2879% |

因此排除零值条件未进入默认生产代码；其完整证据保存在
`/media/cybergeo/12T/CSSJ/datasets_proc_performance_alignment_20260823/color_exposure_zero_hist_20260824/evidence_and_regression.json`。

正确自建深度、有限差分自动曝光和最终 KNN 的实验 wall 为 119.40 s，峰值 3,544,504 KiB；原版 standalone 为 18.90 s，当前约慢 6.32×。输出位于
`/media/cybergeo/12T/CSSJ/datasets_proc_performance_alignment_20260823/color_gt_geometry_auto_gamma_zero_hist_selfdepth_20260824`。

下一步应按 octree leaf/XYZ 对齐原版与净室 0.1 m centroid，并在同一冻结目标中比较 quality 和 residual，而不是继续调最终 RGB。

## 3. 全景

二进制探针确认：

- 全景内部 densifier 使用 `weight-ray=1`；公开 dense depth 的 `100000` 不属于该路径。
- 优化器为四层、每层最多 100 次，未观测区域满足调和约束。
- `head-plus` 平移符号正确。

净室用四层 screened-Laplacian 替换最近邻深度填充。六张原生 8K 输出均改善：

| capture | MAE 1024 前→后 | SSIM 前→后 |
|---|---:|---:|
| 00000 | 7.8913→7.5249 | 0.9455→0.9605 |
| 00001 | 12.6227→12.4158 | 0.9418→0.9531 |
| 00002 | 7.2724→6.4974 | 0.8867→0.9119 |
| 00003 | 16.9791→15.9639 | 0.7779→0.8323 |
| 00004 | 7.2381→7.1916 | 0.9304→0.9331 |
| 00005 | 9.7369→9.6876 | 0.9616→0.9638 |
| 平均 | 10.2901→9.8802 | 0.9073→0.9258 |

原生 8192×4096 MAE 均值 11.6250→10.9627。使用原版 projected inputs、只运行净室 seam/blend 的 2K MAE 为 5.4994，因此剩余主差距仍在深度辅助投影，不能称完全对齐。详细 JSON：
`../navvis_alignment_reference/panorama_depth_optimizer_alignment_20260824.json`。

## 4. 统一验收

- CMake Release build：通过，`-O3 -DNDEBUG` 与 OpenMP 已启用。
- CTest：2/2。
- Python：14/14。
- runner 与 SLAM probe `py_compile`：通过。
- surface occlusion status/keep：exact。
- surface main-input XYZ/origin/intensity/weight：exact。
- surface output voxel keys：exact。

本轮验收 worker：

- surface colorizer SHA-256 `fc713fa7021cdd8389412de923226475e402c919fd679da321da73a19495ca86`
- panorama SHA-256 `560c9e8db51ae3c31c308b0c98659cfd329d31c4e1e6139ff16e2510c8680925`

