# `nv_colorcloud` 着色对齐：dynamic Cauchy 与 HSV255 修复

## 结论

本轮通过只读反汇编和 GDB 动态探针确认，旧净室实现的主要错误不是最终 RGB 融合，而是曝光优化少了原版对每个相机 dynamic-range 三残差块使用的 `ScaledLoss(CauchyLoss(0.1), view_scale)`。同时还存在 `/255`、`/65535` 的 float32 提升顺序，以及 `GammaModel::apply` 没有走原版 `RGB -> HSV255 -> 修改 V -> RGB` 浮点路径的问题。

三项均已在 `cpp/apps/surface_panorama_colorizer.cpp` 修复。`2026-02-08_07.33.20` 的同序 GT 几何全量回归中，净室自动 Gamma 的 RGB MAE 从上一条正确零值 histogram 基线的 `7.3246277/255` 降至 `0.7484060/255`，下降 `89.78%`。几何逐位一致，`98.9848%` 的点最大通道差不超过 5。

当前结果仍不能称为最终 RGB 完全对齐：加载同次原版 Gamma 的隔离基线为 `0.3221219/255`，说明自动 Gamma 与少量上游 OVS/权重差异仍留下约 `0.4263/255` 的 MAE 余量。

## 二进制证据

- 安装程序：`/opt/NavVis/pointcloud-coloring/bin/nv_colorcloud`
- SHA-256：`b582016681f9552cfec69471c51f3f9373a5828a2cefd3e1eddc31324545c234`
- Build ID：`a7586f518009434f5e97891f897aea42675f26a0`
- Software Version：`996799455e02da742b525658d521da45edae9d10`
- 详细反汇编与运行时公式：`color_binary_followup/NV_COLORCLOUD_EXPOSURE_GAMMA_BINARY_EVIDENCE.md`

确认的曝光目标为：

1. 每个多视图点构造一个 2～5 维 joint block，残差为 `q_i * (y_i - weighted_mean)^2`，整块使用 `Cauchy(0.005)`。
2. 每个活跃 view 构造一个三维 dynamic-range block，残差为 `1.1-(H-L)/(h-l)`、`min(L,0)`、`max(H-1,0)`，整块使用 `view_scale * Cauchy(0.1)`。
3. scene-brightness 只选 `high > 203` 的 view，使用 joint block 数作为 `ScaledLoss(nullptr, N_joint)` 的 scale。
4. `V` 和质量权重分别先以 float32 乘 `1/255`、除 `65535`，再提升为 double。
5. Gamma 只作用于 HSV255 的 V；每条观测先校正，再按 float32 权重融合。

旧代码遗漏第 2 项的 Cauchy loss。全量 dynamic 总 scale 是裁剪的约 `107.06x`，因此裁剪回归看似接近，而全量模型会明显偏离。

## 求解器隔离

输入同次原版曝光 OVS `/tmp/nv_original_full_ovs_same_run.bin` 时，修复后的净室 Ceres 问题为：

| 项目 | 原版 | 净室 |
|---|---:|---:|
| joint blocks | 168,827 | 168,827 |
| joint scalar residuals | 516,152 | 516,152 |
| dynamic blocks / residuals | 24 / 72 | 24 / 72 |
| scene blocks / residuals | 1 / 1 | 1 / 1 |
| 初始代价 | 780.9323 | 780.9323 |
| 最终代价 | 0.1066163 | 0.1068101 |
| 迭代行 | 51 | 51 |

24 路模型相对原版的参数误差：gain MAE `0.00116786`、最大 `0.00389445`；exponent MAE `0.00104385`、最大 `0.00324516`。Ceres 2.0 与 2.1 的当前解析 Jacobian 输出一致，因此剩余差异不是编译所用 Ceres 小版本造成的。

## 上游 OVS 隔离

原版与净室曝光云均为 `416,789` 个一米体素代表点、`5,450` 个空间 voxel；全部 XYZ 可一一精确匹配。空间匹配后：

- active mask TP/FP/FN：`261,299 / 109 / 26`；
- 五视图集合完全一致：`416,141 / 416,789`；
- 重叠观测 `709,231`，原版独有 310，净室独有 583；
- 重叠 RGB `99.7604%` 完全一致，通道 MAE `0.01356/255`；
- 量化质量的 uint16 MAE 为 `47.87/65535`。

这些差值很小，但曝光目标非常平坦，50 次未收敛 LM 会把它们放大为不同的最终 Gamma 轨迹。最终点云直接着色 OVS 本身已达到 mask IoU `99.9911%`，同点同 view 的 RGB 几乎全部精确。

## 全量回归

输入：

- GT/同序几何：`/media/cybergeo/12T/CSSJ/datasets_proc_reference_g11_0109/2026-02-08_07.33.20/pointcloud.ply`
- 真实 view ID 深度：`/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/test_resources/color_final_ovs_20260824/clean_depth`
- 输出：`/media/cybergeo/12T/CSSJ/datasets_proc_performance_alignment_20260823/color_dynamic_cauchy_hsv_20260824/pointcloud.ply`
- 机器可读指标：`color_dynamic_cauchy_hsv_full_20260824.json`

| 指标 | 修复后自动 Gamma | 旧零值 histogram 自动 Gamma | 同次原版 Gamma 隔离 |
|---|---:|---:|---:|
| RGB MAE /255 | 0.748406 | 7.324628 | 0.322122 |
| RGB RMSE /255 | 3.233968 | 11.485044 | 3.129700 |
| PSNR | 37.9361 dB | 26.9282 dB | 38.2207 dB |
| RGB 完全一致点 | 32.7871% | 5.2613% | 67.1544% |
| 最大通道差 <= 5 | 98.9848% | 66.1228% | 98.9840% |
| 最大通道差 <= 20 | 99.7435% | 87.2879% | 99.7435% |

输出几何、强度、法线、曲率和 alpha 均与参考逐位一致。wall `95.13 s`、峰值 RSS `3,544,752 KiB`；原版 standalone 为 `18.90 s`，当前约慢 `5.03x`。

## 验证

- Release C++ build：成功；`build-cpp/navvis_recon_surface_colorizer` SHA-256 为 `76e0e7bad881489d5e592f3304b4b15b2c2aa5d7c326af58265b6c7f272b8020`。
- Ceres 2.1 诊断 build：成功；worker SHA-256 为 `3805bb95220ad94899b5efeb37a0daa7739f3e201a79925051cef22007133b14`。
- CTest：两套构建均 `2/2` 通过。
- Python：`14/14` 通过。

## 尚未闭合

1. 原始 score 到 top-5 uint16 质量的少量边界分歧，及原版/净室独有的 893 条曝光观测。
2. 同一目标、同一初值下，原版与净室 Ceres 最后约 `0.000194` 的代价差和约 `1e-3` 量级模型差；现有证据不支持通过经验调参数消除。
3. 最终 RGB MAE `0.7484/255` 已接近视觉无显著差异，但按严格数值验收仍是“基本对齐”，不是逐点完全一致。
