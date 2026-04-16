# Agent Instructions

本文件供 GitHub Copilot Coding Agent 使用，记录项目的工作流命令、参数说明和操作规范。

---

## 常用命令

### ALIKED 特征点检测

#### 编译 custom_ops CUDA 扩展

每次修改 `.cu` / `.cpp` 源码后需重新编译：

```powershell
cd D:\ProjectX\project-3d\reality-capture-toolkit\ALIKED\custom_ops
& d:\ProjectX\project-3d\reality-capture-toolkit\.venv\Scripts\python.exe setup.py build_ext --inplace
```

编译成功后会生成 `get_patches.cp312-win_amd64.pyd`。

#### 测试 demo（10 帧快速验证）

```powershell
cd D:\ProjectX\project-3d\reality-capture-toolkit\ALIKED
& d:\ProjectX\project-3d\reality-capture-toolkit\.venv\Scripts\python.exe demo_seq.py `
  D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\ground_undistort\fisheye_x5_VID_20251017_113930_00_052_cam0 `
  --n_frames 10 `
  --output output_test
```

预期输出：

- 特征提取 median ≈ 104ms（第 1 帧因 GPU 冷启动约 4s，属正常）
- 特征匹配 median ≈ 122ms
- 每帧约 5000 个关键点，≥3000 个 matches

#### 注意事项

- 必须在 `ALIKED/` 目录下执行 `demo_seq.py`，否则找不到 `nets/` 模块
- `custom_ops` 编译依赖 MSVC + CUDA 12.8，需在 VS 2022 Developer 环境下运行
- 测试数据路径：`D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\ground_undistort\fisheye_x5_VID_20251017_113930_00_052_cam0`

---

### run-ar — Insta360 点云重建流水线

将 Insta360 录制的 INSV 视频 + 外置相机 MOV + 激光点云数据合并，完成轨迹对齐、颜色化、建图等全流程处理。

```powershell
.\ztools\run-ar.ps1 `
  -inputdir     D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud `
  -insvpath     D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud\VID_20260415_122738_00_228.insv `
  -outputdir    D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud\output `
  -calibfile    D:\output\calibration.dat `
  -movpath      "D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud\2026-04-16 113900.mov" `
  -trajectoryfile "D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud\Project_2026-04-16_14-49-15\2026-04-15_12-26-00_PointCloud\output\trajectory.txt"
```

#### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `-inputdir` | ✅ | 点云采集数据根目录，包含 INSV、点云等原始文件 |
| `-insvpath` | ✅ | Insta360 全景视频文件路径（`.insv`） |
| `-outputdir` | ✅ | 处理结果输出目录，不存在时自动创建 |
| `-calibfile` | ✅ | 相机标定文件路径（`.dat`），用于外参对齐 |
| `-movpath` | 可选 | 外置相机视频路径（`.mov`），用于颜色化点云 |
| `-trajectoryfile` | 可选 | 预计算的轨迹文件，跳过 SLAM 直接使用已有轨迹 |

#### 目录命名规律

- `inputdir` 目录名格式：`YYYY-MM-DD_HH-MM-SS_PointCloud`
- `insvpath` 文件名格式：`VID_YYYYMMDD_HHMMSS_00_<设备ID>.insv`
- `trajectoryfile` 通常位于 Insta360 Studio 导出的 Project 目录内

---

### run — SLAM点云和全景POS生成流水线

处理 Manifold 采集的标定数据，完成 Livox 点云转换、IMU 时间同步、轨迹计算等步骤（不含颜色化）。

```powershell
.\ztools\run.ps1 `
  -inputdir  D:\ProjectX\project-3d\data\manifold-tech-calib\calib\MT20260326-161907-ikalibr\ `
  -insvpath  D:\ProjectX\project-3d\data\manifold-tech-calib\calib\MT20260326-161907-ikalibr\VID_20260326_161848_00_055.insv `
  -calibfile D:\ProjectX\project-3d\data\manifold-tech-calib\ikalibr_param.yaml `
  -outputdir D:\output2
```

#### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `-inputdir` | ✅ | 采集数据根目录，包含 INSV 和原始点云文件 |
| `-insvpath` | ✅ | Insta360 全景视频文件路径（`.insv`） |
| `-outputdir` | ✅ | 处理结果输出目录，不存在时自动创建 |
| `-calibfile` | 可选 | 标定参数文件路径（`.dat` 或 `.yaml`），用于 Manifold 转换 |

#### 与 run-ar 的区别

- `run.ps1`：无外置相机，不做颜色化，适用于**标定采集场景**
- `run-ar.ps1`：含外置相机 MOV + 颜色化，适用于**实际建图场景**
