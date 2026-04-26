# Agent Instructions

本文件供 GitHub Copilot Coding Agent 使用，记录项目的工作流命令、参数说明和操作规范。

---

## 常用命令

### ALIKED 特征点检测

> 当前仓库中的 ALIKED 代码位于 `sfm-phoenix/raw/ALIKED/`，不是顶层 `ALIKED/`。
> Python 侧实际验证通过的环境为 `C:\Users\rick\miniconda3\envs\cusfm\python.exe`。

#### 编译 custom_ops CUDA 扩展

每次修改 `.cu` / `.cpp` 源码后需重新编译：

```powershell
cd D:\codes\reality-capture-toolkit\sfm-phoenix\raw\ALIKED\custom_ops
$env:DISTUTILS_USE_SDK = "1"
& C:\Users\rick\miniconda3\envs\cusfm\python.exe setup.py build_ext --inplace
```

编译成功后会生成 `get_patches.cp312-win_amd64.pyd`。

#### 推荐测试命令

先设置运行环境，避免 `torch` 导入时缺少 `shm.dll` / `cudnn64_9.dll`：

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONPATH = "D:\codes\reality-capture-toolkit"
$env:PATH = "d:\codes\tmp\colmap-x64-windows-cuda\bin;" +
            "C:\Program Files\NVIDIA\CUDNN\v9.21\bin\12.9\x64;" +
            "C:\Users\rick\miniconda3\envs\cusfm;" +
            "C:\Users\rick\miniconda3\envs\cusfm\Library\bin;" +
            "C:\Users\rick\miniconda3\envs\cusfm\Scripts;" +
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" +
            "C:\Program Files\TensorRT-10.16.1.11\bin;" +
            $env:PATH
```

#### Python 单对参考

```powershell
& C:\Users\rick\miniconda3\envs\cusfm\python.exe sfm-phoenix\tools\test_pair_python.py `
  --image0 d:\codes\tmp\images\00006_5.jpg `
  --image1 d:\codes\tmp\images\00007_6.jpg `
  --output sfm-phoenix\compare_output\matches_py.jpg
```

#### Python / C++ 对比测试

```powershell
& C:\Users\rick\miniconda3\envs\cusfm\python.exe sfm-phoenix\tools\compare_py_cpp.py `
  --image-dir d:\codes\tmp\images `
  --n-pairs 12 `
  --warmup 2 `
  --start 6 `
  --max-edge 1024 `
  --output-dir sfm-phoenix\compare_output
```

2026-04-19 实测结果：

- Python / C++ 都稳定输出约 5000 个关键点
- 关键点重叠率约 94.2%，median dist 约 0.07 px
- 描述子余弦相似度约 0.9856
- 匹配数约为 Py 3421 / C++ 3618
- 匹配重叠率约 91.4%
- 总耗时约为 Py 188 ms / pair，C++ compare 模式 110 ms / pair

#### C++ 纯性能测试

```powershell
& C:\Users\rick\miniconda3\envs\cusfm\python.exe sfm-phoenix\tools\benchmark_cpp.py `
  --image-dir d:\codes\tmp\images `
  --n-pairs 12 `
  --warmup 2 `
  --start 6 `
  --max-edge 1024
```

2026-04-19 实测结果：

- 单图提取约 33.0 ms
- 单对匹配约 40.0 ms
- 单对总耗时约 72.9 ms
- 稳态吞吐约 13.72 pairs/s

#### 注意事项

- Python 入口现已统一按 `sfm-phoenix/raw/ALIKED` 作为包根导入；不要再使用旧的 `feature_extraction.raw.ALIKED` 包路径
- `custom_ops` 编译依赖 MSVC + CUDA 12.8，需在 VS 2022 Developer 环境下运行
- 当前快速测试数据路径：`d:\codes\tmp\images`

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

---

### hloc SfM（ALIKED + LightGlue）

用于新建 SfM 工程，执行局部特征提取、全局检索配对、时序配对融合、
LightGlue 匹配，并导出 COLMAP 格式重建结果。

#### 环境安装（Windows + Python 3.12）

在项目根目录执行：

```powershell
cd D:\codes\reality-capture-toolkit
& C:\Users\rick\miniconda3\envs\cusfm\python.exe -m pip install --upgrade pip
& C:\Users\rick\miniconda3\envs\cusfm\python.exe -m pip install "git+https://github.com/cvg/Hierarchical-Localization.git"
& C:\Users\rick\miniconda3\envs\cusfm\python.exe -m pip install pycolmap
```

说明：`hloc` 会自动安装 `lightglue` 依赖。

#### 代理设置（模型下载）

首次运行会下载 ALIKED / LightGlue / NetVLAD 权重。网络受限时需设置代理：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
```

若手动下载模型，Windows 下可用：

```powershell
curl.exe -k -L "<MODEL_URL>" -o "<OUTPUT_FILE>"
```

#### 关键权重缓存路径

- `C:\Users\rick\.cache\torch\hub\checkpoints\aliked-n16.pth`
- `C:\Users\rick\.cache\torch\hub\checkpoints\aliked_lightglue.pth`
- `C:\Users\rick\.cache\torch\hub\netvlad\VGG16-NetVLAD-Pitts30K.mat`

#### 运行命令（示例）

```powershell
cd D:\codes\reality-capture-toolkit
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
& C:\Users\rick\miniconda3\envs\cusfm\python.exe sfm-phoenix\hloc_sfm.py `
  --image_dir "D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\ground_undistort\fisheye_x5_VID_20251017_113930_00_052_cam0" `
  --output_dir "D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\ground_undistort\fisheye_x5_VID_20251017_113930_00_052_cam0\hloc_output"
```

#### 输出说明

- 局部特征：`hloc_output/feats-aliked-n16.h5`
- 全局描述子：`hloc_output/global-feats-netvlad.h5`
- 配对文件：`hloc_output/pairs-retrieval-netvlad20.txt`、
  `hloc_output/pairs-seq.txt`、`hloc_output/pairs-merged.txt`
- 匹配结果：`hloc_output/matches-aliked-lightglue.h5`
- COLMAP 重建：`hloc_output/sfm/`

---

### Phoenix 特征提取 + 匹配

用于直接运行 `phoenix feature_extractor` 和 `phoenix feature_matcher`，
自动补齐常用运行时 PATH，并串起两步处理。

```powershell
.\scripts.ps1 `
  -ImageDir "D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\ground_undistort\fisheye_x5_VID_20251017_113930_00_052_cam0" `
  -OutputDir "D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\ground_undistort\fisheye_x5_VID_20251017_113930_00_052_cam0\phoenix_output" `
  -DatabasePath "D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\ground_undistort\fisheye_x5_VID_20251017_113930_00_052_cam0\phoenix_output\database.db" `
  -MaxEdge 1024 `
  -TopK 5000 `
  -MaxMatches 4000
```

---

### DINOv3 ONNX（ModelScope 下载 + FP16 导出）

用于 Phoenix retrieval 的 DINOv3 模型文件名固定为：

- `sfm-phoenix/models/dinov3_vitb16_pretrain_lvd1689m.onnx`

#### 依赖安装（cusfm）

```powershell
& C:\Users\rick\miniconda3\envs\cusfm\python.exe -m pip install --upgrade modelscope transformers huggingface_hub safetensors
```

#### 导出命令（优先使用 ModelScope）

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONPATH = "D:\ProjectX\project-3d\reality-capture-toolkit"
& C:\Users\rick\miniconda3\envs\cusfm\python.exe sfm-phoenix\tools\export_dinov3_onnx.py `
  --download-from-modelscope `
  --modelscope-model-id facebook/dinov3-vitb16-pretrain-lvd1689m `
  --modelscope-cache-dir D:\ProjectX\project-3d\reality-capture-toolkit\sfm-phoenix\tmp_modelscope_cache `
  --output D:\ProjectX\project-3d\reality-capture-toolkit\sfm-phoenix\models\dinov3_vitb16_pretrain_lvd1689m.onnx `
  --export-fp16 `
  --opset 19
```

#### 导出后核验

- ONNX 输入名：`pixel_values`，shape `[-1, 3, 224, 224]`
- ONNX 输出名：`embeddings`，shape `[-1, 768]`
- 当前导出脚本默认走 FP16（输入/输出与权重均为 FLOAT16）

#### automatic_reconstructor 验证命令（已通过）

```powershell
$env:PATH = "C:\Program Files\TensorRT-10.16.1.11\bin;F:\Library\vcpkg\installed\x64-windows\bin;" + $env:PATH
& D:\ProjectX\project-3d\reality-capture-toolkit\build\RelWithDebInfo\phoenix.exe automatic_reconstructor `
  --database_path D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\tmp\1.db `
  --image_path D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\tmp\camera\fisheye_x5_VID_20251017_113930_00_052_cam0\ `
  --output_path D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\tmp\xsfm\ `
  --ImageReader.camera_model OPENCV_FISHEYE
```
