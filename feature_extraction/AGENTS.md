# feature_extraction — Agent Instructions

本文件记录 feature_extraction 模块的构建、测试和性能比较流程。

---

## 依赖环境

| 项目 | 版本 / 路径 |
|------|-------------|
| CUDA | 12.8 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`) |
| TensorRT | 10.16.1.11 (`C:\Program Files\TensorRT-10.16.1.11`) |
| Python | 3.12 (venv: `.venv`) |
| PyTorch | 2.11.0+cu128 |
| OpenCV (C++) | 4.11.0 via vcpkg |
| Eigen3 | 3.4 via vcpkg |
| vcpkg toolchain | `F:/Library/vcpkg/scripts/buildsystems/vcpkg.cmake` |
| VS2022 | Professional (`D:\Program Files\Microsoft Visual Studio\2022\Professional`) |

---

## 构建流程

### 1. 编译 ALIKED custom_ops CUDA 扩展

```powershell
cd D:\ProjectX\project-3d\reality-capture-toolkit\feature_extraction\raw\ALIKED\custom_ops
# 需要 VS2022 Developer 环境
$env:DISTUTILS_USE_SDK = "1"
& d:\ProjectX\project-3d\reality-capture-toolkit\.venv\Scripts\python.exe setup.py build_ext --inplace
```

生成 `get_patches.cp312-win_amd64.pyd`。

### 2. 导出 ONNX 模型

```powershell
$env:PYTHONPATH = "D:\ProjectX\project-3d\reality-capture-toolkit"

# Backbone + SDDH
& .venv\Scripts\python.exe feature_extraction\tools\export_aliked_onnx.py `
  --model aliked-n32 --output feature_extraction\models\

# LightGlue
& .venv\Scripts\python.exe feature_extraction\tools\export_lightglue_onnx.py `
  --output feature_extraction\models\
```

输出：`feature_extraction/models/` 下的 `aliked_backbone.onnx`、`aliked_sddh.onnx`、`lightglue.onnx`。

### 3. 构建 TensorRT 引擎

> **重要**：Backbone 必须使用 FP32（`--no-fp16-backbone`）。DeformableConv2d TRT 插件在 FP16 下有严重精度损失，导致特征图退化、描述子坍缩，LightGlue 匹配数降至 0。SDDH 和 LightGlue 可安全使用 FP16。

```powershell
$env:PATH = "C:\Program Files\TensorRT-10.16.1.11\bin;" + $env:PATH
& .venv\Scripts\python.exe feature_extraction\tools\build_trt_engines.py `
  --no-fp16-backbone `
  --input feature_extraction\models\ --output feature_extraction\engines\
```

仅重建单个引擎：

```powershell
& .venv\Scripts\python.exe feature_extraction\tools\build_trt_engines.py `
  --no-fp16-backbone --only backbone
```

输出引擎（RTX 4060）：

| 引擎 | 精度 | 大小 | 构建时间 |
|------|------|------|----------|
| `aliked_backbone.engine` | **FP32** | 2.4 MB | ~101s |
| `aliked_sddh.engine` | FP16 | 1.6 MB | ~12s |
| `lightglue.engine` | FP16 | 25.5 MB | ~67s |

### 4. 编译 C++ 项目

```powershell
cmake -B feature_extraction/build -S feature_extraction `
  -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE="F:/Library/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DTensorRT_ROOT="C:\Program Files\TensorRT-10.16.1.11" `
  -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"

cmake --build feature_extraction/build --config Release -- /m
```

输出：`feature_extraction/build/Release/demo_feature_matching.exe`

---

## 性能测试

测试数据：`fisheye_x5_VID_20251017_113930_00_052_cam0`（604 张 3840×3840 JPG）

### 测试脚本

所有脚本运行前需设置环境：

```powershell
$env:PYTHONPATH = "D:\ProjectX\project-3d\reality-capture-toolkit"
$env:PATH = "C:\Program Files\TensorRT-10.16.1.11\bin;" + $env:PATH
```

| 脚本 | 用途 | 示例 |
|------|------|------|
| `compare_py_cpp.py` | Python vs C++ 正确性 + 性能对比 | `--image-dir <dir> --n-pairs 10 --start 6` |
| `benchmark_cpp.py` | C++ GPU 路径纯计时（无 dump） | `--image-dir <dir> --n-pairs 20 --warmup 3` |
| `bench_cached.py` | C++ 检测缓存 + 匹配计时 | `<dir> [n_pairs] [warmup] [lg_engine]` |
| `test_pair_python.py` | Python 单对参考 | `--image0 <img0> --image1 <img1>` |

### 测试结果（2026-04-17, RTX 4060）

#### 正确性（compare_py_cpp.py, 10 pairs）

| 指标 | 值 |
|------|-----|
| 关键点重叠率 | **93.1%**（median dist 0.04 px） |
| 描述子余弦相似度 | **0.9903** |
| 匹配数 Py / C++ | 2862 / **2815** |
| 匹配重叠率 | **88.6%** |

#### 性能

| 管线 | 耗时/对 | 吞吐 | 加速比 |
|------|---------|-------|--------|
| Python (ALIKED + LightGlue) | ~320 ms | 3.1 pairs/s | 1.0× |
| C++ TRT GPU (backbone FP32 + SDDH/LG FP16) | **147 ms** | **6.8 pairs/s** | **2.2×** |

> C++ 数据来自 `benchmark_cpp.py`（17 pairs, 3 warmup），std=3.0ms。`compare_py_cpp.py` 的 C++ 计时含 GPU→CPU 拷贝和文件 I/O，不代表真实推理性能。

---

## 注意事项

### Backbone 禁止使用 FP16

ALIKED backbone 的 DeformableConv2d（TRT `ModulatedDeformConv2d` v2 插件）在 FP16 下偏移量计算精度严重损失，导致描述子坍缩（内部相似度 0.033→0.61）、LightGlue 匹配归零。必须使用 `--no-fp16-backbone`。SDDH 和 LightGlue 的 FP16 不受影响。
