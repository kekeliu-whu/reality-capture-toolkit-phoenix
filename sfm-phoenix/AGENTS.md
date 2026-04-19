# sfm-phoenix — Agent Instructions

本文件记录 sfm-phoenix 模块的构建、测试和性能比较流程。

---

## 依赖环境

| 项目 | 版本 / 路径 |
|------|-------------|
| CUDA | 12.8 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`) |
| TensorRT | 10.16.1.11 (`C:\Program Files\TensorRT-10.16.1.11`) |
| Python | 3.12 (`C:\Users\rick\miniconda3\envs\cusfm\python.exe`) |
| PyTorch | 2.11.0+cu128 |
| OpenCV (C++) | 4.11.0 via vcpkg |
| Eigen3 | 3.4 via vcpkg |
| vcpkg toolchain | `D:/vcpkg/scripts/buildsystems/vcpkg.cmake` |
| VS2022 | Enterprise (`D:\Program Files\Microsoft Visual Studio\2022\Enterprise`) |

### 当前机器补充（2026-04-19）

- Python 环境实际使用：`C:\Users\rick\miniconda3\envs\cusfm\python.exe`
- 为避免 `torch` 导入时报 `shm.dll` / `cudnn64_9.dll` 缺失，运行 Python 前需把下列目录加入 `PATH` 之一：
  - `d:\codes\tmp\colmap-x64-windows-cuda\bin`
  - `C:\Program Files\NVIDIA\CUDNN\v9.21\bin\12.9\x64`
- ONNX 导出依赖已验证可用：`onnx`、`onnxscript`
- CMake 可执行文件路径：
  - `D:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`

推荐的 PowerShell 环境变量：

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

---

## 构建流程

### 1. 编译 ALIKED custom_ops CUDA 扩展

```powershell
cd D:\codes\reality-capture-toolkit\sfm-phoenix\raw\ALIKED\custom_ops
# 需要 VS2022 Developer 环境
$env:DISTUTILS_USE_SDK = "1"
& C:\Users\rick\miniconda3\envs\cusfm\python.exe setup.py build_ext --inplace
```

生成 `get_patches.cp312-win_amd64.pyd`。

如果 `setup.py build_ext --inplace` 在 Windows 上因 `torch.utils.cpp_extension`
的 `ninja` / `cl.exe` 调度失败，可按下面方式手工生成 `.pyd`：

```powershell
cmd /c '"D:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1 && cd /d d:\codes\reality-capture-toolkit\sfm-phoenix\raw\ALIKED\custom_ops && "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -std=c++17 -Xcompiler /MD -Xcompiler /wd4819 -Xcompiler /wd4251 -Xcompiler /wd4244 -Xcompiler /wd4267 -Xcompiler /wd4275 -Xcompiler /wd4018 -Xcompiler /wd4190 -Xcompiler /wd4624 -Xcompiler /wd4067 -Xcompiler /wd4068 -Xcompiler /EHsc --use-local-env -Xcudafe --diag_suppress=base_class_has_different_dll_interface -Xcudafe --diag_suppress=field_without_dll_interface -Xcudafe --diag_suppress=dll_interface_conflict_none_assumed -Xcudafe --diag_suppress=dll_interface_conflict_dllexport_assumed -IC:\Users\rick\miniconda3\envs\cusfm\Lib\site-packages\torch\include -IC:\Users\rick\miniconda3\envs\cusfm\Lib\site-packages\torch\include\torch\csrc\api\include -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" -IC:\Users\rick\miniconda3\envs\cusfm\include -IC:\Users\rick\miniconda3\envs\cusfm\Include -I"D:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.44.35207\include" -I"D:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.44.35207\ATLMFC\include" -I"D:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\VS\include" -I"D:\Windows Kits\10\include\10.0.26100.0\ucrt" -I"D:\Windows Kits\10\include\10.0.26100.0\um" -I"D:\Windows Kits\10\include\10.0.26100.0\shared" -I"D:\Windows Kits\10\include\10.0.26100.0\winrt" -I"D:\Windows Kits\10\include\10.0.26100.0\cppwinrt" -I"C:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\include\um" -c get_patches_cuda.cu -o build\temp.win-amd64-cpython-312\Release\get_patches_cuda.obj -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=get_patches -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86'

cmd /c '"D:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1 && cd /d d:\codes\reality-capture-toolkit\sfm-phoenix\raw\ALIKED\custom_ops && link.exe /DLL /NOLOGO /INCREMENTAL:NO /OUT:get_patches.cp312-win_amd64.pyd /IMPLIB:get_patches.cp312-win_amd64.lib build\temp.win-amd64-cpython-312\Release\get_patches.obj build\temp.win-amd64-cpython-312\Release\get_patches_cuda.obj /LIBPATH:C:\Users\rick\miniconda3\envs\cusfm\libs /LIBPATH:C:\Users\rick\miniconda3\envs\cusfm\Lib\site-packages\torch\lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" python312.lib c10.lib torch.lib torch_cpu.lib torch_python.lib cudart.lib c10_cuda.lib torch_cuda.lib'
```

### 2. 导出 ONNX 模型

```powershell
$env:PYTHONPATH = "D:\codes\reality-capture-toolkit"

# Backbone + SDDH
& C:\Users\rick\miniconda3\envs\cusfm\python.exe sfm-phoenix\tools\export_aliked_onnx.py `
  --model aliked-n32 --output sfm-phoenix\models\

# LightGlue
& C:\Users\rick\miniconda3\envs\cusfm\python.exe sfm-phoenix\tools\export_lightglue_onnx.py `
  --output sfm-phoenix\models\
```

输出：`sfm-phoenix/models/` 下的 `aliked_backbone.onnx`、`aliked_sddh.onnx`、`lightglue.onnx`。

### 3. 构建 TensorRT 引擎

> **重要**：Backbone 必须使用 FP32（`--no-fp16-backbone`）。DeformableConv2d TRT 插件在 FP16 下有严重精度损失，导致特征图退化、描述子坍缩，LightGlue 匹配数降至 0。SDDH 和 LightGlue 可安全使用 FP16。

```powershell
$env:PATH = "C:\Program Files\TensorRT-10.16.1.11\bin;" + $env:PATH
& C:\Users\rick\miniconda3\envs\cusfm\python.exe sfm-phoenix\tools\build_trt_engines.py `
  --no-fp16-backbone `
  --input sfm-phoenix\models\ --output sfm-phoenix\engines\
```

仅重建单个引擎：

```powershell
& C:\Users\rick\miniconda3\envs\cusfm\python.exe sfm-phoenix\tools\build_trt_engines.py `
  --no-fp16-backbone --only backbone
```

输出引擎（RTX 4060）：

| 引擎 | 精度 | 大小 | 构建时间 |
|------|------|------|----------|
| `aliked_backbone.engine` | **FP32** | 2.4 MB | ~101s |
| `aliked_sddh.engine` | FP16 | 1.6 MB | ~12s |
| `lightglue.engine` | FP16 | 25.5 MB | ~67s |

若 1600 profile 在 8GB GPU 上构建 backbone 时出现 CUDA runtime 失败，可先用
1024 profile 稳定完成验证：

```powershell
$env:PATH = "C:\Program Files\TensorRT-10.16.1.11\bin;" + $env:PATH

& "C:\Program Files\TensorRT-10.16.1.11\bin\trtexec.exe" `
  --onnx="d:\codes\reality-capture-toolkit\sfm-phoenix\models\aliked_backbone.onnx" `
  --saveEngine="d:\codes\reality-capture-toolkit\sfm-phoenix\engines\aliked_backbone.engine" `
  --minShapes=image:1x3x320x320 `
  --optShapes=image:1x3x1024x1024 `
  --maxShapes=image:1x3x1024x1024 `
  --builderOptimizationLevel=3 `
  --avgTiming=1 `
  --skipInference

& "C:\Program Files\TensorRT-10.16.1.11\bin\trtexec.exe" `
  --onnx="d:\codes\reality-capture-toolkit\sfm-phoenix\models\aliked_sddh.onnx" `
  --saveEngine="d:\codes\reality-capture-toolkit\sfm-phoenix\engines\aliked_sddh.engine" `
  --minShapes=feature_map:1x128x320x320,keypoints_wh:100x2,feature_map_hw:2 `
  --optShapes=feature_map:1x128x1024x1024,keypoints_wh:5000x2,feature_map_hw:2 `
  --maxShapes=feature_map:1x128x1024x1024,keypoints_wh:5000x2,feature_map_hw:2 `
  --fp16 `
  --builderOptimizationLevel=3 `
  --avgTiming=1 `
  --skipInference

& "C:\Program Files\TensorRT-10.16.1.11\bin\trtexec.exe" `
  --onnx="d:\codes\reality-capture-toolkit\sfm-phoenix\models\lightglue.onnx" `
  --saveEngine="d:\codes\reality-capture-toolkit\sfm-phoenix\engines\lightglue.engine" `
  --minShapes=kpts0:1x100x2,desc0:1x100x128,kpts1:1x100x2,desc1:1x100x128 `
  --optShapes=kpts0:1x5000x2,desc0:1x5000x128,kpts1:1x5000x2,desc1:1x5000x128 `
  --maxShapes=kpts0:1x5000x2,desc0:1x5000x128,kpts1:1x5000x2,desc1:1x5000x128 `
  --fp16 `
  --builderOptimizationLevel=3 `
  --avgTiming=1 `
  --skipInference
```

### 4. 编译 C++ 项目

```powershell
"D:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -B sfm-phoenix/build -S sfm-phoenix `
  -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DTensorRT_ROOT="C:\Program Files\TensorRT-10.16.1.11" `
  -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"

"D:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build sfm-phoenix/build --config Release -- /m
```

输出：`sfm-phoenix/build/Release/demo_feature_matching.exe`

---

## 性能测试

本轮实际测试数据：`d:\codes\tmp\images`（连续 JPG 序列，按相邻帧配对）

历史基准数据：`fisheye_x5_VID_20251017_113930_00_052_cam0`（604 张 3840×3840 JPG）

### 测试脚本

所有脚本运行前需设置环境：

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

| 脚本 | 用途 | 示例 |
|------|------|------|
| `compare_py_cpp.py` | Python vs C++ 正确性 + 性能对比 | `--image-dir <dir> --n-pairs 10 --start 6` |
| `benchmark_cpp.py` | C++ GPU 路径纯计时（无 dump） | `--image-dir <dir> --n-pairs 20 --warmup 3` |
| `bench_cached.py` | C++ 检测缓存 + 匹配计时 | `<dir> [n_pairs] [warmup] [lg_engine]` |
| `test_pair_python.py` | Python 单对参考 | `--image0 <img0> --image1 <img1>` |

`compare_py_cpp.py` 与 `benchmark_cpp.py` 当前已支持 `--max-edge`，在 8GB GPU 上建议先用 `--max-edge 1024` 做性能对比。

### 测试结果（2026-04-19, `d:\codes\tmp\images`, `max-edge=1024`）

> 说明：该机器在 8GB 显存下构建 `1600x1600` backbone engine 会触发 TensorRT
> CUDA runtime 失败，因此本轮性能数据统一采用 `1024x1024` profile。

#### 正确性（`compare_py_cpp.py`, 12 pairs, 2 warmup）

| 指标 | 值 |
|------|-----|
| 关键点重叠率 | **94.2%**（median dist 0.07 px） |
| 描述子余弦相似度 | **0.9856** |
| 匹配数 Py / C++ | 3421 / **3618** |
| 匹配重叠率 | **91.4%** |

#### 性能

> 本轮重新统计只使用一个 LightGlue 模型 / 引擎：`lightglue.engine`。
> 不再并排比较两个 LG 版本。

| 管线 | 耗时/对 | 吞吐 | 备注 |
|------|---------|-------|------|
| Python 全流程（ALIKED + 单个 LightGlue） | **188 ms** | **5.32 pairs/s** | `compare_py_cpp.py`, 12 pairs, 2 warmup, `max-edge=1024` |
| C++ compare 模式（含 dump / 对齐开销） | **110 ms** | **9.06 pairs/s** | `compare_py_cpp.py`, 12 pairs, 2 warmup, `max-edge=1024` |
| C++ TRT GPU 稳态（backbone FP32 + SDDH/LG FP16） | **72.9 ms** | **13.72 pairs/s** | `benchmark_cpp.py`, 12 pairs, 2 warmup, `max-edge=1024` |

#### 分段耗时拆分（2026-04-19）

> 口径说明：
>
> - Python 分段来自 `compare_py_cpp.py`，每对图像都会重新提取两张图的特征。
> - Python 汇总为 12 对、前 2 对 warmup 丢弃，最终统计 10 对 measured。
> - C++ 分段来自 `demo_feature_matching.exe` 的 `PROFILE=1` 输出。
> - C++ batch / GPU 优化路径会复用上一对的左图特征，所以**稳态**每对只新增提取 1 张图；这里的“提取耗时 / TPS”按新增 1 张图统计。

| 管线 | 单图提取耗时 | 提取 TPS | 单对匹配耗时 | 匹配 TPS | 单对总耗时 | 总吞吐 |
|------|--------------|----------|--------------|----------|------------|--------|
| Python 全流程（ALIKED + 单个 LightGlue） | **约 45 ms / image** | **约 22.2 img/s** | **约 98 ms / pair** | **约 10.2 pair/s** | **188 ms / pair** | **5.32 pair/s** |
| C++ compare 模式（含 dump / 对齐开销） | **约 34.6 ms / image** | **28.88 img/s** | **约 41.1 ms / pair** | **24.31 pair/s** | **110 ms / pair** | **9.06 pair/s** |
| C++ benchmark 稳态（左图缓存，纯 GPU 路径） | **33.0 ms / image** | **30.35 img/s** | **40.0 ms / pair** | **25.03 pair/s** | **72.9 ms / pair** | **13.72 pair/s** |

C++ `PROFILE=1` 首对冷启动参考值：

- `det0=172.69 ms`
- `det1=33.49 ms`
- `match=55.19 ms`
- `total=261.37 ms`

说明：首对包含 TensorRT / CUDA 首次调度与缓存建立，不代表稳态吞吐，汇总时应剔除。

---

## 注意事项

### Python 导入路径

当前仓库布局下，ALIKED Python 代码应通过 `sfm-phoenix/raw/ALIKED` 作为包根运行，入口脚本统一使用 `from nets.aliked import ALIKED`，`raw/ALIKED/nets/aliked.py` 内部统一使用相对导入。

不要再使用旧的 `feature_extraction.raw.ALIKED` 包路径；该路径会导致 `compare_py_cpp.py`、`test_pair_python.py` 等脚本在当前仓库结构下导入失败。

### Backbone 禁止使用 FP16

ALIKED backbone 的 DeformableConv2d（TRT `ModulatedDeformConv2d` v2 插件）在 FP16 下偏移量计算精度严重损失，导致描述子坍缩（内部相似度 0.033→0.61）、LightGlue 匹配归零。必须使用 `--no-fp16-backbone`。SDDH 和 LightGlue 的 FP16 不受影响。
