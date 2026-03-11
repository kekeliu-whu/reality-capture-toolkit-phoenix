# 如何使用`parse_gyro.py`
运行它之前通过以下命令生成 protobuf 文件：
```powershell
python -m grpc_tools.protoc -I. --python_out=. proto/sensors.proto
```

---

## 快速开始 - 使用构建脚本

### PowerShell 版本（推荐）
在项目根目录执行：
```powershell
.\migration\scripts\build-python-tools.ps1
```

**功能：**
- ✅ 自动检查 Python 和 pip
- ✅ 自动安装 PyInstaller
- ✅ 自动安装项目依赖（从 requirements.txt）
- ✅ 编译所有 Python 脚本为 EXE
- ✅ 清理构建产物
- ✅ 生成软件包 (ZIP 文件)
- ✅ 显示构建总结

生成的 EXE 文件会保存到 `python-tools/` 目录

---

### Batch 版本（Windows）
如果不想使用 PowerShell，也可以用 Batch 脚本：
```cmd
.\migration\scripts\build-python-tools.bat
```

---

## 手动编译（无需脚本）

### 1. 安装 PyInstaller（如未安装）
```powershell
pip install pyinstaller
```

### 2. 编译脚本到 python-tools 文件夹
在项目根目录或 `migration/scripts` 目录下执行以下命令：

```powershell
# 编译 insta_data_extraction.py
pyinstaller `
  --onefile `
  --distpath "../../python-tools" `
  --buildpath "build_insta_data_extraction" `
  --specpath "." `
  insta_data_extraction.py

# 编译 insta_time_sync.py
pyinstaller `
  --onefile `
  --distpath "../../python-tools" `
  --buildpath "build_insta_time_sync" `
  --specpath "." `
  insta_time_sync.py

# 编译 insta_compute_poses.py
pyinstaller `
  --onefile `
  --distpath "../../python-tools" `
  --buildpath "build_insta_compute_poses" `
  --specpath "." `
  insta_compute_poses.py
```

### 3. 或使用批量编译脚本
创建 `build_all.ps1` 文件，一键编译所有脚本：
```powershell
$python_files = @(
    "insta_data_extraction.py",
    "insta_time_sync.py",
    "insta_compute_poses.py"
)

foreach ($file in $python_files) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($file)
    Write-Host "编译 $file ..."
    pyinstaller `
        --onefile `
        --distpath "../../python-tools" `
        --buildpath "build_$name" `
        --specpath "." `
        $file
}
```

执行：
```powershell
.\build_all.ps1
```

### 4. 编译后的 EXE 位置
编译完成后，可执行文件会保存到：
- `python-tools/insta_data_extraction.exe`
- `python-tools/insta_time_sync.exe`
- `python-tools/insta_compute_poses.exe`

### 5. 直接使用 EXE（无需 Python 环境）
```powershell
# 数据提取
.\python-tools\insta_data_extraction.exe `
  --input-video-filename "path/to/video.insv" `
  --output-dir "D:/slam/camera/" `
  --time-offset 1735725830.203135

# 时间同步
.\python-tools\insta_time_sync.exe `
  --device "D:\slam\imu.dat" `
  --insta "D:/slam/camera/insv.dat"

# 计算位姿
.\python-tools\insta_compute_poses.exe `
  --poses-file D:/slam/output/traj.txt `
  --calib-file D:/slam/calibration.dat `
  --image-folder D:/slam/camera `
  --output D:/slam/camera/ImgPose.txt
```

---

## 使用 PyInstaller 编译 Python 脚本为 EXE

### 1. 安装 PyInstaller（如未安装）
```powershell
pip install pyinstaller
```

### 2. 编译脚本到 python-tools 文件夹
在项目根目录或 `migration/scripts` 目录下执行以下命令：

```powershell
# 编译 insta_data_extraction.py
pyinstaller `
  --onefile `
  --distpath "../../python-tools" `
  --buildpath "build_insta_data_extraction" `
  --specpath "." `
  insta_data_extraction.py

# 编译 insta_time_sync.py
pyinstaller `
  --onefile `
  --distpath "../../python-tools" `
  --buildpath "build_insta_time_sync" `
  --specpath "." `
  insta_time_sync.py

# 编译 insta_compute_poses.py
pyinstaller `
  --onefile `
  --distpath "../../python-tools" `
  --buildpath "build_insta_compute_poses" `
  --specpath "." `
  insta_compute_poses.py
```

### 3. 或使用批量编译脚本
创建 `build_all.ps1` 文件，一键编译所有脚本：
```powershell
$python_files = @(
    "insta_data_extraction.py",
    "insta_time_sync.py",
    "insta_compute_poses.py"
)

foreach ($file in $python_files) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($file)
    Write-Host "编译 $file ..."
    pyinstaller `
        --onefile `
        --distpath "../../python-tools" `
        --buildpath "build_$name" `
        --specpath "." `
        $file
}
```

执行：
```powershell
.\build_all.ps1
```

### 4. 编译后的 EXE 位置
编译完成后，可执行文件会保存到：
- `python-tools/insta_data_extraction.exe`
- `python-tools/insta_time_sync.exe`
- `python-tools/insta_compute_poses.exe`

### 5. 直接使用 EXE（无需 Python 环境）
```powershell
# 数据提取
.\python-tools\insta_data_extraction.exe `
  --input-video-filename "path/to/video.insv" `
  --output-dir "D:/slam/camera/" `
  --time-offset 1735725830.203135

# 时间同步
.\python-tools\insta_time_sync.exe `
  --device "D:\slam\imu.dat" `
  --insta "D:/slam/camera/insv.dat"

# 计算位姿
.\python-tools\insta_compute_poses.exe `
  --poses-file D:/slam/output/traj.txt `
  --calib-file D:/slam/calibration.dat `
  --image-folder D:/slam/camera `
  --output D:/slam/camera/ImgPose.txt
```

---

## convert_s20 - 从 ROS bag 转换 Livox 激光雷达和 IMU 数据
```powershell
.\convert_s20 `
  --bag_filename="\\wsl.localhost\Ubuntu-24.04\home\rick\iKalibr\src\iKalibr\2026-02-06_11-34-29-s20\all_2026-02-06-11-34-35.bag" `
  --calib_filename="\\wsl.localhost\Ubuntu-24.04\home\rick\iKalibr\src\iKalibr\2026-02-06_11-34-29-s20\ikalibr_output\ikalibr_param.yaml" `
  --output_dir="D:\slam"
```

## laser_mapping - LIDAR 里程计和建图
```powershell
.\laser_mapping `
  --project_dirname="D:\slam" `
  --output_dir="D:\slam\output" `
  --indoor=true
```

## insta_data_extraction.py - 科研相机数据提取
```powershell
python insta_data_extraction.py `
  --input-video-filename "\\wsl.localhost\Ubuntu-24.04\home\rick\video.insv" `
  --output-dir "D:/slam/camera/" `
  --time-offset 1735725830.203135 `
  --export-frames
```

## insta_time_sync.py - IMU 时间同步
```powershell
python insta_time_sync.py `
  --device "D:\slam\imu.dat" `
  --insta "D:/slam/camera/insv.dat"
```

## insta_compute_poses.py - 计算相机位姿
```powershell
python insta_compute_poses.py `
  --poses-file D:/slam/output/traj.txt `
  --calib-file D:/slam/calibration.dat `
  --image-folder D:/slam/camera `
  --output D:/slam/camera/ImgPose.txt
```
