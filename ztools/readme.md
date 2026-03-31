# 数据处理工作流

## 概述
该工作流包含两个主要步骤：数据处理和颜色化处理。

## 步骤 1: 数据处理
运行主处理脚本，将原始传感器数据转换为点云：

```powershell
./ztools/run.ps1 -inputdir d:\ProjectX\project-3d\data\manifold-tech-calib\MT20260326-162323 -insvpath d:\ProjectX\project-3d\data\manifold-tech-calib\MT20260326-162323\VID_20260326_162319_00_057.insv -calibfile d:\ProjectX\project-3d\data\manifold-tech-calib\ikalibr_param.yaml -outputdir D:/output
```

**参数说明：**
- `-inputdir`: 原始输入数据目录（包含 Manifold 导出数据）
- `-insvpath`: INSV 视频文件路径（Insta360 视频格式）
- `-calibfile`: 相机标定参数文件（YAML 格式，推荐填写以启用 Manifold 转换）
- `-outputdir`: 输出目录路径，存放处理结果

## 步骤 2: 颜色化处理
使用 XColor 模块为点云数据添加颜色信息：

```powershell
./ztools/run-xcolor.ps1 D:/output
```

**参数说明：**
- 输入目录：第一步的输出目录，包含点云数据

## 输出
- 第一步生成带地理坐标的点云数据
- 第二步生成带 RGB 颜色信息的点云数据