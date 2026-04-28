# ztools 使用文档

## 概述


### 适用场景

当你已经具备以下数据时，使用 run-ar.ps1：

- 采集目录 inputdir，包含原始数据。
- Insta360 的 .insv 视频。
- 外置相机的 .mov 视频。
- 可用的 calibration.dat。
- 已有轨迹文件 trajectory.txt。

该脚本会串联完成 Manifold 数据转换、Insta360 数据提取、IMU 时间同步、图像导出，以及 ImgPose.txt 生成。

### 处理命令示例

下面这条就是成都 0427 这批数据的处理命令：

```powershell
.\ztools\run-ar.ps1 `
	-inputdir D:\Users\rick\Desktop\chengdu_0427\ `
	-insvpath "D:\Users\rick\Desktop\chengdu_0427\VID_20260427_143745_00_233.insv" `
	-outputdir D:\Users\rick\Desktop\chengdu_0427\output `
	-calibfile D:\output-s20\calibration.dat `
	-movpath "D:\Users\rick\Desktop\chengdu_0427\2026-04-27 141205.mov" `
	-trajectoryfile "D:\Users\rick\Desktop\chengdu_0427\Project_2026-04-28_11-38-54\chengdu_0427\output\trajectory.txt"
```

### 参数说明

| 参数 | 是否建议必填 | 说明 | 成都 0427 示例值 |
|------|--------------|------|------------------|
| -inputdir | 是 | 采集数据根目录，包含原始点云/IMU等文件 | D:\Users\rick\Desktop\chengdu_0427\ |
| -insvpath | 是 | Insta360 全景视频文件 | D:\Users\rick\Desktop\chengdu_0427\VID_20260427_143745_00_233.insv |
| -outputdir | 是 | 输出目录，不存在时自动创建 | D:\Users\rick\Desktop\chengdu_0427\output |
| -calibfile | 是 | 标定文件，用于姿态和相机配置生成 | D:\output-s20\calibration.dat |
| -movpath | 是 | 外置相机视频，用于导出外置相机帧 | D:\Users\rick\Desktop\chengdu_0427\2026-04-27 141205.mov |
| -trajectoryfile | 是 | 预先计算的轨迹文件，用于生成 ImgPose.txt | D:\Users\rick\Desktop\chengdu_0427\Project_2026-04-28_11-38-54\chengdu_0427\output\trajectory.txt |

### 脚本执行流程

run-ar.ps1 按以下顺序执行：

1. 调用 convert_manifold.exe，将 inputdir 中的 Manifold/Livox 数据转换到 outputdir。
2. 调用 insta_data_extraction_ar.exe，对 INSV 先做一次无导帧提取，生成 Insta360 相关元数据。
3. 调用 insta_time_sync.exe，对设备 IMU 与 INSV IMU 做时间同步，并解析最终 time delay。
4. 再次调用 insta_data_extraction_ar.exe，带上 time offset 和 movpath，正式导出图像帧。
5. 调用 insta_compute_poses.exe，基于 trajectoryfile 与 calibfile 尝试生成 ImgPose.txt；若位姿计算工具支持当前目录结构，也会一并生成 rig.json。

### 典型输出

执行成功后，通常可以在 outputdir 下看到以下结果：

- outputdir\imu.dat：设备 IMU 数据。
- outputdir\images\insv.dat：INSV 提取出的 IMU/元数据文件。
- outputdir\images\cam2\：导出的外置相机图像目录。
- outputdir\images\ImgPose.txt：每张图像对应的位姿结果。
- outputdir\images\rig.json：相机 rig 配置文件。

其中 ImgPose.txt 和 rig.json 是否生成，取决于 trajectoryfile 与 calibfile 是否有效。
