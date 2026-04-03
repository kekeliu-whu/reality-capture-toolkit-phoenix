"""
用随机选取的相片对点云进行上色。

流程：
1. 读取 camera/ImgPose.txt —— 每行含相片路径和相机位姿（四元数）
2. 读取 colorized.las —— 点云（世界坐标系）
3. 读取 calibration.dat —— 鱼眼相机内参及畸变系数（protobuf格式）
4. 随机选一张相片，将点云投影到该相片，采样颜色
5. 将带颜色的点云保存为 recolored.las
"""

import os
import random
import sys
import time

import numpy as np
import laspy
import cv2
from scipy.spatial.transform import Rotation as ScipyR
from proto import calib_pb2

BASE_DIR = R"D:/output2"
target_img = "cam0/1749886787_819981.jpg"  # 默认图片


# ──────────────────────────────────────────────
# 1. 读 ImgPose.txt
# ──────────────────────────────────────────────
def load_img_poses(filepath: str) -> list[dict]:
    """
    文件格式（空格分隔）：
      index  x  y  z  roll  pitch  yaw  qx  qy  qz  qw  timestamp
    返回含 img_path / position / quat 的字典列表。
    """
    poses = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:          # 跳过列头
        parts = line.strip().split()
        if len(parts) < 11:
            continue
        poses.append({
            "img_path":  parts[0],
            "position":  np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
            # 四元数：qx qy qz qw（columns 7-10）
            "quat":      np.array([float(parts[7]), float(parts[8]),
                                   float(parts[9]), float(parts[10])]),
        })
    return poses


# ──────────────────────────────────────────────
# 3. 读相机标定
# ──────────────────────────────────────────────
def load_calibration(filepath: str) -> dict:
    """
    从protobuf格式的calibration.dat读取相机内参和畸变参数。
    返回 {相机名称: {fx, fy, cx, cy, k1, k2, k3, k4}} 字典。
    """
    with open(filepath, "rb") as f:
        calib_data = f.read()
    
    sensor_calib = calib_pb2.SensorCalib()
    sensor_calib.ParseFromString(calib_data)
    
    cameras = {}
    for cam_param in sensor_calib.camera_param:
        name = cam_param.name if cam_param.name else "default"
        cameras[name] = {
            "fx": cam_param.fx,
            "fy": cam_param.fy,
            "cx": cam_param.cx,
            "cy": cam_param.cy,
            "k1": cam_param.k1,
            "k2": cam_param.k2,
            "k3": cam_param.k3,
            "k4": cam_param.k4,
        }
    return cameras


# ──────────────────────────────────────────────
# 4. OpenCV 鱼眼投影
# ──────────────────────────────────────────────
def project_fisheye(
    pts_cam: np.ndarray,
    fl_x: float, fl_y: float,
    cx: float, cy: float,
    k1: float, k2: float, k3: float, k4: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OpenCV FISHEYE 畸变模型投影。

    输入：
      pts_cam : (N, 3)  相机坐标系下的三维点 (X, Y, Z)，+Z 为相机朝向
    返回：
      u, v   : (N,) 像素坐标（浮点）
      valid  : (N,) bool，Z > 0 且未溢出
    """
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    valid = Z > 0.01

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r, Z)
    t2 = theta * theta
    theta_d = theta * (1.0 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)

    safe_r = np.where(r > 1e-9, r, 1.0)
    scale  = np.where(r > 1e-9, theta_d / safe_r, 0.0)

    u = fl_x * (scale * X) + cx
    v = fl_y * (scale * Y) + cy

    return u, v, valid


# ──────────────────────────────────────────────
# 5. 主流程
# ──────────────────────────────────────────────
def colorize_pointcloud(seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)

    # ── 加载点云（步长降采样）──
    las_path = os.path.join(BASE_DIR, "map.las")
    las = laspy.read(las_path)
    total = len(las.points)
    step = 2  # 每 100 点取 1，快速测试；改为 4 可做 1/4 采样
    idx = np.arange(0, total, step)
    las.points = las.points[idx]
    pts_world = np.vstack([las.x, las.y, las.z]).T          # (N, 3)
    N = len(pts_world)
    print(f"降采样（1/{step}）：{total} → {N} 个点")
    print(f"点云（降采样后）：{N} 个点")
    print(f"  X: [{pts_world[:,0].min():.4f}, {pts_world[:,0].max():.4f}]")
    print(f"  Y: [{pts_world[:,1].min():.4f}, {pts_world[:,1].max():.4f}]")
    print(f"  Z: [{pts_world[:,2].min():.4f}, {pts_world[:,2].max():.4f}]")

    # ── 加载位姿 & 标定 ──
    poses  = load_img_poses(os.path.join(BASE_DIR, "images", "ImgPose.txt"))
    calibs = load_calibration(os.path.join(BASE_DIR, "calibration.dat"))
    print(f"位姿条目：{len(poses)} 条")

    pose = None
    for p in poses:
        # 规范化路径（处理反斜杠）
        norm_path = p["img_path"].replace("\\", "/")
        if target_img in norm_path or norm_path == target_img or target_img in p["img_path"]:
            pose = p
            print(f"✓ 在ImgPose.txt中找到对应的pose")
            break
    if pose is None:
        print(f"✗ 警告：未找到 '{target_img}'，使用随机图片")
        # pose = poses[0]
        pose = random.choice(poses)
    
    img_rel  = pose["img_path"]
    cam_name = "cam0" if "cam0" in img_rel else "cam1"
    img_path = os.path.join(BASE_DIR, "images", img_rel)

    print(f"\n选中图片：{img_rel}  ({cam_name} 相机)")
    print(f"  位置 (x,y,z)  : {pose['position']}")
    print(f"  四元数(qx,qy,qz,qw): {pose['quat']}")

    # 加载图片
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法打开图片：{img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    print(f"  图片尺寸：{W} × {H}")

    calib = calibs[f"/{cam_name}/image_raw"]

    # ── 坐标变换：世界 → 相机（world_from_camera，即 c2w）──
    # quat 和 position 均为 c2w：P_world = R_c2w @ P_cam + t_c2w
    # 反求：P_cam = R_c2w^T @ (P_world - t_c2w)
    R_c2w = ScipyR.from_quat(pose["quat"]).as_matrix()  # scipy 接受 [qx,qy,qz,qw]
    t_c2w = pose["position"]
    pts_cam = (R_c2w.T @ (pts_world - t_c2w).T).T   # (N, 3)

    print(f"\n相机坐标系 Z 范围：[{pts_cam[:,2].min():.4f}, {pts_cam[:,2].max():.4f}]")

    # ── 鱼眼投影 ──
    u, v, valid_front = project_fisheye(
        pts_cam,
        calib["fx"], calib["fy"],
        calib["cx"],   calib["cy"],
        calib["k1"],   calib["k2"], calib["k3"], calib["k4"],
    )

    # 像素边界检查
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    valid     = valid_front & in_bounds

    hit = int(valid.sum())
    print(f"投影到图片内的点：{hit} / {N}  ({100.0 * hit / N:.1f}%)")

    # ── 采样颜色 ──
    # 未命中的点默认给灰色 (128, 128, 128)
    colors = np.full((N, 3), 128, dtype=np.uint8)
    colors[valid] = img_rgb[vi[valid], ui[valid]]

    # ── 写出带色 LAS ──
    # 转换到支持 RGB 的 Point Format 2
    out_las = laspy.convert(las, point_format_id=2)

    # LAS 颜色字段为 uint16（0-65535），8-bit 值左移 8 位
    out_las.red   = (colors[:, 0].astype(np.uint16)) << 8
    out_las.green = (colors[:, 1].astype(np.uint16)) << 8
    out_las.blue  = (colors[:, 2].astype(np.uint16)) << 8

    out_path = os.path.join(BASE_DIR, "recolored.las")
    out_las.write(out_path)
    print(f"\n已保存：{out_path}")


if __name__ == "__main__":
    seed_arg = None
    
    if len(sys.argv) > 1:
        seed_arg = int(sys.argv[1])
    
    if seed_arg is None:
        seed_arg = int(time.time() * 1e6) % (2**31)
    
    colorize_pointcloud(seed=seed_arg)
