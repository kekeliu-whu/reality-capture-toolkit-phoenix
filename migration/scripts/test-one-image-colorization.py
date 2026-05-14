"""
用随机选取的相片对点云进行上色。

流程：
1. 读取 camera/ImgPose.txt —— 每行含相片路径和相机位姿（四元数）
2. 读取 colorized.las —— 点云（世界坐标系）
3. 读取 calibration.json / calibration.dat —— 鱼眼相机内参及畸变系数
4. 随机选一张相片，将点云投影到该相片，采样颜色
5. 将带颜色的点云保存为 recolored.las
"""

import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import laspy
import cv2
from scipy.spatial.transform import Rotation as ScipyR

try:
    from proto import calib_pb2
except ImportError:
    calib_pb2 = None


# ============================================================================
# [CONFIG] 命令行参数默认值 - 修改此处便于手动运行
# ============================================================================

DEFAULT_BASE_DIR = R"Z:\rick\dataset\q9000\MT20260430-112900-collect-by-app-only\output"
DEFAULT_TARGET_IMG = "cam1/1749887118_188812.jpg"
DEFAULT_MAX_POINTS = 20_000_000
DEFAULT_SEED = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use one selected image to colorize a point cloud"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing images, calibration, and LAS files",
    )
    parser.add_argument(
        "--target-img",
        type=str,
        default=DEFAULT_TARGET_IMG,
        help="Preferred image path relative to images/",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum number of points to process after downsampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used when the target image is unavailable",
    )
    return parser.parse_args()


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
    for line in lines[1:]:  # 跳过列头
        parts = line.strip().split()
        if len(parts) < 11:
            continue
        poses.append(
            {
                "img_path": parts[0],
                "position": np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                ),
                # 四元数：qx qy qz qw（columns 7-10）
                "quat": np.array(
                    [
                        float(parts[7]),
                        float(parts[8]),
                        float(parts[9]),
                        float(parts[10]),
                    ]
                ),
            }
        )
    return poses


# ──────────────────────────────────────────────
# 3. 读相机标定
# ──────────────────────────────────────────────
def load_calibration(filepath: str) -> dict:
    """
    从 calibration.json 或 calibration.dat 读取相机内参和畸变参数。
    返回 {相机名称: {fx, fy, cx, cy, k1, k2, k3, k4}} 字典。
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            calib_json = json.load(f)

        cameras = {}
        for cam_param in calib_json.get("cameras", []):
            intrinsic = cam_param.get("intrinsic", {})
            distortion = cam_param.get("distortion", {}).get("params", {})
            name = cam_param.get("name", "default")
            cameras[name] = {
                "fx": float(intrinsic["fl_x"]),
                "fy": float(intrinsic["fl_y"]),
                "cx": float(intrinsic["cx"]),
                "cy": float(intrinsic["cy"]),
                "k1": float(distortion.get("k1", 0.0)),
                "k2": float(distortion.get("k2", 0.0)),
                "k3": float(distortion.get("k3", 0.0)),
                "k4": float(distortion.get("k4", 0.0)),
            }
        return cameras

    if calib_pb2 is None:
        raise ImportError(
            "protobuf 标定解析不可用，请安装 proto 模块或改用 calibration.json"
        )

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


def get_camera_calibration(calibs: dict, cam_name: str) -> dict:
    aliases = {
        "cam0": ["/cam0/image_raw", "cam0", "left", "left_camera"],
        "cam1": ["/cam1/image_raw", "cam1", "right", "right_camera"],
    }

    for key in aliases.get(cam_name, [cam_name]):
        if key in calibs:
            return calibs[key]

    available = ", ".join(sorted(calibs.keys()))
    raise KeyError(f"未找到相机 '{cam_name}' 的标定参数，可用相机: {available}")


def resolve_las_path(base_dir: str) -> str:
    candidates = ["map_opt.las", "colorized.las", "rtk_all.las", "map.las"]
    for filename in candidates:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            return path

    las_files = sorted(
        name for name in os.listdir(base_dir) if name.lower().endswith(".las")
    )
    if len(las_files) == 1:
        return os.path.join(base_dir, las_files[0])

    raise FileNotFoundError(f"未找到 LAS 文件，尝试过: {', '.join(candidates)}")


def infer_camera_name(img_rel: str) -> str:
    norm_path = img_rel.replace("\\", "/").lower()
    if "cam0" in norm_path or norm_path.startswith("left/"):
        return "cam0"
    if "cam1" in norm_path or norm_path.startswith("right/"):
        return "cam1"
    raise ValueError(f"无法从图片路径推断相机名称: {img_rel}")


def resolve_image_path(base_dir: str, img_rel: str) -> str:
    norm_path = img_rel.replace("\\", "/")
    return os.path.join(base_dir, "images", *norm_path.split("/"))


def write_single_point_pcd(filepath: str, point_xyz: np.ndarray) -> None:
    x, y, z = [float(value) for value in point_xyz]
    content = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 1
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 1
DATA ascii
{x:.9f} {y:.9f} {z:.9f}
""".format(x=x, y=y, z=z)
    with open(filepath, "w", encoding="ascii") as f:
        f.write(content)


# ──────────────────────────────────────────────
# 4. OpenCV 鱼眼投影
# ──────────────────────────────────────────────
def project_fisheye(
    pts_cam: np.ndarray,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
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
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)

    safe_r = np.where(r > 1e-9, r, 1.0)
    scale = np.where(r > 1e-9, theta_d / safe_r, 0.0)

    u = fl_x * (scale * X) + cx
    v = fl_y * (scale * Y) + cy

    return u, v, valid


# ──────────────────────────────────────────────
# 5. 主流程
# ──────────────────────────────────────────────
def colorize_pointcloud(
    base_dir: str,
    target_img: str,
    max_points: int,
    seed: int | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    # ── 加载点云（步长降采样）──
    las_path = resolve_las_path(base_dir)
    las = laspy.read(las_path)
    total = len(las.points)

    # 控制默认处理规模，避免超大点云在交互环境中被中断。
    step = max(1, (total + max_points - 1) // max_points)
    idx = np.arange(0, total, step)
    las.points = las.points[idx]
    pts_world = np.vstack([las.x, las.y, las.z]).T  # (N, 3)
    world_origin = las.header.offsets.astype(np.float64)
    pts_world_local = pts_world - world_origin
    N = len(pts_world_local)
    print(f"降采样（1/{step}）：{total} → {N} 个点")
    print(f"点云（降采样后，已去偏移）：{N} 个点")
    print(f"  计算原点：{world_origin}")
    print(f"  X: [{pts_world_local[:,0].min():.4f}, {pts_world_local[:,0].max():.4f}]")
    print(f"  Y: [{pts_world_local[:,1].min():.4f}, {pts_world_local[:,1].max():.4f}]")
    print(f"  Z: [{pts_world_local[:,2].min():.4f}, {pts_world_local[:,2].max():.4f}]")

    # ── 加载位姿 & 标定 ──
    poses = load_img_poses(os.path.join(base_dir, "images", "ImgPose.txt"))
    calibs = load_calibration(os.path.join(base_dir, "calibration.dat"))
    print(f"位姿条目：{len(poses)} 条")

    existing_poses = [
        p for p in poses if os.path.exists(resolve_image_path(base_dir, p["img_path"]))
    ]
    print(f"可用图片位姿：{len(existing_poses)} 条")
    if not existing_poses:
        raise FileNotFoundError("ImgPose.txt 中没有任何可用图片文件")

    pose = None
    for p in existing_poses:
        # 规范化路径（处理反斜杠）
        norm_path = p["img_path"].replace("\\", "/")
        if (
            target_img in norm_path
            or norm_path == target_img
            or target_img in p["img_path"]
        ):
            pose = p
            print(f"✓ 在ImgPose.txt中找到对应的pose")
            break
    if pose is None:
        print(f"✗ 警告：未找到 '{target_img}'，使用随机图片")
        pose = random.choice(existing_poses)

    img_rel = pose["img_path"]
    cam_name = infer_camera_name(img_rel)
    img_path = resolve_image_path(base_dir, img_rel)
    pose_position = pose["position"].astype(np.float64)
    t_c2w_local = pose_position - world_origin

    print(f"\n选中图片：{img_rel}  ({cam_name} 相机)")
    print(f"  位置 (x,y,z，已去偏移): {t_c2w_local}")
    print(f"  四元数(qx,qy,qz,qw): {pose['quat']}")

    # 加载图片
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法打开图片：{img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    print(f"  图片尺寸：{W} × {H}")

    calib = get_camera_calibration(calibs, cam_name)

    # ── 坐标变换：世界 → 相机（world_from_camera，即 c2w）──
    # quat 和 position 均为 c2w：P_world = R_c2w @ P_cam + t_c2w
    # 反求：P_cam = R_c2w^T @ (P_world - t_c2w)
    R_c2w = ScipyR.from_quat(pose["quat"]).as_matrix()  # scipy 接受 [qx,qy,qz,qw]
    pts_cam = (R_c2w.T @ (pts_world_local - t_c2w_local).T).T  # (N, 3)

    print(f"\n相机坐标系 Z 范围：[{pts_cam[:,2].min():.4f}, {pts_cam[:,2].max():.4f}]")

    # ── 鱼眼投影 ──
    u, v, valid_front = project_fisheye(
        pts_cam,
        calib["fx"],
        calib["fy"],
        calib["cx"],
        calib["cy"],
        calib["k1"],
        calib["k2"],
        calib["k3"],
        calib["k4"],
    )

    # 像素边界检查
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    valid = valid_front & in_bounds

    hit = int(valid.sum())
    print(f"投影到图片内的点：{hit} / {N}  ({100.0 * hit / N:.1f}%)")

    # ── 采样颜色 ──
    # 未命中的点默认给灰色 (128, 128, 128)
    colors = np.full((N, 3), 128, dtype=np.uint8)
    colors[valid] = img_rgb[vi[valid], ui[valid]]

    # ── 写出带色 LAS ──
    # 转换到支持 RGB 的 Point Format 2
    out_las = laspy.convert(las, point_format_id=2)
    out_las.header.scales = las.header.scales.copy()
    out_las.header.offsets = np.zeros(3, dtype=np.float64)
    out_las.x = pts_world_local[:, 0]
    out_las.y = pts_world_local[:, 1]
    out_las.z = pts_world_local[:, 2]

    # LAS 颜色字段为 uint16（0-65535），8-bit 值左移 8 位
    out_las.red = (colors[:, 0].astype(np.uint16)) << 8
    out_las.green = (colors[:, 1].astype(np.uint16)) << 8
    out_las.blue = (colors[:, 2].astype(np.uint16)) << 8

    out_path = os.path.join(base_dir, "recolored.las")
    out_las.write(out_path)
    print(f"\n已保存（局部坐标，无全局偏移）：{out_path}")

    camera_center_path = os.path.join(base_dir, "camera_center.pcd")
    write_single_point_pcd(camera_center_path, t_c2w_local)
    print(f"已保存相机中心点：{camera_center_path}")

    image_ext = os.path.splitext(img_path)[1] or ".jpg"
    output_image_path = os.path.join(base_dir, f"selected_image{image_ext}")
    shutil.copy2(img_path, output_image_path)
    print(f"已复制选中图片到输出目录：{output_image_path}")


def main() -> int:
    args = parse_args()
    seed = args.seed
    if seed is None:
        seed = int(time.time() * 1e6) % (2**31)

    colorize_pointcloud(
        base_dir=args.base_dir,
        target_img=args.target_img,
        max_points=args.max_points,
        seed=seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
