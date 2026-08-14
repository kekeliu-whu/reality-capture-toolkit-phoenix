from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


if os.environ.get("CONVERT_COLMAP_LIMIT_NATIVE_THREADS", "1") != "0":
    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(env_name, "1")

import cv2
import laspy
import numpy as np
import pycolmap
from PIL import Image
import torch


MASK_EXT = ".png"
DEPTH_EXT = ".png"
DEPTH_TILE_SIZE = 16
DEFAULT_DEPTH_CHUNK_SIZE = 10_000_000
DEFAULT_MASK_EXPAND_PIXELS = 8
ROTATION_MATRIX_ATOL = 1e-5
MAX_SPLAT_ELEMENTS = 8_000_000


@dataclass(frozen=True)
class FaceSpec:
    name: str
    rotation_face_to_source: np.ndarray


@dataclass(frozen=True)
class FaceIntrinsics:
    width: int
    height: int
    focal: float
    cx: float
    cy: float


@dataclass(frozen=True)
class PcdHeader:
    fields: tuple[str, ...]
    sizes: tuple[int, ...]
    types: tuple[str, ...]
    counts: tuple[int, ...]
    points: int
    data: str
    data_offset: int


@dataclass(frozen=True)
class FaceExportPlan:
    face_name: str
    image_id: int
    camera_id: int
    output_name: str
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class ImageJob:
    index: int
    total: int
    image_name: str
    source_camera_id: int
    split_source_camera: bool
    source_rotation: np.ndarray
    source_translation: np.ndarray
    point3d_ids: np.ndarray
    source_projected_points: np.ndarray
    world_xyz: np.ndarray
    face_plans: tuple[FaceExportPlan, ...]


@dataclass(frozen=True)
class FaceExportResult:
    face_name: str
    image_id: int
    camera_id: int
    output_name: str
    rotation: np.ndarray
    translation: np.ndarray
    point3d_ids: np.ndarray
    projected_points: np.ndarray
    depth_name: str | None = None
    depth_metadata: dict[str, float | int | str] | None = None


@dataclass(frozen=True)
class ImageExportResult:
    index: int
    image_name: str
    face_results: tuple[FaceExportResult, ...]


@dataclass(frozen=True)
class ExportConfig:
    image_dir: Path
    mask_dir: Path | None
    img_out: Path
    mask_out: Path
    depth_out: Path | None
    depth_colorized_out: Path | None
    image_ext: str
    jpeg_quality: int
    depth_mode: str
    depth_scale: float
    depth_voxel_size: float
    depth_max_distance: float | None
    gpu_chunk_points: int


def ensure_even_size(size: int) -> int:
    size = max(int(size), 2)
    return size if size % 2 == 0 else size + 1


def validate_rotation_matrix(name: str, rotation: np.ndarray, atol: float = ROTATION_MATRIX_ATOL) -> None:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix for {name} must be 3x3, got {matrix.shape}.")

    det = float(np.linalg.det(matrix))
    orthogonality_error = matrix.T @ matrix - np.eye(3, dtype=np.float64)
    max_orthogonality_error = float(np.max(np.abs(orthogonality_error)))
    if not np.isfinite(det) or abs(det - 1.0) > atol or max_orthogonality_error > atol:
        raise ValueError(
            f"Invalid rotation matrix for {name}: det={det:.8f}, "
            f"max|R^T R - I|={max_orthogonality_error:.8e}"
        )


def build_face_specs() -> tuple[FaceSpec, ...]:
    sq2 = np.sqrt(2.0)
    sq3 = np.sqrt(3.0)
    sq6 = np.sqrt(6.0)

    v1 = np.array([1.0 / sq2, 1.0 / sq6, 1.0 / sq3], dtype=np.float64)
    v2 = np.array([-1.0 / sq2, 1.0 / sq6, 1.0 / sq3], dtype=np.float64)
    v3 = np.array([0.0, -np.sqrt(2.0 / 3.0), 1.0 / sq3], dtype=np.float64)

    face_specs = (
        FaceSpec("face0", np.column_stack((v2, v3, v1))),
        FaceSpec("face1", np.column_stack((v3, v1, v2))),
        FaceSpec("face2", np.column_stack((v1, v2, v3))),
    )
    for face_spec in face_specs:
        validate_rotation_matrix(face_spec.name, face_spec.rotation_face_to_source)
    return face_specs


def get_camera_model_name(source_camera: pycolmap.Camera) -> str:
    model_name = getattr(source_camera, "model_name", getattr(source_camera, "model", ""))
    return str(model_name).upper()


def is_fisheye_camera(source_camera: pycolmap.Camera) -> bool:
    return "FISHEYE" in get_camera_model_name(source_camera)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COLMAP model to a symmetric 3-camera setup with automatic masking."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=R"Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\xsfm\sparse\0",
        help="Input COLMAP model directory.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=R"Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\images",
        help="Source image directory.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Optional source mask directory. Masks are matched by image-relative path.",
    )
    parser.add_argument(
        "--no-source-masks",
        dest="mask_dir",
        action="store_const",
        const=None,
        help="Disable per-image source masks and generate masks from remapped black borders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=R"Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\cubemap_colmap",
        help="Output directory.",
    )
    parser.add_argument(
        "--point-cloud-path",
        type=Path,
        default=R"Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\small_plane_refined.las",
        help="Point cloud used for depth rendering.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        help="Image extension for exported images. Masks are always written as PNG.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.92,
        help="Legacy compatibility option; ignored by the current black-border mask generation.",
    )
    parser.add_argument(
        "--mask-expand-pixels",
        type=int,
        default=DEFAULT_MASK_EXPAND_PIXELS,
        help="Expand the generated black-border mask by this many output pixels.",
    )
    parser.add_argument(
        "--model-format",
        choices=("text", "binary", "both"),
        default="binary",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--image-step",
        type=int,
        default=2,
        help="After sorting source images by name, export one image every N images.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="How many source-image workers to run concurrently for export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--side-short-fov",
        type=float,
        default=40.0,
        help="Legacy compatibility option; ignored by the current symmetric 3-camera export.",
    )
    parser.add_argument(
        "--side-long-fov",
        type=float,
        default=90.0,
        help="Legacy compatibility option; ignored by the current symmetric 3-camera export.",
    )
    parser.add_argument(
        "--side-angle-degrees",
        type=float,
        default=70.0,
        help="Legacy compatibility option; ignored by the current symmetric 3-camera export.",
    )
    parser.add_argument(
        "--depth-mode",
        choices=("dense", "sparse"),
        default="dense",
        help="Depth export mode. Dense writes the visible splat winner layer; sparse keeps only visible splat centers.",
    )
    depth_mode_group = parser.add_mutually_exclusive_group()
    depth_mode_group.add_argument(
        "--generate-depths",
        dest="generate_depths",
        action="store_true",
        help="Generate depth maps (default behavior).",
    )
    depth_mode_group.add_argument(
        "--skip-depths",
        dest="generate_depths",
        action="store_false",
        help="Skip depth-map generation.",
    )
    parser.set_defaults(generate_depths=True)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.25,
        help="Scale factor for depth-map resolution and K relative to the exported face image.",
    )
    parser.add_argument(
        "--depth-voxel-size",
        type=float,
        default=0.05,
        help="Voxel size in meters used to downsample the point cloud before rendering depth.",
    )
    parser.add_argument(
        "--depth-max-distance",
        type=float,
        default=30.0,
        help="Discard depth points whose center distance exceeds this many meters before rendering. Set to 0 to disable.",
    )
    parser.add_argument(
        "--gpu-chunk-points",
        type=int,
        default=3_000_000,
        help="How many visible spheres each CUDA depth worker processes per rendering batch.",
    )
    args = parser.parse_args()
    if args.image_step < 1:
        parser.error("--image-step must be >= 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.mask_expand_pixels < 0:
        parser.error("--mask-expand-pixels must be >= 0")
    args.skip_depths = not args.generate_depths
    return args


def pixels_from_full_fov(focal: float, fov_degrees: float) -> int:
    return max(int(round(2.0 * focal * np.tan(np.deg2rad(fov_degrees / 2.0)))), 1)


def build_face_intrinsics(
    face_specs: tuple[FaceSpec, ...],
    shared_focal: float,
) -> dict[str, FaceIntrinsics]:
    size = ensure_even_size(pixels_from_full_fov(shared_focal, 90.0))

    intrinsics_by_name: dict[str, FaceIntrinsics] = {}
    for face_spec in face_specs:
        intrinsics_by_name[face_spec.name] = FaceIntrinsics(
            width=size,
            height=size,
            focal=shared_focal,
            cx=size / 2.0,
            cy=size / 2.0,
        )
    return intrinsics_by_name


def build_scaled_intrinsics(
    per_camera_intrinsics: dict[int, dict[str, FaceIntrinsics]], scale: float
) -> dict[int, dict[str, FaceIntrinsics]]:
    scaled_intrinsics: dict[int, dict[str, FaceIntrinsics]] = {}
    for camera_id, intrinsics_by_face in per_camera_intrinsics.items():
        scaled_intrinsics[camera_id] = {}
        for face_name, intrinsics in intrinsics_by_face.items():
            width = ensure_even_size(int(round(intrinsics.width * scale)))
            height = ensure_even_size(int(round(intrinsics.height * scale)))
            scaled_intrinsics[camera_id][face_name] = FaceIntrinsics(
                width=width,
                height=height,
                focal=intrinsics.focal * scale,
                cx=width / 2.0,
                cy=height / 2.0,
            )
    return scaled_intrinsics


def parse_pcd_header(pcd_path: Path) -> PcdHeader:
    header_map: dict[str, list[str]] = {}
    with pcd_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected end of file while reading PCD header: {pcd_path}")

            decoded = line.decode("ascii", errors="strict").strip()
            if not decoded or decoded.startswith("#"):
                continue

            tokens = decoded.split()
            key, values = tokens[0].upper(), tokens[1:]
            header_map[key] = values
            if key == "DATA":
                return PcdHeader(
                    fields=tuple(header_map["FIELDS"]),
                    sizes=tuple(int(v) for v in header_map["SIZE"]),
                    types=tuple(header_map["TYPE"]),
                    counts=tuple(int(v) for v in header_map.get("COUNT", ["1"] * len(header_map["FIELDS"]))),
                    points=int(header_map.get("POINTS", header_map["WIDTH"])[0]),
                    data=values[0].lower(),
                    data_offset=handle.tell(),
                )


def build_pcd_dtype(header: PcdHeader) -> np.dtype:
    dtype_fields = []
    for field_name, field_size, field_type, field_count in zip(
        header.fields, header.sizes, header.types, header.counts
    ):
        type_key = (field_type.upper(), field_size)
        if type_key == ("F", 4):
            base_dtype = np.float32
        elif type_key == ("F", 8):
            base_dtype = np.float64
        elif type_key == ("U", 1):
            base_dtype = np.uint8
        elif type_key == ("U", 2):
            base_dtype = np.uint16
        elif type_key == ("U", 4):
            base_dtype = np.uint32
        elif type_key == ("I", 1):
            base_dtype = np.int8
        elif type_key == ("I", 2):
            base_dtype = np.int16
        elif type_key == ("I", 4):
            base_dtype = np.int32
        else:
            raise ValueError(f"Unsupported PCD field type: {(field_type, field_size)}")

        if field_count == 1:
            dtype_fields.append((field_name, base_dtype))
        else:
            dtype_fields.append((field_name, base_dtype, (field_count,)))
    return np.dtype(dtype_fields)


def build_row_view(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    row_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    return contiguous.view(row_dtype).reshape(-1)


def aggregate_voxel_centroids(
    xyz: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voxels = np.floor(xyz / voxel_size).astype(np.int64)
    voxel_rows = build_row_view(voxels)
    _, unique_indices, inverse, counts = np.unique(
        voxel_rows,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    voxel_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(voxel_sums, inverse, xyz.astype(np.float64, copy=False))
    return (
        voxels[unique_indices],
        voxel_sums,
        counts.astype(np.int64, copy=False),
    )


def append_downsampled_chunk(
    xyz: np.ndarray,
    voxel_size: float,
    candidate_voxels: list[np.ndarray],
    candidate_sums: list[np.ndarray],
    candidate_counts: list[np.ndarray],
) -> None:
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    if xyz.size == 0:
        return

    chunk_voxels, chunk_sums, chunk_counts = aggregate_voxel_centroids(xyz, voxel_size)
    candidate_voxels.append(chunk_voxels)
    candidate_sums.append(chunk_sums)
    candidate_counts.append(chunk_counts)


def load_downsampled_point_cloud(
    point_cloud_path: Path,
    voxel_size: float,
    chunk_size: int,
) -> np.ndarray:
    candidate_sums: list[np.ndarray] = []
    candidate_counts: list[np.ndarray] = []
    candidate_voxels: list[np.ndarray] = []

    suffix = point_cloud_path.suffix.lower()
    if suffix in {".las", ".laz"}:
        with laspy.open(point_cloud_path) as las_reader:
            total_points = int(las_reader.header.point_count)
            total_chunks = max((total_points + chunk_size - 1) // chunk_size, 1)
            processed_points = 0

            for chunk_index, points in enumerate(las_reader.chunk_iterator(chunk_size), start=1):
                processed_points += len(points)
                print(
                    f"[Depth] Downsampling point cloud chunk {chunk_index}/{total_chunks} "
                    f"({processed_points}/{total_points} samples)",
                    flush=True,
                )
                xyz = np.column_stack(
                    (
                        np.asarray(points.x, dtype=np.float64),
                        np.asarray(points.y, dtype=np.float64),
                        np.asarray(points.z, dtype=np.float64),
                    )
                ).astype(np.float32, copy=False)
                append_downsampled_chunk(
                    xyz,
                    voxel_size,
                    candidate_voxels,
                    candidate_sums,
                    candidate_counts,
                )
    elif suffix == ".pcd":
        header = parse_pcd_header(point_cloud_path)
        if header.data != "binary":
            raise ValueError(
                f"Only binary PCD is supported, got {header.data!r} from {point_cloud_path}"
            )
        if not {"x", "y", "z"}.issubset(header.fields):
            raise ValueError(f"PCD file does not contain x/y/z fields: {point_cloud_path}")

        point_data = np.memmap(
            point_cloud_path,
            dtype=build_pcd_dtype(header),
            mode="r",
            offset=header.data_offset,
            shape=(header.points,),
        )
        total_chunks = max((header.points + chunk_size - 1) // chunk_size, 1)

        for chunk_index, start in enumerate(range(0, header.points, chunk_size), start=1):
            end = min(start + chunk_size, header.points)
            print(
                f"[Depth] Downsampling point cloud chunk {chunk_index}/{total_chunks} "
                f"({end}/{header.points} samples)",
                flush=True,
            )
            xyz = np.empty((end - start, 3), dtype=np.float32)
            xyz[:, 0] = point_data["x"][start:end]
            xyz[:, 1] = point_data["y"][start:end]
            xyz[:, 2] = point_data["z"][start:end]
            append_downsampled_chunk(
                xyz,
                voxel_size,
                candidate_voxels,
                candidate_sums,
                candidate_counts,
            )
    else:
        raise ValueError(
            f"Unsupported point-cloud format {point_cloud_path.suffix!r}; expected .las, .laz, or .pcd"
        )

    if not candidate_voxels:
        return np.empty((0, 3), dtype=np.float32)

    print("[Depth] Merging downsampled chunks", flush=True)
    merged_voxels = np.concatenate(candidate_voxels, axis=0)
    merged_sums = np.concatenate(candidate_sums, axis=0)
    merged_counts = np.concatenate(candidate_counts, axis=0)
    merged_row_view = build_row_view(merged_voxels)
    _, inverse = np.unique(merged_row_view, return_inverse=True)
    voxel_count = int(np.max(inverse)) + 1 if inverse.size != 0 else 0
    final_sums = np.zeros((voxel_count, 3), dtype=np.float64)
    final_counts = np.zeros((voxel_count,), dtype=np.int64)
    np.add.at(final_sums, inverse, merged_sums)
    np.add.at(final_counts, inverse, merged_counts)
    return (final_sums / final_counts[:, None]).astype(np.float32, copy=False)


def select_visible_depth_points(
    points_in_camera: np.ndarray,
    intrinsics: FaceIntrinsics,
    voxel_size: float,
    max_distance: float | None = None,
) -> np.ndarray:
    if points_in_camera.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    voxel_circumradius = float(voxel_size) * float(np.sqrt(3.0)) * 0.5
    z_values = points_in_camera[:, 2]
    valid = z_values > -voxel_circumradius
    visible_points = points_in_camera[valid]
    if visible_points.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if max_distance is not None and max_distance > 0.0:
        max_distance_sq = float(max_distance + voxel_circumradius) ** 2
        distance_sq = np.sum(visible_points * visible_points, axis=1)
        visible_points = visible_points[distance_sq <= max_distance_sq]
        if visible_points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

    x_min = (0.0 - intrinsics.cx) / intrinsics.focal
    x_max = (intrinsics.width - intrinsics.cx) / intrinsics.focal
    y_min = (0.0 - intrinsics.cy) / intrinsics.focal
    y_max = (intrinsics.height - intrinsics.cy) / intrinsics.focal
    left_margin = voxel_circumradius * np.sqrt(1.0 + x_min * x_min)
    right_margin = voxel_circumradius * np.sqrt(1.0 + x_max * x_max)
    top_margin = voxel_circumradius * np.sqrt(1.0 + y_min * y_min)
    bottom_margin = voxel_circumradius * np.sqrt(1.0 + y_max * y_max)
    frustum_mask = (
        (visible_points[:, 0] - x_min * visible_points[:, 2] >= -left_margin)
        & (-visible_points[:, 0] + x_max * visible_points[:, 2] >= -right_margin)
        & (visible_points[:, 1] - y_min * visible_points[:, 2] >= -top_margin)
        & (-visible_points[:, 1] + y_max * visible_points[:, 2] >= -bottom_margin)
    )
    visible_points = visible_points[frustum_mask]
    if visible_points.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    far_enough = visible_points[:, 2] > voxel_circumradius
    near_points = visible_points[~far_enough]
    far_points = visible_points[far_enough]
    if far_points.size != 0:
        u = intrinsics.focal * far_points[:, 0] / far_points[:, 2] + intrinsics.cx
        v = intrinsics.focal * far_points[:, 1] / far_points[:, 2] + intrinsics.cy
        radius_pixels = intrinsics.focal * voxel_circumradius / far_points[:, 2]
        intersects = (
            (u + radius_pixels >= 0.0)
            & (u - radius_pixels < intrinsics.width)
            & (v + radius_pixels >= 0.0)
            & (v - radius_pixels < intrinsics.height)
        )
        far_points = far_points[intersects]

    if near_points.size != 0 and far_points.size != 0:
        visible_points = np.concatenate((far_points, near_points), axis=0)
    elif near_points.size != 0:
        visible_points = near_points
    else:
        visible_points = far_points

    return np.ascontiguousarray(visible_points, dtype=np.float32)


def filter_sparse_projectable_points(
    points_in_camera: np.ndarray,
) -> np.ndarray:
    if points_in_camera.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Sparse depth stores projected sample points, so points behind the camera
    # cannot be represented meaningfully.
    valid = points_in_camera[:, 2] > 0.0
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(points_in_camera[valid], dtype=np.float32)


def render_depth_map(
    points_in_camera: np.ndarray,
    intrinsics: FaceIntrinsics,
    voxel_size: float,
    gpu_chunk_points: int,
    max_distance: float | None = None,
    depth_mode: str = "dense",
) -> tuple[np.ndarray, int, int]:
    if depth_mode not in {"dense", "sparse"}:
        raise ValueError(f"Unsupported depth mode: {depth_mode}")

    ensure_cuda_available()

    depth_map = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)
    visible_points = select_visible_depth_points(
        points_in_camera,
        intrinsics,
        voxel_size,
        max_distance=max_distance,
    )
    sparse_points = filter_sparse_projectable_points(visible_points)
    if sparse_points.size == 0:
        return depth_map, 0, 0

    device = torch.device("cuda")
    with torch.no_grad():
        all_points = torch.from_numpy(sparse_points).to(device=device, dtype=torch.float32)
        point_count = all_points.shape[0]
        focal = float(intrinsics.focal)
        cx = float(intrinsics.cx)
        cy = float(intrinsics.cy)
        width = int(intrinsics.width)
        height = int(intrinsics.height)
        max_splat_radius_pixels = int(np.ceil(np.hypot(width, height)))
        voxel_circumradius = float(voxel_size) * float(np.sqrt(3.0)) * 0.5
        projected_u = focal * all_points[:, 0] / all_points[:, 2] + cx
        projected_v = focal * all_points[:, 1] / all_points[:, 2] + cy
        center_pixel_x = torch.round(projected_u - 0.5).to(torch.int64)
        center_pixel_y = torch.round(projected_v - 0.5).to(torch.int64)
        splat_radius = torch.ceil(focal * voxel_circumradius / all_points[:, 2]).to(torch.int64)
        # Once a projected disk exceeds the image diagonal, larger radii are
        # raster-equivalent for our winner-layer approximation but create
        # pathological CUDA allocations for near-camera points.
        splat_radius = torch.clamp(splat_radius, min=0, max=max_splat_radius_pixels)
        flat_depth = torch.full((height * width,), torch.inf, device=device, dtype=torch.float32)
        disk_offsets_by_radius = {}

        for start in range(0, point_count, gpu_chunk_points):
            end = min(start + gpu_chunk_points, point_count)
            batch_center_x = center_pixel_x[start:end]
            batch_center_y = center_pixel_y[start:end]
            batch_radius = splat_radius[start:end]
            batch_depth = all_points[start:end, 2]

            for radius_tensor in torch.unique(batch_radius):
                radius_pixels = int(max(radius_tensor.item(), 0))
                radius_mask = batch_radius == radius_tensor
                if not bool(radius_mask.any()):
                    continue

                if radius_pixels not in disk_offsets_by_radius:
                    coords = torch.arange(-radius_pixels, radius_pixels + 1, device=device, dtype=torch.int64)
                    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
                    disk_mask = grid_x.square() + grid_y.square() <= radius_pixels * radius_pixels
                    disk_offsets_by_radius[radius_pixels] = (grid_x[disk_mask], grid_y[disk_mask])

                offset_x, offset_y = disk_offsets_by_radius[radius_pixels]
                offset_count = int(offset_x.numel())
                group_indices = torch.nonzero(radius_mask, as_tuple=False).squeeze(1)
                subchunk_size = max(1, MAX_SPLAT_ELEMENTS // max(offset_count, 1))

                for group_start in range(0, int(group_indices.numel()), subchunk_size):
                    point_indices = group_indices[group_start : group_start + subchunk_size]
                    splat_x = batch_center_x[point_indices][:, None] + offset_x[None, :]
                    splat_y = batch_center_y[point_indices][:, None] + offset_y[None, :]
                    in_bounds = (
                        (splat_x >= 0)
                        & (splat_x < width)
                        & (splat_y >= 0)
                        & (splat_y < height)
                    )
                    if not bool(in_bounds.any()):
                        continue

                    linear_indices = (splat_y[in_bounds] * width + splat_x[in_bounds]).to(torch.int64)
                    depth_values = batch_depth[point_indices][:, None].expand(-1, offset_count)[in_bounds]
                    flat_depth.scatter_reduce_(0, linear_indices, depth_values, reduce="amin", include_self=True)

        winner_pixel_count = int(torch.count_nonzero(torch.isfinite(flat_depth)).item())
        if winner_pixel_count == 0:
            return depth_map, 0, 0

        if depth_mode == "dense":
            depth_map = (
                torch.where(torch.isinf(flat_depth), 0.0, flat_depth)
                .reshape(height, width)
                .cpu()
                .numpy()
            )
            return depth_map, winner_pixel_count, winner_pixel_count

        visible_point_mask = torch.zeros((point_count,), device=device, dtype=torch.bool)
        for start in range(0, point_count, gpu_chunk_points):
            end = min(start + gpu_chunk_points, point_count)
            batch_center_x = center_pixel_x[start:end]
            batch_center_y = center_pixel_y[start:end]
            batch_radius = splat_radius[start:end]
            batch_depth = all_points[start:end, 2]
            batch_visible = torch.zeros((end - start,), device=device, dtype=torch.bool)

            for radius_tensor in torch.unique(batch_radius):
                radius_pixels = int(max(radius_tensor.item(), 0))
                radius_mask = batch_radius == radius_tensor
                if not bool(radius_mask.any()):
                    continue

                offset_x, offset_y = disk_offsets_by_radius[radius_pixels]
                offset_count = int(offset_x.numel())
                group_indices = torch.nonzero(radius_mask, as_tuple=False).squeeze(1)
                subchunk_size = max(1, MAX_SPLAT_ELEMENTS // max(offset_count, 1))

                for group_start in range(0, int(group_indices.numel()), subchunk_size):
                    point_indices = group_indices[group_start : group_start + subchunk_size]
                    splat_x = batch_center_x[point_indices][:, None] + offset_x[None, :]
                    splat_y = batch_center_y[point_indices][:, None] + offset_y[None, :]
                    in_bounds = (
                        (splat_x >= 0)
                        & (splat_x < width)
                        & (splat_y >= 0)
                        & (splat_y < height)
                    )
                    if not bool(in_bounds.any()):
                        continue

                    linear_indices = (splat_y[in_bounds] * width + splat_x[in_bounds]).to(torch.int64)
                    depth_values = batch_depth[point_indices][:, None].expand(-1, offset_count)[in_bounds]
                    local_point_indices = point_indices[:, None].expand(-1, offset_count)[in_bounds]
                    winner_splat_mask = depth_values == flat_depth[linear_indices]
                    if bool(winner_splat_mask.any()):
                        batch_visible[local_point_indices[winner_splat_mask]] = True

            visible_point_mask[start:end] = batch_visible

        positive_count = int(torch.count_nonzero(visible_point_mask).item())
        if positive_count == 0:
            return depth_map, winner_pixel_count, 0

        visible_center_x = center_pixel_x[visible_point_mask]
        visible_center_y = center_pixel_y[visible_point_mask]
        visible_depth = all_points[visible_point_mask, 2]
        center_in_bounds = (
            (visible_center_x >= 0)
            & (visible_center_x < width)
            & (visible_center_y >= 0)
            & (visible_center_y < height)
        )
        if not bool(center_in_bounds.any()):
            return depth_map, winner_pixel_count, positive_count

        center_linear_indices = visible_center_y[center_in_bounds] * width + visible_center_x[center_in_bounds]
        center_depth = visible_depth[center_in_bounds]
        flat_sparse_depth = torch.full((height * width,), torch.inf, device=device, dtype=torch.float32)
        flat_sparse_depth.scatter_reduce_(0, center_linear_indices, center_depth, reduce="amin", include_self=True)
        depth_map = torch.where(torch.isinf(flat_sparse_depth), 0.0, flat_sparse_depth).reshape(height, width).cpu().numpy()
        return depth_map, winner_pixel_count, positive_count


def render_dense_depth_map(
    points_in_camera: np.ndarray,
    intrinsics: FaceIntrinsics,
    voxel_size: float,
    gpu_chunk_points: int,
    max_distance: float | None = None,
) -> tuple[np.ndarray, int, int]:
    return render_depth_map(
        points_in_camera,
        intrinsics,
        voxel_size,
        gpu_chunk_points,
        max_distance=max_distance,
        depth_mode="dense",
    )


def render_sparse_depth_map_sparse_only(
    points_in_camera: np.ndarray,
    intrinsics: FaceIntrinsics,
    voxel_size: float,
    gpu_chunk_points: int,
    max_distance: float | None = None,
) -> tuple[np.ndarray, int, int]:
    return render_depth_map(
        points_in_camera,
        intrinsics,
        voxel_size,
        gpu_chunk_points,
        max_distance=max_distance,
        depth_mode="sparse",
    )


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA depth rendering was requested but no CUDA device is available.")


def write_depth(output_path: Path, depth_map: np.ndarray) -> tuple[float, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_mask = depth_map > 0
    depth_mm = np.zeros(depth_map.shape, dtype="<u4")
    if np.any(valid_mask):
        depth_mm[valid_mask] = np.round(depth_map[valid_mask] * 1000.0).astype(np.uint32)
        min_depth = float(depth_map[valid_mask].min())
        max_depth = float(depth_map[valid_mask].max())
    else:
        min_depth = 0.0
        max_depth = 0.0

    rgba = depth_mm.view(np.uint8).reshape(*depth_map.shape, 4)
    Image.fromarray(rgba, mode="RGBA").save(output_path)
    return min_depth, max_depth


def write_colorized_depth(output_path: Path, depth_map: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_mask = depth_map > 0
    colorized_depth = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    if np.any(valid_mask):
        valid_depth = depth_map[valid_mask]
        min_depth = float(valid_depth.min())
        max_depth = float(valid_depth.max())
        normalized = np.zeros(depth_map.shape, dtype=np.uint8)
        if np.isclose(min_depth, max_depth):
            normalized[valid_mask] = 255
        else:
            scaled = (valid_depth - min_depth) / (max_depth - min_depth)
            normalized[valid_mask] = np.round(scaled * 255.0).astype(np.uint8)
        colorized_depth = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colorized_depth[~valid_mask] = 0

    if not cv2.imwrite(str(output_path), colorized_depth):
        raise RuntimeError(f"Failed to write colorized depth image: {output_path}")


def write_json(output_path: Path, content: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def build_depth_metadata(
    image_name: str,
    intrinsics: FaceIntrinsics,
    depth_scale: float,
    voxel_size: float,
    depth_mode: str,
    min_depth: float,
    max_depth: float,
    contributing_voxel_count: int,
    auxiliary_count: int,
) -> dict[str, float | int | str]:
    metadata: dict[str, float | int | str] = {
        "image": image_name,
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.focal),
        "fy": float(intrinsics.focal),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "depth_scale": float(depth_scale),
        "voxel_size_m": float(voxel_size),
        "depth_collision_shape": "projected_disk_splat",
        "depth_encoding": "rgba_uint32_le_mm",
        "depth_unit": "mm",
        "depth_scale_to_meters": 0.001,
        "invalid_depth_value_mm": 0,
        "contributing_voxel_count": int(contributing_voxel_count),
        "min_depth_m": float(min_depth),
        "max_depth_m": float(max_depth),
    }
    if depth_mode == "dense":
        metadata["depth_output_semantics"] = "visible_splat_dense_pixels"
        metadata["filled_pixel_count"] = int(auxiliary_count)
    elif depth_mode == "sparse":
        metadata["depth_output_semantics"] = "visible_splat_center_points"
        metadata["positive_z_center_count"] = int(auxiliary_count)
    else:
        raise ValueError(f"Unsupported depth mode: {depth_mode}")
    return metadata


def export_depth(
    output_path: Path,
    colorized_output_path: Path | None,
    image_name: str,
    points_in_face: np.ndarray,
    intrinsics: FaceIntrinsics,
    depth_scale: float,
    voxel_size: float,
    gpu_chunk_points: int,
    max_distance: float | None,
    depth_mode: str,
) -> dict[str, float | int | str]:
    if depth_mode == "dense":
        depth_map, contributing_voxel_count, auxiliary_count = render_dense_depth_map(
            points_in_face,
            intrinsics=intrinsics,
            voxel_size=voxel_size,
            gpu_chunk_points=gpu_chunk_points,
            max_distance=max_distance,
        )
    elif depth_mode == "sparse":
        depth_map, contributing_voxel_count, auxiliary_count = render_sparse_depth_map_sparse_only(
            points_in_face,
            intrinsics=intrinsics,
            voxel_size=voxel_size,
            gpu_chunk_points=gpu_chunk_points,
            max_distance=max_distance,
        )
    else:
        raise ValueError(f"Unsupported depth mode: {depth_mode}")

    min_depth, max_depth = write_depth(output_path, depth_map)
    if colorized_output_path is not None:
        write_colorized_depth(colorized_output_path, depth_map)
    return build_depth_metadata(
        image_name=image_name,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        voxel_size=voxel_size,
        depth_mode=depth_mode,
        min_depth=min_depth,
        max_depth=max_depth,
        contributing_voxel_count=contributing_voxel_count,
        auxiliary_count=auxiliary_count,
    )


def export_depth_for_face(
    depth_points_world: np.ndarray,
    face_plan: FaceExportPlan,
    intrinsics: FaceIntrinsics,
    config: ExportConfig,
) -> tuple[str, dict[str, float | int | str]]:
    depth_name = str(Path(face_plan.output_name).with_suffix(DEPTH_EXT))
    points_in_face = depth_points_world @ face_plan.rotation.T + face_plan.translation
    depth_metadata = export_depth(
        output_path=config.depth_out / depth_name,
        colorized_output_path=(
            config.depth_colorized_out / depth_name
            if config.depth_colorized_out is not None
            else None
        ),
        image_name=face_plan.output_name,
        points_in_face=points_in_face,
        intrinsics=intrinsics,
        depth_scale=config.depth_scale,
        voxel_size=config.depth_voxel_size,
        gpu_chunk_points=config.gpu_chunk_points,
        max_distance=config.depth_max_distance,
        depth_mode=config.depth_mode,
    )
    return depth_name, depth_metadata


def build_remap_tables(
    reconstruction: pycolmap.Reconstruction,
    face_specs: tuple[FaceSpec, ...],
    per_camera_intrinsics: dict[int, dict[str, FaceIntrinsics]],
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    remap_tables = {}
    for camera_id in sorted(per_camera_intrinsics.keys()):
        source_camera = reconstruction.camera(camera_id)
        remap_tables[camera_id] = {}
        for face_spec in face_specs:
            intrinsics = per_camera_intrinsics[camera_id][face_spec.name]
            grid_x, grid_y = np.meshgrid(np.arange(intrinsics.width), np.arange(intrinsics.height))
            rays = np.stack(((grid_x - intrinsics.cx) / intrinsics.focal, 
                             (grid_y - intrinsics.cy) / intrinsics.focal, 
                             np.ones_like(grid_x)), axis=-1).reshape(-1, 3)
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
            source_rays = rays @ face_spec.rotation_face_to_source.T
            source_pixels = source_camera.img_from_cam(source_rays)
            map_x = source_pixels[:, 0].reshape(intrinsics.height, intrinsics.width).astype(np.float32)
            map_y = source_pixels[:, 1].reshape(intrinsics.height, intrinsics.width).astype(np.float32)
            remap_tables[camera_id][face_spec.name] = (map_x, map_y)
    return remap_tables


def assign_faces(points_in_source: np.ndarray, face_specs: tuple[FaceSpec, ...]) -> np.ndarray:
    forward_directions = np.stack([f.rotation_face_to_source[:, 2] for f in face_specs], axis=0)
    scores = (points_in_source / np.linalg.norm(points_in_source, axis=1, keepdims=True)) @ forward_directions.T
    return np.argmax(scores, axis=1)


def write_image(output_path: Path, image: np.ndarray, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if output_path.suffix.lower() in [".jpg", ".jpeg"] else []
    success, encoded = cv2.imencode(output_path.suffix, image, params)
    if not success:
        raise RuntimeError(f"Failed to encode image: {output_path}")
    output_path.write_bytes(encoded.tobytes())


def build_linked_points2d(
    projected_points: np.ndarray,
    point3d_ids: np.ndarray,
) -> list[pycolmap.Point2D]:
    if projected_points.shape[0] != point3d_ids.shape[0]:
        raise ValueError(
            "Projected point count and Point3D id count must match: "
            f"{projected_points.shape[0]} != {point3d_ids.shape[0]}"
        )

    points2d = []
    for point, point3d_id in zip(projected_points, point3d_ids):
        point2d = pycolmap.Point2D(point)
        point2d.point3D_id = int(point3d_id)
        points2d.append(point2d)
    return points2d


def load_source_image(source_path: Path) -> np.ndarray | None:
    try:
        encoded = np.frombuffer(source_path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def load_source_mask(mask_path: Path) -> np.ndarray | None:
    try:
        encoded = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)


def find_source_mask_path(mask_dir: Path, image_name: str) -> Path | None:
    relative_image_path = Path(image_name)
    camera_name = relative_image_path.parts[0] if len(relative_image_path.parts) > 1 else ""
    candidates = (
        mask_dir / f"{image_name}{MASK_EXT}",
        mask_dir / relative_image_path.with_suffix(MASK_EXT),
        mask_dir / relative_image_path,
        mask_dir / f"{camera_name}{MASK_EXT}",
        mask_dir / camera_name / f"mask{MASK_EXT}",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_optional_source_mask(mask_dir: Path, image_name: str) -> np.ndarray | None:
    source_mask_path = find_source_mask_path(mask_dir, image_name)
    if source_mask_path is None:
        return None
    source_mask = load_source_mask(source_mask_path)
    if source_mask is None:
        raise FileNotFoundError(f"Failed to read source mask: {source_mask_path}")
    return source_mask


def remap_source_mask(source_mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    if source_mask.ndim == 3:
        source_mask = cv2.cvtColor(source_mask, cv2.COLOR_BGR2GRAY)
    return cv2.remap(
        source_mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def build_full_valid_source_mask(source_pixels: np.ndarray) -> np.ndarray:
    return np.full(source_pixels.shape[:2], 255, dtype=np.uint8)


def build_mask_from_generated_image(generated_image: np.ndarray, expand_pixels: int) -> np.ndarray:
    if generated_image.ndim == 2:
        black_pixels = generated_image == 0
    else:
        black_pixels = np.all(generated_image == 0, axis=2)

    if not np.any(black_pixels):
        return np.zeros(black_pixels.shape, dtype=np.uint8)

    _, labels = cv2.connectedComponents(black_pixels.astype(np.uint8), connectivity=4)
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    invalid_mask = np.isin(labels, border_labels)

    if expand_pixels > 0:
        kernel_size = 2 * expand_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        invalid_mask = cv2.dilate(invalid_mask.astype(np.uint8), kernel, iterations=1) > 0

    return np.where(invalid_mask, 255, 0).astype(np.uint8)


def build_face_masks_from_reference_images(
    reconstruction: pycolmap.Reconstruction,
    image_ids: list[int],
    face_specs: tuple[FaceSpec, ...],
    image_dir: Path,
    remap_tables: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    expand_pixels: int,
) -> dict[tuple[int, str], np.ndarray]:
    reference_images_by_camera: dict[int, int] = {}
    for image_id in image_ids:
        source_image = reconstruction.image(image_id)
        if source_image.camera_id not in remap_tables:
            continue
        reference_images_by_camera.setdefault(source_image.camera_id, image_id)

    face_masks: dict[tuple[int, str], np.ndarray] = {}
    for camera_id, image_id in sorted(reference_images_by_camera.items()):
        source_image = reconstruction.image(image_id)
        source_path = image_dir / source_image.name
        source_pixels = load_source_image(source_path)
        if source_pixels is None:
            raise FileNotFoundError(
                f"Failed to read source image for mask generation: {source_path}"
            )

        for face_spec in face_specs:
            map_x, map_y = remap_tables[camera_id][face_spec.name]
            generated_image = cv2.remap(
                source_pixels,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            face_masks[(camera_id, face_spec.name)] = build_mask_from_generated_image(
                generated_image,
                expand_pixels=expand_pixels,
            )

    return face_masks


def build_image_job(
    reconstruction: pycolmap.Reconstruction,
    source_image_id: int,
    image_index: int,
    total_images: int,
    face_specs: tuple[FaceSpec, ...],
    face_camera_id_map: dict[tuple[int, str], int],
    image_ext: str,
    next_image_id: int,
) -> tuple[ImageJob, int]:
    source_image = reconstruction.image(source_image_id)
    source_camera = reconstruction.camera(source_image.camera_id)
    split_source_camera = is_fisheye_camera(source_camera)
    source_pose = np.asarray(source_image.cam_from_world().matrix(), dtype=np.float64)
    validate_rotation_matrix(f"source image {source_image.name}", source_pose[:, :3])

    point3d_ids_list = [int(point.point3D_id) for point in source_image.points2D if point.has_point3D()]
    source_projected_points_list = [
        np.asarray(point.xy, dtype=np.float64)
        for point in source_image.points2D
        if point.has_point3D()
    ]
    if point3d_ids_list:
        point3d_ids = np.asarray(point3d_ids_list, dtype=np.int64)
        source_projected_points = np.asarray(source_projected_points_list, dtype=np.float64)
        world_xyz = np.asarray(
            [reconstruction.point3D(point3d_id).xyz for point3d_id in point3d_ids_list],
            dtype=np.float64,
        )
    else:
        point3d_ids = np.empty((0,), dtype=np.int64)
        source_projected_points = np.empty((0, 2), dtype=np.float64)
        world_xyz = np.empty((0, 3), dtype=np.float64)

    face_plans = []
    if split_source_camera:
        rel_path = Path(source_image.name).parent.name
        stem = Path(source_image.name).stem
        for face_spec in face_specs:
            face_rot = face_spec.rotation_face_to_source.T @ source_pose[:, :3]
            validate_rotation_matrix(f"{source_image.name} {face_spec.name}", face_rot)
            face_trans = face_spec.rotation_face_to_source.T @ source_pose[:, 3]
            out_name = f"{rel_path}_{face_spec.name}/{stem}{image_ext}"
            face_plans.append(
                FaceExportPlan(
                    face_name=face_spec.name,
                    image_id=next_image_id,
                    camera_id=face_camera_id_map[(source_image.camera_id, face_spec.name)],
                    output_name=out_name,
                    rotation=np.asarray(face_rot, dtype=np.float64),
                    translation=np.asarray(face_trans, dtype=np.float64),
                )
            )
            next_image_id += 1
    else:
        face_plans.append(
            FaceExportPlan(
                face_name="original",
                image_id=next_image_id,
                camera_id=face_camera_id_map[(source_image.camera_id, "original")],
                output_name=source_image.name,
                rotation=np.asarray(source_pose[:, :3], dtype=np.float64),
                translation=np.asarray(source_pose[:, 3], dtype=np.float64),
            )
        )
        next_image_id += 1

    return (
        ImageJob(
            index=image_index,
            total=total_images,
            image_name=source_image.name,
            source_camera_id=source_image.camera_id,
            split_source_camera=split_source_camera,
            source_rotation=np.asarray(source_pose[:, :3], dtype=np.float64),
            source_translation=np.asarray(source_pose[:, 3], dtype=np.float64),
            point3d_ids=point3d_ids,
            source_projected_points=source_projected_points,
            world_xyz=world_xyz,
            face_plans=tuple(face_plans),
        ),
        next_image_id,
    )


def process_image_job(
    job: ImageJob,
    face_specs: tuple[FaceSpec, ...],
    face_masks: dict[tuple[int, str], np.ndarray],
    per_camera_intrinsics: dict[int, dict[str, FaceIntrinsics]],
    depth_intrinsics: dict[int, dict[str, FaceIntrinsics]],
    remap_tables: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    depth_points_world: np.ndarray | None,
    config: ExportConfig,
) -> ImageExportResult:
    source_path = config.image_dir / job.image_name
    source_pixels = load_source_image(source_path)
    if source_pixels is None:
        raise FileNotFoundError(f"Failed to read source image: {source_path}")

    source_mask = None
    if config.mask_dir is not None:
        source_mask = load_optional_source_mask(config.mask_dir, job.image_name)
        if source_mask is None:
            print(
                f"[Mask] Missing source mask for {job.image_name}; using full-valid mask",
                flush=True,
            )
            source_mask = build_full_valid_source_mask(source_pixels)

    if job.split_source_camera:
        projected_by_face: dict[str, tuple[np.ndarray, np.ndarray]] = {
            face_spec.name: (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 2), dtype=np.float64),
            )
            for face_spec in face_specs
        }
    else:
        projected_by_face = {
            "original": (
                np.asarray(job.point3d_ids, dtype=np.int64, copy=False),
                np.asarray(job.source_projected_points, dtype=np.float64, copy=False),
            )
        }

    if job.split_source_camera and job.world_xyz.shape[0] != 0:
        pts_in_src = job.world_xyz @ job.source_rotation.T + job.source_translation
        face_idx = assign_faces(pts_in_src, face_specs)

        for face_index, (face_spec, face_plan) in enumerate(zip(face_specs, job.face_plans)):
            mask = face_idx == face_index
            if not np.any(mask):
                continue
            intrinsics = per_camera_intrinsics[job.source_camera_id][face_plan.face_name]
            points_in_face = pts_in_src[mask] @ face_spec.rotation_face_to_source
            projected = np.column_stack(
                (
                    intrinsics.focal * points_in_face[:, 0] / points_in_face[:, 2] + intrinsics.cx,
                    intrinsics.focal * points_in_face[:, 1] / points_in_face[:, 2] + intrinsics.cy,
                )
            )
            valid = (
                (points_in_face[:, 2] > 0.0)
                & (projected[:, 0] >= 0.0)
                & (projected[:, 0] < intrinsics.width)
                & (projected[:, 1] >= 0.0)
                & (projected[:, 1] < intrinsics.height)
            )
            projected_by_face[face_plan.face_name] = (
                job.point3d_ids[mask][valid],
                projected[valid],
            )

    face_results = []
    for face_plan in job.face_plans:
        if job.split_source_camera:
            map_x, map_y = remap_tables[job.source_camera_id][face_plan.face_name]
            resampled_image = cv2.remap(
                source_pixels,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if source_mask is not None:
                resampled_mask = remap_source_mask(source_mask, map_x, map_y)
            else:
                resampled_mask = face_masks[(job.source_camera_id, face_plan.face_name)]
        else:
            resampled_image = source_pixels
            if source_mask is not None:
                resampled_mask = source_mask
            else:
                resampled_mask = np.zeros(source_pixels.shape[:2], dtype=np.uint8)

        mask_name = str(Path(face_plan.output_name).with_suffix(MASK_EXT))
        if job.split_source_camera:
            write_image(config.img_out / face_plan.output_name, resampled_image, config.jpeg_quality)
        else:
            output_path = config.img_out / face_plan.output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, output_path)
        write_image(config.mask_out / mask_name, resampled_mask, config.jpeg_quality)

        depth_name = None
        depth_metadata = None
        if depth_points_world is not None and config.depth_out is not None:
            if not job.split_source_camera:
                raise NotImplementedError(
                    "Depth export is not supported for passthrough non-fisheye cameras."
                )
            depth_intr = depth_intrinsics[job.source_camera_id][face_plan.face_name]
            depth_name, depth_metadata = export_depth_for_face(
                depth_points_world=depth_points_world,
                face_plan=face_plan,
                intrinsics=depth_intr,
                config=config,
            )

        point3d_ids, projected_points = projected_by_face[face_plan.face_name]
        face_results.append(
            FaceExportResult(
                face_name=face_plan.face_name,
                image_id=face_plan.image_id,
                camera_id=face_plan.camera_id,
                output_name=face_plan.output_name,
                rotation=face_plan.rotation,
                translation=face_plan.translation,
                point3d_ids=point3d_ids,
                projected_points=projected_points,
                depth_name=depth_name,
                depth_metadata=depth_metadata,
            )
        )

    return ImageExportResult(index=job.index, image_name=job.image_name, face_results=tuple(face_results))


def export_images_with_workers(
    reconstruction: pycolmap.Reconstruction,
    image_ids: list[int],
    face_specs: tuple[FaceSpec, ...],
    face_camera_id_map: dict[tuple[int, str], int],
    face_masks: dict[tuple[int, str], np.ndarray],
    per_camera_intrinsics: dict[int, dict[str, FaceIntrinsics]],
    depth_intrinsics: dict[int, dict[str, FaceIntrinsics]],
    remap_tables: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    depth_points_world: np.ndarray | None,
    config: ExportConfig,
    num_workers: int,
) -> list[ImageExportResult]:
    total_images = len(image_ids)
    results_by_index: dict[int, ImageExportResult] = {}
    next_image_id = 1

    def store_result(result: ImageExportResult) -> None:
        results_by_index[result.index] = result
        print(
            f"[Image] Finished {len(results_by_index)}/{total_images}: {result.image_name}",
            flush=True,
        )

    if total_images == 0:
        return []

    if num_workers == 1:
        for image_index, source_image_id in enumerate(image_ids, start=1):
            job, next_image_id = build_image_job(
                reconstruction,
                source_image_id,
                image_index,
                total_images,
                face_specs,
                face_camera_id_map,
                config.image_ext,
                next_image_id,
            )
            result = process_image_job(
                job,
                face_specs,
                face_masks,
                per_camera_intrinsics,
                depth_intrinsics,
                remap_tables,
                depth_points_world,
                config,
            )
            store_result(result)
        return [results_by_index[index] for index in range(1, total_images + 1)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pending_futures = {}
        for image_index, source_image_id in enumerate(image_ids, start=1):
            job, next_image_id = build_image_job(
                reconstruction,
                source_image_id,
                image_index,
                total_images,
                face_specs,
                face_camera_id_map,
                config.image_ext,
                next_image_id,
            )
            future = executor.submit(
                process_image_job,
                job,
                face_specs,
                face_masks,
                per_camera_intrinsics,
                depth_intrinsics,
                remap_tables,
                depth_points_world,
                config,
            )
            pending_futures[future] = job.index

            if len(pending_futures) >= num_workers:
                completed_future = next(iter(as_completed(pending_futures)))
                pending_futures.pop(completed_future)
                store_result(completed_future.result())

        for completed_future in as_completed(pending_futures):
            store_result(completed_future.result())

    return [results_by_index[index] for index in range(1, total_images + 1)]


def main() -> None:
    args = parse_args()
    if args.depth_scale <= 0.0:
        raise ValueError("--depth-scale must be positive.")
    if args.depth_voxel_size <= 0.0:
        raise ValueError("--depth-voxel-size must be positive.")
    if args.depth_max_distance < 0.0:
        raise ValueError("--depth-max-distance must be non-negative.")
    if args.gpu_chunk_points <= 0:
        raise ValueError("--gpu-chunk-points must be positive.")
    if not args.skip_depths:
        ensure_cuda_available()
    if args.mask_dir is not None and not args.mask_dir.exists():
        raise FileNotFoundError(f"Source mask directory not found: {args.mask_dir}")

    if args.num_workers > 1:
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    reconstruction = pycolmap.Reconstruction(str(args.model_dir))
    face_specs = build_face_specs()
    fisheye_camera_ids = {
        camera_id
        for camera_id in reconstruction.cameras
        if is_fisheye_camera(reconstruction.camera(camera_id))
    }
    non_fisheye_camera_ids = set(reconstruction.cameras.keys()) - fisheye_camera_ids
    if not args.skip_depths and non_fisheye_camera_ids:
        raise NotImplementedError(
            "Depth export is only supported when all cameras are fisheye-split. "
            "Use --skip-depths to keep non-fisheye cameras unchanged."
        )
    
    per_camera_intrinsics = {
        cam_id: build_face_intrinsics(
            face_specs,
            reconstruction.camera(cam_id).mean_focal_length(),
        )
        for cam_id in fisheye_camera_ids
    }
    depth_intrinsics = build_scaled_intrinsics(per_camera_intrinsics, args.depth_scale)
    print("[Setup] Building remap tables", flush=True)
    remap_tables = build_remap_tables(reconstruction, face_specs, per_camera_intrinsics)

    image_ids = sorted(
        reconstruction.images.keys(),
        key=lambda image_id: reconstruction.image(image_id).name,
    )
    if args.image_step > 1:
        image_ids = image_ids[::args.image_step]
    if args.limit > 0:
        image_ids = image_ids[:args.limit]

    if args.mask_dir is not None:
        print(f"[Setup] Using per-image source masks: {args.mask_dir}", flush=True)
        face_masks = {}
    else:
        print("[Setup] Building shared masks from reference face images", flush=True)
        face_masks = build_face_masks_from_reference_images(
            reconstruction,
            image_ids,
            face_specs,
            args.image_dir,
            remap_tables,
            args.mask_expand_pixels,
        )

    depth_points_world = None
    if not args.skip_depths:
        if not args.point_cloud_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {args.point_cloud_path}")
        print(f"Loading and downsampling point cloud: {args.point_cloud_path}")
        depth_points_world = load_downsampled_point_cloud(
            args.point_cloud_path,
            voxel_size=args.depth_voxel_size,
            chunk_size=DEFAULT_DEPTH_CHUNK_SIZE,
        )
        print(f"Depth points after downsampling: {depth_points_world.shape[0]}")
    
    img_out, mask_out = args.output_dir / "images", args.output_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    depth_out = None
    depth_colorized_out = None
    if not args.skip_depths:
        depth_out = args.output_dir / "depths"
        depth_out.mkdir(parents=True, exist_ok=True)
        depth_colorized_out = args.output_dir / "depth_colorized"
        depth_colorized_out.mkdir(parents=True, exist_ok=True)

    converted = pycolmap.Reconstruction()
    tracks_by_point3d = defaultdict(list)
    depth_metadata_by_name: dict[str, dict[str, float | int | str]] = {}

    face_camera_id_map = {}
    cam_counter = 1
    for src_cam_id in sorted(reconstruction.cameras.keys()):
        source_camera = reconstruction.camera(src_cam_id)
        if src_cam_id in fisheye_camera_ids:
            for face_spec in face_specs:
                intr = per_camera_intrinsics[src_cam_id][face_spec.name]
                cam = pycolmap.Camera(model="PINHOLE", width=intr.width, height=intr.height,
                                      params=[intr.focal, intr.focal, intr.cx, intr.cy], camera_id=cam_counter)
                converted.add_camera_with_trivial_rig(cam)
                face_camera_id_map[(src_cam_id, face_spec.name)] = cam_counter
                cam_counter += 1
        else:
            cam = pycolmap.Camera(
                model=str(source_camera.model_name),
                width=source_camera.width,
                height=source_camera.height,
                params=[float(value) for value in source_camera.params],
                camera_id=cam_counter,
            )
            converted.add_camera_with_trivial_rig(cam)
            face_camera_id_map[(src_cam_id, "original")] = cam_counter
            cam_counter += 1

    depth_mode_summary = f"cuda/{args.depth_mode}" if not args.skip_depths else "skipped"
    print(
        f"[Image] Selected {len(image_ids)}/{len(reconstruction.images)} source images "
        f"(sorted by name, step={args.image_step}, workers={args.num_workers}, depths={depth_mode_summary})",
        flush=True,
    )
    export_config = ExportConfig(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        img_out=img_out,
        mask_out=mask_out,
        depth_out=depth_out,
        depth_colorized_out=depth_colorized_out,
        image_ext=args.image_ext,
        jpeg_quality=args.jpeg_quality,
        depth_mode=args.depth_mode,
        depth_scale=args.depth_scale,
        depth_voxel_size=args.depth_voxel_size,
        depth_max_distance=None if args.depth_max_distance == 0.0 else args.depth_max_distance,
        gpu_chunk_points=args.gpu_chunk_points,
    )
    image_results = export_images_with_workers(
        reconstruction,
        image_ids,
        face_specs,
        face_camera_id_map,
        face_masks,
        per_camera_intrinsics,
        depth_intrinsics,
        remap_tables,
        depth_points_world,
        export_config,
        args.num_workers,
    )

    for image_result in image_results:
        for face_result in image_result.face_results:
            points2d = build_linked_points2d(
                face_result.projected_points,
                face_result.point3d_ids,
            )
            converted.add_image_with_trivial_frame(
                pycolmap.Image(
                    name=face_result.output_name,
                    camera_id=face_result.camera_id,
                    image_id=face_result.image_id,
                    points2D=points2d,
                ),
                pycolmap.Rigid3d(
                    pycolmap.Rotation3d(face_result.rotation),
                    face_result.translation,
                ),
            )
            for point_index, point3d_id in enumerate(face_result.point3d_ids):
                tracks_by_point3d[int(point3d_id)].append(
                    pycolmap.TrackElement(face_result.image_id, point_index)
                )
            if face_result.depth_name is not None and face_result.depth_metadata is not None:
                depth_metadata_by_name[face_result.depth_name] = face_result.depth_metadata

    # 还原 Point3D
    for pid, track_els in tracks_by_point3d.items():
        src_pt = reconstruction.point3D(pid)
        track = pycolmap.Track()
        for el in track_els: track.add_element(el)
        converted.add_point3D_with_id(pid, pycolmap.Point3D(xyz=src_pt.xyz, color=src_pt.color, error=src_pt.error, track=track))

    out_model = args.output_dir / "sparse"
    out_model.mkdir(parents=True, exist_ok=True)
    if args.model_format in ["binary", "both"]: converted.write_binary(str(out_model))
    if args.model_format in ["text", "both"]: converted.write_text(str(out_model))
    if depth_out is not None:
        write_json(args.output_dir / "depth_intrinsics.json", depth_metadata_by_name)
        depth_summary = f"{depth_out} (colorized: {depth_colorized_out})"
    else:
        depth_summary = "skipped"
    print(f"Finished. Images: {img_out}, Masks: {mask_out}, Depths: {depth_summary}")

if __name__ == "__main__":
    main()
