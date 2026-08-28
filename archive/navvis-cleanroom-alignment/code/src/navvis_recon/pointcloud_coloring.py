"""Patch-based multi-view point-cloud coloring.

Class and file names recovered from the binary include DirectPatchColorExtractor,
PatchProjector, VoxelRanking, DepthMap, MultiViewColorBlending, WeightMap2D,
AdaptiveBandwidthSelection, GammaEqualizer and several uncolored-point painters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .models import Camera, ColoredPoint, LaserPoint, Pose, unit


@dataclass(slots=True)
class ColoringConfig:
    max_view_distance: float = 35.0
    max_views: int = 5
    depth_tolerance: float = 0.04
    patch_radius: int = 2
    exposure: str = "global"  # none | global
    extrapolation: str = "paint"  # none | paint | discard | fill
    fill_max_radius: float = 0.20
    grayscale: bool = False


@dataclass(slots=True)
class View:
    camera: Camera
    image_bgr: np.ndarray
    depth: np.ndarray | None = None
    gamma: float = 1.0


def render_depth_map(points: Sequence[LaserPoint], camera: Camera) -> np.ndarray:
    depth = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    for p in points:
        u, v, z = camera.project(p.xyz)
        if camera.inside(u, v) and z > 0:
            x, y = int(round(u)), int(round(v))
            if 0 <= x < camera.width and 0 <= y < camera.height:
                depth[y, x] = min(depth[y, x], z)
    finite = np.isfinite(depth).astype(np.uint8)
    # Expand sparse splats, preserving the nearer sample during overlap.
    for _ in range(2):
        nearest = cv2.dilate(np.where(np.isfinite(depth), -depth, -1e9), np.ones((3, 3), np.uint8))
        depth = np.where(finite, depth, -nearest)
        finite = cv2.dilate(finite, np.ones((3, 3), np.uint8))
    depth[finite == 0] = np.inf
    return depth


def visible_in_camera(point: LaserPoint, view: View, tolerance: float) -> tuple[bool, float, float, float]:
    u, v, z = view.camera.project(point.xyz)
    if not view.camera.inside(u, v, 1) or z <= 0:
        return False, u, v, z
    if view.camera.mask is not None and view.camera.mask[int(v), int(u)] == 0:
        return False, u, v, z
    if view.depth is None:
        return True, u, v, z
    observed = float(view.depth[int(v), int(u)])
    return bool(not np.isfinite(observed) or z <= observed + tolerance), u, v, z


def _boundary_weight(camera: Camera, u: float, v: float) -> float:
    edge = min(u, v, camera.width - 1 - u, camera.height - 1 - v)
    return float(np.clip(edge / max(min(camera.width, camera.height) * 0.08, 1), 0, 1))


def rank_view(point: LaserPoint, view: View, u: float, v: float, z: float, cfg: ColoringConfig) -> float:
    if z > cfg.max_view_distance:
        return 0.0
    camera_center = view.camera.world_from_camera.translation
    toward_camera = unit(camera_center - point.xyz)
    incidence = 1.0 if point.normal is None else max(0.0, float(np.dot(point.normal, toward_camera)))
    distance_weight = np.exp(-((z / cfg.max_view_distance) ** 2))
    return float(incidence * incidence * distance_weight * _boundary_weight(view.camera, u, v))


def direct_patch_color(image: np.ndarray, u: float, v: float, radius: int) -> tuple[np.ndarray, float] | None:
    """Project a local patch and return robust RGB plus patch confidence."""
    x, y = int(round(u)), int(round(v))
    if x - radius < 0 or y - radius < 0 or x + radius >= image.shape[1] or y + radius >= image.shape[0]:
        return None
    patch = image[y - radius : y + radius + 1, x - radius : x + radius + 1, ::-1].astype(float) / 255.0
    color = np.median(patch.reshape(-1, 3), axis=0)
    variance = float(np.mean(np.var(patch, axis=(0, 1))))
    return color, 1.0 / (1.0 + 20.0 * variance)


def estimate_global_exposure(samples: Sequence[tuple[int, int, float, float]], view_count: int) -> np.ndarray:
    """Fit log gains from pairwise brightness correspondences.

    The original uses Ceres GammaModel residuals for dynamic range, pairwise
    brightness, parameters, scene brightness and joint variance.  This compact
    least-squares system reproduces the central pairwise term and anchors view 0.
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for a, b, ia, ib in samples:
        if ia <= 1e-4 or ib <= 1e-4:
            continue
        row = np.zeros(view_count)
        row[a], row[b] = 1.0, -1.0
        rows.append(row)
        rhs.append(np.log(ib) - np.log(ia))
    anchor = np.zeros(view_count)
    anchor[0] = 10.0
    rows.append(anchor)
    rhs.append(0.0)
    gains, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return np.exp(np.clip(gains, -1.5, 1.5))


def collect_exposure_pairs(points: Sequence[LaserPoint], views: Sequence[View], cfg: ColoringConfig) -> list[tuple[int, int, float, float]]:
    pairs: list[tuple[int, int, float, float]] = []
    for p in points[:: max(1, len(points) // 5000)]:
        observations: list[tuple[int, float]] = []
        for i, view in enumerate(views):
            visible, u, v, _ = visible_in_camera(p, view, cfg.depth_tolerance)
            if not visible:
                continue
            sample = direct_patch_color(view.image_bgr, u, v, 1)
            if sample:
                observations.append((i, float(np.mean(sample[0]))))
        for a in range(len(observations)):
            for b in range(a + 1, len(observations)):
                pairs.append((observations[a][0], observations[b][0], observations[a][1], observations[b][1]))
    return pairs


def colorize(points: Sequence[LaserPoint], views: Sequence[View], cfg: ColoringConfig) -> list[ColoredPoint]:
    for view in views:
        if view.depth is None:
            view.depth = render_depth_map(points, view.camera)
    if cfg.exposure == "global" and views:
        gains = estimate_global_exposure(collect_exposure_pairs(points, views, cfg), len(views))
        for view, gain in zip(views, gains):
            view.gamma = float(gain)

    result: list[ColoredPoint] = []
    uncolored: list[int] = []
    for point in points:
        candidates: list[tuple[float, np.ndarray]] = []
        for view in views:
            visible, u, v, z = visible_in_camera(point, view, cfg.depth_tolerance)
            if not visible:
                continue
            score = rank_view(point, view, u, v, z, cfg)
            patch = direct_patch_color(view.image_bgr, u, v, cfg.patch_radius)
            if score <= 0 or patch is None:
                continue
            color, confidence = patch
            candidates.append((score * confidence, np.clip(color * view.gamma, 0, 1)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[: cfg.max_views]
        if candidates:
            weights = np.array([x[0] for x in candidates])
            colors = np.array([x[1] for x in candidates])
            rgb = np.rint(255 * np.average(colors, axis=0, weights=weights)).astype(np.uint8)
            if cfg.grayscale:
                rgb[:] = int(round(float(np.dot(rgb, [0.299, 0.587, 0.114]))))
        else:
            rgb = None
            uncolored.append(len(result))
        result.append(
            ColoredPoint(
                xyz=point.xyz.copy(), timestamp=point.timestamp, intensity=point.intensity,
                ring=point.ring, origin=point.origin.copy(), normal=point.normal,
                ray_weight=point.ray_weight, rgb=rgb,
            )
        )
    _paint_uncolored(result, uncolored, cfg)
    return result


def _paint_uncolored(points: list[ColoredPoint], uncolored: list[int], cfg: ColoringConfig) -> None:
    if not uncolored or cfg.extrapolation == "none":
        return
    if cfg.extrapolation == "discard":
        for i in reversed(uncolored):
            points.pop(i)
        return
    known = [i for i, p in enumerate(points) if p.rgb is not None]
    if not known:
        return
    tree = cKDTree(np.array([points[i].xyz for i in known]))
    for i in uncolored:
        k = min(5, len(known))
        distances, ids = tree.query(points[i].xyz, k=k)
        distances, ids = np.atleast_1d(distances), np.atleast_1d(ids)
        if cfg.extrapolation == "fill" and float(distances[0]) > cfg.fill_max_radius:
            continue
        weights = 1.0 / np.maximum(distances, 1e-4)
        colors = np.array([points[known[int(j)]].rgb for j in ids])
        points[i].rgb = np.rint(np.average(colors, axis=0, weights=weights)).astype(np.uint8)
