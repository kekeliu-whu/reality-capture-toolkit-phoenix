"""Voxel surface filtering reconstructed from RTTI, imports and control flow.

The stripped program proves the following order: voxel aggregation, multi-scale
normal estimation, optional global FreespaceOctree construction, density and
adaptive statistical filtering, normal/position smoothing, and tiled output.
The proprietary tile cache is replaced by ordinary arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from .models import LaserPoint, unit


@dataclass(slots=True)
class SurfaceFilterConfig:
    resolution: float = 0.01
    normal_min_radius: float = 0.025
    normal_max_radius: float = 0.15
    normal_levels: int = 6
    smoothing_radius: float = 0.04
    normal_smoothing_radius: float = 0.02
    density_k: int = 16
    max_effective_planar_resolution: float = 0.05
    sor_k: int = 16
    sor_stddev: float = 2.0
    regularize_grid: bool = False
    clean_freespace: bool = True
    freespace_min_intersections: int = 3
    freespace_intersection_hit_ratio: float = 1.5
    freespace_min_origin_distance: float = 0.15
    freespace_max_origin_distance: float = 50.0
    freespace_ray_endpoint_margin: float = 0.04
    freespace_max_incidence_angle_deg: float = 80.0

    @classmethod
    def g11_standard(cls, resolution: float = 0.01) -> "SurfaceFilterConfig":
        """Recovered active branch for G11 + standard.

        The 0.025/0.15/6 normal pyramid and k=16 are direct constants from the
        preset constructor.  Other thresholds are conservative equivalents
        because the stripped struct member names cannot be paired uniquely.
        """
        scale = resolution / 0.01
        return cls(
            resolution=resolution,
            normal_min_radius=max(0.025, 2.5 * resolution),
            normal_max_radius=max(0.15, 15.0 * resolution),
            normal_levels=6,
            smoothing_radius=max(0.04, 4.0 * resolution),
            normal_smoothing_radius=max(0.02, 2.0 * resolution),
            density_k=16,
            sor_k=16,
            freespace_ray_endpoint_margin=0.04 * scale,
        )


@dataclass(slots=True)
class SurfacePoint:
    xyz: np.ndarray
    origin: np.ndarray
    intensity: float
    weight: float
    normal: np.ndarray | None = None


def _key(xyz: np.ndarray, resolution: float) -> tuple[int, int, int]:
    return tuple(np.floor(xyz / resolution).astype(np.int64))


def aggregate_voxels(points: Iterable[LaserPoint], resolution: float, regularize: bool = False) -> list[SurfacePoint]:
    buckets: dict[tuple[int, int, int], list[LaserPoint]] = {}
    for p in points:
        buckets.setdefault(_key(p.xyz, resolution), []).append(p)
    out: list[SurfacePoint] = []
    for key, members in buckets.items():
        weights = np.array([max(p.ray_weight, 1e-6) for p in members])
        weights /= weights.sum()
        xyz = (np.array([p.xyz for p in members]) * weights[:, None]).sum(axis=0)
        if regularize:
            xyz = (np.asarray(key, dtype=float) + 0.5) * resolution
        out.append(
            SurfacePoint(
                xyz=xyz,
                origin=(np.array([p.origin for p in members]) * weights[:, None]).sum(axis=0),
                intensity=float(np.dot([p.intensity for p in members], weights)),
                weight=float(sum(p.ray_weight for p in members)),
            )
        )
    return out


def _normal_from_neighbors(center: np.ndarray, neighbors: np.ndarray) -> tuple[np.ndarray, float]:
    offsets = neighbors - neighbors.mean(axis=0)
    covariance = offsets.T @ offsets / max(len(offsets), 1)
    values, vectors = np.linalg.eigh(covariance)
    normal = unit(vectors[:, int(np.argmin(values))])
    planarity = float(values[0] / max(values.sum(), 1e-12))
    return normal, planarity


def estimate_multiscale_normals(points: list[SurfacePoint], cfg: SurfaceFilterConfig) -> None:
    if len(points) < 3:
        return
    xyz = np.array([p.xyz for p in points])
    tree = cKDTree(xyz)
    radii = np.geomspace(cfg.normal_min_radius, cfg.normal_max_radius, cfg.normal_levels)
    for i, p in enumerate(points):
        selected: tuple[np.ndarray, float] | None = None
        for radius in radii:
            ids = tree.query_ball_point(p.xyz, float(radius))
            if len(ids) < 6:
                continue
            candidate = _normal_from_neighbors(p.xyz, xyz[ids])
            if selected is None or candidate[1] < selected[1]:
                selected = candidate
        if selected:
            n = selected[0]
            if np.dot(n, p.xyz - p.origin) > 0:
                n = -n
            p.normal = n


def density_filter(points: list[SurfacePoint], cfg: SurfaceFilterConfig) -> list[SurfacePoint]:
    if len(points) <= cfg.density_k:
        return points
    xyz = np.array([p.xyz for p in points])
    distances, _ = cKDTree(xyz).query(xyz, k=cfg.density_k + 1)
    effective_resolution = distances[:, -1] / np.sqrt(cfg.density_k)
    return [p for p, spacing in zip(points, effective_resolution) if spacing <= cfg.max_effective_planar_resolution]


def adaptive_statistical_outlier_removal(points: list[SurfacePoint], cfg: SurfaceFilterConfig) -> list[SurfacePoint]:
    """Adaptive SOR: compare local mean neighbor distance at equal density."""
    if len(points) <= cfg.sor_k:
        return points
    xyz = np.array([p.xyz for p in points])
    distances, _ = cKDTree(xyz).query(xyz, k=cfg.sor_k + 1)
    local = distances[:, 1:].mean(axis=1)
    # Median/MAD is robust to mixed indoor densities while retaining the
    # familiar mean+sigma SOR behavior in the central population.
    median = float(np.median(local))
    sigma = 1.4826 * float(np.median(np.abs(local - median)))
    threshold = median + cfg.sor_stddev * max(sigma, cfg.resolution * 0.05)
    return [p for p, d in zip(points, local) if d <= threshold]


def _ray_voxels(origin: np.ndarray, endpoint: np.ndarray, voxel_size: float) -> list[tuple[int, int, int]]:
    """Conservative voxel traversal used in place of the proprietary octree."""
    delta = endpoint - origin
    length = float(np.linalg.norm(delta))
    if length <= voxel_size:
        return []
    steps = max(1, int(ceil(length / (voxel_size * 0.5))))
    samples = origin[None, :] + np.linspace(0, 1, steps, endpoint=False)[:, None] * delta[None, :]
    return list(dict.fromkeys(_key(p, voxel_size) for p in samples))


def clean_freespace(points: list[SurfacePoint], cfg: SurfaceFilterConfig) -> list[SurfacePoint]:
    """Remove occupied voxels contradicted by many laser rays.

    This mirrors recovered FreespaceOctree calls: addPoint, computeCentroidNormals,
    computeIntersections, removeFreespaceVoxels, then compact occupancy conversion.
    """
    occupied: dict[tuple[int, int, int], int] = {_key(p.xyz, cfg.resolution): i for i, p in enumerate(points)}
    hits: dict[tuple[int, int, int], int] = {k: 1 for k in occupied}
    intersections: dict[tuple[int, int, int], int] = {}
    max_angle_cos = np.cos(np.deg2rad(cfg.freespace_max_incidence_angle_deg))
    for p in points:
        ray = p.xyz - p.origin
        distance = float(np.linalg.norm(ray))
        if not cfg.freespace_min_origin_distance <= distance <= cfg.freespace_max_origin_distance:
            continue
        direction = ray / max(distance, 1e-12)
        if p.normal is not None and abs(float(np.dot(p.normal, direction))) < max_angle_cos:
            continue
        endpoint = p.xyz - direction * cfg.freespace_ray_endpoint_margin
        for key in _ray_voxels(p.origin, endpoint, cfg.resolution):
            intersections[key] = intersections.get(key, 0) + 1
    keep: list[SurfacePoint] = []
    for p in points:
        key = _key(p.xyz, cfg.resolution)
        free = intersections.get(key, 0)
        hit = hits.get(key, 1)
        contradicted = free >= cfg.freespace_min_intersections and free / hit >= cfg.freespace_intersection_hit_ratio
        if not contradicted:
            keep.append(p)
    return keep


def smooth_surface(points: list[SurfacePoint], cfg: SurfaceFilterConfig) -> list[SurfacePoint]:
    if not points:
        return []
    xyz = np.array([p.xyz for p in points])
    tree = cKDTree(xyz)
    result: list[SurfacePoint] = []
    sigma = max(cfg.smoothing_radius / 2, 1e-6)
    for i, p in enumerate(points):
        ids = tree.query_ball_point(p.xyz, cfg.smoothing_radius)
        if len(ids) < 3 or p.normal is None:
            result.append(p)
            continue
        q = xyz[ids]
        spatial = np.exp(-np.sum((q - p.xyz) ** 2, axis=1) / (2 * sigma * sigma))
        normals = np.array([points[j].normal if points[j].normal is not None else p.normal for j in ids])
        compatible = np.clip(normals @ p.normal, 0, 1) ** 2
        weights = spatial * compatible
        if weights.sum() <= 1e-12:
            result.append(p)
            continue
        center = (q * weights[:, None]).sum(axis=0) / weights.sum()
        # Tangential smoothing preserves the measured surface along its normal.
        displacement = center - p.xyz
        center -= np.dot(displacement, p.normal) * p.normal
        n = unit((normals * weights[:, None]).sum(axis=0), p.normal)
        result.append(replace(p, xyz=center, normal=n))
    return result


def filter_surface(points: Iterable[LaserPoint], cfg: SurfaceFilterConfig) -> tuple[list[SurfacePoint], list[SurfacePoint]]:
    """Return ``(smoothed, raw_filtered)`` like output-cloud/output-cloud-raw."""
    cloud = aggregate_voxels(points, cfg.resolution, cfg.regularize_grid)
    estimate_multiscale_normals(cloud, cfg)
    cloud = [p for p in cloud if p.normal is not None]
    if cfg.clean_freespace:
        cloud = clean_freespace(cloud, cfg)
    cloud = density_filter(cloud, cfg)
    cloud = adaptive_statistical_outlier_removal(cloud, cfg)
    return smooth_surface(cloud, cfg), cloud
