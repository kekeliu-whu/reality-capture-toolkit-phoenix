"""Reference reconstruction of the non-SLAM laser cloud builder.

Recovered evidence identifies an ordered chain containing NonFiniteXYZFilter,
IntensityNormalizer, IntensityRegionFilter, MultilayerFringeFilter,
RegionFilter, RingRangeFilter and PlaneFilter.  The implementation below keeps
that order and exposes the device-specific geometry as configuration instead of
embedding proprietary G11 calibration numbers.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .models import LaserPoint, Pose, unit


@dataclass(slots=True)
class AxisAlignedRegion:
    minimum: np.ndarray
    maximum: np.ndarray

    def contains(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.minimum) and np.all(p <= self.maximum))


@dataclass(slots=True)
class BuilderConfig:
    min_range_by_ring: dict[int, float] = field(default_factory=dict)
    max_range_by_ring: dict[int, float] = field(default_factory=dict)
    intensity_floor: float = 0.05
    normalize_intensity: bool = True
    unskew: bool = True
    remove_regions: list[AxisAlignedRegion] = field(default_factory=list)
    fringe_regions: list[AxisAlignedRegion] = field(default_factory=list)
    no_motion_translation: float = 0.005
    no_motion_angle_deg: float = 0.05
    plane_distance_threshold: float | None = None


def pose_at(trajectory: Sequence[Pose], timestamp: float) -> Pose:
    if not trajectory:
        raise ValueError("trajectory is empty")
    times = [p.timestamp for p in trajectory]
    i = bisect_right(times, timestamp)
    if i == 0:
        return trajectory[0]
    if i == len(trajectory):
        return trajectory[-1]
    return Pose.interpolate(trajectory[i - 1], trajectory[i], timestamp)


def estimate_ordered_normals(rows: Sequence[Sequence[LaserPoint]]) -> None:
    """Estimate normals from adjacent laser/ring samples in an ordered cloud."""
    for r in range(1, len(rows) - 1):
        width = min(len(rows[r - 1]), len(rows[r]), len(rows[r + 1]))
        for c in range(1, width - 1):
            horizontal = rows[r][c + 1].xyz - rows[r][c - 1].xyz
            vertical = rows[r + 1][c].xyz - rows[r - 1][c].xyz
            n = unit(np.cross(horizontal, vertical))
            ray = rows[r][c].xyz - rows[r][c].origin
            if np.dot(n, ray) > 0:
                n = -n
            rows[r][c].normal = n


def _valid_range(p: LaserPoint, cfg: BuilderConfig) -> bool:
    distance = float(np.linalg.norm(p.xyz - p.origin))
    return cfg.min_range_by_ring.get(p.ring, 0.0) <= distance <= cfg.max_range_by_ring.get(p.ring, np.inf)


def _normalize_intensity(points: list[LaserPoint]) -> None:
    by_ring: dict[int, list[LaserPoint]] = {}
    for p in points:
        by_ring.setdefault(p.ring, []).append(p)
    for ring_points in by_ring.values():
        values = np.array([p.intensity for p in ring_points], dtype=float)
        lo, hi = np.percentile(values, [5, 95]) if len(values) > 1 else (0.0, max(values[0], 1.0))
        span = max(float(hi - lo), 1e-6)
        for p in ring_points:
            p.intensity = float(np.clip((p.intensity - lo) / span, 0, 1))


def _plane_inliers(points: list[LaserPoint], threshold: float, iterations: int = 96) -> set[int]:
    """Small RANSAC plane model matching the PCL model proven in the ELF."""
    if len(points) < 3:
        return set()
    rng = np.random.default_rng(0)
    xyz = np.array([p.xyz for p in points])
    best: np.ndarray = np.array([], dtype=int)
    for _ in range(iterations):
        ids = rng.choice(len(points), 3, replace=False)
        a, b, c = xyz[ids]
        n = unit(np.cross(b - a, c - a))
        if np.linalg.norm(n) < 1e-9:
            continue
        inliers = np.flatnonzero(np.abs((xyz - a) @ n) <= threshold)
        if len(inliers) > len(best):
            best = inliers
    return set(map(int, best))


def build_cloud(
    scans: Iterable[Sequence[LaserPoint]],
    trajectory: Sequence[Pose],
    cfg: BuilderConfig,
    sensor_to_rig: Callable[[LaserPoint], np.ndarray] | None = None,
) -> list[LaserPoint]:
    """Filter, unskew and transform raw points into the map frame.

    SLAM is deliberately absent: ``trajectory`` is an already-estimated input.
    """
    accepted: list[LaserPoint] = []
    for scan in scans:
        for raw in scan:
            if not np.all(np.isfinite(raw.xyz)) or not _valid_range(raw, cfg):
                continue
            p_rig = sensor_to_rig(raw) if sensor_to_rig else raw.xyz
            if any(region.contains(p_rig) for region in cfg.remove_regions):
                continue
            if raw.intensity < cfg.intensity_floor:
                continue
            if any(region.contains(p_rig) for region in cfg.fringe_regions):
                continue
            pose = pose_at(trajectory, raw.timestamp) if cfg.unskew else pose_at(trajectory, scan[0].timestamp)
            accepted.append(
                LaserPoint(
                    xyz=pose.apply(p_rig),
                    timestamp=raw.timestamp,
                    intensity=raw.intensity,
                    ring=raw.ring,
                    origin=pose.translation.copy(),
                    normal=None if raw.normal is None else pose.rotation @ raw.normal,
                    ray_weight=raw.ray_weight,
                )
            )
    if cfg.normalize_intensity and accepted:
        _normalize_intensity(accepted)
    if cfg.plane_distance_threshold:
        # The binary constructs PCL RandomSampleConsensus<ModelPlane>.  Keeping
        # the inliers is useful for calibration/ground extraction modes.
        inliers = _plane_inliers(accepted, cfg.plane_distance_threshold)
        accepted = [p for i, p in enumerate(accepted) if i in inliers]
    return accepted
