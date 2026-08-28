from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


Array = np.ndarray


def unit(v: Array, fallback: Optional[Array] = None) -> Array:
    n = float(np.linalg.norm(v))
    if n > 1e-12:
        return np.asarray(v, dtype=float) / n
    return np.zeros(3) if fallback is None else np.asarray(fallback, dtype=float)


def quaternion_to_matrix(q_xyzw: Array) -> Array:
    x, y, z, w = unit(np.asarray(q_xyzw, dtype=float), np.array([0, 0, 0, 1]))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def slerp(a: Array, b: Array, t: float) -> Array:
    a = unit(np.asarray(a, dtype=float), np.array([0, 0, 0, 1]))
    b = unit(np.asarray(b, dtype=float), np.array([0, 0, 0, 1]))
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        return unit(a + t * (b - a), a)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


@dataclass(slots=True)
class Pose:
    timestamp: float
    translation: Array
    quaternion_xyzw: Array

    @property
    def rotation(self) -> Array:
        return quaternion_to_matrix(self.quaternion_xyzw)

    def apply(self, point: Array) -> Array:
        return self.rotation @ np.asarray(point, dtype=float) + self.translation

    def inverse_apply(self, point: Array) -> Array:
        return self.rotation.T @ (np.asarray(point, dtype=float) - self.translation)

    @staticmethod
    def interpolate(a: "Pose", b: "Pose", timestamp: float) -> "Pose":
        if b.timestamp <= a.timestamp:
            return a
        t = float(np.clip((timestamp - a.timestamp) / (b.timestamp - a.timestamp), 0, 1))
        return Pose(
            timestamp,
            (1 - t) * a.translation + t * b.translation,
            slerp(a.quaternion_xyzw, b.quaternion_xyzw, t),
        )


@dataclass(slots=True)
class LaserPoint:
    xyz: Array
    timestamp: float
    intensity: float = 0.0
    ring: int = 0
    origin: Array = field(default_factory=lambda: np.zeros(3))
    normal: Optional[Array] = None
    ray_weight: float = 1.0


@dataclass(slots=True)
class ColoredPoint(LaserPoint):
    rgb: Optional[Array] = None
    alpha: int = 255


@dataclass(slots=True)
class Camera:
    """Pinhole camera plus the data needed by the recovered view selector."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_from_camera: Pose
    rolling_shutter_seconds: float = 0.0
    mask: Optional[Array] = None

    def project(self, point_world: Array) -> tuple[float, float, float]:
        p = self.world_from_camera.inverse_apply(point_world)
        if p[2] <= 1e-9:
            return np.nan, np.nan, float(p[2])
        return self.fx * p[0] / p[2] + self.cx, self.fy * p[1] / p[2] + self.cy, float(p[2])

    def inside(self, u: float, v: float, margin: float = 0.0) -> bool:
        return margin <= u < self.width - margin and margin <= v < self.height - margin
