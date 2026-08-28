"""Exact-light reference for mapped-space quality and passive-radio export.

The standard production path is the C++ worker.  This module intentionally
keeps the same candidate-grid sampling, projected points, 255 direction bins,
range compression and compact fields for small unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class QualityConfig:
    grid_resolution: float = 0.16666666666666669
    min_num_rays_per_voxel: int = 36
    max_ray_length: float = 50.0
    use_every_nth_point: int = 1


@dataclass(slots=True)
class Ray:
    origin: np.ndarray
    endpoint: np.ndarray


@dataclass(slots=True)
class QualityVoxel:
    key: tuple[int, int, int]
    ray_count: int
    directional_diversity: int
    minimum_range: int


def _key(point: np.ndarray, resolution: float) -> tuple[int, int, int]:
    return tuple(int(value) for value in np.floor(point * (1.0 / resolution)))


def _spherical_directions() -> np.ndarray:
    index = np.arange(255, dtype=np.float32)
    reciprocal_golden = np.float32(0.6180340051651001)
    product = index * reciprocal_golden
    # float64 exactly represents a float32 product/add pair, so the final cast
    # has the same single rounding as the C++ fmaf used by the installed code.
    turns = np.float32(
        index.astype(np.float64) * float(reciprocal_golden)
        - np.trunc(product).astype(np.float64)
    )
    z = np.float32(1.0) - np.float32(
        (np.float32(2.0) * index + np.float32(1.0)) / np.float32(255.0)
    )
    angle = np.float32(6.2831854820251465) * turns
    radius = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - z * z)).astype(np.float32)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z)).astype(np.float32)


_SPHERICAL_DIRECTIONS = _spherical_directions()


def _encode_direction(ray: Ray) -> int:
    direction = np.asarray(ray.origin - ray.endpoint, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
    direction32 = direction.astype(np.float32)
    delta = (_SPHERICAL_DIRECTIONS - direction32).astype(np.float32)
    # Preserve the float32, left-associated squaredNorm used by Eigen.
    distances = np.float32(
        np.float32(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
        + delta[:, 2] * delta[:, 2]
    )
    return int(np.argmin(distances))


def _compress_range(distance: float) -> int:
    index = int(np.float32(distance) * np.float32(100.0))
    index = max(0, min(7000, index))
    compressed = int(np.floor(np.float32(index) ** np.float32(0.625) + np.float32(0.5)))
    return max(1, compressed)


def _traverse(ray: Ray, resolution: float) -> list[tuple[tuple[int, int, int], int]]:
    """Return projected contribution cells and compressed origin ranges.

    Consecutive *sample* cells are de-duplicated before projection. Projected
    cells are deliberately not de-duplicated; one ray can therefore add more
    than one count to a voxel, matching the installed estimator.
    """
    origin = np.asarray(ray.origin, dtype=np.float64)
    endpoint = np.asarray(ray.endpoint, dtype=np.float64)
    delta = endpoint - origin
    length = float(np.linalg.norm(delta))
    direction = delta / length if length > 0.0 else np.zeros(3, dtype=np.float64)
    endpoint_key = _key(endpoint, resolution)
    previous_sample_key: tuple[int, int, int] | None = None
    result: list[tuple[tuple[int, int, int], int]] = []
    distance = resolution * 0.001
    while distance <= length:
        sample = origin + direction * distance
        sample_key = _key(sample, resolution)
        if sample_key != previous_sample_key:
            contribution = endpoint
            if sample_key != endpoint_key:
                center = (np.asarray(sample_key, dtype=np.float64) + 0.5) * resolution
                projection = origin + direction * float(np.dot(center - origin, direction))
                contribution = projection if float(np.dot(projection - origin, delta)) > 0.0 else sample
            result.append((_key(contribution, resolution), _compress_range(np.linalg.norm(contribution - origin))))
            previous_sample_key = sample_key
        distance += resolution / 3.0
    if previous_sample_key != endpoint_key:
        result.append((endpoint_key, _compress_range(length)))
    return result


def _compact_diversity(ranges: list[tuple[int, int]]) -> tuple[int, int]:
    total = np.float32(0.0)
    minimum = 255
    for _, compressed in ranges:
        decompressed = np.float32(np.float32(compressed) ** np.float32(1.6) * np.float32(0.01))
        weight = np.float32(
            decompressed * np.float32(-0.035357143729925156)
            + np.float32(1.0707142353057861)
        )
        weight = np.float32(min(1.0, max(0.01, float(weight))))
        total = np.float32(total + weight)
        minimum = min(minimum, compressed)
    diversity = np.float32(total / np.float32(255.0))
    encoded = 65535 if diversity >= 1.0 else int(np.float32(diversity * np.float32(65535.0)))
    return encoded, minimum


def mapped_space_quality(rays: Iterable[Ray], cfg: QualityConfig = QualityConfig()) -> list[QualityVoxel]:
    """Aggregate and compact mapped-space quality with official field semantics."""
    if cfg.grid_resolution <= 0 or cfg.use_every_nth_point <= 0 or cfg.min_num_rays_per_voxel < 0:
        raise ValueError("invalid mapped-space quality configuration")
    states: dict[tuple[int, int, int], tuple[int, list[tuple[int, int]]]] = {}
    for index, ray in enumerate(rays):
        if index % cfg.use_every_nth_point:
            continue
        direction = _encode_direction(ray)
        for key, compressed_range in _traverse(ray, cfg.grid_resolution):
            count, direction_ranges = states.setdefault(key, (0, []))
            count = min(65535, count + 1)
            for pair_index, (known_direction, known_range) in enumerate(direction_ranges):
                if known_direction == direction:
                    direction_ranges[pair_index] = (known_direction, min(known_range, compressed_range))
                    break
            else:
                direction_ranges.append((direction, compressed_range))
            states[key] = (count, direction_ranges)
    result = []
    for key, (count, direction_ranges) in states.items():
        count = min(65535, round(float(np.float32(count) * np.float32(cfg.use_every_nth_point))))
        if count >= cfg.min_num_rays_per_voxel:
            diversity, minimum_range = _compact_diversity(direction_ranges)
            result.append(QualityVoxel(key, count, diversity, minimum_range))
    return result


def export_radio_observations(records: Iterable[dict], kind: str) -> list[dict]:
    """Normalize Wi-Fi/Bluetooth scans; these pipeline stages are extraction, not SLAM."""
    allowed = {"wifi": ("bssid", "ssid"), "bluetooth": ("address", "name")}
    if kind not in allowed:
        raise ValueError(f"unknown radio kind: {kind}")
    identity, label = allowed[kind]
    result = []
    for record in records:
        if identity not in record or "timestamp" not in record:
            continue
        result.append(
            {
                "timestamp": record["timestamp"],
                "id": record[identity],
                "label": record.get(label, ""),
                "rssi": record.get("rssi"),
                "position": record.get("position"),
            }
        )
    return result
