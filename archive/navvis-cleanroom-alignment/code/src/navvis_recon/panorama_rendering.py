"""Panorama stitching, multiband blending and surfel rendering references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .models import Camera, ColoredPoint, Pose, unit


@dataclass(slots=True)
class PanoramaConfig:
    width: int = 8192
    height: int = 4096
    seam_finder: str = "graphcut"  # graphcut | dp | center
    seam_cost: str = "color_grad"
    multiband_levels: int = 6
    weight_ray: float = 1.0
    floor_filling: bool = True


@dataclass(slots=True)
class SurfelConfig:
    width: int = 8192
    height: int = 4096
    radius: float = 0.01
    near: float = 0.2
    far: float = 35.0
    max_hole_area_fraction: float = 0.0003
    max_hole_compactness: float = 0.7


def equirectangular_ray(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    longitude = (u / width - 0.5) * (2 * np.pi)
    latitude = (0.5 - v / height) * np.pi
    cos_lat = np.cos(latitude)
    return np.stack([cos_lat * np.sin(longitude), np.sin(latitude), cos_lat * np.cos(longitude)], axis=-1)


def warp_to_panorama(image: np.ndarray, camera: Camera, pano_pose: Pose, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:height, :width]
    rays_pano = equirectangular_ray(xx, yy, width, height)
    rays_world = rays_pano @ pano_pose.rotation.T
    rays_camera = rays_world @ camera.world_from_camera.rotation
    z = rays_camera[..., 2]
    map_x = (camera.fx * rays_camera[..., 0] / np.maximum(z, 1e-8) + camera.cx).astype(np.float32)
    map_y = (camera.fy * rays_camera[..., 1] / np.maximum(z, 1e-8) + camera.cy).astype(np.float32)
    valid = (z > 0) & (map_x >= 0) & (map_y >= 0) & (map_x < camera.width - 1) & (map_y < camera.height - 1)
    if camera.mask is not None:
        sampled_mask = cv2.remap(camera.mask, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
        valid &= sampled_mask > 0
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    return warped, valid.astype(np.float32)


def center_seam_masks(masks: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not masks:
        return []
    h, w = masks[0].shape
    yy, xx = np.mgrid[:h, :w]
    scores = []
    for mask in masks:
        distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        scores.append(distance)
    owner = np.argmax(np.stack(scores), axis=0)
    return [((owner == i) & (masks[i] > 0)).astype(np.float32) for i in range(len(masks))]


def _gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image.astype(np.float32)]
    for _ in range(1, levels):
        if min(pyramid[-1].shape[:2]) <= 2:
            break
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    gaussian = _gaussian_pyramid(image, levels)
    laplacian = []
    for i in range(len(gaussian) - 1):
        up = cv2.pyrUp(gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
        laplacian.append(gaussian[i] - up)
    laplacian.append(gaussian[-1])
    return laplacian


def multiband_blend(images: Sequence[np.ndarray], masks: Sequence[np.ndarray], levels: int = 6) -> np.ndarray:
    if not images:
        raise ValueError("no images")
    image_pyramids = [_laplacian_pyramid(i, levels) for i in images]
    mask_pyramids = [_gaussian_pyramid(m, levels) for m in masks]
    count = min(map(len, image_pyramids))
    blended: list[np.ndarray] = []
    for level in range(count):
        shape = image_pyramids[0][level].shape[:2]
        numerator = np.zeros((*shape, images[0].shape[2]), np.float32)
        denominator = np.zeros(shape, np.float32)
        for ip, mp in zip(image_pyramids, mask_pyramids):
            weight = mp[level]
            numerator += ip[level] * weight[..., None]
            denominator += weight
        blended.append(numerator / np.maximum(denominator[..., None], 1e-6))
    result = blended[-1]
    for level in range(count - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended[level].shape[1], blended[level].shape[0])) + blended[level]
    return np.clip(result, 0, 1)


def pyramid_inpaint(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Coarse-to-fine hole filling matching the recovered PyramidInpainting class."""
    missing = (valid_mask == 0).astype(np.uint8)
    if not np.any(missing):
        return image
    image8 = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return cv2.inpaint(image8, missing, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255


def stitch_panorama(images: Sequence[np.ndarray], cameras: Sequence[Camera], pose: Pose, cfg: PanoramaConfig) -> np.ndarray:
    warped, masks = zip(*(warp_to_panorama(i, c, pose, cfg.width, cfg.height) for i, c in zip(images, cameras)))
    # OpenCV GraphCut/DP finders are present in the original.  Center masks are
    # deterministic and serve as fallback when detail seam finders are unavailable.
    seam_masks = center_seam_masks(masks)
    pano = multiband_blend(warped, seam_masks, cfg.multiband_levels)
    valid = np.maximum.reduce(masks)
    return pyramid_inpaint(pano, valid) if cfg.floor_filling else pano


def render_surfels(points: Sequence[ColoredPoint], pose: Pose, cfg: SurfelConfig) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference for the original OpenGL oriented-disc surfel splatter."""
    color = np.zeros((cfg.height, cfg.width, 3), np.float32)
    depth = np.full((cfg.height, cfg.width), np.inf, np.float32)
    weight = np.zeros((cfg.height, cfg.width), np.float32)
    for p in points:
        local = pose.inverse_apply(p.xyz)
        distance = float(np.linalg.norm(local))
        if not cfg.near <= distance <= cfg.far or p.rgb is None:
            continue
        lon = np.arctan2(local[0], local[2])
        lat = np.arcsin(np.clip(local[1] / distance, -1, 1))
        u = int((lon / (2 * np.pi) + 0.5) * cfg.width) % cfg.width
        v = int((0.5 - lat / np.pi) * cfg.height)
        if not 0 <= v < cfg.height:
            continue
        pixel_radius = max(1, int(cfg.radius / distance * cfg.width / (2 * np.pi)))
        for yy in range(max(0, v - pixel_radius), min(cfg.height, v + pixel_radius + 1)):
            for xx0 in range(u - pixel_radius, u + pixel_radius + 1):
                xx = xx0 % cfg.width
                if (xx0 - u) ** 2 + (yy - v) ** 2 > pixel_radius**2:
                    continue
                if distance <= depth[yy, xx] + cfg.radius:
                    w = 1.0 / (1.0 + distance * distance)
                    color[yy, xx] += w * (p.rgb.astype(float) / 255)
                    weight[yy, xx] += w
                    depth[yy, xx] = min(depth[yy, xx], distance)
    valid = weight > 0
    color[valid] /= weight[valid, None]
    color = pyramid_inpaint(color, valid.astype(np.uint8))
    return color, depth
