"""DNG/image post-processing reference path.

The binary imports LibRaw/DNG SDK and OpenCV's MergeMertens, GaussianBlur,
medianBlur, fastNlMeansDenoising, addWeighted and JPEG encoder.  Those observed
operations define the pipeline below; raw decoding itself remains delegated to
the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(slots=True)
class ImageConfig:
    preset: str = "high-quality"  # high-quality | fast | plain
    sharpen: float = 1.0
    denoise: str | float = -1.0  # adaptive wavelet, 0/off, positive/manual, nlm, nlm-fused
    exposure_stops: float = 0.0
    white_balance: str = "camera"  # camera | auto | custom | per-pano
    custom_white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hdr_ev_shifts: tuple[float, ...] = (-1.0, 0.0, 1.0)
    max_file_size: int | None = None


def gray_world_white_balance(linear_rgb: np.ndarray) -> np.ndarray:
    means = np.maximum(linear_rgb.reshape(-1, 3).mean(axis=0), 1e-6)
    gains = means.mean() / means
    return np.clip(linear_rgb * gains[None, None, :], 0, 1)


def apply_vignetting(image: np.ndarray, polynomial: Sequence[float]) -> np.ndarray:
    """Apply an inverse radial vignetting polynomial centered in the image."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    x = (xx - (w - 1) / 2) / max(w / 2, 1)
    y = (yy - (h - 1) / 2) / max(h / 2, 1)
    r2 = x * x + y * y
    attenuation = np.zeros_like(r2, dtype=float)
    power = np.ones_like(r2, dtype=float)
    for coefficient in polynomial:
        attenuation += coefficient * power
        power *= r2
    return np.clip(image / np.maximum(attenuation[..., None], 0.1), 0, 1)


def exposure_fusion(linear_rgb: np.ndarray, ev_shifts: Iterable[float]) -> np.ndarray:
    exposures = []
    for ev in ev_shifts:
        shifted = np.clip(linear_rgb * (2.0 ** ev), 0, 1)
        srgb = np.where(shifted <= 0.0031308, 12.92 * shifted, 1.055 * shifted ** (1 / 2.4) - 0.055)
        exposures.append(np.rint(255 * srgb[..., ::-1]).astype(np.uint8))
    fused = cv2.createMergeMertens().process(exposures)
    return np.clip(fused, 0, 1)


def denoise(image_bgr: np.ndarray, mode: str | float) -> np.ndarray:
    if mode in (0, 0.0, "0", "off"):
        return image_bgr
    image8 = np.rint(np.clip(image_bgr, 0, 1) * 255).astype(np.uint8)
    if mode in ("nlm", "nlm-fused"):
        h = 5 if mode == "nlm-fused" else 7
        return cv2.fastNlMeansDenoisingColored(image8, None, h, h, 7, 21).astype(np.float32) / 255
    strength = 1.0 if float(mode) < 0 else float(mode)
    # Edge-preserving wavelet equivalent: subtract a bilateral base and shrink
    # small high-frequency coefficients.
    base = cv2.bilateralFilter(image8, 7, 25, 5).astype(np.float32) / 255
    detail = image_bgr - base
    threshold = 0.008 * strength
    detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
    return np.clip(base + detail, 0, 1)


def unsharp_mask(image: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return np.clip(cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0), 0, 1)


def blur_regions(image: np.ndarray, regions: Iterable[tuple[int, int, int, int]], kernel: int = 31) -> np.ndarray:
    result = image.copy()
    kernel = max(3, kernel | 1)
    for x, y, w, h in regions:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        if x1 > x0 and y1 > y0:
            result[y0:y1, x0:x1] = cv2.GaussianBlur(result[y0:y1, x0:x1], (kernel, kernel), 0)
    return result


def process_linear_rgb(
    linear_rgb: np.ndarray,
    cfg: ImageConfig,
    camera_white_balance: tuple[float, float, float] = (1, 1, 1),
    vignetting: Sequence[float] = (1.0,),
    blur_boxes: Iterable[tuple[int, int, int, int]] = (),
) -> np.ndarray:
    image = np.asarray(linear_rgb, dtype=np.float32)
    if cfg.white_balance == "auto":
        image = gray_world_white_balance(image)
    else:
        gains = cfg.custom_white_balance if cfg.white_balance == "custom" else camera_white_balance
        image = np.clip(image * np.asarray(gains)[None, None, :], 0, 1)
    image = apply_vignetting(image, vignetting)
    image = np.clip(image * (2.0 ** cfg.exposure_stops), 0, 1)
    if cfg.preset == "plain":
        image_bgr = image[..., ::-1]
    else:
        shifts = (0.0,) if cfg.preset == "fast" else cfg.hdr_ev_shifts
        image_bgr = exposure_fusion(image, shifts)
        image_bgr = denoise(image_bgr, cfg.denoise)
        image_bgr = unsharp_mask(image_bgr, cfg.sharpen)
    return blur_regions(image_bgr, blur_boxes)


def encode_jpeg(image_bgr: np.ndarray, max_bytes: int | None = None) -> bytes:
    image8 = np.rint(np.clip(image_bgr, 0, 1) * 255).astype(np.uint8)
    qualities = range(95, 34, -5) if max_bytes else (95,)
    last = b""
    for quality in qualities:
        ok, encoded = cv2.imencode(".jpg", image8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        last = encoded.tobytes()
        if max_bytes is None or len(last) <= max_bytes:
            return last
    return last


def write_jpeg(path: str | Path, image_bgr: np.ndarray, cfg: ImageConfig) -> None:
    Path(path).write_bytes(encode_jpeg(image_bgr, cfg.max_file_size))

