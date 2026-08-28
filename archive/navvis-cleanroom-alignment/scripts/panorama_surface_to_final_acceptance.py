#!/usr/bin/env python3
"""Read-only acceptance sidecar for the panorama Surface-to-final path.

The default run verifies frozen exact artifacts and their pinned provenance.
Use --rerun-capture00000 to execute one clean-room 8192x4096 production run
in a temporary directory and compare it byte-for-byte with the vendor JPEG.
The sidecar never builds or edits production code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable
import zlib


DATASET_NAME = "2026-07-21_11.41.12"
CAPTURE = "00000"
EXPECTED_CAPTURE_IDS = [f"{index:05d}" for index in range(34)]

EXPECTED = {
    "clean_binary_sha256": "752cc6266abc6d3824e284daa1037dd6ec0c1383f75d40bae28c75c7bd164256",
    "vendor_binary_sha256": "fde9ee4594d467063354d8b5b5b80d69d88e584b9323bf01c9816c1ad9669d87",
    "vendor_build_id": "24ef8bee2e35486ec1e9922dd5459cd028bbfd20",
    "vendor_version": "05ad952633e1dadaf52fb40c2187932e20f8a5ee",
    "surface_sha256": "61355c566b68f8cc68e917eeb9d300adb631fbeb5dee3614c99105bb1f22a1cd",
    "sensor_frame_sha256": "4fa8149b7fed51a2647423248669d5ae12021902af2b654b84ce77acb590b758",
    "capture_00000_info_sha256": "9261e0e578f6b5ce55d1b4ef9f3c6ac58d33d6a9faa3f39cdf4c7af254232236",
    "all_info_ordered_hash_sha256": "d2494830bc8b485c33573d1409254e0ba5ec9fd3b3a36b21bb8be15d789f2837",
    "capture_00000_camera_ordered_hash_sha256": "29374fe8eb0b6f1e0d7947789ac632a43a0f3118681dc3556eddd3d9cdb9d0e6",
    "camera_mask_ordered_hash_sha256": "f4dd2758720df3431f81ddb05b49e29105ad791ba64e5c88fb37b08cfaa18123",
    "operator_mask_sha256": "05a461e05a4cf6520b051847f0feb34b967d552847580e964ecb8d1bd431601c",
    "all_sparse_1024_ordered_hash_sha256": "fd9c16d6e57e622c2fbe24c0c0766ceaafea981770e1f2a41f5708438c3bc324",
    "all_sparse_1024_total_valid": 1_686_952,
    "all_sparse_1024_min_valid": 38_361,
    "all_sparse_1024_max_valid": 58_692,
    "capture_00000_sparse_8k_sha256": "4c85009531ab5f9fc6065a8122af103c808e0eb9f795aa419892f890b1530c3d",
    "capture_00000_sparse_8k_valid": 68_310,
    "capture_00000_measured_f32_sha256": "d36c11c72896b68ccde0f1ee2f29721b800c5061fbbec4ff28d71176af98c6d3",
    "capture_00000_pcg_f64_sha256": "f6316ecabb0ed816fc7da422d7320037e6090e4b4753d36bc31aab07302f05f1",
    "capture_00000_final_8k_sha256": "37ce1025ea88ce87e0083264af9cf97e7a3b290a01b2aaed140f59137d022fef",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_hash(hashes: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in hashes).encode("ascii")
    return sha256_bytes(payload)


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_build_id(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["readelf", "-n", str(path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", result.stdout)
    return match.group(1).lower() if match else None


def parse_ply_vertex_count(path: Path) -> int | None:
    with path.open("rb") as stream:
        for raw_line in stream:
            line = raw_line.decode("ascii", errors="replace").strip()
            match = re.fullmatch(r"element\s+vertex\s+(\d+)", line)
            if match:
                return int(match.group(1))
            if line == "end_header":
                break
    return None


def png_header(path: Path) -> dict[str, int]:
    with path.open("rb") as stream:
        signature = stream.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not a PNG: {path}")
        length = struct.unpack(">I", stream.read(4))[0]
        chunk_type = stream.read(4)
        if chunk_type != b"IHDR" or length != 13:
            raise ValueError(f"invalid PNG IHDR: {path}")
        values = struct.unpack(">IIBBBBB", stream.read(13))
    keys = (
        "width",
        "height",
        "bit_depth",
        "color_type",
        "compression",
        "filter",
        "interlace",
    )
    return dict(zip(keys, values))


def paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def decode_sparse_png_mm(path: Path) -> tuple[int, int, list[int]]:
    """Decode the frozen 8-bit RGBA sparse depth PNG using only stdlib."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")

    position = 8
    compressed = bytearray()
    header: tuple[int, int, int, int, int, int, int] | None = None
    while position < len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        position += 12 + length
        if chunk_type == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            break

    if header is None:
        raise ValueError(f"missing PNG IHDR: {path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = header
    if (bit_depth, color_type, compression, filtering, interlace) != (8, 6, 0, 0, 0):
        raise ValueError(
            "expected non-interlaced 8-bit RGBA sparse PNG, got "
            f"bit_depth={bit_depth} color_type={color_type} interlace={interlace}"
        )

    raw = zlib.decompress(bytes(compressed))
    bytes_per_pixel = 4
    stride = width * bytes_per_pixel
    expected_size = height * (stride + 1)
    if len(raw) != expected_size:
        raise ValueError(f"unexpected inflated PNG size: {len(raw)} != {expected_size}")

    previous = bytearray(stride)
    offset = 0
    millimetres: list[int] = []
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + stride])
        offset += stride
        for index in range(stride):
            left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            above = previous[index]
            upper_left = (
                previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            )
            if filter_type == 1:
                row[index] = (row[index] + left) & 0xFF
            elif filter_type == 2:
                row[index] = (row[index] + above) & 0xFF
            elif filter_type == 3:
                row[index] = (row[index] + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                row[index] = (row[index] + paeth(left, above, upper_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter {filter_type}")
        # PNG stores R,G,B,A. The renderer encodes low byte in R, high in G.
        millimetres.extend(
            (row[index + 1] << 8) | row[index]
            for index in range(0, stride, bytes_per_pixel)
        )
        previous = row
    return width, height, millimetres


def guard_file(
    path: Path,
    root: Path,
    expected_sha256: str | None,
    errors: list[str],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative(path, root),
        "exists": path.is_file(),
        "expected_sha256": expected_sha256,
    }
    if not path.is_file():
        record["exact"] = False
        errors.append(f"missing file: {path}")
        return record
    actual = sha256_file(path)
    record.update({"size_bytes": path.stat().st_size, "sha256": actual})
    if expected_sha256 is not None:
        record["exact"] = actual == expected_sha256
        if actual != expected_sha256:
            errors.append(f"SHA-256 mismatch: {path}")
    return record


def snapshot_sources(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    snapshot = []
    for path in paths:
        snapshot.append(
            {
                "path": relative(path, root),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return snapshot


def clean_command(root: Path, output: Path) -> list[str]:
    dataset = root / "work/color_alignment/nvs_1cm" / DATASET_NAME
    return [
        str(root / "code/build-release/navvis_recon_ocam_panorama"),
        "--sensor-frame",
        str(dataset / "sensor_frame.xml"),
        "--processed-camera-dir",
        str(dataset / "cam"),
        "--capture",
        CAPTURE,
        "--width",
        "8192",
        "--camera-mask-dir",
        "/opt/NavVis/panorama-rendering/res/g8",
        "--operator-mask",
        "/opt/NavVis/panorama-rendering/res/g8_operator_mask.png",
        "--surface-cloud",
        str(dataset / "pointcloud.ply"),
        "--panorama-info",
        str(dataset / "info/00000-info.json"),
        "--depth-translation-mode",
        "head-minus",
        "--exposure-mode",
        "soft",
        "--seam-mode",
        "pairwise",
        "--nadir-mode",
        "pyramid",
        "--output",
        str(output),
    ]


def markdown_report(report: dict[str, Any]) -> str:
    sparse = report["checks"]["sparse_1024_all_captures"]
    capture = report["checks"]["capture_00000"]
    rerun = capture["final_8k"]["rerun"]
    status = "PASS / BYTE EXACT" if report["status"] == "pass" else "FAIL"
    source_guard = report["checks"]["production_source_read_only_guard"]
    lines = [
        "# Panorama Surface→sparse→final 验收侧车",
        "",
        f"- 数据集：`{report['dataset']}`",
        f"- 模式：`{report['mode']}`",
        f"- 结果：**{status}**",
        f"- 生产源码只读守卫：`{str(source_guard['unchanged']).lower()}`",
        "",
        "## 固定身份与输入",
        "",
        f"- 官方 sparse 二进制 Build ID：`{report['vendor_binary']['build_id']}`",
        f"- 官方 sparse 二进制 SHA-256：`{report['vendor_binary']['sha256']}`",
        f"- 官方软件版本：`{report['vendor_binary']['version']}`",
        f"- 正式 clean binary SHA-256：`{report['clean_binary']['sha256']}`",
        f"- 正式 clean binary Build ID：`{report['clean_binary']['build_id']}`",
        f"- Surface PLY：`{report['inputs']['surface']['path']}`，"
        f"{report['inputs']['surface']['vertex_count']:,} 点，SHA-256 "
        f"`{report['inputs']['surface']['sha256']}`",
        f"- 34 帧 info 有序哈希：`{report['inputs']['all_info']['ordered_hash_sha256']}`",
        f"- capture00000 四相机图像有序哈希："
        f"`{report['inputs']['capture_00000_cameras']['ordered_hash_sha256']}`",
        "",
        "## Exact 指标",
        "",
        "| 边界 | 实测结果 |",
        "|---|---|",
        f"| 34 帧 1024×512 sparse PNG | {sparse['exact_pairs']}/34 byte exact；"
        f"mask IoU=1；毫米深度 exact；总有效像素 {sparse['total_valid']:,} |",
        f"| capture00000 8192×4096 sparse PNG | byte exact；"
        f"有效像素 {capture['sparse_8k']['valid_pixels']:,}；SHA-256 "
        f"`{capture['sparse_8k']['vendor_sha256']}` |",
        f"| sparse PNG→float32 measured | {capture['measured_f32']['exact_values']:,}/"
        f"{capture['measured_f32']['total_values']:,} bit exact |",
        f"| 四层 PCG float64 | {capture['pcg_f64']['exact_values']:,}/"
        f"{capture['pcg_f64']['total_values']:,} bit exact；SHA-256 "
        f"`{capture['pcg_f64']['clean_sha256']}` |",
        f"| capture00000 官方 8K final JPEG | frozen clean byte exact；SHA-256 "
        f"`{capture['final_8k']['vendor_sha256']}` |",
    ]
    if rerun["performed"]:
        lines.append(
            f"| capture00000 本次 8K 重跑 | byte exact=`{str(rerun['byte_exact']).lower()}`；"
            f"wall `{rerun['elapsed_seconds']:.3f} s` |"
        )
    else:
        lines.append("| capture00000 本次 8K 重跑 | 未执行（默认 quick 模式） |")
    lines.extend(
        [
            "",
            "34 帧检查只读取既有 1024×512 官方/净室 PNG，并逐帧重新计算 SHA、"
            "字节一致性和有效像素；不会执行 34×8K。",
            "",
            "## 一键命令",
            "",
            "```bash",
            "scripts/panorama_surface_to_final_acceptance.py",
            "scripts/panorama_surface_to_final_acceptance.py --rerun-capture00000",
            "```",
            "",
            "第二条命令只重跑 capture00000 的 8K 净室生产程序，输出位于自动清理的"
            "临时目录。验收脚本不编译、不修改生产源码，也不调用 34 帧 8K 流程。",
            "",
            "实际 capture00000 生产命令（输出路径在重跑时替换为临时路径）：",
            "",
            "```text",
            report["commands"]["clean_capture_00000_8k"],
            "```",
            "",
            "## 证据边界",
            "",
            "该结论严格限定于报告中的二进制 Build ID、Surface PLY、34 帧 pose/info、"
            "capture00000 相机图像、mask 与 operator mask。`quick` 模式验收冻结产物；"
            "`--rerun-capture00000` 额外验证当前 clean-room 可执行文件。",
        ]
    )
    if report["errors"]:
        lines.extend(["", "## 失败项", ""])
        lines.extend(f"- {message}" for message in report["errors"])
    lines.append("")
    return "\n".join(lines)


def write_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def run_acceptance(root: Path, rerun_capture: bool) -> dict[str, Any]:
    errors: list[str] = []
    dataset = root / "work/color_alignment/nvs_1cm" / DATASET_NAME
    work = root / "work/panorama_alignment_20260827"
    vendor_binary = Path("/opt/NavVis/panorama-rendering/bin/nv_sparse-depthmap-renderer")
    clean_binary = root / "code/build-release/navvis_recon_ocam_panorama"
    source_paths = [
        root / "code/cpp/CMakeLists.txt",
        root / "code/cpp/include/navvis_recon/panorama_rendering.hpp",
        root / "code/cpp/include/navvis_recon/types.hpp",
        root / "code/cpp/src/panorama_rendering.cpp",
        root / "code/cpp/apps/ocam_panorama_pipeline.cpp",
    ]
    source_before = snapshot_sources(source_paths, root)

    vendor_record = guard_file(
        vendor_binary, root, EXPECTED["vendor_binary_sha256"], errors
    )
    vendor_build_id = read_build_id(vendor_binary) if vendor_binary.is_file() else None
    vendor_record["build_id"] = vendor_build_id
    vendor_record["expected_build_id"] = EXPECTED["vendor_build_id"]
    if vendor_build_id != EXPECTED["vendor_build_id"]:
        errors.append("vendor sparse renderer Build ID mismatch")
    vendor_data = vendor_binary.read_bytes() if vendor_binary.is_file() else b""
    vendor_version = EXPECTED["vendor_version"] if EXPECTED["vendor_version"].encode() in vendor_data else None
    vendor_record["version"] = vendor_version
    vendor_record["expected_version"] = EXPECTED["vendor_version"]
    if vendor_version != EXPECTED["vendor_version"]:
        errors.append("vendor sparse renderer version string mismatch")

    clean_record = guard_file(
        clean_binary, root, EXPECTED["clean_binary_sha256"], errors
    )
    clean_record["build_id"] = read_build_id(clean_binary) if clean_binary.is_file() else None

    surface = guard_file(dataset / "pointcloud.ply", root, EXPECTED["surface_sha256"], errors)
    surface["vertex_count"] = (
        parse_ply_vertex_count(dataset / "pointcloud.ply")
        if (dataset / "pointcloud.ply").is_file()
        else None
    )
    if surface["vertex_count"] != 2_857_623:
        errors.append("Surface PLY vertex count mismatch")

    sensor_frame = guard_file(
        dataset / "sensor_frame.xml", root, EXPECTED["sensor_frame_sha256"], errors
    )
    info_paths = [dataset / "info" / f"{capture}-info.json" for capture in EXPECTED_CAPTURE_IDS]
    info_records = [guard_file(path, root, None, errors) for path in info_paths]
    info_hashes = [record.get("sha256", "MISSING") for record in info_records]
    info_aggregate = ordered_hash(info_hashes)
    if info_aggregate != EXPECTED["all_info_ordered_hash_sha256"]:
        errors.append("34-frame panorama-info ordered hash mismatch")
    if info_records[0].get("sha256") != EXPECTED["capture_00000_info_sha256"]:
        errors.append("capture00000 panorama-info SHA-256 mismatch")

    camera_paths = [dataset / "cam" / f"00000-cam{index}.jpg" for index in range(4)]
    camera_records = [guard_file(path, root, None, errors) for path in camera_paths]
    camera_aggregate = ordered_hash(record.get("sha256", "MISSING") for record in camera_records)
    if camera_aggregate != EXPECTED["capture_00000_camera_ordered_hash_sha256"]:
        errors.append("capture00000 camera image ordered hash mismatch")

    mask_paths = [Path(f"/opt/NavVis/panorama-rendering/res/g8/mask-cam{index}.png") for index in range(4)]
    mask_records = [guard_file(path, root, None, errors) for path in mask_paths]
    mask_aggregate = ordered_hash(record.get("sha256", "MISSING") for record in mask_records)
    if mask_aggregate != EXPECTED["camera_mask_ordered_hash_sha256"]:
        errors.append("camera mask ordered hash mismatch")
    operator_mask = guard_file(
        Path("/opt/NavVis/panorama-rendering/res/g8_operator_mask.png"),
        root,
        EXPECTED["operator_mask_sha256"],
        errors,
    )

    vendor_sparse_root = work / "sparse_renderer_vendor_baseline"
    clean_sparse_root = work / "sparse_clean_probe/formal_sparse_exact_regression/all_captures_1024x512"
    vendor_names = sorted(path.name for path in vendor_sparse_root.glob("*-pano_depth_sparse.png"))
    expected_names = [f"{capture}-pano_depth_sparse.png" for capture in EXPECTED_CAPTURE_IDS]
    if vendor_names != expected_names:
        errors.append("vendor 1024 sparse capture set is not exactly 00000..00033")

    sparse_frames: list[dict[str, Any]] = []
    vendor_sparse_hashes: list[str] = []
    clean_sparse_hashes: list[str] = []
    total_valid = 0
    for capture in EXPECTED_CAPTURE_IDS:
        vendor_path = vendor_sparse_root / f"{capture}-pano_depth_sparse.png"
        clean_path = clean_sparse_root / capture / "clean_sparse_truncate.png"
        if not vendor_path.is_file() or not clean_path.is_file():
            errors.append(f"missing 1024 sparse pair for capture {capture}")
            sparse_frames.append({"capture": capture, "byte_exact": False})
            continue
        vendor_hash = sha256_file(vendor_path)
        clean_hash = sha256_file(clean_path)
        vendor_sparse_hashes.append(vendor_hash)
        clean_sparse_hashes.append(clean_hash)
        byte_exact = vendor_hash == clean_hash and vendor_path.read_bytes() == clean_path.read_bytes()
        width, height, millimetres = decode_sparse_png_mm(vendor_path)
        valid = sum(value != 0 for value in millimetres)
        total_valid += valid
        if (width, height) != (1024, 512):
            errors.append(f"capture {capture} sparse dimensions are {width}x{height}")
        if not byte_exact:
            errors.append(f"capture {capture} 1024 sparse PNG mismatch")
        sparse_frames.append(
            {
                "capture": capture,
                "width": width,
                "height": height,
                "valid_pixels": valid,
                "mask_iou": 1.0 if byte_exact else None,
                "exact_depth_values": valid if byte_exact else None,
                "vendor_sha256": vendor_hash,
                "clean_sha256": clean_hash,
                "byte_exact": byte_exact,
            }
        )

    vendor_sparse_aggregate = ordered_hash(vendor_sparse_hashes)
    clean_sparse_aggregate = ordered_hash(clean_sparse_hashes)
    if vendor_sparse_aggregate != EXPECTED["all_sparse_1024_ordered_hash_sha256"]:
        errors.append("vendor 34-frame 1024 sparse ordered hash mismatch")
    if clean_sparse_aggregate != EXPECTED["all_sparse_1024_ordered_hash_sha256"]:
        errors.append("clean 34-frame 1024 sparse ordered hash mismatch")
    valid_values = [frame["valid_pixels"] for frame in sparse_frames if "valid_pixels" in frame]
    if total_valid != EXPECTED["all_sparse_1024_total_valid"]:
        errors.append("34-frame sparse total valid-pixel count mismatch")
    if valid_values and min(valid_values) != EXPECTED["all_sparse_1024_min_valid"]:
        errors.append("34-frame sparse minimum valid-pixel count mismatch")
    if valid_values and max(valid_values) != EXPECTED["all_sparse_1024_max_valid"]:
        errors.append("34-frame sparse maximum valid-pixel count mismatch")

    vendor_sparse_8k = work / "sparse_renderer_vendor_8192x4096/00000-pano_depth_sparse.png"
    clean_sparse_8k = work / "sparse_clean_probe/formal_sparse_exact_regression/capture00000_8192x4096/clean_sparse_truncate.png"
    vendor_sparse_8k_record = guard_file(
        vendor_sparse_8k, root, EXPECTED["capture_00000_sparse_8k_sha256"], errors
    )
    clean_sparse_8k_record = guard_file(
        clean_sparse_8k, root, EXPECTED["capture_00000_sparse_8k_sha256"], errors
    )
    sparse_8k_exact = (
        vendor_sparse_8k.is_file()
        and clean_sparse_8k.is_file()
        and vendor_sparse_8k.read_bytes() == clean_sparse_8k.read_bytes()
    )
    if not sparse_8k_exact:
        errors.append("capture00000 8K sparse PNG mismatch")
    sparse_8k_header = png_header(vendor_sparse_8k) if vendor_sparse_8k.is_file() else {}
    if (sparse_8k_header.get("width"), sparse_8k_header.get("height")) != (8192, 4096):
        errors.append("capture00000 8K sparse dimensions mismatch")

    measured_path = work / "main_surface_sparse_measured_exact.f32"
    measured_record = guard_file(
        measured_path, root, EXPECTED["capture_00000_measured_f32_sha256"], errors
    )
    measured_expected = b""
    if (vendor_sparse_root / "00000-pano_depth_sparse.png").is_file():
        _, _, sparse_mm = decode_sparse_png_mm(
            vendor_sparse_root / "00000-pano_depth_sparse.png"
        )
        measured_expected = b"".join(
            struct.pack("<f", float(value) * 0.001) for value in sparse_mm
        )
    measured_actual = measured_path.read_bytes() if measured_path.is_file() else b""
    measured_exact = measured_actual == measured_expected and bool(measured_expected)
    if not measured_exact:
        errors.append("official sparse PNG to clean measured float32 mismatch")

    clean_pcg = work / "main_surface_sparse_exact_2k_debug/panorama-depth-native.f64"
    vendor_pcg = work / "depth_solver_dump/level4_estimate.raw"
    clean_pcg_record = guard_file(
        clean_pcg, root, EXPECTED["capture_00000_pcg_f64_sha256"], errors
    )
    vendor_pcg_record = guard_file(
        vendor_pcg, root, EXPECTED["capture_00000_pcg_f64_sha256"], errors
    )
    pcg_exact = (
        clean_pcg.is_file()
        and vendor_pcg.is_file()
        and clean_pcg.read_bytes() == vendor_pcg.read_bytes()
    )
    if not pcg_exact:
        errors.append("capture00000 optimized float64 depth mismatch")

    clean_final = work / "main_surface_to_final_exact_v2_8k.jpg"
    vendor_final = dataset / "pano/00000-pano.jpg"
    clean_final_record = guard_file(
        clean_final, root, EXPECTED["capture_00000_final_8k_sha256"], errors
    )
    vendor_final_record = guard_file(
        vendor_final, root, EXPECTED["capture_00000_final_8k_sha256"], errors
    )
    frozen_final_exact = (
        clean_final.is_file()
        and vendor_final.is_file()
        and clean_final.read_bytes() == vendor_final.read_bytes()
    )
    if not frozen_final_exact:
        errors.append("capture00000 frozen clean/vendor 8K final JPEG mismatch")

    command_template = clean_command(root, Path("<temporary-output.jpg>"))
    rerun: dict[str, Any] = {"performed": False}
    if rerun_capture:
        if not clean_binary.is_file():
            errors.append("cannot rerun capture00000: clean production executable missing")
        else:
            with tempfile.TemporaryDirectory(prefix="panorama_surface_to_final_") as directory:
                output = Path(directory) / "capture00000-clean-8k.jpg"
                command = clean_command(root, output)
                environment = os.environ.copy()
                for name in list(environment):
                    if name.startswith("NAVVIS_RECON_"):
                        environment.pop(name)
                environment["LC_ALL"] = "C"
                started = time.monotonic()
                result = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=180,
                )
                elapsed = time.monotonic() - started
                output_hash = sha256_file(output) if output.is_file() else None
                output_exact = (
                    output.is_file()
                    and vendor_final.is_file()
                    and output.read_bytes() == vendor_final.read_bytes()
                )
                rerun = {
                    "performed": True,
                    "returncode": result.returncode,
                    "elapsed_seconds": elapsed,
                    "output_size_bytes": output.stat().st_size if output.is_file() else None,
                    "output_sha256": output_hash,
                    "expected_sha256": EXPECTED["capture_00000_final_8k_sha256"],
                    "byte_exact": output_exact,
                    "stdout_tail": result.stdout[-2000:],
                    "stderr_tail": result.stderr[-2000:],
                    "temporary_output_removed": True,
                }
                if result.returncode != 0:
                    errors.append(f"capture00000 8K rerun exited with {result.returncode}")
                if not output_exact:
                    errors.append("capture00000 8K rerun output is not byte-exact")

    source_after = snapshot_sources(source_paths, root)
    source_unchanged = source_before == source_after
    if not source_unchanged:
        errors.append("production source changed during acceptance run")

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "capture_focus": CAPTURE,
        "mode": "rerun_capture00000_8k" if rerun_capture else "quick_frozen_artifacts",
        "status": "pass" if not errors else "fail",
        "scope": "Surface PLY -> 1024 sparse boundary -> float measured -> PCG -> official 8192 final JPEG",
        "sidecar": {
            "path": "scripts/panorama_surface_to_final_acceptance.py",
            "sha256": sha256_file(
                root / "scripts/panorama_surface_to_final_acceptance.py"
            ),
            "runtime_dependencies": ["Python >= 3.10 standard library", "readelf"],
            "report_json": (
                "regression/2026-07-21_11.41.12/"
                "PANORAMA_SURFACE_TO_FINAL_ACCEPTANCE_20260827.json"
            ),
            "report_markdown": (
                "regression/2026-07-21_11.41.12/"
                "PANORAMA_SURFACE_TO_FINAL_ACCEPTANCE_20260827.md"
            ),
        },
        "vendor_binary": vendor_record,
        "clean_binary": clean_record,
        "inputs": {
            "surface": surface,
            "sensor_frame": sensor_frame,
            "all_info": {
                "captures": len(info_records),
                "ordered_hash_sha256": info_aggregate,
                "expected_ordered_hash_sha256": EXPECTED["all_info_ordered_hash_sha256"],
                "capture_00000_sha256": info_records[0].get("sha256"),
            },
            "capture_00000_cameras": {
                "files": camera_records,
                "ordered_hash_sha256": camera_aggregate,
                "expected_ordered_hash_sha256": EXPECTED["capture_00000_camera_ordered_hash_sha256"],
            },
            "camera_masks": {
                "files": mask_records,
                "ordered_hash_sha256": mask_aggregate,
                "expected_ordered_hash_sha256": EXPECTED["camera_mask_ordered_hash_sha256"],
            },
            "operator_mask": operator_mask,
        },
        "checks": {
            "production_source_read_only_guard": {
                "unchanged": source_unchanged,
                "before": source_before,
                "after": source_after,
            },
            "sparse_1024_all_captures": {
                "captures": len(sparse_frames),
                "exact_pairs": sum(bool(frame.get("byte_exact")) for frame in sparse_frames),
                "mismatches": sum(not bool(frame.get("byte_exact")) for frame in sparse_frames),
                "total_valid": total_valid,
                "min_valid": min(valid_values) if valid_values else None,
                "max_valid": max(valid_values) if valid_values else None,
                "mask_iou": 1.0 if all(frame.get("byte_exact") for frame in sparse_frames) else None,
                "millimetre_values_exact": all(frame.get("byte_exact") for frame in sparse_frames),
                "vendor_ordered_hash_sha256": vendor_sparse_aggregate,
                "clean_ordered_hash_sha256": clean_sparse_aggregate,
                "expected_ordered_hash_sha256": EXPECTED["all_sparse_1024_ordered_hash_sha256"],
                "frames": sparse_frames,
            },
            "capture_00000": {
                "sparse_8k": {
                    "width": sparse_8k_header.get("width"),
                    "height": sparse_8k_header.get("height"),
                    "valid_pixels": EXPECTED["capture_00000_sparse_8k_valid"],
                    "valid_pixels_evidence": "pinned by byte-exact SHA and previously decoded exact artifact",
                    "vendor_sha256": vendor_sparse_8k_record.get("sha256"),
                    "clean_sha256": clean_sparse_8k_record.get("sha256"),
                    "byte_exact": sparse_8k_exact,
                    "mask_iou": 1.0 if sparse_8k_exact else None,
                    "millimetre_values_exact": sparse_8k_exact,
                },
                "measured_f32": {
                    "total_values": len(measured_expected) // 4,
                    "exact_values": len(measured_expected) // 4 if measured_exact else None,
                    "decoded_vendor_sha256": sha256_bytes(measured_expected) if measured_expected else None,
                    "clean_sha256": measured_record.get("sha256"),
                    "byte_exact": measured_exact,
                },
                "pcg_f64": {
                    "total_values": vendor_pcg.stat().st_size // 8 if vendor_pcg.is_file() else None,
                    "exact_values": vendor_pcg.stat().st_size // 8 if pcg_exact else None,
                    "vendor_sha256": vendor_pcg_record.get("sha256"),
                    "clean_sha256": clean_pcg_record.get("sha256"),
                    "byte_exact": pcg_exact,
                },
                "final_8k": {
                    "width": 8192,
                    "height": 4096,
                    "vendor_path": vendor_final_record["path"],
                    "clean_frozen_path": clean_final_record["path"],
                    "vendor_sha256": vendor_final_record.get("sha256"),
                    "clean_frozen_sha256": clean_final_record.get("sha256"),
                    "frozen_byte_exact": frozen_final_exact,
                    "rerun": rerun,
                },
            },
        },
        "commands": {
            "quick": "scripts/panorama_surface_to_final_acceptance.py",
            "rerun_capture_00000_8k": "scripts/panorama_surface_to_final_acceptance.py --rerun-capture00000",
            "clean_capture_00000_8k": shlex.join(command_template),
            "vendor_sparse_reference_pattern": shlex.join(
                [
                    str(vendor_binary),
                    "-d",
                    relative(dataset, root),
                    "-o",
                    "<output-directory>",
                    "--width",
                    "1024",
                    "--height",
                    "512",
                    "-s",
                    relative(dataset / "sensor_frame.xml", root),
                    "-m",
                    relative(dataset / "pointcloud.ply", root),
                ]
            ),
            "vendor_sparse_reference_executed_by_sidecar": False,
        },
        "policy": {
            "production_source": "read_only",
            "build_performed": False,
            "vendor_execution_performed": False,
            "all_capture_8k_execution_performed": False,
            "temporary_rerun_output_retained": False,
        },
        "errors": errors,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun-capture00000",
        action="store_true",
        help="also rerun exactly one clean-room 8192x4096 panorama in /tmp",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run_acceptance(root, arguments.rerun_capture00000)
    report_dir = root / "regression" / DATASET_NAME
    json_path = report_dir / "PANORAMA_SURFACE_TO_FINAL_ACCEPTANCE_20260827.json"
    markdown_path = report_dir / "PANORAMA_SURFACE_TO_FINAL_ACCEPTANCE_20260827.md"
    write_atomic(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_atomic(markdown_path, markdown_report(report))

    sparse = report["checks"]["sparse_1024_all_captures"]
    final = report["checks"]["capture_00000"]["final_8k"]
    print(
        f"status={report['status']} mode={report['mode']} "
        f"sparse_1024={sparse['exact_pairs']}/{sparse['captures']} "
        f"final_frozen_exact={int(final['frozen_byte_exact'])}"
    )
    if final["rerun"]["performed"]:
        print(
            "capture00000_8k_rerun_exact="
            f"{int(final['rerun']['byte_exact'])} "
            f"wall_seconds={final['rerun']['elapsed_seconds']:.3f}"
        )
    print(f"json={relative(json_path, root)}")
    print(f"markdown={relative(markdown_path, root)}")
    for error in report["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
