#!/usr/bin/env python3
"""Build the bounded nv_colorcloud/Ceres fingerprint report.

This script deliberately avoids whole-library string intersections.  It reads
the frozen runtime captures, checks a fixed list of RTTI tokens directly in the
vendor ELF, and disassembles only the vendor CGNR window plus the two clean
CGNR object files.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import struct
import subprocess
from pathlib import Path


VENDOR = Path("/opt/NavVis/pointcloud-coloring/bin/nv_colorcloud")
VENDOR_CGNR_START = "0x745000"
VENDOR_CGNR_STOP = "0x750000"
VEX_PREFIXES = ("vadd", "vsub", "vmul", "vdiv", "vsqrt", "vmov", "vfm", "vfnm")
SSE_PACKED = {
    "addpd", "subpd", "mulpd", "divpd", "sqrtpd", "movapd", "movupd",
    "addps", "subps", "mulps", "divps", "sqrtps", "movaps", "movups",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_text(arguments: list[str]) -> str:
    return subprocess.run(
        arguments, check=True, text=True, errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=30
    ).stdout


def summary(directory: Path, vendor: bool = False) -> dict:
    name = "vendor_solver_summary.json" if vendor else "clean_solver_summary.json"
    return json.loads((directory / name).read_text(encoding="utf-8"))


def iterations(directory: Path) -> list[dict]:
    return json.loads((directory / "iterations.json").read_text(encoding="utf-8"))


def models(path: Path) -> dict[int, tuple[float, float]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        view, gain, exponent = line.split()
        result[int(view)] = (float(gain), float(exponent))
    return result


def model_metrics(reference: dict[int, tuple[float, float]], candidate: dict[int, tuple[float, float]]) -> dict:
    views = sorted(reference)
    if views != sorted(candidate):
        raise ValueError("model view sets differ")
    gain = [abs(candidate[view][0] - reference[view][0]) for view in views]
    exponent = [abs(candidate[view][1] - reference[view][1]) for view in views]
    return {
        "view_count": len(views),
        "gain_mae": math.fsum(gain) / len(gain),
        "gain_max": max(gain),
        "exponent_mae": math.fsum(exponent) / len(exponent),
        "exponent_max": max(exponent),
    }


def parse_layout(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = int(value)
    return result


def decode_options(path: Path, layout: dict[str, int]) -> dict:
    raw = path.read_bytes()
    return {
        "captured_bytes": len(raw),
        "trust_region_strategy_type": struct.unpack_from("<i", raw, 88)[0],
        "max_num_iterations": struct.unpack_from("<i", raw, 104)[0],
        "num_threads_before_runtime_override": struct.unpack_from("<i", raw, 120)[0],
        "initial_trust_region_radius": struct.unpack_from("<d", raw, 128)[0],
        "function_tolerance": struct.unpack_from("<d", raw, 184)[0],
        "linear_solver_type": struct.unpack_from("<i", raw, 208)[0],
        "preconditioner_type": struct.unpack_from("<i", raw, 212)[0],
        "min_linear_solver_iterations": struct.unpack_from(
            "<i", raw, layout["ceres::Solver::Options.min_linear_solver_iterations"]
        )[0],
        "max_linear_solver_iterations": struct.unpack_from(
            "<i", raw, layout["ceres::Solver::Options.max_linear_solver_iterations"]
        )[0],
        "eta": struct.unpack_from("<d", raw, layout["ceres::Solver::Options.eta"])[0],
        "jacobi_scaling": bool(raw[layout["ceres::Solver::Options.jacobi_scaling"]]),
        "logging_type": struct.unpack_from(
            "<i", raw, layout["ceres::Solver::Options.logging_type"]
        )[0],
        "minimizer_progress_to_stdout": bool(
            raw[layout["ceres::Solver::Options.minimizer_progress_to_stdout"]]
        ),
    }


def instruction_profile(arguments: list[str]) -> dict:
    disassembly = run_text(arguments)
    mnemonics: collections.Counter[str] = collections.Counter()
    has_wide_register = False
    for line in disassembly.splitlines():
        match = re.match(r"^\s*[0-9a-f]+:\s+([a-z][a-z0-9.]*)", line)
        if not match:
            continue
        mnemonic = match.group(1)
        mnemonics[mnemonic] += 1
        has_wide_register |= "ymm" in line or "zmm" in line
    vex = sum(count for name, count in mnemonics.items() if name.startswith(VEX_PREFIXES))
    return {
        "vex_avx_fma_instruction_count": vex,
        "has_ymm_or_zmm_operand": has_wide_register,
        "sse_packed_instruction_counts": {
            name: mnemonics[name] for name in sorted(SSE_PACKED) if mnemonics[name]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / "work/color_alignment/gamma_ceres_fingerprint_20260827"

    layout = {
        version: parse_layout(work / f"layout/ceres{short}-fingerprint-layout.txt")
        for version, short in (("2.0.0", "20"), ("2.1.0", "21"), ("2.2.0", "22"))
    }
    vendor_dir = work / "run-vendor"
    clean_dirs = {
        "2.0.0": work / "run-ceres20",
        "2.1.0": work / "run-ceres21",
        "2.2.0": work / "run-ceres22",
        "2.1.0_native_float": work / "run-ceres21-native-float",
        "2.1.0_eigen_no_vectorize": work / "run-ceres21-novec",
    }
    vendor_model = models(vendor_dir / "gamma_models.txt")

    vendor_raw = VENDOR.read_bytes()
    note = run_text(["readelf", "-n", str(VENDOR)])
    comment = run_text(["readelf", "-p", ".comment", str(VENDOR)])
    build_id_match = re.search(r"Build ID:\s*([0-9a-f]+)", note)
    compiler_match = re.search(r"GCC:[^\n]+", comment)

    cgnr_objects = [
        root / "work/color_alignment/gamma_auto_analysis/ceres-2.1.0-build-make/internal/ceres/CMakeFiles/ceres_internal.dir/cgnr_solver.cc.o",
        root / "work/color_alignment/gamma_auto_analysis/ceres-2.1.0-build-make/internal/ceres/CMakeFiles/ceres_internal.dir/conjugate_gradients_solver.cc.o",
    ]
    clean_isa_parts = [
        instruction_profile(["objdump", "-d", "-M", "intel", "--no-show-raw-insn", str(path)])
        for path in cgnr_objects
    ]

    clean_runs = {}
    for name, directory in clean_dirs.items():
        clean_runs[name] = {
            "summary": summary(directory),
            "iteration_0": iterations(directory)[0],
            "iteration_1": iterations(directory)[1],
            "iteration_2": iterations(directory)[2],
            "gamma_sha256": sha256(directory / "gamma_models.txt"),
            "model_vs_vendor": model_metrics(vendor_model, models(directory / "gamma_models.txt")),
        }

    result = {
        "verdict": {
            "vendor_ceres": "2.1.x",
            "exact_patch_level_observable": False,
            "version_explains_remaining_model_gap": False,
            "isa_explains_remaining_model_gap": False,
            "remaining_difference_location": (
                "nonlinear objective/functor value or Jacobian after parameters leave identity; "
                "not initial objective, initial gradient norm, first CGNR solve, Ceres release, or ISA"
            ),
        },
        "vendor_binary": {
            "path": str(VENDOR),
            "sha256": sha256(VENDOR),
            "build_id": build_id_match.group(1) if build_id_match else None,
            "compiler_comment": compiler_match.group(0) if compiler_match else None,
            "elf_isa_note": "x86-64-baseline",
        },
        "version_evidence": {
            "runtime_options_bytes": len((vendor_dir / "solver_options_before.bin").read_bytes()),
            "vendor_options_decoded_as_2_1": decode_options(
                vendor_dir / "solver_options_before.bin", layout["2.1.0"]
            ),
            "layouts": layout,
            "fixed_rtti_tokens": {
                "Manifold": b"N5ceres8ManifoldE" in vendor_raw,
                "SubsetManifold": b"N5ceres14SubsetManifoldE" in vendor_raw,
                "ManifoldAdapter": b"N5ceres8internal15ManifoldAdapterE" in vendor_raw,
                "vendor_manifold_source_path": b"internal/ceres/manifold.cc" in vendor_raw,
            },
            "source_release_boundaries": {
                "2.0.0_has_manifold_header": (
                    root / "work/color_alignment/toolchain/ceres-2.0.0/include/ceres/manifold.h"
                ).exists(),
                "2.1.0_has_manifold_header": (
                    root / "work/color_alignment/gamma_auto_analysis/ceres-2.1.0/include/ceres/manifold.h"
                ).exists(),
                "2.2.0_has_manifold_header": (
                    root / "work/color_alignment/toolchain/ceres-2.2.0/include/ceres/manifold.h"
                ).exists(),
            },
            "solve_wrapper_control_flow": {
                "vendor_address": "0x52a890",
                "matches": ["2.1.0", "2.2.0"],
                "differs_from": ["2.0.0"],
                "method": "fixed 0x46-byte disassembly; relative branch displacements ignored",
            },
        },
        "runtime": {
            "vendor": {
                "summary": summary(vendor_dir, vendor=True),
                "iteration_0": iterations(vendor_dir)[0],
                "iteration_1": iterations(vendor_dir)[1],
                "iteration_2": iterations(vendor_dir)[2],
                "gamma_sha256": sha256(vendor_dir / "gamma_models.txt"),
            },
            "clean": clean_runs,
            "all_standard_clean_versions_same_gamma": len({
                clean_runs[version]["gamma_sha256"] for version in ("2.0.0", "2.1.0", "2.2.0")
            }) == 1,
        },
        "isa": {
            "vendor_cgnr_window": instruction_profile([
                "objdump", "-d", "-M", "intel", "--no-show-raw-insn",
                f"--start-address={VENDOR_CGNR_START}", f"--stop-address={VENDOR_CGNR_STOP}",
                str(VENDOR),
            ]),
            "clean_ceres21_cgnr_objects": clean_isa_parts,
            "clean_release_flags": (
                root / "work/color_alignment/gamma_auto_analysis/ceres-2.1.0-build-make/internal/ceres/CMakeFiles/ceres_internal.dir/flags.make"
            ).read_text(encoding="utf-8"),
            "no_vectorize_variant_flags": (
                work / "ceres21-novec-build/internal/ceres/CMakeFiles/ceres_internal.dir/flags.make"
            ).read_text(encoding="utf-8"),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
