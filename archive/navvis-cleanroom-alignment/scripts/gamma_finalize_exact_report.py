#!/usr/bin/env python3
"""Build the machine-readable automatic-Gamma exact-alignment report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import struct


IDENTITY_FILES = {
    "state": ("identity_state.bin", 272),
    "residual": ("identity_residual_b.bin", 139_307),
    "scaled_jacobian_values": ("identity_jacobian_scaled_A.bin", 1_264_288),
    "gradient": ("identity_gradient.bin", 272),
    "jacobi_scaling": ("identity_jacobian_scaling.bin", 272),
}
FIRST_JOINT_FILES = {
    "parameters": ("first_joint_raw_parameters.bin", 4),
    "residual": ("first_joint_raw_residual.bin", 2),
    "jacobian": ("first_joint_raw_jacobian.bin", 8),
}
VENDOR_SUMMARY = re.compile(
    r"SUMMARY termination=(?P<termination>-?\d+) "
    r"initial_cost=(?P<initial>[-+0-9.eE]+) "
    r"final_cost=(?P<final>[-+0-9.eE]+) "
    r"successful_steps=(?P<successful>\d+) "
    r"unsuccessful_steps=(?P<unsuccessful>\d+)"
)
CLEAN_BRIEF = re.compile(
    r"Iterations: (?P<iterations>\d+), Initial cost: (?P<initial>[-+0-9.eE]+), "
    r"Final cost: (?P<final>[-+0-9.eE]+), Termination: (?P<termination>\w+)"
)
ELAPSED = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (?P<elapsed>\S+)")
MAX_RSS = re.compile(r"Maximum resident set size \(kbytes\): (?P<rss>\d+)")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_bits(value: float) -> int:
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def ulp_distance(left: float, right: float) -> int:
    return abs(ordered_bits(left) - ordered_bits(right))


def read_models(path: pathlib.Path) -> dict[int, tuple[float, float]]:
    models = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        view, gain, exponent = line.split()
        models[int(view)] = (float(gain), float(exponent))
    return models


def compare_model_files(reference: pathlib.Path, candidate: pathlib.Path) -> dict:
    left = read_models(reference)
    right = read_models(candidate)
    if sorted(left) != sorted(right):
        raise ValueError(f"model view sets differ: {reference} vs {candidate}")
    scalar_exact = sum(
        left[view][component] == right[view][component]
        for view in left
        for component in (0, 1)
    )
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_sha256": sha256(reference),
        "candidate_sha256": sha256(candidate),
        "byte_exact": reference.read_bytes() == candidate.read_bytes(),
        "views": len(left),
        "parameter_scalars": 2 * len(left),
        "parameter_scalars_exact": scalar_exact,
        "gain_mae": sum(abs(left[v][0] - right[v][0]) for v in left) / len(left),
        "exponent_mae": sum(abs(left[v][1] - right[v][1]) for v in left) / len(left),
    }


def compare_binary_set(reference_dir: pathlib.Path, candidate_dir: pathlib.Path,
                       files: dict[str, tuple[str, int]]) -> dict:
    result = {}
    for label, (name, scalar_count) in files.items():
        reference = reference_dir / name
        candidate = candidate_dir / name
        exact = reference.read_bytes() == candidate.read_bytes()
        result[label] = {
            "reference": str(reference),
            "candidate": str(candidate),
            "scalar_count": scalar_count,
            "bytes": candidate.stat().st_size,
            "reference_sha256": sha256(reference),
            "candidate_sha256": sha256(candidate),
            "byte_exact": exact,
        }
    return result


def parse_vendor_summary(path: pathlib.Path) -> dict:
    match = VENDOR_SUMMARY.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"vendor summary missing: {path}")
    return {
        "termination": int(match.group("termination")),
        "initial_cost": float(match.group("initial")),
        "final_cost": float(match.group("final")),
        "successful_steps": int(match.group("successful")),
        "unsuccessful_steps": int(match.group("unsuccessful")),
    }


def parse_clean_run(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8")
    brief = CLEAN_BRIEF.search(text)
    elapsed = ELAPSED.search(text)
    rss = MAX_RSS.search(text)
    if not brief:
        raise ValueError(f"clean Ceres report missing: {path}")
    return {
        "iterations_printed": int(brief.group("iterations")),
        "initial_cost_printed": float(brief.group("initial")),
        "final_cost_printed": float(brief.group("final")),
        "termination_printed": brief.group("termination"),
        "wall_time": elapsed.group("elapsed") if elapsed else None,
        "max_rss_kib": int(rss.group("rss")) if rss else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    work = root / "work/color_alignment"
    exact = work / "gamma_dynamic_exact_20260827"
    identity = exact / "identity"
    vendor_identity = work / "gamma_jacobian_first_joint_20260827/vendor"
    probe_path = work / "gamma_dynamic_scale_probe3_20260827/dynamic_scale_inputs.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))

    raw_weights = probe["raw_weights"]
    normalized_weights = probe["normalized_weights"]
    count = probe["count"]
    average_scale = probe["stack_08"]
    total_scale = probe["stack_38"]
    grouped_scales = [average_scale * (float(count) * value) for value in normalized_weights]
    simplified_scales = [total_scale * value for value in normalized_weights]
    normalization_ulps = [
        ulp_distance(left, right)
        for left, right in zip(raw_weights, normalized_weights)
    ]
    scale_ulps = [
        ulp_distance(left, right)
        for left, right in zip(grouped_scales, simplified_scales)
    ]

    identity_arrays = compare_binary_set(vendor_identity, identity, IDENTITY_FILES)
    first_joint = compare_binary_set(vendor_identity, identity, FIRST_JOINT_FILES)

    vendor_max2 = work / "gamma_jacobian_same_solve_20260827/vendor_max2"
    clean_max2 = exact / "max2"
    vendor_full_runs = [work / f"gamma_auto_vendor_t1/run{index}" for index in (1, 2, 3)]
    clean_full = exact / "full_t1"
    max2_models = compare_model_files(
        vendor_max2 / "gamma_models.txt", clean_max2 / "gamma_models.txt"
    )
    full_models = compare_model_files(
        vendor_full_runs[0] / "gamma_models.txt", clean_full / "gamma_models.txt"
    )
    vendor_full_hashes = [sha256(run / "gamma_models.txt") for run in vendor_full_runs]
    vendor_summaries = [parse_vendor_summary(run / "capture.txt") for run in vendor_full_runs]

    final_context_path = exact / "final_clean_pct/regression.json"
    final_context = json.loads(final_context_path.read_text(encoding="utf-8"))
    fixed_ovs = work / "gamma_auto_vendor_order_detail/effective_exposure_ovs.bin"
    official_binary = pathlib.Path("/opt/NavVis/pointcloud-coloring/bin/nv_colorcloud")
    production_source = root / "code/cpp/apps/surface_panorama_colorizer.cpp"
    production_binary = root / "code/build-release/navvis_recon_surface_colorizer"

    result = {
        "schema": "color-gamma-automatic-exact-alignment-v1",
        "date": "2026-08-27",
        "status": "byte_exact_on_fixed_exposure_ovs",
        "dataset": "2026-07-21_11.41.12",
        "inputs": {
            "fixed_exposure_ovs": str(fixed_ovs.relative_to(root)),
            "fixed_exposure_ovs_bytes": fixed_ovs.stat().st_size,
            "fixed_exposure_ovs_records": fixed_ovs.stat().st_size // 40,
            "fixed_exposure_ovs_sha256": sha256(fixed_ovs),
            "views": 136,
            "joint_blocks": 32_274,
            "joint_residuals": 138_898,
            "dynamic_blocks": 136,
            "scene_ranges": 136,
        },
        "binaries": {
            "vendor_path": str(official_binary),
            "vendor_sha256": sha256(official_binary),
            "vendor_build_id": "a7586f518009434f5e97891f897aea42675f26a0",
            "vendor_software_revision": "996799455e02da742b525658d521da45edae9d10",
            "production_source": str(production_source.relative_to(root)),
            "production_source_sha256": sha256(production_source),
            "production_binary": str(production_binary.relative_to(root)),
            "production_binary_sha256": sha256(production_binary),
        },
        "recovered_semantics": {
            "joint": {
                "expression": "weight * (difference * difference)",
                "old_expression": "(weight * difference) * difference",
                "first_vendor_residual_hex": "0x1.901b30270c734p-14",
                "first_old_clean_residual_hex": "0x1.901b30270c733p-14",
                "post_fix_first_joint_raw_byte_exact": all(
                    item["byte_exact"] for item in first_joint.values()
                ),
            },
            "dynamic": {
                "normalization": "serial L1 normalization of the already globally-normalized surviving ranges",
                "raw_weight_sum": math.fsum(raw_weights),
                "raw_weight_sum_serial": sum(raw_weights),
                "raw_weight_sum_serial_hex": sum(raw_weights).hex(),
                "normalized_weight_sum_serial": sum(normalized_weights),
                "normalized_weight_sum_serial_hex": sum(normalized_weights).hex(),
                "weights_changed_by_second_l1": sum(value > 0 for value in normalization_ulps),
                "second_l1_ulp_histogram": {
                    str(value): normalization_ulps.count(value)
                    for value in sorted(set(normalization_ulps))
                },
                "scale_expression": "average_scale * (range_count * normalized_weight)",
                "average_scale": average_scale,
                "average_scale_hex": average_scale.hex(),
                "total_scale": total_scale,
                "total_scale_hex": total_scale.hex(),
                "first_captured_scale": probe["xmm0_scale"],
                "first_captured_scale_hex": probe["xmm0_scale"].hex(),
                "first_grouped_scale_exact": grouped_scales[0] == probe["xmm0_scale"],
                "grouped_vs_simplified_changed": sum(value > 0 for value in scale_ulps),
                "grouped_vs_simplified_ulp_histogram": {
                    str(value): scale_ulps.count(value)
                    for value in sorted(set(scale_ulps))
                },
                "grouped_vs_simplified_max_abs": max(
                    abs(left - right)
                    for left, right in zip(grouped_scales, simplified_scales)
                ),
                "probe": str(probe_path.relative_to(root)),
                "raw_weights_sha256": sha256(probe_path.parent / "raw_weights.bin"),
                "normalized_weights_sha256": sha256(
                    probe_path.parent / "normalized_weights.bin"
                ),
            },
            "scene": {
                "normalized_weight_storage": "float32 promoted to double/Jet",
                "initial_cost_after_scene_float": 145.50208948799332,
            },
        },
        "identity": {
            "candidate": str(identity.relative_to(root)),
            "reference": str(vendor_identity.relative_to(root)),
            "first_joint": first_joint,
            "full_objective": identity_arrays,
            "all_byte_exact": all(
                item["byte_exact"] for item in (*first_joint.values(), *identity_arrays.values())
            ),
            "initial_cost": json.loads(
                (identity / "solver_summary.json").read_text(encoding="utf-8")
            )["initial_cost"],
        },
        "max2": {
            "models": max2_models,
            "vendor_summary": json.loads(
                (vendor_max2 / "solver_summary.json").read_text(encoding="utf-8")
            ),
            "clean_printed": parse_clean_run(clean_max2 / "run.stderr"),
        },
        "full_t1": {
            "max_num_iterations": 50,
            "solver_threads": 1,
            "models": full_models,
            "clean_printed": parse_clean_run(clean_full / "run.stderr"),
            "vendor_model_sha256_runs": vendor_full_hashes,
            "vendor_three_runs_byte_deterministic": len(set(vendor_full_hashes)) == 1,
            "vendor_summaries": vendor_summaries,
            "vendor_three_summaries_exact": all(
                summary == vendor_summaries[0] for summary in vendor_summaries[1:]
            ),
        },
        "acceptance": {
            "identity_state_residual_jacobian_gradient_scaling_byte_exact": all(
                item["byte_exact"] for item in identity_arrays.values()
            ),
            "first_joint_raw_byte_exact": all(
                item["byte_exact"] for item in first_joint.values()
            ),
            "max2_models_byte_exact": max2_models["byte_exact"],
            "full_models_byte_exact": full_models["byte_exact"],
            "automatic_gamma_fixed_ovs_exact": all(
                [
                    *[item["byte_exact"] for item in identity_arrays.values()],
                    *[item["byte_exact"] for item in first_joint.values()],
                    max2_models["byte_exact"],
                    full_models["byte_exact"],
                ]
            ),
        },
        "scope_boundary": {
            "proven": "automatic Gamma objective and solve on the frozen exposure OVS, one solver thread",
            "not_proven": [
                "arbitrary datasets or different OVS inputs",
                "multi-thread automatic Gamma determinism",
                "upstream PCT/final-OVS generation",
                "end-to-end final PLY byte equality when upstream OVS differs",
            ],
            "current_clean_pct_context": {
                "path": str(final_context_path.relative_to(root)),
                "geometry_bit_exact": final_context["ply"]["geometry_bit_exact"],
                "rgb_mae_255": final_context["ply"]["rgb_mae_255"],
                "rgb_changed_points": final_context["ply"]["rgb_changed_points"],
                "note": "This residual is outside the fixed-exposure-OVS automatic-Gamma proof.",
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["acceptance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
