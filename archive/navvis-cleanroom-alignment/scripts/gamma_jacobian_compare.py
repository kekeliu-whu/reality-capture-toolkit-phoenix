#!/usr/bin/env python3
"""Compare frozen vendor/clean identity linearization and CGNR arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


ARRAYS = (
    "identity_state.bin",
    "identity_residual_b.bin",
    "identity_gradient.bin",
    "identity_jacobian_scaled_A.bin",
    "identity_jacobian_scaling.bin",
    "identity_cgnr_A.bin",
    "identity_cgnr_b.bin",
    "identity_cgnr_D.bin",
    "identity_cgnr_rhs_Atb.bin",
    "identity_cgnr_solution_y.bin",
)

# Packed OVS creates 138,898 Joint rows, 136 three-row Dynamic blocks and one
# Scene row.  BlockSparseMatrix stores values in the same row-block order.
RESIDUAL_FAMILIES = {
    "joint": (0, 138_898),
    "dynamic": (138_898, 139_306),
    "scene": (139_306, 139_307),
}
JACOBIAN_FAMILIES = {
    "joint": (0, 1_263_200),
    "dynamic": (1_263_200, 1_264_016),
    "scene": (1_264_016, 1_264_288),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_bits(bits: int) -> int:
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def scalar_record(index: int, vendor_bits: int, clean_bits: int) -> dict:
    vendor = struct.unpack("<d", struct.pack("<Q", vendor_bits))[0]
    clean = struct.unpack("<d", struct.pack("<Q", clean_bits))[0]
    return {
        "index": index,
        "vendor": vendor,
        "clean": clean,
        "vendor_hexfloat": vendor.hex(),
        "clean_hexfloat": clean.hex(),
        "vendor_bits": f"0x{vendor_bits:016x}",
        "clean_bits": f"0x{clean_bits:016x}",
        "absolute_delta": abs(clean - vendor),
        "ulp_distance": abs(ordered_bits(clean_bits) - ordered_bits(vendor_bits)),
    }


def compare_payloads(vendor: bytes, clean: bytes, begin: int = 0, end: int | None = None) -> dict:
    if len(vendor) != len(clean) or len(vendor) % 8:
        raise ValueError("array byte lengths differ or are not f64 aligned")
    count = len(vendor) // 8
    if end is None:
        end = count
    if not (0 <= begin <= end <= count):
        raise ValueError("invalid comparison range")

    exact = 0
    first = None
    maximum_abs = (-1.0, None)
    maximum_ulp = (-1, None)
    absolute_sum = 0.0
    for index in range(begin, end):
        offset = index * 8
        vendor_bits = struct.unpack_from("<Q", vendor, offset)[0]
        clean_bits = struct.unpack_from("<Q", clean, offset)[0]
        if vendor_bits == clean_bits:
            exact += 1
            continue
        record = scalar_record(index, vendor_bits, clean_bits)
        if first is None:
            first = record
        absolute_sum += record["absolute_delta"]
        if record["absolute_delta"] > maximum_abs[0]:
            maximum_abs = (record["absolute_delta"], record)
        if record["ulp_distance"] > maximum_ulp[0]:
            maximum_ulp = (record["ulp_distance"], record)
    span = end - begin
    changed = span - exact
    return {
        "begin": begin,
        "end": end,
        "count": span,
        "exact_count": exact,
        "changed_count": changed,
        "exact_fraction": exact / span if span else 1.0,
        "mean_absolute_delta": absolute_sum / span if span else 0.0,
        "first_difference": first,
        "maximum_absolute_difference": maximum_abs[1],
        "maximum_ulp_difference": maximum_ulp[1],
    }


def compare_file(vendor_path: Path, clean_path: Path) -> dict:
    vendor = vendor_path.read_bytes()
    clean = clean_path.read_bytes()
    result = compare_payloads(vendor, clean)
    result.update({
        "vendor_path": str(vendor_path),
        "clean_path": str(clean_path),
        "vendor_sha256": hashlib.sha256(vendor).hexdigest(),
        "clean_sha256": hashlib.sha256(clean).hexdigest(),
        "byte_exact": vendor == clean,
    })
    if vendor_path.name in {"identity_residual_b.bin", "identity_cgnr_b.bin"}:
        result["families"] = {
            name: compare_payloads(vendor, clean, begin, end)
            for name, (begin, end) in RESIDUAL_FAMILIES.items()
        }
    if vendor_path.name in {"identity_jacobian_scaled_A.bin", "identity_cgnr_A.bin"}:
        result["families"] = {
            name: compare_payloads(vendor, clean, begin, end)
            for name, (begin, end) in JACOBIAN_FAMILIES.items()
        }
    return result


def load_models(path: Path) -> dict[int, tuple[float, float]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        view, gain, exponent = line.split()
        result[int(view)] = (float(gain), float(exponent))
    return result


def model_metrics(vendor_path: Path, clean_path: Path) -> dict:
    vendor = load_models(vendor_path)
    clean = load_models(clean_path)
    if sorted(vendor) != sorted(clean):
        raise ValueError("gamma model view sets differ")
    result = {"view_count": len(vendor)}
    for component, column in (("gain", 0), ("exponent", 1)):
        deltas = [abs(clean[view][column] - vendor[view][column]) for view in sorted(vendor)]
        result[component] = {
            "mae": math.fsum(deltas) / len(deltas),
            "max": max(deltas),
            "changed": sum(delta != 0.0 for delta in deltas),
        }
    result["vendor_sha256"] = sha256(vendor_path)
    result["clean_sha256"] = sha256(clean_path)
    return result


def verify_injection(clean_dir: Path) -> dict:
    capture = json.loads((clean_dir / "iter1_state_injection.json").read_text(encoding="utf-8"))
    before = (clean_dir / "iter1_state_before.bin").read_bytes()
    after = (clean_dir / "iter1_state_vendor_injected.bin").read_bytes()
    if len(before) != len(after) or len(after) != 272 * 8:
        raise ValueError("unexpected injected state size")
    expected = bytearray()
    rows = sorted(capture["rows"], key=lambda row: row["block"])
    for row in rows:
        expected.extend(struct.pack("<dd", row["vendor_gain"], row["vendor_exponent"]))
    changed_scalars = 0
    for index in range(272):
        if before[index * 8 : (index + 1) * 8] != after[index * 8 : (index + 1) * 8]:
            changed_scalars += 1
    return {
        "timing": {
            "iteration": capture["iteration"],
            "new_evaluation_point": capture["new_evaluation_point"],
            "meaning": "entry to iter1 HandleSuccessfulStep derivative evaluation, after x=candidate_x and before iter2 CGNR",
        },
        "state_bytes": len(after),
        "state_scalars": len(after) // 8,
        "parameter_blocks": len(rows),
        "mapped_views_are_0_to_135_permutation": sorted(capture["mapped_views"]) == list(range(136)),
        "after_matches_serialized_vendor_rows": after == bytes(expected),
        "changed_scalars": changed_scalars,
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arrays = {
        name: compare_file(args.vendor_dir / name, args.clean_dir / name)
        for name in ARRAYS
    }
    vendor_iterations = json.loads(
        (args.vendor_dir / "iteration_summaries.json").read_text(encoding="utf-8")
    )
    clean_iterations = json.loads(
        (args.clean_dir / "iteration_summaries.json").read_text(encoding="utf-8")
    )
    result = {
        "verdict": {
            "nonlinear_identity_summary_exact": all(
                vendor_iterations[0][name] == clean_iterations[0][name]
                for name in ("cost", "gradient_max_norm", "gradient_norm")
            ),
            "vendor_max1_state_injection_complete": verify_injection(args.clean_dir)[
                "after_matches_serialized_vendor_rows"
            ],
            "identity_linearization_byte_exact": arrays[
                "identity_jacobian_scaled_A.bin"
            ]["byte_exact"],
            "first_cgnr_solution_byte_exact": arrays[
                "identity_cgnr_solution_y.bin"
            ]["byte_exact"],
        },
        "vendor_dir": str(args.vendor_dir),
        "clean_dir": str(args.clean_dir),
        "dimensions": json.loads(
            (args.vendor_dir / "identity_evaluation.json").read_text(encoding="utf-8")
        ),
        "injection": verify_injection(args.clean_dir),
        "iterations": {"vendor": vendor_iterations, "clean": clean_iterations},
        "arrays": arrays,
        "max2_models": model_metrics(
            args.vendor_dir / "gamma_models.txt", args.clean_dir / "gamma_models.txt"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
