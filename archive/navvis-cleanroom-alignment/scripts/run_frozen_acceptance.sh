#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$bundle_root/test_resources"
build_root="${NAVVIS_RECON_BUILD_DIR:-$bundle_root/code/build-release}"
acceptance="$build_root/navvis_recon_surface_capture_acceptance"

if [[ ! -x "$acceptance" ]]; then
  echo "Missing $acceptance; run $bundle_root/scripts/build_and_test.sh first." >&2
  exit 2
fi

"$acceptance" \
  --occlusion \
  "$test_root/g11_pre_surface_filter_status_mask_probe_v1_raw_status52.bin"

"$acceptance" \
  --occlusion-main-input \
  "$test_root/g11_pre_surface_filter_status_mask_probe_v1_raw_status52.bin" \
  "$test_root/surface_main_input_clean_occlusion_probe_v1.bin"

"$acceptance" \
  --voxel-compare \
  "$test_root/surface_intermediates_probe_v1_03_output_voxel_aggregation_before_input.bin" \
  "$test_root/surface_intermediates_probe_v1_03_output_voxel_aggregation_after_output.bin"

echo "Frozen surface acceptance passed."
