#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/new/output/directory" >&2
  exit 2
fi

output_dir="$1"
if [[ "$output_dir" != /* ]]; then
  echo "Output directory must be absolute: $output_dir" >&2
  exit 2
fi
if [[ -e "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty output directory: $output_dir" >&2
  exit 2
fi

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
code_root="$bundle_root/code"
recording="/media/cybergeo/12T/CSSJ/datasets_rec/2026-02-08_07.33.20"
reference="/media/cybergeo/12T/CSSJ/datasets_proc_reference_g11_0109/2026-02-08_07.33.20"
proc_base="$(dirname "$output_dir")"

for required in \
  "$recording/sensor_frame.xml" \
  "$reference/artifacts/trajectory.bag" \
  "$code_root/build-release/navvis_recon_pandar" \
  "$code_root/build-release/navvis_recon_shard_surface_filter" \
  "$code_root/build-release/navvis_recon_surface_colorizer" \
  "$code_root/build-release/navvis_recon_ocam_panorama"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input or worker: $required" >&2
    echo "Run $bundle_root/scripts/build_and_test.sh before this script." >&2
    exit 2
  fi
done

mkdir -p "$output_dir"
python3 "$code_root/runner/navvis_postprocessing_recon.py" \
  "$recording" \
  --bagplayer-args=--quiet \
  --proc-base-dir "$proc_base" \
  --output-dir "$output_dir" \
  --caller sitemaker \
  --trajectory-bag "$reference/artifacts/trajectory.bag" \
  --slam-reference-bag "$reference/artifacts/trajectory.bag" \
  --pandar-worker "$code_root/build-release/navvis_recon_pandar" \
  --surface-worker "$code_root/build-release/navvis_recon_shard_surface_filter" \
  --colorizer-worker "$code_root/build-release/navvis_recon_surface_colorizer" \
  --panorama-worker "$code_root/build-release/navvis_recon_ocam_panorama" \
  --aligned-standard \
  --force \
  --res 0.01 \
  --cloud-format ply \
  --preset standard \
  --num-threads-panos 32 \
  --surface-preprocess-threads 8 \
  --surface-tile-threads 8 \
  --log-file "$output_dir/postprocessing-regression.log"

echo "Full G11 post-processing regression completed: $output_dir"
echo "Important: this run used the reference trajectory and is not an offline SLAM regression."
