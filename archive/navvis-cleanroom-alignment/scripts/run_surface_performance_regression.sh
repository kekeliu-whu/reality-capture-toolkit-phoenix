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
worker="$bundle_root/code/build-release/navvis_recon_shard_surface_filter"
input_shards="/media/cybergeo/12T/CSSJ/datasets_proc_regression_20260823/2026-02-08_07.33.20/raw_shards"
frozen_surface="/media/cybergeo/12T/CSSJ/datasets_proc_regression_20260823/2026-02-08_07.33.20/pointcloud_surface.ply"

for required in "$worker" "$input_shards" "$frozen_surface"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required worker or frozen input: $required" >&2
    exit 2
  fi
done

mkdir -p "$output_dir"
worker_sha="$(sha256sum "$worker" | awk '{print $1}')"
frozen_sha="$(sha256sum "$frozen_surface" | awk '{print $1}')"
printf '%s  %s\n' "$worker_sha" "$worker" >"$output_dir/worker.sha256"
printf '%s  %s\n' "$frozen_sha" "$frozen_surface" >"$output_dir/expected-output.sha256"

OMP_NUM_THREADS=4 \
OMP_MAX_ACTIVE_LEVELS=1 \
OMP_DYNAMIC=FALSE \
OMP_WAIT_POLICY=PASSIVE \
/usr/bin/time -v "$worker" \
  --input-shards "$input_shards" \
  --output "$output_dir/pointcloud_surface.ply" \
  --work-directory "$output_dir/work" \
  --resolution 0.01 \
  --output-cell 0.01 \
  --free-space-carving \
  --free-space-mode sparse \
  --preprocess-threads 8 \
  --tile-threads 8 \
  >"$output_dir/stdout.log" \
  2>"$output_dir/stderr.log"

actual_sha="$(sha256sum "$output_dir/pointcloud_surface.ply" | awk '{print $1}')"
printf '%s  %s\n' "$actual_sha" "$output_dir/pointcloud_surface.ply" >"$output_dir/actual-output.sha256"
if [[ "$actual_sha" != "$frozen_sha" ]]; then
  echo "Surface output SHA mismatch: expected $frozen_sha, got $actual_sha" >&2
  exit 1
fi

rg 'Phase timing:|Elapsed \(wall|Maximum resident|Percent of CPU|Exit status' "$output_dir/stderr.log"
echo "Surface performance regression passed byte-exact: $output_dir"
