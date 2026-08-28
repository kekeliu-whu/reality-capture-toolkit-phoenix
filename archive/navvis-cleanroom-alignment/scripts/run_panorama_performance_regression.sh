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
dataset="/media/cybergeo/12T/CSSJ/datasets_rec/2026-02-08_07.33.20"
frozen="/media/cybergeo/12T/CSSJ/datasets_proc_regression_20260823/2026-02-08_07.33.20"
worker="$code_root/build-release/navvis_recon_ocam_panorama"

for required in "$dataset/sensor_frame.xml" "$frozen/panoramas" "$frozen/cam" "$worker"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input, baseline, or worker: $required" >&2
    exit 2
  fi
done

mkdir -p "$output_dir"
sha256sum "$worker" >"$output_dir/worker.sha256"
/usr/bin/time -v python3 "$code_root/runner/navvis_postprocessing_recon.py" \
  "$dataset" \
  --proc-base-dir "$(dirname "$output_dir")" \
  --output-dir "$output_dir" \
  --caller acceptance \
  --panorama-worker "$worker" \
  --skip-cloud \
  --aligned-standard \
  --force \
  --res 0.01 \
  --preset standard \
  --pano-width 8192 \
  --max-panos 6 \
  --num-threads-panos 32 \
  >"$output_dir/stdout.log" \
  2>"$output_dir/stderr.log"

diff -rq "$frozen/panoramas" "$output_dir/panoramas"
diff -rq "$frozen/cam" "$output_dir/cam"
echo "Panorama performance regression passed byte-exact: $output_dir"
