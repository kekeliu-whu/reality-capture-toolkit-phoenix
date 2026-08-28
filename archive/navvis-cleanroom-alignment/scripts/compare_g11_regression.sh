#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/candidate/output/directory" >&2
  exit 2
fi

candidate="$1"
bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tools_root="$bundle_root/test_resources"
code_root="$bundle_root/code"
reference="/media/cybergeo/12T/CSSJ/datasets_proc_reference_g11_0109/2026-02-08_07.33.20"
recording="/media/cybergeo/12T/CSSJ/datasets_rec/2026-02-08_07.33.20"
report_root="$candidate/regression_reports"

for required in \
  "$reference/pointcloud.ply" \
  "$reference/artifacts/trajectory.bag" \
  "$candidate/pointcloud_surface.ply" \
  "$candidate/pointcloud.ply" \
  "$candidate/trajectory.csv" \
  "$candidate/panoramas"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing comparison input: $required" >&2
    exit 2
  fi
done

mkdir -p "$report_root"

python3 "$tools_root/compare_full_clouds.py" \
  "$reference/pointcloud.ply" \
  "$candidate/pointcloud_surface.ply" \
  --trajectory "$candidate/trajectory.csv" \
  --anchor-indices 0,2000,4000,6000,8080 \
  --output "$report_root/geometry_regression_vs_navvis.json"

python3 "$tools_root/compare_colored_clouds.py" \
  "$reference/pointcloud.ply" \
  "$candidate/pointcloud.ply" \
  --trajectory "$candidate/trajectory.csv" \
  --anchor-indices 0,2000,4000,6000,8080 \
  --output "$report_root/color_regression_vs_navvis.json"

python3 "$tools_root/compare_all_panoramas.py" \
  --reference "$reference/pano" \
  --candidate "$candidate/panoramas" \
  --output-json "$report_root/panorama_regression_vs_navvis.json" \
  --output-csv "$report_root/panorama_regression_vs_navvis.csv" \
  --width 1024

python3 "$code_root/tools/evaluate_slam_trajectory.py" \
  --global-bag "$recording/internal/trajectory_slam.bag" \
  --local-bag "$recording/internal/artifacts/trajectory_local.bag" \
  --reference-bag "$reference/artifacts/trajectory.bag" \
  --upsampling-factor 5 \
  --output-csv "$report_root/recorded_slam_fused_trajectory.csv" \
  --report "$report_root/recorded_slam_regression_vs_navvis.json"

echo "Regression reports written to: $report_root"
