#!/usr/bin/env bash
set -euo pipefail

RECON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECON_BUILD_DIR="${NAVVIS_RECON_BUILD_DIR:-${RECON_ROOT}/build-release}"
SLAM_STAGE_BUILD_DIR="${NAVVIS_RECON_SLAM_STAGE_BUILD_DIR:-${RECON_BUILD_DIR}}"

if [[ $# -eq 0 ]]; then
  python3 "${RECON_ROOT}/runner/navvis_complete_slam.py" --help >&2
  exit 2
fi
for argument in "$@"; do
  if [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    exec python3 "${RECON_ROOT}/runner/navvis_complete_slam.py" --help
  fi
done

cmake -S "${RECON_ROOT}/cpp" -B "${RECON_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${RECON_BUILD_DIR}" --target \
  navvis_recon_pandar \
  navvis_recon_slam \
  navvis_recon_slam_frontend_native -j "$(nproc)"

if [[ "${SLAM_STAGE_BUILD_DIR}" == "${RECON_BUILD_DIR}" ]]; then
  cmake --build "${RECON_BUILD_DIR}" --target \
    navvis_recon_stage1_imu_ceres_solver \
    navvis_recon_stage2_imu_ceres_solver -j "$(nproc)"
else
  cmake -S "${RECON_ROOT}/cpp" -B "${SLAM_STAGE_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${SLAM_STAGE_BUILD_DIR}" --target \
    navvis_recon_stage1_imu_ceres_solver \
    navvis_recon_stage2_imu_ceres_solver -j "$(nproc)"
fi

exec python3 "${RECON_ROOT}/runner/navvis_complete_slam.py" \
  --build-dir "${RECON_BUILD_DIR}" \
  --stage-build-dir "${SLAM_STAGE_BUILD_DIR}" "$@"
