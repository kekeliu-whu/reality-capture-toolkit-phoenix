#!/usr/bin/env bash
set -euo pipefail

RECON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RECON_BUILD_DIR="${NAVVIS_RECON_BUILD_DIR:-${RECON_ROOT}/build-release}"

cmake -S "${RECON_ROOT}/cpp" -B "${RECON_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${RECON_BUILD_DIR}" --target \
  navvis_recon_smoke \
  navvis_recon_panorama_sampling \
  navvis_recon_mapped_space_quality_test \
  navvis_recon_slam_imu_test \
  navvis_recon_surface_capture_acceptance \
  navvis_recon_stage1_imu_ceres_solver \
  navvis_recon_stage2_imu_ceres_solver -j "$(nproc)"
ctest --test-dir "${RECON_BUILD_DIR}" --output-on-failure

mapfile -d '' PYTHON_FILES < <(
  find \
    "${RECON_ROOT}/src" \
    "${RECON_ROOT}/runner" \
    "${RECON_ROOT}/tools" \
    "${RECON_ROOT}/tests" \
    -type f -name '*.py' -print0
)
python3 -m py_compile "${PYTHON_FILES[@]}"

export PYTHONPATH="${RECON_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NAVVIS_RECON_SLAM_NATIVE="${RECON_BUILD_DIR}/libnavvis_recon_slam_frontend_native.so"
python3 "${RECON_ROOT}/tests/test_reconstruction.py"
python3 "${RECON_ROOT}/tests/test_floor_estimator.py"
python3 "${RECON_ROOT}/tests/test_complete_slam_evaluator.py"
python3 "${RECON_ROOT}/tests/test_recording_io.py"

echo "All clean-room build, syntax and unit checks passed."
