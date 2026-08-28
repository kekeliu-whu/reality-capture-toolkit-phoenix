#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_root="$(cd "${script_dir}/.." && pwd)"
bundle_root="$(cd "${code_root}/.." && pwd)"

fixture_root="${bundle_root}/test_resources/slam_stage1_20260827"
dataset="${1:-${fixture_root}/2026-07-21_11.41.12}"
build_dir="${NAVVIS_RECON_BUILD_DIR:-${code_root}/build-release}"
solver="${STAGE1_CERES_SOLVER:-${build_dir}/navvis_recon_stage1_imu_ceres_solver}"
reference_state="${fixture_root}/reference/optimization_state.pb"
measurements="${fixture_root}/reference/measurements.log"
result_root="${NAVVIS_RECON_TEST_OUTPUT_DIR:-${build_dir}/test-results}"
work_dir="${result_root}/stage1_work"
report="${result_root}/slam_imu_backend_stage1_alignment.json"

if [[ ! -x "${solver}" ]]; then
    echo "Stage1 Ceres solver is missing or not executable: ${solver}" >&2
    exit 2
fi
if [[ ! -f "${reference_state}" || ! -f "${measurements}" ]]; then
    echo "Frozen Stage1 acceptance resources are missing under ${fixture_root}" >&2
    exit 2
fi

mkdir -p "${result_root}"

python3 "${code_root}/tools/evaluate_fast_imu_stage1.py" "${dataset}" \
    --reference-state "${reference_state}" \
    --vendor-measurements "${measurements}" \
    --ceres-solver "${solver}" \
    --ceres-work-dir "${work_dir}" \
    --official-final-cost-precise 401551.5707926048 \
    --output "${report}"

python3 - "${report}" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report["acceptance"]["complete_stage1_backend_result_aligned"]:
    raise SystemExit("Stage1 backend acceptance failed")
print("Stage1 backend acceptance: PASS")
PY
