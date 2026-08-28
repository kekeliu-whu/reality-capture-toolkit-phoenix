#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
code_root="$bundle_root/code"
build_root="${NAVVIS_RECON_BUILD_DIR:-$code_root/build-release}"
build_jobs="${BUILD_JOBS:-8}"

cmake -S "$code_root/cpp" -B "$build_root" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_root" -j"$build_jobs"
ctest --test-dir "$build_root" --output-on-failure
PYTHONPATH="$code_root/src" python3 "$code_root/tests/test_reconstruction.py"
python3 -m py_compile "$code_root/runner/navvis_postprocessing_recon.py"

echo "Build and tests passed: $build_root"
