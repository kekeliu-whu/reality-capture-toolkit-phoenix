#!/usr/bin/env bash
set -euo pipefail

RECON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECON_BUILD_DIR="${NAVVIS_RECON_BUILD_DIR:-${RECON_ROOT}/build-release}"

if [[ $# -eq 0 ]]; then
    python3 "${RECON_ROOT}/runner/navvis_postprocessing_recon.py" --help >&2
    exit 2
fi
for argument in "$@"; do
    if [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
        exec python3 "${RECON_ROOT}/runner/navvis_postprocessing_recon.py" --help
    fi
done

if [[ ! -x "${RECON_BUILD_DIR}/navvis_recon_pandar" || \
      ! -x "${RECON_BUILD_DIR}/navvis_recon_shard_surface_filter" || \
      ! -x "${RECON_BUILD_DIR}/navvis_recon_mapped_space_quality" || \
      ! -x "${RECON_BUILD_DIR}/navvis_recon_surface_colorizer" || \
      ! -x "${RECON_BUILD_DIR}/navvis_recon_ocam_panorama" ]]; then
    cmake -S "${RECON_ROOT}/cpp" -B "${RECON_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${RECON_BUILD_DIR}" --target \
      navvis_recon_pandar \
      navvis_recon_shard_surface_filter \
      navvis_recon_mapped_space_quality \
      navvis_recon_surface_colorizer \
      navvis_recon_ocam_panorama -j "$(nproc)"
fi

PANORAMA_WORKER="${RECON_BUILD_DIR}/navvis_recon_ocam_panorama"
if [[ ! -x "${PANORAMA_WORKER}" ]]; then
    echo "navvis_recon_ocam_panorama is missing; install the documented dependencies and rebuild cpp/" >&2
    exit 1
fi

exec python3 "${RECON_ROOT}/runner/navvis_postprocessing_recon.py" \
    --pandar-worker="${RECON_BUILD_DIR}/navvis_recon_pandar" \
    --surface-worker="${RECON_BUILD_DIR}/navvis_recon_shard_surface_filter" \
    --quality-worker="${RECON_BUILD_DIR}/navvis_recon_mapped_space_quality" \
    --colorizer-worker="${RECON_BUILD_DIR}/navvis_recon_surface_colorizer" \
    --panorama-worker="${PANORAMA_WORKER}" "$@"
