#!/usr/bin/env bash
set -euo pipefail

alignment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
solver_dump="${alignment_root}/work/panorama_alignment_20260827/depth_solver_dump"
work_root="${alignment_root}/work/panorama_alignment_20260827/world_map_clean_alignment"
probe_binary="${work_root}/panorama_world_map_solver_probe"
probe_output="${work_root}/solver_probe_exact"

mkdir -p "${probe_output}"

g++ \
  -std=c++17 \
  -O3 \
  -DNDEBUG \
  -Wall \
  -Wextra \
  -pedantic \
  -I/usr/include/eigen3 \
  "${alignment_root}/scripts/panorama_world_map_solver_probe.cpp" \
  -o "${probe_binary}"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${probe_binary}" "${solver_dump}" "${probe_output}"

for level in 1 2 3 4; do
  clean="${probe_output}/level${level}_estimate_clean.raw"
  vendor="${solver_dump}/level${level}_estimate.raw"
  cmp "${clean}" "${vendor}"
  sha256sum "${clean}" "${vendor}"
done

echo "All four depth estimates are bit-exact."
