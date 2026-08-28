#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"

probe_root=work/panorama_alignment_20260827/final_stage_probe
cpp_output="$probe_root/cpp_reconstructed"
python_output="$probe_root/reconstructed"
mkdir -p "$probe_root" "$cpp_output" "$python_output"

g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
  scripts/panorama_final_stage_probe.cpp \
  -o "$probe_root/panorama_final_stage_probe" \
  $(pkg-config --cflags --libs opencv4)

"$probe_root/panorama_final_stage_probe" \
  work/panorama_alignment_20260827/second_blend_capture/blend2_output.tga \
  work/panorama_alignment_20260827/second_blend_capture/blend2_arg2_cam0.pgm \
  work/panorama_alignment_20260827/second_blend_capture/blend2_arg2_cam1.pgm \
  work/panorama_alignment_20260827/second_blend_capture/blend2_arg2_cam2.pgm \
  work/panorama_alignment_20260827/second_blend_capture/blend2_arg2_cam3.pgm \
  "$cpp_output"

python3 scripts/panorama_final_stage_regression.py \
  --projection-mask work/panorama_alignment_20260827/second_blend_capture/blend2_arg2_cam0.pgm \
  --projection-mask work/panorama_alignment_20260827/second_blend_capture/blend2_arg2_cam1.pgm \
  --projection-mask work/panorama_alignment_20260827/second_blend_capture/blend2_arg2_cam2.pgm \
  --projection-mask work/panorama_alignment_20260827/second_blend_capture/blend2_arg2_cam3.pgm \
  --binary-mask-png "$probe_root/vendor_probe/pano_tmp/stitching/BinaryMasks/00000_binaryMask.png" \
  --floor-mask work/panorama_alignment_20260827/floor_inpaint_capture/floor_inpaint_mask.pgm \
  --blend-output work/panorama_alignment_20260827/second_blend_capture/blend2_output.tga \
  --no-floor-jpeg work/panorama_alignment_20260827/vendor_final_imwrite_probe/pano_tmp/stitching/Images/00000-pano.jpg \
  --floor-input work/panorama_alignment_20260827/floor_inpaint_capture/floor_inpaint_input.tga \
  --floor-output work/panorama_alignment_20260827/floor_inpaint_capture/floor_inpaint_output.tga \
  --filled-jpeg work/panorama_alignment_20260827/vendor_final_imwrite_probe/pano_tmp/stitching/ImagesFilled/00000-pano.jpg \
  --output-directory "$python_output" \
  --cpp-candidate-directory "$cpp_output" \
  --json "$probe_root/final_stage_metrics.json"
