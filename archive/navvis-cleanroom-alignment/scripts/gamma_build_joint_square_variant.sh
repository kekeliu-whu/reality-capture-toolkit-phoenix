#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
baseline_root="${root_dir}/work/color_alignment/gamma_ceres21_sse2"
variant_root="${root_dir}/work/color_alignment/gamma_joint_square_order_20260827"
app_build="${variant_root}/app-build"
source_file="${baseline_root}/app-build/surface_panorama_colorizer.cpp"
variant_source="${app_build}/surface_panorama_colorizer.cpp"
ceres_install="${baseline_root}/ceres-install"

mkdir -p "${app_build}"
cp "${source_file}" "${variant_source}"

old='residuals[row] = T(sample_.observations[row].weight) * difference * difference;'
new='residuals[row] = T(sample_.observations[row].weight) * (difference * difference);'
old_count="$(grep -Fxc "            ${old}" "${variant_source}")"
if [[ "${old_count}" != 1 ]]; then
  echo "expected exactly one Joint residual expression, found ${old_count}" >&2
  exit 1
fi
sed -i 's@ \* difference \* difference;@ * (difference * difference);@' "${variant_source}"
new_count="$(grep -Fxc "            ${new}" "${variant_source}")"
if [[ "${new_count}" != 1 ]]; then
  echo "failed to install parenthesized Joint residual expression" >&2
  exit 1
fi

app_flags=(
  -O3 -DNDEBUG -msse2 -mno-avx -mno-avx2 -mno-fma
  -fopenmp -std=c++17
)

/usr/bin/c++ \
  -DNAVVIS_RECON_HAVE_CERES=1 \
  -I"${root_dir}/code/cpp/include" \
  -isystem /usr/include/eigen3 \
  -isystem /usr/include/opencv4 \
  -isystem "${ceres_install}/include/ceres/internal/miniglog" \
  -isystem "${ceres_install}/include" \
  "${app_flags[@]}" \
  -c "${variant_source}" \
  -o "${app_build}/surface_panorama_colorizer.cpp.o"

/usr/bin/c++ \
  "${app_flags[@]}" \
  -Wl,--disable-new-dtags \
  "${app_build}/surface_panorama_colorizer.cpp.o" \
  -o "${variant_root}/navvis_recon_surface_colorizer" \
  "${root_dir}/work/color_alignment/build-ceres22/libnavvis_recon.a" \
  "${ceres_install}/lib/libceres.a" \
  /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d \
  /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d \
  /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d \
  /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d \
  /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d \
  /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d \
  /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d \
  /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d \
  /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so \
  /usr/lib/x86_64-linux-gnu/libpthread.a \
  /usr/lib/x86_64-linux-gnu/libopenblas.so \
  -lm -ldl

{
  printf 'compiler=' && /usr/bin/c++ --version | head -n 1
  printf 'flags=' && printf '%q ' "${app_flags[@]}" && printf '\n'
  printf 'old_expression=%s\n' "${old}"
  printf 'new_expression=%s\n' "${new}"
  sha256sum \
    "${source_file}" \
    "${variant_source}" \
    "${ceres_install}/lib/libceres.a" \
    "${variant_root}/navvis_recon_surface_colorizer"
} > "${variant_root}/build_manifest.txt"

cat "${variant_root}/build_manifest.txt"
