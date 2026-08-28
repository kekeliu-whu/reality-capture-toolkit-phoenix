#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {O3|O2|O1|Os}" >&2
  exit 2
fi

variant="$1"
case "${variant}" in
  O3|O2|O1|Os) ;;
  *)
    echo "unsupported optimization variant: ${variant}" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
baseline_root="${root_dir}/work/color_alignment/gamma_ceres21_sse2"
variant_root="${root_dir}/work/color_alignment/gamma_app_opt_${variant}"
app_build="${variant_root}/app-build"
source_file="${baseline_root}/app-build/surface_panorama_colorizer.cpp"
ceres_install="${baseline_root}/ceres-install"

mkdir -p "${app_build}"
cp "${source_file}" "${app_build}/surface_panorama_colorizer.cpp"

app_flags=(
  "-${variant}" -DNDEBUG -msse2 -mno-avx -mno-avx2 -mno-fma
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
  -c "${app_build}/surface_panorama_colorizer.cpp" \
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
  printf 'variant=%s\n' "${variant}"
  printf 'compiler=' && /usr/bin/c++ --version | head -n 1
  printf 'flags=' && printf '%q ' "${app_flags[@]}" && printf '\n'
  sha256sum \
    "${source_file}" \
    "${ceres_install}/lib/libceres.a" \
    "${variant_root}/navvis_recon_surface_colorizer"
} > "${variant_root}/build_manifest.txt"

cat "${variant_root}/build_manifest.txt"
