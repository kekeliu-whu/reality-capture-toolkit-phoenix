#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {sse2|eigen_scalar|eigen_scalar_noautovec|avx2_fma|native}" >&2
  exit 2
fi

variant="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
ceres_source="${root_dir}/work/color_alignment/gamma_auto_analysis/ceres-2.1.0"
variant_root="${root_dir}/work/color_alignment/gamma_ceres21_${variant}"
ceres_build="${variant_root}/ceres-build"
ceres_install="${variant_root}/ceres-install"
app_build="${variant_root}/app-build"
jobs="${GAMMA_BUILD_JOBS:-8}"

case "${variant}" in
  sse2)
    ceres_flags="-O3 -DNDEBUG -msse2 -mno-avx -mno-avx2 -mno-fma"
    ;;
  eigen_scalar)
    ceres_flags="-O3 -DNDEBUG -msse2 -mno-avx -mno-avx2 -mno-fma -DEIGEN_DONT_VECTORIZE"
    ;;
  eigen_scalar_noautovec)
    ceres_flags="-O3 -DNDEBUG -msse2 -mno-avx -mno-avx2 -mno-fma -DEIGEN_DONT_VECTORIZE -fno-tree-vectorize"
    ;;
  avx2_fma)
    ceres_flags="-O3 -DNDEBUG -mavx2 -mfma"
    ;;
  native)
    ceres_flags="-O3 -DNDEBUG -march=native"
    ;;
  *)
    echo "unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

# Keep the application/objective side on the vendor-observed x86-64 SSE2 path.
# Only the Ceres/Eigen library changes between these variants.
app_flags=(
  -O3 -DNDEBUG -msse2 -mno-avx -mno-avx2 -mno-fma
  -fopenmp -std=c++17
)

cmake -S "${ceres_source}" -B "${ceres_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${ceres_install}" \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DGFLAGS=OFF \
  -DMINIGLOG=ON \
  -DSUITESPARSE=OFF \
  -DCXSPARSE=OFF \
  -DEIGENSPARSE=ON \
  -DLAPACK=ON \
  -DSCHUR_SPECIALIZATIONS=ON \
  -DCMAKE_CXX_FLAGS_RELEASE="${ceres_flags}"

cmake --build "${ceres_build}" --target install --parallel "${jobs}"

mkdir -p "${app_build}"
cp "${root_dir}/code/cpp/apps/surface_panorama_colorizer.cpp" \
  "${app_build}/surface_panorama_colorizer.cpp"

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

sha256sum \
  "${ceres_install}/lib/libceres.a" \
  "${variant_root}/navvis_recon_surface_colorizer" \
  "${app_build}/surface_panorama_colorizer.cpp"
