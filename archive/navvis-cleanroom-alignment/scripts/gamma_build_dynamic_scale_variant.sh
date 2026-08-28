#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
baseline_root="${root_dir}/work/color_alignment/gamma_joint_square_order_20260827"
variant_root="${root_dir}/work/color_alignment/gamma_joint_square_dynamic_scale_order_20260827"
app_build="${variant_root}/app-build"
source_file="${baseline_root}/app-build/surface_panorama_colorizer.cpp"
variant_source="${app_build}/surface_panorama_colorizer.cpp"
ceres_install="${root_dir}/work/color_alignment/gamma_ceres21_sse2/ceres-install"

mkdir -p "${app_build}"
cp "${source_file}" "${variant_source}"

python3 - "${variant_source}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
needle = """        problem.dynamic_ranges.push_back({static_cast<int>(view), low, high, normalized_weight,
                                          joint_count * 1.0e-4 * normalized_weight});
    }
    for (std::size_t view = 0; view <"""
replacement = """        problem.dynamic_ranges.push_back({static_cast<int>(view), low, high, normalized_weight,
                                          joint_count * 1.0e-4 * normalized_weight});
    }

    // nv_colorcloud first normalizes all per-view dynamic weights, then
    // renormalizes the subset whose percentile range survived validation.
    // Preserve its explicit average/block scale grouping; the algebraically
    // simplified product rounds differently inside ScaledLoss.
    double selected_dynamic_weight = 0.0;
    for (const ExposureDynamicRange& range : problem.dynamic_ranges) {
        selected_dynamic_weight += std::abs(range.normalized_weight);
    }
    if (selected_dynamic_weight > 0.0 && !problem.dynamic_ranges.empty()) {
        const double dynamic_count = static_cast<double>(problem.dynamic_ranges.size());
        const double average_loss_scale = (joint_count * 1.0e-4) / dynamic_count;
        for (ExposureDynamicRange& range : problem.dynamic_ranges) {
            range.normalized_weight /= selected_dynamic_weight;
            range.loss_scale =
                average_loss_scale * (dynamic_count * range.normalized_weight);
        }
    }
    for (std::size_t view = 0; view <"""
count = source.count(needle)
if count != 2:
    raise SystemExit(f"expected two dynamic construction sites, found {count}")
path.write_text(source.replace(needle, replacement), encoding="utf-8")
PY

if [[ "$(grep -Fc 'average_loss_scale * (dynamic_count * range.normalized_weight)' "${variant_source}")" != 2 ]]; then
  echo "failed to install both Dynamic scale grouping sites" >&2
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
  printf 'joint_expression=weight * (difference * difference)\n'
  printf 'dynamic_normalization=two_stage_serial_l1\n'
  printf 'dynamic_scale=average_scale * (count * normalized_weight)\n'
  sha256sum \
    "${source_file}" \
    "${variant_source}" \
    "${ceres_install}/lib/libceres.a" \
    "${variant_root}/navvis_recon_surface_colorizer"
} > "${variant_root}/build_manifest.txt"

cat "${variant_root}/build_manifest.txt"
