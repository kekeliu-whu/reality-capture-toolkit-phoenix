#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/.." && pwd)
dataset=${DATASET_ROOT:-$root/work/color_alignment/nvs_1cm/2026-07-21_11.41.12}
work=${WORK_ROOT:-$root/work/panorama_alignment_20260827}
pcl_include=${PCL_INCLUDE:-$work/pcl_reference_package/extracted/usr/include/pcl-1.12}
probe_source=$root/scripts/panorama_world_map_sparse_octree_probe.cpp
probe_dir=$work/sparse_clean_probe
probe=$probe_dir/panorama_world_map_sparse_octree_probe
run_root=${RUN_ROOT:-$probe_dir/formal_sparse_exact_regression}
vendor_1024=${VENDOR_1024_ROOT:-$work/sparse_renderer_vendor_baseline}

if [[ ! -f "$pcl_include/pcl/octree/octree_search.h" ]]; then
    echo "Missing exact PCL 1.12 headers: $pcl_include" >&2
    exit 2
fi
if [[ ! -f "$dataset/pointcloud.ply" ]]; then
    echo "Missing Surface PLY: $dataset/pointcloud.ply" >&2
    exit 2
fi

mkdir -p "$probe_dir" "$run_root"

g++ -std=c++17 -O3 -DNDEBUG -Wno-deprecated-declarations \
    -ffp-contract=off -I"$pcl_include" -I/usr/include/eigen3 \
    $(pkg-config --cflags opencv4) "$probe_source" -o "$probe" \
    $(pkg-config --libs opencv4)

vendor_for_resolution() {
    local width=$1
    local height=$2
    if [[ $width == 1024 && $height == 512 ]]; then
        printf '%s\n' "$vendor_1024"
    else
        printf '%s\n' "$work/sparse_renderer_vendor_${width}x${height}"
    fi
}

resolution_summary=$run_root/capture00000_resolutions.tsv
: > "$resolution_summary"
for dimensions in "256 128" "512 256" "1024 512" "2048 1024" "8192 4096"; do
    read -r width height <<< "$dimensions"
    vendor_root=$(vendor_for_resolution "$width" "$height")
    vendor=$vendor_root/00000-pano_depth_sparse.png
    output=$run_root/capture00000_${width}x${height}
    log=$run_root/capture00000_${width}x${height}.log
    if [[ ! -f "$vendor" ]]; then
        echo "Missing frozen vendor sparse depth: $vendor" >&2
        exit 2
    fi
    "$probe" \
        "$dataset/pointcloud.ply" \
        "$dataset/info/00000-info.json" \
        "$vendor" "$output" "$width" "$height" --compact > "$log"
    exact=0
    if cmp -s "$output/clean_sparse_truncate.png" "$vendor"; then
        exact=1
    fi
    valid=$(sed -n 's/^truncate clean_valid=\([0-9]*\).*/\1/p' "$log")
    render_ms=$(sed -n \
        's/^stage_load_ms=.* stage_octree_ms=.* stage_render_ms=\([^ ]*\)$/\1/p' \
        "$log")
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$width" "$height" "$valid" "$exact" "$render_ms" \
        >> "$resolution_summary"
done

capture_summary=$run_root/all_captures_1024x512.tsv
: > "$capture_summary"
for vendor in "$vendor_1024"/*-pano_depth_sparse.png; do
    name=$(basename -- "$vendor")
    capture=${name%%-*}
    info=$dataset/info/$capture-info.json
    output=$run_root/all_captures_1024x512/$capture
    log=$run_root/all_captures_1024x512/$capture.log
    mkdir -p "$(dirname -- "$log")"
    "$probe" "$dataset/pointcloud.ply" "$info" "$vendor" \
        "$output" 1024 512 --compact > "$log"
    exact=0
    if cmp -s "$output/clean_sparse_truncate.png" "$vendor"; then
        exact=1
    fi
    valid=$(sed -n 's/^truncate clean_valid=\([0-9]*\).*/\1/p' "$log")
    render_ms=$(sed -n \
        's/^stage_load_ms=.* stage_octree_ms=.* stage_render_ms=\([^ ]*\)$/\1/p' \
        "$log")
    printf '%s\t%s\t%s\t%s\n' \
        "$capture" "$valid" "$exact" "$render_ms" >> "$capture_summary"
done

awk -F '\t' '
    { count++; exact += $4; }
    END {
        printf "resolutions=%d exact=%d mismatches=%d\n", count, exact, count-exact;
        if (count != exact) exit 1;
    }
' "$resolution_summary"
awk -F '\t' '
    { count++; valid += $2; exact += $3; render += $4; }
    END {
        printf "captures=%d exact=%d mismatches=%d total_valid=%d render_ms_sum=%.6f\n",
               count, exact, count-exact, valid, render;
        if (count != exact) exit 1;
    }
' "$capture_summary"
