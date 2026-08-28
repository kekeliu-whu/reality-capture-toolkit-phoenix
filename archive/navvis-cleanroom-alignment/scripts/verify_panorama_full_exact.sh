#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
reference_dir=${1:-"${project_root}/work/color_alignment/nvs_1cm/2026-07-21_11.41.12/pano"}
candidate_root=${2:-"${project_root}/work/panorama_alignment_20260827"}

capture_count=0
mismatch_count=0

for number in $(seq 0 33); do
    capture=$(printf '%05d' "${number}")
    filename="${capture}-pano.jpg"
    candidate="${candidate_root}/all34/${filename}"

    if [[ ! -f "${candidate}" ]]; then
        candidate="${candidate_root}/clean_standard_seamfix_${capture}.jpg"
    fi
    if [[ ! -f "${candidate}" ]]; then
        candidate="${candidate_root}/clean_standard_exact_${capture}.jpg"
    fi
    if [[ ! -f "${reference_dir}/${filename}" || ! -f "${candidate}" ]]; then
        printf 'MISSING %s reference=%s candidate=%s\n' \
            "${capture}" "${reference_dir}/${filename}" "${candidate}"
        mismatch_count=$((mismatch_count + 1))
        capture_count=$((capture_count + 1))
        continue
    fi
    if ! cmp -s "${reference_dir}/${filename}" "${candidate}"; then
        printf 'MISMATCH %s candidate=%s\n' "${capture}" "${candidate}"
        mismatch_count=$((mismatch_count + 1))
    fi
    capture_count=$((capture_count + 1))
done

printf 'captures=%d byte_exact=%d mismatches=%d\n' \
    "${capture_count}" "$((capture_count - mismatch_count))" "${mismatch_count}"

test "${mismatch_count}" -eq 0
