#!/bin/bash

set -ex

# 支持多个RAW_DIR作为参数输入
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 RAW_DIR1 [RAW_DIR2 ...]"
  exit 1
fi

for RAW_DIR in "$@"; do
  OUT_DIR="${RAW_DIR}/sfm"

  echo "processing $RAW_DIR ..."

  /d/ProjectX/project-3d/xsfm/build/RelWithDebInfo/xsfm.exe -database_filename "${OUT_DIR}/xsfm.db" -images_path "${OUT_DIR}/images" -initial_pose_filename "${OUT_DIR}/images/ImgPose.txt" -point_cloud_filename "${OUT_DIR}/colorized.las_normals.pcd" -point_cloud_offset_filename "${OUT_DIR}/colorized.las_offset.csv" -output_path "${OUT_DIR}"

done
