#!/bin/bash

set -ex

# 支持多个RAW_DIR作为参数输入
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 RAW_DIR1 [RAW_DIR2 ...]"
  exit 1
fi

CODE_BIN=todokk
COLMAP_EXE=/d/Library/vcpkg/packages/colmap_x64-windows/tools/colmap/colmap.exe
CAMERA_PARAMS="1483,3300,4400,0,0,0,0"

for RAW_DIR in "$@"; do
  OUT_DIR="${RAW_DIR}/sfm"

  echo "processing $RAW_DIR ..."

  rm -rf "${OUT_DIR}"
  mkdir -p "${OUT_DIR}"

  mkdir -p "${OUT_DIR}/images"
  cp -r "${RAW_DIR}/output/undistort/left" "${RAW_DIR}/output/undistort/right" "${RAW_DIR}/output/undistort/ImgPose.txt" "${OUT_DIR}/images"
  cp /d/ProjectX/project-3d/data/sfm/calibration.yaml "${OUT_DIR}"
  if ls "${RAW_DIR}"/output/*_colorized.las 1>/dev/null 2>&1; then
    cp "${RAW_DIR}"/output/*_colorized.las "${OUT_DIR}/colorized.las"
  else
    cp "${RAW_DIR}"/output/*_uncolorized.las "${OUT_DIR}/colorized.las"
  fi

  /d/ProjectX/project-3d/reality-capture-toolkit/build/RelWithDebInfo/process_point_cloud_s10.exe -las_filename "${OUT_DIR}/colorized.las" -nooutput_full

  "${COLMAP_EXE}" feature_extractor --image_path "${OUT_DIR}/images" --database_path "${OUT_DIR}/xsfm.db" --ImageReader.camera_model OPENCV_CUSTOM --ImageReader.camera_params ${CAMERA_PARAMS} --ImageReader.single_camera_per_folder 1

  /d/ProjectX/project-3d/xsfm/build/RelWithDebInfo/xsfm_reset_s20_cameras.exe -database_filename "${OUT_DIR}/xsfm.db" -calibration_filename "${OUT_DIR}/calibration.yaml"

  "${COLMAP_EXE}" sequential_matcher --database_path "${OUT_DIR}/xsfm.db" --SequentialMatching.vocab_tree_path /d/Users/rick/Downloads/vocab_tree_flickr100K_words32K.bin --SequentialMatching.loop_detection 1 --SequentialMatching.loop_detection_period 1 --SequentialMatching.loop_detection_num_nearest_neighbors 2

done
