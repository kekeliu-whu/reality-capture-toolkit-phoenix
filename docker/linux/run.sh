#!/bin/bash

set -ex

RAW_DIR=/buildspace/s10-data/$1
PRECESS_DIR=/buildspace/s10-data/$2
OUT_DIR=/buildspace/s10-data/$3



CODE_BIN=/buildspace/bin
camera_params="1184.8924263895674,1185.1170631149612,1559.652852771795,1974.3553852739133,-0.010548472940355621,0.00020134273373369012,-0.0022613277269270763,0.00020493257874673932"
COLMAP_EXE=/buildspace/vcpkg/installed/x64-linux/tools/colmap/colmap


rm -rf ${OUT_DIR}
mkdir -p ${OUT_DIR}

cp ${PRECESS_DIR}/transforms.json ${OUT_DIR}
cp ${PRECESS_DIR}/colorized.las ${OUT_DIR}
mkdir -p ${OUT_DIR}/images
cp -r ${RAW_DIR}/camera/left ${RAW_DIR}/camera/right ${OUT_DIR}/images


${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -nooutput_full
mv ${OUT_DIR}/colorized.las_normals.pcd ${OUT_DIR}/colorized.las_normals.full.pcd

${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -output_full

export LD_LIBRARY_PATH=/buildspace/vcpkg/installed/x64-linux/lib/manual-link/:${LD_LIBRARY_PATH}
sudo ldconfig


$COLMAP_EXE feature_extractor --image_path ${OUT_DIR}/images --database_path ${OUT_DIR}/test.db --ImageReader.camera_model OPENCV_FISHEYE --ImageReader.camera_params $camera_params --ImageReader.single_camera_per_folder 1

$COLMAP_EXE exhaustive_matcher --database_path ${OUT_DIR}/test.db
# $COLMAP_EXE sequential_matcher --database_path ${OUT_DIR}/test.db

${CODE_BIN}/sfm -database_filename ${OUT_DIR}/test.db -image_path ${OUT_DIR}/images -output_path ${OUT_DIR}/ -point_cloud_filename ${OUT_DIR}/colorized.las_normals.pcd -initial_pose_filename ${OUT_DIR}/transforms.json
