#!/bin/bash

set -ex

RAW_DIR=/buildspace/s10-data/$1
PROCESS_DIR=/buildspace/s10-data/$2
OUT_DIR=/buildspace/s10-data/$3
TYPE=$4


CODE_BIN=/buildspace/bin
COLMAP_EXE=/buildspace/vcpkg/installed/x64-linux/tools/colmap/colmap


echo "TYPE=1，processing s20 data..."

CAMERA_PARAMS="1483,1483,3300,4400,0,0,0,0"

rm -rf ${OUT_DIR}
mkdir -p ${OUT_DIR}

mkdir -p ${OUT_DIR}/images
cp -r ${RAW_DIR}/output/undistort/left ${RAW_DIR}/output/undistort/right ${RAW_DIR}/output/undistort/ImgPose.txt ${OUT_DIR}/images
cp ${PROCESS_DIR}/output/calibration.yaml ${OUT_DIR}
cp ${PROCESS_DIR}/output/*_colorized.las ${OUT_DIR}/colorized.las


# ${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -output_full
# mv ${OUT_DIR}/colorized.las_normals.pcd ${OUT_DIR}/colorized.las_normals.full.pcd

${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -nooutput_full

${COLMAP_EXE} feature_extractor --image_path ${OUT_DIR}/images --database_path ${OUT_DIR}/xsfm.db --ImageReader.camera_model OPENCV --ImageReader.camera_params ${CAMERA_PARAMS} --ImageReader.single_camera_per_folder 1

${CODE_BIN}/xsfm_reset_s20_cameras -database_filename ${OUT_DIR}/xsfm.db -calibration_filename ${OUT_DIR}/calibration.yaml

# ${COLMAP_EXE} exhaustive_matcher --database_path ${OUT_DIR}/xsfm.db
${COLMAP_EXE} sequential_matcher --database_path ${OUT_DIR}/xsfm.db --SequentialMatching.vocab_tree_path ${CODE_BIN}/vocab_tree_flickr100K_words32K.bin --SequentialMatching.loop_detection 1 --SequentialMatching.loop_detection_period 1 --SequentialMatching.loop_detection_num_nearest_neighbors 2

# ${CODE_BIN}/xsfm -database_filename ${OUT_DIR}/xsfm.db -output_path ${OUT_DIR}/ -point_cloud_filename ${OUT_DIR}/colorized.las_normals.pcd -initial_pose_dirname ${OUT_DIR} -pose_type 1

