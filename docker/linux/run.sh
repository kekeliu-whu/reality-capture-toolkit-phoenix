#!/bin/bash

set -ex

RAW_DIR=/buildspace/s10-data/$1
PROCESS_DIR=/buildspace/s10-data/$2
OUT_DIR=/buildspace/s10-data/$3
TYPE=$4



CODE_BIN=/buildspace/bin
COLMAP_EXE=/buildspace/vcpkg/installed/x64-linux/tools/colmap/colmap


if [ "${TYPE}" = "0" ]; then
  echo "TYPE=0，processing s10 data..."

  CAMERA_PARAMS="1184.8924263895674,1185.1170631149612,1559.652852771795,1974.3553852739133,-0.010548472940355621,0.00020134273373369012,-0.0022613277269270763,0.00020493257874673932"

  rm -rf ${OUT_DIR}
  mkdir -p ${OUT_DIR}

  cp ${PROCESS_DIR}/transforms.json ${OUT_DIR}
  cp ${PROCESS_DIR}/colorized.las ${OUT_DIR}
  mkdir -p ${OUT_DIR}/images
  cp -r ${RAW_DIR}/camera/left ${RAW_DIR}/camera/right ${OUT_DIR}/images


  ${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -output_full
  mv ${OUT_DIR}/colorized.las_normals.pcd ${OUT_DIR}/colorized.las_normals.full.pcd

  ${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -nooutput_full

  ${COLMAP_EXE} feature_extractor --image_path ${OUT_DIR}/images --database_path ${OUT_DIR}/xsfm.db --ImageReader.camera_model OPENCV_FISHEYE --ImageReader.camera_params ${CAMERA_PARAMS} --ImageReader.single_camera_per_folder 1

  # ${COLMAP_EXE} exhaustive_matcher --database_path ${OUT_DIR}/xsfm.db
  ${COLMAP_EXE} sequential_matcher --database_path ${OUT_DIR}/xsfm.db --SequentialMatching.vocab_tree_path ${CODE_BIN}/vocab_tree_flickr100K_words32K.bin --SequentialMatching.loop_detection 1 --SequentialMatching.loop_detection_period 1 --SequentialMatching.loop_detection_num_nearest_neighbors 2

  ${CODE_BIN}/xsfm -database_filename ${OUT_DIR}/xsfm.db -output_path ${OUT_DIR}/ -point_cloud_filename ${OUT_DIR}/colorized.las_normals.pcd -initial_pose_dirname ${OUT_DIR} -pose_type 0
elif [ "${TYPE}" = "1" ]; then
  echo "TYPE=1，processing s20 data..."

  CAMERA_PARAMS="1483,1483,3300,4400,0,0,0,0"

  rm -rf ${OUT_DIR}
  mkdir -p ${OUT_DIR}

  cp ${PROCESS_DIR}/output/calibration.yaml ${OUT_DIR}
  cp ${PROCESS_DIR}/output/*Pose.txt ${OUT_DIR}
  cp ${PROCESS_DIR}/output/colorized.las ${OUT_DIR}
  mkdir -p ${OUT_DIR}/images
  cp -r ${RAW_DIR}/output/undistort/left ${RAW_DIR}/output/undistort/right ${OUT_DIR}/images


  ${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -output_full
  mv ${OUT_DIR}/colorized.las_normals.pcd ${OUT_DIR}/colorized.las_normals.full.pcd

  ${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -nooutput_full

  ${COLMAP_EXE} feature_extractor --image_path ${OUT_DIR}/images --database_path ${OUT_DIR}/xsfm.db --ImageReader.camera_model OPENCV --ImageReader.camera_params ${CAMERA_PARAMS} --ImageReader.single_camera_per_folder 1

  ${CODE_BIN}/xsfm_reset_s20_cameras -database_filename ${OUT_DIR}/xsfm.db -calibration_filename ${OUT_DIR}/calibration.yaml

  # ${COLMAP_EXE} exhaustive_matcher --database_path ${OUT_DIR}/xsfm.db
  ${COLMAP_EXE} sequential_matcher --database_path ${OUT_DIR}/xsfm.db --SequentialMatching.vocab_tree_path ${CODE_BIN}/vocab_tree_flickr100K_words32K.bin --SequentialMatching.loop_detection 1 --SequentialMatching.loop_detection_period 1 --SequentialMatching.loop_detection_num_nearest_neighbors 2

  ${CODE_BIN}/xsfm -database_filename ${OUT_DIR}/xsfm.db -output_path ${OUT_DIR}/ -point_cloud_filename ${OUT_DIR}/colorized.las_normals.pcd -initial_pose_dirname ${OUT_DIR} -pose_type 1
elif [ "${TYPE}" = "2" ]; then
  echo "TYPE=1，processing s20 data, simply export pose priors..."

  rm -rf ${OUT_DIR}
  mkdir -p ${OUT_DIR}

  cp ${PROCESS_DIR}/output/calibration.yaml ${OUT_DIR}
  cp ${PROCESS_DIR}/output/*Pose.txt ${OUT_DIR}
  cp ${PROCESS_DIR}/output/colorized.las ${OUT_DIR}
  mkdir -p ${OUT_DIR}/images
  cp -r ${RAW_DIR}/output/undistort/left ${RAW_DIR}/output/undistort/right ${OUT_DIR}/images

  ${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -output_full
  mv ${OUT_DIR}/colorized.las_normals.pcd ${OUT_DIR}/colorized.las_normals.full.pcd

  ${CODE_BIN}/process_point_cloud_s10 -las_filename ${OUT_DIR}/colorized.las -nooutput_full

  ${CODE_BIN}/xsfm -database_filename ${OUT_DIR}/xsfm.db -output_path ${OUT_DIR}/ -point_cloud_filename ${OUT_DIR}/colorized.las_normals.pcd -initial_pose_dirname ${OUT_DIR} -pose_type 2 -calibration_filename ${OUT_DIR}/calibration.yaml -images_path ${OUT_DIR}/images
else
  echo "Error: invalid TYPE ${TYPE}"
  exit 1
fi

