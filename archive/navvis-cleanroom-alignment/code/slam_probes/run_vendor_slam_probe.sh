#!/usr/bin/env bash
set -euo pipefail

if (($# != 2)); then
  echo "usage: $0 GDB_SCRIPT ASSETS_OUTPUT_DIR" >&2
  exit 2
fi

probe_script=$1
probe_assets_dir=$2
probe_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
probe_dataset=${NAVVIS_PROBE_DATASET:-/media/cybergeo/12T/CSSJ/datasets_rec/2026-07-21_11.41.12}
probe_bag_dir=${NAVVIS_REWRITTEN_BAG_DIR:-/tmp/navvis_postproc_dataset_2026-07-21_11.41.12@2026-08-27_08.00.43_74zl8dn5/tmp_shared_between_action_groups/rewritten}
probe_official=${NAVVIS_PROBE_OFFICIAL:-${probe_root}/work/color_alignment/nvs_1cm/2026-07-21_11.41.12}

mkdir -p "${probe_assets_dir}/internal"

exec gdb -q -batch -x "${probe_script}" --args \
  /opt/NavVis/slam/lib/surveyor_ros/surveyorslam_processing_node \
  --configuration-directories \
  /opt/NavVis/slam/share/surveyor_ros/configuration_files \
  /opt/NavVis/mapper/share/navvis_mapper/launch/Z1 \
  "${probe_root}/code/slam_probes" \
  --configuration-basename=debug_slam_config.lua \
  --clock --wall-clock --publish-speed \
  --assets-output-dir="${probe_assets_dir}" \
  --assets-output-dir-dev="${probe_assets_dir}/internal" \
  --anchor-frame=anchor_cross \
  --sensor-frame="${probe_dataset}/sensor_frame.xml" \
  --dynamic-frames-tf-topics head_body \
  --bags \
  "${probe_bag_dir}/bag_laser_horiz_2.bag" \
  "${probe_bag_dir}/bag_laser_vert_1.bag" \
  "${probe_bag_dir}/bag_laser_horiz_0.bag" \
  "${probe_bag_dir}/bag_laser_vert_0.bag" \
  "${probe_bag_dir}/bag_2026-07-21-11-41-16_0.bag" \
  "${probe_bag_dir}/bag_laser_horiz_1.bag" \
  "${probe_bag_dir}/bag_laser_vert_2.bag" \
  --rec-certificate="${probe_dataset}/trolley.cer" \
  --proc-certificate=/etc/NavVis/workstation.cer \
  --publish-tf-tree --no-roscore \
  --imu-topic /imu/imu_raw/data \
  --hesai-packets-topics /laser_horiz/packets /laser_vert/packets \
  --laser-temperature-topics /laser_vert/temperature /laser_horiz/temperature \
  --laser-status-topics /laser_horiz/pandar_status \
  --topics \
  /controller/parameter_descriptions /controller/parameter_updates \
  /carrierboard_cx3_init /trigger_event /trigger_event_header \
  /trigger_processed /image_saving /heartbeat \
  /hw_trigger_button_pressed /hw_trigger_button_released \
  /user_interaction /beacon_data /imu/imu_raw/data /imu/imu_sync_status \
  /imu/magnetic_field /imu/acc_pure /imu/delta_vel_orient \
  /imu_left/imu_raw/data /imu_right/imu_raw/data \
  /laser_horiz/packets /laser_horiz/temperature /laser_horiz/pandar_status \
  /laser_horiz/clouds /laser_vert/packets /laser_vert_left/packets \
  /laser_vert_right/packets /laser_vert/temperature /laser_vert/pandar_status \
  /laser_vert/clouds /laser_left/clouds /laser_right/clouds \
  /cam0/temperature /cam1/temperature /cam2/temperature /cam3/temperature \
  /slam_anchor /flir_sys_init /hook_size /detected_hook_size \
  /capture_location_delete /capture_location_modify_pano_mode \
  /last_pps /clock_info /visual_odom0/image /visual_odom1/image \
  /visual_odom0/metadata /visual_odom1/metadata \
  /carrierboard_sys_init /visual_odom_mipi_sys_init /pano_mipi_sys_init \
  --anchors="${probe_official}/anchors/anchors.txt" \
  --serialize-submaps --serialize-trajectory-nodes --serialize-inter-constraints \
  --quiet
