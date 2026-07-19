#pragma once

#include "exception.h"
#include "parameters.h"

struct RosWrapperParameters
{
  std::string imu_topic;
  std::string encoder_topic;
  std::string lidar_topic;
  std::string gnss_topic;
  int offline_mode_start_frame_id;
  int offline_mode_end_frame_id;
  int offline_mode_speed_ratio;
};

struct FullParameters : public lixel::LioParameters
{
  RosWrapperParameters ros_param;
};

enum class CalibModel
{
  L2_LIKE = 0,
  K1
};

int ExtractYamlFileFromHbc(const std::string &bag_filename, const std::string &external_calib_filepath);

LioCoreRosErrorCode ReadParameters(
    const std::string &algorithm_config_filename,
    const std::string &external_calib_filepath,
    FullParameters &params,
    CalibModel calib_model);

// clang-format off
/**
 *
 * @brief Reads calibration configuration.
 *
 * IMU intrinsics file:    If the file does not exist, it is not read; if it exists, the configuration must be read successfully, otherwise an error is thrown.
 * Radar intrinsics file:  If the file does not exist, it is not read; if it exists, the configuration must be read successfully, otherwise an error is thrown.
 * Radar motor extrinsics: If the file does not exist, it is not read; if it exists, the configuration must be read successfully, otherwise an error is thrown.
 * Motor IMU extrinsics:   The file must exist and be read successfully, otherwise an error is thrown.
 * RTK extrinsics:         The file must exist and be read successfully, otherwise an error is thrown.
 *
 * @param external_calib_filepath
 * @param params
 * @return
 */
LioCoreRosErrorCode ReadCalibParameters(const std::string &external_calib_filepath, FullParameters &params, CalibModel calib_model);

/**
 * @brief Reads algorithm configuration.
 *
 * The function reacts differently in the following scenarios:
 * 1. If the file does not exist, an error is thrown through CHECK.
 * 2. If the configuration key does not exist, a YAML::Exception is thrown.
 * 3. If there is a parsing error in the configuration value, a YAML::Exception is thrown.
 * 4. If the length of the configuration value is incorrect, an error is thrown through CHECK.
 *
 * @param filename
 * @param params
 * @return
 */
LioCoreRosErrorCode ReadLioParameters(const std::string &filename, FullParameters &params);
// clang-format on
