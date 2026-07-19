#pragma once

#include <rosbag/bag.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <mutex>
#include <string>

#include "lixel_msgs/LioFullStates.h"

/**
 * @brief DataIO
 *
 * WARNING: THIS STRUCTURE IS FOR THE EXCLUSIVE USE OF LIXEL STUDIO.
 */
class DataIO
{
 public:
  DataIO(const std::string& output_path);

  void AddUndistortedLidarScan(const sensor_msgs::PointCloud2& scan);

  void AddLocalLioStates(const lixel_msgs::LioFullStates& lio_state);

  void AddCorrectedImu(const sensor_msgs::Imu& imu);

  void Close();

  ~DataIO();

 private:
  void TryInit();

 private:
  std::shared_ptr<rosbag::Bag> xbc_writer_;
  std::mutex mtx_;
  std::string output_dir_;
#ifdef __linux__
  std::ofstream poses_csv_;
#endif

  constexpr static char TOPIC_PC_BIN[] = "/algorithm/l1f_pcbin";
  constexpr static char TOPIC_FULL_STATES[] = "/algorithm/fullstates_bin";
  constexpr static char TOPIC_IMU_CORRECTED[] = "/imu_corrected";
};
