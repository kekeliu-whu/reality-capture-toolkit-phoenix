#pragma once

#include <deque>
#include <iomanip>
#include <mutex>
#include <queue>

#include "common/common_struct.h"
#include "lio_msgs.h"

namespace lixel
{

struct InputDataBuffer
{
  std::deque<PointT> lidar_points_queue;
  std::deque<int> lidar_frame_sizes_queue;
  std::deque<double> lidar_frameids_queue;
  std::deque<ImuMsg::Ptr> imu_queue;
  std::deque<MotorMsg::Ptr> motor_queue;
  std::deque<GnssMsg::Ptr> gnss_queue;
  std::deque<ImageMsg::Ptr> image_queue;
  std::deque<OdometryMsg::Ptr> prior_odom_queue;
};

struct OutputDataBuffer
{
  std::queue<LioResultMsg::Ptr> lio_result_queue;
};

class IOUtils
{
 public:
  IOUtils(bool motor_enabled, bool sweep_cut_auto);
  ~IOUtils();

  void addLidar(const PointCloudMsg::Ptr &msg);

  void addImu(const ImuMsg::Ptr &msg);

  void addMotor(const MotorMsg::Ptr &msg);

  void addGnss(const GnssMsg::Ptr &msg);

  void addImage(const ImageMsg::Ptr &msg);

  void addOdometry(const OdometryMsg::Ptr &msg);

  bool containsEnoughDataForSyncPackages(double sweep_duration);

  bool syncPackagesImpl(double sweep_duration, MeaureGroup &mg);

  void addLioResult(const LioResultMsg::Ptr &lio_result);

  LioResultMsg::Ptr getLioResult();

  double getInputDataCommonDuration();

 private:
  double getCommonBeginTime();

  double getCommonEndTime();

  void printSyncPackagesResult(MeaureGroup &mg);

  void syncPackagesImplHandleImu(MeaureGroup &mg);

  bool syncPackagesImplCutByDuration(double sweep_duration, MeaureGroup &mg);

  bool syncPackagesImplCutByOriginalSize(MeaureGroup &mg);

  void syncPackagesImplHandleMotor(MeaureGroup &mg);

 private:
  InputDataBuffer input_data_buff_;
  std::recursive_mutex mtx_input_data_buff_;
  std::mutex mtx_output_data_buff_;
  OutputDataBuffer output_data_buff_;
  bool motor_enabled_;
  bool sweep_cut_auto_;
  int sweep_id = 0;

  const double PADDING_DURATION_FOR_IMU_MOTOR_BUFFER = 3.0;
  const double MIN_COMMON_QUEUE_DURATION = 0.2;
};

}  // namespace lixel