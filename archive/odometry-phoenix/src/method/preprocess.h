#pragma once

#include "common/common_struct.h"
#include "lio_msgs.h"
#include "parameters.h"

namespace lixel
{

class PreProcess
{
 private:
  /* data */
 public:
  PreProcess(/* args */);
  ~PreProcess();

  void process(const LioParameters &param, MeaureGroup &mg);

 private:
  // TODO：Mask operation will under the point cloud of world frame instead of body frame
  void doTrimCloudByMask(
      float range_min,
      float range_max,
      Eigen::Vector3f body_mask_min,
      Eigen::Vector3f body_mask_max,
      PointCloud::Ptr &cloud);

  void doCorrectImu(const SensorParam::ImuParam &param, std::vector<ImuMsg> &imus);

  void doCorrectLidar(const SensorParam::LidarParam &param, PointCloud::Ptr &cloud);

  void doCompensateMotorMotion(
      const ExtrinsicParam::MotorParam &param,
      const std::vector<MotorMsg> &motor_msgs,
      PointCloud::Ptr &cloud);

  void doTransformToImuFrame(const Eigen::Matrix4d &T, PointCloud::Ptr &cloud);
};

};  // namespace lixel