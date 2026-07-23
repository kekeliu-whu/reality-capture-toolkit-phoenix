#include "preprocess.h"
#include <glog/logging.h>

namespace
{

inline void TransformPointInPlace(const Eigen::Matrix4f &T, lixel::PointCloud::PointType &p)
{
  p.getVector3fMap() = T.block<3, 3>(0, 0) * p.getVector3fMap() + T.block<3, 1>(0, 3);
}

}  // namespace

namespace lixel
{
PreProcess::PreProcess(/* args */)
{
}

PreProcess::~PreProcess()
{
}

void PreProcess::process(const LioParameters &param, MeaureGroup &mg)
{
  doTrimCloudByMask(
      param.preprocess_param.range_min,
      param.preprocess_param.range_max,
      param.preprocess_param.body_mask_min.cast<float>(),
      param.preprocess_param.body_mask_max.cast<float>(),
      mg.lidar_points);

  doCorrectImu(param.sensor_param.imu_param, mg.imu_vec);

  doCorrectLidar(param.sensor_param.lidar_param, mg.lidar_points);

  doCompensateMotorMotion(param.extrinsic_param.motor_param, mg.motor_vec, mg.lidar_points);

  doTransformToImuFrame(param.extrinsic_param.ext_imu_motor, mg.lidar_points);
}

void PreProcess::doTrimCloudByMask(
    float range_min,
    float range_max,
    Eigen::Vector3f body_mask_min,
    Eigen::Vector3f body_mask_max,
    PointCloud::Ptr &cloud)
{
  PointCloud::Ptr cloud_new(new PointCloud);
  cloud_new->reserve(cloud->points.size());
  for (auto &pcl_point : cloud->points)
  {
    double range = pcl_point.getVector3fMap().norm();
    if (range < range_min || range > range_max)
    {
      continue;
    }
    if (pcl_point.x >= body_mask_min.x() && pcl_point.x <= body_mask_max.x() && pcl_point.y >= body_mask_min.y() &&
        pcl_point.y <= body_mask_max.y() && pcl_point.z >= body_mask_min.z() && pcl_point.z <= body_mask_max.z())
    {
      continue;
    }
    cloud_new->points.push_back(pcl_point);
  }
  cloud_new->width = cloud_new->points.size();
  cloud_new->height = 1;

  cloud = cloud_new;
}

void PreProcess::doCorrectImu(const SensorParam::ImuParam &param, std::vector<ImuMsg> &imus)
{
  if (!param.enabled)
  {
    LOG(WARNING) << "Imu instrinsic calibration is disabled";
    return;
  }
  for (auto &imu : imus)
  {
    imu.acc = param.Ta * param.Ka * (imu.acc - param.Ba);
    imu.gyro = param.Tg * param.Kg * (imu.gyro - param.Bg);
  }
}

void PreProcess::doCorrectLidar(const SensorParam::LidarParam &param, PointCloud::Ptr &cloud)
{
  if (!param.enabled)
  {
    LOG(WARNING) << "Lidar instrinsic calibration is disabled";
    return;
  }

  if (fabs(param.elevation_offset) < 1e-6)
    return;

  for (auto &pcl_point : cloud->points)
  {
    float x = pcl_point.x;
    float y = pcl_point.y;
    float z = pcl_point.z;
    float depth = Eigen::Vector3d(x, y, z).norm();

    if (depth > 0 && depth < 1e3)
    {
      float elevation = asin(z / depth) + param.elevation_offset;
      float dist_xy = cos(elevation) * depth;
      float yaw = atan2(y, x);

      pcl_point.x = dist_xy * cos(yaw);
      pcl_point.y = dist_xy * sin(yaw);
      pcl_point.z = sin(elevation) * depth;
    }
  }
}

void PreProcess::doCompensateMotorMotion(
    const ExtrinsicParam::MotorParam &param,
    const std::vector<MotorMsg> &motor_msgs,
    PointCloud::Ptr &cloud)
{
  if (!param.enabled)
  {
    LOG(WARNING) << "Motor motion compensation is disabled";
    return;
  }

  int idx_m = 0;
  int counter = 0;
  for (int i = 0; i < cloud->points.size(); i++)
  {
    while (idx_m < motor_msgs.size() - 1)
    {
      if (cloud->points[i].timestamp > motor_msgs[idx_m].timestamp &&
          cloud->points[i].timestamp <= motor_msgs[idx_m + 1].timestamp)
      {
        break;
      }
      idx_m++;
    }

    if (idx_m >= motor_msgs.size() - 1)
    {
      break;
    }

    // transform to rotation motor
    TransformPointInPlace(param.ext_motor_lidar.cast<float>(), cloud->points[i]);

    // transform to motor zero position
    float factor = (cloud->points[i].timestamp - motor_msgs[idx_m].timestamp) /
                   (motor_msgs[idx_m + 1].timestamp - motor_msgs[idx_m].timestamp);
    Eigen::Quaternionf R = motor_msgs[idx_m].q.slerp(factor, motor_msgs[idx_m + 1].q);
    cloud->points[i].getVector3fMap() = R * cloud->points[i].getVector3fMap();

    ++counter;
  }

  // DCHECK_EQ(counter, cloud->points.size());
}

void PreProcess::doTransformToImuFrame(const Eigen::Matrix4d &T, PointCloud::Ptr &cloud)
{
  for (auto &pcl_point : cloud->points)
  {
    TransformPointInPlace(T.cast<float>(), pcl_point);
  }
}

};  // namespace lixel