#pragma once
#include "common/common_lib.h"
#include "ieskf/ieskf.h"
#include "ieskf/state_group.h"
#include "uniform_sampling.h"
#include "xmap.h"

#include <limits>

namespace lixel
{

constexpr double IMU_INIT_TS = 4.0;  // unit: s
class Initialization
{
 public:
  Initialization(const InitParam& param, int window_size);

  bool initialize(const std::vector<ImuMsg>& imu_vec, const PointCloud::Ptr& lidar_point, double pcl_end_time);
  bool staticInit(const std::vector<ImuMsg>& imu_vec, const PointCloud::Ptr& lidar_point, double pcl_end_time);
  void setXmap(std::shared_ptr<xmap::Xmap>& xmap_ptr);
  void setIESKF(IESKF::Ptr& ieskf_ptr);

 private:
  struct DynamicFrame
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestamp = 0.0;
    PointCloud::Ptr raw_cloud;
    PointCloudXYZINormal::Ptr registration_cloud;
    std::vector<ImuMsg> imu_vec;
    Mat4 pose_ref_body = Mat4::Identity();
    bool registration_valid = false;
    double fitness = std::numeric_limits<double>::infinity();
  };

  struct MotionStatistics
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vec3 acc_ref_mean = Vec3::Zero();
    Vec3 acc_ref_std = Vec3::Zero();
    Vec3 acc_body_mean = Vec3::Zero();
    Vec3 gyr_mean = Vec3::Zero();
    Vec3 gyr_std = Vec3::Zero();
    bool stationary = false;
  };

  bool dynamicInit(const std::vector<ImuMsg>& imu_vec, const PointCloud::Ptr& lidar_point, double pcl_end_time);
  void appendDynamicFrame(const std::vector<ImuMsg>& imu_vec, const PointCloud::Ptr& lidar_point,
                          double pcl_end_time);
  MotionStatistics calculateMotionStatistics(bool robust) const;
  void finishDynamicInitialization(double pcl_end_time, bool stationary);
  void buildDynamicMap(const Mat3& rot_world_ref);
  Vec3 estimateVelocityRef() const;
  Vec3 estimateDynamicGyroBias() const;
  Mat4 interpolateReferencePose(size_t frame_index, double timestamp) const;

  void staticStateInit(const std::vector<ImuMsg>& imu_vec, double pcl_end_time);
  void staticMapInit(const PointCloud::Ptr& lidar_point, double pcl_end_time);

  UniformSampling<PointXYZINormal> uniform_sampling_map_;
  PointCloud::Ptr init_pcl;
  IESKF::Ptr ieskf_ptr_;
  KFState::Ptr init_state_;
  std::vector<ImuMsg> static_imu_vec_;
  std::vector<DynamicFrame, Eigen::aligned_allocator<DynamicFrame>> dynamic_frames_;
  PointCloudXYZINormal::Ptr registration_map_ref_;
  Mat4 last_relative_pose_ = Mat4::Identity();
  double dynamic_start_time_ = -1.0;
  size_t registration_success_count_ = 0;
  bool motion_wait_logged_ = false;
  std::shared_ptr<xmap::Xmap> xmap_ = nullptr;
  InitParam param_;
  bool init_ = false;
};
}  // namespace lixel
