#pragma once
#include "common/common_lib.h"
#include "ieskf/ieskf.h"
#include "ieskf/state_group.h"
#include "uniform_sampling.h"
#include "xmap.h"

namespace lixel
{

constexpr double IMU_INIT_TS = 4.0;  // unit: s
class Initialization
{
 public:
  Initialization(const InitParam& param);

  bool staticInit(const std::vector<ImuMsg>& imu_vec, const PointCloud::Ptr& lidar_point, double pcl_end_time);
  void setXmap(std::shared_ptr<xmap::Xmap>& xmap_ptr);
  void setIESKF(IESKF::Ptr& ieskf_ptr);

 private:
  void staticStateInit(const std::vector<ImuMsg>& imu_vec, double pcl_end_time);
  void staticMapInit(const PointCloud::Ptr& lidar_point, double pcl_end_time);

  UniformSampling<PointXYZINormal> uniform_sampling_map_;
  PointCloud::Ptr init_pcl;
  IESKF::Ptr ieskf_ptr_;
  KFState::Ptr init_state_;
  std::vector<ImuMsg> static_imu_vec_;
  std::shared_ptr<xmap::Xmap> xmap_ = nullptr;
  InitParam param_;
  bool init_ = false;
};
}  // namespace lixel