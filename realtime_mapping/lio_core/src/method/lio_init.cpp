//
// Created by youyuan on 24-3-16.
//

#include "lio_init.h"

namespace lixel
{

Initialization::Initialization(const InitParam &param)
{
  init_ = false;
  xmap_ = nullptr;
  init_state_ = std::make_shared<KFState>();
  init_pcl.reset(new PointCloud);
  uniform_sampling_map_.setRandomSeed(0);
  param_ = param;
}

void Initialization::setXmap(std::shared_ptr<xmap::Xmap> &xmap_ptr)
{
  xmap_ = xmap_ptr;
}

void Initialization::setIESKF(IESKF::Ptr &ieskf_ptr)
{
  ieskf_ptr_ = ieskf_ptr;
}

bool Initialization::staticInit(
    const std::vector<ImuMsg> &imu_vec,
    const PointCloud::Ptr &lidar_point,
    double pcl_end_time)
{
  if (init_)
    return true;

  if (imu_vec.empty())
  {
    lslog(LSLOG_ERROR) << "input imu is empty";
    return false;
  }

  staticStateInit(imu_vec, pcl_end_time);
  staticMapInit(lidar_point, pcl_end_time);
  return false;
}

void Initialization::staticStateInit(const std::vector<ImuMsg> &imu_vec, double pcl_end_time)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance **/
  /** 2. normalize the acceleration measurenments to unit gravity **/
  static_imu_vec_.insert(static_imu_vec_.end(), imu_vec.begin(), imu_vec.end());
  if (static_imu_vec_.back().timestamp - static_imu_vec_.front().timestamp < IMU_INIT_TS)
  {
    init_ = false;
    return;
  }
  Vec3 mean_acc = Vec3::Zero(), mean_gyr = Vec3::Zero();
  for (const auto &imu : static_imu_vec_)
  {
    const auto &imu_acc = imu.acc;
    const auto &gyr_acc = imu.gyro;

    mean_acc += imu_acc;
    mean_gyr += gyr_acc;
  }
  mean_acc /= (double)static_imu_vec_.size();
  mean_gyr /= (double)static_imu_vec_.size();

  // state initialization
  init_state_->sw_pos_[0] = Vec3::Zero();
  init_state_->sw_rot_[0] =
      Quaterniond::FromTwoVectors(-mean_acc / mean_acc.norm(), DEFAULT_GRIVITY_VEC).toRotationMatrix();
  init_state_->gyo_bias_ = mean_gyr;
  init_state_->acc_bias_ = mean_acc + init_state_->sw_rot_[0].transpose() * DEFAULT_GRIVITY_VEC;
  for (int i = 1; i < WINDOW_SIZE; ++i)
  {
    init_state_->sw_rot_[i] = init_state_->sw_rot_[0];
    init_state_->sw_pos_[i] = init_state_->sw_pos_[0];
  }

  // covariance initialization
  MatDIM &mill_P = init_state_->mill_cov;
  init_state_->timestamp = pcl_end_time;
  mill_P.setIdentity();
  mill_P(0, 0) = mill_P(1, 1) = mill_P(2, 2) = pow(param_.init_rot_std, 2);
  mill_P(3, 3) = mill_P(4, 4) = mill_P(5, 5) = pow(param_.init_pos_std, 2);
  mill_P(6, 6) = mill_P(7, 7) = mill_P(8, 8) = pow(param_.init_vel_std, 2);
  mill_P(9, 9) = mill_P(10, 10) = mill_P(11, 11) = pow(param_.init_gyro_bias_std, 2);
  mill_P(12, 12) = mill_P(13, 13) = mill_P(14, 14) = pow(param_.init_acc_bias_std, 2);
  for (int i = 1; i < WINDOW_SIZE; ++i)
  {
    int dim = DIM_CURR_STATE + (i - 1) * 6;
    mill_P(dim, dim) = mill_P(dim + 1, dim + 1) = mill_P(dim + 2, dim + 2) = pow(param_.init_rot_std, 2);
    mill_P(dim + 3, dim + 3) = mill_P(dim + 4, dim + 4) = mill_P(dim + 5, dim + 5) = pow(param_.init_pos_std, 2);
  }
  mill_P *= SCALE;
  init_state_->timestamp = pcl_end_time;

#if GRAVITY_CALIBRATION
  P(15, 15) = P(16, 16) = P(17, 17) = INIT_STD_GRAVITY * INIT_STD_GRAVITY;
  states_ptr_->gravity = default_gravity_;
#endif

  lslog(LSLOG_INFO) << std::fixed << "init_state_->timestamp:" << init_state_->timestamp;
  init_ = true;
  ieskf_ptr_->init(*init_state_);
}

void Initialization::staticMapInit(const PointCloud::Ptr &lidar_point, double pcl_end_time)
{
  *init_pcl += *lidar_point;
  if (!init_)
    return;

  init_pcl->header.stamp = pcl_end_time * 1e6;
  PointCloudXYZINormal::Ptr init_map_body = transformPCL(init_pcl);
  PointCloudXYZINormal::Ptr downsample_map_body(new PointCloudXYZINormal);

  uniform_sampling_map_.setRadius(xmap_->getConfigs().resolution);
  uniform_sampling_map_.setInputCloud(init_map_body);
  uniform_sampling_map_.filter(*downsample_map_body);

  PointCloudXYZINormal::Ptr downsample_map_world(new PointCloudXYZINormal);
  transformToWorld(downsample_map_body, init_state_, downsample_map_world);
  xmap_->mapIncremental(downsample_map_world, init_state_->sw_pos_[0].cast<float>());

  lslog(LSLOG_INFO) << "init pcl size:" << init_pcl->size();
  init_pcl.reset();
}

}  // namespace lixel