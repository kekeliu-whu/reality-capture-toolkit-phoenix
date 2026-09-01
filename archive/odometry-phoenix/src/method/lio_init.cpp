//
// Created by youyuan on 24-3-16.
//

#include "lio_init.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace
{

using AlignedVec3Vector = std::vector<lixel::Vec3, Eigen::aligned_allocator<lixel::Vec3>>;

void ComputeMeanAndStd(
    const AlignedVec3Vector &samples,
    const std::vector<size_t> &indices,
    lixel::Vec3 &mean,
    lixel::Vec3 &stddev)
{
  mean.setZero();
  stddev.setZero();
  if (indices.empty())
    return;

  for (const size_t index : indices)
    mean += samples[index];
  mean /= static_cast<lixel::FloatDataType>(indices.size());

  if (indices.size() < 2)
    return;
  for (const size_t index : indices)
  {
    const lixel::Vec3 residual = samples[index] - mean;
    stddev += residual.cwiseProduct(residual);
  }
  stddev =
      (stddev / static_cast<lixel::FloatDataType>(indices.size() - 1))
          .cwiseSqrt();
}

}  // namespace

namespace lixel
{

Initialization::Initialization(const InitParam &param, int window_size)
{
  init_ = false;
  xmap_ = nullptr;
  init_state_ = std::make_shared<KFState>(window_size);
  init_pcl.reset(new PointCloud);
  registration_map_ref_.reset(new PointCloudXYZINormal);
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

bool Initialization::initialize(
    const std::vector<ImuMsg> &imu_vec,
    const PointCloud::Ptr &lidar_point,
    double pcl_end_time)
{
  if (init_)
    return true;

  if (param_.dynamic_init_enabled)
    return dynamicInit(imu_vec, lidar_point, pcl_end_time);
  return staticInit(imu_vec, lidar_point, pcl_end_time);
}

bool Initialization::dynamicInit(
    const std::vector<ImuMsg> &imu_vec,
    const PointCloud::Ptr &lidar_point,
    double pcl_end_time)
{
  if (init_)
    return true;
  if (imu_vec.empty() || !lidar_point || lidar_point->empty())
  {
    LOG(ERROR) << "dynamic initialization input is empty";
    return false;
  }

  appendDynamicFrame(imu_vec, lidar_point, pcl_end_time);
  if (dynamic_frames_.empty() || dynamic_start_time_ < 0.0)
    return false;

  const double duration = pcl_end_time - dynamic_start_time_;
  if (duration < param_.dynamic_init_min_duration)
    return false;

  const MotionStatistics statistics = calculateMotionStatistics(false);
  const double registration_ratio = static_cast<double>(registration_success_count_) /
                                    static_cast<double>(dynamic_frames_.size());
  if (statistics.stationary && registration_ratio >= param_.dynamic_init_min_registration_ratio)
  {
    finishDynamicInitialization(pcl_end_time, true);
    return false;
  }

  if (duration < param_.dynamic_init_max_duration)
  {
    if (!motion_wait_logged_)
    {
      std::cout << "[PHOENIX_INIT] motion detected; extending bootstrap to "
                << param_.dynamic_init_max_duration << " s"
                << " acc_std=" << statistics.acc_ref_std.norm()
                << " gyr_std=" << statistics.gyr_std.norm()
                << " registration_ratio=" << registration_ratio << std::endl;
      motion_wait_logged_ = true;
    }
    return false;
  }

  if (registration_ratio < param_.dynamic_init_min_registration_ratio)
  {
    LOG(WARNING) << "dynamic initialization registration ratio is low: " << registration_ratio;
  }
  finishDynamicInitialization(pcl_end_time, false);
  return false;
}

void Initialization::appendDynamicFrame(
    const std::vector<ImuMsg> &imu_vec,
    const PointCloud::Ptr &lidar_point,
    double pcl_end_time)
{
  DynamicFrame frame;
  frame.timestamp = pcl_end_time;
  frame.raw_cloud.reset(new PointCloud(*lidar_point));
  frame.imu_vec = imu_vec;

  PointCloudXYZINormal::Ptr body_cloud = transformPCL(lidar_point);
  frame.registration_cloud.reset(new PointCloudXYZINormal);
  pcl::VoxelGrid<PointXYZINormal> voxel_filter;
  const float voxel_size = static_cast<float>(std::max(0.1, param_.dynamic_init_icp_voxel_size));
  voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
  voxel_filter.setInputCloud(body_cloud);
  voxel_filter.filter(*frame.registration_cloud);

  if (dynamic_start_time_ < 0.0)
  {
    dynamic_start_time_ = imu_vec.front().timestamp;
    frame.pose_ref_body.setIdentity();
    frame.registration_valid =
        frame.registration_cloud->size() >=
        static_cast<size_t>(param_.dynamic_init_min_registration_points);
    frame.fitness = 0.0;
    if (frame.registration_valid)
    {
      *registration_map_ref_ = *frame.registration_cloud;
      ++registration_success_count_;
    }
    dynamic_frames_.push_back(frame);
    std::cout << "[PHOENIX_INIT] dynamic bootstrap started with "
              << frame.registration_cloud->size() << " registration points" << std::endl;
    return;
  }

  const DynamicFrame &previous = dynamic_frames_.back();
  Mat4 initial_guess = previous.pose_ref_body * last_relative_pose_;
  if (dynamic_frames_.size() == 1)
  {
    Vec3 mean_gyr = Vec3::Zero();
    for (const ImuMsg &imu : imu_vec)
      mean_gyr += imu.gyro.cast<FloatDataType>();
    mean_gyr /= static_cast<FloatDataType>(imu_vec.size());
    const double dt = std::max(0.0, pcl_end_time - previous.timestamp);
    Mat4 imu_relative = Mat4::Identity();
    imu_relative.block<3, 3>(0, 0) =
        lixel::exp(mean_gyr * static_cast<FloatDataType>(dt));
    initial_guess = previous.pose_ref_body * imu_relative;
  }

  const size_t min_registration_points =
      static_cast<size_t>(param_.dynamic_init_min_registration_points);
  bool registration_valid =
      frame.registration_cloud->size() >= min_registration_points &&
      registration_map_ref_ &&
      registration_map_ref_->size() >= min_registration_points;
  if (registration_valid)
  {
    pcl::IterativeClosestPoint<PointXYZINormal, PointXYZINormal> icp;
    icp.setMaximumIterations(30);
    icp.setMaxCorrespondenceDistance(param_.dynamic_init_icp_max_correspondence);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-5);
    icp.setInputSource(frame.registration_cloud);
    icp.setInputTarget(registration_map_ref_);
    PointCloudXYZINormal aligned;
    icp.align(aligned, initial_guess.cast<float>());
    frame.fitness = icp.getFitnessScore(param_.dynamic_init_icp_max_correspondence);
    frame.pose_ref_body = icp.getFinalTransformation().cast<FloatDataType>();

    const Mat4 relative_pose = previous.pose_ref_body.inverse() * frame.pose_ref_body;
    const double translation_step = relative_pose.block<3, 1>(0, 3).norm();
    const double rotation_step = lixel::log(relative_pose.block<3, 3>(0, 0)).norm();
    registration_valid = icp.hasConverged() && std::isfinite(frame.fitness) &&
                         frame.fitness <= param_.dynamic_init_icp_fitness_threshold &&
                         frame.pose_ref_body.allFinite() &&
                         translation_step <= 2.0 * param_.dynamic_init_icp_max_correspondence &&
                         rotation_step <= 0.8;
    if (registration_valid)
      last_relative_pose_ = relative_pose;
  }

  if (!registration_valid)
  {
    frame.pose_ref_body = initial_guess;
    frame.fitness = std::numeric_limits<double>::infinity();
  }
  frame.registration_valid = registration_valid;

  if (frame.registration_valid)
  {
    PointCloudXYZINormal::Ptr transformed(new PointCloudXYZINormal);
    pcl::transformPointCloud(*frame.registration_cloud, *transformed, frame.pose_ref_body.cast<float>());
    *registration_map_ref_ += *transformed;
    PointCloudXYZINormal::Ptr filtered_map(new PointCloudXYZINormal);
    voxel_filter.setInputCloud(registration_map_ref_);
    voxel_filter.filter(*filtered_map);
    registration_map_ref_ = filtered_map;
    ++registration_success_count_;
  }

  dynamic_frames_.push_back(frame);
}

Initialization::MotionStatistics Initialization::calculateMotionStatistics(bool robust) const
{
  MotionStatistics result;
  AlignedVec3Vector acc_ref_samples;
  AlignedVec3Vector acc_body_samples;
  AlignedVec3Vector gyr_samples;

  for (const DynamicFrame &frame : dynamic_frames_)
  {
    if (!frame.registration_valid)
      continue;
    const Mat3 rot_ref_body = frame.pose_ref_body.block<3, 3>(0, 0);
    for (const ImuMsg &imu : frame.imu_vec)
    {
      if (!imu.acc.allFinite() || !imu.gyro.allFinite())
        continue;
      const Vec3 acc = imu.acc.cast<FloatDataType>();
      acc_body_samples.push_back(acc);
      acc_ref_samples.push_back(rot_ref_body * acc);
      gyr_samples.push_back(imu.gyro.cast<FloatDataType>());
    }
  }

  if (acc_ref_samples.empty())
    return result;

  std::vector<size_t> indices(acc_ref_samples.size());
  std::iota(indices.begin(), indices.end(), 0);
  if (robust)
  {
    std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
      const double lhs_score = std::abs(acc_ref_samples[lhs].norm() - DEFAULT_GRAVITY) /
                                   std::max(0.1, param_.dynamic_init_max_acc_norm_error) +
                               gyr_samples[lhs].norm() /
                                   std::max(0.01, param_.dynamic_init_max_mean_gyr);
      const double rhs_score = std::abs(acc_ref_samples[rhs].norm() - DEFAULT_GRAVITY) /
                                   std::max(0.1, param_.dynamic_init_max_acc_norm_error) +
                               gyr_samples[rhs].norm() /
                                   std::max(0.01, param_.dynamic_init_max_mean_gyr);
      return lhs_score < rhs_score;
    });
    const size_t keep_count = std::min(
        indices.size(),
        std::max<size_t>(20, static_cast<size_t>(std::ceil(
                                 indices.size() * param_.dynamic_init_robust_sample_ratio))));
    indices.resize(keep_count);
  }

  Vec3 unused_std;
  ComputeMeanAndStd(acc_ref_samples, indices, result.acc_ref_mean, result.acc_ref_std);
  ComputeMeanAndStd(acc_body_samples, indices, result.acc_body_mean, unused_std);
  ComputeMeanAndStd(gyr_samples, indices, result.gyr_mean, result.gyr_std);

  result.stationary =
      result.acc_ref_std.norm() <= param_.dynamic_init_max_acc_std &&
      result.gyr_std.norm() <= param_.dynamic_init_max_gyr_std &&
      result.gyr_mean.norm() <= param_.dynamic_init_max_mean_gyr &&
      std::abs(result.acc_ref_mean.norm() - DEFAULT_GRAVITY) <= param_.dynamic_init_max_acc_norm_error;
  return result;
}

Vec3 Initialization::estimateVelocityRef() const
{
  std::vector<size_t> indices;
  for (size_t i = dynamic_frames_.size(); i > 0 && indices.size() < 12; --i)
  {
    if (dynamic_frames_[i - 1].registration_valid)
      indices.push_back(i - 1);
  }
  std::reverse(indices.begin(), indices.end());
  if (indices.size() < 3)
    return Vec3::Zero();

  double mean_time = 0.0;
  Vec3 mean_position = Vec3::Zero();
  for (const size_t index : indices)
  {
    mean_time += dynamic_frames_[index].timestamp;
    mean_position += dynamic_frames_[index].pose_ref_body.block<3, 1>(0, 3);
  }
  mean_time /= static_cast<double>(indices.size());
  mean_position /= static_cast<FloatDataType>(indices.size());

  double denominator = 0.0;
  Vec3 numerator = Vec3::Zero();
  for (const size_t index : indices)
  {
    const double dt = dynamic_frames_[index].timestamp - mean_time;
    denominator += dt * dt;
    numerator += static_cast<FloatDataType>(dt) *
                 (dynamic_frames_[index].pose_ref_body.block<3, 1>(0, 3) -
                  mean_position);
  }
  if (denominator < 1e-6)
    return Vec3::Zero();
  return numerator / static_cast<FloatDataType>(denominator);
}

Vec3 Initialization::estimateDynamicGyroBias() const
{
  AlignedVec3Vector estimates;
  for (size_t i = 1; i < dynamic_frames_.size(); ++i)
  {
    const DynamicFrame &previous = dynamic_frames_[i - 1];
    const DynamicFrame &current = dynamic_frames_[i];
    if (!previous.registration_valid || !current.registration_valid || current.imu_vec.empty())
      continue;
    const double dt = current.timestamp - previous.timestamp;
    if (dt <= 1e-4)
      continue;

    Vec3 mean_gyr = Vec3::Zero();
    for (const ImuMsg &imu : current.imu_vec)
      mean_gyr += imu.gyro.cast<FloatDataType>();
    mean_gyr /= static_cast<FloatDataType>(current.imu_vec.size());
    const Mat3 relative_rotation = previous.pose_ref_body.block<3, 3>(0, 0).transpose() *
                                   current.pose_ref_body.block<3, 3>(0, 0);
    const Vec3 estimate =
        mean_gyr - lixel::log(relative_rotation) / static_cast<FloatDataType>(dt);
    if (estimate.allFinite() && estimate.norm() < 0.2)
      estimates.push_back(estimate);
  }

  if (estimates.empty())
    return Vec3::Zero();
  std::vector<size_t> indices(estimates.size());
  std::iota(indices.begin(), indices.end(), 0);
  Vec3 mean, stddev;
  ComputeMeanAndStd(estimates, indices, mean, stddev);
  if (mean.norm() > 0.1 || stddev.norm() > 0.15)
    return Vec3::Zero();
  return mean;
}

Mat4 Initialization::interpolateReferencePose(size_t frame_index, double timestamp) const
{
  const DynamicFrame &current = dynamic_frames_[frame_index];
  if (frame_index == 0)
    return current.pose_ref_body;

  const DynamicFrame &previous = dynamic_frames_[frame_index - 1];
  const double duration = current.timestamp - previous.timestamp;
  const double alpha = duration > 1e-6 ?
      std::clamp((timestamp - previous.timestamp) / duration, 0.0, 1.0) : 1.0;
  const QUAT q_previous(previous.pose_ref_body.block<3, 3>(0, 0));
  const QUAT q_current(current.pose_ref_body.block<3, 3>(0, 0));
  Mat4 pose = Mat4::Identity();
  pose.block<3, 3>(0, 0) =
      q_previous.slerp(static_cast<FloatDataType>(alpha), q_current)
          .normalized().toRotationMatrix();
  pose.block<3, 1>(0, 3) =
      static_cast<FloatDataType>(1.0 - alpha) *
          previous.pose_ref_body.block<3, 1>(0, 3) +
      static_cast<FloatDataType>(alpha) *
          current.pose_ref_body.block<3, 1>(0, 3);
  return pose;
}

void Initialization::finishDynamicInitialization(double pcl_end_time, bool stationary)
{
  const MotionStatistics statistics = calculateMotionStatistics(!stationary);
  if (statistics.acc_ref_mean.norm() < 1e-6)
  {
    std::cout << "[PHOENIX_INIT] bootstrap rejected: no valid LiDAR/IMU "
                 "registration support; restarting" << std::endl;
    dynamic_frames_.clear();
    registration_map_ref_.reset(new PointCloudXYZINormal);
    last_relative_pose_.setIdentity();
    dynamic_start_time_ = -1.0;
    registration_success_count_ = 0;
    motion_wait_logged_ = false;
    return;
  }

  const DynamicFrame &last_frame = dynamic_frames_.back();
  Mat3 rot_world_ref;
  if (param_.use_initial_pose)
  {
    const auto &q = param_.initial_quaternion_xyzw;
    Eigen::Quaterniond initial_q(q.w(), q.x(), q.y(), q.z());
    if (initial_q.norm() == 0.0)
      throw std::runtime_error("Initial quaternion must be non-zero");
    rot_world_ref = initial_q.normalized().toRotationMatrix().cast<FloatDataType>() *
                     last_frame.pose_ref_body.block<3, 3>(0, 0).transpose();
  }
  else
  {
    rot_world_ref = QUAT::FromTwoVectors(
                        -statistics.acc_ref_mean.normalized(), DEFAULT_GRIVITY_VEC.normalized())
                        .toRotationMatrix();
  }

  init_state_->timestamp = pcl_end_time;
  for (int state_index = 0; state_index < init_state_->windowSize(); ++state_index)
  {
    const size_t offset = std::min<size_t>(state_index, dynamic_frames_.size() - 1);
    const DynamicFrame &frame = dynamic_frames_[dynamic_frames_.size() - 1 - offset];
    init_state_->sw_rot_[state_index] = rot_world_ref * frame.pose_ref_body.block<3, 3>(0, 0);
    init_state_->sw_pos_[state_index] = rot_world_ref * frame.pose_ref_body.block<3, 1>(0, 3);
    init_state_->sw_timestamp[state_index] = frame.timestamp;
  }

  if (param_.use_initial_pose)
    init_state_->vel_ = param_.initial_velocity.cast<FloatDataType>();
  else
    init_state_->vel_ = rot_world_ref * estimateVelocityRef();
  if (stationary && init_state_->vel_.norm() < 0.2)
    init_state_->vel_.setZero();

  if (param_.use_initial_pose)
  {
    init_state_->gyo_bias_.setZero();
    init_state_->acc_bias_.setZero();
  }
  else if (stationary)
  {
    init_state_->gyo_bias_ = statistics.gyr_mean;
    init_state_->acc_bias_ =
        statistics.acc_body_mean + init_state_->sw_rot_[0].transpose() * DEFAULT_GRIVITY_VEC;
  }
  else
  {
    init_state_->gyo_bias_ = estimateDynamicGyroBias();
    init_state_->acc_bias_.setZero();
  }

  const double rot_std = stationary ? param_.init_rot_std : std::max(param_.init_rot_std, 0.05);
  const double pos_std = stationary ? param_.init_pos_std : std::max(param_.init_pos_std, 0.10);
  const double vel_std = stationary ? param_.init_vel_std : std::max(param_.init_vel_std, 0.50);
  const double gyro_bias_std = stationary ? param_.init_gyro_bias_std : std::max(param_.init_gyro_bias_std, 0.02);
  const double acc_bias_std = stationary ? param_.init_acc_bias_std : std::max(param_.init_acc_bias_std, 0.20);
  MatDIM &mill_P = init_state_->mill_cov;
  mill_P.setIdentity();
  mill_P.block<3, 3>(0, 0).diagonal().setConstant(rot_std * rot_std);
  mill_P.block<3, 3>(3, 3).diagonal().setConstant(pos_std * pos_std);
  mill_P.block<3, 3>(6, 6).diagonal().setConstant(vel_std * vel_std);
  mill_P.block<3, 3>(9, 9).diagonal().setConstant(gyro_bias_std * gyro_bias_std);
  mill_P.block<3, 3>(12, 12).diagonal().setConstant(acc_bias_std * acc_bias_std);
  for (int i = 1; i < init_state_->windowSize(); ++i)
  {
    const int dim = DIM_CURR_STATE + (i - 1) * 6;
    mill_P.block<3, 3>(dim, dim).diagonal().setConstant(rot_std * rot_std);
    mill_P.block<3, 3>(dim + 3, dim + 3).diagonal().setConstant(pos_std * pos_std);
  }
  mill_P *= SCALE;

  buildDynamicMap(rot_world_ref);
  init_ = true;
  ieskf_ptr_->init(*init_state_);

  const double duration = pcl_end_time - dynamic_start_time_;
  const double registration_ratio = static_cast<double>(registration_success_count_) /
                                    static_cast<double>(dynamic_frames_.size());
  std::cout << "[PHOENIX_INIT] complete mode=" << (stationary ? "stationary" : "moving")
            << " duration=" << duration
            << " frames=" << dynamic_frames_.size()
            << " registration_ratio=" << registration_ratio
            << " velocity=" << init_state_->vel_.transpose()
            << " ba=" << init_state_->acc_bias_.transpose()
            << " bg=" << init_state_->gyo_bias_.transpose() << std::endl;

  dynamic_frames_.clear();
  registration_map_ref_.reset();
}

void Initialization::buildDynamicMap(const Mat3 &rot_world_ref)
{
  PointCloudXYZINormal::Ptr init_map_world(new PointCloudXYZINormal);
  size_t reserve_size = 0;
  for (const DynamicFrame &frame : dynamic_frames_)
  {
    if (frame.registration_valid && frame.raw_cloud)
      reserve_size += frame.raw_cloud->size();
  }
  init_map_world->reserve(reserve_size);

  for (size_t frame_index = 0; frame_index < dynamic_frames_.size(); ++frame_index)
  {
    const DynamicFrame &frame = dynamic_frames_[frame_index];
    if (!frame.registration_valid || !frame.raw_cloud)
      continue;
    for (const PointIRT &point : frame.raw_cloud->points)
    {
      const Mat4 pose_ref_body = interpolateReferencePose(frame_index, point.timestamp);
      const Vec3 point_body(point.x, point.y, point.z);
      const Vec3 point_ref = pose_ref_body.block<3, 3>(0, 0) * point_body +
                             pose_ref_body.block<3, 1>(0, 3);
      const Vec3 point_world = rot_world_ref * point_ref;
      PointXYZINormal output{};
      output.x = static_cast<float>(point_world.x());
      output.y = static_cast<float>(point_world.y());
      output.z = static_cast<float>(point_world.z());
      output.intensity = point.intensity;
      init_map_world->push_back(output);
    }
  }

  init_map_world->width = static_cast<uint32_t>(init_map_world->size());
  init_map_world->height = 1;
  init_map_world->header.stamp = dynamic_frames_.back().timestamp * 1e6;
  PointCloudXYZINormal::Ptr downsample_map_world(new PointCloudXYZINormal);
  uniform_sampling_map_.setRadius(xmap_->getConfigs().resolution);
  uniform_sampling_map_.setInputCloud(init_map_world);
  uniform_sampling_map_.filter(*downsample_map_world);
  xmap_->mapIncremental(
      downsample_map_world,
      init_state_->sw_pos_[0].cast<xmap::FloatDataType>());
  std::cout << "[PHOENIX_INIT] dynamic map raw_points=" << init_map_world->size()
            << " downsampled_points=" << downsample_map_world->size() << std::endl;
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
    LOG(ERROR) << "input imu is empty";
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

    mean_acc += imu_acc.cast<FloatDataType>();
    mean_gyr += gyr_acc.cast<FloatDataType>();
  }
  mean_acc /= static_cast<FloatDataType>(static_imu_vec_.size());
  mean_gyr /= static_cast<FloatDataType>(static_imu_vec_.size());

  // state initialization
  init_state_->sw_pos_[0] = Vec3::Zero();
  init_state_->sw_rot_[0] =
      QUAT::FromTwoVectors(-mean_acc / mean_acc.norm(), DEFAULT_GRIVITY_VEC).toRotationMatrix();
  init_state_->vel_ = Vec3::Zero();
  if (param_.use_initial_pose)
  {
    const auto &q = param_.initial_quaternion_xyzw;
    Quaterniond initial_q(q.w(), q.x(), q.y(), q.z());
    if (initial_q.norm() == 0.0)
      throw std::runtime_error("Initial quaternion must be non-zero");
    init_state_->sw_rot_[0] =
        initial_q.normalized().toRotationMatrix().cast<FloatDataType>();
    init_state_->vel_ = param_.initial_velocity.cast<FloatDataType>();
  }
  if (param_.use_initial_pose)
  {
    // The external pose is used when the recorder was already moving during
    // the normal four-second static initialization window.  In that case the
    // mean specific force and angular rate contain real motion and must not be
    // mistaken for sensor bias.
    init_state_->gyo_bias_ = Vec3::Zero();
    init_state_->acc_bias_ = Vec3::Zero();
  }
  else
  {
    init_state_->gyo_bias_ = mean_gyr;
    init_state_->acc_bias_ = mean_acc + init_state_->sw_rot_[0].transpose() * DEFAULT_GRIVITY_VEC;
  }
  for (int i = 1; i < init_state_->windowSize(); ++i)
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
  for (int i = 1; i < init_state_->windowSize(); ++i)
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

  LOG(INFO) << std::fixed << "init_state_->timestamp:" << init_state_->timestamp;
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

  LOG(INFO) << "init pcl size:" << init_pcl->size();
  init_pcl.reset();
}

}  // namespace lixel
