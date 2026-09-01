//
// Created by youyuan on 24-2-18.
//

#include <omp.h>
#include <glog/logging.h>
#include <Eigen/Eigenvalues>
#include "omp.h"
#include "sensor_fusion/lidar_fusion.h"
#include "xmap_util.h"

namespace lixel
{

void LiDARFusion::calculateMeas(VecX& residual, SparseMat& H, double& R)
{
  /** Point to Plane LiDAR Registration **/
  std::vector<LiDARMeasTemp> meas_vec_omp, meas_vec;
  int lidar_meas_total_size = 0;
  int search_success_num = 0;
  int plane_success_num = 0;
  if (!states_group_ptr_ || states_group_ptr_->windowSize() != window_size_)
    throw std::runtime_error("LiDAR fusion state window does not match configuration");

  for (int i = 0; i < window_size_; ++i)
    lidar_meas_total_size += static_cast<int>(sw_lidar_surf_[i]->size());
  meas_vec_omp.resize(lidar_meas_total_size);
  LOG(INFO) << "input point size:" << lidar_meas_total_size;

  int start_index = 0;
  for (int j = 0; j < window_size_; ++j)
  {
#ifdef MP_EN
#pragma omp parallel for schedule(dynamic) \
    reduction(+ : search_success_num, plane_success_num)
#endif
    for (int i = 0; i < sw_lidar_surf_[j]->size(); ++i)
    {
      PointXYZINormal& p = sw_lidar_surf_[j]->points[i];
      Vec3 p_I(p.x, p.y, p.z);
      double point_time_absolute = states_group_ptr_->timestamp;
      Vec3 p_W = states_group_ptr_->sw_rot_[j] * p_I + states_group_ptr_->sw_pos_[j];
      Vec3 pos = states_group_ptr_->sw_pos_[j];

      LiDARMatchCache& cache = sw_match_cache_[j][i];
      if (faster_model_ && cache.status != LiDARMatchStatus::ACTIVE)
        continue;

      xmap::PlaneConstPtr plane = cache.plane;
      if (!faster_model_ || refresh_matches_ || !plane)
      {
        bool success = xmap_ptr_->knnSearch(
            p_W.cast<xmap::FloatDataType>(),
            pos.cast<xmap::FloatDataType>(),
            point_time_absolute,
            plane);
        if (!success)
        {
          cache.status = LiDARMatchStatus::KNN_FAIL;
          cache.plane.reset();
          continue;
        }
        ++search_success_num;

        if (!plane || !plane->is_plane)
        {
          cache.status = LiDARMatchStatus::NO_PLANE;
          cache.plane.reset();
          continue;
        }
        cache.plane = plane;
      }
      else
      {
        ++search_success_num;
      }
      ++plane_success_num;

      p.normal_x = plane->normal[0];
      p.normal_y = plane->normal[1];
      p.normal_z = plane->normal[2];
      // outlier remove
      double dist_to_viewpoint = p_I.norm();
      double dist_to_plane =
          static_cast<double>(p_W.dot(plane->normal.template cast<FloatDataType>())) +
          static_cast<double>(plane->d);

      // FAST-LIO
      //    float s = 1 - 0.9 * fabs(dist_to_plane) / sqrt(dist_to_viewpoint);
      //    if (s < 0.9)
      //      continue;

      if (fabs(dist_to_plane) >
          knn_search_slope_ * dist_to_viewpoint + knn_search_min_dist_)
      {
        cache.status = LiDARMatchStatus::OUTLIER;
        continue;
      }

      // 0 <= weight <= 1
      double weight = 1 - plane->planarity;
      weight = std::min(1.0, weight);
      weight = std::max(0.0, weight);
      LiDARMeasTemp lidar_meas_temp;
      lidar_meas_temp.residual = dist_to_plane;
      lidar_meas_temp.p_I = p_I.cast<FloatDataType>();
      lidar_meas_temp.normal = plane->normal.cast<FloatDataType>();
      lidar_meas_temp.weight = weight;
      lidar_meas_temp.useful = true;
      lidar_meas_temp.index = j;
      cache.status = LiDARMatchStatus::ACTIVE;
      meas_vec_omp[start_index + i] = lidar_meas_temp;
    }
    start_index += sw_lidar_surf_[j]->size();
  }
  refresh_matches_ = false;

  // Prevent randomness caused by omp disorder
  for (auto& meas : meas_vec_omp)
  {
    if (!meas.useful)
      continue;
    meas_vec.emplace_back(meas);
  }
  /** H, r construction **/
  int row = static_cast<int>(meas_vec.size());
  residual = VecX::Zero(row, 1);
  const int dim_state = states_group_ptr_->dimState();
  H.resize(row, dim_state);
  Eigen::VectorXi sizes(row);
  sizes.setConstant(6);
  H.reserve(sizes);

  LOG(INFO) << "effect points num:" << meas_vec.size();
  attributeJacobi_ = AttributeJacobi{};
  attributeJacobi_.use_point_num = static_cast<uint32_t>(meas_vec.size());
  attributeJacobi_.total_point_num = static_cast<uint32_t>(lidar_meas_total_size);
  attributeJacobi_.search_success_num = static_cast<uint32_t>(search_success_num);
  attributeJacobi_.plane_success_num = static_cast<uint32_t>(plane_success_num);
  for (const auto& meas : meas_vec)
  {
    if (meas.index == 0)
      ++attributeJacobi_.current_use_point_num;
  }
  if (!sw_lidar_surf_[0]->empty())
  {
    attributeJacobi_.overlap_radio =
        static_cast<float>(attributeJacobi_.current_use_point_num) /
        static_cast<float>(sw_lidar_surf_[0]->size());
  }
  if (meas_vec.size() <= WARNING_POINTS_NUM)
  {
    LOG(WARNING) << "points num is not enough, maybe error: " << meas_vec.size();
  }

  if (meas_vec.size() <= ERROR_POINTS_NUM)
  {
    LOG(ERROR) << "Too few effective points, stop current update: " << meas_vec.size();
    residual.resize(0);
    H.resize(0, dim_state);
    R = 0.0;
    return;
  }

  // TODO: OPENMP PARALLEL !!!
  Eigen::Matrix<double, 6, 6> current_pose_information =
      Eigen::Matrix<double, 6, 6>::Zero();
  for (int i = 0; i < meas_vec.size(); i++)
  {
    Mat3 point_crossmat = skewSymMatrix(meas_vec[i].p_I);
    Vec3 normal = meas_vec[i].normal;
    double weight = meas_vec[i].weight;

    // 计算公共部分
    Eigen::Matrix<FloatDataType, 1, 3> J_rot =
        -normal.transpose() * states_group_ptr_->sw_rot_[meas_vec[i].index] * point_crossmat * weight;
    Eigen::Matrix<FloatDataType, 1, 3> J_pos = normal.transpose() * weight;
    int dim = (meas_vec[i].index == 0) ? 0 : DIM_CURR_STATE + (meas_vec[i].index - 1) * 6;

    H.insert(i, dim) = J_rot[0];
    H.insert(i, dim + 1) = J_rot[1];
    H.insert(i, dim + 2) = J_rot[2];

    H.insert(i, dim + 3) = J_pos[0];
    H.insert(i, dim + 4) = J_pos[1];
    H.insert(i, dim + 5) = J_pos[2];

    if (meas_vec[i].index == 0)
    {
      Eigen::Matrix<double, 1, 6> jacobian;
      jacobian << J_rot[0], J_rot[1], J_rot[2],
                  J_pos[0], J_pos[1], J_pos[2];
      current_pose_information.noalias() += jacobian.transpose() * jacobian;
    }

    residual(i) = meas_vec[i].residual * weight;
  }
  /** LiDAR Covariance Calculation **/
  double var = (residual.array() - residual.mean()).square().sum() / (residual.size() - 1);
  for (auto& p : meas_vec) p.var = var;

  R = var;
  attributeJacobi_.residual_mean = static_cast<float>(residual.mean());
  attributeJacobi_.residual_rms =
      static_cast<float>(std::sqrt(residual.squaredNorm() / residual.size()));
  if (attributeJacobi_.current_use_point_num > 0)
  {
    current_pose_information /= attributeJacobi_.current_use_point_num;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(
        current_pose_information, Eigen::EigenvaluesOnly);
    if (solver.info() == Eigen::Success)
      attributeJacobi_.current_pose_information_eig =
          solver.eigenvalues().cast<float>();
  }
  LOG(INFO) << "var:" << var;
  /** LOG intermediate results of LiDARFusion **/
  calculateAttrJacobi(meas_vec);
}

void LiDARFusion::calculateAttrJacobi(const std::vector<LiDARMeasTemp>& meas_vec)
{
  int numPoints = meas_vec.size();
  std::vector<V3F> data_points;
  std::vector<V3F> data_normal;
  for (int i = 0; i < numPoints; i++)
  {
    data_points.push_back(meas_vec[i].p_I.cast<float>());
    data_normal.push_back(meas_vec[i].normal.cast<float>());
  }
  V3F sigma_points, sigma_normal;
  M3F V_points, V_normal;
  SVD(data_points, V_points, sigma_points);
  SVD(data_normal, V_normal, sigma_normal);

  attributeJacobi_.point_eig << sigma_points(0), sigma_points(1), sigma_points(2);
  attributeJacobi_.norm_eig << sigma_normal(0), sigma_normal(1), sigma_normal(2);
  attributeJacobi_.flat_ness = sqrt(std::pow(sigma_points(2), 2) / (sigma_points(1) * sigma_points(0)));
  attributeJacobi_.smooth_ness = sqrt(std::pow(sigma_normal(2), 2) / (sigma_normal(1) * sigma_normal(0)));
  attributeJacobi_.use_point_num = static_cast<uint32_t>(meas_vec.size());
}

LiDARFusion::LiDARFusion(const IESKFParam& param)
    : window_size_(param.window_size),
      reset_window_size_(param.reset_window_size),
      faster_model_(param.faster_model),
      knn_search_slope_(param.knn_search_slope),
      knn_search_min_dist_(param.knn_search_min_dist)
{
  if (window_size_ < 1 || reset_window_size_ < 1)
    throw std::invalid_argument("LiDAR fusion window sizes must be positive");
  sw_lidar_surf_.resize(window_size_);
  sw_lidar_map_.resize(window_size_);
  sw_lidar_undistort_.resize(window_size_);
  sw_match_cache_.resize(window_size_);
  LOG(INFO) << "LiDAR fusion configured window_size=" << window_size_
            << " reset_window_size=" << reset_window_size_
            << " faster_model=" << faster_model_
            << " knn_search_slope=" << knn_search_slope_
            << " knn_search_min_dist=" << knn_search_min_dist_;
}

void LiDARFusion::setXmap(std::shared_ptr<xmap::Xmap>& xmap_ptr)
{
  xmap_ptr_ = xmap_ptr;
  for (int i = 0; i < window_size_; ++i)
    sw_lidar_surf_[i].reset(new PointCloudXYZINormal);
}

// TODO: fix the bug about normal filter
void LiDARFusion::setLidarMeas(
    const PointCloudXYZINormal::Ptr& surf_pcl,
    const PointCloudXYZINormal::Ptr& map_pcl,
    const PointCloud::Ptr& undistort_pcl)
{
  // lidar_meas_ sliding
  for (int i = window_size_ - 1; i >= 1; --i)
  {
    sw_lidar_surf_[i] = sw_lidar_surf_[i - 1];
    sw_lidar_map_[i] = sw_lidar_map_[i - 1];
    sw_lidar_undistort_[i] = sw_lidar_undistort_[i - 1];
    sw_match_cache_[i] = std::move(sw_match_cache_[i - 1]);
    if (i < reset_window_size_)
    {
      for (auto& cache : sw_match_cache_[i])
      {
        cache.status = LiDARMatchStatus::ACTIVE;
        cache.plane.reset();
      }
    }
  }
  // Keep the caller-owned shared clouds instead of deep copies.  The LiDAR
  // fusion writes fitted plane normals into surf_pcl during calculateMeas().
  // LioCore then propagates those normals to map_pcl before the delayed frame
  // is inserted into Xmap.  Deep-copying here disconnected those two stages,
  // so every inserted map point had a zero normal and Xmap replaced it with a
  // sensor viewing ray rather than the fitted surface normal.
  sw_lidar_surf_[0] = surf_pcl;
  sw_lidar_map_[0] = map_pcl;
  sw_lidar_undistort_[0] = undistort_pcl;
  sw_match_cache_[0].assign(surf_pcl->size(), LiDARMatchCache{});
  refresh_matches_ = true;
}

bool LiDARFusion::getUpdateFrame(PointCloudXYZINormal::Ptr& update_frame)
{
  update_frame = sw_lidar_map_.back();
  if (update_frame == nullptr)
    return false;
  else
    return true;
}

bool LiDARFusion::getPublishFrame(PointCloud::Ptr& publish_frame)
{
  publish_frame = sw_lidar_undistort_.back();
  if (publish_frame == nullptr)
    return false;
  else
    return true;
}

const AttributeJacobi& LiDARFusion::getAttributeJacobi() const
{
  return attributeJacobi_;
}

}  // namespace lixel
