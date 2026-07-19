//
// Created by youyuan on 24-2-18.
//

#include <omp.h>

#include <omp.h>
#include "log/lsLogger.h"
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
  for (int i = 0; i < WINDOW_SIZE; ++i) lidar_meas_total_size += sw_lidar_surf_[i]->size();
  meas_vec_omp.resize(lidar_meas_total_size);
  lslog(LSLOG_INFO) << "input point size:" << lidar_meas_total_size;

  int start_index = 0;
  for (int j = 0; j < WINDOW_SIZE; ++j)
  {
#ifdef MP_EN
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < sw_lidar_surf_[j]->size(); ++i)
    {
      PointXYZINormal& p = sw_lidar_surf_[j]->points[i];
      Vec3 p_I(p.x, p.y, p.z);
      double point_time_absolute = states_group_ptr_->timestamp;
      std::vector<xmap::V3F> nearest_points;
      Vec3 p_W = states_group_ptr_->sw_rot_[j] * p_I.cast<double>() + states_group_ptr_->sw_pos_[j];
      Vec3 pos = states_group_ptr_->sw_pos_[j];

      // TODO: KNN Search should not fix the param about min/max/search_dist
      bool success = xmap_ptr_->knnSearch(
          p_W.cast<xmap::FloatDataType>(), pos.cast<xmap::FloatDataType>(), nearest_points, point_time_absolute);
      if (!success)
        continue;

      xmap::PlanePtr plane = xmap::planeFitting(nearest_points, pos.cast<xmap::FloatDataType>());
      if (!plane->is_plane)
        continue;

      p.normal_x = plane->normal[0];
      p.normal_y = plane->normal[1];
      p.normal_z = plane->normal[2];
      // outlier remove
      double dist_to_viewpoint = p_I.norm();
      double dist_to_plane = p_W.dot(plane->normal.cast<double>()) + plane->d;

      // FAST-LIO
      //    float s = 1 - 0.9 * fabs(dist_to_plane) / sqrt(dist_to_viewpoint);
      //    if (s < 0.9)
      //      continue;

      if (fabs(dist_to_plane) > k_for_adaptive_search_ * dist_to_viewpoint + S_MIN)
        continue;

      // 0 <= weight <= 1
      double weight = 1 - plane->planarity;
      weight = std::min(1.0, weight);
      weight = std::max(0.0, weight);
      LiDARMeasTemp lidar_meas_temp;
      lidar_meas_temp.residual = dist_to_plane;
      lidar_meas_temp.p_I = p_I.cast<FloatDataType>();
      lidar_meas_temp.normal = plane->normal.cast<double>();
      lidar_meas_temp.weight = weight;
      lidar_meas_temp.useful = true;
      lidar_meas_temp.index = j;
      meas_vec_omp[start_index + i] = lidar_meas_temp;
    }
    start_index += sw_lidar_surf_[j]->size();
  }

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
  H.resize(row, DIM_STATE);
  Eigen::VectorXi sizes(row);
  sizes.setConstant(6);
  H.reserve(sizes);

  lslog(LSLOG_INFO) << "effect points num:" << meas_vec.size();
  if (meas_vec.size() <= WARNING_POINTS_NUM)
  {
    lslog(LSLOG_WARNING) << "effect points less than 30";
  }

  if (meas_vec.size() <= ERROR_POINTS_NUM)
  {
    lslog(LSLOG_ERROR) << "effect points less than 5";
    return;
  }

  // TODO: OPENMP PARALLEL !!!
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

    residual(i) = meas_vec[i].residual * weight;
  }
  /** LiDAR Covariance Calculation **/
  double var = (residual.array() - residual.mean()).square().sum() / (residual.size() - 1);
  for (auto& p : meas_vec) p.var = var;

  R = var;
  lslog(LSLOG_INFO) << "var:" << var;
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
  attributeJacobi_.total_point_num = static_cast<uint32_t>(sw_lidar_surf_[0]->size());
  attributeJacobi_.overlap_radio = float(meas_vec.size()) / float(sw_lidar_surf_[0]->size());
}

LiDARFusion::LiDARFusion(double k_for_adaptive_search) : k_for_adaptive_search_(k_for_adaptive_search)
{
}

void LiDARFusion::setXmap(std::shared_ptr<xmap::Xmap>& xmap_ptr)
{
  xmap_ptr_ = xmap_ptr;
  for (int i = 0; i < WINDOW_SIZE; ++i) sw_lidar_surf_[i].reset(new PointCloudXYZINormal);
}

// TODO: fix the bug about normal filter
void LiDARFusion::setLidarMeas(
    const PointCloudXYZINormal::Ptr& surf_pcl,
    const PointCloudXYZINormal::Ptr& map_pcl,
    const PointCloud::Ptr& undistort_pcl)
{
  PointCloudXYZINormal::Ptr surf_pcl_copy(new PointCloudXYZINormal(*surf_pcl));
  PointCloudXYZINormal::Ptr map_pcl_copy(new PointCloudXYZINormal(*map_pcl));
  PointCloud::Ptr undistort_pcl_copy(new PointCloud(*undistort_pcl));
  // lidar_meas_ sliding
  for (int i = WINDOW_SIZE - 1; i >= 1; --i)
  {
    sw_lidar_surf_[i] = sw_lidar_surf_[i - 1];
    sw_lidar_map_[i] = sw_lidar_map_[i - 1];
    sw_lidar_undistort_[i] = sw_lidar_undistort_[i - 1];
  }
  sw_lidar_surf_[0] = surf_pcl_copy;
  sw_lidar_map_[0] = map_pcl_copy;
  sw_lidar_undistort_[0] = undistort_pcl_copy;
}

bool LiDARFusion::getUpdateFrame(PointCloudXYZINormal::Ptr& update_frame)
{
  update_frame = sw_lidar_map_[WINDOW_SIZE - 1];
  if (update_frame == nullptr)
    return false;
  else
    return true;
}

bool LiDARFusion::getPublishFrame(PointCloud::Ptr& publish_frame)
{
  publish_frame = sw_lidar_undistort_[WINDOW_SIZE - 1];
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
