#include "xivt.hpp"

#include "context/logging.hpp"
#include "xutil/base/basic_definition.hpp"

#include <omp.h>
#include <sstream>

namespace xivt
{
XiVT_hyper_grid_context::~XiVT_hyper_grid_context()
{
  m_point_cloud = nullptr;
  m_kd_tree = nullptr;
  m_oc_tree = nullptr;
}

bool XiVT_node::Empty() const
{
  return m_point_offsets.empty();
}

std::size_t XiVT_node::Size() const
{
  return m_point_offsets.size();
}

int XiVT_node::incremental_knn(
    std::vector<Point_with_distance>& dis_points,
    const XiVT_point_type& point,
    const Vec3f& view_point,
    const int& K,
    float& max_squared_range,
    bool by_normal,
    XiVT_hyper_grid_context* src_context)
{
  std::size_t old_size = dis_points.size();
  const auto& p_vec = m_hyper_grid_context->m_point_cloud->points;
  // if (m_point_offsets.size() > 10) {
  //     exit(0);
  // }

  int indexed_till_position = -1;

  if (src_context != nullptr)
  {
    indexed_till_position = src_context->m_indexed_till;
  }

  if (src_context != m_hyper_grid_context)
  {
    indexed_till_position = -1;
  }

  for (int i = 0; i < m_point_offsets.size(); i++)
  {
    int pt_offset = m_point_offsets[i];

    if (pt_offset <= indexed_till_position)
    {
      continue;
    }

    const XiVT_point_type& pt = p_vec[pt_offset];

    double d = (pt.x - point.x) * (pt.x - point.x) + (pt.y - point.y) * (pt.y - point.y) +
               (pt.z - point.z) * (pt.z - point.z);

    if (d < max_squared_range)
    {
      // // filter via normal if needed
      // if (by_normal) {
      //     Vec3f pos_vec(pt.x,
      //                     pt.y,
      //                     pt.z);
      //     Vec3f norm_vec(pt.normal_x,
      //                     pt.normal_y,
      //                     pt.normal_z);

      //     Vec3f view_vec = view_point - pos_vec;

      //     float dot = view_vec.dot(norm_vec);
      //     if (dot < 0) {
      //         continue;
      //     }
      // }

      dis_points.template emplace_back(Point_with_distance(d, m_hyper_grid_context, pt_offset));
      if (dis_points.size() >= K)
      {
        // for (int off = 0; off < K; off++) {
        max_squared_range = std::min(dis_points[0].dist, max_squared_range);
        max_squared_range = std::min(dis_points[K - 1].dist, max_squared_range);
        // }
      }
      // if (dis_points.size() == K) {
      //     // for (int off = 0; off < K; off++) {
      //     max_squared_range = std::min(dis_points[0].dist, max_squared_range);
      //     // max_squared_range = std::min(dis_points[K - 1].dist, max_squared_range);
      //     // }
      // }
    }

    if (dis_points.size() - old_size == K)
      break;
  }
  // sort by distance
  // if (old_size + K >= dis_points.size()) {
  // } else {
  //     std::nth_element(dis_points.begin() + old_size, dis_points.begin() + old_size + K - 1,
  //     dis_points.end()); dis_points.resize(old_size + K);
  // }

  return dis_points.size();
}

static constexpr int t_hyper_grid_bin_offset = 3;

XiVT::XiVT(Options options) : m_options(options)
{
  m_options.m_downsample_ratio = 4;

  m_options.m_resolution = m_options.m_downsample_resolution * m_options.m_downsample_ratio;
  m_options.m_capacity = 10000000;

  m_options.m_inv_resolution = 1.0 / m_options.m_resolution;
  // m_options.m_downsample_resolution = m_options.m_resolution / m_options.m_downsample_ratio;
  m_options.m_downsample_resulution_inv = 1.0 / m_options.m_downsample_resolution;

  m_options.m_downsample_half_sphere_sq_distance =
      m_options.m_downsample_resolution * m_options.m_downsample_resolution / 2;

  GLOG(INFO) << xutil::strprintf(
      "[XiVT::XiVT] Setup finished, grid resolution = %f, inv = %f, downsample ratio = %d, "
      "downsample voxel resolution = %f, inv = %f, near = %d, capacity = %d\n",
      m_options.m_resolution,
      m_options.m_inv_resolution,
      m_options.m_downsample_ratio,
      m_options.m_downsample_resolution,
      m_options.m_downsample_resulution_inv,
      m_options.m_nearby_type,
      m_options.m_capacity);

  GLOG(INFO) << xutil::strprintf(
      "[XiVT::XiVT] With hyper grid size = %f\n",
      m_options.m_resolution * (2 << t_hyper_grid_bin_offset));

  GLOG(INFO) << xutil::strprintf(
      "[XiVT::XiVT] sizeof(XiVT_point_type) = %zu, sizeof(XiVT_node) = %zu\n",
      sizeof(XiVT_point_type),
      sizeof(XiVT_node));

  GenerateNearbyGrids();
}

bool esti_plane_dynamic(
    Plane& plane,
    const XiVT_point_vector& points,
    const float& threshold,
    float& planarity)
{
  if (points.size() < 3)
    return false;

  Eigen::Matrix<float, -1, 3> A;
  Eigen::Matrix<float, -1, 1> b;
  A.resize(points.size(), 3);
  b.resize(points.size());
  A.setZero();
  b.setOnes();
  b *= -1.0f;

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  int points_size = points.size();
  for (int j = 0; j < points_size; j++)
  {
    Eigen::Vector3d pv(points[j].x, points[j].y, points[j].z);
    covariance += pv * pv.transpose();
    center += pv;
  }

  center = center / points_size;
  covariance = covariance / points_size - center * center.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance);
  Eigen::VectorXd D = eig.eigenvalues();
  planarity = fabs(D(2)) > 1e-8 ? (D(1) - D(0)) / D(2) : 0;
  double planer_threshold = 0.01;
  if (D(0) > planer_threshold || planarity < 0.1)
    return false;

  // 1207
  // double planer_threshold = 0.001;
  // if (D(0) > planer_threshold || planarity < 0.3)
  //     return false;

  // int inliers = 0;
  double inlier_ratio = 0;
  Eigen::Vector3d normal = eig.eigenvectors().col(0);
  double d = -normal.dot(center);
  // for (int j = 0; j < points_size; j++)
  // {
  //     Eigen::Vector3d pv(points[j].x, points[j].y, points[j].z);
  //     double dist = fabs(pv.dot(normal) + d);
  //     if (dist < threshold)
  //         inliers++;
  // }
  // inlier_ratio = (double)inliers / points_size;
  double standard_deviation = sqrt(D(0));

  // result
  plane.normal = normal;
  plane.center = center;
  plane.covariance = covariance;
  plane.eigen_values = eig.eigenvalues();
  plane.eigen_vectors = eig.eigenvectors();
  plane.d = d;
  plane.planarity = planarity;
  plane.inlier_ratio = inlier_ratio;
  plane.standard_deviation = standard_deviation;
  plane.valid = true;

  return true;
}

bool esti_plane_dynamic(
    Eigen::Matrix<float, 9, 1>& pca_result,
    const XiVT_point_vector& points,
    const float& threshold,
    float& planarity)
{
  Eigen::Matrix<float, -1, 3> A;
  Eigen::Matrix<float, -1, 1> b;
  A.resize(points.size(), 3);
  b.resize(points.size());
  A.setZero();
  b.setOnes();
  b *= -1.0f;

  Eigen::Matrix3d plane_covariance = Eigen::Matrix3d::Zero();
  Eigen::Vector3d plane_center = Eigen::Vector3d::Zero();
  int points_size = points.size();
  for (int j = 0; j < points_size; j++)
  {
    Eigen::Vector3d pv(points[j].x, points[j].y, points[j].z);
    plane_covariance += pv * pv.transpose();
    plane_center += pv;
  }

  plane_center = plane_center / points_size;
  plane_covariance = plane_covariance / points_size - plane_center * plane_center.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(plane_covariance);
  Eigen::VectorXd D = eig.eigenvalues();
  Eigen::Vector3d n = eig.eigenvectors().col(0);
  n.normalize();
  planarity = fabs(D(2)) > 1e-8 ? (D(1) - D(0)) / D(2) : 0;

  double planer_threshold = 0.01;
  if (D(0) > planer_threshold || planarity < 0.1)
    return false;

  for (int j = 0; j < points.size(); j++)
  {
    A(j, 0) = points[j].x;
    A(j, 1) = points[j].y;
    A(j, 2) = points[j].z;
  }

  // Matrix<float, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
  Eigen::Vector3d evals_real;
  evals_real = D.real();

  Eigen::Matrix3f::Index evals_max, evals_min;
  evals_real.rowwise().sum().maxCoeff(&evals_max);
  evals_real.rowwise().sum().minCoeff(&evals_min);
  int evals_mid = 3 - evals_max - evals_min;
  float min_eigen_value = evals_real(evals_min);
  float mid_eigen_value = evals_real(evals_mid);
  float max_eigen_value = evals_real(evals_max);
  double plane_radius = 3.5 * sqrt(max_eigen_value);

  // plane_center.normalize();

  // T inv_norm = 1.0/normvec.norm();
  pca_result(0) = n(0);
  pca_result(1) = n(1);
  pca_result(2) = n(2);
  pca_result(3) = -plane_center.dot(n);
  pca_result(4) = plane_center(0);
  pca_result(5) = plane_center(1);
  pca_result(6) = plane_center(2);
  pca_result(7) = plane_radius;
  pca_result(8) = planarity;

  for (int j = 0; j < points.size(); j++)
  {
    if (fabs(
            pca_result(0) * points[j].x + pca_result(1) * points[j].y +
            pca_result(2) * points[j].z + pca_result(3)) > threshold)
    {
      return false;
    }
  }
  return true;
}

#ifdef V1207
XiVT_knn_query_status XiVT::get_closest_point(
    const Point_type& query_point,
    const Vec3f& view_point,
    Point_vector& closest_pt,
    Eigen::Matrix<float, 9, 1>& pca_result,
    int _min_num,
    int _max_num,
    double max_range,
    bool by_normal,
    int zero_dot_filter_num)
{
  std::vector<Point_with_distance> candidates;
  int max_num_local = _max_num;

  // firstly, filter by distance

  Vec3f pos_vec(query_point.x, query_point.y, query_point.z);

  Vec3f view_vec = view_point - pos_vec;

  float distance = view_vec.norm();

  // float ratio = 1 - 1.0 / (sqrtf(distance) / 2.0 + 1);
  float ratio = 1;

  XiVT_knn_query_status ret;

  static __thread unsigned int seed = omp_get_thread_num();

  if (rand_r(&seed) * 1.0 / RAND_MAX >= ratio)
  {
    return ret;
  }

  if (by_normal)
  {
    max_num_local *= 2;
  }

  candidates.reserve(max_num_local);

  auto pt_eigen = ToEigen<float, t_dim>(query_point);

  // XiVT_voxel_index node_key = Pos2Grid(pt_eigen);
  XiVT_key_type voxel_key = position_to_voxel(pt_eigen);

  XiVT_key_type added_voxel_key = voxel_key;
  added_voxel_key[0] += 1 << 30;
  added_voxel_key[1] += 1 << 30;
  added_voxel_key[2] += 1 << 30;

  XiVT_key_type node_key = added_voxel_key / m_options.m_downsample_ratio;

  // auto node_key = Pos2Grid(ToEigen<float, t_dim>(query_point));

  float max_squared_range = max_range * max_range;

  XiVT_hyper_grid_context* center_context = nullptr;
  bool instant_break = false;

  decltype(m_nodes_map)::iterator iter, first_iter;
  bool first_node_used = false;

  for (const XiVT_key_type& delta : m_nearby_grids)
  {
    auto dkey = node_key + delta;
    iter = m_nodes_map.find(dkey);
    if (iter != m_nodes_map.end())
    {
      if (center_context == nullptr && !first_node_used)
      {
        if (!iter->second->second.m_valid)
        {
          break;
        }
        center_context = iter->second->second.m_hyper_grid_context;
        if (center_context->m_indexed_till != -1)
        {
          // search in KDTree
          std::vector<int> k_indices;
          std::vector<float> k_sqr_distances;

          // int knn_num = max_num;
          // if (by_normal) {
          //     knn_num *= 2;
          // }
          center_context->m_kd_tree->nearestKSearch(
              query_point, max_num_local, k_indices, k_sqr_distances);

          for (int i = 0; i < k_indices.size(); i++)
          {
            if (k_sqr_distances[i] < max_squared_range)
            {
              // if (by_normal) {
              //     Vec3f pos_vec(center_context->m_point_cloud->points[k_indices[i]].x,
              //                   center_context->m_point_cloud->points[k_indices[i]].y,
              //                   center_context->m_point_cloud->points[k_indices[i]].z);
              //     Vec3f norm_vec(center_context->m_point_cloud->points[k_indices[i]].normal_x,
              //                   center_context->m_point_cloud->points[k_indices[i]].normal_y,
              //                   center_context->m_point_cloud->points[k_indices[i]].normal_z);

              //     Vec3f view_vec = view_point - pos_vec;

              //     float dot = view_vec.dot(norm_vec);
              //     if (dot < 0) {
              //         continue;
              //     }
              // }
              candidates.emplace_back(k_sqr_distances[i], center_context, k_indices[i]);
            }
          }

          if (candidates.size() >= max_num_local && m_options.m_instant_break_policy != 0)
          {
            instant_break = true;
            break;
          }
        }
      }
      auto tmp = iter->second->second.incremental_knn(
          candidates,
          query_point,
          view_point,
          max_num_local,
          max_squared_range,
          by_normal,
          center_context);
    }
    if (!first_node_used)
    {
      first_node_used = true;
      first_iter = iter;
      // if (candidates.empty()) {  // bad for precision
      //     return ret;
      // }
    }
  }

  if (candidates.empty())
  {
    return ret;
  }

  if (!instant_break)
  {
    if (candidates.size() <= max_num_local)
    {
    }
    else
    {
      std::nth_element(
          candidates.begin(), candidates.begin() + max_num_local - 1, candidates.end());
      candidates.resize(max_num_local);
    }
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());
  }

  ret.m_knn_count = candidates.size();

  closest_pt.clear();
  for (auto& it : candidates)
  {
    auto candidate_point = it.m_hyper_grid->m_point_cloud->points[it.idx];
    if (by_normal)
    {
      // Vec3f pos_vec(candidate_point.x,
      //               candidate_point.y,
      //               candidate_point.z);
      Vec3f norm_vec(candidate_point.normal_x, candidate_point.normal_y, candidate_point.normal_z);

      Vec3f view_vec = view_point - pos_vec;

      // float dot = view_vec.dot(norm_vec);
      float dot = view_vec.transpose() * norm_vec;
      // if (dot < 0 || (dot == 0 && rand_r(&seed) % 2 == 0)) {
      //     ret.m_filtered_by_normal_count++;
      //     continue;
      // }
      if (dot <= 0)
      {
        if (dot == 0)
        {
          ret.m_zero_normal_count++;
          if ((ret.m_zero_normal_count + 1) % zero_dot_filter_num != 0)
          {
            ret.m_filtered_by_zero_normal_count++;
            continue;
          }
        }
        else
        {
          ret.m_filtered_by_normal_count++;
          continue;
        }
      }
    }
    closest_pt.emplace_back(candidate_point);
    if (closest_pt.size() >= _max_num)
    {
      break;
    }
  }

  if (ret.m_filtered_by_normal_count >= closest_pt.size() * m_options.m_rev_normal_threshold)
  {
    ret.m_filtered_via_rev = true;
    return ret;
  }
  float close_sq_distance_limit =
      (m_options.m_close_normal_threshold * m_options.m_downsample_resolution) *
      (m_options.m_close_normal_threshold * m_options.m_downsample_resolution);

  if (closest_pt.size() >= _min_num)
  {
    ret.m_valid_plane = esti_plane_dynamic(pca_result, closest_pt, 0.1f, ret.m_planarity);
    if (!ret.m_valid_plane)
    {
      ret.m_filtered_via_plane = true;
    }

    // also, give normal, if very close, and with good planarity
    // if (by_normal && omp_get_thread_num() == 0) {
    if (ret.m_valid_plane && m_options.m_use_dynamic_normal && by_normal &&
        ret.m_planarity > m_options.m_dynamic_normal_planarity_threshold)
    {
      for (auto& it : candidates)
      {
        auto& pt = it.m_hyper_grid->m_point_cloud->points[it.idx];
        auto& status = it.m_hyper_grid->m_point_status[it.idx];

        float sq_distance = (pt.x - pos_vec[0]) * (pt.x - pos_vec[0]) +
                            (pt.y - pos_vec[1]) * (pt.y - pos_vec[1]) +
                            (pt.z - pos_vec[2]) * (pt.z - pos_vec[2]);

        if (close_sq_distance_limit > sq_distance &&
            ((pt.normal_x == 0.0f && pt.normal_y == 0.0f && pt.normal_z == 0.0f) ||
             !status.m_with_valid_normal))
        {
          Vec3f pos_vec(pt.x, pt.y, pt.z);
          Vec3f norm_vec(pca_result(0), pca_result(1), pca_result(2));

          Vec3f view_vec = view_point - pos_vec;

          float dot = view_vec.transpose() * norm_vec;
          float coeff = dot > 0 ? 1 : -1;

          norm_vec *= coeff;

          {
            // check point in front of plane
            Vec3f norm_center_vec(pca_result(4), pca_result(5), pca_result(6));
            Vec3f center_to_point_vec = pos_vec - norm_center_vec;
            float dot = center_to_point_vec.transpose() * norm_vec;
            if (dot < 0)
            {
              // behind the plane, bad
              continue;
            }
          }
          // estimate the angle of view-vec and norm-vec
          float angle = acos(std::abs(dot) / (view_vec.norm() * norm_vec.norm()));
          float angle_deg = angle * 180 / M_PI;

          // GLOG(INFO) << "angle_deg: " << angle_deg << " " << dot << " " << view_vec.norm() << " "
          // << norm_vec.norm();

          // only keep norm with small angle (less than 60 deg)
          if (angle_deg < m_options.m_dynamic_normal_deg_max)
          {
            // status.m_with_valid_normal = true;
            // compare with stored deg

            Simple_spinlock_guard spinlock_guard(&m_lock_table.m_spinlocks[it.idx % 1024]);
            if (status.m_fake_normal_angle > angle_deg)
            {
              status.m_fake_normal_angle = angle_deg;

              pt.normal_x = norm_vec(0);
              pt.normal_y = norm_vec(1);
              pt.normal_z = norm_vec(2);
            }
          }
        }
      }
    }
  }
  else
  {
    ret.m_filtered_via_count = true;
  }

  return ret;
}
#else
XiVT_knn_query_status XiVT::get_closest_point(
    const XiVT_point_type& query_point,
    const Vec3f& view_point,
    XiVT_point_vector& closest_pt,
    Plane& plane_result,
    int _min_num,
    int _max_num,
    double max_range,
    bool by_normal,
    int zero_dot_filter_num)
{
  std::vector<Point_with_distance> candidates;
  int max_num_local = _max_num;

  XiVT_point_type query_point_local = query_point;

  // firstly, filter by distance

  Vec3f pos_vec(query_point.x, query_point.y, query_point.z);

  Vec3f view_vec = view_point - pos_vec;

  float distance = view_vec.norm();

  // float ratio = 1 - 1.0 / (sqrtf(distance) / 2.0 + 1);
  float ratio = 1;

  XiVT_knn_query_status ret;

  static __thread unsigned int seed = omp_get_thread_num();

  if (rand_r(&seed) * 1.0 / RAND_MAX >= ratio)
  {
    return ret;
  }

  if (by_normal)
  {
    max_num_local *= 2;  // TODO: -> 2.2
  }

  candidates.reserve(max_num_local);

  auto pt_eigen = ToEigen<float, t_dim>(query_point);

  // XiVT_voxel_index node_key = Pos2Grid(pt_eigen);
  XiVT_key_type voxel_key = position_to_voxel(pt_eigen);

  XiVT_key_type added_voxel_key = voxel_key;
  added_voxel_key[0] += 1 << 30;
  added_voxel_key[1] += 1 << 30;
  added_voxel_key[2] += 1 << 30;

  XiVT_key_type node_key = added_voxel_key / m_options.m_downsample_ratio;

  // auto node_key = Pos2Grid(ToEigen<float, t_dim>(query_point));

  float max_squared_range = max_range * max_range;

  XiVT_hyper_grid_context* center_context = nullptr;
  bool instant_break = false;

  decltype(m_nodes_map)::iterator iter, first_iter;
  bool first_node_used = false;

  for (const XiVT_key_type& delta : m_nearby_grids)
  {
    auto dkey = node_key + delta;
    iter = m_nodes_map.find(dkey);
    if (iter != m_nodes_map.end())
    {
      if (center_context == nullptr && !first_node_used)
      {
        if (!iter->second->second.m_valid)
        {
          break;
        }
        center_context = iter->second->second.m_hyper_grid_context;
        if (center_context->m_indexed_till != -1)
        {
          // search in KDTree
          std::vector<int> k_indices;
          std::vector<float> k_sqr_distances;

          // int knn_num = max_num;
          // if (by_normal) {
          //     knn_num *= 2;
          // }
          center_context->m_kd_tree->nearestKSearch(
              query_point_local, max_num_local, k_indices, k_sqr_distances);

          for (int i = 0; i < k_indices.size(); i++)
          {
            if (k_sqr_distances[i] < max_squared_range)
            {
              candidates.emplace_back(k_sqr_distances[i], center_context, k_indices[i]);
            }
          }

          if (candidates.size() >= max_num_local && m_options.m_instant_break_policy != 0)
          {
            instant_break = true;
            break;
          }
        }
      }
      auto tmp = iter->second->second.incremental_knn(
          candidates,
          query_point_local,
          view_point,
          max_num_local,
          max_squared_range,
          by_normal,
          center_context);
    }
    if (!first_node_used)
    {
      first_node_used = true;
      first_iter = iter;
      // if (candidates.empty()) {  // bad for precision
      //     return ret;
      // }
    }
  }

  if (candidates.empty())
  {
    return ret;
  }

  if (!instant_break)
  {
    if (candidates.size() <= max_num_local)
    {
    }
    else
    {
      std::nth_element(
          candidates.begin(), candidates.begin() + max_num_local - 1, candidates.end());
      candidates.resize(max_num_local);
    }
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());
  }

  ret.m_knn_count = candidates.size();

  closest_pt.clear();
  for (auto& it : candidates)
  {
    auto candidate_point = it.m_hyper_grid->m_point_cloud->points[it.idx];
    Vec3f norm_vec = it.m_hyper_grid->m_point_status[it.idx].get_normal();
    if (by_normal)
    {
      // Vec3f pos_vec(candidate_point.x,
      //               candidate_point.y,
      //               candidate_point.z);
      // Vec3f norm_vec(candidate_point.normal_x,
      //               candidate_point.normal_y,
      //               candidate_point.normal_z);

      Vec3f view_vec = view_point - pos_vec;

      // float dot = view_vec.dot(norm_vec);
      float dot = view_vec.transpose() * norm_vec;
      // if (dot < 0 || (dot == 0 && rand_r(&seed) % 2 == 0)) {
      //     ret.m_filtered_by_normal_count++;
      //     continue;
      // }

      // filt points far away from current query plane
      // if (query_point.normal_x != 0 || query_point.normal_y != 0 || query_point.normal_z != 0 &&
      // query_point.curvature != 0)  {
      //     auto d = query_point.normal_x * candidate_point.x +
      //              query_point.normal_y * candidate_point.y +
      //              query_point.normal_z * candidate_point.z +
      //              query_point.curvature;

      //     Vec3f cand_vec(candidate_point.x, candidate_point.y, candidate_point.z);
      //     Vec3f que_norm(query_point.normal_x, query_point.normal_y, query_point.normal_z);
      //     Vec3f p_vec = cand_vec - pos_vec;
      //     auto d1 = p_vec.dot(que_norm)/p_vec.norm();
      //     // GLOG(WARNING) << "p:" << pos_vec.transpose();
      //     // GLOG(WARNING) << "p_norm:" << que_norm.transpose() << " " << query_point.curvature;
      //     // GLOG(WARNING) << "candidate:" << cand_vec.transpose();
      //     // GLOG(WARNING) << d << "_" << p_vec.norm() <<"_"<<d1;
      //     if (fabs(d) > 0.2)
      //     {
      //         ret.m_filtered_by_plane_dist_count++;
      //         continue;
      //     }
      // }

      if (dot <= 0)
      {
        if (dot == 0)
        {
          ret.m_zero_normal_count++;
          if ((ret.m_zero_normal_count + 1) % zero_dot_filter_num != 0)
          {
            ret.m_filtered_by_zero_normal_count++;
            continue;
          }
        }
        else
        {
          ret.m_filtered_by_normal_count++;
          continue;
        }
      }
    }
    Point_type p;
    closest_pt.emplace_back(candidate_point);
    if (closest_pt.size() >= _max_num)
    {
      break;
    }
  }

  if (ret.m_filtered_by_normal_count >= closest_pt.size() * m_options.m_rev_normal_threshold)
  {
    ret.m_filtered_via_rev = true;
    return ret;
  }
  float close_sq_distance_limit =
      (m_options.m_close_normal_threshold * m_options.m_downsample_resolution) *
      (m_options.m_close_normal_threshold * m_options.m_downsample_resolution);

  if (closest_pt.size() >= _min_num)
  {
    ret.m_valid_plane = esti_plane_dynamic(plane_result, closest_pt, 0.1f, ret.m_planarity);
    if (!ret.m_valid_plane)
    {
      ret.m_filtered_via_plane = true;
    }

    // also, give normal, if very close, and with good planarity
    // if (by_normal && omp_get_thread_num() == 0) {
    if (ret.m_valid_plane && m_options.m_use_dynamic_normal && by_normal &&
        ret.m_planarity > m_options.m_dynamic_normal_planarity_threshold)
    {
      for (auto& it : candidates)
      {
        auto& pt = it.m_hyper_grid->m_point_cloud->points[it.idx];
        auto& status = it.m_hyper_grid->m_point_status[it.idx];
        Vec3f norm_pt = status.get_normal();

        float sq_distance = (pt.x - pos_vec[0]) * (pt.x - pos_vec[0]) +
                            (pt.y - pos_vec[1]) * (pt.y - pos_vec[1]) +
                            (pt.z - pos_vec[2]) * (pt.z - pos_vec[2]);

        if (close_sq_distance_limit > sq_distance &&
            ((norm_pt(0) == 0.0f && norm_pt(1) == 0.0f && norm_pt(2) == 0.0f) ||
             !status.m_with_valid_normal))
        {
          Vec3f pos_vec(pt.x, pt.y, pt.z);
          Vec3f norm_vec(plane_result.normal(0), plane_result.normal(1), plane_result.normal(2));

          Vec3f view_vec = view_point - pos_vec;

          float dot = view_vec.transpose() * norm_vec;
          float coeff = dot > 0 ? 1 : -1;

          norm_vec *= coeff;

          {
            // check point in front of plane
            Vec3f norm_center_vec(
                plane_result.center(0), plane_result.center(1), plane_result.center(2));
            Vec3f center_to_point_vec = pos_vec - norm_center_vec;
            float dot = center_to_point_vec.transpose() * norm_vec;
            if (dot < 0)
            {
              // behind the plane, bad
              continue;
            }
          }
          // estimate the angle of view-vec and norm-vec
          float angle = acos(std::abs(dot) / (view_vec.norm() * norm_vec.norm()));
          float angle_deg = angle * 180 / M_PI;

          // GLOG(INFO) << "angle_deg: " << angle_deg << " " << dot << " " << view_vec.norm() << " "
          // << norm_vec.norm();

          // only keep norm with small angle (less than 60 deg)
          if (angle_deg < m_options.m_dynamic_normal_deg_max)
          {
            // status.m_with_valid_normal = true;
            // compare with stored deg

            Simple_spinlock_guard spinlock_guard(&m_lock_table.m_spinlocks[it.idx % 1024]);
            if (status.m_fake_normal_angle > angle_deg)
            {
              status.m_fake_normal_angle = angle_deg;

              status.set_normal(norm_vec);
            }
          }
        }
      }
    }
  }
  else
  {
    ret.m_filtered_via_count = true;
  }

  return ret;
}
#endif

size_t XiVT::NumValidGrids() const
{
  return m_nodes_map.size();
}

void XiVT::GenerateNearbyGrids()
{
  if (m_options.m_nearby_type == NearbyType::CENTER)
  {
    m_nearby_grids.emplace_back(XiVT_key_type::Zero());
  }
  else if (m_options.m_nearby_type == NearbyType::NEARBY6)
  {
    m_nearby_grids = {
        XiVT_key_type(0, 0, 0),
        XiVT_key_type(-1, 0, 0),
        XiVT_key_type(1, 0, 0),
        XiVT_key_type(0, 1, 0),
        XiVT_key_type(0, -1, 0),
        XiVT_key_type(0, 0, -1),
        XiVT_key_type(0, 0, 1)};
  }
  else if (m_options.m_nearby_type == NearbyType::NEARBY18)
  {
    m_nearby_grids = {
        XiVT_key_type(0, 0, 0),
        XiVT_key_type(-1, 0, 0),
        XiVT_key_type(1, 0, 0),
        XiVT_key_type(0, 1, 0),
        XiVT_key_type(0, -1, 0),
        XiVT_key_type(0, 0, -1),
        XiVT_key_type(0, 0, 1),
        XiVT_key_type(1, 1, 0),
        XiVT_key_type(-1, 1, 0),
        XiVT_key_type(1, -1, 0),
        XiVT_key_type(-1, -1, 0),
        XiVT_key_type(1, 0, 1),
        XiVT_key_type(-1, 0, 1),
        XiVT_key_type(1, 0, -1),
        XiVT_key_type(-1, 0, -1),
        XiVT_key_type(0, 1, 1),
        XiVT_key_type(0, -1, 1),
        XiVT_key_type(0, 1, -1),
        XiVT_key_type(0, -1, -1)};
  }
  else if (m_options.m_nearby_type == NearbyType::NEARBY26)
  {
    m_nearby_grids = {
        XiVT_key_type(0, 0, 0),   XiVT_key_type(-1, 0, 0),  XiVT_key_type(1, 0, 0),
        XiVT_key_type(0, 1, 0),   XiVT_key_type(0, -1, 0),  XiVT_key_type(0, 0, -1),
        XiVT_key_type(0, 0, 1),   XiVT_key_type(1, 1, 0),   XiVT_key_type(-1, 1, 0),
        XiVT_key_type(1, -1, 0),  XiVT_key_type(-1, -1, 0), XiVT_key_type(1, 0, 1),
        XiVT_key_type(-1, 0, 1),  XiVT_key_type(1, 0, -1),  XiVT_key_type(-1, 0, -1),
        XiVT_key_type(0, 1, 1),   XiVT_key_type(0, -1, 1),  XiVT_key_type(0, 1, -1),
        XiVT_key_type(0, -1, -1), XiVT_key_type(1, 1, 1),   XiVT_key_type(-1, 1, 1),
        XiVT_key_type(1, -1, 1),  XiVT_key_type(1, 1, -1),  XiVT_key_type(-1, -1, 1),
        XiVT_key_type(-1, 1, -1), XiVT_key_type(1, -1, -1), XiVT_key_type(-1, -1, -1)};
  }
  else
  {
    LOG(ERROR) << "Unknown nearby_type!";
  }
}

Eigen::Matrix<float, t_dim, 1> XiVT::get_voxel_center_for_point(float x, float y, float z)
{
  Eigen::Matrix<float, t_dim, 1> pt_eigen;
  pt_eigen[0] = x;
  pt_eigen[1] = y;
  pt_eigen[2] = z;
  XiVT_key_type voxel_key = position_to_voxel(pt_eigen);
  Eigen::Matrix<float, t_dim, 1> voxel_center =
      voxel_key.template cast<float>() * m_options.m_downsample_resolution;
  return voxel_center;
}

// template <class XiVT_point_type>
// int get_direction_key_inner(const Point_type& pt) {
//     return pt.normal_x + pt.normal_y + pt.normal_z > 0 ? 1 : 0;
//     // return 0;
// }

int get_direction_key_inner(Vec3f pt)
{
  return pt.x() + pt.y() + pt.z() > 0 ? 1 : 0;
  // return 0;
}

template <class XiVT_point_type>
int get_direction_key_voxel_map(const XiVT_point_type& pt)
{
  // return pt.normal_x + pt.normal_y + pt.normal_z > 0 ? 1 : 0;
  return 0;
}

// TODO: order points by hyper grid, and add them once
// #pragma GCC optimize("O0")
void XiVT::add_points(
    const Point_vector& points_to_add,
    int current_frame_id,
    const std::vector<XiVT_point_status>& point_status,
    bool normal_need_recompute)
{
  // std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), [this](const
  // auto& pt) {
  std::lock_guard<std::mutex> lock(m_mutex);

  // #pragma omp parallel for
  for (int i = 0; i < points_to_add.size(); i++)
  {
    const auto& pt = points_to_add[i];
    XiVT_point_type pt_xivt;
    pt_xivt.x = pt.x;
    pt_xivt.y = pt.y;
    pt_xivt.z = pt.z;

    // TODO: do not use round index
    auto pt_eigen = ToEigen<float, t_dim>(pt);
    XiVT_key_type voxel_key = position_to_voxel(pt_eigen);
    // int direction_key = get_direction_key_voxel_map(pt);
    int direction_key = 0;
    Eigen::Matrix<float, t_dim, 1> voxel_center =
        voxel_key.template cast<float>() * m_options.m_downsample_resolution;
    float sq_distance_to_center = (pt_eigen - voxel_center).squaredNorm();

    // if (sq_distance_to_center > m_options.m_downsample_half_sphere_sq_distance) {
    //     continue;
    // }

    // XiVT_voxel_index node_key = Pos2Grid(pt_eigen);

    XiVT_key_type added_voxel_key = voxel_key;
    added_voxel_key[0] += 1 << 30;
    added_voxel_key[1] += 1 << 30;
    added_voxel_key[2] += 1 << 30;

    XiVT_key_type node_key = added_voxel_key / m_options.m_downsample_ratio;

    // printf("node_key = %d %d %d, voxel_key = %d %d %d\n", node_key[0], node_key[1], node_key[2],
    // voxel_key[0], voxel_key[1], voxel_key[2]);

    XiVT_key_type hyper_grid_key = node_key / (2 << t_hyper_grid_bin_offset);

    if (m_hyper_grids.count(hyper_grid_key) == 0)
    {
      m_hyper_grids[hyper_grid_key] =
          new XiVT_hyper_grid_context(m_options.m_downsample_resolution);
    }

    XiVT_hyper_grid_context* hyper_grid_ptr = m_hyper_grids.at(hyper_grid_key);

    auto iter = m_nodes_map.find(node_key);

    if (iter == m_nodes_map.end())
    {
      // center.getVector3fMap() = node_key.template cast<float>() * m_options.m_resolution;

      m_grids_lru_cache.push_front({node_key, XiVT_node(node_key)});
      iter = m_nodes_map.insert({node_key, m_grids_lru_cache.begin()}).first;

      hyper_grid_ptr->m_nodes.push_back(&m_grids_lru_cache.front().second);

      m_grids_lru_cache.front().second.m_hyper_grid_context = hyper_grid_ptr;
      int sz = hyper_grid_ptr->m_point_cloud->points.size();
      hyper_grid_ptr->m_point_cloud->push_back(pt_xivt);
      // hyper_grid_ptr->m_deleted_mark.push_back(false);
      hyper_grid_ptr->m_point_status.push_back(point_status.at(i));
      hyper_grid_ptr->m_point_status.back().m_squared_distances_to_center =
          XiVT_fake_float_01(sq_distance_to_center * m_options.m_downsample_resulution_inv);
      hyper_grid_ptr->m_point_status.back().m_frame_id = current_frame_id;
      hyper_grid_ptr->m_point_status.back().set_normal(pt.normal_x, pt.normal_y, pt.normal_z);

      // hyper_grid_ptr->m_squared_distances_to_center.push_back(sq_distance_to_center);
      // hyper_grid_ptr->m_frame_id.push_back(current_frame_id);
      if (!point_status.at(i).m_with_valid_normal && normal_need_recompute)
      {
        hyper_grid_ptr->m_points_without_normal_offsets.push_back(
            hyper_grid_ptr->m_point_status.size() - 1);
      }

      iter->second->second.m_point_offsets.push_back(sz);
      hyper_grid_ptr->m_down_sample_voxels[voxel_key] = sz;

      if (m_nodes_map.size() >= m_options.m_capacity)
      {
        // batch erase: erase multiple number of grids at the end of LRU
        auto end_iter = m_grids_lru_cache.end();
        int erased_grid = 0, iter_id = 0;
        while (erased_grid < m_options.m_erase_batch_size)
        {
          end_iter--;
          if (iter_id % m_options.m_erase_skip == 0)
          {
            // TODO: consider also erase its near26

            XiVT_hyper_grid_context* src_hyper_grid = end_iter->second.m_hyper_grid_context;
            for (int point_offset : end_iter->second.m_point_offsets)
            {
              // mark points as deleted, but to not erase them
              src_hyper_grid->m_point_status[point_offset].m_deleted_mark = true;
            }

            src_hyper_grid->m_erased_point_count += end_iter->second.m_point_offsets.size();
            src_hyper_grid->m_erased_grid_count++;

            m_nodes_map.erase(end_iter->first);
            m_grids_lru_cache.erase(end_iter);
          }
          iter_id++;
        }
      }
    }
    else
    {
      auto voxel_iter = hyper_grid_ptr->m_down_sample_voxels.find(voxel_key);
      if (voxel_iter == hyper_grid_ptr->m_down_sample_voxels.end())
      {
        int sz = hyper_grid_ptr->m_point_cloud->points.size();
        // manual reserve
        if (sz > 100 && sz + 2 > hyper_grid_ptr->m_point_cloud->points.capacity())
        {
          hyper_grid_ptr->m_point_cloud->points.reserve(sz + 100);
          hyper_grid_ptr->m_point_status.reserve(sz + 100);
        }
        hyper_grid_ptr->m_point_cloud->push_back(pt_xivt);
        // hyper_grid_ptr->m_deleted_mark.push_back(false);
        hyper_grid_ptr->m_point_status.push_back(point_status.at(i));
        hyper_grid_ptr->m_point_status.back().m_squared_distances_to_center =
            XiVT_fake_float_01(sq_distance_to_center * m_options.m_downsample_resulution_inv);
        hyper_grid_ptr->m_point_status.back().m_frame_id = current_frame_id;
        hyper_grid_ptr->m_point_status.back().set_normal(pt.normal_x, pt.normal_y, pt.normal_z);
        if (!point_status.at(i).m_with_valid_normal && normal_need_recompute)
        {
          hyper_grid_ptr->m_points_without_normal_offsets.push_back(
              hyper_grid_ptr->m_point_status.size() - 1);
        }

        iter->second->second.m_point_offsets.push_back(sz);
        hyper_grid_ptr->m_down_sample_voxels[voxel_key] = sz;
      }
      else
      {
        // perform downsample
        int offset = voxel_iter->second;

        if (hyper_grid_ptr->m_point_status.at(offset).m_deleted_mark
            // || hyper_grid_ptr->m_squared_distances_to_center.at(offset) > sq_distance_to_center
            // || sq_distance_to_center < m_options.m_downsample_half_sphere_sq_distance
            // && offset > hyper_grid_ptr->m_indexed_till
            || point_status.at(i).m_with_valid_normal  // always trust new normal
        )
        {
          Vec3f ori_normal = hyper_grid_ptr->m_point_status.at(offset).get_normal();
          Vec3f new_normal = point_status.at(i).get_normal();

          int ori_direction = get_direction_key_inner(ori_normal);
          int new_direction = get_direction_key_inner(new_normal);
          bool old_with_normal = hyper_grid_ptr->m_point_status.at(offset).m_with_valid_normal;
          bool new_with_normal = point_status.at(i).m_with_valid_normal;
          int target_direction = (voxel_key(0) + voxel_key(1) + voxel_key(2)) % 2;

          bool perform_replacement = false;

          if (!new_with_normal)
          {
            // do nothing
          }
          else if (new_with_normal && (!old_with_normal))
          {
            perform_replacement = true;
          }
          else if (target_direction != ori_direction && target_direction == new_direction)
          {
            perform_replacement = true;
          }
          // remove the deleted point or replace the point further to the center of the voxel

          // never replace a point
          if (perform_replacement)
          {
            hyper_grid_ptr->m_point_cloud->points.at(offset) = pt_xivt;
            hyper_grid_ptr->m_point_status.at(offset) = point_status.at(i);
            hyper_grid_ptr->m_point_status.at(offset).m_frame_id = current_frame_id;
            hyper_grid_ptr->m_point_status.at(offset).m_deleted_mark = false;
            hyper_grid_ptr->m_point_status.at(offset).m_squared_distances_to_center =
                XiVT_fake_float_01(sq_distance_to_center * m_options.m_downsample_resulution_inv);
            hyper_grid_ptr->m_modified_count++;
          }
        }
      }

      m_grids_lru_cache.splice(m_grids_lru_cache.begin(), m_grids_lru_cache, iter->second);
      m_nodes_map[node_key] = m_grids_lru_cache.begin();
    }
  }

  // exit(0);
  // });
}

void XiVT::perform_forget(int frame_threshold)
{
  // first, collect hyper grids that needs to perform forget
  GLOG(INFO) << xutil::strprintf("xivt pfg %d\n", frame_threshold);

  std::vector<XiVT_hyper_grid_context*> hyper_grids;

  for (auto p : m_hyper_grids)
  {
    if (p.second->m_point_status.size() > 0 &&
        p.second->m_point_status[0].m_frame_id < frame_threshold)
    {
      hyper_grids.push_back(p.second);
    }
  }

#pragma omp parallel for schedule(dynamic)
  for (int hyper_grid_idx = 0; hyper_grid_idx < hyper_grids.size(); hyper_grid_idx++)
  {
    XiVT_hyper_grid_context* hyper_grid_context = hyper_grids[hyper_grid_idx];

    // std::binary_search(p.second->m_frame_id.begin, p.second->m_frame_id.end, )

    int low = 0, high = hyper_grid_context->m_point_status.size(), middle;
    int position = -1;

    while (low < high)
    {
      middle = (low + high) / 2;
      if (frame_threshold == hyper_grid_context->m_point_status[middle].m_frame_id)
      {
        position = middle;
        break;
      }
      else if (frame_threshold < hyper_grid_context->m_point_status[middle].m_frame_id)
      {
        high = middle;
      }
      else if (frame_threshold > hyper_grid_context->m_point_status[middle].m_frame_id)
      {
        low = middle + 1;
      }
    }
    if (position == -1)
    {
      position = std::min(low, (int)(hyper_grid_context->m_point_status.size() - 1));
    }
    else
    {
      while (position < hyper_grid_context->m_point_status.size() - 1 &&
             hyper_grid_context->m_point_status[position].m_frame_id == frame_threshold)
      {
        position++;
      }
    }

    if (position == hyper_grid_context->m_point_status.size() - 1)
    {
      position = hyper_grid_context->m_point_status.size();
    }

    // GLOG(INFO) << xutil::strprintf("erase points before frame %d, count = %d in %d\n",
    // frame_threshold, position, hyper_grid_context->m_point_status.size());

    // hyper_grid_context->m_down_sample_voxels.erase(hyper_grid_context->m_down_sample_voxels.begin(),
    // hyper_grid_context->m_down_sample_voxels.begin() + position);

    robin_hood::unordered_flat_map<XiVT_key_type, int, hash_vec<3>> down_sample_voxels;

    for (int i = position; i < hyper_grid_context->m_point_cloud->points.size(); i++)
    {
      auto pt_eigen = ToEigen<float, t_dim>(hyper_grid_context->m_point_cloud->points[i]);
      XiVT_key_type voxel_key = position_to_voxel(pt_eigen);
      // int direction_key = 0;

      // if (i < position) {
      //     hyper_grid_context->m_down_sample_voxels.erase(voxel_key);
      // } else {
      //     hyper_grid_context->m_down_sample_voxels.at(voxel_key) -= position;
      // }
      down_sample_voxels[voxel_key] = i - position;
    }

    hyper_grid_context->m_down_sample_voxels.swap(down_sample_voxels);

    // hyper_grid_context->m_deleted_mark.erase(hyper_grid_context->m_deleted_mark.begin(),
    // hyper_grid_context->m_deleted_mark.begin() + position);
    // hyper_grid_context->m_frame_id.erase(hyper_grid_context->m_frame_id.begin(),
    // hyper_grid_context->m_frame_id.begin() + position);
    hyper_grid_context->m_point_cloud->points.erase(
        hyper_grid_context->m_point_cloud->points.begin(),
        hyper_grid_context->m_point_cloud->points.begin() + position);
    hyper_grid_context->m_point_cloud->points.shrink_to_fit();
    hyper_grid_context->m_point_status.erase(
        hyper_grid_context->m_point_status.begin(),
        hyper_grid_context->m_point_status.begin() + position);
    hyper_grid_context->m_point_status.shrink_to_fit();

    std::vector<int> new_points_without_normal;

    for (int i = 0; i < hyper_grid_context->m_points_without_normal_offsets.size(); i++)
    {
      if (hyper_grid_context->m_points_without_normal_offsets[i] >= position)
      {
        new_points_without_normal.push_back(
            hyper_grid_context->m_points_without_normal_offsets[i] - position);
      }
    }
    hyper_grid_context->m_points_without_normal_offsets.swap(new_points_without_normal);

    std::vector<XiVT_node*> empty_nodes, non_empty_nodes;

    // for each node, reduce its counter
    for (auto* node : hyper_grid_context->m_nodes)
    {
      std::vector<int> new_offsets;
      for (auto i : node->m_point_offsets)
      {
        if (i >= position)
        {
          new_offsets.push_back(i - position);
        }
      }
      node->m_point_offsets.swap(new_offsets);

      if (node->m_point_offsets.size() == 0)
      {
        empty_nodes.push_back(node);
      }
      else
      {
        non_empty_nodes.push_back(node);
      }
    }

    // hyper_grid_context->m_nodes = non_empty_nodes;

    // clear empty nodes
    // for (auto* node : empty_nodes) {
    //     auto item = m_nodes_map.find(node->m_node_key);
    //     m_grids_lru_cache.erase(item->second);
    //     m_nodes_map.erase(item);
    // }

    // clear index
    hyper_grid_context->m_indexed_till = -1;
    hyper_grid_context->m_kd_tree.reset(new pcl::search::KdTree<XiVT_point_type>);
    hyper_grid_context->m_modified_count = hyper_grid_context->m_point_cloud->points.size();
  }
}

void XiVT::recompute_normal(
    int cur_frame_id,
    int frame_stride,
    int min_interval,
    int max_attempt,
    int _min_num,
    int _max_num,
    double max_range)
{
  std::vector<XiVT_hyper_grid_context*> hyper_grids;

  int total_normal_remaining = 0;
  for (auto p : m_hyper_grids)
  {
    if (p.second->m_points_without_normal_offsets.size() > 0)
    {
      hyper_grids.push_back(p.second);
      total_normal_remaining += p.second->m_points_without_normal_offsets.size();
    }
  }
  GLOG(INFO) << "[XiVT::recompute_normal] cur_frame_id = " << cur_frame_id
             << ", recompute normal: " << total_normal_remaining;

  float tmp_rev_normal_threshold = m_options.m_rev_normal_threshold;
  m_options.m_rev_normal_threshold = 99999;

  // #pragma omp parallel for schedule(dynamic, 1)
  for (int hyper_grid_idx = 0; hyper_grid_idx < hyper_grids.size(); hyper_grid_idx++)
  {
    XiVT_hyper_grid_context* hyper_grid_context = hyper_grids[hyper_grid_idx];

    int count_remain = 0;
    std::vector<int> remain;
    for (auto i : hyper_grid_context->m_points_without_normal_offsets)
    {
      // get all ref
      // if (i >= hyper_grid_context->m_point_cloud->points.size()) {
      //     continue;
      // }
      XiVT_point_status& point_status = hyper_grid_context->m_point_status.at(i);

      if (point_status.m_with_valid_normal)
      {
        continue;
      }

      XiVT_point_type& point = hyper_grid_context->m_point_cloud->points.at(i);
      // if (hyper_grid_context->m_frame_id.at(i) >= m_viewpoints.size()) {
      //     continue;
      // }
      const Vec3f& viewpoint = m_viewpoints.at(hyper_grid_context->m_point_status.at(i).m_frame_id);

      if (cur_frame_id - point_status.m_last_attempt_frame < min_interval)
      {
        remain.push_back(i);
        continue;
      }
      XiVT_point_vector closest_pt;
#ifdef V1207
      Eigen::Matrix<float, 9, 1> plane_result;
#else
      Plane plane_result;
#endif
      // perform knn
      XiVT_knn_query_status query_status = get_closest_point(
          point,
          viewpoint,
          closest_pt,
          plane_result,
          _min_num * 1,
          _max_num * 1,
          max_range,
          true,
          1);

      point_status.m_last_attempt_frame = cur_frame_id;
      if (query_status.m_valid_plane)
      {
        point_status.m_with_valid_normal = true;

        // assign normal
        Vec3f pos_vec(point.x, point.y, point.z);
#ifdef V1207
        Vec3f norm_vec(plane_result(0), plane_result(1), plane_result(2));
#else
        Vec3f norm_vec(plane_result.normal(0), plane_result.normal(1), plane_result.normal(2));
#endif
        Vec3f view_vec = viewpoint - pos_vec;

        float dot = view_vec.transpose() * norm_vec;

        float coeff = dot > 0 ? 1 : -1;

        norm_vec *= coeff;

        // {
        //     // check point in front of plane
        //     Vec3f norm_center_vec(pca_result(4),
        //                             pca_result(5),
        //                             pca_result(6));
        //     Vec3f center_to_point_vec = pos_vec - norm_center_vec;
        //     float dot = center_to_point_vec.transpose() * norm_vec;
        //     if (dot < 0) {
        //         // behind the plane, bad
        //         continue;
        //     }
        // }
#ifdef V1207
        point.normal_x = norm_vec(0);
        point.normal_y = norm_vec(1);
        point.normal_z = norm_vec(2);
#endif
        point_status.set_normal(norm_vec);
      }
      else
      {
        point_status.m_total_attempt_count++;
        if (point_status.m_total_attempt_count >= max_attempt)
        {
          // point_status.m_with_valid_normal = false;
        }
        else
        {
          remain.push_back(i);
        }
      }
    }

    hyper_grid_context->m_points_without_normal_offsets.swap(remain);
    // only reserve those with
  }

  m_options.m_rev_normal_threshold = tmp_rev_normal_threshold;

  int remain_after_recompute = 0;
  for (int hyper_grid_idx = 0; hyper_grid_idx < hyper_grids.size(); hyper_grid_idx++)
  {
    XiVT_hyper_grid_context* hyper_grid_context = hyper_grids[hyper_grid_idx];
    remain_after_recompute += hyper_grid_context->m_points_without_normal_offsets.size();
  }

  // glog
  GLOG(INFO) << "[XiVT::recompute_normal] cur_frame_id = " << cur_frame_id
             << ", recompute normal: " << total_normal_remaining << " -> "
             << remain_after_recompute;
}

void XiVT::add_viewpoint(const Vec3f& viewpoint, int frame_id)
{
  assert(m_viewpoints.size() == frame_id);
  m_viewpoints.push_back(viewpoint);
}

void XiVT::build_index(bool force, int kdtree_threshold)
{
  // constexpr int kdtree_threshold = 500;

  std::vector<XiVT_hyper_grid_context*> hyper_grids;

  std::string force_str = force ? "forced" : "not forced";

  // #pragma omp master
  for (auto p : m_hyper_grids)
  {
    // if (p.second->m_point_cloud->points.size() > kdtree_threshold) {
    if (p.second->m_point_cloud->points.size() - p.second->m_indexed_till > kdtree_threshold ||
        force)
    {
      // printf("index hyper with coord %d %d %d, deleted mark size = %zu, pc size = %zu, index_till
      // = %d, %s\n",
      //         p.first[0], p.first[1], p.first[2], p.second->m_deleted_mark.size(),
      //         p.second->m_point_cloud->points.size(), p.second->m_indexed_till,
      //         force_str.c_str());
      // if (p.second->m_point_cloud->points.size() - p.second->m_indexed_till +
      // p.second->m_modified_count > kdtree_threshold) {
      hyper_grids.push_back(p.second);
    }
  }
  // int min_index_threshold = omp_get_max_threads() * 2;
  int min_index_threshold = 1;

  // do this only for last threads
  if (min_index_threshold <= hyper_grids.size())
  {
#pragma omp parallel for
    for (int hyper_grid_idx = 0; hyper_grid_idx < hyper_grids.size(); hyper_grid_idx++)
    {
      XiVT_hyper_grid_context* hyper_grid_context = hyper_grids[hyper_grid_idx];

      hyper_grid_context->m_kd_tree->setInputCloud(hyper_grid_context->m_point_cloud);

      hyper_grid_context->m_indexed_till = hyper_grid_context->m_point_cloud->points.size() - 1;
      hyper_grid_context->m_modified_count = 0;
    }
  }
}

void XiVT::garbage_collect()
{
  // erase points marked deleted

  // create a point vector with new capacity, and
}

std::string XiVT::statistics_string() const
{
  std::stringstream ss;

  int total_grids = m_nodes_map.size(), total_points = 0, total_points_vec = 0, erased_points = 0,
      total_indexed_points = 0, erased_grids = 0;

  ss << "[XiVT][STA] hyper grids = " << m_hyper_grids.size();

  for (auto p : m_hyper_grids)
  {
    XiVT_hyper_grid_context* hyper_grid = p.second;
    erased_grids += hyper_grid->m_erased_grid_count;
    erased_points += hyper_grid->m_erased_point_count;
    total_indexed_points += hyper_grid->m_indexed_till + 1;
    total_points += hyper_grid->m_down_sample_voxels.size();
    total_points_vec += hyper_grid->m_point_cloud->points.capacity();
  }

  ss << ", total grids = " << total_grids;
  ss << ", total points = " << total_points;
  ss << ", total points vec = " << total_points_vec << ", total size = "
     << total_points_vec * (sizeof(PointT) + sizeof(XiVT_point_status)) / 1024.0 / 1024.0 << " MB";
  ss << ", total indexed points = " << total_indexed_points;
  ss << ", avg point per grid = " << total_points * 1.0 / total_grids;
  ss << ", avg point per hyper grid = " << total_points * 1.0 / m_hyper_grids.size();
  ss << ", erased grids = " << erased_grids;
  ss << ", erased points = " << erased_points;

  return ss.str();
}

void XiVT::test_hyper_grids_consistency() const
{
  return;

  std::map<int, int> mmp;
  std::vector<int> v;
  for (int i = 0; i < 10; i++)
  {
    mmp[i] = i;
    v.push_back(i);
  }

  if (&mmp.at(9) == nullptr)
  {
    printf("holy dear! %d, %d\n", mmp.at(8), v.at(11));
    ((int*)(nullptr))[0] = 0;
    std::map<int, int> a;
    printf("%d\n", a.at(0));
  }
  // return;

  // for (auto p : m_hyper_grids) {
  //         if (p.second->m_point_cloud->points.size() != p.second->m_deleted_mark.size()) {
  //             printf("invalid hyper with coord %d %d %d, deleted mark size = %zu, pc size = %zu,
  //             index_till = %d\n",
  //                     p.first[0], p.first[1], p.first[2], p.second->m_deleted_mark.size(),
  //                     p.second->m_point_cloud->points.size(), p.second->m_indexed_till);
  //             fflush(stdout);
  //             assert(false);
  //             ((int*)(nullptr))[0] = 0;
  //             std::map<int, int> a;
  //             printf("%d\n", a.at(0));
  //         }
  // }
}

pcl::PointCloud<XiVT_point_type>::Ptr XiVT::get_all_points(bool last_call)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  pcl::PointCloud<XiVT_point_type>::Ptr cloud(new pcl::PointCloud<XiVT_point_type>());
  size_t total_size = 0;
  for (auto p : m_hyper_grids)
  {
    XiVT_hyper_grid_context* hyper_grid = p.second;
    if (last_call)
    {
      std::vector<XiVT_point_status> point_status;
      total_size += hyper_grid->m_point_cloud->points.size();
      hyper_grid->m_point_status.swap(point_status);
    }
  }

  cloud->points.reserve(total_size);

  for (auto p : m_hyper_grids)
  {
    XiVT_hyper_grid_context* hyper_grid = p.second;

    cloud->points.insert(
        cloud->points.end(),
        hyper_grid->m_point_cloud->points.begin(),
        hyper_grid->m_point_cloud->points.end());
  }
  cloud->height = 1;
  cloud->width = cloud->points.size();
  return cloud;
}

pcl::PointCloud<XiVT_point_type>::Ptr XiVT::get_downsampled_points(int count, bool last_call)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  pcl::PointCloud<XiVT_point_type>::Ptr cloud(new pcl::PointCloud<XiVT_point_type>());
  size_t total_size = 0;
  for (auto p : m_hyper_grids)
  {
    XiVT_hyper_grid_context* hyper_grid = p.second;
    total_size += hyper_grid->m_point_cloud->points.size();
    if (last_call)
    {
      std::vector<XiVT_point_status> point_status;
      hyper_grid->m_point_status.swap(point_status);
    }
  }

  GLOG(INFO) << "[XiVT::get_downsampled_points] reserving " << count;

  cloud->points.reserve(count);
  double sample_ratio = count * 1.0 / total_size;
  if (sample_ratio > 0.999)
  {
    sample_ratio = 0.999;
  }

  // sample count / total_size points
  for (auto p : m_hyper_grids)
  {
    XiVT_hyper_grid_context* hyper_grid = p.second;
    int sample_count = hyper_grid->m_point_cloud->points.size() * sample_ratio;
    // GLOG(INFO) << "[XiVT::get_downsampled_points] sampling " << sample_count << " points from
    // hyper grid with points count = " << hyper_grid->m_point_cloud->points.size();

    if (sample_count > 0)
    {
      // sample with stride
      int stride = hyper_grid->m_point_cloud->points.size() / sample_count;
      for (int i = 0; i < hyper_grid->m_point_cloud->points.size(); i += stride)
      {
        cloud->points.push_back(hyper_grid->m_point_cloud->points[i]);
      }
    }
  }
  return cloud;
}

std::vector<float> XiVT::StatGridPoints() const
{
  // int num = m_grids_lru_cache.size(), valid_num = 0, max = 0, min = 100000000;
  // int sum = 0, sum_square = 0;
  // for (auto& it : m_grids_lru_cache) {
  //     int s = it.second.Size();
  //     valid_num += s > 0;
  //     max = s > max ? s : max;
  //     min = s < min ? s : min;
  //     sum += s;
  //     sum_square += s * s;
  // }
  // float ave = float(sum) / num;
  // float stddev = num > 1 ? sqrt((float(sum_square) - num * ave * ave) / (num - 1)) : 0;
  // return std::vector<float>{valid_num, ave, max, min, stddev};
  return std::vector<float>();
}

}  // namespace xivt