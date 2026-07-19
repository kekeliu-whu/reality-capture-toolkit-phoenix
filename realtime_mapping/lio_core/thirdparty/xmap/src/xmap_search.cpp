//
// Created by youyuan on 24-1-17.
//
#include <glog/logging.h>
#include "xmap.h"
#include "xmap_util.h"

namespace xmap {

bool Xmap::knnSearch(
    const V3F& point_search,
    const V3F& view_point,
    std::vector<V3F>& nearest_points,
    double ts_absolute) {
  /** 1. 查找所有可能邻居 **/
  nearest_points.clear();
  std::vector<VoxelLoc> searchVoxel = generateSearchVoxelKey(point_search);
  std::vector<V3F> nearest_point_temp;
  std::vector<float> nearest_dist_temp;
  float ts_relative = static_cast<float>(ts_absolute - start_mapping_ts_);
  PointType pointType = V3F2PointType(point_search);
  /** 2. 在所有可能的邻居中进行KNN搜索 **/
  for (int i = 0; i < searchVoxel.size(); ++i) {
    std::vector<int> pointIdxSearch;          // 存储k近邻搜索点的索引结果
    std::vector<float> pointSquaredDistance;  // 存储k近邻搜索的平方距离
    std::vector<PointType, Eigen::aligned_allocator<PointType>> pointsVector;

    auto it = small_voxel_map_.find(searchVoxel[i]);
    if (it == small_voxel_map_.end() || it->second.cloud_->empty() || it->second.is_dynamic_IO)
      continue;

    // 执行本格子的KNN搜索
    int search_max = configs_.knn_max_points;
    if (configs_.enable_normal_filter) search_max *= 2;
    it->second.kd_tree_->nearestKSearch(
        pointType, search_max, pointIdxSearch, pointSquaredDistance);
    for (int j = 0; j < pointIdxSearch.size(); ++j) {
      PointType nearest_j_point = it->second.cloud_->points[pointIdxSearch[j]];
      // 法向过滤
      if (configs_.enable_normal_filter) {
        V3F point_normal(
            nearest_j_point.normal_x, nearest_j_point.normal_y, nearest_j_point.normal_z);
        V3F view_vec = view_point - point_search;
        view_vec /= view_vec.norm();
        double cos_theta = point_normal.dot(view_vec);
        // cos_theta < 0排除了钝角的case，留下了锐角和normal = (0,0,0) 的 case
        if (cos_theta < 0) continue;
      }

      V3F searched_point = pointType2V3F(nearest_j_point);
      if (pointSquaredDistance[j] < configs_.knn_distance_limit * configs_.knn_distance_limit) {
        nearest_point_temp.emplace_back(pointType2V3F(nearest_j_point));
        nearest_dist_temp.emplace_back(pointSquaredDistance[j]);
      }
    }
    // 更新本格子的active时间
    if (ts_relative > it->second.time_mark_.end_ts) it->second.time_mark_.end_ts = ts_relative;
    // 近似搜索，降低了约3%的召回率
    if (nearest_point_temp.size() >= configs_.knn_min_points) break;
  }

  /** 3. 得到真正的k近邻 **/
  // 排序，创建索引数组
  std::vector<int> indices(nearest_dist_temp.size());
  for (int i = 0; i < indices.size(); ++i) indices[i] = i;
  // 使用自定义比较函数对索引数组进行排序
  std::sort(indices.begin(), indices.end(), [&nearest_dist_temp](int a, int b) {
    return nearest_dist_temp[a] < nearest_dist_temp[b];
  });

  // 将距离最小的k个点返回
  for (int i = 0; i < std::min(configs_.knn_max_points, (int)nearest_point_temp.size()); ++i) {
    nearest_points.emplace_back(nearest_point_temp[indices[i]]);
  }
  return nearest_dist_temp.size() >= configs_.knn_min_points;
}

bool Xmap::radiusSearch(
    const V3F& point_search,
    const V3F& view_point,
    std::vector<V3F>& nearest_points,
    double ts_absolute) {
  /** 1. 查找所有可能邻居 **/
  float ts_relative = static_cast<float>(ts_absolute - start_mapping_ts_);
  nearest_points.clear();
  std::vector<VoxelLoc> searchVoxel = generateSearchVoxelKey(point_search);
  std::vector<float> nearest_dist_temp;

  /** 2. 在所有可能的邻居中进行radius搜索 **/
  for (int i = 0; i < searchVoxel.size(); ++i) {
    std::vector<int> pointIdxSearch;          // 存储k近邻搜索点的索引结果
    std::vector<float> pointSquaredDistance;  // 存储k近邻搜索的平方距离

    auto it = small_voxel_map_.find(searchVoxel[i]);
    if (it == small_voxel_map_.end()) continue;

    PointType pointType = V3F2PointType(point_search);
    it->second.kd_tree_->radiusSearch(
        pointType,
        configs_.knn_distance_limit,
        pointIdxSearch,
        pointSquaredDistance,
        configs_.knn_max_points);

    for (int j = 0; j < pointIdxSearch.size(); ++j) {
      PointType nearest_j_point = it->second.cloud_->points[pointIdxSearch[j]];
      // 法向过滤
      if (configs_.enable_normal_filter) {
        V3F point_normal(
            nearest_j_point.normal_x, nearest_j_point.normal_y, nearest_j_point.normal_z);
        V3F view_vec = view_point - point_search;
        view_vec /= view_vec.norm();
        double cos_theta = point_normal.dot(view_vec);
        // cos_theta < 0包含了锐角点和normal=(0,0,0)的无法向点
        if (cos_theta < 0) continue;
      }
      nearest_points.emplace_back(pointType2V3F(nearest_j_point));

      // 近似搜索
      // if (nearest_points.size() >= configs_.knn_min_points) break;
    }

    // 更新本格子的active时间
    if (ts_relative > it->second.time_mark_.end_ts) it->second.time_mark_.end_ts = ts_relative;
  }

  return nearest_dist_temp.size() >= configs_.knn_min_points;
}

bool Xmap::planeSearch(const V3F& point_search, Plane& plane) { return false; }

std::vector<VoxelLoc> Xmap::generateSearchVoxelKey(const V3F& point) {
  /** 1. 根据点的位置计算体素坐标 **/
  VoxelLoc voxelLoc = pos2VoxelLoc(point, configs_.small_voxel_size);
  V3F voxel_center(
      voxelLoc.x_ * configs_.small_voxel_size,
      voxelLoc.y_ * configs_.small_voxel_size,
      voxelLoc.z_ * configs_.small_voxel_size);

  /** 2. 跨格判断 **/
  V3F diff = point - voxel_center;
  std::vector<int> index[3] = {{0}, {0}, {0}};
  // 计算每个维度上的索引变化
  // 只要在边界的configs_.knn_distance_limit内，就认为需要跨格搜索
  for (int i = 0; i < 3; ++i) {
    if (diff[i] > configs_.small_voxel_size / 2 - configs_.knn_distance_limit) {
      index[i].push_back(1);
    } else if (diff[i] < -configs_.small_voxel_size / 2 + configs_.knn_distance_limit) {
      index[i].push_back(-1);
    }
  }

  /** 3. 生成邻居体素坐标 **/
  std::vector<VoxelLoc> neighborVoxelLoc;
  for (int i : index[0])
    for (int j : index[1])
      for (int k : index[2])
        neighborVoxelLoc.emplace_back(voxelLoc.x_ + i, voxelLoc.y_ + j, voxelLoc.z_ + k);
  return neighborVoxelLoc;
}

}  // namespace xmap