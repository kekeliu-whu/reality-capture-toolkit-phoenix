#include "point_cloud_utils.h"

#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>
#include <spdlog/spdlog.h>

#include <vector>

// Hash value
namespace std {
template <>
struct hash<Eigen::Vector3i> {
  int64_t operator()(const Eigen::Vector3i &s) const {
    using std::hash;
    using std::size_t;
    constexpr int64_t kHashP = 116101;
    constexpr int64_t kMaxN  = 10000000000;
    return ((((s.z()) * kHashP) % kMaxN + (s.y())) * kHashP) % kMaxN + (s.x());
  }
};
}  // namespace std

int GetVoxelCount(const PointCloudXYZI &points, double voxel_size) {
  if (voxel_size <= 0.0) {
    return static_cast<double>(points.size());
  }

  absl::flat_hash_map<Eigen::Vector3i, int> voxel_map;
  for (const auto &point : points) {
    Eigen::Vector3d point_pos(point.x, point.y, point.z);
    Eigen::Vector3i voxel_loc = ((point_pos / voxel_size).array()).round().cast<int>();
    voxel_map[voxel_loc]++;
  }

  return static_cast<double>(voxel_map.size());
}

double GetVoxelSizeForTargetSize(const PointCloudXYZI &points, int target_size) {
  if (points.size() <= static_cast<size_t>(target_size)) {
    return 0.0;
  }

  double voxel_size_lower = 0.03;
  double voxel_size_upper = 0.4;

  int upper_count = GetVoxelCount(points, voxel_size_lower);
  if (upper_count <= target_size) {
    return voxel_size_lower;
  }

  int lower_count = GetVoxelCount(points, voxel_size_upper);
  if (lower_count >= target_size) {
    return voxel_size_upper;
  }

  int iteration = 0;

  while (voxel_size_upper - voxel_size_lower > 0.01) {
    double voxel_size_mid = 0.5 * (voxel_size_lower + voxel_size_upper);
    int    voxel_count    = GetVoxelCount(points, voxel_size_mid);

    if (voxel_count > target_size) {
      voxel_size_lower = voxel_size_mid;
    } else {
      voxel_size_upper = voxel_size_mid;
    }

    iteration++;
  }

  LOG(INFO) << "Voxel size for target size " << target_size << " found after " << iteration
            << " iterations: " << 0.5 * (voxel_size_lower + voxel_size_upper);

  return 0.5 * (voxel_size_lower + voxel_size_upper);
}

void DownsamplePoints(const PointCloudXYZI &sweep, PointCloudXYZI &sweep_downsampled, int target_size) {
  sweep_downsampled.clear();
  if (sweep.size() <= static_cast<size_t>(target_size)) {
    sweep_downsampled = sweep;
    return;
  }

  double voxel_size = GetVoxelSizeForTargetSize(sweep, target_size);

  absl::flat_hash_map<Eigen::Vector3i, std::vector<size_t>> grid_map;

  Eigen::Vector3d random_offset = Eigen::Vector3d::Random() * voxel_size;
  for (size_t i = 0; i < sweep.size(); ++i) {
    const auto     &point     = sweep[i];
    Eigen::Vector3d point_pos = Eigen::Vector3d(point.x, point.y, point.z) + random_offset;
    Eigen::Vector3i voxel_loc = ((point_pos / voxel_size).array()).round().cast<int>();
    grid_map[voxel_loc].push_back(i);
  }

  for (auto &grid_entry : grid_map) {
    auto &indices = grid_entry.second;

    PointType center_point;
    double    sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

    for (size_t idx : indices) {
      const auto &p = sweep[idx];
      sum_x += p.x;
      sum_y += p.y;
      sum_z += p.z;
    }

    size_t cluster_size    = indices.size();
    center_point.x         = static_cast<float>(sum_x / cluster_size);
    center_point.y         = static_cast<float>(sum_y / cluster_size);
    center_point.z         = static_cast<float>(sum_z / cluster_size);
    center_point.curvature = 0;
    center_point.intensity = 0;

    sweep_downsampled.push_back(center_point);
  }

  spdlog::info("Downsampled by voxel from {} to {} points with voxel size {}", sweep.size(),
               sweep_downsampled.size(), voxel_size);
}
