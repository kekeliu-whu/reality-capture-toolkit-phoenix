#include <cmath>
#include <iostream>

#include "xmap.h"

namespace {

xmap::Configs makeConfigs() {
  xmap::Configs configs;
  configs.resolution = 0.1f;
  configs.small_scale = 20;
  configs.knn_min_points = 5;
  configs.knn_max_points = 8;
  configs.knn_distance_limit = 0.35f;
  configs.convergence_num = 100;
  configs.enable_normal_filter = true;
  configs.enable_forget = true;
  configs.forget_ts = 200.0f;
  configs.forget_range = 200.0f;
  return configs;
}

xmap::PointCloudPtr makePlaneCloud() {
  xmap::PointCloudPtr cloud(new xmap::PointCloud);
  cloud->header.stamp = 1000000;
  for (int x = 0; x < 15; ++x) {
    for (int y = 0; y < 15; ++y) {
      xmap::PointType point;
      point.x = 0.05f + 0.1f * static_cast<float>(x);
      point.y = 0.05f + 0.1f * static_cast<float>(y);
      point.z = 0.05f;
      cloud->push_back(point);
    }
  }
  return cloud;
}

}  // namespace

int main() {
  xmap::Xmap map(makeConfigs());
  const xmap::V3F view_point(0.75f, 0.75f, 1.5f);
  const xmap::V3F query(0.75f, 0.75f, 0.05f);
  map.mapIncremental(makePlaneCloud(), view_point);

  if (map.pointSize() != 225) {
    std::cerr << "dense cell index did not retain exactly one point per cell\n";
    return 1;
  }

  xmap::PlaneConstPtr plane;
  if (!map.knnSearch(query, view_point, 1.1, plane) || !plane || !plane->is_plane ||
      std::abs(plane->normal.z()) < 0.99f) {
    std::cerr << "active KNN path did not fit a valid horizontal plane\n";
    return 2;
  }

  std::cout << "XMap dense indexing and active KNN plane fitting passed\n";
  return 0;
}
