//
// Created by youyuan on 23-12-5.
//

#include "xmap.h"
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace xmap {
namespace {

template <typename T>
void readOptional(const YAML::Node& config, const char* key, T& value) {
  const YAML::Node node = config[key];
  if (node && !node.IsNull()) value = node.as<T>();
}

}  // namespace

Xmap::Xmap(const Configs& configs) : configs_(configs) {
  finalizeConfigs();
  LOG(INFO) << "Init Xmap Success!";
}

Xmap::Xmap(const std::string& path) {
  loadConfigsFromYaml(path);
  LOG(INFO) << "Init Xmap Success!";
}

Xmap::~Xmap() { std::unordered_map<VoxelLoc, SmallVoxelValue>().swap(small_voxel_map_); }

void Xmap::loadConfigsFromYaml(const std::string& path) {
  try {
    const YAML::Node root = YAML::LoadFile(path);
    const YAML::Node config = root["xmap"] ? root["xmap"] : root;

    readOptional(config, "resolution", configs_.resolution);
    readOptional(config, "small_scale", configs_.small_scale);
    readOptional(config, "knn_min_points", configs_.knn_min_points);
    readOptional(config, "knn_max_points", configs_.knn_max_points);
    readOptional(config, "knn_distance_limit", configs_.knn_distance_limit);
    readOptional(config, "forget_ts", configs_.forget_ts);
    readOptional(config, "forget_range", configs_.forget_range);

    readOptional(config, "convergence_num", configs_.convergence_num);
    readOptional(config, "enable_normal_filter", configs_.enable_normal_filter);
    readOptional(config, "enable_forget", configs_.enable_forget);
    readOptional(config, "test_data_path", configs_.test_data_path);
    int replace_points_flag = static_cast<int>(configs_.replace_points_flag);
    readOptional(config, "replace_points_flag", replace_points_flag);
    configs_.replace_points_flag = static_cast<PointUpdateStrategy>(replace_points_flag);

    finalizeConfigs();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to load YAML file: " << e.what();
    throw;
  }
}

void Xmap::finalizeConfigs() {
  configs_.knn_distance_limit_squared =
      configs_.knn_distance_limit * configs_.knn_distance_limit;
  configs_.small_voxel_size = configs_.resolution * configs_.small_scale;

  if (configs_.resolution <= 0 || configs_.small_scale <= 0) {
    throw std::invalid_argument("XMap voxel scale must be positive");
  }
  if (configs_.knn_min_points < 3 || configs_.knn_max_points < configs_.knn_min_points ||
      configs_.knn_distance_limit <= 0) {
    throw std::invalid_argument("XMap KNN configuration is invalid");
  }
  if (configs_.convergence_num < configs_.knn_min_points) {
    throw std::invalid_argument("XMap KD-Tree convergence threshold is invalid");
  }
  if (configs_.replace_points_flag < OLD_REMAIN || configs_.replace_points_flag > FUSION) {
    throw std::invalid_argument("XMap replace_points_flag is invalid");
  }
  // clang-format off
  LOG(INFO) << "resolution:" << configs_.resolution;
  LOG(INFO) << "small_scale:" << configs_.small_scale << ", small_size:" << configs_.small_voxel_size;
  LOG(INFO) << "forget_ts:" << configs_.forget_ts << ", forget_range:" << configs_.forget_range;
  LOG(INFO) << "knn_min_points:" << configs_.knn_min_points << ", knn_max_points:" << configs_.knn_max_points;
  LOG(INFO) << "knn_distance_limit:" << configs_.knn_distance_limit;
  LOG(INFO) << "convergence_num:" << configs_.convergence_num;
  LOG(INFO) << "replace_points_flag:" << configs_.replace_points_flag;
  LOG(INFO) << "enable_normal_filter:" << configs_.enable_normal_filter;
  LOG(INFO) << "enable_forget:" << configs_.enable_forget;
  // clang-format on
}

const Configs& Xmap::getConfigs() const { return configs_; }

int Xmap::pointSize() const {
  int total_pointcloud_size = 0;
  for (const auto& entry : small_voxel_map_) {
    total_pointcloud_size += entry.second.cloud_->size();
  }
  return total_pointcloud_size;
}

int Xmap::capacity() const {
  int total_pointcloud_size = 0;
  for (const auto& entry : small_voxel_map_) {
    total_pointcloud_size += entry.second.cloud_->points.capacity();
  }
  return total_pointcloud_size;
}

void Xmap::reBuildKDTree() {
  for (const auto& entry : small_voxel_map_) {
    entry.second.kd_tree_->setInputCloud(entry.second.cloud_);
  }
}

PointCloudConstPtr Xmap::getMapPointCloud() const {
  PointCloudPtr cloud_ptr(new xmap::PointCloud);
  for (const auto& entry : small_voxel_map_) {
    for (const auto& point : *entry.second.cloud_) {
      cloud_ptr->points.push_back(point);
    }
  }
  cloud_ptr->points.resize(pointSize());
  cloud_ptr->points.shrink_to_fit();
  cloud_ptr->height = 1;
  cloud_ptr->width = cloud_ptr->points.size();
  return cloud_ptr;
}

}  // namespace xmap
