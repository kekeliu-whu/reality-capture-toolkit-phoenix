//
// Created by youyuan on 23-12-5.
//

#include "xmap.h"
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include "xmap_util.h"

namespace xmap {

Xmap::Xmap(const std::string& path) {
  loadConfigsFromYaml(path);
  LOG(INFO) << "Init Xmap Success!";
}

Xmap::~Xmap() { std::unordered_map<VoxelLoc, SmallVoxelValue>().swap(small_voxel_map_); }

void Xmap::loadConfigsFromYaml(const std::string& path) {
  try {
    YAML::Node config = YAML::LoadFile(path);

    configs_.resolution = config["resolution"].as<float>();
    configs_.small_scale = config["small_scale"].as<int>();
    configs_.large_scale = config["large_scale"].as<int>();
    configs_.map_scale = config["map_scale"].as<int>();

    configs_.knn_min_points = config["knn_min_points"].as<int>();
    configs_.knn_max_points = config["knn_max_points"].as<int>();
    configs_.knn_distance_limit = config["knn_distance_limit"].as<float>();

    configs_.forget_ts = config["forget_ts"].as<float>();
    configs_.forget_range = config["forget_range"].as<float>();

    configs_.enable_normal_filter = config["enable_normal_filter"].as<bool>();
    configs_.enable_forget_point = config["enable_forget_point"].as<bool>();
    configs_.enable_forget = config["enable_forget"].as<bool>();
    configs_.enable_dynamic = config["enable_dynamic"].as<bool>();
    configs_.enable_dynamic_backend = config["enable_dynamic_backend"].as<bool>();
#ifdef __x86_64__
    configs_.pcd_path = config["pcd_path"].as<std::string>();
#else
    configs_.pcd_path = config["pcd_path_onboard"].as<std::string>();
#endif
    configs_.test_data_path = config["test_data_path"].as<std::string>();
    configs_.replace_points_flag =
        static_cast<PointUpdateStrategy>(config["replace_points_flag"].as<int>());

    configs_.voxel_diff = (configs_.map_scale - 1) / 2;
    configs_.small_voxel_size = configs_.resolution * configs_.small_scale;
    configs_.large_voxel_size = configs_.small_voxel_size * configs_.large_scale;
    configs_.map_size = configs_.large_voxel_size * configs_.map_scale;

    // clang-format off
    LOG(INFO) << "resolution:" << configs_.resolution;
    LOG(INFO) << "small_scale:" << configs_.small_scale << ", small_size:" << configs_.small_voxel_size;
    LOG(INFO) << "large_scale:" << configs_.large_scale << ", large_size:" << configs_.large_voxel_size;
    LOG(INFO) << "map_scale  :" << configs_.map_scale <<   ", map_size  :" << configs_.map_size;
    LOG(INFO) << "map_size:" << configs_.map_size;
    LOG(INFO) << "forget_ts:" << configs_.forget_ts;
    LOG(INFO) << "forget_range:" << configs_.forget_range;
    LOG(INFO) << "knn_min_points:" << configs_.knn_min_points;
    LOG(INFO) << "knn_max_points:" << configs_.knn_max_points;
    LOG(INFO) << "replace_points_flag:" << configs_.replace_points_flag;
    LOG(INFO) << "pcd_path:" << configs_.pcd_path;

    LOG(INFO) << "enable_normal_filter:" << configs_.enable_normal_filter;
    LOG(INFO) << "enable_forget_point:" << configs_.enable_forget_point;
    LOG(INFO) << "enable_dynamic:" << configs_.enable_dynamic;
    LOG(INFO) << "enable_dynamic_backend:" << configs_.enable_dynamic_backend;

    // clang-format on
    if (configs_.enable_dynamic) {
      createFolders(configs_.pcd_path);
      deleteFolderContents(configs_.pcd_path);
    }
    // dynamicIO_thread_->Start();

    if (configs_.map_scale % 2 != 1) {
      LOG(ERROR) << "map_scale not satisfy odd!";
      // 一级告警：错误码 101 配置内容有误
      exit(101);
    }
    if (configs_.enable_dynamic == configs_.enable_forget && configs_.enable_dynamic) {
      LOG(ERROR) << "enable_dynamic == enable_forget == true, params can not be true sync!";
      // 一级告警：错误码 101 配置内容有误
      exit(101);
    }

  } catch (const YAML::Exception& e) {
    // 一级告警：处理加载文件失败的情况
    LOG(ERROR) << "Failed to load YAML file: " << e.what();
    // 错误码 102 配置格式有误
    exit(102);
  }
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

bool Xmap::checkGridDataSync() {
  for (const auto& it : large_voxel_map_) {
    for (const auto& small_voxel : it.second) {
      auto it_small = small_voxel_map_.find(small_voxel);
      if (it_small == small_voxel_map_.end()) {
        LOG(ERROR) << "small_voxel:" << small_voxel;
        LOG(ERROR) << "small_voxel_map_ not sync with large_voxel_map_!";
      }
    }
  }
  return true;
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