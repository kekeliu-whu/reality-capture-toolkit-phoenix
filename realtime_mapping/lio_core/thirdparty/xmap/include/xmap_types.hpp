// 用于管理和维护Xmap的基本成员类型，不涉及Xmap的算法实现
// 涉及Xmap中常量的定义，类型的定义与初始化，空间哈希的计算方式
// 设计文档：https://pecivkvtit.feishu.cn/docx/Vq1OdiUmjoq3D5xa9IbczkMhnsb

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
namespace xmap {

using FloatDataType = float;
using IntDataType = int32_t;

using V3F = Eigen::Matrix<FloatDataType, 3, 1>;
using M3F = Eigen::Matrix<FloatDataType, 3, 3>;

// PointType中data[3]保存协方差, curvature保存相对start_time的时间
using PointType = pcl::PointXYZINormal;
using PointPtr = std::shared_ptr<PointType>;
using PointCloud = pcl::PointCloud<PointType>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

constexpr int64_t HASH_PRIME = 116101;
constexpr int64_t MAX_HASH_NUM = (int64_t(1) << 32);
constexpr int64_t PRIMES[] = {73856093, 19349663, 83492791};
constexpr double EPSILON = 1e-6;
constexpr double INIT_STARTING_MAPPING_TS = -1;
constexpr int MAX_ATTEMP = 10;
constexpr double PLANARITY_THRESHOLD = 0.3;

enum PointUpdateStrategy {
  OLD_REMAIN = 0,   // 0
  NEW_REPLACE = 1,  // 1
  FUSION = 2        // 2
};

struct Plane {
  V3F center = V3F::Zero();
  V3F normal = V3F::Zero();
  FloatDataType d = 0.0;
  FloatDataType planarity = 0.0;

  int points_size = 0;
  bool is_plane = false;
  bool operator==(const Plane& p) const {
    double error = 0.0;
    error += (center - p.center).norm();
    error += (normal - p.normal).norm();
    error += is_plane - p.is_plane;
    return error < 1e-12;
  }
  bool operator!=(const Plane& p) const { return !(*this == p); }
};
using PlanePtr = std::shared_ptr<Plane>;
using PlaneConstPtr = std::shared_ptr<const Plane>;

struct Configs {
  int knn_min_points = 5;                                // 配置输入
  int knn_max_points = 5;                                // 配置输入
  float knn_distance_limit = 1;                          // 配置输入

  float resolution = 0.1;                                // 配置输入
  int small_scale = 30;                                  // 配置输入
  int large_scale = 20;                                  // 配置输入
  int map_scale = 3;                                     // 配置输入
  float forget_ts = 10;                                  // 配置输入
  float forget_range = 120;                              // 配置输入

  int voxel_diff = 1;                                    // 计算获得
  float map_size = 180.0;                                // 计算获得
  float small_voxel_size = 3.0;                          // 计算获得
  float large_voxel_size = 60.0;                         // 计算获得

  bool enable_normal_filter = false;                     // 配置输入
  bool enable_dynamic = false;                           // 配置输入
  bool enable_dynamic_backend = false;                   // 配置输入
  bool enable_forget = false;                            // 配置输入
  bool enable_forget_point = false;                      // 配置输入
  std::string pcd_path = "null";                         // 配置输入
  std::string test_data_path = "null";                   // 配置输入
  PointUpdateStrategy replace_points_flag = OLD_REMAIN;  // 配置输入
};

struct TimeMark {
  float start_ts = -1;
  float end_ts = -1;
};
}  // namespace xmap