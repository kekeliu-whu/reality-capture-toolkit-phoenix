#pragma once

#include <colmap/geometry/rigid3.h>
#include <colmap/scene/camera.h>

#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace xcolor {

constexpr int kMinCandidateNum = 3;
constexpr int kColorInlierMaxNum = 10;
constexpr int kColorInlierThreshold = 60;
constexpr int kRangeInlier = 40;
constexpr double kRayCasterVoxelResolution = 0.08;
constexpr double kDepthVisibilityTolerance = 0.15;
constexpr int kDepthNeighborRadius = 1;

struct DepthIntrinsics {
  int width = 0;
  int height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

struct Image {
  std::string filename;
  std::string depth_filename;
  std::string mask_filename;
  colmap::Rigid3d pose;
  colmap::Camera camera;
  DepthIntrinsics depth_intrinsics;
  bool has_depth = false;
  bool has_mask = false;
};

// 存储每个点的颜色候选值
struct PointColorCandidates {
  std::vector<std::pair<double, cv::Vec3b>> candidates;  // (distance, color)
};

bool IsMaskPixelAllowed(const cv::Mat& mask, const Eigen::Vector2d& pixel);
bool IsNormalFacingCamera(const Eigen::Vector3d& point_in_camera,
                          const Eigen::Vector3d& normal_in_camera);
DepthIntrinsics DepthIntrinsicsFromCamera(const colmap::Camera& camera);

void PerformXColor(const std::vector<Image>& images,
                   pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud_rgb,
                   std::string output_path,
                   int color_candidate_limit = kColorInlierMaxNum);

}  // namespace xcolor
