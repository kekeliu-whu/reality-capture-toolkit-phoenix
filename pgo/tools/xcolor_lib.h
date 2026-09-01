#pragma once

#include <colmap/geometry/rigid3.h>
#include <colmap/scene/camera.h>

#include <string>
#include <unordered_map>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

namespace xcolor {

constexpr int kMinCandidateNum             = 3;
constexpr int kColorInlierMaxNum           = 10;
constexpr int kColorInlierThreshold        = 60;
constexpr int kRangeInlier                 = 40;
constexpr double kRayCasterVoxelResolution = 0.08;
constexpr double kDepthVisibilityTolerance = 0.15;
constexpr int kDepthNeighborRadius         = 1;

struct FisheyeDepthOptions {
  bool generate         = false;
  bool gpu_visibility   = true;
  bool gpu_color_fusion = true;
  bool smooth_fusion    = true;
  double scale          = 0.25;
  float voxel_size      = 0.03f;
  float max_distance    = 30.0f;
  float visibility_tolerance = static_cast<float>(kDepthVisibilityTolerance);
  int gpu_chunk_points  = 3000000;
  std::string output_path;
};

struct Image {
  std::string name;
  std::string filename;
  std::string depth_filename;
  std::string mask_filename;
  colmap::Rigid3d pose;
  colmap::Camera camera;
  colmap::Camera depth_camera;
  bool has_depth = false;
  bool has_mask  = false;
};

// 存储每个点的颜色候选值
struct PointColorCandidates {
  std::vector<std::pair<double, cv::Vec3b>> candidates;  // (distance, color)
};

bool IsMaskPixelAllowed(const cv::Mat& mask, const Eigen::Vector2d& pixel);
bool IsNormalFacingCamera(const Eigen::Vector3d& point_in_camera,
                          const Eigen::Vector3d& normal_in_camera);
double ComputeCandidateDistance(const Eigen::Vector3d& point_in_camera);

void PerformXColor(const std::vector<Image>& images,
                   pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud_rgb,
                   std::string output_path,
                   int color_candidate_limit                        = kColorInlierMaxNum,
                   const FisheyeDepthOptions& fisheye_depth_options = {});

}  // namespace xcolor
