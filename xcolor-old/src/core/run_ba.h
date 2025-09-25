#pragma once

#include <ceres/ceres.h>
#include <colmap/estimators/cost_functions.h>
#include <colmap/scene/database.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

#include "core/xsfm_lib.h"

namespace xcolor {

void RunMultipleViewBA(const xcolor::SfmConfig &config, const std::string &output_path, const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree,
                       const pcl::PointCloud<pcl::PointXYZINormal> &point_cloud,
                       const std::unordered_map<colmap::image_t, colmap::Rigid3d> &pose_priors,
                       std::unordered_map<colmap::image_t, colmap::Image> &images, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras,
                       std::vector<xcolor::MatchTrack> &match_tracks);

void RunTwoViewBA(const xcolor::SfmConfig &config, const std::string &output_path, const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree,
                  const pcl::PointCloud<pcl::PointXYZINormal> &point_cloud, const std::unordered_map<colmap::image_t, colmap::Rigid3d> &pose_priors,
                  std::unordered_map<colmap::image_t, colmap::Image> &images, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras,
                  std::vector<xcolor::MatchTrack> &match_tracks);

}  // namespace xcolor
