#pragma once

#include "octo_tree.hpp"

typedef pcl::PointCloud<pcl::PointXYZINormal> PointCloud;
// typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

namespace xvmp
{
class VoxelMap
{
 public:
  VoxelMap();

  ~VoxelMap();

  void init(
      const PointCloud &input_points,
      const float voxel_size,
      const int max_layer,
      const std::vector<int> &layer_point_size,
      const int max_points_size,
      const int max_cov_points_size,
      const float planer_threshold,
      const Eigen::Vector3d &view_point);

  void update(const PointCloud &input_points, const Eigen::Vector3d &view_point);

  void clear_cache(const double &curr_time);

  void publish();

  bool search_plane(
      const Eigen::Vector3d &point,
      const Eigen::Vector3d &view_point,
      Plane::Ptr &plane,
      bool use_normal = true,
      double plane_threlshold = 0.01);

  void get_all_planes(std::vector<Plane::Ptr> &planes);

  void print_statistics(std::string &str);

 private:
  VOXEL_LOC get_voxel_index(const Eigen::Vector3d &pt);
  Eigen::Vector3d get_voxel_index2(const Eigen::Vector3d &pt);

  float m_voxel_size;
  int m_max_layer;
  std::vector<int> m_layer_point_size;
  int m_max_points_size;
  int m_max_cov_points_size;
  float m_planer_threshold;

  std::unordered_map<VOXEL_LOC, OctoTree *> m_voxel_map;
};

}  // namespace xvmp