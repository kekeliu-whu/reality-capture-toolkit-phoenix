#include "voxel_map.hpp"

#include <gflags/gflags.h>

#define XVMP_LOG(severity) \
  LOG(severity) << std::fixed << std::setprecision(6) << std::setfill(' ') << "[xvmp]"

namespace xvmp
{
VoxelMap::VoxelMap() {}

VoxelMap::~VoxelMap() {}

void VoxelMap::init(
    const PointCloud &input_points,
    const float voxel_size,
    const int max_layer,
    const std::vector<int> &layer_point_size,
    const int max_points_size,
    const int max_cov_points_size,
    const float planer_threshold,
    const Eigen::Vector3d &view_point)
{
  XVMP_LOG(INFO) << "init, voxel_size=" << m_voxel_size << ", max_layer=" << max_layer;

  m_voxel_size = voxel_size;
  m_max_layer = max_layer;
  m_layer_point_size = layer_point_size;
  m_max_points_size = max_points_size;
  m_max_cov_points_size = max_cov_points_size;
  m_planer_threshold = planer_threshold;

  // 点放入voxel对应的octo tree中
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    pointStamped p_v(
        input_points[i].x, input_points[i].y, input_points[i].z, input_points.header.stamp / 1e6);
    p_v.view_point = view_point;
    p_v.view_vec = p_v.view_point - p_v.point;

    VOXEL_LOC position = get_voxel_index(p_v.point);
    auto iter = m_voxel_map.find(position);
    if (iter != m_voxel_map.end())
    {
      m_voxel_map[position]->temp_points_.push_back(p_v);
      m_voxel_map[position]->new_points_num_++;
    }
    else
    {
      OctoTree *octo_tree = new OctoTree(
          m_max_layer,
          0,
          m_layer_point_size,
          m_max_points_size,
          m_max_cov_points_size,
          m_planer_threshold);
      m_voxel_map[position] = octo_tree;
      m_voxel_map[position]->quater_length_ = voxel_size / 4;
      m_voxel_map[position]->voxel_center_[0] = (0.5 + position.x) * m_voxel_size;
      m_voxel_map[position]->voxel_center_[1] = (0.5 + position.y) * m_voxel_size;
      m_voxel_map[position]->voxel_center_[2] = (0.5 + position.z) * m_voxel_size;
      m_voxel_map[position]->temp_points_.push_back(p_v);
      m_voxel_map[position]->new_points_num_++;
      m_voxel_map[position]->layer_point_size_ = m_layer_point_size;
    }
  }
  for (auto iter = m_voxel_map.begin(); iter != m_voxel_map.end(); ++iter)
  {
    iter->second->init_octo_tree();
  }
}

void VoxelMap::update(const PointCloud &input_points, const Eigen::Vector3d &view_point)
{
  uint plsize = input_points.size();
  XVMP_LOG(INFO) << "update, input points size=" << plsize;

  for (uint i = 0; i < plsize; i++)
  {
    pointStamped p_v(
        input_points[i].x, input_points[i].y, input_points[i].z, input_points.header.stamp / 1e6);
    // printf("pv.stamp %lf\n", p_v.timestamp);
    p_v.view_point = view_point;
    p_v.view_vec = p_v.view_point - p_v.point;

    VOXEL_LOC position = get_voxel_index(p_v.point);
    auto iter = m_voxel_map.find(position);
    if (iter != m_voxel_map.end())
    {
      m_voxel_map[position]->UpdateOctoTree(p_v);
    }
    else
    {
      OctoTree *octo_tree = new OctoTree(
          m_max_layer,
          0,
          m_layer_point_size,
          m_max_points_size,
          m_max_cov_points_size,
          m_planer_threshold);
      m_voxel_map[position] = octo_tree;
      m_voxel_map[position]->quater_length_ = m_voxel_size / 4;
      m_voxel_map[position]->voxel_center_[0] = (0.5 + position.x) * m_voxel_size;
      m_voxel_map[position]->voxel_center_[1] = (0.5 + position.y) * m_voxel_size;
      m_voxel_map[position]->voxel_center_[2] = (0.5 + position.z) * m_voxel_size;
      m_voxel_map[position]->UpdateOctoTree(p_v);
    }
  }
}

void VoxelMap::clear_cache(const double &curr_time)
{
  static double last_clear_time = curr_time;
  double clear_period = 60.0;

  if (curr_time - last_clear_time > clear_period)
  {
    for (auto &iter : m_voxel_map)
    {
      auto &octree_ptr = iter.second;
      if (0 == octree_ptr->layer_ && fabs(curr_time - octree_ptr->last_visit_time_) > clear_period)
      {
        octree_ptr->clear_points();
      }
    }
    last_clear_time = curr_time;
    XVMP_LOG(INFO) << "clear points before " << curr_time - clear_period;
    // XVMP_LOG(INFO) << "clear points before " << curr_time - clear_period << ", points_size=" <<
    // nn
    //                << ", release mem " << nn * sizeof(pointStamped) / 1024 << "kb";
  }
}

void VoxelMap::publish() {}

bool VoxelMap::search_plane(
    const Eigen::Vector3d &point,
    const Eigen::Vector3d &view_point,
    Plane::Ptr &plane,
    bool use_normal,
    double plane_threlshold)
{
  Eigen::Vector3d view_vec = view_point - point;
  VOXEL_LOC position = get_voxel_index(point);
  Eigen::Vector3d loc_xyz = get_voxel_index2(point);

  auto iter = m_voxel_map.find(position);
  if (iter != m_voxel_map.end())
  {
    if (iter->second->search_plane(point, plane, view_vec, use_normal, plane_threlshold))
    {
      return true;
    }
    else
    {
      OctoTree *current_octo = iter->second;
      VOXEL_LOC near_position = position;
      if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_))
        near_position.x = near_position.x + 1;
      else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_))
        near_position.x = near_position.x - 1;
      if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_))
        near_position.y = near_position.y + 1;
      else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_))
        near_position.y = near_position.y - 1;
      if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_))
        near_position.z = near_position.z + 1;
      else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_))
        near_position.z = near_position.z - 1;

      auto iter_near = m_voxel_map.find(near_position);
      if (iter_near != m_voxel_map.end())
      {
        if (iter_near->second->search_plane(point, plane, view_vec, use_normal, plane_threlshold))
        {
          return true;
        }
      }
    }
    // std::vector<Plane::Ptr > node_planes;
    // iter->second->get_planes(node_planes);
    // if (node_planes.size() > 0)
    // {
    //     plane = *node_planes[0];
    //     return true;
    // }
  }

  return false;
}

void VoxelMap::get_all_planes(std::vector<Plane::Ptr> &planes)
{
  std::vector<Plane::Ptr>().swap(planes);
  for (auto &iter : m_voxel_map)
  {
    std::vector<Plane::Ptr> node_planes;
    iter.second->get_all_planes(node_planes);
    planes.insert(planes.end(), node_planes.begin(), node_planes.end());
  }
  printf("voxel_size=%lu plane_size=%lu\n", m_voxel_map.size(), planes.size());
}

VOXEL_LOC VoxelMap::get_voxel_index(const Eigen::Vector3d &pt)
{
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = pt[j] / m_voxel_size;
    if (loc_xyz[j] < 0)
    {
      loc_xyz[j] -= 1.0;
    }
  }
  return VOXEL_LOC((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
}

Eigen::Vector3d VoxelMap::get_voxel_index2(const Eigen::Vector3d &pt)
{
  Eigen::Vector3d loc_xyz;
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = pt[j] / m_voxel_size;
    if (loc_xyz[j] < 0)
    {
      loc_xyz[j] -= 1.0;
    }
  }
  return loc_xyz;
}

void VoxelMap::print_statistics(std::string &str)
{
  std::vector<Plane::Ptr> planes;
  get_all_planes(planes);

  size_t n_points = 0;
  for (auto &iter : m_voxel_map)
  {
    n_points += iter.second->get_all_points_count();
  }
  char desc[2056];
  sprintf(
      desc,
      "voxel_size=%lu plane_size=%lu points_size=%lu\n",
      m_voxel_map.size(),
      planes.size(),
      n_points);

  str = std::string(desc);
}

}  // namespace xvmp
