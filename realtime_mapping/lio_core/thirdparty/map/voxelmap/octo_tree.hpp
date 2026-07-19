#pragma once

// #include "common_lib.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "omp.h"

#include <pcl/common/io.h>
#include <stdio.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <string>
#include <unordered_map>

#define HASH_P 116101
#define MAX_N 10000000000

class VOXEL_LOC
{
 public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const
  {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std
{
template <>
struct hash<VOXEL_LOC>
{
  int64_t operator()(const VOXEL_LOC &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
}  // namespace std

namespace xvmp
{
enum OCT_STATE
{
  OCTREE_UNINITIALIZED = 0,
  OCTREE_MIDDLE = 1,
  OCTREE_FIXED = 2
};

// a point to plane matching structure
typedef struct ptpl
{
  Eigen::Vector3d point;
  Eigen::Vector3d normal;
  Eigen::Vector3d center;
  Eigen::Matrix<double, 6, 6> plane_cov;
  double d;
  int layer;
} ptpl;

// 3D point with covariance
typedef struct pointStamped
{
  pointStamped(const double x, const double y, const double z, const double t)
  {
    point = Eigen::Vector3d(x, y, z);
    timestamp = t;
    // normal.setZero();
    view_point.setZero();
    view_vec.setZero();
  }

  pointStamped()
  {
    // normal.setZero();
    view_point.setZero();
    view_vec.setZero();
    timestamp = 0;
  }

  Eigen::Vector3d point;
  // Eigen::Vector3d normal;
  Eigen::Vector3d view_point;
  Eigen::Vector3d view_vec;
  // Eigen::Vector3d point_world;
  // Eigen::Matrix3d cov;
  double timestamp;
} pointStamped;

enum PlaneType
{
  PLANE_TYPE_IDENTITY = 0,
  PLANE_TYPE_PARALLEL,
  PLANE_TYPE_ORTHORHOMBIC,
  PLANE_TYPE_NEGATIVE,
  PLANE_TYPE_OTHERS,
  PLANE_TYPE_INVALID
};

class Plane
{
 public:
  using Ptr = std::shared_ptr<Plane>;
  using ConstPtr = std::shared_ptr<const Plane>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Plane() = default;

  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Matrix3d covariance;
  Eigen::Matrix3d eigen_vectors;
  float d = 0;
  float planarity = 0;
  float min_eigen_value = 1;
  float mid_eigen_value = 1;
  float max_eigen_value = 1;
  int points_size = 0;
  int layer = 0;
  double std_deviation;
  double radius;
  double timestamp = 0;

  bool is_plane = false;
  bool is_init = false;
  int id;
  // is_update and last_update_points_size are only for publish plane
  bool is_update = false;
  int last_update_points_size = 0;
  bool update_enable = true;
};

class OctoTree
{
 public:
  using Ptr = std::shared_ptr<OctoTree>;
  using ConstPtr = std::shared_ptr<const OctoTree>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::vector<pointStamped> temp_points_;  // all points in an octo tree
  std::vector<pointStamped> new_points_;   // new points in an octo tree
  std::vector<pointStamped> recent_points_;
  std::vector<Plane::Ptr> plane_vec_;  // 每颗八叉树中包含的所有平面
  Plane::Ptr active_plane_ptr_;        // 当前被激活的平面
  // Plane::Ptr recent_plane_ptr_;        // 最近点组成的平面

  int max_layer_;
  bool indoor_mode_;
  int layer_;
  int octo_state_;  // not_inited, middle, fixed
  OctoTree *leaves_[8];
  double voxel_center_[3];  // x, y, z
  std::vector<int> layer_point_size_;
  float quater_length_;
  float planer_threshold_;
  int min_plane_update_threshold_;
  int update_size_threshold_;
  int all_points_num_;
  int new_points_num_;
  int max_points_size_;
  int max_cov_points_size_;

  bool octree_update_enable_;
  bool plane_reinit_enable_;

  double last_visit_time_;
  int invalid_obs_count_ = 0;

  OctoTree(
      int max_layer,
      int layer,
      std::vector<int> layer_point_size,
      int max_point_size,
      int max_cov_points_size,
      float planer_threshold)
      : max_layer_(max_layer),
        layer_(layer),
        layer_point_size_(layer_point_size),
        max_points_size_(max_point_size),
        max_cov_points_size_(max_cov_points_size),
        planer_threshold_(planer_threshold)
  {
    temp_points_.clear();
    octo_state_ = OCTREE_UNINITIALIZED;
    new_points_num_ = 0;
    all_points_num_ = 0;
    // when new points num > 5, do a update
    update_size_threshold_ = 5;

    octree_update_enable_ = true;
    plane_reinit_enable_ = false;

    min_plane_update_threshold_ = layer_point_size_[layer_];

    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    active_plane_ptr_.reset(new Plane);
    // recent_plane_ptr_.reset(new Plane);
  }

  // check is plane , calc plane parameters including plane covariance
  void init_plane(const std::vector<pointStamped> &points, Plane::Ptr plane);

  // only update plane normal, center and radius with new points
  void update_plane(const std::vector<pointStamped> &points, Plane::Ptr plane);

  bool find_plane(const Plane::Ptr plane_ptr, Plane::Ptr &target_plane);

  int compare_plane(const Plane::Ptr plane0_ptr, const Plane::Ptr plane1_ptr);

  bool insert_plane(Plane::Ptr plane_ptr);

  void init_octo_tree();

  void cut_octo_tree();

  void reset_octo_tree();

  void clear_points();

  Plane::Ptr get_active_plane();

  void get_all_planes(std::vector<Plane::Ptr> &planes);

  size_t get_all_points_count();

  void get_planes_in_current_layer(
      std::vector<Plane::Ptr> &planes,
      const Eigen::Vector3d &view_vec,
      bool use_normal,
      double plane_threlshold);

  bool search_plane(
      const Eigen::Vector3d &pt,
      Plane::Ptr &plane,
      const Eigen::Vector3d &view_vec,
      bool use_normal,
      double plane_threlshold);

  bool planarity_and_normal_check(
      const Plane::Ptr plane,
      const Eigen::Vector3d &view_vec,
      bool use_normal,
      double plane_threlshold);

  void get_active_planes(std::vector<Plane::Ptr> &planes);

  void UpdateOctoTree(pointStamped &pv);
};
}  // namespace xvmp