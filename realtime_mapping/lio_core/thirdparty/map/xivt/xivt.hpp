#pragma once

#include <glog/logging.h>
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/octree.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#include "geo/geo.hpp"
#include "robin_hood/robin_hood.h"
#include "xutil/arch/spinlock_table.hpp"
// #include "tsl/sparse_map.h"

namespace xivt
{
/// alias for eigen
using Vec2i = Eigen::Vector2i;
using Vec3i = Eigen::Vector3i;

using Vec2d = Eigen::Vector2d;
using Vec2f = Eigen::Vector2f;
using Vec3d = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Vec5d = Eigen::Matrix<double, 5, 1>;
using Vec5f = Eigen::Matrix<float, 5, 1>;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Vec6f = Eigen::Matrix<float, 6, 1>;
using Vec15d = Eigen::Matrix<double, 15, 15>;

using Mat1d = Eigen::Matrix<double, 1, 1>;
using Mat3d = Eigen::Matrix3d;
using Mat3f = Eigen::Matrix3f;
using Mat4d = Eigen::Matrix4d;
using Mat4f = Eigen::Matrix4f;
using Mat5d = Eigen::Matrix<double, 5, 5>;
using Mat5f = Eigen::Matrix<float, 5, 5>;
using Mat6d = Eigen::Matrix<double, 6, 6>;
using Mat6f = Eigen::Matrix<float, 6, 6>;
using Mat15d = Eigen::Matrix<double, 15, 15>;

using Quatd = Eigen::Quaterniond;
using Quatf = Eigen::Quaternionf;

/// less of vector
template <int N>
struct less_vec
{
  inline bool operator()(const Eigen::Matrix<int, N, 1>& v1, const Eigen::Matrix<int, N, 1>& v2)
      const;
};

/// hash of vector
template <int N>
struct hash_vec
{
  inline size_t operator()(const Eigen::Matrix<int, N, 1>& v) const;
};

// template <int N>
// struct less_vec_ptr {
//     inline bool operator()(const Eigen::Matrix<int, N, 1>* v1, const Eigen::Matrix<int, N, 1>*
//     v2) const;
// };

/// implementation
template <>
inline bool less_vec<2>::operator()(
    const Eigen::Matrix<int, 2, 1>& v1,
    const Eigen::Matrix<int, 2, 1>& v2) const
{
  return v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]);
}

template <>
inline bool less_vec<3>::operator()(
    const Eigen::Matrix<int, 3, 1>& v1,
    const Eigen::Matrix<int, 3, 1>& v2) const
{
  return v1[0] < v2[0] ||
         (v1[0] == v2[0] && v1[1] < v2[1]) && (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] < v2[2]);
}

// template <>
// inline bool less_vec_ptr<3>::operator()(const Eigen::Matrix<int, 3, 1>* v1, const
// Eigen::Matrix<int, 3, 1>* v2) const {
//     return (*v1)[0] < (*v2)[0] || ((*v1)[0] == (*v2)[0] && (*v1)[1] < (*v2)[1]) && ((*v1)[0] ==
//     (*v2)[0] && (*v1)[1] == (*v2)[1] && (*v1)[2] < (*v2)[2]);
// }

/// vec 2 hash
/// @see Optimized Spatial Hashing for Collision Detection of Deformable Objects, Matthias Teschner
/// et. al., VMV 2003
template <>
inline size_t hash_vec<2>::operator()(const Eigen::Matrix<int, 2, 1>& v) const
{
  return size_t((size_t(v[0]) * 73856093) ^ (size_t(v[1]) * 471943)) % 10000000;
}

/// vec 3 hash
template <>
inline size_t hash_vec<3>::operator()(const Eigen::Matrix<int, 3, 1>& v) const
{
  return size_t((size_t(v[0]) * 73856093) ^ (size_t(v[1]) * 471943) ^ (size_t(v[2]) * 83492791));
}

constexpr auto less_vec2i = [](const Vec2i& v1, const Vec2i& v2)
{ return v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]); };

constexpr int t_dim = 3;

using Point_type = pcl::PointXYZINormal;
#ifdef V1207
using XiVT_point_type = pcl::PointXYZINormal;
#else
using XiVT_point_type = pcl::PointXYZI;
#endif
using Point_vector = std::vector<Point_type, Eigen::aligned_allocator<Point_type>>;
using XiVT_point_vector = std::vector<XiVT_point_type, Eigen::aligned_allocator<XiVT_point_type>>;

using PointT = Eigen::Matrix<float, t_dim, 1>;

using XiVT_key_type = Eigen::Matrix<int, t_dim, 1>;

// convert from pcl point to eigen
template <typename T, int t_dim, typename PointType>
inline Eigen::Matrix<T, t_dim, 1> ToEigen(const PointType& pt)
{
  return Eigen::Matrix<T, t_dim, 1>(pt.x, pt.y, pt.z);
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZ>(const pcl::PointXYZ& pt)
{
  return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZI>(const pcl::PointXYZI& pt)
{
  return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZINormal>(
    const pcl::PointXYZINormal& pt)
{
  return pt.getVector3fMap();
}

class XiVT_node;

struct XiVT_fake_float_01
{
  uint8_t m_value_int = 0;
  XiVT_fake_float_01()
  {
  }
  XiVT_fake_float_01(float value)
  {
    m_value_int = uint8_t(value * 255);
  }
  float get() const
  {
    return m_value_int / 255.0f;
  }
  void set(float value)
  {
    m_value_int = uint8_t(value * 255);
  }
} __attribute__((packed));

// struct XiVT_fake_float_neg1_to1 {
//     uint8_t m_value_int = 0;
//     XiVT_fake_float_neg1_to1() {}
//     XiVT_fake_float_neg1_to1(float value) {
//         m_value_int = uint8_t((value + 1) * 127.5f);
//     }
//     float get() const {
//         return (m_value_int / 127.5f) - 1;
//     }
//     void set(float value) {
//         m_value_int = uint8_t((value + 1) * 127.5f);
//     }
// } __attribute__((packed));

struct XiVT_fake_float_neg1_to1
{
  float m_value = 0;
  XiVT_fake_float_neg1_to1()
  {
  }
  XiVT_fake_float_neg1_to1(float value)
  {
    m_value = value;
  }
  float get() const
  {
    return m_value;
  }
  void set(float value)
  {
    m_value = value;
  }
} __attribute__((packed));

struct XiVT_point_status
{
  bool m_with_valid_normal = false;
  bool m_deleted_mark = false;
  uint8_t m_fake_normal_angle = 0;
  uint8_t m_total_attempt_count = 0;
  int m_last_attempt_frame = -9999;
  int m_frame_id;

  XiVT_fake_float_01 m_squared_distances_to_center;

  XiVT_fake_float_neg1_to1 m_normal_x;
  XiVT_fake_float_neg1_to1 m_normal_y;
  XiVT_fake_float_neg1_to1 m_normal_z;

  Vec3f get_normal() const
  {
    return Vec3f(m_normal_x.get(), m_normal_y.get(), m_normal_z.get());
  }
  void set_normal(const Vec3f& normal)
  {
    m_normal_x.set(normal[0]);
    m_normal_y.set(normal[1]);
    m_normal_z.set(normal[2]);
  }
  void set_normal(float x, float y, float z)
  {
    m_normal_x.set(x);
    m_normal_y.set(y);
    m_normal_z.set(z);
  }
} __attribute__((packed));

struct XiVT_hyper_grid_context
{
  // Point_vector& m_points;
  // std::vector<bool> m_deleted_mark;
  // std::vector<float> m_squared_distances_to_center;
  // tsl::sparse_map<XiVT_key_type, int, hash_vec<3>> m_down_sample_voxels;
  robin_hood::unordered_flat_map<XiVT_key_type, int, hash_vec<3>> m_down_sample_voxels;
  std::vector<XiVT_node*> m_nodes;
  std::vector<XiVT_point_status> m_point_status;

  std::vector<int> m_points_without_normal_offsets;

  // KDTree and lock
  float m_voxel_size_inv;
  float m_voxel_size;
  pcl::PointCloud<XiVT_point_type>::Ptr m_point_cloud;  // TODO: compress normal
  pcl::search::KdTree<XiVT_point_type>::Ptr m_kd_tree;
  pcl::search::Octree<XiVT_point_type>::Ptr m_oc_tree;
  // std::vector<int> m_frame_id;

  decltype(m_point_cloud->points)& m_points_ref;  // for GDB

  int m_indexed_till = -1;  // m_points[0:m_indexed_till] is indexed
  int m_modified_count = 0;

  // statistics
  int m_erased_point_count = 0;
  int m_grid_count = 0;
  int m_erased_grid_count = 0;

  XiVT_hyper_grid_context(float voxel_size)
      : m_voxel_size(voxel_size),
        m_point_cloud(new pcl::PointCloud<XiVT_point_type>()),
        m_kd_tree(new pcl::search::KdTree<XiVT_point_type>()),
        m_oc_tree(new pcl::search::Octree<XiVT_point_type>(m_voxel_size * 10)),
        m_points_ref(m_point_cloud->points)
  {
    // m_points.resize(0);
  }
  // m_points(m_point_cloud->points)

  ~XiVT_hyper_grid_context();
};

class XiVT_node
{
 public:
  XiVT_node(XiVT_key_type key)
  {
    m_node_key = key;
    m_enough_point_count = 0;
    m_good_plane_count = 0;
    m_valid = true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  struct Point_with_distance;

  XiVT_node()
  {
    m_enough_point_count = 0;
    m_good_plane_count = 0;
    m_valid = true;
  }

  XiVT_node(const XiVT_node& node)
  {
    m_node_key = node.m_node_key;
    m_hyper_grid_context = node.m_hyper_grid_context;
    m_point_offsets = node.m_point_offsets;
    m_enough_point_count = (int)(node.m_enough_point_count);
    m_good_plane_count = (int)(node.m_good_plane_count);
    m_valid = (bool)(node.m_valid);
  }

  inline bool Empty() const;

  inline std::size_t Size() const;

  int incremental_knn(
      std::vector<Point_with_distance>& dis_points,
      const XiVT_point_type& point,
      const Vec3f& view_point,
      const int& K,
      float& max_squared_range,
      bool by_normal,
      XiVT_hyper_grid_context* src_context);

  XiVT_key_type m_node_key;
  XiVT_hyper_grid_context* m_hyper_grid_context;
  std::vector<int> m_point_offsets, m_point_offsets_2;

  std::atomic<int> m_enough_point_count, m_good_plane_count;
  std::atomic<bool> m_valid;
};

struct XiVT_node::Point_with_distance
{
  float dist = 0;
  XiVT_hyper_grid_context* m_hyper_grid = nullptr;
  int idx = 0;

  Point_with_distance() = default;
  Point_with_distance(const double d, XiVT_hyper_grid_context* n, const int i)
      : dist(d), m_hyper_grid(n), idx(i)
  {
  }

  inline bool operator()(const Point_with_distance& p1, const Point_with_distance& p2)
  {
    return p1.dist < p2.dist;
  }

  inline bool operator<(const Point_with_distance& rhs)
  {
    return dist < rhs.dist;
  }
};

struct XiVT_knn_query_status
{
  int m_filtered_by_normal_count = 0, m_zero_normal_count = 0, m_filtered_by_zero_normal_count = 0,
      m_knn_count = 0, m_filtered_by_plane_dist_count = 0;

  float m_planarity, pd2, s;

  bool m_filtered_via_count = false;
  bool m_filtered_via_rev = false;
  bool m_filtered_via_plane = false;
  bool m_valid_plane = false;
};

class XiVT
{
 public:
  using Point_with_distance = typename XiVT_node::Point_with_distance;

  enum class NearbyType
  {
    CENTER,  // center only
    NEARBY6,
    NEARBY18,
    NEARBY26,
  };

  struct Options
  {
    float m_resolution = 0.2;
    float m_inv_resolution = 10.0;
    int m_downsample_ratio = 2;
    float m_downsample_resolution = 0.2;
    float m_downsample_resulution_inv;
    float m_downsample_half_sphere_sq_distance;
    NearbyType m_nearby_type = NearbyType::NEARBY6;  // nearby range

    std::size_t m_capacity = 100000000;  // capacity
    int m_erase_skip = 1;                // if == 1, no skip
    int m_erase_batch_size = 1000;
    bool m_use_index;

    int m_instant_break_policy = 1;
    float m_dynamic_normal_deg_max = 80, m_dynamic_normal_planarity_threshold = 0.45;
    int m_use_dynamic_normal = 0, m_init_normal_mode = 0;
    float m_rev_normal_threshold = 0.9;
    float m_close_normal_threshold = 2;
  };

  explicit XiVT(Options options);

  Eigen::Matrix<float, t_dim, 1> get_voxel_center_for_point(float x, float y, float z);

  void add_points(
      const Point_vector& points_to_add,
      int current_frame_id,
      const std::vector<XiVT_point_status>& point_status,
      bool recompute_normal);

  /// get nn with condition
#ifdef V1207
  XiVT_knn_query_status get_closest_point(
      const Point_type& query_point,
      const Vec3f& view_point,
      Point_vector& closest_pt,
      Eigen::Matrix<float, 9, 1>& pca_result,
      int _min_num,
      int _max_num,
      double max_range,
      bool by_normal,
      int zero_dot_filter_num = 2);
#else
  XiVT_knn_query_status get_closest_point(
      const XiVT_point_type& pt,
      const Vec3f& view_point,
      XiVT_point_vector& closest_pt,
      Plane& plane_result,
      int _min_num,
      int _max_num,
      double max_range,
      bool by_normal,
      int zero_filter_num = 2);
#endif

  void perform_forget(int frame_threshold);
  void recompute_normal(
      int frame_id,
      int frame_stride,
      int min_interval,
      int max_attempt,
      int min_num,
      int max_num,
      double max_range);
  void add_viewpoint(const Vec3f& viewpoint, int frame_id);

  /// get number of points
  size_t NumPoints() const;

  /// get number of valid grids
  size_t NumValidGrids() const;

  /// get statistics of the points
  std::vector<float> StatGridPoints() const;

  void build_index(bool force, int threshold = 500);
  void garbage_collect();
  std::string statistics_string() const;

  std::atomic<bool> m_busy_flag;  // set true when next frame comes

  void test_hyper_grids_consistency() const;

  void memory_test();

  pcl::PointCloud<XiVT_point_type>::Ptr get_all_points(bool last_call = false);

  pcl::PointCloud<XiVT_point_type>::Ptr get_downsampled_points(int count, bool last_call = false);

 private:
  /// generate the nearby grids according to the given options
  void GenerateNearbyGrids();

  /// position to grid
  inline XiVT_key_type position_to_voxel(const PointT& pt) const
  {
    return (pt * m_options.m_downsample_resulution_inv).array().round().template cast<int>();
  }

  Options m_options;
  std::unordered_map<
      XiVT_key_type,
      typename std::list<std::pair<XiVT_key_type, XiVT_node>>::iterator,
      hash_vec<t_dim>>
      m_nodes_map;  // voxel hash map

  std::unordered_map<XiVT_key_type, XiVT_hyper_grid_context*, hash_vec<t_dim>> m_hyper_grids;

  std::list<std::pair<XiVT_key_type, XiVT_node>> m_grids_lru_cache;  // voxel cache

  std::vector<XiVT_key_type> m_nearby_grids;  // nearbys

  std::vector<Vec3f> m_viewpoints;

  Spinlock_table<1024> m_lock_table;
  std::mutex m_mutex;
};

}  // namespace xivt