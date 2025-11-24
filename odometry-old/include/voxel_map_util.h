#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP

#include <openssl/md5.h>
#include <pcl/common/io.h>
#include <rosbag/bag.h>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <execution>
#include <string>
#include <unordered_map>
#include "common_lib.h"
#include "omp.h"

#define HASH_P 116101
#define MAX_N 10000000000

typedef struct PointWithCovMeta {
  Eigen::Vector3d pw;   // point in world frame
  Eigen::Matrix3d cov;  // point covariance in world frame
} pointWithCovMeta;

// 3D point with covariance
typedef struct PointWithCov : pointWithCovMeta {
  Eigen::Vector3d pl;        // point in lidar frame
  Eigen::Vector3d pi;        // point in imu frame
  Eigen::Matrix3d body_cov;  // point covariance in lidar/body frame
} pointWithCov;

// a point to plane matching structure
typedef struct PointPlaneMatchInfo {
  pointWithCov pv;
  Eigen::Vector3d normal;                 // plane normal vector in world frame
  Eigen::Vector3d center;                 // plane center point in world frame
  Eigen::Matrix<double, 6, 6> plane_cov;  // plane covariance in world frame
  double d;                               // used to compute point-plane distance
  int layer;
} ptpl;

typedef struct Plane {
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Vector3d y_normal;
  Eigen::Vector3d x_normal;
  Eigen::Matrix3d covariance;
  Eigen::Matrix<double, 6, 6> plane_cov;
  float radius          = 0;  // if the plane points are evenly distributed in a circle, then radius*2 will be real radius of the circle
  float min_eigen_value = 1;
  float mid_eigen_value = 1;
  float max_eigen_value = 1;
  float d               = 0;
  int points_size       = 0;

  bool is_plane = false;
  int id;
} Plane;

class VoxelLoc {
 public:
  int64_t x, y, z;

  VoxelLoc(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VoxelLoc &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <>
struct hash<VoxelLoc> {
  int64_t operator()(const VoxelLoc &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
}  // namespace std

class OctoTree {
 public:
  std::vector<pointWithCovMeta> temp_points_;  // all points in an octo tree
  std::vector<pointWithCovMeta> new_points_;   // new points in an octo tree
  Plane *plane_ptr_;
  int max_layer_;
  int layer_;
  int octo_state_;  // 0 is end of tree, 1 is not
  OctoTree *leaves_[8];
  double voxel_center_[3];  // x, y, z
  std::vector<int> layer_point_size_;
  float quarter_length_;
  float planer_threshold_;
  int max_plane_update_threshold_;
  int update_size_threshold_;
  int all_points_num_;
  int new_points_num_;
  int max_points_size_;
  int max_cov_points_size_;
  bool init_octo_;
  bool update_cov_enable_;
  bool update_enable_;

  OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
           int max_point_size, int max_cov_points_size, float planer_threshold);

  // check is plane , calc plane parameters including plane covariance
  void InitPlane(const std::vector<pointWithCovMeta> &points, Plane *plane);

  // only update plane normal, center and radius with new points
  void UpdatePlane(const std::vector<pointWithCovMeta> &points, Plane *plane);

  void InitOctoTree();

  void CutOctoTree();

  void UpdateOctoTree(const pointWithCovMeta &pv);
};

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
            uint8_t &b);

void BuildVoxelMap(const std::vector<pointWithCov> &input_points,
                   const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size, const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VoxelLoc, OctoTree *> &feat_map);

void UpdateVoxelMap(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VoxelLoc, OctoTree *> &feat_map);

void BuildSingleResidual(const pointWithCov &pv, const OctoTree *current_octo,
                         const int current_layer, const int max_layer,
                         const double sigma_num, bool &is_success,
                         double &prob, ptpl &single_ptpl);

void GetUpdatePlane(const OctoTree *current_octo, const int pub_max_voxel_layer,
                    std::vector<Plane> &plane_list);

void BuildResidualListOMP(const unordered_map<VoxelLoc, OctoTree *> &voxel_map,
                          const double voxel_size, const double sigma_num,
                          const int max_layer,
                          const std::vector<pointWithCov> &pv_list,
                          std::vector<ptpl> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match);

void BuildResidualListNormal(
    const unordered_map<VoxelLoc, OctoTree *> &voxel_map,
    const double voxel_size, const double sigma_num, const int max_layer,
    const std::vector<pointWithCovMeta> &pv_list, std::vector<ptpl> &ptpl_list,
    std::vector<Eigen::Vector3d> &non_match);

void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec,
                     const Eigen::Vector3d &z_vec,
                     geometry_msgs::Quaternion &q);

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q);

// eq.1
void CalcBodyCov(Eigen::Vector3d &pb, const float range_inc,
                 const float degree_inc, Eigen::Matrix3d &cov);

template <typename T>
void DownSamplingVoxel(const pcl::PointCloud<PointType> &cloud_in,
                       pcl::PointCloud<PointType> &cloud_out,
                       double voxel_size);

template <typename T>
void DownSamplingVoxelRandom(const pcl::PointCloud<PointType> &cloud_in,
                             pcl::PointCloud<PointType> &cloud_out,
                             double voxel_size);

void InitVoxelMapParams(double min_plane_likeness);

#endif