#pragma once

#include <ceres/ceres.h>
#include <colmap/estimators/cost_functions.h>
#include <colmap/geometry/triangulation.h>
#include <colmap/scene/database.h>
#include <colmap/scene/database_cache.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/scene/reconstruction_io.h>
#include <glog/logging.h>
#include <omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>

namespace xcolor {

struct SfmConfig {
  int min_num_matches         = 15;
  bool ignore_watermarks      = false;
  bool refine_focal_length    = true;
  bool refine_principal_point = true;
  bool refine_extra_params    = true;
  double min_tri_angle        = 5;

  int outer_opt_num_two_view                           = 1;
  double reproject_error_outlier_thresholds_twoview[1] = {6};    // pixels
  double lidar_error_outlier_thresholds_twoview[1]     = {0.6};  // meters

  int outer_opt_num_multi_view                           = 3;
  double reproject_error_outlier_thresholds_multiview[3] = {5, 4, 3};        // pixels
  double lidar_error_outlier_thresholds_multiview[3]     = {0.4, 0.3, 0.15};  // meters

  double reproject_cauchy_loss_scale = 1.5;  // pixels, half of the smallest reproject_error_outlier_thresholds

  double lidar_cauchy_weight  = 0.1;  // meters, half of the smallest lidar_error_outlier_thresholds
  double lidar_weight_scale   = 0.2;
  bool enable_weight_by_depth = true;

  double pose_prior_rotation_weight    = 0.2 * M_PI / 180.0;
  double pose_prior_translation_weight = 0.05;
  double pose_prior_weight_scale       = 1;

  int ba_optimization_num_threads = 1;

  bool use_all_track = false;
  int min_track_len  = 3;

  SfmConfig(int cores_used) { ba_optimization_num_threads = cores_used; }
};

struct Point2DInfo {
  typedef std::shared_ptr<Point2DInfo> Ptr;

  colmap::image_t image_id;
  colmap::camera_t camera_id;
  Eigen::Vector2d point_pixel;
};

struct Point3DInfo {
  typedef std::shared_ptr<Point3DInfo> Ptr;

  bool valid = false;  // =false means this point3D is not used in the optimization
  Eigen::Vector3d point3D;
  ceres::ResidualBlockId residual_block_id       = nullptr;
  ceres::ResidualBlockId lidar_residual_block_id = nullptr;
};

struct MatchTrack {
  std::vector<Point2DInfo::Ptr> point2D_on_imageN;
  Point3DInfo point3D;
};

class LidarPlaneCostFunction {
 public:
  explicit LidarPlaneCostFunction(const Eigen::Vector3d &center, const Eigen::Vector3d &normal) : center_(center), normal_(normal) {}

  static ceres::CostFunction *Create(const Eigen::Vector3d &center, const Eigen::Vector3d &normal) {
    return new ceres::AutoDiffCostFunction<LidarPlaneCostFunction, 1, 3>(new LidarPlaneCostFunction(center, normal));
  }

  template <typename T>
  bool operator()(const T *const point3D_ptr, T *residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point3D{point3D_ptr};
    residuals[0] = normal_.cast<T>().dot(point3D - center_.cast<T>());
    return true;
  }

 private:
  const Eigen::Vector3d center_;
  const Eigen::Vector3d normal_;
};

class PosePriorCostFunction {
 public:
  PosePriorCostFunction(const Eigen::Quaterniond &R_prior, const Eigen::Vector3d &t_prior, Eigen::Matrix<double, 6, 6> sqrt_information)
      : R_prior_(R_prior), t_prior_(t_prior), sqrt_information_(std::move(sqrt_information)) {}

  template <typename T>
  bool operator()(const T *const R_ptr, const T *const t_ptr, T *residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T>> R_a(R_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_a(t_ptr);

    Eigen::Quaternion<T> R_error   = R_a.conjugate() * R_prior_.cast<T>();
    Eigen::Matrix<T, 3, 1> t_error = R_a.conjugate() * (t_prior_.cast<T>() - t_a);

    // Compute the residuals
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template head<3>() = t_error;
    residuals.template tail<3>() = T(2.0) * R_error.vec();

    // Apply the square root information matrix
    residuals = sqrt_information_.template cast<T>() * residuals;

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Quaterniond &R_prior, const Eigen::Vector3d &t_prior,
                                     const Eigen::Matrix<double, 6, 6> &sqrt_information) {
    return new ceres::AutoDiffCostFunction<PosePriorCostFunction, 6, 4, 3>(new PosePriorCostFunction(R_prior, t_prior, sqrt_information));
  }

 private:
  Eigen::Quaterniond R_prior_;
  Eigen::Vector3d t_prior_;
  Eigen::Matrix<double, 6, 6> sqrt_information_;
};

std::vector<MatchTrack> GenerateMatchPairs(const colmap::CorrespondenceGraph &corr_graph,
                                           const std::unordered_map<colmap::image_t, colmap::Image> &images, const SfmConfig &config);

void ParameterizeCameras(const SfmConfig &config, ceres::Problem &problem, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras);

void ParameterizePoses(const SfmConfig &config, ceres::Problem &problem, std::unordered_map<colmap::camera_t, colmap::Image> &images);

Eigen::Vector2d ComputePixelError(Eigen::Vector3d &point3D, const Eigen::Vector2d &point2D, const colmap::Image &image, const colmap::Camera &camera);

double ComputeLidarError(Eigen::Vector3d &point3D, const Eigen::Vector3d &center, const Eigen::Vector3d &normal);

void AddReprojectFactorToProblem(ceres::Problem &problem, Eigen::Vector3d &point3D, const Eigen::Vector2d &point2D, colmap::Image &image,
                                 colmap::Camera &camera, ceres::LossFunction *loss_function, std::vector<ceres::ResidualBlockId> &residual_block_ids);

void AddLidarFactorToProblem(ceres::Problem &problem, Eigen::Vector3d &point3D, const Eigen::Vector3d &center, const Eigen::Vector3d &normal,
                             ceres::LossFunction *loss_function, std::vector<ceres::ResidualBlockId> &residual_block_ids);

void AddPosePriorsToProblem(const SfmConfig &config, ceres::Problem &problem, const std::unordered_map<colmap::image_t, colmap::Rigid3d> &pose_priors,
                            std::unordered_set<colmap::image_t> &optimized_image_ids, std::unordered_map<colmap::image_t, colmap::Image> &images);

void PrintResidualHistogram(double threshold, ceres::Problem &problem, const std::vector<ceres::ResidualBlockId> &residual_block_ids,
                            const std::string &name);

bool MergeTrack(const std::vector<MatchTrack> &match_tracks, std::vector<MatchTrack> &match_tracks_merged, bool use_all_track, int min_track_size);

}  // namespace xcolor