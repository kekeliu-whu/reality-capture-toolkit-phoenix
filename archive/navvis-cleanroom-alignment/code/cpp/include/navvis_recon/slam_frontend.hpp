#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "navvis_recon/slam_imu.hpp"

namespace navvis_recon::slam {

struct Pose {
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();

  static Pose identity();
  Pose inverse() const;
  Pose compose(const Pose& other, bool normalize_rotation = false) const;
};

struct TimedRangeBatch {
  std::int64_t timestamp_ns = 0;
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> origins;
  std::vector<std::int64_t> point_timestamps_ns;
};

struct DeskewedRangeBatch {
  std::int64_t timestamp_ns = 0;
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> origins;
};

class PosePredictor {
 public:
  virtual ~PosePredictor() = default;
  virtual Pose predict(std::int64_t timestamp_ns) = 0;
  virtual DeskewedRangeBatch deskew(const TimedRangeBatch& batch) = 0;
  virtual void correct(std::int64_t timestamp_ns, const Pose& pose) = 0;
  virtual Eigen::Vector3d gravityObservation() const = 0;
};

// Production raw-IMU adapter. It keeps the two observable float boundaries
// used by Surveyor's range accumulator: ray -> batch start, then one
// start -> batch end matrix cast.
class ImuPosePredictor final : public PosePredictor {
 public:
  explicit ImuPosePredictor(
      std::vector<ImuSample> samples, double maximum_range_m = 60.0,
      Pose initial_pose = Pose::identity(),
      RawImuTrackerOptions options = RawImuTrackerOptions{});
  ~ImuPosePredictor() override;
  ImuPosePredictor(ImuPosePredictor&&) noexcept;
  ImuPosePredictor& operator=(ImuPosePredictor&&) noexcept;
  ImuPosePredictor(const ImuPosePredictor&) = delete;
  ImuPosePredictor& operator=(const ImuPosePredictor&) = delete;

  Pose predict(std::int64_t timestamp_ns) override;
  DeskewedRangeBatch deskew(const TimedRangeBatch& batch) override;
  void correct(std::int64_t timestamp_ns, const Pose& pose) override;
  Eigen::Vector3d gravityObservation() const override;

 private:
  class Implementation;
  std::unique_ptr<Implementation> implementation_;
};

struct FrontendConfig {
  // Optional trajectory gauge supplied by the surrounding mapping session.
  // It is applied after the first batch has been deskewed, matching the
  // installed frontend's first-node correction boundary.
  std::optional<Pose> initial_tracking_pose;

  float minimum_range_m = 0.5F;
  float maximum_range_m = 60.0F;
  float scan_voxel_m = 0.04F;
  float high_resolution_min_voxel_m = 0.02F;
  float high_resolution_max_voxel_m = 0.40F;
  std::size_t high_resolution_max_points = 5000;

  std::array<float, 3> correspondence_limits_m{0.15F, 0.45F, 0.90F};
  float initial_plane_distance_m = 0.20F;
  float contracted_plane_distance_m = 0.02F;
  float maximum_incidence_angle_degrees = 86.0F;
  int icp_threads = 8;
  int icp_iterations = 20;
  int icp_minimum_iterations = 6;
  int icp_contraction_iterations = 6;
  int icp_minimum_correspondences = 6;

  float submap_overlap_displacement_m = 5.0F;
  float submap_finish_displacement_m = 10.0F;

  // Cartographer-style motion filter. A batch is rejected only while all
  // three deltas from the most recently retained node remain below limits.
  double motion_filter_maximum_time_s = 0.50;
  double motion_filter_maximum_distance_m = 0.005;
  double motion_filter_maximum_angle_rad = 0.02 * 3.14159265358979323846 / 180.0;
};

struct IcpResult {
  Pose target_from_source;
  float plane_fitness_m = std::numeric_limits<float>::infinity();
  float euclidean_fitness_m = std::numeric_limits<float>::infinity();
  std::size_t correspondence_count = 0;
  int iteration_count = 0;
  bool converged = false;
};

struct FrontendNode {
  std::size_t index = 0;
  std::int64_t timestamp_ns = 0;
  Pose local_pose;
  std::vector<Eigen::Vector3f> points;
  IcpResult scan_match;
  Eigen::Vector3d gravity_observation{0.0, 0.0, 9.81};
};

struct SubmapSummary {
  std::size_t index = 0;
  std::int64_t start_timestamp_ns = 0;
  std::int64_t end_timestamp_ns = 0;
  Pose local_pose;
  std::vector<std::size_t> node_indices;
  std::array<std::size_t, 3> surfel_counts{};
  std::size_t probability_grid_cells = 0;
  bool finished = false;
  // Optional production-state payload. The normal trajectory-only command
  // ignores these arrays; --state-output serializes them for clean-room loop
  // closure and the Stage1/Stage2 backends.
  std::array<std::vector<Eigen::Vector3f>, 3> surfel_points;
  std::array<std::vector<Eigen::Vector3f>, 3> surfel_normals;
  std::vector<std::array<std::int32_t, 3>> probability_grid_indices;
  std::vector<std::uint16_t> probability_grid_values;
};

struct FrontendResult {
  struct Timing {
    double deskew_seconds = 0.0;
    double centroid_filter_seconds = 0.0;
    double prediction_and_icp_seconds = 0.0;
    double correction_and_motion_filter_seconds = 0.0;
    double node_filter_seconds = 0.0;
    double submap_insertion_seconds = 0.0;
    double submap_transform_seconds = 0.0;
    std::array<double, 3> surfel_update_seconds{};
    double probability_grid_seconds = 0.0;
    double finish_seconds = 0.0;
  } timing;
  std::vector<FrontendNode> nodes;
  std::vector<SubmapSummary> submaps;
  std::size_t processed_batches = 0;
  std::size_t motion_filtered_batches = 0;
};

// Persistent, fully C++ local trajectory builder. It owns the three-level
// split-surfel maps, probability grids and ICP indices; no Python callback is
// used on the processing path.
class Frontend {
 public:
  explicit Frontend(FrontendConfig config = {});
  ~Frontend();
  Frontend(Frontend&&) noexcept;
  Frontend& operator=(Frontend&&) noexcept;
  Frontend(const Frontend&) = delete;
  Frontend& operator=(const Frontend&) = delete;

  void process(const TimedRangeBatch& batch, PosePredictor& predictor);
  FrontendResult finish();

 private:
  class Implementation;
  std::unique_ptr<Implementation> implementation_;
};

}  // namespace navvis_recon::slam
