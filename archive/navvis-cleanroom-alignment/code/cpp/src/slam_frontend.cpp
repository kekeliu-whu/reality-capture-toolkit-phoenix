#include "navvis_recon/slam_frontend.hpp"

#include "navvis_recon/slam_frontend_native.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace navvis_recon::slam {
namespace {

static_assert(sizeof(Eigen::Vector3f) == 3U * sizeof(float));

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kIcpTranslationStepM = static_cast<double>(1.0e-5F);
constexpr double kIcpRotationStepRad =
    static_cast<double>(static_cast<float>(1.0e-5 * kPi / 180.0));
constexpr std::int64_t kSurfelMinimumCell = -(1LL << 20);
constexpr std::int64_t kSurfelMaximumCell = (1LL << 20) - 1LL;
constexpr std::uint64_t kSurfelCellMask = (1ULL << 21) - 1ULL;

double elapsedSeconds(const std::chrono::steady_clock::time_point started) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       started)
      .count();
}

std::uint64_t packSurfelCell(const std::int64_t x, const std::int64_t y,
                             const std::int64_t z) {
  if (x < kSurfelMinimumCell || x > kSurfelMaximumCell ||
      y < kSurfelMinimumCell || y > kSurfelMaximumCell ||
      z < kSurfelMinimumCell || z > kSurfelMaximumCell) {
    throw std::runtime_error("SLAM surfel cell is outside packed-key support");
  }
  std::uint64_t key =
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) &
       kSurfelCellMask)
      << 42;
  key |= (static_cast<std::uint64_t>(static_cast<std::uint32_t>(y)) &
          kSurfelCellMask)
         << 21;
  key |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(z)) &
         kSurfelCellMask;
  return key;
}

const float* data(const std::vector<Eigen::Vector3f>& values) {
  return values.empty() ? nullptr : values.front().data();
}

float* data(std::vector<Eigen::Vector3f>& values) {
  return values.empty() ? nullptr : values.front().data();
}

void requireNative(const int status, const char* operation) {
  if (status != 0) {
    throw std::runtime_error(std::string(operation) +
                             " failed with native status " +
                             std::to_string(status));
  }
}

Eigen::Quaterniond arrayQuaternionProduct(const Eigen::Quaterniond& left,
                                          const Eigen::Quaterniond& right) {
  const double ax = left.x();
  const double ay = left.y();
  const double az = left.z();
  const double aw = left.w();
  const double bx = right.x();
  const double by = right.y();
  const double bz = right.z();
  const double bw = right.w();
  return Eigen::Quaterniond(
      aw * bw - ((ax * bx + ay * by) + az * bz),
      (aw * bx + bw * ax) + (ay * bz - az * by),
      (aw * by + bw * ay) + (az * bx - ax * bz),
      (aw * bz + bw * az) + (ax * by - ay * bx));
}

Eigen::Quaterniond rawQuaternionInverse(const Eigen::Quaterniond& value) {
  const double squared_norm =
      (value.z() * value.z() + value.x() * value.x()) +
      (value.w() * value.w() + value.y() * value.y());
  return Eigen::Quaterniond(value.w() / squared_norm,
                            -value.x() / squared_norm,
                            -value.y() / squared_norm,
                            -value.z() / squared_norm);
}

Eigen::Vector3d rawQuaternionTransformVector(
    const Eigen::Quaterniond& quaternion, const Eigen::Vector3d& vector) {
  const double qx = quaternion.x();
  const double qy = quaternion.y();
  const double qz = quaternion.z();
  const double qw = quaternion.w();
  double tx = qy * vector.z() - qz * vector.y();
  tx += tx;
  double ty = qz * vector.x() - qx * vector.z();
  ty += ty;
  double tz = qx * vector.y() - qy * vector.x();
  tz += tz;
  return Eigen::Vector3d(
      (qy * tz - qz * ty) + (qw * tx + vector.x()),
      (qz * tx - qx * tz) + (qw * ty + vector.y()),
      (qx * ty - qy * tx) + (vector.z() + qw * tz));
}

std::array<double, 7> poseArray(const Pose& pose) {
  return {pose.translation.x(), pose.translation.y(), pose.translation.z(),
          pose.rotation.x(), pose.rotation.y(), pose.rotation.z(),
          pose.rotation.w()};
}

Pose poseFromArray(const double* const values) {
  Pose pose;
  pose.translation = Eigen::Vector3d(values[0], values[1], values[2]);
  pose.rotation = Eigen::Quaterniond(values[6], values[3], values[4], values[5]);
  return pose;
}

std::vector<Eigen::Vector3f> transformMatrix(
    const std::vector<Eigen::Vector3f>& points, const Pose& pose) {
  std::vector<Eigen::Vector3f> output(points.size());
  const auto value = poseArray(pose);
  requireNative(navvis_recon_slam_transform_points_double_matrix_cast(
                    points.size(), data(points), value.data(), value.data() + 3,
                    data(output)),
                "SLAM float-matrix point transform");
  return output;
}

std::vector<Eigen::Vector3f> transformFloatQuaternion(
    const std::vector<Eigen::Vector3f>& points, const Pose& pose) {
  std::vector<Eigen::Vector3f> output(points.size());
  const auto value = poseArray(pose);
  requireNative(navvis_recon_slam_transform_points_raw_float_quaternion(
                    points.size(), data(points), value.data(), value.data() + 3,
                    data(output)),
                "SLAM float-quaternion point transform");
  return output;
}

struct Cell {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  bool operator==(const Cell& other) const noexcept {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct CellHash {
  std::size_t operator()(const Cell& cell) const noexcept {
    std::uint64_t value = static_cast<std::uint64_t>(cell.x);
    value ^= static_cast<std::uint64_t>(cell.y) + 0x9e3779b97f4a7c15ULL +
             (value << 6U) + (value >> 2U);
    value ^= static_cast<std::uint64_t>(cell.z) + 0x9e3779b97f4a7c15ULL +
             (value << 6U) + (value >> 2U);
    return static_cast<std::size_t>(value);
  }
};

std::vector<std::size_t> firstPointIndices(
    const std::vector<Eigen::Vector3f>& points, const float resolution) {
  std::unordered_set<Cell, CellHash> cells;
  cells.reserve(points.size());
  std::vector<std::size_t> indices;
  indices.reserve(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Eigen::Vector3f& point = points[index];
    const Cell cell{static_cast<std::int64_t>(std::floor(point.x() / resolution)),
                    static_cast<std::int64_t>(std::floor(point.y() / resolution)),
                    static_cast<std::int64_t>(std::floor(point.z() / resolution))};
    if (cells.insert(cell).second) {
      indices.push_back(index);
    }
  }
  return indices;
}

std::vector<Eigen::Vector3f> adaptiveFirstPointFilter(
    const std::vector<Eigen::Vector3f>& points, const FrontendConfig& config) {
  if (points.size() <= config.high_resolution_max_points) {
    return points;
  }

  float maximum = config.high_resolution_max_voxel_m;
  float minimum = config.high_resolution_min_voxel_m;
  std::vector<std::size_t> sparse = firstPointIndices(points, maximum);
  std::vector<std::size_t> dense = firstPointIndices(points, minimum);
  std::vector<std::size_t>* selected = nullptr;
  if (sparse.size() > config.high_resolution_max_points) {
    selected = &sparse;
  } else {
    for (int iteration = 0; iteration < 10; ++iteration) {
      const double excess =
          static_cast<double>(dense.size() - config.high_resolution_max_points) /
          static_cast<double>(config.high_resolution_max_points);
      if (excess <= 0.1) {
        break;
      }
      const float middle = 0.5F * (maximum + minimum);
      std::vector<std::size_t> candidate = firstPointIndices(points, middle);
      if (candidate.size() < config.high_resolution_max_points) {
        maximum = middle;
        sparse = std::move(candidate);
      } else {
        minimum = middle;
        dense = std::move(candidate);
      }
    }
    selected = &dense;
  }

  std::vector<Eigen::Vector3f> result;
  result.reserve(std::min(selected->size(), config.high_resolution_max_points));
  if (selected->size() <= config.high_resolution_max_points) {
    for (const std::size_t index : *selected) {
      result.push_back(points[index]);
    }
    return result;
  }
  for (std::size_t index = 0; index < config.high_resolution_max_points;
       ++index) {
    const std::size_t source =
        index * selected->size() / config.high_resolution_max_points + 1U;
    result.push_back(points[selected->at(source)]);
  }
  return result;
}

struct FilteredScan {
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> origins;
};

FilteredScan centroidFilter(const DeskewedRangeBatch& input,
                            const FrontendConfig& config) {
  if (input.points.size() != input.origins.size()) {
    throw std::invalid_argument("SLAM points and ray origins must have equal size");
  }
  // The archive producer already applied the lidar fringe/minimum-range
  // policy. Long endpoints are removed by ImuPosePredictor only after they
  // have participated in the prediction schedule, matching RangeDataCollator.
  if (input.points.size() < 4U) {
    throw std::runtime_error("too few SLAM returns remain after range filtering");
  }

  FilteredScan output;
  output.points.resize(input.points.size());
  output.origins.resize(input.origins.size());
  std::uint64_t output_count = 0;
  requireNative(navvis_recon_slam_range_centroid_filter(
                    input.points.size(), data(input.origins), data(input.points),
                    config.scan_voxel_m, input.points.size(), &output_count,
                    data(output.origins), data(output.points)),
                "SLAM centroid range filter");
  output.points.resize(output_count);
  output.origins.resize(output_count);
  return output;
}

class NativeOctree {
 public:
  NativeOctree() = default;
  explicit NativeOctree(const std::vector<Eigen::Vector3f>& points) {
    reset(points);
  }
  ~NativeOctree() { reset(); }
  NativeOctree(const NativeOctree&) = delete;
  NativeOctree& operator=(const NativeOctree&) = delete;
  NativeOctree(NativeOctree&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  NativeOctree& operator=(NativeOctree&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  void reset() noexcept {
    if (handle_ != nullptr) {
      navvis_recon_slam_octree_destroy(handle_);
      handle_ = nullptr;
    }
  }
  void reset(const std::vector<Eigen::Vector3f>& points) {
    reset();
    if (!points.empty()) {
      handle_ = navvis_recon_slam_octree_create(points.size(), data(points));
      if (handle_ == nullptr) {
        throw std::runtime_error("failed to build SLAM octree");
      }
    }
  }
  [[nodiscard]] void* get() const noexcept { return handle_; }

 private:
  void* handle_ = nullptr;
};

struct SurfelState {
  std::vector<std::int64_t> keys;
  std::vector<float> primary_weights;
  std::vector<std::uint32_t> primary_counts;
  std::vector<float> primary_means;
  std::vector<float> primary_covariances;
  std::vector<float> primary_viewpoints;
  std::vector<float> secondary_weights;
  std::vector<std::uint32_t> secondary_counts;
  std::vector<float> secondary_means;
  std::vector<float> secondary_covariances;
  std::vector<float> secondary_viewpoints;
  std::vector<std::uint8_t> is_split;
  std::vector<float> split_normals;
  std::vector<std::uint8_t> primary_dirty;
  std::vector<std::uint8_t> secondary_dirty;

  [[nodiscard]] std::size_t size() const noexcept { return primary_weights.size(); }

  void resize(const std::size_t size) {
    keys.resize(3U * size);
    primary_weights.resize(size);
    primary_counts.resize(size);
    primary_means.resize(3U * size);
    primary_covariances.resize(9U * size);
    primary_viewpoints.resize(3U * size);
    secondary_weights.resize(size);
    secondary_counts.resize(size);
    secondary_means.resize(3U * size);
    secondary_covariances.resize(9U * size);
    secondary_viewpoints.resize(3U * size);
    is_split.resize(size);
    split_normals.resize(3U * size);
    primary_dirty.resize(size);
    secondary_dirty.resize(size);
  }
};

struct SurfelLevel {
  SurfelLevel() = default;
  SurfelLevel(const double input_resolution, const double input_offset)
      : resolution(input_resolution), offset(input_offset) {}

  double resolution = 0.1;
  double offset = 0.05;
  SurfelState state;
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> normals;
  NativeOctree octree;
  std::unordered_map<std::uint64_t, std::uint64_t> cell_labels;
  std::vector<std::uint64_t> labels;
  std::vector<std::uint64_t> sources;
  std::vector<std::uint32_t> seen_generations;
  std::uint32_t seen_generation = 0U;
  std::vector<float> geometry_weights;
  std::vector<float> geometry_means;
  std::vector<float> geometry_covariances;
  std::vector<float> geometry_viewpoints;
  std::vector<float> geometry_normals;
  std::vector<float> geometry_eigenvalues;

  void update(const std::vector<Eigen::Vector3f>& new_points,
              const std::vector<Eigen::Vector3f>& new_origins,
              const bool maintain) {
    const std::size_t previous_count = state.size();
    labels.resize(new_points.size());
    if (cell_labels.empty() && previous_count != 0U) {
      cell_labels.reserve(previous_count + new_points.size());
      for (std::size_t index = 0; index < previous_count; ++index) {
        cell_labels.emplace(
            packSurfelCell(state.keys[3U * index],
                           state.keys[3U * index + 1U],
                           state.keys[3U * index + 2U]),
            index);
      }
    }
    std::uint64_t output_count = previous_count;
    const double inverse_resolution = 1.0 / resolution;
    for (std::size_t point_index = 0; point_index < new_points.size();
         ++point_index) {
      std::int64_t cell[3]{};
      for (int axis = 0; axis < 3; ++axis) {
        const double coordinate =
            (static_cast<double>(new_points[point_index][axis]) + offset) *
            inverse_resolution;
        const double floored = std::floor(coordinate);
        if (floored < static_cast<double>(kSurfelMinimumCell) ||
            floored > static_cast<double>(kSurfelMaximumCell)) {
          throw std::runtime_error(
              "SLAM surfel point is outside packed-key support");
        }
        cell[axis] = static_cast<std::int64_t>(floored);
      }
      const std::uint64_t key = packSurfelCell(cell[0], cell[1], cell[2]);
      const auto found = cell_labels.find(key);
      if (found != cell_labels.end()) {
        labels[point_index] = found->second;
        continue;
      }
      labels[point_index] = output_count;
      cell_labels.emplace(key, output_count++);
      state.keys.push_back(cell[0]);
      state.keys.push_back(cell[1]);
      state.keys.push_back(cell[2]);
    }
    state.resize(output_count);

    requireNative(navvis_recon_slam_update_split_surfels(
                      state.size(), state.primary_weights.data(),
                      state.primary_counts.data(), state.primary_means.data(),
                      state.primary_covariances.data(),
                      state.primary_viewpoints.data(),
                      state.secondary_weights.data(),
                      state.secondary_counts.data(), state.secondary_means.data(),
                      state.secondary_covariances.data(),
                      state.secondary_viewpoints.data(), state.is_split.data(),
                      state.split_normals.data(), state.primary_dirty.data(),
                      state.secondary_dirty.data(), new_points.size(), labels.data(),
                      data(new_origins), data(new_points), maintain ? 1U : 0U),
                  "SLAM split-surfel update");
    if (!maintain) {
      return;
    }

    sources.clear();
    if (previous_count == 0U) {
      sources.resize(state.size());
      std::iota(sources.begin(), sources.end(), std::uint64_t{0});
    } else {
      seen_generations.resize(state.size(), 0U);
      if (++seen_generation == 0U) {
        std::fill(seen_generations.begin(), seen_generations.end(), 0U);
        seen_generation = 1U;
      }
      for (const std::uint64_t label : labels) {
        if (seen_generations[label] != seen_generation) {
          seen_generations[label] = seen_generation;
          sources.push_back(label);
        }
      }
    }
    merge(sources);
    rebuildCloud();
  }

  void activate() {
    requireNative(navvis_recon_slam_maintain_split_surfels(
                      state.size(), state.primary_weights.data(),
                      state.primary_counts.data(), state.primary_means.data(),
                      state.primary_covariances.data(),
                      state.primary_viewpoints.data(),
                      state.secondary_counts.data(), state.secondary_means.data(),
                      state.secondary_covariances.data(),
                      state.secondary_viewpoints.data(), state.is_split.data(),
                      state.split_normals.data(), state.primary_dirty.data(),
                      state.secondary_dirty.data()),
                  "SLAM deferred surfel maintenance");
    sources.resize(state.size());
    std::iota(sources.begin(), sources.end(), std::uint64_t{0});
    merge(sources);
    rebuildCloud();
  }

 private:
  void merge(const std::vector<std::uint64_t>& sources) {
    std::uint64_t merge_count = 0;
    requireNative(navvis_recon_slam_merge_surfels(
                      state.size(), state.keys.data(),
                      state.primary_weights.data(), state.primary_counts.data(),
                      state.primary_means.data(),
                      state.primary_covariances.data(),
                      state.primary_viewpoints.data(),
                      state.secondary_weights.data(),
                      state.secondary_counts.data(), state.secondary_means.data(),
                      state.secondary_covariances.data(),
                      state.secondary_viewpoints.data(), state.is_split.data(),
                      state.split_normals.data(), state.primary_dirty.data(),
                      state.secondary_dirty.data(), offset, offset, offset,
                      1.0 / resolution, sources.size(), sources.data(),
                      &merge_count),
                  "SLAM cross-cell surfel merge");
  }

  void rebuildCloud() {
    const std::size_t count = 2U * state.size();
    geometry_weights.resize(count);
    geometry_means.resize(3U * count);
    geometry_covariances.resize(9U * count);
    geometry_viewpoints.resize(3U * count);
    for (std::size_t index = 0; index < state.size(); ++index) {
      geometry_weights[2U * index] = state.primary_weights[index];
      geometry_weights[2U * index + 1U] = state.secondary_weights[index];
      for (int axis = 0; axis < 3; ++axis) {
        geometry_means[3U * (2U * index) + axis] =
            state.primary_means[3U * index + axis];
        geometry_means[3U * (2U * index + 1U) + axis] =
            state.secondary_means[3U * index + axis];
        geometry_viewpoints[3U * (2U * index) + axis] =
            state.primary_viewpoints[3U * index + axis];
        geometry_viewpoints[3U * (2U * index + 1U) + axis] =
            state.secondary_viewpoints[3U * index + axis];
      }
      for (int value = 0; value < 9; ++value) {
        geometry_covariances[9U * (2U * index) + value] =
            state.primary_covariances[9U * index + value];
        geometry_covariances[9U * (2U * index + 1U) + value] =
            state.secondary_covariances[9U * index + value];
      }
    }
    geometry_normals.resize(3U * count);
    geometry_eigenvalues.resize(3U * count);
    requireNative(navvis_recon_slam_oriented_surfel_geometry(
                      count, geometry_covariances.data(), geometry_means.data(),
                      geometry_viewpoints.data(), geometry_normals.data(),
                      geometry_eigenvalues.data()),
                  "SLAM surfel PCA");

    points.clear();
    normals.clear();
    points.reserve(count);
    normals.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      const float first = geometry_eigenvalues[3U * index];
      const float second = geometry_eigenvalues[3U * index + 1U];
      const float third = geometry_eigenvalues[3U * index + 2U];
      const float total_float = (second + third) + first;
      const double total = static_cast<double>(total_float);
      const float planarity = static_cast<float>(
          2.0 * static_cast<double>(second - first) / total);
      const float curvature =
          static_cast<float>(3.0 * static_cast<double>(first) / total);
      const float major_spread =
          static_cast<float>(2.0 * std::sqrt(3.0 * static_cast<double>(third)));
      const float minor_spread =
          static_cast<float>(2.0 * std::sqrt(3.0 * static_cast<double>(second)));
      const float normal_spread =
          static_cast<float>(2.0 * std::sqrt(3.0 * static_cast<double>(first)));
      bool valid = geometry_weights[index] >= 7.5F && planarity >= 0.50F &&
                   curvature <= 0.30F &&
                   major_spread >= static_cast<float>(resolution) * 0.0F &&
                   minor_spread >= 0.075F && normal_spread <= 0.05F;
      if ((index & 1U) != 0U) {
        valid = valid && state.is_split[index / 2U] != 0U;
      }
      if (!valid) {
        continue;
      }
      points.emplace_back(geometry_means[3U * index],
                          geometry_means[3U * index + 1U],
                          geometry_means[3U * index + 2U]);
      normals.emplace_back(geometry_normals[3U * index],
                           geometry_normals[3U * index + 1U],
                           geometry_normals[3U * index + 2U]);
    }
    octree.reset(points);
  }
};

struct TransformedRangeData {
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector3f> origins;
};

TransformedRangeData transformToSubmap(
    const Pose& node_pose, const Pose& submap_pose,
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<Eigen::Vector3f>& origins) {
  if (points.size() != origins.size()) {
    throw std::invalid_argument("submap range points/origins size mismatch");
  }
  const std::vector<Eigen::Vector3f> zero_normals(points.size(),
                                                  Eigen::Vector3f::Zero());
  TransformedRangeData output;
  output.points.resize(points.size());
  output.normals.resize(points.size());
  output.origins.resize(points.size());
  const auto node = poseArray(node_pose);
  const auto submap = poseArray(submap_pose);
  std::array<double, 7> relative{};
  requireNative(navvis_recon_slam_transform_submap_range_data(
                    points.size(), data(points), data(zero_normals), data(origins),
                    node.data(), node.data() + 3, submap.data(),
                    submap.data() + 3, data(output.points), data(output.normals),
                    data(output.origins), relative.data()),
                "SLAM submap range transform");
  return output;
}

struct MatchingLevel {
  const std::vector<Eigen::Vector3f>* points = nullptr;
  const std::vector<Eigen::Vector3f>* normals = nullptr;
  const NativeOctree* octree = nullptr;
};

struct Correspondences {
  std::vector<std::uint8_t> selected;
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> normals;
  std::vector<float> distances_squared;
};

Correspondences findCorrespondences(
    const std::vector<Eigen::Vector3f>& queries,
    const std::array<MatchingLevel, 3>& levels,
    const FrontendConfig& config) {
  Correspondences output;
  output.selected.assign(queries.size(), 0U);
  output.points.resize(queries.size());
  output.normals.resize(queries.size());
  output.distances_squared.assign(queries.size(),
                                  std::numeric_limits<float>::infinity());
  std::vector<std::size_t> unresolved(queries.size());
  std::iota(unresolved.begin(), unresolved.end(), std::size_t{0});
  for (std::size_t level_index = 0; level_index < levels.size(); ++level_index) {
    const MatchingLevel& level = levels[level_index];
    if (unresolved.empty() || level.octree == nullptr ||
        level.octree->get() == nullptr) {
      continue;
    }
    std::vector<Eigen::Vector3f> candidates;
    candidates.reserve(unresolved.size());
    for (const std::size_t index : unresolved) {
      candidates.push_back(queries[index]);
    }
    std::vector<std::uint8_t> found(candidates.size());
    std::vector<std::uint64_t> target_indices(candidates.size());
    std::vector<float> squared_distances(candidates.size());
    requireNative(navvis_recon_slam_octree_nearest(
                      level.octree->get(), candidates.size(), data(candidates),
                      config.correspondence_limits_m[level_index],
                      config.icp_threads, found.data(), target_indices.data(),
                      squared_distances.data()),
                  "SLAM ICP nearest-neighbour query");
    std::vector<std::size_t> next_unresolved;
    next_unresolved.reserve(unresolved.size());
    for (std::size_t local_index = 0; local_index < unresolved.size();
         ++local_index) {
      const std::size_t query_index = unresolved[local_index];
      if (found[local_index] == 0U) {
        next_unresolved.push_back(query_index);
        continue;
      }
      const std::size_t target_index = target_indices[local_index];
      output.selected[query_index] = 1U;
      output.points[query_index] = level.points->at(target_index);
      output.normals[query_index] = level.normals->at(target_index);
      output.distances_squared[query_index] = squared_distances[local_index];
    }
    unresolved = std::move(next_unresolved);
  }
  return output;
}

std::size_t applyGeometricFilters(
    Correspondences& matches,
    const std::vector<Eigen::Vector3f>& prepared_points,
    const std::vector<Eigen::Vector3f>& prepared_origins,
    const Pose& filter_target_from_search, const float plane_limit,
    const FrontendConfig& config) {
  const std::vector<Eigen::Vector3f> transformed_targets =
      transformFloatQuaternion(matches.points, filter_target_from_search);
  Pose rotation_only = filter_target_from_search;
  rotation_only.translation.setZero();
  const std::vector<Eigen::Vector3f> transformed_normals =
      transformFloatQuaternion(matches.normals, rotation_only);
  const float incidence_threshold = static_cast<float>(std::cos(
      static_cast<double>(config.maximum_incidence_angle_degrees) * kPi / 180.0));
  std::size_t count = 0;
  for (std::size_t index = 0; index < matches.selected.size(); ++index) {
    if (matches.selected[index] == 0U) {
      continue;
    }
    const Eigen::Vector3f delta = prepared_points[index] - transformed_targets[index];
    const Eigen::Vector3f& normal = transformed_normals[index];
    const float residual =
        (normal.z() * delta.z() + normal.y() * delta.y()) +
        normal.x() * delta.x();
    if (std::abs(residual) >= plane_limit) {
      matches.selected[index] = 0U;
      continue;
    }
    const Eigen::Vector3f ray = prepared_points[index] - prepared_origins[index];
    const float squared_norm =
        (ray.z() * ray.z() + ray.y() * ray.y()) + ray.x() * ray.x();
    const float norm = std::max(std::sqrt(squared_norm), 1.0e-12F);
    const float numerator =
        (normal.z() * ray.z() + normal.y() * ray.y()) + normal.x() * ray.x();
    if (-(numerator / norm) < incidence_threshold) {
      matches.selected[index] = 0U;
      continue;
    }
    ++count;
  }
  return count;
}

IcpResult pointToPlaneIcp(const FilteredScan& scan,
                          const std::array<MatchingLevel, 3>& levels,
                          const Pose& initial,
                          const FrontendConfig& config) {
  IcpResult result;
  result.target_from_source = initial;
  if (scan.points.empty() || levels[0].points == nullptr ||
      levels[0].points->empty()) {
    return result;
  }

  const auto initial_values = poseArray(initial);
  std::array<double, 7> normalization_values{};
  requireNative(navvis_recon_slam_icp_normalization_pose(
                    initial_values.data(), initial_values.data() + 3,
                    normalization_values.data()),
                "SLAM ICP normalization");
  const Pose normalization = poseFromArray(normalization_values.data());
  const std::vector<Eigen::Vector3f> prepared_points =
      transformMatrix(scan.points, initial);
  const std::vector<Eigen::Vector3f> prepared_origins =
      transformMatrix(scan.origins, initial);
  Pose solver_pose = Pose::identity();

  for (int iteration = 1; iteration <= config.icp_iterations; ++iteration) {
    float plane_limit = config.contracted_plane_distance_m;
    if (iteration < config.icp_contraction_iterations) {
      const float current = static_cast<float>(iteration - 1);
      const float denominator =
          static_cast<float>(std::max(1, config.icp_contraction_iterations - 1));
      const float contraction = current / denominator;
      const float difference = config.contracted_plane_distance_m -
                               config.initial_plane_distance_m;
      plane_limit = contraction * difference + config.initial_plane_distance_m;
    }

    const Pose inverse_solver = solver_pose.inverse();
    const std::vector<Eigen::Vector3f> query_points =
        transformMatrix(prepared_points, inverse_solver);
    Correspondences matches = findCorrespondences(query_points, levels, config);
    const Pose filter_target_from_search = inverse_solver.inverse();
    const std::size_t used = applyGeometricFilters(
        matches, prepared_points, prepared_origins, filter_target_from_search,
        plane_limit, config);
    result.iteration_count = iteration;
    result.correspondence_count = used;
    if (used < static_cast<std::size_t>(config.icp_minimum_correspondences)) {
      break;
    }

    std::vector<Eigen::Vector3f> source;
    std::vector<Eigen::Vector3f> target;
    std::vector<Eigen::Vector3f> normals;
    source.reserve(used);
    target.reserve(used);
    normals.reserve(used);
    for (std::size_t index = 0; index < matches.selected.size(); ++index) {
      if (matches.selected[index] != 0U) {
        source.push_back(query_points[index]);
        target.push_back(matches.points[index]);
        normals.push_back(matches.normals[index]);
      }
    }

    std::array<double, 3> increment_translation{};
    std::array<double, 4> increment_quaternion{};
    std::array<double, 6> delta{};
    double scale = 0.0;
    std::array<double, 36> normal_matrix{};
    std::array<double, 6> right_hand_side{};
    const auto normalizer = poseArray(normalization);
    requireNative(navvis_recon_slam_point_plane_step(
                      used, data(source), data(target), data(normals),
                      normalizer.data(), normalizer.data() + 3,
                      increment_translation.data(), increment_quaternion.data(),
                      delta.data(), &scale, normal_matrix.data(),
                      right_hand_side.data()),
                  "SLAM point-to-plane step");
    Pose increment;
    increment.translation = Eigen::Vector3d(increment_translation[0],
                                             increment_translation[1],
                                             increment_translation[2]);
    increment.rotation = Eigen::Quaterniond(
        increment_quaternion[3], increment_quaternion[0],
        increment_quaternion[1], increment_quaternion[2]);
    solver_pose = increment.compose(solver_pose);
    if (iteration >= config.icp_minimum_iterations &&
        increment.translation.norm() <= kIcpTranslationStepM &&
        Eigen::AngleAxisd(increment.rotation.normalized()).angle() <=
            kIcpRotationStepRad) {
      result.converged = true;
      break;
    }
  }

  result.target_from_source = solver_pose.inverse().compose(initial);
  const std::vector<Eigen::Vector3f> final_points =
      transformMatrix(scan.points, result.target_from_source);
  const std::vector<Eigen::Vector3f> final_origins =
      transformMatrix(scan.origins, result.target_from_source);
  Correspondences final_matches = findCorrespondences(final_points, levels, config);
  const std::size_t final_count = applyGeometricFilters(
      final_matches, final_points, final_origins, Pose::identity(),
      config.contracted_plane_distance_m, config);
  result.correspondence_count = final_count;
  if (final_count != 0U) {
    double plane_squared_sum = 0.0;
    double euclidean_squared_sum = 0.0;
    for (std::size_t index = 0; index < final_matches.selected.size(); ++index) {
      if (final_matches.selected[index] == 0U) {
        continue;
      }
      const Eigen::Vector3f delta = final_points[index] - final_matches.points[index];
      const float residual = final_matches.normals[index].dot(delta);
      plane_squared_sum += static_cast<double>(residual) * residual;
      euclidean_squared_sum += final_matches.distances_squared[index];
    }
    result.plane_fitness_m =
        static_cast<float>(std::sqrt(plane_squared_sum / final_count));
    result.euclidean_fitness_m =
        static_cast<float>(std::sqrt(euclidean_squared_sum / final_count));
  }
  result.converged = result.converged &&
                     final_count >=
                         static_cast<std::size_t>(config.icp_minimum_correspondences);
  return result;
}

}  // namespace

Pose Pose::identity() { return {}; }

Pose Pose::inverse() const {
  const auto value = poseArray(*this);
  std::array<double, 7> output{};
  requireNative(navvis_recon_slam_inverse_pose(value.data(), value.data() + 3,
                                               output.data()),
                "SLAM pose inverse");
  return poseFromArray(output.data());
}

Pose Pose::compose(const Pose& other, const bool normalize_rotation) const {
  const auto left = poseArray(*this);
  const auto right = poseArray(other);
  std::array<double, 7> output{};
  const int status = normalize_rotation
                         ? navvis_recon_slam_compose_pose_normalized(
                               left.data(), left.data() + 3, right.data(),
                               right.data() + 3, output.data())
                         : navvis_recon_slam_compose_pose(
                               left.data(), left.data() + 3, right.data(),
                               right.data() + 3, output.data());
  requireNative(status, "SLAM pose composition");
  return poseFromArray(output.data());
}

class ImuPosePredictor::Implementation {
 public:
  Implementation(std::vector<ImuSample> samples, const double maximum_range,
                 const Pose& initial_pose, RawImuTrackerOptions options)
      : predictor(
            std::move(samples),
            RigidPose{initial_pose.translation, initial_pose.rotation},
            std::move(options)),
        maximum_range_squared(maximum_range * maximum_range) {
    if (!std::isfinite(maximum_range) || maximum_range <= 0.0) {
      throw std::invalid_argument("SLAM maximum range must be finite and positive");
    }
  }

  RawConstantVelocityPosePredictor predictor;
  double maximum_range_squared = 3600.0;
  std::vector<Eigen::Quaterniond> relative_rotations;
  std::vector<Eigen::Vector3d> relative_translations;
  std::vector<double> input_points;
  std::vector<double> input_origins;
  std::vector<double> start_quaternions;
  std::vector<double> start_translations;
  std::vector<Eigen::Vector3f> points_at_start;
  std::vector<Eigen::Vector3f> origins_at_start;

  DeskewedRangeBatch deskew(const TimedRangeBatch& batch) {
    const std::size_t count = batch.points.size();
    if (count == 0U || batch.origins.size() != count ||
        batch.point_timestamps_ns.size() != count) {
      throw std::invalid_argument("timed SLAM range batch has inconsistent arrays");
    }
    if (!std::is_sorted(batch.point_timestamps_ns.begin(),
                        batch.point_timestamps_ns.end()) ||
        batch.point_timestamps_ns.back() > batch.timestamp_ns) {
      throw std::invalid_argument("SLAM ray timestamps must be sorted and bounded");
    }
    predictor.reserveRayScratch(count);
    relative_rotations.resize(count);
    relative_translations.resize(count);
    predictor.relativeMotion(batch.point_timestamps_ns.data(), count,
                             batch.timestamp_ns, relative_rotations.data(),
                             relative_translations.data());

    input_points.resize(3U * count);
    input_origins.resize(3U * count);
    start_quaternions.resize(4U * count);
    start_translations.resize(3U * count);
    points_at_start.resize(count);
    origins_at_start.resize(count);

    const Eigen::Quaterniond start_from_end =
        rawQuaternionInverse(relative_rotations.front());
    const Eigen::Vector3d end_from_start_translation =
        relative_translations.front();
    for (std::size_t index = 0; index < count; ++index) {
      for (int axis = 0; axis < 3; ++axis) {
        input_points[3U * index + axis] = batch.points[index][axis];
        input_origins[3U * index + axis] = batch.origins[index][axis];
      }
      const Eigen::Quaterniond start_from_ray =
          arrayQuaternionProduct(start_from_end, relative_rotations[index]);
      start_quaternions[4U * index] = start_from_ray.x();
      start_quaternions[4U * index + 1U] = start_from_ray.y();
      start_quaternions[4U * index + 2U] = start_from_ray.z();
      start_quaternions[4U * index + 3U] = start_from_ray.w();
      const Eigen::Vector3d start_from_ray_translation =
          rawQuaternionTransformVector(
              start_from_end,
              relative_translations[index] - end_from_start_translation);
      for (int axis = 0; axis < 3; ++axis) {
        start_translations[3U * index + axis] =
            start_from_ray_translation[axis];
      }
    }
    requireNative(navvis_recon_slam_deskew_points(
                      count, input_points.data(), start_quaternions.data(),
                      start_translations.data(), data(points_at_start)),
                  "SLAM ray-to-batch-start deskew");
    requireNative(navvis_recon_slam_deskew_points(
                      count, input_origins.data(), start_quaternions.data(),
                      start_translations.data(), data(origins_at_start)),
                  "SLAM origin-to-batch-start deskew");

    const RigidPose& end_from_start = predictor.lastEndFromStartPose();
    const Pose end_pose{end_from_start.translation, end_from_start.rotation};
    const std::vector<Eigen::Vector3f> end_points =
        transformMatrix(points_at_start, end_pose);
    const std::vector<Eigen::Vector3f> end_origins =
        transformMatrix(origins_at_start, end_pose);

    DeskewedRangeBatch output;
    output.timestamp_ns = batch.timestamp_ns;
    output.points.reserve(count);
    output.origins.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      const Eigen::Vector3d raw_delta =
          batch.points[index].cast<double>() - batch.origins[index].cast<double>();
      if (raw_delta.squaredNorm() <= maximum_range_squared) {
        output.points.push_back(end_points[index]);
        output.origins.push_back(end_origins[index]);
      }
    }
    return output;
  }
};

ImuPosePredictor::ImuPosePredictor(std::vector<ImuSample> samples,
                                   const double maximum_range_m,
                                   Pose initial_pose,
                                   RawImuTrackerOptions options)
    : implementation_(std::make_unique<Implementation>(
          std::move(samples), maximum_range_m, initial_pose,
          std::move(options))) {}

ImuPosePredictor::~ImuPosePredictor() = default;
ImuPosePredictor::ImuPosePredictor(ImuPosePredictor&&) noexcept = default;
ImuPosePredictor& ImuPosePredictor::operator=(ImuPosePredictor&&) noexcept =
    default;

Pose ImuPosePredictor::predict(const std::int64_t timestamp_ns) {
  const RigidPose value = implementation_->predictor.predict(timestamp_ns);
  return Pose{value.translation, value.rotation};
}

DeskewedRangeBatch ImuPosePredictor::deskew(const TimedRangeBatch& batch) {
  return implementation_->deskew(batch);
}

void ImuPosePredictor::correct(const std::int64_t timestamp_ns,
                               const Pose& pose) {
  implementation_->predictor.correct(
      timestamp_ns, RigidPose{pose.translation, pose.rotation});
}

Eigen::Vector3d ImuPosePredictor::gravityObservation() const {
  return implementation_->predictor.tracker().gravityObservation();
}

class Frontend::Implementation {
 public:
  explicit Implementation(FrontendConfig input_config)
      : config(std::move(input_config)) {}

  struct Submap {
    std::size_t index = 0;
    std::int64_t start_timestamp_ns = 0;
    std::int64_t end_timestamp_ns = 0;
    Pose local_pose;
    std::vector<std::size_t> node_indices;
    std::array<SurfelLevel, 3> levels{
        SurfelLevel{0.10, 0.05}, SurfelLevel{0.30, 0.15},
        SurfelLevel{0.60, 0.0}};
    void* probability_grid = nullptr;
    bool deferred = false;
    bool finished = false;

    Submap() {
      probability_grid = navvis_recon_slam_probability_grid_create(0.20F);
      if (probability_grid == nullptr) {
        throw std::runtime_error("failed to allocate SLAM probability grid");
      }
    }
    ~Submap() {
      if (probability_grid != nullptr) {
        navvis_recon_slam_probability_grid_destroy(probability_grid);
      }
    }
    Submap(const Submap&) = delete;
    Submap& operator=(const Submap&) = delete;
    Submap(Submap&& other) noexcept
        : index(other.index),
          start_timestamp_ns(other.start_timestamp_ns),
          end_timestamp_ns(other.end_timestamp_ns),
          local_pose(std::move(other.local_pose)),
          node_indices(std::move(other.node_indices)),
          levels(std::move(other.levels)),
          probability_grid(std::exchange(other.probability_grid, nullptr)),
          deferred(other.deferred),
          finished(other.finished) {}
    Submap& operator=(Submap&&) = delete;

    void insert(const FrontendNode& node,
                const DeskewedRangeBatch& raw,
                const FilteredScan& filtered,
                const bool maintain,
                FrontendResult::Timing* const timing) {
      auto phase_started = std::chrono::steady_clock::now();
      const TransformedRangeData map = transformToSubmap(
          node.local_pose, local_pose, raw.points, raw.origins);
      timing->submap_transform_seconds += elapsedSeconds(phase_started);
      for (std::size_t index = 0; index < levels.size(); ++index) {
        phase_started = std::chrono::steady_clock::now();
        levels[index].update(map.points, map.origins, maintain);
        timing->surfel_update_seconds[index] += elapsedSeconds(phase_started);
      }
      phase_started = std::chrono::steady_clock::now();
      const TransformedRangeData grid = transformToSubmap(
          node.local_pose, local_pose, filtered.points, filtered.origins);
      requireNative(navvis_recon_slam_probability_grid_insert(
                        probability_grid, grid.points.size(), data(grid.points),
                        data(grid.origins)),
                    "SLAM probability-grid insertion");
      timing->probability_grid_seconds += elapsedSeconds(phase_started);
      node_indices.push_back(node.index);
      end_timestamp_ns = node.timestamp_ns;
      deferred = deferred || !maintain;
    }

    void activate() {
      if (!deferred) {
        return;
      }
      for (SurfelLevel& level : levels) {
        level.activate();
      }
      deferred = false;
    }

    [[nodiscard]] SubmapSummary snapshot() {
      SubmapSummary value;
      value.index = index;
      value.start_timestamp_ns = start_timestamp_ns;
      value.end_timestamp_ns = end_timestamp_ns;
      value.local_pose = local_pose;
      value.node_indices = node_indices;
      for (std::size_t level = 0; level < levels.size(); ++level) {
        value.surfel_counts[level] = levels[level].points.size();
        value.surfel_points[level] = std::move(levels[level].points);
        value.surfel_normals[level] = std::move(levels[level].normals);
      }
      value.probability_grid_cells =
          navvis_recon_slam_probability_grid_size(probability_grid);
      value.probability_grid_indices.resize(value.probability_grid_cells);
      value.probability_grid_values.resize(value.probability_grid_cells);
      std::uint64_t exported = 0U;
      static_assert(sizeof(std::array<std::int32_t, 3>) ==
                    3U * sizeof(std::int32_t));
      requireNative(navvis_recon_slam_probability_grid_export(
                        probability_grid, value.probability_grid_cells,
                        &exported,
                        value.probability_grid_indices.empty()
                            ? nullptr
                            : value.probability_grid_indices.front().data(),
                        value.probability_grid_values.empty()
                            ? nullptr
                            : value.probability_grid_values.data()),
                    "SLAM probability-grid export");
      if (exported != value.probability_grid_cells) {
        throw std::runtime_error("SLAM probability-grid export count mismatch");
      }
      value.finished = finished;
      return value;
    }
  };

  FrontendConfig config;
  FrontendResult result;
  std::vector<std::unique_ptr<Submap>> submaps;
  std::vector<Submap*> active;
  std::optional<Pose> last_retained_pose;
  std::int64_t last_retained_timestamp_ns = 0;

  Submap& startSubmap(const FrontendNode& node) {
    auto submap = std::make_unique<Submap>();
    submap->index = submaps.size();
    submap->start_timestamp_ns = node.timestamp_ns;
    // The first submap defines the trajectory-local gravity frame and keeps
    // an exact identity rotation.  Later submaps preserve the binary's full
    // gravity-alignment quaternion chain; replacing either case with the node
    // rotation changes the coordinates in which incremental ICP is solved.
    Eigen::Quaterniond submap_rotation = Eigen::Quaterniond::Identity();
    if (!submaps.empty()) {
      const double node_quaternion_xyzw[4]{
          node.local_pose.rotation.x(), node.local_pose.rotation.y(),
          node.local_pose.rotation.z(), node.local_pose.rotation.w()};
      const double gravity[3]{node.gravity_observation.x(),
                              node.gravity_observation.y(),
                              node.gravity_observation.z()};
      double output_quaternion_xyzw[4]{};
      requireNative(navvis_recon_slam_submap_rotation(
                        node_quaternion_xyzw, gravity,
                        output_quaternion_xyzw),
                    "SLAM submap gravity alignment");
      submap_rotation = Eigen::Quaterniond(
          output_quaternion_xyzw[3], output_quaternion_xyzw[0],
          output_quaternion_xyzw[1], output_quaternion_xyzw[2]);
    }
    submap->local_pose =
        Pose{node.local_pose.translation, submap_rotation};
    Submap& reference = *submap;
    submaps.push_back(std::move(submap));
    active.push_back(&reference);
    return reference;
  }

  bool retain(const std::int64_t timestamp_ns, const Pose& pose) {
    if (!last_retained_pose.has_value()) {
      return true;
    }
    const double elapsed =
        static_cast<double>(timestamp_ns - last_retained_timestamp_ns) * 1.0e-9;
    const Pose delta = last_retained_pose->inverse().compose(pose);
    const double distance = delta.translation.norm();
    const double angle = Eigen::AngleAxisd(delta.rotation.normalized()).angle();
    return elapsed >= config.motion_filter_maximum_time_s ||
           distance >= config.motion_filter_maximum_distance_m ||
           angle >= config.motion_filter_maximum_angle_rad;
  }

  void addNode(FrontendNode node, const DeskewedRangeBatch& raw,
               const FilteredScan& filtered) {
    if (active.empty()) {
      startSubmap(node);
    }
    for (std::size_t index = 0; index < active.size(); ++index) {
      active[index]->insert(node, raw, filtered, index == 0U,
                            &result.timing);
    }
    Submap* newest = active.back();
    const double displacement =
        (node.local_pose.translation - newest->local_pose.translation).norm();
    if (displacement >= config.submap_overlap_displacement_m) {
      if (active.size() >= 2U) {
        active.front()->finished = true;
        active.erase(active.begin());
        active.front()->activate();
      }
      startSubmap(node);
    }
    result.nodes.push_back(std::move(node));
  }

  void process(const TimedRangeBatch& input, PosePredictor& predictor) {
    ++result.processed_batches;
    auto phase_started = std::chrono::steady_clock::now();
    DeskewedRangeBatch raw = predictor.deskew(input);
    result.timing.deskew_seconds += elapsedSeconds(phase_started);
    if (raw.points.size() != raw.origins.size() || raw.points.empty()) {
      throw std::runtime_error("invalid deskewed SLAM range batch");
    }
    phase_started = std::chrono::steady_clock::now();
    const FilteredScan filtered = centroidFilter(raw, config);
    result.timing.centroid_filter_seconds += elapsedSeconds(phase_started);
    phase_started = std::chrono::steady_clock::now();
    Pose predicted = predictor.predict(input.timestamp_ns);
    if (result.processed_batches == 1U &&
        config.initial_tracking_pose.has_value()) {
      predictor.correct(input.timestamp_ns, *config.initial_tracking_pose);
      predicted = predictor.predict(input.timestamp_ns);
    }
    IcpResult match;
    match.target_from_source = predicted;
    Pose pose = predicted;
    if (!active.empty()) {
      active.front()->activate();
      std::array<MatchingLevel, 3> levels{};
      for (std::size_t index = 0; index < levels.size(); ++index) {
        levels[index] = MatchingLevel{&active.front()->levels[index].points,
                                      &active.front()->levels[index].normals,
                                      &active.front()->levels[index].octree};
      }
      const Pose matching_from_local = active.front()->local_pose.inverse();
      const Pose matching_initial =
          matching_from_local.compose(predicted, true);
      match = pointToPlaneIcp(filtered, levels, matching_initial, config);
      if (match.correspondence_count >=
          static_cast<std::size_t>(config.icp_minimum_correspondences)) {
        pose = active.front()->local_pose.compose(match.target_from_source, true);
      }
    }
    result.timing.prediction_and_icp_seconds += elapsedSeconds(phase_started);
    phase_started = std::chrono::steady_clock::now();
    predictor.correct(input.timestamp_ns, pose);
    if (!retain(input.timestamp_ns, pose)) {
      result.timing.correction_and_motion_filter_seconds +=
          elapsedSeconds(phase_started);
      ++result.motion_filtered_batches;
      return;
    }
    result.timing.correction_and_motion_filter_seconds +=
        elapsedSeconds(phase_started);

    phase_started = std::chrono::steady_clock::now();
    FrontendNode node;
    node.index = result.nodes.size();
    node.timestamp_ns = input.timestamp_ns;
    node.local_pose = pose;
    node.points = adaptiveFirstPointFilter(filtered.points, config);
    result.timing.node_filter_seconds += elapsedSeconds(phase_started);
    node.scan_match = match;
    node.gravity_observation = predictor.gravityObservation();
    phase_started = std::chrono::steady_clock::now();
    addNode(std::move(node), raw, filtered);
    result.timing.submap_insertion_seconds += elapsedSeconds(phase_started);
    last_retained_pose = pose;
    last_retained_timestamp_ns = input.timestamp_ns;
  }

  FrontendResult finish() {
    const auto finish_started = std::chrono::steady_clock::now();
    if (active.size() >= 2U) {
      // The unfinished trailing overlap map is intentionally omitted from
      // serialized topology, matching Surveyor's FinishTrajectory behavior.
      Submap* discarded = active.back();
      active.pop_back();
      const auto found = std::find_if(
          submaps.begin(), submaps.end(),
          [discarded](const std::unique_ptr<Submap>& value) {
            return value.get() == discarded;
          });
      if (found != submaps.end()) {
        submaps.erase(found);
      }
    }
    if (!active.empty()) {
      active.back()->end_timestamp_ns = std::numeric_limits<std::int64_t>::max();
    }
    for (Submap* submap : active) {
      submap->finished = true;
      submap->activate();
    }
    active.clear();
    result.submaps.clear();
    result.submaps.reserve(submaps.size());
    for (const std::unique_ptr<Submap>& submap : submaps) {
      result.submaps.push_back(submap->snapshot());
    }
    result.timing.finish_seconds += elapsedSeconds(finish_started);
    return std::move(result);
  }
};

Frontend::Frontend(FrontendConfig config)
    : implementation_(std::make_unique<Implementation>(std::move(config))) {}

Frontend::~Frontend() = default;
Frontend::Frontend(Frontend&&) noexcept = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;

void Frontend::process(const TimedRangeBatch& batch, PosePredictor& predictor) {
  implementation_->process(batch, predictor);
}

FrontendResult Frontend::finish() { return implementation_->finish(); }

}  // namespace navvis_recon::slam
