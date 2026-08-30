#include "navvis_recon/binary_surface_pipeline.hpp"

#include "navvis_recon/cloud_surface_filter.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace navvis_recon {
namespace {

struct Neighbor {
    std::uint32_t index = 0U;
    float squared_distance = 0.0F;
};

// CompactOctree radius-query ordering used by the installed surface binary.
// Its root is a power-of-two cube centered on the float32 input bounds.  AABB
// queries visit occupied leaves in ascending octant order (x is the high
// child bit, then y, then z), while records inside a leaf retain insertion
// order.  Reproducing that order is material because the normal accumulator
// is an online float32 update.
class CompactRadiusIndex {
  public:
    CompactRadiusIndex(const std::vector<Vec3f>& points, double resolution)
        : points_(points), resolution_(resolution) {
        if (points.empty()) {
            return;
        }
        Vec3f minimum = points.front();
        Vec3f maximum = minimum;
        for (std::size_t index = 1U; index < points.size(); ++index) {
            minimum = minimum.cwiseMin(points[index]);
            maximum = maximum.cwiseMax(points[index]);
        }
        const Eigen::Vector3d extent =
            maximum.cast<double>() - minimum.cast<double>();
        const double required_side = extent.maxCoeff();
        side_ = resolution_;
        while (side_ < required_side) {
            side_ *= 2.0;
            ++depth_;
        }
        lower_ = (minimum.cast<double>() + maximum.cast<double>()) * 0.5 -
                 Eigen::Vector3d::Constant(side_ * 0.5);
        cells_.reserve(points.size());
        for (std::uint32_t index = 0U; index < points.size(); ++index) {
            cells_[key(points[index])].push_back(index);
        }
    }

    void radius(const Vec3f& center, float radius, std::vector<Neighbor>& result) const {
        result.clear();
        if (points_.empty()) {
            return;
        }
        const Eigen::Vector3d center_f64 = center.cast<double>();
        const Eigen::Vector3d half =
            Eigen::Vector3d::Constant(static_cast<double>(radius));
        const VoxelKey minimum = keyF64(Eigen::Vector3d(center_f64 - half));
        const VoxelKey maximum = keyF64(Eigen::Vector3d(center_f64 + half));
        thread_local std::vector<OccupiedCell> occupied;
        occupied.clear();
        for (int x = minimum.x; x <= maximum.x; ++x) {
            for (int y = minimum.y; y <= maximum.y; ++y) {
                for (int z = minimum.z; z <= maximum.z; ++z) {
                    const VoxelKey voxel{x, y, z};
                    const auto found = cells_.find(voxel);
                    if (found != cells_.end()) {
                        occupied.push_back({voxel, &found->second});
                    }
                }
            }
        }
        std::sort(occupied.begin(), occupied.end(),
                  [&](const OccupiedCell& first, const OccupiedCell& second) {
                      return mortonLess(first.key, second.key);
                  });
        for (const OccupiedCell& cell : occupied) {
            for (const std::uint32_t index : *cell.indices) {
                const Vec3f delta = center - points_[index];
                const float squared_distance =
                    (delta.z() * delta.z() + delta.y() * delta.y()) +
                    delta.x() * delta.x();
                result.push_back({index, squared_distance});
            }
        }
    }

  private:
    struct OccupiedCell {
        VoxelKey key{};
        const std::vector<std::uint32_t>* indices = nullptr;
    };

    [[nodiscard]] VoxelKey key(const Vec3f& point) const {
        return keyF64(Eigen::Vector3d(point.cast<double>()));
    }

    [[nodiscard]] VoxelKey keyF64(const Eigen::Vector3d& point) const {
        const std::uint64_t cell_count = std::uint64_t{1} << depth_;
        VoxelKey result;
        for (int axis = 0; axis < 3; ++axis) {
            const double local = (point[axis] - lower_[axis]) / resolution_;
            const auto raw = static_cast<std::int64_t>(std::floor(local));
            resultAxis(result, axis) = static_cast<int>(std::clamp<std::int64_t>(
                raw, 0, static_cast<std::int64_t>(cell_count - 1U)));
        }
        return result;
    }

    static int& resultAxis(VoxelKey& key, int axis) {
        if (axis == 0) return key.x;
        if (axis == 1) return key.y;
        return key.z;
    }

    [[nodiscard]] bool mortonLess(const VoxelKey& first, const VoxelKey& second) const {
        for (int bit = depth_ - 1; bit >= 0; --bit) {
            const int first_child = (((first.x >> bit) & 1) << 2) |
                                    (((first.y >> bit) & 1) << 1) |
                                    ((first.z >> bit) & 1);
            const int second_child = (((second.x >> bit) & 1) << 2) |
                                     (((second.y >> bit) & 1) << 1) |
                                     ((second.z >> bit) & 1);
            if (first_child != second_child) {
                return first_child < second_child;
            }
        }
        return false;
    }

    const std::vector<Vec3f>& points_;
    double resolution_ = 1.0;
    double side_ = 0.0;
    int depth_ = 0;
    Eigen::Vector3d lower_ = Eigen::Vector3d::Zero();
    std::unordered_map<VoxelKey, std::vector<std::uint32_t>, VoxelKeyHash> cells_;
};

class SpatialIndex {
  public:
    SpatialIndex(const std::vector<Vec3f>& points, float cell) : points_(points), cell_(cell) {
        cells_.reserve(points.size());
        for (std::uint32_t index = 0U; index < points.size(); ++index) {
            cells_[key(points[index])].push_back(index);
        }
    }

    void radius(const Vec3f& center, float radius, std::vector<Neighbor>& result) const {
        result.clear();
        const VoxelKey middle = key(center);
        const int extent = static_cast<int>(std::ceil(radius / cell_));
        const float squared_radius = radius * radius;
        for (int x = middle.x - extent; x <= middle.x + extent; ++x) {
            for (int y = middle.y - extent; y <= middle.y + extent; ++y) {
                for (int z = middle.z - extent; z <= middle.z + extent; ++z) {
                    const auto found = cells_.find(VoxelKey{x, y, z});
                    if (found == cells_.end()) {
                        continue;
                    }
                    for (const std::uint32_t index : found->second) {
                        const Vec3f delta = points_[index] - center;
                        float distance = delta.x() * delta.x();
                        distance += delta.y() * delta.y();
                        distance += delta.z() * delta.z();
                        if (distance <= squared_radius) {
                            result.push_back({index, distance});
                        }
                    }
                }
            }
        }
    }

    void nearest(const Vec3f& center, std::size_t count, std::vector<Neighbor>& result,
                 float required_maximum_radius = 0.0F) const {
        float radius = cell_;
        const int maximum_attempts = required_maximum_radius > 0.0F ? 16 : 7;
        for (int attempt = 0; attempt < maximum_attempts; ++attempt) {
            this->radius(center, radius, result);
            if (result.size() >= count ||
                (required_maximum_radius > 0.0F && radius >= required_maximum_radius) ||
                (required_maximum_radius <= 0.0F &&
                 (radius >= 3.2F || attempt + 1 >= maximum_attempts))) {
                break;
            }
            radius *= 2.0F;
        }
        const auto order = [](const Neighbor& first, const Neighbor& second) {
            if (first.squared_distance != second.squared_distance) {
                return first.squared_distance < second.squared_distance;
            }
            return first.index < second.index;
        };
        if (result.size() > count) {
            std::nth_element(result.begin(), result.begin() + count, result.end(), order);
            result.resize(count);
        }
        std::sort(result.begin(), result.end(), order);
    }

    [[nodiscard]] bool anyWithinRadius(const Vec3f& center, float radius) const {
        const VoxelKey middle = key(center);
        const int extent = static_cast<int>(std::ceil(radius / cell_));
        const float squared_radius = radius * radius;
        for (int x = middle.x - extent; x <= middle.x + extent; ++x) {
            for (int y = middle.y - extent; y <= middle.y + extent; ++y) {
                for (int z = middle.z - extent; z <= middle.z + extent; ++z) {
                    const auto found = cells_.find(VoxelKey{x, y, z});
                    if (found == cells_.end()) {
                        continue;
                    }
                    for (const std::uint32_t index : found->second) {
                        if ((points_[index] - center).squaredNorm() <= squared_radius) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

  private:
    [[nodiscard]] VoxelKey key(const Vec3f& point) const {
        return {static_cast<int>(std::floor(point.x() / cell_)),
                static_cast<int>(std::floor(point.y() / cell_)),
                static_cast<int>(std::floor(point.z() / cell_))};
    }

    const std::vector<Vec3f>& points_;
    float cell_;
    std::unordered_map<VoxelKey, std::vector<std::uint32_t>, VoxelKeyHash> cells_;
};

// The installed binary uses compact trees for radius searches and
// PCL FLANN/NanoFLANN for exact small-k searches.  A hash grid gives the same
// neighbours, but large radii enumerate many empty cells and nearly parallel
// normals create dense buckets.  Both cases make it scan far more candidates
// than a balanced tree.
//
// Keep the tree and its query scratch contiguous.  Results are maintained in
// exact (float32 squared distance, original point index) order so replacing the
// acceleration structure cannot change the two-pass SOR decisions.
class ExactKdTree3f {
  public:
    explicit ExactKdTree3f(const std::vector<Vec3f>& points) : points_(points) {
        std::vector<std::uint32_t> indices(points.size());
        std::iota(indices.begin(), indices.end(), 0U);
        nodes_.reserve(points.size());
        root_ = build(indices, 0U, indices.size());
    }

    void nearest(const Vec3f& center, std::size_t count, Neighbor* result,
                 std::size_t& result_size) const {
        result_size = 0U;
        if (count == 0U || root_ < 0) {
            return;
        }
        search(root_, center, count, result, result_size);
    }

    void radius(const Vec3f& center, float radius, std::vector<Neighbor>& result) const {
        result.clear();
        if (root_ < 0) {
            return;
        }
        radiusSearch(root_, center, radius * radius, result);
    }

  private:
    struct Node {
        std::uint32_t point_index = 0U;
        std::int32_t left = -1;
        std::int32_t right = -1;
        std::uint8_t axis = 0U;
    };

    static bool neighborLess(const Neighbor& first, const Neighbor& second) {
        if (first.squared_distance != second.squared_distance) {
            return first.squared_distance < second.squared_distance;
        }
        return first.index < second.index;
    }

    std::int32_t build(std::vector<std::uint32_t>& indices, std::size_t begin, std::size_t end) {
        if (begin == end) {
            return -1;
        }

        Vec3f minimum = points_[indices[begin]];
        Vec3f maximum = minimum;
        for (std::size_t position = begin + 1U; position < end; ++position) {
            minimum = minimum.cwiseMin(points_[indices[position]]);
            maximum = maximum.cwiseMax(points_[indices[position]]);
        }
        const Vec3f extent = maximum - minimum;
        int axis = 0;
        if (extent.y() > extent.x()) {
            axis = 1;
        }
        if (extent.z() > extent[axis]) {
            axis = 2;
        }

        const std::size_t middle = begin + (end - begin) / 2U;
        std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(begin),
                         indices.begin() + static_cast<std::ptrdiff_t>(middle),
                         indices.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](std::uint32_t first, std::uint32_t second) {
                             const float first_coordinate = points_[first][axis];
                             const float second_coordinate = points_[second][axis];
                             if (first_coordinate != second_coordinate) {
                                 return first_coordinate < second_coordinate;
                             }
                             return first < second;
                         });

        const std::int32_t node_index = static_cast<std::int32_t>(nodes_.size());
        nodes_.push_back({indices[middle], -1, -1, static_cast<std::uint8_t>(axis)});
        const std::int32_t left = build(indices, begin, middle);
        const std::int32_t right = build(indices, middle + 1U, end);
        nodes_[static_cast<std::size_t>(node_index)].left = left;
        nodes_[static_cast<std::size_t>(node_index)].right = right;
        return node_index;
    }

    static void insertCandidate(const Neighbor& candidate, std::size_t count, Neighbor* result,
                                std::size_t& result_size) {
        std::size_t position = result_size;
        if (result_size < count) {
            ++result_size;
        } else {
            if (!neighborLess(candidate, result[count - 1U])) {
                return;
            }
            position = count - 1U;
        }
        while (position > 0U && neighborLess(candidate, result[position - 1U])) {
            result[position] = result[position - 1U];
            --position;
        }
        result[position] = candidate;
    }

    void search(std::int32_t node_index, const Vec3f& center, std::size_t count, Neighbor* result,
                std::size_t& result_size) const {
        if (node_index < 0) {
            return;
        }
        const Node& node = nodes_[static_cast<std::size_t>(node_index)];
        const Vec3f& point = points_[node.point_index];
        insertCandidate({node.point_index, (point - center).squaredNorm()}, count, result,
                        result_size);

        const int axis = static_cast<int>(node.axis);
        const float delta = center[axis] - point[axis];
        const std::int32_t near_child = delta < 0.0F ? node.left : node.right;
        const std::int32_t far_child = delta < 0.0F ? node.right : node.left;
        search(near_child, center, count, result, result_size);

        const float axis_distance = delta * delta;
        if (result_size < count || axis_distance <= result[count - 1U].squared_distance) {
            search(far_child, center, count, result, result_size);
        }
    }

    void radiusSearch(std::int32_t node_index, const Vec3f& center, float squared_radius,
                      std::vector<Neighbor>& result) const {
        if (node_index < 0) {
            return;
        }
        const Node& node = nodes_[static_cast<std::size_t>(node_index)];
        const Vec3f& point = points_[node.point_index];
        const float squared_distance = (point - center).squaredNorm();
        if (squared_distance <= squared_radius) {
            result.push_back({node.point_index, squared_distance});
        }

        const int axis = static_cast<int>(node.axis);
        const float delta = center[axis] - point[axis];
        const std::int32_t near_child = delta < 0.0F ? node.left : node.right;
        const std::int32_t far_child = delta < 0.0F ? node.right : node.left;
        radiusSearch(near_child, center, squared_radius, result);
        if (delta * delta <= squared_radius) {
            radiusSearch(far_child, center, squared_radius, result);
        }
    }

    const std::vector<Vec3f>& points_;
    std::vector<Node> nodes_;
    std::int32_t root_ = -1;
};

std::vector<float> normalRadii(const BinarySurfaceOptions& options) {
    std::vector<float> radii;
    radii.reserve(static_cast<std::size_t>(options.normal_radius_levels));
    for (int level = 0; level < options.normal_radius_levels; ++level) {
        const float alpha =
            options.normal_radius_levels > 1
                ? static_cast<float>(level) / static_cast<float>(options.normal_radius_levels - 1)
                : 0.0F;
        // 0x16b0c0..0x16b103 evaluates this in the written order.  The
        // algebraically equivalent lerp form changes intermediate float32
        // rounding for some levels.
        radii.push_back(options.normal_radius_minimum +
                        (options.normal_radius_maximum - options.normal_radius_minimum) * alpha);
    }
    return radii;
}

struct NormalPcaResult {
    Vec3f centroid = Vec3f::Zero();
    Vec3f mean_origin = Vec3f::Zero();
    Vec3f normal = Vec3f::Zero();
    Eigen::Vector3f eigenvalues = Eigen::Vector3f::Zero();
    std::size_t count = 0U;
    bool valid = false;
};

NormalPcaResult weightedNormalPca(const std::vector<BinarySurfaceInput>& input,
                                  const std::vector<Neighbor>& neighbors) {
    NormalPcaResult result;
    result.count = neighbors.size();
    if (neighbors.empty()) {
        return result;
    }

    // The binary's accumulator at 0x168970 is a single-precision online
    // weighted covariance update.  A batch/double covariance is numerically
    // close but perturbs normals near the cross-scale quality gates.
    float weight_sum = 0.0F;
    Vec3f centroid = Vec3f::Zero();
    Vec3f mean_origin = Vec3f::Zero();
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> covariance =
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Zero();
    for (const auto& neighbor : neighbors) {
        const BinarySurfaceInput& point = input[neighbor.index];
        const float weight = point.weight;
        const float old_weight = weight_sum;
        const float new_weight = old_weight + weight;
        const float inverse_weight = 1.0F / new_weight;
        const float alpha = weight * inverse_weight;
        const Vec3f old_centroid = centroid;
        Vec3f new_centroid;
        for (int axis = 0; axis < 3; ++axis) {
            const float delta = point.xyz[axis] - old_centroid[axis];
            new_centroid[axis] = old_centroid[axis] + alpha * delta;
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                const float right = weight * (point.xyz[column] - old_centroid[column]);
                const float update = (point.xyz[row] - new_centroid[row]) * right;
                covariance(row, column) =
                    (covariance(row, column) * old_weight + update) * inverse_weight;
            }
        }
        for (int axis = 0; axis < 3; ++axis) {
            mean_origin[axis] =
                (mean_origin[axis] * old_weight + point.origin[axis] * weight) * inverse_weight;
        }
        centroid = new_centroid;
        weight_sum = new_weight;
    }
    if (weight_sum == 0.0F || !std::isfinite(weight_sum)) {
        return result;
    }
    result.centroid = centroid;
    result.mean_origin = mean_origin;
    if (neighbors.size() <= 2U) {
        return result;
    }

    // 0x11c0a0 reads the accumulator's row-major upper triangle at
    // +0x20/+0x24/+0x30.  Eigen's public solver consumes the lower triangle,
    // so mirror those exact (independently rounded) entries explicitly.
    Eigen::Matrix3f covariance_matrix;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            covariance_matrix(row, column) =
                row >= column ? covariance(column, row) : covariance(row, column);
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance_matrix);
    if (solver.info() != Eigen::Success) {
        return result;
    }
    result.eigenvalues = solver.eigenvalues();
    Vec3f normal = solver.eigenvectors().col(0);
    const Vec3f view = result.centroid - result.mean_origin;
    const float view_dot =
        (normal.z() * view.z() + normal.y() * view.y()) + normal.x() * view.x();
    if (view_dot > 0.0F) {
        normal = -normal;
    }
    result.normal = normal;
    result.valid = normal.allFinite() && result.eigenvalues.allFinite();
    return result;
}

bool normalQuality(const NormalPcaResult& result, const BinarySurfaceOptions& options) {
    if (!result.valid || result.count < options.normal_minimum_support ||
        result.eigenvalues.x() < 0.0F || result.eigenvalues.y() < 0.0F) {
        return false;
    }
    const float in_plane_spread =
        static_cast<float>(2.0 * std::sqrt(3.0 * static_cast<double>(result.eigenvalues.y())));
    const float normal_spread =
        static_cast<float>(2.0 * std::sqrt(3.0 * static_cast<double>(result.eigenvalues.x())));
    const float eigenvalue_difference = result.eigenvalues.y() - result.eigenvalues.x();
    const float eigenvalue_sum =
        (result.eigenvalues.y() + result.eigenvalues.z()) + result.eigenvalues.x();
    const float planarity =
        eigenvalue_sum != 0.0F
            ? static_cast<float>(2.0 * static_cast<double>(eigenvalue_difference) /
                                 static_cast<double>(eigenvalue_sum))
            : std::numeric_limits<float>::quiet_NaN();
    return in_plane_spread >= options.normal_minimum_in_plane_spread &&
           normal_spread <= options.normal_maximum_direction_spread &&
           planarity >= options.normal_minimum_planarity;
}

NormalPcaResult normalCandidate(const std::vector<BinarySurfaceInput>& input,
                                const BinarySurfaceInput& query,
                                const std::vector<Neighbor>& neighbors,
                                const BinarySurfaceOptions& options,
                                std::vector<Neighbor>& incidence_neighbors) {
    const NormalPcaResult initial = weightedNormalPca(input, neighbors);
    if (initial.count <= 2U || !initial.valid) {
        return initial;
    }

    Vec3f incidence_normal = initial.normal;
    const Vec3f query_view = query.xyz - query.origin;
    const float query_dot =
        (incidence_normal.z() * query_view.z() + incidence_normal.y() * query_view.y()) +
        incidence_normal.x() * query_view.x();
    if (query_dot > 0.0F) {
        incidence_normal = -incidence_normal;
    }

    incidence_neighbors.clear();
    if (incidence_neighbors.capacity() < neighbors.size()) {
        incidence_neighbors.reserve(neighbors.size());
    }
    for (const auto& neighbor : neighbors) {
        const Vec3f ray = input[neighbor.index].origin - input[neighbor.index].xyz;
        const float numerator =
            (ray.z() * incidence_normal.z() + ray.y() * incidence_normal.y()) +
            ray.x() * incidence_normal.x();
        const float squared_length =
            (ray.z() * ray.z() + ray.y() * ray.y()) + ray.x() * ray.x();
        const float incidence = numerator / std::sqrt(squared_length);
        if (incidence >= options.normal_minimum_incidence_cosine) {
            incidence_neighbors.push_back(neighbor);
        }
    }
    const NormalPcaResult robust = weightedNormalPca(input, incidence_neighbors);
    return normalQuality(robust, options) ? robust : initial;
}

std::vector<BinarySurfacePoint> estimateNormals(const std::vector<BinarySurfaceInput>& input,
                                                const BinarySurfaceOptions& options) {
    std::vector<Vec3f> xyz;
    xyz.reserve(input.size());
    for (const auto& point : input) {
        xyz.push_back(point.xyz);
    }
    CompactRadiusIndex index(
        xyz, static_cast<double>(options.normal_radius_minimum) * 1.01);
    const auto radii = normalRadii(options);
    std::vector<BinarySurfacePoint> output(input.size());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Neighbor> neighbors;
        std::vector<Neighbor> incidence_neighbors;
        neighbors.reserve(128U);
        incidence_neighbors.reserve(128U);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t signed_point_index = 0;
             signed_point_index < static_cast<std::int64_t>(input.size());
             ++signed_point_index) {
            const std::size_t point_index = static_cast<std::size_t>(signed_point_index);
            Vec3f selected_normal = Vec3f::Zero();
            bool fallback_written = false;
            for (const float radius : radii) {
                index.radius(input[point_index].xyz, radius, neighbors);
                const float squared_radius = radius * radius;
                neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
                                               [&](const Neighbor& neighbor) {
                                                   const Vec3f delta = input[point_index].xyz -
                                                       input[neighbor.index].xyz;
                                                   const float squared_distance =
                                                       (delta.z() * delta.z() +
                                                        delta.y() * delta.y()) +
                                                       delta.x() * delta.x();
                                                   return !(squared_distance < squared_radius);
                                               }),
                                neighbors.end());
                const NormalPcaResult candidate = normalCandidate(
                    input, input[point_index], neighbors, options, incidence_neighbors);
                if (normalQuality(candidate, options)) {
                    selected_normal = candidate.normal;
                    break;
                }
                if (!fallback_written && neighbors.size() >= 3U && candidate.valid &&
                    candidate.normal.squaredNorm() > 0.0F) {
                    selected_normal = candidate.normal;
                    fallback_written = true;
                }
            }
            output[point_index] = {input[point_index].xyz, selected_normal,
                                   input[point_index].intensity, 0.0F, input[point_index].weight};
        }
    }
    return output;
}

struct RankedSurfaceCandidate {
    std::uint32_t point = 0U;
    std::size_t local_index = 0U;
    float radial_squared = 0.0F;
};

std::vector<BinarySurfacePoint> selectSurfacePoints(const std::vector<BinarySurfacePoint>& input,
                                                    const BinarySurfaceOptions& options) {
    std::vector<Vec3f> xyz;
    xyz.reserve(input.size());
    for (const auto& point : input) {
        xyz.push_back(point.xyz);
    }
    CompactRadiusIndex index(
        xyz, static_cast<double>(options.selection_radius) * 1.01);
    std::vector<BinarySurfacePoint> output = input;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Neighbor> neighbors;
        std::vector<RankedSurfaceCandidate> ranked;
        neighbors.reserve(64U);
        ranked.reserve(64U);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t signed_point_index = 0;
             signed_point_index < static_cast<std::int64_t>(input.size());
             ++signed_point_index) {
            const std::size_t point_index = static_cast<std::size_t>(signed_point_index);
            index.radius(input[point_index].xyz, options.selection_radius, neighbors);
            ranked.clear();
            if (ranked.capacity() < neighbors.size()) {
                ranked.reserve(neighbors.size());
            }
            const Vec3f query_xyz = input[point_index].xyz;
            const Vec3f query_normal = input[point_index].normal;
            const float search_radius_squared = options.selection_radius * options.selection_radius;
            for (const auto& neighbor : neighbors) {
                const auto& candidate = input[neighbor.index];
                const Vec3f delta = query_xyz - candidate.xyz;
                const float normal_dot =
                    (candidate.normal.z() * query_normal.z() +
                     candidate.normal.y() * query_normal.y()) +
                    candidate.normal.x() * query_normal.x();
                if (!(normal_dot > 0.0F)) {
                    continue;
                }
                const float distance_squared =
                    (delta.z() * delta.z() + delta.y() * delta.y()) +
                    delta.x() * delta.x();
                if (!(distance_squared < search_radius_squared)) {
                    continue;
                }
                const float axial =
                    (delta.z() * query_normal.z() + delta.y() * query_normal.y()) +
                    delta.x() * query_normal.x();
                const Vec3f radial = delta - axial * query_normal;
                const float radial_squared =
                    (radial.z() * radial.z() + radial.y() * radial.y()) +
                    radial.x() * radial.x();
                if (!(radial_squared < std::numeric_limits<float>::infinity())) {
                    continue;
                }
                ranked.push_back({neighbor.index, ranked.size(), radial_squared});
            }
            std::sort(ranked.begin(), ranked.end(), [](const auto& first, const auto& second) {
                if (first.radial_squared != second.radial_squared) {
                    return first.radial_squared < second.radial_squared;
                }
                return first.local_index < second.local_index;
            });

            const float minimum_radius_squared =
                options.minimum_cylinder_radius * options.minimum_cylinder_radius;
            std::size_t selected_count = 0U;
            double selected_weight = 0.0;
            float selected_radius_squared = 0.0F;
            for (const auto& candidate : ranked) {
                ++selected_count;
                selected_weight += static_cast<double>(input[candidate.point].weight);
                selected_radius_squared = candidate.radial_squared;
                if (selected_count >= options.selection_minimum_points &&
                    candidate.radial_squared > minimum_radius_squared &&
                    selected_weight > static_cast<double>(options.selection_target_weight)) {
                    break;
                }
            }

            Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
            double centroid_weight = 0.0;
            for (std::size_t index_in_selection = 0U; index_in_selection < selected_count;
                 ++index_in_selection) {
                const auto& candidate = input[ranked[index_in_selection].point];
                const double weight = static_cast<double>(candidate.weight);
                centroid_weight += weight;
                weighted_sum += weight * candidate.xyz.cast<double>();
            }
            Vec3f centroid = query_xyz;
            if (centroid_weight != 0.0) {
                // 0x15a821 computes one double reciprocal and multiplies all
                // three accumulated coordinates by it.  Three independent
                // divisions are usually equal, but differ by one float ULP
                // for a small number of high-weight helper neighborhoods.
                const double inverse_centroid_weight = 1.0 / centroid_weight;
                centroid.x() =
                    static_cast<float>(weighted_sum.x() * inverse_centroid_weight);
                centroid.y() =
                    static_cast<float>(weighted_sum.y() * inverse_centroid_weight);
                centroid.z() =
                    static_cast<float>(weighted_sum.z() * inverse_centroid_weight);
            }

            auto& point = output[point_index];
            point.weight = static_cast<float>(centroid_weight);
            const float displacement =
                ((query_xyz.z() - centroid.z()) * query_normal.z() +
                 (query_xyz.y() - centroid.y()) * query_normal.y()) +
                (query_xyz.x() - centroid.x()) * query_normal.x();
            const Vec3f projected = query_xyz - displacement * query_normal;
            const Vec3f projected_delta = projected - centroid;
            const float projected_distance_squared =
                (projected_delta.z() * projected_delta.z() +
                 projected_delta.y() * projected_delta.y()) +
                projected_delta.x() * projected_delta.x();
            point.xyz = projected_distance_squared < selected_radius_squared ? projected : centroid;
        }
    }
    return output;
}

struct VoxelAccumulator {
    Vec3f xyz = Vec3f::Zero();
    Vec3f normal = Vec3f::Zero();
    float intensity = 0.0F;
    float curvature = 0.0F;
    float weight = 0.0F;
    VoxelKey key{};

    void add(const BinarySurfacePoint& point) {
        // A newly inserted hash voxel copies its first record directly.  The
        // binary only enters 0x1872b0 for a subsequent point with the same
        // key; multiplying and dividing a single point by its weight changes
        // boundary-anchor coordinates by one ULP.
        if (weight == 0.0F) {
            xyz = point.xyz;
            normal = point.normal;
            intensity = point.intensity;
            curvature = point.curvature;
            weight = point.weight;
            return;
        }

        const float old_weight = weight;
        const float merged_weight = old_weight + point.weight;
        const float inverse_weight = 1.0F / merged_weight;
        for (int axis = 0; axis < 3; ++axis) {
            float merged_xyz = point.weight * point.xyz[axis];
            merged_xyz += old_weight * xyz[axis];
            xyz[axis] = merged_xyz * inverse_weight;

            float merged_normal = point.weight * point.normal[axis];
            merged_normal += old_weight * normal[axis];
            normal[axis] = merged_normal * inverse_weight;
        }
        float normal_squared = normal.z() * normal.z();
        normal_squared += normal.y() * normal.y();
        normal_squared += normal.x() * normal.x();
        if (normal_squared > 0.0F) {
            const float normal_length = std::sqrt(normal_squared);
            normal.x() /= normal_length;
            normal.y() /= normal_length;
            normal.z() /= normal_length;
        }

        float merged_intensity = point.weight * point.intensity;
        merged_intensity += old_weight * intensity;
        intensity = merged_intensity * inverse_weight;
        float merged_curvature = point.weight * point.curvature;
        merged_curvature += old_weight * curvature;
        curvature = merged_curvature * inverse_weight;
        weight = merged_weight;
    }

    [[nodiscard]] BinarySurfacePoint point() const {
        return {xyz, normal, intensity, curvature, weight};
    }
};

VoxelAccumulator mergeOutputVoxelPair(const VoxelAccumulator& first,
                                      const VoxelAccumulator& second) {
    VoxelAccumulator merged;
    merged.weight = first.weight + second.weight;
    const float inverse_weight = 1.0F / merged.weight;

    merged.xyz.x() = first.xyz.x() * first.weight;
    merged.xyz.x() += second.xyz.x() * second.weight;
    merged.xyz.x() *= inverse_weight;
    merged.xyz.y() = first.xyz.y() * first.weight;
    merged.xyz.y() += second.xyz.y() * second.weight;
    merged.xyz.y() *= inverse_weight;
    merged.xyz.z() = second.xyz.z() * second.weight;
    merged.xyz.z() += first.xyz.z() * first.weight;
    merged.xyz.z() *= inverse_weight;

    // 0x16cec0 normalizes the weighted normal sum directly.  Dividing by the
    // total weight first is algebraically equivalent, but changes float32
    // rounding in most of the captured secondary merges.
    merged.normal.x() = second.normal.x() * second.weight;
    merged.normal.x() += first.normal.x() * first.weight;
    merged.normal.y() = second.normal.y() * second.weight;
    merged.normal.y() += first.normal.y() * first.weight;
    merged.normal.z() = second.normal.z() * second.weight;
    merged.normal.z() += first.normal.z() * first.weight;
    const float normal_squared =
        (merged.normal.z() * merged.normal.z() +
         merged.normal.y() * merged.normal.y()) +
        merged.normal.x() * merged.normal.x();
    if (normal_squared > 0.0F) {
        const float normal_length = std::sqrt(normal_squared);
        merged.normal.x() /= normal_length;
        merged.normal.y() /= normal_length;
        merged.normal.z() /= normal_length;
    }

    merged.intensity = first.intensity * first.weight;
    merged.intensity += second.intensity * second.weight;
    merged.intensity *= inverse_weight;
    merged.curvature = first.curvature * first.weight;
    merged.curvature += second.curvature * second.weight;
    merged.curvature *= inverse_weight;
    return merged;
}

std::vector<VoxelAccumulator>
outputVoxelPrimaryAccumulators(const std::vector<BinarySurfacePoint>& input,
                               const BinarySurfaceOptions& options) {
    const double inverse = 1.0 / static_cast<double>(options.output_resolution);
    const auto key = [inverse](const Vec3f& point) {
        return VoxelKey{static_cast<int>(std::floor(static_cast<double>(point.x()) * inverse)),
                        static_cast<int>(std::floor(static_cast<double>(point.y()) * inverse)),
                        static_cast<int>(std::floor(static_cast<double>(point.z()) * inverse))};
    };
    std::vector<VoxelAccumulator> accumulators;
    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> lookup;
    lookup.reserve(input.size());
    for (const auto& point : input) {
        const VoxelKey voxel = key(point.xyz);
        const auto inserted = lookup.emplace(voxel, accumulators.size());
        if (inserted.second) {
            accumulators.push_back({});
            accumulators.back().key = voxel;
        }
        accumulators[inserted.first->second].add(point);
    }
    return accumulators;
}

std::vector<std::uint32_t>
outputVoxelMergeChoices(const std::vector<VoxelAccumulator>& accumulators,
                        const BinarySurfaceOptions& options) {
    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> lookup;
    lookup.reserve(accumulators.size());
    for (std::size_t index = 0U; index < accumulators.size(); ++index) {
        lookup.emplace(accumulators[index].key, index);
    }
    const std::uint32_t no_choice = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> choices(accumulators.size(), no_choice);
    const double merge_distance_squared = static_cast<double>(options.output_merge_distance) *
                                          static_cast<double>(options.output_merge_distance);
    for (std::size_t index = 0U; index < accumulators.size(); ++index) {
        const BinarySurfacePoint point = accumulators[index].point();
        const Vec3f normal = point.normal;
        const std::array<Vec3f, 2U> directions{normal, -normal};
        std::size_t best_neighbor = accumulators.size();
        float best_normal_dot = -std::numeric_limits<float>::max();
        for (const Vec3f& direction : directions) {
            int axis = 0;
            if (std::abs(direction.y()) > std::abs(direction.x())) axis = 1;
            if (std::abs(direction.z()) > std::abs(direction[axis])) axis = 2;
            const int sign = direction[axis] < 0.0F ? -1 : 1;
            VoxelKey neighbor_key = accumulators[index].key;
            if (axis == 0) neighbor_key.x += sign;
            if (axis == 1) neighbor_key.y += sign;
            if (axis == 2) neighbor_key.z += sign;
            const auto found = lookup.find(neighbor_key);
            if (found == lookup.end() || found->second == index) continue;
            const BinarySurfacePoint neighbor = accumulators[found->second].point();
            const Vec3f delta = neighbor.xyz - point.xyz;
            float distance_squared = delta.z() * delta.z();
            distance_squared += delta.y() * delta.y();
            distance_squared += delta.x() * delta.x();
            float normal_dot = normal.z() * neighbor.normal.z();
            normal_dot += normal.y() * neighbor.normal.y();
            normal_dot += normal.x() * neighbor.normal.x();
            if (static_cast<double>(distance_squared) < merge_distance_squared &&
                normal_dot > best_normal_dot &&
                static_cast<double>(normal_dot) >
                    static_cast<double>(options.output_normal_cosine)) {
                best_neighbor = found->second;
                best_normal_dot = normal_dot;
            }
        }
        if (best_neighbor != accumulators.size()) {
            choices[index] = static_cast<std::uint32_t>(best_neighbor);
        }
    }
    return choices;
}

std::vector<BinarySurfacePoint> outputVoxelAggregation(const std::vector<BinarySurfacePoint>& input,
                                                       const BinarySurfaceOptions& options) {
    const double resolution = static_cast<double>(options.output_resolution);
    const double inverse = 1.0 / resolution;
    auto key = [inverse](const Vec3f& point) {
        return VoxelKey{static_cast<int>(std::floor(static_cast<double>(point.x()) * inverse)),
                        static_cast<int>(std::floor(static_cast<double>(point.y()) * inverse)),
                        static_cast<int>(std::floor(static_cast<double>(point.z()) * inverse))};
    };
    std::vector<VoxelAccumulator> accumulators =
        outputVoxelPrimaryAccumulators(input, options);
    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> lookup;
    lookup.reserve(accumulators.size());
    for (std::size_t index = 0U; index < accumulators.size(); ++index) {
        lookup.emplace(accumulators[index].key, index);
    }

    // 0x172f50 first computes the base key from XYZ, then reuses its temporary
    // storage for the normal and its negation.  It therefore emits exactly two
    // directed neighbor keys: +normal first, followed by -normal. Components
    // are compared strictly, so ties retain x before y before z.
    // 0x1630a0 examines those two cells and emits at most one pair: among
    // candidates passing the strict distance and normal gates it selects the
    // largest oriented-normal dot product (the first candidate wins a tie).
    // 0x16d2a0 materializes that pair list before any merges, then visits it
    // once.  It neither scans all six faces nor repeats to a fixed point.
    std::vector<std::pair<std::size_t, std::size_t>> merge_pairs;
    merge_pairs.reserve(2U * accumulators.size());
    const double merge_distance_squared = static_cast<double>(options.output_merge_distance) *
                                          static_cast<double>(options.output_merge_distance);
    for (std::size_t index = 0U; index < accumulators.size(); ++index) {
        const BinarySurfacePoint point = accumulators[index].point();
        const Vec3f normal = point.normal;
        const std::array<Vec3f, 2U> directions{normal, -normal};
        std::size_t best_neighbor = accumulators.size();
        float best_normal_dot = -std::numeric_limits<float>::max();
        for (const Vec3f& direction : directions) {
            int axis = 0;
            if (std::abs(direction.y()) > std::abs(direction.x())) {
                axis = 1;
            }
            if (std::abs(direction.z()) > std::abs(direction[axis])) {
                axis = 2;
            }
            const int sign = direction[axis] < 0.0F ? -1 : 1;
            VoxelKey neighbor_key = accumulators[index].key;
            if (axis == 0) {
                neighbor_key.x += sign;
            }
            if (axis == 1) {
                neighbor_key.y += sign;
            }
            if (axis == 2) {
                neighbor_key.z += sign;
            }
            const auto found = lookup.find(neighbor_key);
            if (found != lookup.end() && found->second != index) {
                const BinarySurfacePoint neighbor = accumulators[found->second].point();
                const Vec3f delta = neighbor.xyz - point.xyz;
                float distance_squared = delta.z() * delta.z();
                distance_squared += delta.y() * delta.y();
                distance_squared += delta.x() * delta.x();
                float normal_dot = normal.z() * neighbor.normal.z();
                normal_dot += normal.y() * neighbor.normal.y();
                normal_dot += normal.x() * neighbor.normal.x();
                // 0x163304 selects a neighbor only when both gates are
                // strict: float distance squared converted to double is less
                // than resolution^2, and the oriented normal dot is greater
                // than cos(20 deg). It does not use abs(dot).
                if (static_cast<double>(distance_squared) < merge_distance_squared &&
                    normal_dot > best_normal_dot &&
                    static_cast<double>(normal_dot) >
                        static_cast<double>(options.output_normal_cosine)) {
                    best_neighbor = found->second;
                    best_normal_dot = normal_dot;
                }
            }
        }
        if (best_neighbor != accumulators.size()) {
            merge_pairs.emplace_back(index, best_neighbor);
        }
    }

    for (const auto& [first_index, second_index] : merge_pairs) {
        auto& first = accumulators[first_index];
        auto& second = accumulators[second_index];
        // 0x16d2a0 calls the merge for every preselected pair even when an
        // earlier pair has already zeroed one side's weight.  Re-merging the
        // surviving point with a zero-weight record is not a no-op at the bit
        // level: its multiply/divide and renormalization can shift fields by
        // one ULP.  The installed implementation uses weight==0 as the only
        // consumed marker and does not skip these calls.
        VoxelAccumulator merged = mergeOutputVoxelPair(first, second);
        if (!(merged.weight > 0.0F)) {
            continue;
        }
        const VoxelKey merged_key = key(merged.xyz);
        VoxelAccumulator* retained = nullptr;
        VoxelAccumulator* consumed = nullptr;
        if (merged_key == first.key) {
            retained = &first;
            consumed = &second;
        } else if (merged_key == second.key) {
            retained = &second;
            consumed = &first;
        } else {
            continue;
        }
        retained->xyz = merged.xyz;
        retained->normal = merged.normal;
        retained->intensity = merged.intensity;
        retained->curvature = merged.curvature;
        retained->weight = merged.weight;
        consumed->weight = 0.0F;
    }

    std::vector<BinarySurfacePoint> output;
    output.reserve(accumulators.size());
    for (const auto& accumulator : accumulators) {
        if (accumulator.weight != 0.0F) {
            output.push_back(accumulator.point());
        }
    }
    return output;
}

std::vector<float> meanNeighborDistances(const std::vector<BinarySurfacePoint>& points,
                                         std::size_t count) {
    std::vector<Vec3f> xyz;
    xyz.reserve(points.size());
    for (const auto& point : points) {
        xyz.push_back(point.xyz);
    }
    ExactKdTree3f index(xyz);
    std::vector<float> result(points.size(), std::numeric_limits<float>::infinity());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Neighbor> neighbors(count);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t signed_point_index = 0;
             signed_point_index < static_cast<std::int64_t>(points.size());
             ++signed_point_index) {
            const std::size_t point_index = static_cast<std::size_t>(signed_point_index);
            std::size_t neighbor_count = 0U;
            index.nearest(points[point_index].xyz, count, neighbors.data(), neighbor_count);
            float sum = 0.0F;
            std::size_t used = 0U;
            // The captured G11 density worker scales the latter half of its
            // sorted KNN squared-distance buffer by 0.5 before evaluating the
            // density.  For the captured k=9 this is entries [5, 9): the self
            // sample and four closest neighbours retain their Euclidean
            // distances, while the four farther neighbours are divided by
            // sqrt(2).  Runtime probes at 0x161fd7/0x1624aa confirm the transform
            // point-for-point; omitting it caused 882 false removals on the
            // captured 87,255-point density input.
            const std::size_t scaled_begin = neighbor_count / 2U + 1U;
            for (std::size_t neighbor_index = 0U; neighbor_index < neighbor_count;
                 ++neighbor_index) {
                float squared_distance = neighbors[neighbor_index].squared_distance;
                if (neighbor_index >= scaled_begin) {
                    squared_distance *= 0.5F;
                }
                const float distance = std::sqrt(squared_distance);
                if (distance <= 1.0e-5F) {
                    continue;
                }
                sum += distance;
                ++used;
            }
            if (used + 1U >= count && used != 0U) {
                result[point_index] = sum / used;
            }
        }
    }
    return result;
}

std::vector<BinarySurfacePoint> densityFilter(const std::vector<BinarySurfacePoint>& input,
                                              const BinarySurfaceOptions& options) {
    const auto mean_distance = meanNeighborDistances(input, options.density_neighbors);
    std::vector<BinarySurfacePoint> output;
    output.reserve(input.size());
    for (std::size_t index = 0U; index < input.size(); ++index) {
        const float distance = mean_distance[index];
        const float density = 1.0F / (distance * distance);
        if (std::isfinite(density) && density >= options.minimum_planar_density) {
            output.push_back(input[index]);
        }
    }
    return output;
}

std::vector<BinarySurfacePoint> adaptiveSor(const std::vector<BinarySurfacePoint>& input,
                                            const BinarySurfaceOptions& options) {
    if (input.size() <= options.sor_neighbors) {
        return input;
    }
    // Despite the generic SOR name, both captured search objects use the
    // point-normal representation.  Running these searches in XYZ space is
    // the source of the former severe over-removal.
    std::vector<Vec3f> normal_space;
    normal_space.reserve(input.size());
    for (const auto& point : input) {
        normal_space.push_back(point.normal);
    }
    ExactKdTree3f spatial(normal_space);
    const std::size_t query_count = options.sor_neighbors + 1U;
    std::vector<float> means(input.size(), std::numeric_limits<float>::infinity());
    std::vector<float> deviations(input.size(), std::numeric_limits<float>::infinity());
    std::vector<std::uint32_t> neighbor_indices(input.size() * query_count,
                                                std::numeric_limits<std::uint32_t>::max());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        // The binary-standard path always queries mean_k + self = 11 points.
        // Allocate once per worker for the non-default API case as well; the
        // query itself performs no allocation or general-purpose sorting.
        std::vector<Neighbor> neighbors(query_count);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t signed_index = 0;
             signed_index < static_cast<std::int64_t>(input.size()); ++signed_index) {
            const std::size_t index = static_cast<std::size_t>(signed_index);
            std::size_t neighbor_count = 0U;
            spatial.nearest(input[index].normal, query_count, neighbors.data(), neighbor_count);
            for (std::size_t neighbor = 0U; neighbor < neighbor_count; ++neighbor) {
                neighbor_indices[index * query_count + neighbor] = neighbors[neighbor].index;
            }
            if (neighbor_count < query_count) {
                continue;
            }
            const std::size_t half = neighbor_count / 2U;
            if (half == 0U) {
                continue;
            }
            float sum = 0.0F;
            for (std::size_t neighbor = half; neighbor < neighbor_count; ++neighbor) {
                sum += std::sqrt(neighbors[neighbor].squared_distance);
            }
            const float mean = std::max(sum / static_cast<float>(half), options.output_resolution);
            float deviation = 0.0F;
            for (std::size_t neighbor = half; neighbor < neighbor_count; ++neighbor) {
                deviation += std::abs(std::sqrt(neighbors[neighbor].squared_distance) - mean);
            }
            means[index] = mean;
            deviations[index] =
                std::max(deviation / static_cast<float>(half), 0.5F * options.output_resolution);
        }
    }
    std::vector<BinarySurfacePoint> output;
    output.reserve(input.size());
    for (std::size_t index = 0U; index < input.size(); ++index) {
        if (!std::isfinite(means[index])) {
            continue;
        }
        std::size_t reference = index;
        for (std::size_t neighbor = 0U; neighbor < query_count; ++neighbor) {
            const std::uint32_t neighbor_index = neighbor_indices[index * query_count + neighbor];
            if (neighbor_index == std::numeric_limits<std::uint32_t>::max()) {
                break;
            }
            if (deviations[neighbor_index] < deviations[reference]) {
                reference = neighbor_index;
            }
        }
        const float reference_deviation =
            std::max(deviations[reference], options.output_resolution);
        if (std::isfinite(means[reference]) && std::isfinite(reference_deviation) &&
            means[index] <= reference_deviation + options.sor_sigma * means[reference]) {
            output.push_back(input[index]);
        }
    }
    return output;
}

std::vector<BinarySurfacePoint> postFilter(const std::vector<BinarySurfacePoint>& input,
                                           const BinarySurfaceOptions& options) {
    std::vector<Vec3f> input_xyz;
    input_xyz.reserve(input.size());
    for (const auto& point : input) {
        input_xyz.push_back(point.xyz);
    }
    CompactRadiusIndex index(
        input_xyz, static_cast<double>(options.post_radius) * 1.01);
    std::vector<BinarySurfacePoint> output = input;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Neighbor> neighbors;
        neighbors.reserve(64U);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t signed_point_index = 0;
             signed_point_index < static_cast<std::int64_t>(input.size());
             ++signed_point_index) {
            const std::size_t point_index = static_cast<std::size_t>(signed_point_index);
            index.radius(input[point_index].xyz, options.post_radius, neighbors);
            const BinarySurfacePoint& query = input[point_index];
            const float squared_radius = options.post_radius * options.post_radius;
            Vec3f normal_sum = Vec3f::Zero();
            std::size_t normal_count = 0U;
            for (const auto& neighbor : neighbors) {
                if (!(neighbor.squared_distance < squared_radius)) {
                    continue;
                }
                const BinarySurfacePoint& candidate = input[neighbor.index];
                const Vec3f delta = query.xyz - candidate.xyz;
                const float plane_distance = std::abs(
                    (delta.z() * query.normal.z() + delta.y() * query.normal.y()) +
                    delta.x() * query.normal.x());
                if (!(plane_distance < options.post_plane_distance)) {
                    continue;
                }
                const float cosine =
                    (query.normal.z() * candidate.normal.z() +
                     query.normal.y() * candidate.normal.y()) +
                    query.normal.x() * candidate.normal.x();
                if (!(cosine > options.post_normal_cosine)) {
                    continue;
                }
                normal_sum.x() += candidate.normal.x();
                normal_sum.y() += candidate.normal.y();
                normal_sum.z() += candidate.normal.z();
                ++normal_count;
            }
            const float divisor = static_cast<float>(std::max<std::size_t>(normal_count, 1U));
            Vec3f filtered_normal;
            filtered_normal.z() = normal_sum.z() / divisor;
            filtered_normal.y() = normal_sum.y() / divisor;
            filtered_normal.x() = normal_sum.x() / divisor;
            const float squared_norm =
                (filtered_normal.z() * filtered_normal.z() +
                 filtered_normal.y() * filtered_normal.y()) +
                filtered_normal.x() * filtered_normal.x();
            if (squared_norm > 0.0F) {
                const float norm = std::sqrt(squared_norm);
                filtered_normal.x() /= norm;
                filtered_normal.y() /= norm;
                filtered_normal.z() /= norm;
            }
            output[point_index].normal = filtered_normal;

            float weight_sum = 0.0F;
            Vec3f centroid = Vec3f::Zero();
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> covariance =
                Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Zero();
            for (const auto& neighbor : neighbors) {
                if (!(neighbor.squared_distance < squared_radius)) {
                    continue;
                }
                const BinarySurfacePoint& candidate = input[neighbor.index];
                const float cosine =
                    (query.normal.z() * candidate.normal.z() +
                     query.normal.y() * candidate.normal.y()) +
                    query.normal.x() * candidate.normal.x();
                if (!(static_cast<double>(cosine) > -0.7071067811865475)) {
                    continue;
                }
                const float old_weight = weight_sum;
                const float new_weight = old_weight + 1.0F;
                const float inverse_weight = 1.0F / new_weight;
                const Vec3f old_centroid = centroid;
                Vec3f new_centroid;
                for (int axis = 0; axis < 3; ++axis) {
                    const float delta_axis = candidate.xyz[axis] - old_centroid[axis];
                    new_centroid[axis] = old_centroid[axis] + inverse_weight * delta_axis;
                }
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        const float right = candidate.xyz[column] - old_centroid[column];
                        const float update =
                            (candidate.xyz[row] - new_centroid[row]) * right;
                        covariance(row, column) =
                            (covariance(row, column) * old_weight + update) * inverse_weight;
                    }
                }
                centroid = new_centroid;
                weight_sum = new_weight;
            }
            if (weight_sum <= 2.0F) {
                output[point_index].curvature = 0.5F;
                continue;
            }
            Eigen::Matrix3f covariance_matrix;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    covariance_matrix(row, column) =
                        row >= column ? covariance(column, row) : covariance(row, column);
                }
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance_matrix);
            if (solver.info() != Eigen::Success) {
                output[point_index].curvature = 0.5F;
                continue;
            }
            const Eigen::Vector3f eigenvalues = solver.eigenvalues();
            const float eigenvalue_sum =
                (eigenvalues.z() + eigenvalues.y()) + eigenvalues.x();
            output[point_index].curvature = static_cast<float>(
                (3.0 * static_cast<double>(eigenvalues.x())) /
                static_cast<double>(eigenvalue_sum));
        }
    }
    return output;
}

struct OcclusionHelperAccumulator {
    Vec3f xyz = Vec3f::Zero();
    Vec3f origin = Vec3f::Zero();
    float intensity = 0.0F;
    float weight = 0.0F;
};

std::vector<BinarySurfaceInput>
aggregateOcclusionHelperInput(const std::vector<BinarySurfaceInput>& input, float resolution) {
    const double inverse_resolution = 1.0 / static_cast<double>(resolution);
    std::vector<OcclusionHelperAccumulator> accumulators;
    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> lookup;
    lookup.reserve(input.size());
    for (const auto& point : input) {
        const VoxelKey key{
            static_cast<int>(std::floor(static_cast<double>(point.xyz.x()) * inverse_resolution)),
            static_cast<int>(std::floor(static_cast<double>(point.xyz.y()) * inverse_resolution)),
            static_cast<int>(std::floor(static_cast<double>(point.xyz.z()) * inverse_resolution))};
        const auto inserted = lookup.emplace(key, accumulators.size());
        if (inserted.second) {
            accumulators.push_back({});
        }
        auto& accumulator = accumulators[inserted.first->second];
        const float old_weight = accumulator.weight;
        const float new_weight = old_weight + point.weight;
        const float inverse_weight = 1.0F / new_weight;
        for (int axis = 0; axis < 3; ++axis) {
            float xyz_sum = old_weight * accumulator.xyz[axis];
            xyz_sum += point.weight * point.xyz[axis];
            accumulator.xyz[axis] = xyz_sum * inverse_weight;
            float origin_sum = old_weight * accumulator.origin[axis];
            origin_sum += point.weight * point.origin[axis];
            accumulator.origin[axis] = origin_sum * inverse_weight;
        }
        float intensity_sum = old_weight * accumulator.intensity;
        intensity_sum += point.weight * point.intensity;
        accumulator.intensity = intensity_sum * inverse_weight;
        accumulator.weight = new_weight;
    }
    std::vector<BinarySurfaceInput> output;
    output.reserve(accumulators.size());
    for (const auto& accumulator : accumulators) {
        if (accumulator.weight == 0.0F) {
            continue;
        }
        output.push_back(
            {accumulator.xyz, accumulator.origin, accumulator.intensity, accumulator.weight});
    }
    return output;
}

std::vector<BinarySurfacePoint> buildOcclusionHelperSurfaceFromInput(
    const std::vector<BinarySurfaceInput>& helper_input,
    const BinarySurfaceOptions& surface_options,
    const BinaryOcclusionOptions& occlusion_options, BinaryOcclusionStageCounts* counts) {
    if (counts != nullptr) {
        counts->helper_input = helper_input.size();
    }
    auto helper_options = surface_options;
    helper_options.minimum_cylinder_radius = occlusion_options.helper_minimum_cylinder_radius;
    helper_options.output_resolution = occlusion_options.helper_output_resolution;
    // Captured helper finalize object+0x10 is the double constant
    // 0.0002249999899417163, i.e. float32(0.015)^2.  This differs from the
    // main 1 cm pass, whose merge distance is 1 cm.
    helper_options.output_merge_distance = 0.015F;
    auto helper = estimateNormals(helper_input, helper_options);
    if (counts != nullptr) {
        counts->helper_normals = helper.size();
    }
    helper = selectSurfacePoints(helper, helper_options);
    if (counts != nullptr) {
        counts->helper_selected = helper.size();
    }
    helper.erase(std::remove_if(helper.begin(), helper.end(),
                                [](const auto& point) {
                                    return !point.xyz.allFinite() || !point.normal.allFinite() ||
                                           point.normal.squaredNorm() < 0.5F;
                                }),
                 helper.end());
    if (counts != nullptr) {
        counts->helper_valid = helper.size();
    }
    if (counts != nullptr) {
        const double inverse_resolution =
            1.0 / static_cast<double>(helper_options.output_resolution);
        std::unordered_map<VoxelKey, std::uint8_t, VoxelKeyHash> initial_voxels;
        initial_voxels.reserve(helper.size());
        for (const auto& point : helper) {
            initial_voxels.emplace(
                VoxelKey{static_cast<int>(
                             std::floor(static_cast<double>(point.xyz.x()) * inverse_resolution)),
                         static_cast<int>(
                             std::floor(static_cast<double>(point.xyz.y()) * inverse_resolution)),
                         static_cast<int>(
                             std::floor(static_cast<double>(point.xyz.z()) * inverse_resolution))},
                0U);
        }
        counts->helper_initial_voxels = initial_voxels.size();
    }
    helper = outputVoxelAggregation(helper, helper_options);
    if (counts != nullptr) {
        counts->helper_output = helper.size();
    }
    return helper;
}

std::vector<BinarySurfacePoint> buildOcclusionHelperSurface(
    const std::vector<BinarySurfaceInput>& raw_rays, const BinarySurfaceOptions& surface_options,
    const BinaryOcclusionOptions& occlusion_options, BinaryOcclusionStageCounts* counts) {
    const auto helper_input =
        aggregateOcclusionHelperInput(raw_rays, occlusion_options.helper_input_resolution);
    return buildOcclusionHelperSurfaceFromInput(helper_input, surface_options, occlusion_options,
                                                counts);
}

// CompactOctree path used by the binary's clean-occlusions classifier. The
// index starts from the first inserted surfel, with a two-cell root whose
// bounds are [p-resolution, p+resolution-epsilon]. Every out-of-bounds
// insertion doubles the root. On each axis the old root becomes the lower
// child only when the new point is beyond the old upper bound; otherwise it
// becomes the upper child. These details reproduce object+0x68..0x90 and
// object+0x24 from 0x171770 for the captured G11 helper surface.
//
// 0x177340 asks the parametric octree walk for at most one occupied leaf. It
// therefore returns every surfel in the first occupied 2 cm leaf crossed by
// the forward infinite ray; the finite range is applied by the primitive.
class FirstOccupiedLeafOctree {
  public:
    FirstOccupiedLeafOctree(const std::vector<BinarySurfacePoint>& surfels, float resolution)
        : resolution_(static_cast<double>(resolution)) {
        if (surfels.empty() || !(resolution_ > 0.0)) {
            return;
        }

        constexpr double binary_bound_epsilon =
            static_cast<double>(std::numeric_limits<float>::epsilon());
        minimum_ = surfels.front().xyz.cast<double>() - Eigen::Vector3d::Constant(resolution_);
        root_cell_count_ = 2;
        double width = 2.0 * resolution_;
        maximum_ = minimum_ + Eigen::Vector3d::Constant(width - binary_bound_epsilon);

        for (const auto& surfel : surfels) {
            const Eigen::Vector3d point = surfel.xyz.cast<double>();
            while ((point.array() < minimum_.array()).any() ||
                   (point.array() >= maximum_.array()).any()) {
                for (int axis = 0; axis < 3; ++axis) {
                    if (point[axis] < maximum_[axis]) {
                        minimum_[axis] -= width;
                    }
                }
                width *= 2.0;
                root_cell_count_ *= 2;
                maximum_ = minimum_ + Eigen::Vector3d::Constant(width - binary_bound_epsilon);
            }
        }

        depth_ = 0;
        for (int cells = root_cell_count_; cells > 1; cells >>= 1) {
            ++depth_;
        }
        nodes_.reserve(surfels.size() * 4U);
        nodes_.emplace_back();
        for (std::uint32_t index = 0U; index < surfels.size(); ++index) {
            const VoxelKey cell = key(surfels[index].xyz.cast<double>());
            std::uint32_t node = 0U;
            for (int bit = depth_ - 1; bit >= 0; --bit) {
                const unsigned child = (static_cast<unsigned>((cell.x >> bit) & 1) << 2U) |
                                       (static_cast<unsigned>((cell.y >> bit) & 1) << 1U) |
                                       static_cast<unsigned>((cell.z >> bit) & 1);
                std::int32_t next = nodes_[node].children[child];
                if (next < 0) {
                    next = static_cast<std::int32_t>(nodes_.size());
                    nodes_[node].children[child] = next;
                    nodes_.emplace_back();
                }
                node = static_cast<std::uint32_t>(next);
            }
            nodes_[node].indices.push_back(index);
        }
    }

    [[nodiscard]] const std::vector<std::uint32_t>* firstOccupiedLeaf(const Vec3f& ray_origin,
                                                                      Vec3f ray_direction) const {
        if (nodes_.empty()) {
            return nullptr;
        }

        // 0x1774c6/0x1774db/0x1774f0 substitute this float constant for an
        // exactly-zero component before converting the slab math to double.
        constexpr float zero_direction_substitute = 1.0e-10F;
        Vec3f reflected_ray_origin = ray_origin;
        for (int axis = 0; axis < 3; ++axis) {
            if (ray_direction[axis] == 0.0F) {
                ray_direction[axis] = zero_direction_substitute;
            }
        }
        unsigned reflection_mask = 0U;
        for (int axis = 0; axis < 3; ++axis) {
            if (ray_direction[axis] < 0.0F) {
                // 0x1778f8..0x177921 (and the y/z siblings) convert each
                // double bound to float separately, then execute addss and
                // subss before the reflected origin is widened back to
                // double for slab parameters.  A direct double expression
                // selects an adjacent leaf for boundary-grazing rays.
                float reflected_origin = static_cast<float>(minimum_[axis]);
                reflected_origin += static_cast<float>(maximum_[axis]);
                reflected_origin -= reflected_ray_origin[axis];
                reflected_ray_origin[axis] = reflected_origin;
                ray_direction[axis] = -ray_direction[axis];
                reflection_mask |= 1U << (2 - axis);
            }
        }
        const Eigen::Vector3d origin = reflected_ray_origin.cast<double>();
        const Eigen::Vector3d direction = ray_direction.cast<double>();

        ParametricBox root{(minimum_.x() - origin.x()) / direction.x(),
                           (minimum_.y() - origin.y()) / direction.y(),
                           (minimum_.z() - origin.z()) / direction.z(),
                           (maximum_.x() - origin.x()) / direction.x(),
                           (maximum_.y() - origin.y()) / direction.y(),
                           (maximum_.z() - origin.z()) / direction.z()};
        const double entry = std::max({root.x0, root.y0, root.z0});
        const double exit = std::min({root.x1, root.y1, root.z1});
        if (!(entry < exit) || root.x1 < 0.0 || root.y1 < 0.0 || root.z1 < 0.0) {
            return nullptr;
        }
        return traverse(0U, root, reflection_mask);
    }

  private:
    struct Node {
        Node() {
            children.fill(-1);
        }
        std::array<std::int32_t, 8> children{};
        std::vector<std::uint32_t> indices;
    };

    struct ParametricBox {
        double x0;
        double y0;
        double z0;
        double x1;
        double y1;
        double z1;
    };

    static unsigned firstNode(double x0, double y0, double z0, double xm, double ym, double zm) {
        unsigned answer = 0U;
        if (x0 > y0) {
            if (x0 > z0) {
                if (ym < x0) {
                    answer |= 2U;
                }
                if (zm < x0) {
                    answer |= 1U;
                }
                return answer;
            }
        } else if (y0 > z0) {
            if (xm < y0) {
                answer |= 4U;
            }
            if (zm < y0) {
                answer |= 1U;
            }
            return answer;
        }
        if (xm < z0) {
            answer |= 4U;
        }
        if (ym < z0) {
            answer |= 2U;
        }
        return answer;
    }

    static unsigned nextNode(double x, unsigned x_case, double y, unsigned y_case, double z,
                             unsigned z_case) {
        if (x < y) {
            if (x < z) {
                return x_case;
            }
        } else if (y < z) {
            return y_case;
        }
        return z_case;
    }

    [[nodiscard]] const std::vector<std::uint32_t>*
    traverse(std::uint32_t node_index, const ParametricBox& box, unsigned reflection_mask) const {
        if (box.x1 < 0.0 || box.y1 < 0.0 || box.z1 < 0.0) {
            return nullptr;
        }
        const Node& node = nodes_[node_index];
        if (!node.indices.empty()) {
            return &node.indices;
        }

        const double xm = (box.x0 + box.x1) * 0.5;
        const double ym = (box.y0 + box.y1) * 0.5;
        const double zm = (box.z0 + box.z1) * 0.5;
        unsigned current = firstNode(box.x0, box.y0, box.z0, xm, ym, zm);
        while (current != 8U) {
            ParametricBox child{};
            unsigned next = 8U;
            switch (current) {
                case 0U:
                    child = {box.x0, box.y0, box.z0, xm, ym, zm};
                    next = nextNode(xm, 4U, ym, 2U, zm, 1U);
                    break;
                case 1U:
                    child = {box.x0, box.y0, zm, xm, ym, box.z1};
                    next = nextNode(xm, 5U, ym, 3U, box.z1, 8U);
                    break;
                case 2U:
                    child = {box.x0, ym, box.z0, xm, box.y1, zm};
                    next = nextNode(xm, 6U, box.y1, 8U, zm, 3U);
                    break;
                case 3U:
                    child = {box.x0, ym, zm, xm, box.y1, box.z1};
                    next = nextNode(xm, 7U, box.y1, 8U, box.z1, 8U);
                    break;
                case 4U:
                    child = {xm, box.y0, box.z0, box.x1, ym, zm};
                    next = nextNode(box.x1, 8U, ym, 6U, zm, 5U);
                    break;
                case 5U:
                    child = {xm, box.y0, zm, box.x1, ym, box.z1};
                    next = nextNode(box.x1, 8U, ym, 7U, box.z1, 8U);
                    break;
                case 6U:
                    child = {xm, ym, box.z0, box.x1, box.y1, zm};
                    next = nextNode(box.x1, 8U, box.y1, 8U, zm, 7U);
                    break;
                case 7U:
                    child = {xm, ym, zm, box.x1, box.y1, box.z1};
                    next = 8U;
                    break;
                default:
                    return nullptr;
            }
            const std::int32_t physical_child = node.children[current ^ reflection_mask];
            if (physical_child >= 0) {
                const auto* hit =
                    traverse(static_cast<std::uint32_t>(physical_child), child, reflection_mask);
                if (hit != nullptr) {
                    return hit;
                }
            }
            current = next;
        }
        return nullptr;
    }

    [[nodiscard]] VoxelKey key(const Eigen::Vector3d& point) const {
        return {static_cast<int>(std::floor((point.x() - minimum_.x()) / resolution_)),
                static_cast<int>(std::floor((point.y() - minimum_.y()) / resolution_)),
                static_cast<int>(std::floor((point.z() - minimum_.z()) / resolution_))};
    }

    double resolution_ = 0.0;
    Eigen::Vector3d minimum_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d maximum_ = Eigen::Vector3d::Zero();
    int root_cell_count_ = 0;
    int depth_ = 0;
    std::vector<Node> nodes_;
};

float orderedDot(const Vec3f& first, const Vec3f& second) {
    // 0x15e175..0x15e1a6 and 0x15e245..0x15e26d accumulate the primitive's
    // plane products as z+y+x. The association matters for grazing rays:
    // x+y+z moves the reconstructed plane intersection by one float ULP.
    float result = first.z() * second.z();
    result += first.y() * second.y();
    result += first.x() * second.x();
    return result;
}

float orderedSquaredNorm(const Vec3f& vector) {
    // The primitive's endpoint, segment and disc checks use z^2+y^2+x^2.
    float result = vector.z() * vector.z();
    result += vector.y() * vector.y();
    result += vector.x() * vector.x();
    return result;
}

std::uint8_t occlusionPrimitive(const BinarySurfacePoint& surfel, const Vec3f& origin,
                                const Vec3f& direction, float length,
                                const BinaryOcclusionOptions& options) {
    if (!surfel.normal.allFinite() || !surfel.xyz.allFinite()) {
        return 0U;
    }
    const float denominator = orderedDot(surfel.normal, direction);
    const Vec3f endpoint = origin + length * direction;
    if (orderedSquaredNorm(surfel.xyz - endpoint) <
        options.ray_hit_tolerance * options.ray_hit_tolerance) {
        return denominator >= 0.0F ? 2U : 1U;
    }
    const float intersection_distance =
        (orderedDot(surfel.normal, surfel.xyz) - orderedDot(surfel.normal, origin)) / denominator;
    const Vec3f intersection = origin + intersection_distance * direction;
    const float allowed_length = length + options.ray_hit_tolerance;
    if (orderedSquaredNorm(intersection - origin) > allowed_length * allowed_length ||
        orderedSquaredNorm(surfel.xyz - intersection) >
            options.surfel_disc_radius * options.surfel_disc_radius) {
        return 6U;
    }
    return denominator >= 0.0F ? 5U : 4U;
}

bool occlusionPairConsistent(const BinarySurfacePoint& first, const BinarySurfacePoint& second,
                             const Vec3f& direction, const BinaryOcclusionOptions& options) {
    const Vec3f delta = first.xyz - second.xyz;
    // 0x15e340 uses a different scalar instruction order from the primitive:
    // pair length is y^2+z^2+x^2 and all pair dots are z+y+x.
    const auto pair_dot = [](const Vec3f& left, const Vec3f& right) {
        float value = left.z() * right.z();
        value += left.y() * right.y();
        value += left.x() * right.x();
        return value;
    };
    float distance_squared = delta.y() * delta.y();
    distance_squared += delta.z() * delta.z();
    distance_squared += delta.x() * delta.x();
    const float distance = std::sqrt(distance_squared);
    const float sine = std::sin(options.normal_tolerance_radians);
    const float cosine = std::cos(options.normal_tolerance_radians);
    const float plane_tolerance = std::max(options.pair_distance_floor, sine * distance);
    const float alpha = std::acos(-pair_dot(second.normal, direction));
    const double slope =
        std::abs(1.0 / std::tan(1.57079632679489661923 - static_cast<double>(alpha)));
    const float radial_from_incidence =
        static_cast<float>(slope) * options.index_resolution * 1.7320507764816284F;
    const float radial_tolerance = std::max(options.ray_hit_tolerance, radial_from_incidence);
    // The branch at 0x15e58c returns false when A's plane gate fails. It is
    // mandatory, not an optional condition around the final normal checks.
    return plane_tolerance > std::abs(pair_dot(second.normal, delta)) &&
           radial_tolerance > distance &&
           plane_tolerance > std::abs(pair_dot(first.normal, delta)) &&
           pair_dot(second.normal, first.normal) > cosine &&
           static_cast<double>(alpha) > 0.78539816339744830962;
}

std::uint8_t classifyOcclusionRay(const BinarySurfaceInput& ray,
                                  const std::vector<BinarySurfacePoint>& helper,
                                  const FirstOccupiedLeafOctree& octree,
                                  const SpatialIndex& local_index,
                                  const BinaryOcclusionOptions& options,
                                  std::vector<Neighbor>& endpoint_neighbors,
                                  BinaryOcclusionRayDiagnostic* diagnostic = nullptr) {
    const Vec3f segment = ray.xyz - ray.origin;
    // The ray wrapper feeding 0x177340 evaluates the norm as y^2+z^2+x^2.
    // Reassociating it as x^2+y^2+z^2 moves a reconstructed endpoint by one
    // float ULP for boundary rays and can change the global endpoint kNN tie.
    float length_squared = segment.y() * segment.y();
    length_squared += segment.z() * segment.z();
    length_squared += segment.x() * segment.x();
    const float length = std::sqrt(length_squared);
    if (!segment.allFinite() || !std::isfinite(length) || length == 0.0F) {
        return 0U;
    }
    const Vec3f direction = segment / length;
    const Vec3f endpoint = ray.origin + length * direction;
    const auto* candidates = octree.firstOccupiedLeaf(ray.origin, direction);
    if (diagnostic != nullptr && candidates != nullptr) {
        diagnostic->first_leaf_candidates = *candidates;
    }

    // 0x177a50 is the empty-leaf path. For every ordinary finite range it
    // performs a k=1 endpoint query and applies the primitive directly; it
    // does not enter the pair-consistency recovery used below.
    if (candidates == nullptr) {
        local_index.nearest(endpoint, 1U, endpoint_neighbors);
        if (diagnostic != nullptr && !endpoint_neighbors.empty()) {
            diagnostic->endpoint_neighbor = endpoint_neighbors.front().index;
        }
        const std::uint8_t status =
            endpoint_neighbors.empty()
                ? 6U
                : occlusionPrimitive(helper[endpoint_neighbors.front().index], ray.origin,
                                     direction, length, options);
        if (diagnostic != nullptr) {
            diagnostic->status = status;
            diagnostic->first_non_miss_primitive = status;
        }
        return status;
    }

    std::uint8_t status = 6U;
    // The primitive loop stops at its first non-6 result, but 0x1779c2 takes
    // A from dda_candidates.back() regardless of where that early stop
    // occurred. Dynamic examples [123,1207] and [280,866] prove this is not
    // the current primitive iterator.
    const std::uint32_t selected = candidates->back();
    for (const std::uint32_t candidate : *candidates) {
        const std::uint8_t candidate_status =
            occlusionPrimitive(helper[candidate], ray.origin, direction, length, options);
        if (candidate_status == 6U) {
            continue;
        }
        status = candidate_status;
        if (diagnostic != nullptr) {
            diagnostic->first_non_miss_primitive = candidate_status;
        }
        break;
    }
    if (status == 1U || status == 2U) {
        if (diagnostic != nullptr) {
            diagnostic->status = status;
        }
        return status;
    }

    // 0x17792f calls the index's k-nearest query with k=1, then invokes the
    // pair predicate once: A is the current primitive candidate and B is the
    // nearest surfel to the finite ray endpoint.  The previous broad-radius
    // search was deliberately removed because it over-recovered status 3.
    local_index.nearest(endpoint, 1U, endpoint_neighbors);
    if (diagnostic != nullptr && !endpoint_neighbors.empty()) {
        diagnostic->endpoint_neighbor = endpoint_neighbors.front().index;
    }
    const bool consistent =
        !endpoint_neighbors.empty() &&
        occlusionPairConsistent(helper[selected], helper[endpoint_neighbors.front().index],
                                direction, options);
    if (diagnostic != nullptr) {
        diagnostic->pair_consistent = consistent;
    }
    if (consistent) {
        if (diagnostic != nullptr) {
            diagnostic->status = 3U;
        }
        return 3U;
    }
    if (diagnostic != nullptr) {
        diagnostic->status = status;
    }
    return status;
}

} // namespace

std::vector<BinarySurfacePoint>
applyBinaryOutputVoxelAggregation(const std::vector<BinarySurfacePoint>& input,
                                   const BinarySurfaceOptions& options) {
    return outputVoxelAggregation(input, options);
}

std::vector<BinarySurfacePoint>
applyBinaryOutputVoxelPrimaryAggregation(const std::vector<BinarySurfacePoint>& input,
                                          const BinarySurfaceOptions& options) {
    const auto accumulators = outputVoxelPrimaryAccumulators(input, options);
    std::vector<BinarySurfacePoint> output;
    output.reserve(accumulators.size());
    for (const auto& accumulator : accumulators) {
        output.push_back(accumulator.point());
    }
    return output;
}

std::vector<std::uint32_t>
inspectBinaryOutputVoxelMergeChoices(const std::vector<BinarySurfacePoint>& input,
                                      const BinarySurfaceOptions& options) {
    return outputVoxelMergeChoices(outputVoxelPrimaryAccumulators(input, options), options);
}

std::vector<BinarySurfacePoint>
applyBinaryMultiScaleNormalEstimation(const std::vector<BinarySurfaceInput>& input,
                                      const BinarySurfaceOptions& options) {
    return estimateNormals(input, options);
}

std::vector<BinarySurfaceInput>
applyBinaryOcclusionHelperInputAggregation(const std::vector<BinarySurfaceInput>& raw_rays,
                                           float resolution) {
    return aggregateOcclusionHelperInput(raw_rays, resolution);
}

std::vector<BinarySurfacePoint> applyBinaryOcclusionHelperSurface(
    const std::vector<BinarySurfaceInput>& raw_rays, const BinarySurfaceOptions& surface_options,
    const BinaryOcclusionOptions& occlusion_options, BinaryOcclusionStageCounts* counts) {
    BinaryOcclusionStageCounts local_counts;
    local_counts.raw_input = raw_rays.size();
    auto helper =
        buildOcclusionHelperSurface(raw_rays, surface_options, occlusion_options, &local_counts);
    if (counts != nullptr) {
        *counts = local_counts;
    }
    return helper;
}

std::vector<BinarySurfacePoint> applyBinaryOcclusionHelperSurfaceFromInput(
    const std::vector<BinarySurfaceInput>& helper_input,
    const BinarySurfaceOptions& surface_options,
    const BinaryOcclusionOptions& occlusion_options, BinaryOcclusionStageCounts* counts) {
    BinaryOcclusionStageCounts local_counts;
    auto helper = buildOcclusionHelperSurfaceFromInput(
        helper_input, surface_options, occlusion_options, &local_counts);
    if (counts != nullptr) {
        *counts = local_counts;
    }
    return helper;
}

std::vector<std::uint8_t>
classifyBinaryOcclusionRays(const std::vector<BinarySurfaceInput>& raw_rays,
                            const std::vector<BinarySurfacePoint>& helper,
                            const BinaryOcclusionOptions& occlusion_options,
                            std::vector<BinaryOcclusionRayDiagnostic>* diagnostics) {
    std::vector<Vec3f> helper_xyz;
    helper_xyz.reserve(helper.size());
    for (const auto& point : helper) {
        helper_xyz.push_back(point.xyz);
    }
    const SpatialIndex local_index(helper_xyz, occlusion_options.index_resolution);
    const FirstOccupiedLeafOctree octree(helper, occlusion_options.index_resolution);
    std::vector<std::uint8_t> statuses(raw_rays.size());
    if (diagnostics != nullptr) {
        diagnostics->assign(raw_rays.size(), {});
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Neighbor> endpoint_neighbors;
        endpoint_neighbors.reserve(16U);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t signed_index = 0;
             signed_index < static_cast<std::int64_t>(raw_rays.size()); ++signed_index) {
            const std::size_t index = static_cast<std::size_t>(signed_index);
            statuses[index] = classifyOcclusionRay(
                raw_rays[index], helper, octree, local_index, occlusion_options, endpoint_neighbors,
                diagnostics != nullptr ? &(*diagnostics)[index] : nullptr);
        }
    }
    return statuses;
}

std::vector<BinarySurfaceInput> applyBinaryOcclusionCleaning(
    const std::vector<BinarySurfaceInput>& raw_rays, std::vector<std::uint8_t>* statuses,
    const BinarySurfaceOptions& surface_options, const BinaryOcclusionOptions& occlusion_options,
    BinaryOcclusionStageCounts* counts) {
    BinaryOcclusionStageCounts local_counts;
    local_counts.raw_input = raw_rays.size();
    const auto helper =
        buildOcclusionHelperSurface(raw_rays, surface_options, occlusion_options, &local_counts);
    const auto classified = classifyBinaryOcclusionRays(raw_rays, helper, occlusion_options);
    if (statuses != nullptr) {
        *statuses = classified;
    }
    std::vector<BinarySurfaceInput> kept;
    kept.reserve(raw_rays.size());
    for (std::size_t index = 0U; index < raw_rays.size(); ++index) {
        if (classified[index] >= 1U && classified[index] <= 3U) {
            kept.push_back(raw_rays[index]);
        }
    }
    local_counts.kept = kept.size();
    if (counts != nullptr) {
        *counts = local_counts;
    }
    return kept;
}

std::vector<BinarySurfaceInput> applyBinaryOcclusionCleaningFromHelperInput(
    const std::vector<BinarySurfaceInput>& raw_rays,
    const std::vector<BinarySurfaceInput>& helper_input, std::vector<std::uint8_t>* statuses,
    const BinarySurfaceOptions& surface_options, const BinaryOcclusionOptions& occlusion_options,
    BinaryOcclusionStageCounts* counts) {
    BinaryOcclusionStageCounts local_counts;
    local_counts.raw_input = raw_rays.size();
    const auto helper = buildOcclusionHelperSurfaceFromInput(
        helper_input, surface_options, occlusion_options, &local_counts);
    const auto classified = classifyBinaryOcclusionRays(raw_rays, helper, occlusion_options);
    if (statuses != nullptr) {
        *statuses = classified;
    }
    std::vector<BinarySurfaceInput> kept;
    kept.reserve(raw_rays.size());
    for (std::size_t index = 0U; index < raw_rays.size(); ++index) {
        if (classified[index] >= 1U && classified[index] <= 3U) {
            kept.push_back(raw_rays[index]);
        }
    }
    local_counts.kept = kept.size();
    if (counts != nullptr) {
        *counts = local_counts;
    }
    return kept;
}

std::vector<BinarySurfacePoint>
applyBinarySurfacePointSelection(const std::vector<BinarySurfacePoint>& input,
                                 const BinarySurfaceOptions& options) {
    return selectSurfacePoints(input, options);
}

std::vector<BinarySurfacePoint>
applyBinaryDensityFilter(const std::vector<BinarySurfacePoint>& input,
                         const BinarySurfaceOptions& options) {
    return densityFilter(input, options);
}

std::vector<BinarySurfacePoint> applyBinaryAdaptiveSor(const std::vector<BinarySurfacePoint>& input,
                                                       const BinarySurfaceOptions& options) {
    return adaptiveSor(input, options);
}

std::vector<BinarySurfacePoint>
applyBinarySurfaceSupportPruning(const std::vector<BinarySurfacePoint>& support,
                                 const std::vector<BinarySurfacePoint>& target,
                                 float radius) {
    std::vector<Vec3f> target_xyz;
    target_xyz.reserve(target.size());
    for (const auto& point : target) {
        target_xyz.push_back(point.xyz);
    }
    SpatialIndex target_index(target_xyz, radius);
    std::vector<BinarySurfacePoint> output = support;
    output.erase(std::remove_if(output.begin(), output.end(), [&](const auto& point) {
                     return !target_index.anyWithinRadius(point.xyz, radius);
                 }),
                 output.end());
    return output;
}

std::vector<BinarySurfacePoint>
applyBinaryPostSmoothingFilter(const std::vector<BinarySurfacePoint>& input,
                               const BinarySurfaceOptions& options) {
    return postFilter(input, options);
}

std::vector<BinarySurfacePoint>
runBinarySurfacePipeline(const std::vector<BinarySurfaceInput>& input,
                         const BinarySurfaceOptions& options, BinarySurfaceStageCounts* counts) {
    BinarySurfaceStageCounts local;
    local.input = input.size();
    using SurfaceClock = std::chrono::steady_clock;
    const auto elapsed = [](const auto& begin, const auto& end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    auto stage_started = SurfaceClock::now();
    auto normal_support = estimateNormals(input, options);
    auto stage_finished = SurfaceClock::now();
    local.seconds_normals = elapsed(stage_started, stage_finished);
    local.normals = normal_support.size();
    stage_started = stage_finished;
    auto points = selectSurfacePoints(normal_support, options);
    stage_finished = SurfaceClock::now();
    local.seconds_selection = elapsed(stage_started, stage_finished);
    local.selected = input.size();
    points.erase(std::remove_if(points.begin(), points.end(),
                                [](const auto& point) {
                                    return !point.xyz.allFinite() || !point.normal.allFinite() ||
                                           point.normal.squaredNorm() < 0.5F;
                                }),
                 points.end());
    local.valid = points.size();
    stage_started = SurfaceClock::now();
    points = outputVoxelAggregation(points, options);
    stage_finished = SurfaceClock::now();
    local.seconds_output_voxels = elapsed(stage_started, stage_finished);
    local.output_voxels = points.size();
    stage_started = stage_finished;
    points = densityFilter(points, options);
    stage_finished = SurfaceClock::now();
    local.seconds_density = elapsed(stage_started, stage_finished);
    local.density = points.size();
    stage_started = stage_finished;
    points = adaptiveSor(points, options);
    stage_finished = SurfaceClock::now();
    local.seconds_adaptive_sor = elapsed(stage_started, stage_finished);
    local.adaptive_sor = points.size();
    // Captured support pruning keeps only normal-stage points within 3 cm of
    // a final target point.  It never moves or removes a target point.
    stage_started = stage_finished;
    normal_support = applyBinarySurfaceSupportPruning(normal_support, points, 0.03F);
    stage_finished = SurfaceClock::now();
    local.seconds_support_pruning = elapsed(stage_started, stage_finished);
    stage_started = stage_finished;
    points = postFilter(points, options);
    stage_finished = SurfaceClock::now();
    local.seconds_post = elapsed(stage_started, stage_finished);
    local.post = points.size();
    if (counts != nullptr) {
        *counts = local;
    }
    return points;
}

} // namespace navvis_recon
