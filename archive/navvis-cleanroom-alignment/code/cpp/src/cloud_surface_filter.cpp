#include "navvis_recon/cloud_surface_filter.hpp"

#include <Eigen/Eigenvalues>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <tuple>

namespace navvis_recon {
namespace {

VoxelKey voxelKey(const Vec3f& point, float resolution) {
    return {static_cast<int>(std::floor(point.x() / resolution)),
            static_cast<int>(std::floor(point.y() / resolution)),
            static_cast<int>(std::floor(point.z() / resolution))};
}

VoxelKey voxelKey(const Vec3f& point, float resolution, const Vec3f& anchor) {
    const double inverse = 1.0 / static_cast<double>(resolution);
    return {static_cast<int>(std::floor(
                (static_cast<double>(point.x()) - static_cast<double>(anchor.x())) * inverse)),
            static_cast<int>(std::floor(
                (static_cast<double>(point.y()) - static_cast<double>(anchor.y())) * inverse)),
            static_cast<int>(std::floor(
                (static_cast<double>(point.z()) - static_cast<double>(anchor.z())) * inverse))};
}

std::vector<std::pair<float, std::size_t>> sortedDistances(const std::vector<SurfacePoint>& cloud,
                                                           std::size_t center) {
    std::vector<std::pair<float, std::size_t>> result;
    result.reserve(cloud.size() - 1U);
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        if (i != center) {
            result.emplace_back((cloud[i].xyz - cloud[center].xyz).norm(), i);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

float median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0F;
    }
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

struct GroupedFreespaceRay {
    int origin_x = 0;
    int origin_y = 0;
    int origin_z = 0;
    int azimuth = 0;
    int elevation = 0;
    float range = 0.0F;
    VoxelKey endpoint;
};

bool sameGroup(const GroupedFreespaceRay& first, const GroupedFreespaceRay& second) {
    return first.origin_x == second.origin_x && first.origin_y == second.origin_y &&
           first.origin_z == second.origin_z && first.azimuth == second.azimuth &&
           first.elevation == second.elevation;
}

std::vector<GroupedFreespaceRay> groupRays(const std::vector<FreespaceRayObservation>& observations,
                                           const DirectionalFreespaceOptions& options,
                                           float shift_fraction) {
    constexpr float pi = 3.14159265358979323846F;
    const float angular_bin = options.angular_bin_degrees * pi / 180.0F;
    const int azimuth_bins = std::max(1, static_cast<int>(std::ceil(2.0F * pi / angular_bin)));
    const float origin_shift = shift_fraction * options.origin_cell;
    const float angular_shift = shift_fraction * angular_bin;
    std::vector<GroupedFreespaceRay> grouped;
    grouped.reserve(observations.size());
    for (const auto& observation : observations) {
        const Vec3f ray = observation.endpoint - observation.origin;
        const float range = ray.norm();
        if (range < options.minimum_origin_distance || range > options.maximum_origin_distance) {
            continue;
        }
        const Vec3f direction = ray / range;
        float azimuth = std::atan2(direction.y(), direction.x());
        if (azimuth < 0.0F) {
            azimuth += 2.0F * pi;
        }
        int azimuth_key = static_cast<int>(std::floor((azimuth + angular_shift) / angular_bin));
        azimuth_key %= azimuth_bins;
        const float elevation = std::atan2(direction.z(), std::hypot(direction.x(), direction.y()));
        grouped.push_back(
            {static_cast<int>(
                 std::floor((observation.origin.x() + origin_shift) / options.origin_cell)),
             static_cast<int>(
                 std::floor((observation.origin.y() + origin_shift) / options.origin_cell)),
             static_cast<int>(
                 std::floor((observation.origin.z() + origin_shift) / options.origin_cell)),
             azimuth_key, static_cast<int>(std::floor((elevation + angular_shift) / angular_bin)),
             range, observation.endpoint_voxel});
    }
    std::sort(grouped.begin(), grouped.end(), [](const auto& first, const auto& second) {
        return std::tie(first.origin_x, first.origin_y, first.origin_z, first.azimuth,
                        first.elevation, first.range) < std::tie(second.origin_x, second.origin_y,
                                                                 second.origin_z, second.azimuth,
                                                                 second.elevation, second.range);
    });
    return grouped;
}

std::unordered_map<VoxelKey, std::uint64_t, VoxelKeyHash>
intersectionCounts(const std::vector<GroupedFreespaceRay>& grouped, float endpoint_margin) {
    std::unordered_map<VoxelKey, std::uint64_t, VoxelKeyHash> result;
    result.reserve(grouped.size() / 4U + 1U);
    std::size_t begin = 0U;
    while (begin < grouped.size()) {
        std::size_t end = begin + 1U;
        while (end < grouped.size() && sameGroup(grouped[begin], grouped[end])) {
            ++end;
        }
        std::size_t beyond = begin;
        for (std::size_t index = begin; index < end; ++index) {
            beyond = std::max(beyond, index + 1U);
            const float threshold = grouped[index].range + endpoint_margin;
            while (beyond < end && grouped[beyond].range <= threshold) {
                ++beyond;
            }
            result[grouped[index].endpoint] += static_cast<std::uint64_t>(end - beyond);
        }
        begin = end;
    }
    return result;
}

std::uint32_t clampCount(std::uint64_t value) {
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(value, std::numeric_limits<std::uint32_t>::max()));
}

Vec3f decodeSphericalFibonacci(std::uint32_t index) {
    // navvis::geometry::SphericalFibonacci<uint32_t, double> reserves
    // UINT32_MAX as invalid, leaving UINT32_MAX samples indexed 0..N-1.
    constexpr double sample_count = static_cast<double>(std::numeric_limits<std::uint32_t>::max());
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double golden_fraction = 0.618033988749894848204586834365638118;
    const double value = static_cast<double>(index);
    const double z = 1.0 - (2.0 * value + 1.0) / sample_count;
    const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double turns = value * golden_fraction;
    // The binary uses fma(index, golden_fraction, -floor(turns)) here.
    // Keeping the multiply and subtraction separate changes the decoded
    // normal by a few ULPs at large uint32 indices and can also change which
    // inverse-mapping candidate is selected.
    const double fractional_turn =
        std::fma(value, golden_fraction, -std::floor(turns));
    const double angle = 2.0 * pi * fractional_turn;
    double sine = 0.0;
    double cosine = 0.0;
#if defined(_MSC_VER)
    sine = std::sin(angle);
    cosine = std::cos(angle);
#else
    ::sincos(angle, &sine, &cosine);
#endif
    return Vec3f(static_cast<float>(radius * cosine), static_cast<float>(radius * sine),
                 static_cast<float>(z));
}

Vec3f quantizeSphericalFibonacci(const Vec3f& input) {
    // computeCentroidNormals already normalizes the PCA eigenvector in float.
    // The original worker converts those float components directly to double
    // for the inverse Spherical Fibonacci lookup; a second normalization here
    // can move boundary cases to a neighboring quantized normal.
    const Eigen::Vector3d point = input.cast<double>();
    if (!point.allFinite()) {
        return Vec3f::Zero();
    }

    // Constant-time inverse from the Spherical Fibonacci mapping used inline
    // by FreespaceOctree::computeCentroidNormals.  Its four candidates and
    // double-precision arithmetic are visible in the worker disassembly.
    constexpr double sample_count = static_cast<double>(std::numeric_limits<std::uint32_t>::max());
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double tau = 2.0 * pi;
    constexpr double sqrt_five = 2.23606797749978969640917366873127623;
    constexpr double golden_ratio = 1.618033988749894848204586834365638118;
    constexpr double golden_fraction = golden_ratio - 1.0;

    const double latitude_scale =
        sample_count * pi * sqrt_five * std::max(0.0, 1.0 - point.z() * point.z());
    const int level =
        std::max(2, latitude_scale > 0.0
                        ? static_cast<int>(std::floor(std::log(latitude_scale) /
                                                      std::log(golden_ratio * golden_ratio)))
                        : 2);
    const double fibonacci_real = std::pow(golden_ratio, static_cast<double>(level)) / sqrt_five;
    const std::int64_t fibonacci_first = static_cast<std::int64_t>(std::round(fibonacci_real));
    const std::int64_t fibonacci_second =
        static_cast<std::int64_t>(std::round(fibonacci_real * golden_ratio));
    const auto fractional = [](double value) { return value - std::floor(value); };
    const double basis_00 =
        tau * (fractional((static_cast<double>(fibonacci_first) + 1.0) * golden_fraction) -
               golden_fraction);
    const double basis_01 =
        tau * (fractional((static_cast<double>(fibonacci_second) + 1.0) * golden_fraction) -
               golden_fraction);
    const double basis_10 = -2.0 * static_cast<double>(fibonacci_first) / sample_count;
    const double basis_11 = -2.0 * static_cast<double>(fibonacci_second) / sample_count;
    const double determinant = basis_00 * basis_11 - basis_01 * basis_10;
    const double azimuth = std::atan2(point.y(), point.x());
    const double latitude = point.z() - (1.0 - 1.0 / sample_count);
    const std::int64_t cell_first = static_cast<std::int64_t>(
        std::floor((basis_11 * azimuth - basis_01 * latitude) / determinant));
    const std::int64_t cell_second = static_cast<std::int64_t>(
        std::floor((-basis_10 * azimuth + basis_00 * latitude) / determinant));

    std::uint32_t best_index = 0U;
    double best_distance = std::numeric_limits<double>::infinity();
    const std::int64_t maximum_index =
        static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()) - 1;
    for (int candidate = 0; candidate < 4; ++candidate) {
        const std::int64_t first = cell_first + (candidate & 1);
        const std::int64_t second = cell_second + (candidate >> 1);
        const std::int64_t raw_index = fibonacci_first * first + fibonacci_second * second;
        const auto index =
            static_cast<std::uint32_t>(std::clamp(raw_index, std::int64_t{0}, maximum_index));
        const Vec3f decoded = decodeSphericalFibonacci(index);
        const double distance = (decoded.cast<double>() - point).squaredNorm();
        if (distance < best_distance) {
            best_distance = distance;
            best_index = index;
        }
    }
    return decodeSphericalFibonacci(best_index);
}

bool sampledRay(const FreespaceRayObservation& observation, float resolution,
                std::uint32_t stride) {
    if (stride <= 1U) {
        return true;
    }
    const VoxelKey origin = voxelKey(observation.origin, resolution);
    std::size_t seed = VoxelKeyHash{}(observation.endpoint_voxel);
    const std::size_t origin_hash = VoxelKeyHash{}(origin);
    seed ^= origin_hash + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    return seed % stride == 0U;
}

} // namespace

Vec3f detail::decodeSphericalFibonacciNormal(std::uint32_t index) {
    return decodeSphericalFibonacci(index);
}

std::size_t VoxelKeyHash::operator()(const VoxelKey& key) const noexcept {
    std::size_t seed = std::hash<int>{}(key.x);
    seed ^= std::hash<int>{}(key.y) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<int>{}(key.z) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    return seed;
}

SurfaceFilterOptions SurfaceFilterOptions::g11Standard(float requested_resolution) {
    SurfaceFilterOptions options;
    options.resolution = requested_resolution;
    // These four values are direct captures from the G11/standard constructor.
    options.minimum_normal_radius = std::max(0.025F, 2.5F * requested_resolution);
    options.maximum_normal_radius = std::max(0.15F, 15.0F * requested_resolution);
    options.number_of_normal_levels = 6;
    options.density_filter_k_neighbors = 16;
    options.statistical_k_neighbors = 16;
    options.search_radius_smoothing = std::max(0.04F, 4.0F * requested_resolution);
    options.normal_smoothing_radius = std::max(0.02F, 2.0F * requested_resolution);
    options.freespace_endpoint_margin = 4.0F * requested_resolution;
    return options;
}

std::unordered_map<VoxelKey, DirectionalFreespaceEvidence, VoxelKeyHash>
DirectionalFreespaceCarver::compute(const std::vector<FreespaceRayObservation>& observations,
                                    const DirectionalFreespaceOptions& options) {
    if (!(options.origin_cell > 0.0F) || !(options.angular_bin_degrees > 0.0F) ||
        !(options.endpoint_margin >= 0.0F) || !(options.intersection_hit_ratio >= 0.0F)) {
        throw std::invalid_argument("invalid directional free-space options");
    }
    std::unordered_map<VoxelKey, DirectionalFreespaceEvidence, VoxelKeyHash> evidence;
    evidence.reserve(observations.size() / 2U + 1U);
    for (const auto& observation : observations) {
        ++evidence[observation.endpoint_voxel].hit_viewpoints;
    }
    const auto primary =
        intersectionCounts(groupRays(observations, options, 0.0F), options.endpoint_margin);
    const auto shifted =
        intersectionCounts(groupRays(observations, options, 0.5F), options.endpoint_margin);
    for (auto& item : evidence) {
        const auto first = primary.find(item.first);
        const auto second = shifted.find(item.first);
        const std::uint64_t first_count = first == primary.end() ? 0U : first->second;
        const std::uint64_t second_count = second == shifted.end() ? 0U : second->second;
        item.second.intersections = clampCount(std::min(first_count, second_count));
        const float ratio = static_cast<float>(item.second.intersections) /
                            static_cast<float>(std::max(item.second.hit_viewpoints, 1U));
        item.second.removed = item.second.intersections >= options.minimum_intersections &&
                              ratio >= options.intersection_hit_ratio;
    }
    return evidence;
}

std::unordered_map<VoxelKey, DirectionalFreespaceEvidence, VoxelKeyHash>
SparseFreespaceCarver::compute(const std::vector<FreespaceRayObservation>& observations,
                               const std::vector<FreespaceCandidate>& candidates,
                               const SparseFreespaceOptions& options,
                               std::vector<SparseFreespaceLeafDiagnostic>* leaf_diagnostics,
                               std::vector<DirectionalFreespaceEvidence>* candidate_evidence,
                               bool build_evidence_map,
                               std::vector<std::uint8_t>* candidate_removed,
                               SparseFreespaceTiming* timing) {
    if (!(options.traversal_resolution > 0.0F) || !(options.ray_radius >= 0.0F) ||
        !(options.endpoint_margin >= 0.0F) || options.ray_stride == 0U ||
        !(options.maximum_incidence_angle_degrees > 0.0F) ||
        !(options.maximum_incidence_angle_degrees <= 90.0F) ||
        !(options.intersection_hit_ratio >= 0.0F)) {
        throw std::invalid_argument("invalid sparse free-space options");
    }

    using Clock = std::chrono::steady_clock;
    auto subphase_started = Clock::now();
    const auto finish_subphase = [&](double& output) {
        output = std::chrono::duration<double>(Clock::now() - subphase_started).count();
        subphase_started = Clock::now();
    };
    SparseFreespaceTiming measured;

    struct CompactLeaf {
        Vec3f centroid = Vec3f::Zero();
        Vec3f normal = Vec3f::Zero();
        std::uint64_t hits = 0U;
        std::uint32_t sampled_hits = 0U;
        std::uint16_t intersections = 0U;
        bool has_normal = false;
        bool removed = false;
    };

    std::unordered_map<VoxelKey, DirectionalFreespaceEvidence, VoxelKeyHash> evidence;
    if (build_evidence_map) {
        evidence.reserve(candidates.size() + 1U);
    }
    if (candidate_evidence != nullptr) {
        candidate_evidence->assign(candidates.size(), {});
    }
    if (candidate_removed != nullptr) {
        candidate_removed->assign(candidates.size(), 0U);
    }
    if (candidates.empty()) {
        return evidence;
    }
    // The binary grows CompactOctree around the first inserted point.  Root
    // expansions are exact powers of two, so the final lattice has the same
    // phase as this anchor even after the root reaches depth 12.
    const Vec3f grid_anchor =
        options.has_grid_anchor ? options.grid_anchor : candidates.front().point;

    // FreespaceOctree::addPoint aggregates raw hits at 2 cm before either
    // normal or intersection computation.  The ray-history shards already
    // collapse nearby returns, so hit_count restores the original raw-point
    // multiplicity instead of treating one origin cluster as one hit.
    std::vector<CompactLeaf> leaves;
    leaves.reserve(candidates.size() / 2U + 1U);
    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> occupancy;
    occupancy.reserve(candidates.size() / 2U + 1U);
    std::vector<std::size_t> candidate_leaves(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const auto key =
            voxelKey(candidates[index].point, options.traversal_resolution, grid_anchor);
        auto inserted = occupancy.emplace(key, leaves.size());
        if (inserted.second) {
            leaves.emplace_back();
        }
        auto& leaf = leaves[inserted.first->second];
        const std::uint64_t hits = std::max(candidates[index].hit_count, 1U);
        if (leaf.hits == 0U) {
            leaf.centroid = candidates[index].point;
            leaf.hits = hits;
        } else {
            leaf.hits += hits;
            // FreespaceOctree updates its float32 centroid online for every
            // inserted point.  The operation order matters at the 6 mm ray
            // gate; a double accumulator followed by one division moves
            // high-support centroids by several float ULPs.
            const float weight = static_cast<float>(hits);
            const float total = static_cast<float>(leaf.hits);
            leaf.centroid += weight * (candidates[index].point - leaf.centroid) / total;
        }
        candidate_leaves[index] = inserted.first->second;
        if (build_evidence_map) {
            evidence[candidates[index].voxel];
        }
    }
    finish_subphase(measured.occupancy_seconds);

    // Index occupied leaves as contiguous z-columns.  Querying every one of
    // the 125 possible keys through the global hash table dominated this
    // phase; the compact octree used by the original only visits populated
    // branches.  The column scan keeps the exact dx -> dy -> dz accumulation
    // order while reducing empty-key probes by a factor of five.
    using ColumnEntry = std::pair<int, std::size_t>;
    int minimum_column_x = occupancy.begin()->first.x;
    int maximum_column_x = minimum_column_x;
    int minimum_column_y = occupancy.begin()->first.y;
    int maximum_column_y = minimum_column_y;
    for (const auto& occupied : occupancy) {
        minimum_column_x = std::min(minimum_column_x, occupied.first.x);
        maximum_column_x = std::max(maximum_column_x, occupied.first.x);
        minimum_column_y = std::min(minimum_column_y, occupied.first.y);
        maximum_column_y = std::max(maximum_column_y, occupied.first.y);
    }
    const std::size_t column_width = static_cast<std::size_t>(
        static_cast<std::int64_t>(maximum_column_x) - minimum_column_x + 1);
    const std::size_t column_height = static_cast<std::size_t>(
        static_cast<std::int64_t>(maximum_column_y) - minimum_column_y + 1);
    const bool use_dense_columns = column_width <= 2'000'000U / column_height &&
                                   column_width * column_height <= 4U * occupancy.size();
    std::vector<std::vector<ColumnEntry>> dense_columns;
    std::unordered_map<VoxelKey, std::vector<ColumnEntry>, VoxelKeyHash> columns;
    if (use_dense_columns) {
        dense_columns.resize(column_width * column_height);
        for (const auto& occupied : occupancy) {
            const std::size_t x = static_cast<std::size_t>(occupied.first.x - minimum_column_x);
            const std::size_t y = static_cast<std::size_t>(occupied.first.y - minimum_column_y);
            dense_columns[x * column_height + y].emplace_back(occupied.first.z, occupied.second);
        }
        for (auto& column : dense_columns) {
            std::sort(column.begin(), column.end(),
                      [](const ColumnEntry& first, const ColumnEntry& second) {
                          return first.first < second.first;
                      });
        }
    } else {
        columns.reserve(occupancy.size() / 4U + 1U);
        for (const auto& occupied : occupancy) {
            columns[VoxelKey{occupied.first.x, occupied.first.y, 0}].emplace_back(occupied.first.z,
                                                                                  occupied.second);
        }
        for (auto& column : columns) {
            std::sort(column.second.begin(), column.second.end(),
                      [](const ColumnEntry& first, const ColumnEntry& second) {
                          return first.first < second.first;
                      });
        }
    }
    const auto find_column = [&](int x, int y) -> const std::vector<ColumnEntry>* {
        if (use_dense_columns) {
            if (x < minimum_column_x || x > maximum_column_x || y < minimum_column_y ||
                y > maximum_column_y) {
                return nullptr;
            }
            const std::size_t dense_x = static_cast<std::size_t>(x - minimum_column_x);
            const std::size_t dense_y = static_cast<std::size_t>(y - minimum_column_y);
            const auto& column = dense_columns[dense_x * column_height + dense_y];
            return column.empty() ? nullptr : &column;
        }
        const auto column = columns.find({x, y, 0});
        return column == columns.end() ? nullptr : &column->second;
    };

    // The original ignores any input normal here.  Its worker queries the
    // CompactOctree in a fixed +/-2-key cube, excludes the center from that
    // query, then prepends the center centroid and computes an unweighted PCA.
    // The disassembly takes the PCA path only when at least three other leaves
    // were found, i.e. four centroid samples including the center.
    for (const auto& occupied : occupancy) {
        auto& leaf = leaves[occupied.second];
        pcl::PointCloud<pcl::PointXYZ> neighbors;
        neighbors.reserve(125U);
        neighbors.emplace_back(leaf.centroid.x(), leaf.centroid.y(), leaf.centroid.z());
        for (int dx = -2; dx <= 2; ++dx) {
            for (int dy = -2; dy <= 2; ++dy) {
                const auto* column = find_column(occupied.first.x + dx, occupied.first.y + dy);
                if (column == nullptr) {
                    continue;
                }
                const int minimum_z = occupied.first.z - 2;
                const int maximum_z = occupied.first.z + 2;
                auto neighbor = std::lower_bound(
                    column->begin(), column->end(), minimum_z,
                    [](const ColumnEntry& entry, int z) { return entry.first < z; });
                for (; neighbor != column->end() && neighbor->first <= maximum_z; ++neighbor) {
                    if (neighbor->second == occupied.second) {
                        continue;
                    }
                    const Vec3f& point = leaves[neighbor->second].centroid;
                    neighbors.emplace_back(point.x(), point.y(), point.z());
                }
            }
        }
        if (neighbors.size() >= 4U) {
            // The installed worker calls PCL's double-precision centroid and
            // normalized covariance helpers in two separate passes, followed
            // by pcl::eigen33.  Besides avoiding the cancellation of E[xx]-
            // E[x]E[x], eigen33 has an observable eigenvector sign convention.
            Eigen::Vector4d mean = Eigen::Vector4d::Zero();
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            pcl::compute3DCentroid(neighbors, mean);
            pcl::computeCovarianceMatrixNormalized(neighbors, mean, covariance);
            double eigenvalue = 0.0;
            Eigen::Vector3d eigenvector = Eigen::Vector3d::Zero();
            pcl::eigen33(covariance, eigenvalue, eigenvector);
            Vec3f normal = eigenvector.cast<float>();
            // Match the worker's scalar float normalization.  It accumulates
            // z^2 + y^2 + x^2, takes one sqrt, then divides each component;
            // Eigen's reduction order can select the adjacent Fibonacci cell
            // for normals lying on a quantization boundary.
            float normal_squared = normal.z() * normal.z();
            normal_squared += normal.y() * normal.y();
            normal_squared += normal.x() * normal.x();
            const float normal_length = std::sqrt(normal_squared);
            if (normal_length > 0.0F) {
                normal.x() /= normal_length;
                normal.y() /= normal_length;
                normal.z() /= normal_length;
            }
            leaf.normal = quantizeSphericalFibonacci(normal);
        }
        leaf.has_normal = leaf.normal.squaredNorm() > 1.0e-12F;
    }
    finish_subphase(measured.normal_seconds);
    // CompactOctree::collectIntersected does not use a voxel DDA.  It walks
    // only occupied octree branches and applies a strict slab intersection at
    // every node.  Recreate that sparse tree here so edge/corner contacts have
    // the same inclusion semantics as the binary.
    struct TraceNode {
        std::array<int, 8> children;
        std::size_t leaf = std::numeric_limits<std::size_t>::max();

        TraceNode() {
            children.fill(-1);
        }
    };

    VoxelKey root_lower = occupancy.begin()->first;
    int root_size = 1;
    int root_depth = 0;
    if (options.has_grid_root) {
        root_lower = options.grid_root_lower;
        root_size = options.grid_root_size;
        for (int size = root_size; size > 1; size /= 2) {
            if ((size % 2) != 0) {
                throw std::invalid_argument("free-space root size must be a power of two");
            }
            ++root_depth;
        }
        for (const auto& occupied : occupancy) {
            if (occupied.first.x < root_lower.x ||
                occupied.first.x >= root_lower.x + root_size ||
                occupied.first.y < root_lower.y ||
                occupied.first.y >= root_lower.y + root_size ||
                occupied.first.z < root_lower.z ||
                occupied.first.z >= root_lower.z + root_size) {
                throw std::invalid_argument("occupied free-space leaf is outside global root");
            }
        }
    } else {
        VoxelKey root_upper = root_lower;
        for (const auto& occupied : occupancy) {
            root_lower.x = std::min(root_lower.x, occupied.first.x);
            root_lower.y = std::min(root_lower.y, occupied.first.y);
            root_lower.z = std::min(root_lower.z, occupied.first.z);
            root_upper.x = std::max(root_upper.x, occupied.first.x);
            root_upper.y = std::max(root_upper.y, occupied.first.y);
            root_upper.z = std::max(root_upper.z, occupied.first.z);
        }
        const std::int64_t maximum_span =
            std::max({static_cast<std::int64_t>(root_upper.x) - root_lower.x + 1,
                      static_cast<std::int64_t>(root_upper.y) - root_lower.y + 1,
                      static_cast<std::int64_t>(root_upper.z) - root_lower.z + 1});
        while (static_cast<std::int64_t>(root_size) < maximum_span) {
            root_size *= 2;
            ++root_depth;
        }
    }

    std::vector<TraceNode> trace_nodes(1);
    trace_nodes.reserve(2U * leaves.size());
    for (const auto& occupied : occupancy) {
        int node_index = 0;
        int offset_x = occupied.first.x - root_lower.x;
        int offset_y = occupied.first.y - root_lower.y;
        int offset_z = occupied.first.z - root_lower.z;
        for (int depth = root_depth; depth > 0; --depth) {
            const int half = 1 << (depth - 1);
            int child = 0;
            if (offset_x >= half) {
                child |= 4;
                offset_x -= half;
            }
            if (offset_y >= half) {
                child |= 2;
                offset_y -= half;
            }
            if (offset_z >= half) {
                child |= 1;
                offset_z -= half;
            }
            int next = trace_nodes[static_cast<std::size_t>(node_index)]
                           .children[static_cast<std::size_t>(child)];
            if (next < 0) {
                next = static_cast<int>(trace_nodes.size());
                trace_nodes[static_cast<std::size_t>(node_index)]
                    .children[static_cast<std::size_t>(child)] = next;
                trace_nodes.emplace_back();
            }
            node_index = next;
        }
        trace_nodes[static_cast<std::size_t>(node_index)].leaf = occupied.second;
    }
    finish_subphase(measured.trace_seconds);

    const double traversal_resolution = static_cast<double>(options.traversal_resolution);
    const Eigen::Vector3d grid_anchor_d = grid_anchor.cast<double>();
    auto intervalsIntersectSegment = [](const std::array<double, 3>& axis_near,
                                        const std::array<double, 3>& axis_far,
                                        double segment_length) {
        double near_distance = -std::numeric_limits<double>::infinity();
        double far_distance = std::numeric_limits<double>::infinity();
        for (int axis = 0; axis < 3; ++axis) {
            near_distance = std::max(near_distance, axis_near[axis]);
            far_distance = std::min(far_distance, axis_far[axis]);
        }
        // These are the strict comparisons used by
        // CompactOctree::collectIntersectedRecursive.
        return near_distance < far_distance && far_distance >= 0.0 &&
               near_distance <= segment_length;
    };

    const float radius_squared = options.ray_radius * options.ray_radius;
    constexpr float pi = 3.14159265358979323846F;
    const float maximum_incidence_angle = options.maximum_incidence_angle_degrees * pi / 180.0F;

    for (const auto& observation : observations) {
        if (!sampledRay(observation, options.traversal_resolution, options.ray_stride)) {
            continue;
        }
        const Vec3f ray = observation.endpoint - observation.origin;
        // The installed worker's scalar/SSE reduction evaluates the squared
        // norm as x^2 + (y^2 + z^2).  A different association shifts the
        // shortened endpoint by one float ULP and changes a handful of exact
        // octree and 6 mm distance-boundary decisions.
        float length_squared = ray.y() * ray.y();
        length_squared += ray.z() * ray.z();
        length_squared += ray.x() * ray.x();
        const float length = std::sqrt(length_squared);
        if (!std::isfinite(length) || length <= options.endpoint_margin) {
            continue;
        }
        // The worker computes one reciprocal and multiplies all components;
        // three component-wise divisions round differently at float32.
        const float inverse_length = 1.0F / length;
        const Vec3f direction = inverse_length * ray;
        float entry = options.minimum_origin_distance;
        float exit = std::min(length - options.endpoint_margin, options.maximum_origin_distance);
        if (!(entry < exit)) {
            continue;
        }
        const Vec3f start = observation.origin + entry * direction;
        const Vec3f end = observation.origin + exit * direction;
        const Vec3f segment_delta = end - start;
        float projection_denominator = segment_delta.y() * segment_delta.y();
        projection_denominator += segment_delta.z() * segment_delta.z();
        projection_denominator += segment_delta.x() * segment_delta.x();
        const VoxelKey terminal_cell = voxelKey(end, options.traversal_resolution, grid_anchor);
        const Eigen::Vector3d segment_start = start.cast<double>();
        const Eigen::Vector3d segment_end = end.cast<double>();
        const Eigen::Vector3d segment = segment_end - segment_start;
        const double segment_length = segment.norm();
        if (!(segment_length > 0.0) || !std::isfinite(segment_length)) {
            continue;
        }
        Eigen::Vector3d segment_direction = segment / segment_length;
        const Eigen::Vector3d root_minimum =
            grid_anchor_d +
            traversal_resolution * Eigen::Vector3d(root_lower.x, root_lower.y, root_lower.z);
        const Eigen::Vector3d root_maximum =
            root_minimum + Eigen::Vector3d::Constant(traversal_resolution * root_size);

        // CompactOctree reflects negative ray axes about the root bounds and
        // then traverses using an all-positive direction plus an octant XOR
        // mask.  Computing the negative-direction slabs directly and swapping
        // them is mathematically equivalent, but changes the last few double
        // bits for boundary contacts.
        constexpr double direction_epsilon = std::numeric_limits<double>::epsilon();
        Eigen::Vector3d reflected_start = segment_start;
        std::array<bool, 3> direction_negative{};
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(segment_direction[axis]) < direction_epsilon) {
                segment_direction[axis] =
                    std::signbit(segment_direction[axis]) ? -direction_epsilon : direction_epsilon;
            }
            direction_negative[axis] = segment_direction[axis] < 0.0;
            if (direction_negative[axis]) {
                segment_direction[axis] = -segment_direction[axis];
                reflected_start[axis] =
                    root_minimum[axis] + root_maximum[axis] - reflected_start[axis];
            }
        }

        // CompactOctree::collectIntersected computes the three root slab
        // intervals once.  Recursive children inherit one half of each
        // interval, so no division is repeated below the root.
        std::array<double, 3> root_near{};
        std::array<double, 3> root_far{};
        for (int axis = 0; axis < 3; ++axis) {
            root_near[axis] =
                (root_minimum[axis] - reflected_start[axis]) / segment_direction[axis];
            root_far[axis] =
                (root_maximum[axis] - reflected_start[axis]) / segment_direction[axis];
        }

        auto collectIntersected = [&](auto&& self, int node_index, int depth,
                                      const VoxelKey& node_lower, int node_size,
                                      const std::array<double, 3>& axis_near,
                                      const std::array<double, 3>& axis_far) -> void {
            if (!intervalsIntersectSegment(axis_near, axis_far, segment_length)) {
                return;
            }
            const auto& node = trace_nodes[static_cast<std::size_t>(node_index)];
            if (depth == 0) {
                if (node.leaf == std::numeric_limits<std::size_t>::max()) {
                    return;
                }
                if (node_lower == terminal_cell) {
                    return;
                }
                auto& leaf = leaves[node.leaf];
                // Preserve the worker's float32 operation order exactly.  It
                // forms the shortened segment, computes an unclamped line
                // projection parameter, constructs the closest point, and
                // only then accumulates the three squared residuals.  The
                // algebraically equivalent |p-o|^2-projection^2 expression
                // loses enough precision to flip decisions at the 6 mm gate.
                const Vec3f relative = leaf.centroid - start;
                float projection_numerator = relative.y() * segment_delta.y();
                projection_numerator += relative.z() * segment_delta.z();
                projection_numerator += relative.x() * segment_delta.x();
                const float projection = projection_numerator / projection_denominator;
                const Vec3f closest = start + projection * segment_delta;
                const Vec3f residual = leaf.centroid - closest;
                float perpendicular_squared = residual.y() * residual.y();
                perpendicular_squared += residual.z() * residual.z();
                perpendicular_squared += residual.x() * residual.x();
                float incidence_angle = 0.0F;
                if (leaf.has_normal) {
                    // The installed worker evaluates the float dot product in
                    // y, z, x order before calling acosf.  One-ULP differences
                    // here are observable at the inclusive 85-degree gate.
                    float normal_dot = leaf.normal.y() * direction.y();
                    normal_dot += leaf.normal.z() * direction.z();
                    normal_dot += leaf.normal.x() * direction.x();
                    const float absolute_dot =
                        std::clamp(std::abs(normal_dot), 0.0F, 1.0F);
                    incidence_angle = std::acos(absolute_dot);
                }
                if (perpendicular_squared <= radius_squared &&
                    (!leaf.has_normal || incidence_angle <= maximum_incidence_angle)) {
                    const std::uint32_t increment = std::max(observation.return_count, 1U);
                    const std::uint32_t total =
                        static_cast<std::uint32_t>(leaf.intersections) + increment;
                    leaf.intersections = static_cast<std::uint16_t>(std::min(total, 65535U));
                }
                return;
            }
            const int half = node_size / 2;
            const std::array<double, 3> axis_middle{0.5 * (axis_near[0] + axis_far[0]),
                                                    0.5 * (axis_near[1] + axis_far[1]),
                                                    0.5 * (axis_near[2] + axis_far[2])};
            for (int child = 0; child < 8; ++child) {
                const int child_index = node.children[static_cast<std::size_t>(child)];
                if (child_index < 0) {
                    continue;
                }
                const VoxelKey child_lower{node_lower.x + ((child & 4) != 0 ? half : 0),
                                           node_lower.y + ((child & 2) != 0 ? half : 0),
                                           node_lower.z + ((child & 1) != 0 ? half : 0)};
                std::array<double, 3> child_near{};
                std::array<double, 3> child_far{};
                const std::array<bool, 3> high_child{(child & 4) != 0, (child & 2) != 0,
                                                     (child & 1) != 0};
                for (int axis = 0; axis < 3; ++axis) {
                    // For a negative direction, the spatially high child is
                    // encountered before the spatially low child.  This is
                    // the octant XOR/reflection mask in the original code.
                    const bool near_half = high_child[axis] == direction_negative[axis];
                    child_near[axis] = near_half ? axis_near[axis] : axis_middle[axis];
                    child_far[axis] = near_half ? axis_middle[axis] : axis_far[axis];
                }
                self(self, child_index, depth - 1, child_lower, half, child_near, child_far);
            }
        };
        collectIntersected(collectIntersected, 0, root_depth, root_lower, root_size, root_near,
                           root_far);
    }
    finish_subphase(measured.intersection_seconds);

    for (const auto& occupied : occupancy) {
        auto& leaf = leaves[occupied.second];
        leaf.sampled_hits = static_cast<std::uint32_t>(std::max<std::uint64_t>(
            1U, std::min<std::uint64_t>(65535U, (leaf.hits + options.ray_stride - 1U) /
                                                    options.ray_stride)));
        leaf.removed = leaf.intersections >= options.minimum_intersections &&
                       static_cast<float>(leaf.intersections) >=
                           options.intersection_hit_ratio * static_cast<float>(leaf.sampled_hits);
        if (leaf_diagnostics != nullptr) {
            leaf_diagnostics->push_back({occupied.first, leaf.centroid, leaf.normal,
                                         static_cast<std::uint32_t>(std::min<std::uint64_t>(
                                             leaf.hits, std::numeric_limits<std::uint32_t>::max())),
                                         leaf.intersections, leaf.removed});
        }
    }
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
        const auto& leaf = leaves[candidate_leaves[index]];
        if (build_evidence_map || candidate_evidence != nullptr) {
            DirectionalFreespaceEvidence item;
            item.hit_viewpoints = leaf.sampled_hits;
            item.intersections = leaf.intersections;
            item.removed = leaf.removed;
            if (build_evidence_map) {
                evidence[candidates[index].voxel] = item;
            }
            if (candidate_evidence != nullptr) {
                (*candidate_evidence)[index] = item;
            }
        }
        if (candidate_removed != nullptr) {
            (*candidate_removed)[index] = static_cast<std::uint8_t>(leaf.removed);
        }
    }
    finish_subphase(measured.finalize_seconds);
    if (timing != nullptr) {
        *timing = measured;
    }
    return evidence;
}

std::vector<SurfacePoint> VoxelAggregator::aggregate(const std::vector<LaserPoint>& input,
                                                     float resolution, bool regularize_grid) {
    struct Accumulator {
        Vec3f xyz = Vec3f::Zero();
        Vec3f origin = Vec3f::Zero();
        float intensity = 0.0F;
        float weight = 0.0F;
    };
    std::unordered_map<VoxelKey, Accumulator, VoxelKeyHash> buckets;
    for (const auto& point : input) {
        const float weight = std::max(point.ray_weight, 1.0e-6F);
        auto& bucket = buckets[voxelKey(point.xyz, resolution)];
        bucket.xyz += weight * point.xyz;
        bucket.origin += weight * point.origin;
        bucket.intensity += weight * point.intensity;
        bucket.weight += weight;
    }
    std::vector<SurfacePoint> result;
    result.reserve(buckets.size());
    for (const auto& entry : buckets) {
        const auto& key = entry.first;
        const auto& bucket = entry.second;
        SurfacePoint point;
        point.weight = bucket.weight;
        point.xyz = bucket.xyz / bucket.weight;
        point.origin = bucket.origin / bucket.weight;
        point.intensity = bucket.intensity / bucket.weight;
        if (regularize_grid) {
            point.xyz = resolution * (Vec3f(key.x, key.y, key.z) + Vec3f::Constant(0.5F));
        }
        result.push_back(point);
    }
    return result;
}

void MultiScaleNormalEstimator::compute(std::vector<SurfacePoint>& cloud,
                                        const SurfaceFilterOptions& options) {
    if (cloud.size() < 6U) {
        return;
    }
    std::vector<float> radii;
    for (int level = 0; level < options.number_of_normal_levels; ++level) {
        const float alpha = options.number_of_normal_levels == 1
                                ? 0.0F
                                : static_cast<float>(level) /
                                      static_cast<float>(options.number_of_normal_levels - 1);
        radii.push_back(
            options.minimum_normal_radius *
            std::pow(options.maximum_normal_radius / options.minimum_normal_radius, alpha));
    }

#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(cloud.size()); ++index) {
        float best_curvature = std::numeric_limits<float>::infinity();
        Vec3f best_normal = Vec3f::Zero();
        for (const float radius : radii) {
            std::vector<Vec3f> neighbors;
            for (const auto& candidate : cloud) {
                if ((candidate.xyz - cloud[index].xyz).squaredNorm() <= radius * radius) {
                    neighbors.push_back(candidate.xyz);
                }
            }
            if (neighbors.size() < 6U) {
                continue;
            }
            Vec3f mean = Vec3f::Zero();
            for (const auto& value : neighbors) {
                mean += value;
            }
            mean /= static_cast<float>(neighbors.size());
            Mat3f covariance = Mat3f::Zero();
            for (const auto& value : neighbors) {
                const Vec3f offset = value - mean;
                covariance += offset * offset.transpose();
            }
            covariance /= static_cast<float>(neighbors.size());
            Eigen::SelfAdjointEigenSolver<Mat3f> solver(covariance);
            if (solver.info() != Eigen::Success) {
                continue;
            }
            const float curvature =
                solver.eigenvalues().x() / std::max(solver.eigenvalues().sum(), 1.0e-12F);
            if (curvature < best_curvature) {
                best_curvature = curvature;
                best_normal = solver.eigenvectors().col(0).normalized();
            }
        }
        if (best_normal.squaredNorm() > 0.5F) {
            if (best_normal.dot(cloud[index].xyz - cloud[index].origin) > 0.0F) {
                best_normal = -best_normal;
            }
            cloud[index].normal = best_normal;
            cloud[index].has_normal = true;
        }
    }
}

FreespaceOctree::FreespaceOctree(float resolution) : resolution_(resolution) {}

void FreespaceOctree::addPoint(const SurfacePoint& point) {
    auto& node = nodes_[voxelKey(point.xyz, resolution_)];
    node.points.push_back(point);
    ++node.hits;
}

void FreespaceOctree::computeCentroidNormals() {
    for (auto& entry : nodes_) {
        auto& node = entry.second;
        Vec3f xyz = Vec3f::Zero();
        Vec3f origin = Vec3f::Zero();
        Vec3f normal = Vec3f::Zero();
        float intensity = 0.0F;
        float weight = 0.0F;
        for (const auto& point : node.points) {
            const float w = std::max(point.weight, 1.0e-6F);
            xyz += w * point.xyz;
            origin += w * point.origin;
            intensity += w * point.intensity;
            if (point.has_normal) {
                normal += w * point.normal;
            }
            weight += w;
        }
        node.centroid.xyz = xyz / weight;
        node.centroid.origin = origin / weight;
        node.centroid.intensity = intensity / weight;
        node.centroid.normal = normalizedOr(normal);
        node.centroid.has_normal = node.centroid.normal.squaredNorm() > 0.5F;
        node.centroid.weight = weight;
    }
}

void FreespaceOctree::computeIntersections(const SurfaceFilterOptions& options) {
    constexpr float pi = 3.14159265358979323846F;
    const float minimum_incidence =
        std::cos(options.freespace_maximum_incidence_angle_deg * pi / 180.0F);
    for (const auto& source_entry : nodes_) {
        const SurfacePoint& point = source_entry.second.centroid;
        Vec3f ray = point.xyz - point.origin;
        const float length = ray.norm();
        if (length < options.freespace_minimum_origin_distance ||
            length > options.freespace_maximum_origin_distance) {
            continue;
        }
        ray /= length;
        if (point.has_normal && std::abs(point.normal.dot(ray)) < minimum_incidence) {
            continue;
        }
        const float ray_length = std::max(0.0F, length - options.freespace_endpoint_margin);
        const int steps =
            std::max(1, static_cast<int>(std::ceil(ray_length / (0.5F * resolution_))));
        std::unordered_map<VoxelKey, bool, VoxelKeyHash> visited;
        for (int step = 0; step < steps; ++step) {
            const float distance =
                ray_length * static_cast<float>(step) / static_cast<float>(steps);
            visited[voxelKey(point.origin + distance * ray, resolution_)] = true;
        }
        for (const auto& item : visited) {
            const auto found = nodes_.find(item.first);
            if (found != nodes_.end()) {
                ++found->second.intersections;
            }
        }
    }
}

void FreespaceOctree::removeFreespaceVoxels(const SurfaceFilterOptions& options) {
    for (auto& entry : nodes_) {
        auto& node = entry.second;
        const float ratio =
            static_cast<float>(node.intersections) / static_cast<float>(std::max(node.hits, 1));
        node.removed = node.intersections >= options.freespace_minimum_intersections &&
                       ratio >= options.freespace_intersection_hit_ratio;
    }
}

std::vector<SurfacePoint> FreespaceOctree::toCompactOccupancyOctree() const {
    std::vector<SurfacePoint> result;
    result.reserve(nodes_.size());
    for (const auto& entry : nodes_) {
        if (!entry.second.removed) {
            result.push_back(entry.second.centroid);
        }
    }
    return result;
}

std::vector<SurfacePoint>
AdaptiveStatisticalOutlierRemoval::filter(const std::vector<SurfacePoint>& cloud, int k_neighbors,
                                          float standard_deviations) {
    if (cloud.size() <= static_cast<std::size_t>(k_neighbors)) {
        return cloud;
    }
    std::vector<float> local_distances(cloud.size());
#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(cloud.size()); ++i) {
        auto distances = sortedDistances(cloud, static_cast<std::size_t>(i));
        const int k = std::min(k_neighbors, static_cast<int>(distances.size()));
        local_distances[i] =
            std::accumulate(distances.begin(), distances.begin() + k, 0.0F,
                            [](float sum, const auto& item) { return sum + item.first; }) /
            static_cast<float>(k);
    }
    const float center = median(local_distances);
    std::vector<float> deviations;
    deviations.reserve(local_distances.size());
    for (const float distance : local_distances) {
        deviations.push_back(std::abs(distance - center));
    }
    const float sigma = 1.4826F * median(std::move(deviations));
    const float threshold = center + standard_deviations * std::max(sigma, 1.0e-6F);
    std::vector<SurfacePoint> result;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        if (local_distances[i] <= threshold) {
            result.push_back(cloud[i]);
        }
    }
    return result;
}

CloudSurfaceFilter::CloudSurfaceFilter(SurfaceFilterOptions options)
    : options_(std::move(options)) {}

std::vector<SurfacePoint> CloudSurfaceFilter::densityFilter(std::vector<SurfacePoint> cloud) const {
    if (cloud.size() <= static_cast<std::size_t>(options_.density_filter_k_neighbors)) {
        return cloud;
    }
    std::vector<SurfacePoint> result;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        auto distances = sortedDistances(cloud, i);
        const float radius = distances[options_.density_filter_k_neighbors - 1].first;
        const float effective_resolution =
            radius / std::sqrt(static_cast<float>(options_.density_filter_k_neighbors));
        if (effective_resolution <= options_.maximum_effective_planar_resolution) {
            result.push_back(cloud[i]);
        }
    }
    return result;
}

std::vector<SurfacePoint> CloudSurfaceFilter::smooth(const std::vector<SurfacePoint>& cloud) const {
    std::vector<SurfacePoint> result = cloud;
    const float sigma = std::max(options_.search_radius_smoothing * 0.5F, 1.0e-6F);
#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(cloud.size()); ++i) {
        if (!cloud[i].has_normal) {
            continue;
        }
        Vec3f center = Vec3f::Zero();
        Vec3f normal = Vec3f::Zero();
        float total_weight = 0.0F;
        for (const auto& candidate : cloud) {
            const float squared_distance = (candidate.xyz - cloud[i].xyz).squaredNorm();
            if (squared_distance >
                    options_.search_radius_smoothing * options_.search_radius_smoothing ||
                !candidate.has_normal) {
                continue;
            }
            const float spatial = std::exp(-squared_distance / (2.0F * sigma * sigma));
            const float compatible =
                std::pow(std::max(0.0F, candidate.normal.dot(cloud[i].normal)), 2.0F);
            const float weight = spatial * compatible;
            center += weight * candidate.xyz;
            normal += weight * candidate.normal;
            total_weight += weight;
        }
        if (total_weight <= 1.0e-8F) {
            continue;
        }
        center /= total_weight;
        const Vec3f displacement = center - cloud[i].xyz;
        // Remove movement normal to the surface: smooth only in the tangent plane.
        result[i].xyz = center - displacement.dot(cloud[i].normal) * cloud[i].normal;
        result[i].normal = normalizedOr(normal, cloud[i].normal);
    }
    return result;
}

SurfaceFilterResult CloudSurfaceFilter::filter(const std::vector<LaserPoint>& input) const {
    auto cloud = VoxelAggregator::aggregate(input, options_.resolution, options_.regularize_grid);
    MultiScaleNormalEstimator::compute(cloud, options_);
    cloud.erase(std::remove_if(cloud.begin(), cloud.end(),
                               [](const SurfacePoint& point) { return !point.has_normal; }),
                cloud.end());
    if (options_.clean_freespace) {
        FreespaceOctree octree(options_.resolution);
        for (const auto& point : cloud) {
            octree.addPoint(point);
        }
        octree.computeCentroidNormals();
        octree.computeIntersections(options_);
        octree.removeFreespaceVoxels(options_);
        cloud = octree.toCompactOccupancyOctree();
    }
    cloud = densityFilter(std::move(cloud));
    cloud = AdaptiveStatisticalOutlierRemoval::filter(cloud, options_.statistical_k_neighbors,
                                                      options_.statistical_stddev);
    return SurfaceFilterResult{smooth(cloud), cloud};
}

} // namespace navvis_recon
