#include "navvis_recon/binary_surface_pipeline.hpp"
#include "navvis_recon/cloud_surface_filter.hpp"
#include "navvis_recon/types.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using navvis_recon::normalizedOr;
using navvis_recon::Vec3f;
using navvis_recon::VoxelKey;
using navvis_recon::VoxelKeyHash;

namespace {

// Version-1 shard layout emitted by pandar_cloud_pipeline.cpp.  Keeping this
// adapter separate makes completed long runs reproducible while later shard
// versions can retain additional ray-origin statistics.
struct DiskRecordV1 {
    std::int32_t key_x;
    std::int32_t key_y;
    std::int32_t key_z;
    float xyz_x;
    float xyz_y;
    float xyz_z;
    float normal_x;
    float normal_y;
    float normal_z;
    float intensity;
    std::uint32_t count;
};
static_assert(sizeof(DiskRecordV1) == 44U);

// Version-2 ray-history layout emitted as .raytile by pandar_cloud_pipeline.
// New shards contain one exact raw ray per record.  count > 1 identifies an
// older clustered shard and is retained only for backward compatibility.
struct DiskRecordV2 {
    std::int32_t key_x;
    std::int32_t key_y;
    std::int32_t key_z;
    float xyz_x;
    float xyz_y;
    float xyz_z;
    float origin_x;
    float origin_y;
    float origin_z;
    float normal_x;
    float normal_y;
    float normal_z;
    float intensity;
    std::uint32_t count;
};
static_assert(sizeof(DiskRecordV2) == 56U);

#pragma pack(push, 1)
struct DiskRayObservation {
    float endpoint_x;
    float endpoint_y;
    float endpoint_z;
    float origin_x;
    float origin_y;
    float origin_z;
    std::uint32_t return_count;
};
#pragma pack(pop)
static_assert(sizeof(DiskRayObservation) == 28U);

struct Accumulator {
    Vec3f xyz_sum = Vec3f::Zero();
    Vec3f origin_sum = Vec3f::Zero();
    Vec3f normal_sum = Vec3f::Zero();
    float intensity_sum = 0.0F;
    std::uint32_t count = 0U;
};

struct RayVoxelKey {
    VoxelKey endpoint;
    VoxelKey origin;
    bool operator==(const RayVoxelKey& other) const {
        return endpoint == other.endpoint && origin == other.origin;
    }
};

struct RayVoxelKeyHash {
    std::size_t operator()(const RayVoxelKey& key) const noexcept {
        std::size_t seed = VoxelKeyHash{}(key.endpoint);
        const std::size_t origin = VoxelKeyHash{}(key.origin);
        seed ^= origin + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

struct CellStatistics {
    Eigen::Vector3d xyz_sum = Eigen::Vector3d::Zero();
    Eigen::Matrix3d second_moment = Eigen::Matrix3d::Zero();
    Vec3f ordered_normal_sum = Vec3f::Zero();
    std::uint32_t count = 0U;

    void add(const Vec3f& point, const Vec3f& ordered_normal) {
        const Eigen::Vector3d precise = point.cast<double>();
        xyz_sum += precise;
        second_moment += precise * precise.transpose();
        ordered_normal_sum += ordered_normal;
        ++count;
    }
};

struct SurfaceModel {
    Vec3f center = Vec3f::Zero();
    Vec3f normal = Vec3f::UnitZ();
    float curvature = 0.5F;
    std::uint32_t support = 0U;
};

#pragma pack(push, 1)
struct NavVisSurfacePoint {
    float x;
    float y;
    float z;
    float intensity;
    float nx;
    float ny;
    float nz;
    float curvature;
};
#pragma pack(pop)
static_assert(sizeof(NavVisSurfacePoint) == 32U);

#pragma pack(push, 1)
struct InspectionPoint {
    float x;
    float y;
    float z;
    float intensity;
};
#pragma pack(pop)
static_assert(sizeof(InspectionPoint) == 16U);

#pragma pack(push, 1)
struct SurfaceInputDisk {
    float x;
    float y;
    float z;
    float origin_x;
    float origin_y;
    float origin_z;
    float intensity;
    float weight;
};
#pragma pack(pop)
static_assert(sizeof(SurfaceInputDisk) == 32U);

#pragma pack(push, 1)
struct CapturedSurfaceInputDisk {
    float field[12];
};
#pragma pack(pop)
static_assert(sizeof(CapturedSurfaceInputDisk) == 48U);

struct Options {
    fs::path input_directory;
    fs::path output;
    fs::path after_freespace_output;
    fs::path before_surface_output;
    fs::path work_directory;
    fs::path freespace_diagnostics;
    fs::path surface_diagnostics;
    fs::path surface_stage_counts;
    bool has_surface_diagnostic_tile = false;
    VoxelKey surface_diagnostic_tile{};
    float resolution = 0.01F;
    float coarse_cell = 0.025F;
    float normal_cell = 0.10F;
    float output_cell = 0.01F;
    float maximum_planar_curvature = 0.10F;
    std::uint32_t minimum_density_floor = 2U;
    std::uint32_t minimum_support = 4U;
    std::uint32_t minimum_normal_support = 6U;
    std::uint32_t minimum_hits = 1U;
    std::uint32_t tile_threads = 1U;
    std::uint32_t preprocess_threads = 8U;
    bool retain_merged = false;
    bool reuse_merged = false;
    bool inspection_only = false;
    bool adaptive_density = false;
    bool freespace_carving = false;
    bool global_ray_history = true;
    bool nonstandard_freespace = false;
    std::string freespace_mode = "sparse";
    navvis_recon::DirectionalFreespaceOptions freespace;
    navvis_recon::SparseFreespaceOptions sparse_freespace;
};

using Clock = std::chrono::steady_clock;

double secondsSince(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct ShardPreprocessResult {
    std::vector<navvis_recon::BinarySurfaceInput> surface_inputs;
    std::vector<VoxelKey> kept_freespace_leaves;
    std::vector<navvis_recon::SparseFreespaceLeafDiagnostic> leaf_diagnostics;
    VoxelKey diagnostic_shard{};
    std::uint64_t rejected_hits = 0U;
    std::uint64_t rejected_freespace = 0U;
    double seconds_read = 0.0;
    double seconds_candidate_build = 0.0;
    double seconds_freespace = 0.0;
    double seconds_surface_input = 0.0;
    navvis_recon::SparseFreespaceTiming sparse_timing;
};

struct SurfaceTileResult {
    bool processed = false;
    VoxelKey owner{};
    std::vector<NavVisSurfacePoint> points;
    fs::path after_freespace_records;
    fs::path before_surface_records;
    std::uint64_t after_freespace_count = 0U;
    std::uint64_t before_surface_count = 0U;
    navvis_recon::BinarySurfaceStageCounts counts;
    navvis_recon::BinaryOcclusionStageCounts occlusion_counts;
};

struct PreparedShards {
    bool has_ray_history = false;
    std::vector<fs::path> files;
};

void accumulateSurfaceCounts(navvis_recon::BinarySurfaceStageCounts& totals,
                             const navvis_recon::BinarySurfaceStageCounts& counts) {
    totals.input += counts.input;
    totals.normals += counts.normals;
    totals.selected += counts.selected;
    totals.valid += counts.valid;
    totals.output_voxels += counts.output_voxels;
    totals.density += counts.density;
    totals.adaptive_sor += counts.adaptive_sor;
    totals.post += counts.post;
    totals.seconds_normals += counts.seconds_normals;
    totals.seconds_selection += counts.seconds_selection;
    totals.seconds_output_voxels += counts.seconds_output_voxels;
    totals.seconds_density += counts.seconds_density;
    totals.seconds_adaptive_sor += counts.seconds_adaptive_sor;
    totals.seconds_support_pruning += counts.seconds_support_pruning;
    totals.seconds_post += counts.seconds_post;
}

void accumulateOcclusionCounts(navvis_recon::BinaryOcclusionStageCounts& totals,
                               const navvis_recon::BinaryOcclusionStageCounts& counts) {
    totals.raw_input += counts.raw_input;
    totals.helper_input += counts.helper_input;
    totals.helper_normals += counts.helper_normals;
    totals.helper_selected += counts.helper_selected;
    totals.helper_valid += counts.helper_valid;
    totals.helper_initial_voxels += counts.helper_initial_voxels;
    totals.helper_output += counts.helper_output;
    totals.kept += counts.kept;
}

Options parseArguments(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument("missing value after " + argument);
            }
            return argv[i];
        };
        if (argument == "--input-shards") {
            options.input_directory = value();
        } else if (argument == "--output") {
            options.output = value();
        } else if (argument == "--after-freespace-output") {
            options.after_freespace_output = value();
        } else if (argument == "--before-surface-output") {
            options.before_surface_output = value();
        } else if (argument == "--work-directory") {
            options.work_directory = value();
        } else if (argument == "--free-space-diagnostics") {
            options.freespace_diagnostics = value();
        } else if (argument == "--surface-diagnostics") {
            options.surface_diagnostics = value();
        } else if (argument == "--surface-stage-counts") {
            options.surface_stage_counts = value();
        } else if (argument == "--surface-diagnostic-tile") {
            const std::string encoded = value();
            if (std::sscanf(encoded.c_str(), "%d,%d,%d", &options.surface_diagnostic_tile.x,
                            &options.surface_diagnostic_tile.y,
                            &options.surface_diagnostic_tile.z) != 3) {
                throw std::invalid_argument("invalid --surface-diagnostic-tile: " + encoded);
            }
            options.has_surface_diagnostic_tile = true;
        } else if (argument == "--resolution") {
            options.resolution = std::stof(value());
        } else if (argument == "--coarse-cell") {
            options.coarse_cell = std::stof(value());
        } else if (argument == "--normal-cell") {
            options.normal_cell = std::stof(value());
        } else if (argument == "--output-cell") {
            options.output_cell = std::stof(value());
        } else if (argument == "--maximum-planar-curvature") {
            options.maximum_planar_curvature = std::stof(value());
        } else if (argument == "--minimum-support") {
            options.minimum_support = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--minimum-density-floor") {
            options.minimum_density_floor = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--minimum-normal-support") {
            options.minimum_normal_support = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--minimum-hits") {
            options.minimum_hits = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--tile-threads") {
            options.tile_threads = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--preprocess-threads") {
            options.preprocess_threads = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--retain-merged") {
            options.retain_merged = true;
        } else if (argument == "--reuse-merged") {
            options.reuse_merged = true;
        } else if (argument == "--inspection-only") {
            options.inspection_only = true;
        } else if (argument == "--adaptive-density") {
            options.adaptive_density = true;
        } else if (argument == "--free-space-carving") {
            options.freespace_carving = true;
        } else if (argument == "--free-space-local-rays") {
            options.global_ray_history = false;
        } else if (argument == "--free-space-nonstandard") {
            options.nonstandard_freespace = true;
        } else if (argument == "--free-space-mode") {
            options.freespace_mode = value();
        } else if (argument == "--ray-origin-cell") {
            options.freespace.origin_cell = std::stof(value());
        } else if (argument == "--ray-angular-bin-deg") {
            options.freespace.angular_bin_degrees = std::stof(value());
        } else if (argument == "--free-space-endpoint-margin") {
            options.freespace.endpoint_margin = std::stof(value());
            options.sparse_freespace.endpoint_margin = options.freespace.endpoint_margin;
        } else if (argument == "--free-space-min-intersections") {
            options.freespace.minimum_intersections =
                static_cast<std::uint32_t>(std::stoul(value()));
            options.sparse_freespace.minimum_intersections =
                options.freespace.minimum_intersections;
        } else if (argument == "--free-space-intersection-hit-ratio") {
            options.freespace.intersection_hit_ratio = std::stof(value());
            options.sparse_freespace.intersection_hit_ratio =
                options.freespace.intersection_hit_ratio;
        } else if (argument == "--free-space-traversal-resolution") {
            options.sparse_freespace.traversal_resolution = std::stof(value());
        } else if (argument == "--free-space-ray-radius") {
            options.sparse_freespace.ray_radius = std::stof(value());
        } else if (argument == "--free-space-ray-stride") {
            options.sparse_freespace.ray_stride = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--help") {
            std::cout
                << "Usage: navvis_recon_shard_surface_filter --input-shards DIR --output FILE "
                   "[--work-directory DIR] [--coarse-cell 0.025] "
                   "[--after-freespace-output FILE] [--before-surface-output FILE] "
                   "[--minimum-density-floor 2] [--minimum-support 4] "
                   "[--normal-cell 0.10] [--minimum-normal-support 6] "
                   "[--maximum-planar-curvature 0.10] [--output-cell 0.01] "
                   "[--minimum-hits 1] [--tile-threads 1] [--preprocess-threads 8] "
                   "[--retain-merged] [--reuse-merged] [--inspection-only] "
                   "[--adaptive-density] [--free-space-carving] "
                   "[--free-space-local-rays] "
                   "[--free-space-nonstandard] "
                   "[--free-space-mode sparse|directional] "
                   "[--ray-origin-cell 0.50] [--ray-angular-bin-deg 0.12] "
                   "[--free-space-traversal-resolution 0.02] "
                   "[--free-space-ray-radius 0.006] [--free-space-ray-stride 1] "
                   "[--free-space-endpoint-margin 0.05] "
                   "[--free-space-min-intersections 1] "
                   "[--free-space-intersection-hit-ratio 1.0]\n"
                   "                   [--free-space-diagnostics leaves.csv]\n"
                   "                   [--surface-diagnostics DIR]\n"
                   "                   [--surface-stage-counts counts.csv]\n"
                   "                   [--surface-diagnostic-tile x,y,z]\n"
                   "--inspection-only writes the two intermediate PLY files and preserves the "
                   "existing final Surface output.\n"
                   "Standard sparse mode fixes these values to the captured G11 binary "
                   "parameters; pass --free-space-nonstandard for diagnostic overrides.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.input_directory.empty() || options.output.empty()) {
        throw std::invalid_argument("--input-shards and --output are required");
    }
    if (!(options.resolution > 0.0F) || !(options.coarse_cell > 0.0F) ||
        !(options.normal_cell > 0.0F) || !(options.output_cell > 0.0F)) {
        throw std::invalid_argument("resolutions must be positive");
    }
    if (!(options.maximum_planar_curvature >= 0.0F) ||
        !(options.maximum_planar_curvature <= 1.0F)) {
        throw std::invalid_argument("maximum planar curvature must be in [0, 1]");
    }
    if (options.tile_threads == 0U || options.preprocess_threads == 0U) {
        throw std::invalid_argument("thread counts must be positive");
    }
    if (!(options.freespace.origin_cell > 0.0F) ||
        !(options.freespace.angular_bin_degrees > 0.0F) ||
        !(options.freespace.endpoint_margin >= 0.0F) ||
        !(options.freespace.intersection_hit_ratio >= 0.0F)) {
        throw std::invalid_argument("free-space options are invalid");
    }
    if ((options.freespace_mode != "sparse" && options.freespace_mode != "directional") ||
        !(options.sparse_freespace.traversal_resolution > 0.0F) ||
        !(options.sparse_freespace.ray_radius >= 0.0F) ||
        options.sparse_freespace.ray_stride == 0U) {
        throw std::invalid_argument("sparse free-space options are invalid");
    }
    if (options.work_directory.empty()) {
        options.work_directory = options.output.string() + ".surface_tmp";
    }
    if (options.after_freespace_output.empty()) {
        options.after_freespace_output =
            options.output.parent_path() / "pointcloud_after_freespace.ply";
    }
    if (options.before_surface_output.empty()) {
        options.before_surface_output =
            options.output.parent_path() / "pointcloud_before_surface.ply";
    }
    if (options.freespace_mode == "sparse" && !options.nonstandard_freespace) {
        // "sparse" is the G11/standard compatibility path.  Historical
        // runner flags are still accepted for command-line compatibility but
        // must not override parameters captured from the original binary.
        options.sparse_freespace = navvis_recon::SparseFreespaceOptions{};
    }
    return options;
}

VoxelKey coarseKey(const Vec3f& point, float cell, float shift) {
    return {static_cast<int>(std::floor((point.x() + shift) / cell)),
            static_cast<int>(std::floor((point.y() + shift) / cell)),
            static_cast<int>(std::floor((point.z() + shift) / cell))};
}

VoxelKey sparseFreespaceLeafKey(const Vec3f& point,
                                const navvis_recon::SparseFreespaceOptions& options) {
    const Vec3f anchor = options.has_grid_anchor ? options.grid_anchor : Vec3f::Zero();
    const double inverse = 1.0 / static_cast<double>(options.traversal_resolution);
    return {static_cast<int>(std::floor(
                (static_cast<double>(point.x()) - static_cast<double>(anchor.x())) * inverse)),
            static_cast<int>(std::floor(
                (static_cast<double>(point.y()) - static_cast<double>(anchor.y())) * inverse)),
            static_cast<int>(std::floor(
                (static_cast<double>(point.z()) - static_cast<double>(anchor.z())) * inverse))};
}

void mergeRecord(Accumulator& output, const DiskRecordV1& record) {
    output.xyz_sum += Vec3f(record.xyz_x, record.xyz_y, record.xyz_z);
    output.normal_sum += Vec3f(record.normal_x, record.normal_y, record.normal_z);
    output.intensity_sum += record.intensity;
    output.count += record.count;
}

void mergeRecord(Accumulator& output, const DiskRecordV2& record) {
    output.xyz_sum += Vec3f(record.xyz_x, record.xyz_y, record.xyz_z);
    output.origin_sum += Vec3f(record.origin_x, record.origin_y, record.origin_z);
    output.normal_sum += Vec3f(record.normal_x, record.normal_y, record.normal_z);
    output.intensity_sum += record.intensity;
    output.count += record.count;
}

DiskRecordV1 toRecord(const VoxelKey& key, const Accumulator& value) {
    return {key.x,
            key.y,
            key.z,
            value.xyz_sum.x(),
            value.xyz_sum.y(),
            value.xyz_sum.z(),
            value.normal_sum.x(),
            value.normal_sum.y(),
            value.normal_sum.z(),
            value.intensity_sum,
            value.count};
}

DiskRecordV2 toRayRecord(const VoxelKey& key, const Accumulator& value) {
    return {key.x,
            key.y,
            key.z,
            value.xyz_sum.x(),
            value.xyz_sum.y(),
            value.xyz_sum.z(),
            value.origin_sum.x(),
            value.origin_sum.y(),
            value.origin_sum.z(),
            value.normal_sum.x(),
            value.normal_sum.y(),
            value.normal_sum.z(),
            value.intensity_sum,
            value.count};
}

SurfaceModel surfaceModel(const CellStatistics& statistics, const Vec3f& orientation_hint) {
    SurfaceModel model;
    model.support = statistics.count;
    if (statistics.count < 3U) {
        model.normal = normalizedOr(orientation_hint, Vec3f::UnitZ());
        return model;
    }
    const float inverse = 1.0F / static_cast<float>(statistics.count);
    const Eigen::Vector3d mean = static_cast<double>(inverse) * statistics.xyz_sum;
    model.center = mean.cast<float>();
    const Eigen::Matrix3d covariance = inverse * statistics.second_moment - mean * mean.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success) {
        model.normal = normalizedOr(orientation_hint, Vec3f::UnitZ());
        return model;
    }
    model.normal = solver.eigenvectors().col(0).normalized().cast<float>();
    const Vec3f hint =
        normalizedOr(orientation_hint, normalizedOr(statistics.ordered_normal_sum, Vec3f::UnitZ()));
    if (model.normal.dot(hint) < 0.0F) {
        model.normal = -model.normal;
    }
    model.curvature = static_cast<float>(std::clamp(
        solver.eigenvalues().x() / std::max(solver.eigenvalues().sum(), 1.0e-12), 0.0, 1.0));
    return model;
}

std::vector<fs::path> shardFiles(const fs::path& directory, const std::string& extension) {
    std::vector<fs::path> result;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == extension) {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

constexpr float kSurfaceShardSize = 10.0F;
constexpr float kBinarySurfaceTileSize = 5.0F;
constexpr float kBinarySurfaceHalo = 0.15F;

VoxelKey spatialShardKey(const Vec3f& point) {
    return coarseKey(point, kSurfaceShardSize, 0.0F);
}

VoxelKey spatialShardKey(const fs::path& merged_path) {
    std::ifstream input(merged_path, std::ios::binary);
    DiskRecordV2 record{};
    while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
        if (record.count == 0U) {
            continue;
        }
        const float inverse = 1.0F / static_cast<float>(record.count);
        return spatialShardKey(
            Vec3f(inverse * record.xyz_x, inverse * record.xyz_y, inverse * record.xyz_z));
    }
    throw std::runtime_error("empty ray-history shard " + merged_path.string());
}

fs::path rayBucketPath(const fs::path& directory, const VoxelKey& key) {
    return directory / ("tile_" + std::to_string(key.x) + "_" + std::to_string(key.y) + "_" +
                        std::to_string(key.z) + ".raybucket");
}

fs::path candidateHaloBucketPath(const fs::path& directory, const VoxelKey& key) {
    return directory / ("tile_" + std::to_string(key.x) + "_" + std::to_string(key.y) + "_" +
                        std::to_string(key.z) + ".candidate_halo");
}

VoxelKey binarySurfaceTileKey(const Vec3f& point) {
    return coarseKey(point, kBinarySurfaceTileSize, 0.0F);
}

fs::path binarySurfaceTilePath(const fs::path& directory, const VoxelKey& key) {
    return directory / ("tile_" + std::to_string(key.x) + "_" + std::to_string(key.y) + "_" +
                        std::to_string(key.z) + ".surface_input");
}

VoxelKey binarySurfaceTileKey(const fs::path& path) {
    VoxelKey key{};
    if (std::sscanf(path.stem().string().c_str(), "tile_%d_%d_%d", &key.x, &key.y, &key.z) != 3) {
        throw std::runtime_error("invalid surface tile name " + path.string());
    }
    return key;
}

std::uint64_t tilecloudMortonKey(const VoxelKey& key) {
    // Tilecloud stores 21-bit signed tile coordinates after adding a 2^20
    // bias, then interleaves x/y/z into bits 0/1/2.  Its loader visits tiles
    // by this identifier; helper normal accumulation retains that order.
    constexpr std::int64_t bias = 1LL << 20;
    const auto spread = [](std::uint32_t value) {
        std::uint64_t output = 0U;
        for (unsigned bit = 0U; bit < 21U; ++bit) {
            output |= static_cast<std::uint64_t>((value >> bit) & 1U) << (3U * bit);
        }
        return output;
    };
    const auto biased = [&](int value) {
        const std::int64_t result = static_cast<std::int64_t>(value) + bias;
        if (result < 0 || result >= (1LL << 21)) {
            throw std::out_of_range("surface tile coordinate exceeds Tilecloud Morton range");
        }
        return static_cast<std::uint32_t>(result);
    };
    return spread(biased(key.x)) | (spread(biased(key.y)) << 1U) |
           (spread(biased(key.z)) << 2U);
}

void appendSurfaceInputs(const fs::path& directory,
                         const std::vector<navvis_recon::BinarySurfaceInput>& points) {
    std::unordered_map<VoxelKey, std::vector<SurfaceInputDisk>, VoxelKeyHash> buckets;
    for (const auto& point : points) {
        buckets[binarySurfaceTileKey(point.xyz)].push_back(
            {point.xyz.x(), point.xyz.y(), point.xyz.z(), point.origin.x(), point.origin.y(),
             point.origin.z(), point.intensity, point.weight});
    }
    for (const auto& bucket : buckets) {
        std::ofstream output(binarySurfaceTilePath(directory, bucket.first),
                             std::ios::binary | std::ios::app);
        if (!output) {
            throw std::runtime_error("cannot append surface input tile");
        }
        output.write(reinterpret_cast<const char*>(bucket.second.data()),
                     static_cast<std::streamsize>(bucket.second.size() * sizeof(SurfaceInputDisk)));
    }
}

std::vector<std::vector<navvis_recon::BinarySurfaceInput>>
loadAllSurfaceTilesWithHalo(const fs::path& directory, const std::vector<fs::path>& tile_paths) {
    // The former per-tile loader opened all 3x3x3 neighbour files and scanned
    // every record before retaining the 15 cm border.  On the canonical G11
    // run that turned 2.39 GiB / 80.2 M source records into 52.61 GiB / 1.765
    // B examined records.  Build the same ordered halo views in one source
    // pass instead.  A point can belong to at most two 5 m tiles per axis,
    // because the halo is much smaller than the tile.
    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> target_indices;
    target_indices.reserve(tile_paths.size());
    std::vector<std::vector<navvis_recon::BinarySurfaceInput>> result(tile_paths.size());
    for (std::size_t index = 0U; index < tile_paths.size(); ++index) {
        const VoxelKey key = binarySurfaceTileKey(tile_paths[index]);
        target_indices.emplace(key, index);
        const std::uintmax_t bytes = fs::file_size(tile_paths[index]);
        const std::size_t core_records = static_cast<std::size_t>(bytes / sizeof(SurfaceInputDisk));
        result[index].reserve(core_records + core_records / 4U);
    }

    struct SourceTile {
        VoxelKey key;
        fs::path path;
    };
    std::vector<SourceTile> source_tiles;
    source_tiles.reserve(tile_paths.size());
    for (const auto& path : tile_paths) {
        source_tiles.push_back({binarySurfaceTileKey(path), path});
    }
    std::sort(source_tiles.begin(), source_tiles.end(), [](const auto& first, const auto& second) {
        return tilecloudMortonKey(first.key) < tilecloudMortonKey(second.key);
    });

    std::uint64_t source_records = 0U;
    std::uint64_t expanded_records = 0U;
    std::size_t source_index = 0U;
    for (const auto& source : source_tiles) {
        std::ifstream input(source.path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot read surface input tile " + source.path.string());
        }
        SurfaceInputDisk point{};
        while (input.read(reinterpret_cast<char*>(&point), sizeof(point))) {
            ++source_records;
            const Vec3f xyz(point.x, point.y, point.z);
            const VoxelKey core = binarySurfaceTileKey(xyz);
            std::array<std::array<int, 3>, 3> axis_keys{};
            std::array<std::size_t, 3> axis_counts{};
            for (int axis = 0; axis < 3; ++axis) {
                const int core_key = axis == 0 ? core.x : (axis == 1 ? core.y : core.z);
                const float coordinate = xyz[axis];
                for (int candidate = core_key - 1; candidate <= core_key + 1; ++candidate) {
                    const float minimum =
                        kBinarySurfaceTileSize * static_cast<float>(candidate) - kBinarySurfaceHalo;
                    const float maximum = kBinarySurfaceTileSize * static_cast<float>(candidate) +
                                          kBinarySurfaceTileSize + kBinarySurfaceHalo;
                    if (coordinate >= minimum && coordinate <= maximum) {
                        axis_keys[axis][axis_counts[axis]++] = candidate;
                    }
                }
            }
            const navvis_recon::BinarySurfaceInput converted{
                xyz, Vec3f(point.origin_x, point.origin_y, point.origin_z), point.intensity,
                point.weight};
            for (std::size_t x = 0U; x < axis_counts[0]; ++x) {
                for (std::size_t y = 0U; y < axis_counts[1]; ++y) {
                    for (std::size_t z = 0U; z < axis_counts[2]; ++z) {
                        const VoxelKey target_key{axis_keys[0][x], axis_keys[1][y],
                                                  axis_keys[2][z]};
                        const auto target = target_indices.find(target_key);
                        if (target == target_indices.end()) {
                            continue;
                        }
                        result[target->second].push_back(converted);
                        ++expanded_records;
                    }
                }
            }
        }
        std::cerr << "Prepared halo source " << ++source_index << '/' << source_tiles.size()
                  << " tiles\r";
    }
    std::cerr << "\nPrepared " << result.size() << " ordered in-memory halo tiles from "
              << source_records << " core records (" << expanded_records
              << " records including true boundary copies)\n";
    return result;
}

bool segmentIntersectsBox(const Vec3f& origin, const Vec3f& direction, const Vec3f& minimum,
                          const Vec3f& maximum, float entry, float exit) {
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(direction[axis]) < 1.0e-9F) {
            if (origin[axis] < minimum[axis] || origin[axis] > maximum[axis]) {
                return false;
            }
            continue;
        }
        float first = (minimum[axis] - origin[axis]) / direction[axis];
        float second = (maximum[axis] - origin[axis]) / direction[axis];
        if (first > second) {
            std::swap(first, second);
        }
        entry = std::max(entry, first);
        exit = std::min(exit, second);
        if (entry > exit) {
            return false;
        }
    }
    return true;
}

void buildGlobalRayBuckets(const std::vector<fs::path>& merged_files,
                           const fs::path& bucket_directory,
                           const fs::path& candidate_halo_directory,
                           navvis_recon::SparseFreespaceOptions& options) {
    fs::remove_all(bucket_directory);
    fs::create_directories(bucket_directory);
    fs::remove_all(candidate_halo_directory);
    fs::create_directories(candidate_halo_directory);

    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> targets;
    targets.reserve(merged_files.size());
    std::vector<std::ofstream> buckets;
    buckets.reserve(merged_files.size());
    std::vector<std::ofstream> candidate_halos;
    candidate_halos.reserve(merged_files.size());
    for (const auto& merged_path : merged_files) {
        const VoxelKey key = spatialShardKey(merged_path);
        const auto inserted = targets.emplace(key, buckets.size());
        if (!inserted.second) {
            throw std::runtime_error("multiple merged files occupy one 10 m surface shard");
        }
        buckets.emplace_back(rayBucketPath(bucket_directory, key), std::ios::binary);
        if (!buckets.back()) {
            throw std::runtime_error("cannot create global ray bucket");
        }
        candidate_halos.emplace_back(candidateHaloBucketPath(candidate_halo_directory, key),
                                     std::ios::binary);
        if (!candidate_halos.back()) {
            throw std::runtime_error("cannot create free-space candidate halo bucket");
        }
    }

    std::uint64_t input_rays = 0U;
    std::uint64_t bucket_copies = 0U;
    std::size_t source_index = 0U;
    VoxelKey minimum_tile{};
    VoxelKey maximum_tile{};
    bool has_tile_bounds = false;
    for (const auto& merged_path : merged_files) {
        std::ifstream input(merged_path, std::ios::binary);
        DiskRecordV2 record{};
        while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
            if (record.count == 0U) {
                continue;
            }
            ++input_rays;
            const float inverse = 1.0F / static_cast<float>(record.count);
            const Vec3f endpoint(inverse * record.xyz_x, inverse * record.xyz_y,
                                 inverse * record.xyz_z);
            const Vec3f origin(inverse * record.origin_x, inverse * record.origin_y,
                               inverse * record.origin_z);
            const VoxelKey tilecloud_tile = coarseKey(endpoint, kBinarySurfaceTileSize, 0.0F);
            if (!has_tile_bounds) {
                minimum_tile = maximum_tile = tilecloud_tile;
                has_tile_bounds = true;
            } else {
                minimum_tile.x = std::min(minimum_tile.x, tilecloud_tile.x);
                minimum_tile.y = std::min(minimum_tile.y, tilecloud_tile.y);
                minimum_tile.z = std::min(minimum_tile.z, tilecloud_tile.z);
                maximum_tile.x = std::max(maximum_tile.x, tilecloud_tile.x);
                maximum_tile.y = std::max(maximum_tile.y, tilecloud_tile.y);
                maximum_tile.z = std::max(maximum_tile.z, tilecloud_tile.z);
            }
            const VoxelKey source_shard = spatialShardKey(endpoint);
            // The original computes +/-2-leaf PCA normals in one global
            // CompactOctree.  Copy only the thin 10 m boundary halo needed by
            // that query so streaming shards see identical neighbours and a
            // 2 cm leaf split by a shard plane is aggregated as one leaf.
            const float candidate_halo_width = 3.0F * options.traversal_resolution;
            const VoxelKey halo_minimum =
                spatialShardKey(endpoint - Vec3f::Constant(candidate_halo_width));
            const VoxelKey halo_maximum =
                spatialShardKey(endpoint + Vec3f::Constant(candidate_halo_width));
            for (int x = halo_minimum.x; x <= halo_maximum.x; ++x) {
                for (int y = halo_minimum.y; y <= halo_maximum.y; ++y) {
                    for (int z = halo_minimum.z; z <= halo_maximum.z; ++z) {
                        const VoxelKey target_key{x, y, z};
                        if (target_key == source_shard) {
                            continue;
                        }
                        const auto target = targets.find(target_key);
                        if (target == targets.end()) {
                            continue;
                        }
                        candidate_halos[target->second].write(
                            reinterpret_cast<const char*>(&record), sizeof(record));
                    }
                }
            }
            const Vec3f ray = endpoint - origin;
            // Match FreespaceOctree's observable SSE reduction order.  This
            // also keeps the streaming ray buckets consistent with the exact
            // shortened segment used by the sparse carving worker.
            float length_squared = ray.y() * ray.y();
            length_squared += ray.z() * ray.z();
            length_squared += ray.x() * ray.x();
            const float length = std::sqrt(length_squared);
            if (!std::isfinite(length) || length <= options.endpoint_margin) {
                continue;
            }
            const float entry = options.minimum_origin_distance;
            const float exit =
                std::min(length - options.endpoint_margin, options.maximum_origin_distance);
            if (!(entry < exit)) {
                continue;
            }
            const float inverse_length = 1.0F / length;
            const Vec3f direction = inverse_length * ray;
            const Vec3f first = origin + entry * direction;
            const Vec3f last = origin + exit * direction;
            // A 2 cm CompactOctree leaf can straddle a 10 m streaming-shard
            // boundary because its lattice is anchored to the first point.
            // A shard therefore needs rays crossing the complete boundary
            // leaf, plus the subsequent 6 mm centroid-distance allowance.
            const float shard_padding = options.traversal_resolution + options.ray_radius;
            const Vec3f lower = first.cwiseMin(last) - Vec3f::Constant(shard_padding);
            const Vec3f upper = first.cwiseMax(last) + Vec3f::Constant(shard_padding);
            const VoxelKey minimum_key = spatialShardKey(lower);
            const VoxelKey maximum_key = spatialShardKey(upper);
            const DiskRayObservation output_record{endpoint.x(), endpoint.y(), endpoint.z(),
                                                   origin.x(),   origin.y(),   origin.z(),
                                                   record.count};
            for (int x = minimum_key.x; x <= maximum_key.x; ++x) {
                for (int y = minimum_key.y; y <= maximum_key.y; ++y) {
                    for (int z = minimum_key.z; z <= maximum_key.z; ++z) {
                        const VoxelKey key{x, y, z};
                        const auto target = targets.find(key);
                        if (target == targets.end()) {
                            continue;
                        }
                        const Vec3f box_min =
                            kSurfaceShardSize * Vec3f(static_cast<float>(x), static_cast<float>(y),
                                                      static_cast<float>(z)) -
                            Vec3f::Constant(shard_padding);
                        const Vec3f box_max =
                            box_min + Vec3f::Constant(kSurfaceShardSize + 2.0F * shard_padding);
                        if (!segmentIntersectsBox(origin, direction, box_min, box_max, entry,
                                                  exit)) {
                            continue;
                        }
                        buckets[target->second].write(reinterpret_cast<const char*>(&output_record),
                                                      sizeof(output_record));
                        ++bucket_copies;
                    }
                }
            }
        }
        std::cerr << "Indexed global rays " << ++source_index << '/' << merged_files.size()
                  << " shards\r";
    }
    if (!has_tile_bounds || !options.has_grid_anchor) {
        throw std::runtime_error("global sparse free-space bounds require points and an anchor");
    }
    const Eigen::Vector3d bounding_minimum =
        kBinarySurfaceTileSize *
        Eigen::Vector3d(minimum_tile.x, minimum_tile.y, minimum_tile.z);
    const Eigen::Vector3d bounding_maximum =
        kBinarySurfaceTileSize *
        Eigen::Vector3d(maximum_tile.x + 1, maximum_tile.y + 1, maximum_tile.z + 1);
    const double resolution = static_cast<double>(options.traversal_resolution);
    Eigen::Vector3d center = 0.5 * (bounding_minimum + bounding_maximum);
    for (int axis = 0; axis < 3; ++axis) {
        center[axis] -= std::fmod(center[axis], resolution);
        center[axis] +=
            std::fmod(static_cast<double>(options.grid_anchor[axis]), resolution);
    }
    int root_size = 2;
    double root_width = 2.0 * resolution;
    while (((center - Eigen::Vector3d::Constant(0.5 * root_width)).array() >
            bounding_minimum.array())
               .any() ||
           ((center + Eigen::Vector3d::Constant(0.5 * root_width)).array() <
            bounding_maximum.array())
               .any()) {
        root_size *= 2;
        root_width *= 2.0;
    }
    const Eigen::Vector3d root_minimum =
        center - Eigen::Vector3d::Constant(0.5 * root_width);
    const Eigen::Vector3d anchor = options.grid_anchor.cast<double>();
    const Eigen::Vector3d root_key = (root_minimum - anchor) / resolution;
    options.grid_root_lower = {
        static_cast<int>(std::llround(root_key.x())),
        static_cast<int>(std::llround(root_key.y())),
        static_cast<int>(std::llround(root_key.z()))};
    options.grid_root_size = root_size;
    options.has_grid_root = true;
    std::cerr << "Using binary-compatible CompactOctree root lower "
              << options.grid_root_lower.x << ',' << options.grid_root_lower.y << ','
              << options.grid_root_lower.z << " size " << options.grid_root_size << '\n';
    std::cerr << "\nIndexed " << input_rays << " captured ray records into " << bucket_copies
              << " occupied-shard ray copies\n";
}

PreparedShards prepareInputShards(const Options& options, const fs::path& merged_directory) {
    const auto ray_inputs = shardFiles(options.input_directory, ".raytile");
    const auto legacy_inputs = shardFiles(options.input_directory, ".tile");
    if (!ray_inputs.empty() && !legacy_inputs.empty()) {
        throw std::runtime_error("mixed .tile and .raytile inputs are not supported");
    }

    PreparedShards prepared;
    prepared.has_ray_history = !ray_inputs.empty();
    const auto& inputs = prepared.has_ray_history ? ray_inputs : legacy_inputs;
    if (inputs.empty()) {
        throw std::runtime_error("no .tile or .raytile shards in " +
                                 options.input_directory.string());
    }
    if (options.freespace_carving && !prepared.has_ray_history) {
        throw std::runtime_error(
            "--free-space-carving requires version-2 .raytile inputs with sensor origins");
    }

    std::uint64_t raw_records = 0U;
    if (prepared.has_ray_history) {
        // Ray-history shards already contain the final clusters in stable input order.
        prepared.files = inputs;
        for (const auto& input_path : prepared.files) {
            const auto bytes = fs::file_size(input_path);
            if ((bytes % sizeof(DiskRecordV2)) != 0U) {
                throw std::runtime_error("truncated ray-history shard: " + input_path.string());
            }
            raw_records += bytes / sizeof(DiskRecordV2);
        }
        std::cerr << "Using " << prepared.files.size() << " ray-history shards directly with "
                  << raw_records << " records\n";
        return prepared;
    }

    std::uint64_t occupied_voxels = 0U;
    const std::string merged_extension = ".merged";
    prepared.files = shardFiles(merged_directory, merged_extension);
    if (options.reuse_merged && !prepared.files.empty()) {
        for (const auto& merged_path : prepared.files) {
            occupied_voxels += fs::file_size(merged_path) / sizeof(DiskRecordV1);
        }
        std::cerr << "Reusing " << prepared.files.size() << " merged shards with "
                  << occupied_voxels << " occupied voxels\n";
        return prepared;
    }

    std::size_t merged_index = 0U;
    for (const auto& input_path : inputs) {
        const fs::path merged_path =
            merged_directory / (input_path.stem().string() + merged_extension);
        ++merged_index;
        std::ofstream merged(merged_path, std::ios::binary);
        std::unordered_map<VoxelKey, Accumulator, VoxelKeyHash> voxels;
        std::ifstream input(input_path, std::ios::binary);
        DiskRecordV1 record{};
        while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
            ++raw_records;
            mergeRecord(voxels[VoxelKey{record.key_x, record.key_y, record.key_z}], record);
        }
        for (const auto& item : voxels) {
            const DiskRecordV1 output_record = toRecord(item.first, item.second);
            merged.write(reinterpret_cast<const char*>(&output_record), sizeof(output_record));
            if (item.second.count != 0U) {
                ++occupied_voxels;
            }
        }
        std::cerr << "Merged " << merged_index << '/' << inputs.size() << " shards\r";
    }
    std::cerr << "\nMerged " << raw_records << " records into " << occupied_voxels
              << " occupied voxels\n";
    prepared.files = shardFiles(merged_directory, merged_extension);
    return prepared;
}

Options loadRuntimeOptions(const Options& options, bool has_ray_history) {
    Options runtime_options = options;
    const fs::path anchor_path = options.input_directory / "freespace_anchor.txt";
    if (!has_ray_history || !fs::is_regular_file(anchor_path)) {
        return runtime_options;
    }

    std::ifstream anchor(anchor_path);
    if (!(anchor >> runtime_options.sparse_freespace.grid_anchor.x() >>
          runtime_options.sparse_freespace.grid_anchor.y() >>
          runtime_options.sparse_freespace.grid_anchor.z())) {
        throw std::runtime_error("invalid free-space anchor: " + anchor_path.string());
    }
    runtime_options.sparse_freespace.has_grid_anchor = true;
    std::cerr << "Using binary-compatible CompactOctree anchor "
              << runtime_options.sparse_freespace.grid_anchor.transpose() << '\n';
    return runtime_options;
}

std::ofstream openFreespaceDiagnostics(const fs::path& output_path) {
    std::ofstream diagnostics;
    if (output_path.empty()) {
        return diagnostics;
    }
    if (!output_path.parent_path().empty()) {
        fs::create_directories(output_path.parent_path());
    }
    diagnostics.open(output_path);
    if (!diagnostics) {
        throw std::runtime_error("cannot create free-space diagnostics: " + output_path.string());
    }
    diagnostics << "key_x,key_y,key_z,centroid_x,centroid_y,centroid_z,normal_x,normal_y,normal_z,"
                   "hits,intersections,removed\n";
    diagnostics.precision(9);
    return diagnostics;
}

ShardPreprocessResult preprocessShard(const fs::path& merged_path, bool has_ray_history,
                                      bool use_global_sparse_rays,
                                      const fs::path& ray_bucket_directory,
                                      const fs::path& candidate_halo_directory,
                                      const Options& options, const Options& runtime_options,
                                      bool emit_freespace_diagnostics) {
    ShardPreprocessResult result;
    auto subphase_started = Clock::now();
    result.diagnostic_shard = spatialShardKey(merged_path);

    std::vector<DiskRecordV1> records;
    std::vector<DiskRecordV2> raw_ray_records;
    std::vector<DiskRecordV2> freespace_candidate_records;
    std::vector<navvis_recon::FreespaceRayObservation> ray_observations;
    std::vector<std::uint8_t> sparse_raw_removed;
    if (has_ray_history) {
        const std::size_t ray_count =
            static_cast<std::size_t>(fs::file_size(merged_path) / sizeof(DiskRecordV2));
        raw_ray_records.reserve(ray_count);
        ray_observations.reserve(ray_count);
        std::ifstream input(merged_path, std::ios::binary);
        DiskRecordV2 ray_record{};
        while (input.read(reinterpret_cast<char*>(&ray_record), sizeof(ray_record))) {
            if (ray_record.count == 0U) {
                continue;
            }
            raw_ray_records.push_back(ray_record);
            const VoxelKey endpoint{ray_record.key_x, ray_record.key_y, ray_record.key_z};
            const float inverse = 1.0F / static_cast<float>(ray_record.count);
            if (!use_global_sparse_rays) {
                ray_observations.push_back(
                    {endpoint,
                     Vec3f(inverse * ray_record.xyz_x, inverse * ray_record.xyz_y,
                           inverse * ray_record.xyz_z),
                     Vec3f(inverse * ray_record.origin_x, inverse * ray_record.origin_y,
                           inverse * ray_record.origin_z),
                     ray_record.count});
            }
        }
        if (use_global_sparse_rays) {
            const VoxelKey shard_key = spatialShardKey(merged_path);
            std::ifstream rays(rayBucketPath(ray_bucket_directory, shard_key), std::ios::binary);
            DiskRayObservation ray{};
            while (rays.read(reinterpret_cast<char*>(&ray), sizeof(ray))) {
                const Vec3f endpoint(ray.endpoint_x, ray.endpoint_y, ray.endpoint_z);
                ray_observations.push_back({coarseKey(endpoint, options.resolution, 0.0F), endpoint,
                                            Vec3f(ray.origin_x, ray.origin_y, ray.origin_z),
                                            ray.return_count});
            }
            std::ifstream halo(candidateHaloBucketPath(candidate_halo_directory, shard_key),
                               std::ios::binary);
            while (halo.read(reinterpret_cast<char*>(&ray_record), sizeof(ray_record))) {
                if (ray_record.count != 0U) {
                    freespace_candidate_records.push_back(ray_record);
                }
            }
        }
    } else {
        records.reserve(
            static_cast<std::size_t>(fs::file_size(merged_path) / sizeof(DiskRecordV1)));
        std::ifstream input(merged_path, std::ios::binary);
        DiskRecordV1 record{};
        while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
            records.push_back(record);
        }
    }
    result.seconds_read = secondsSince(subphase_started);

    std::unordered_map<VoxelKey, navvis_recon::DirectionalFreespaceEvidence, VoxelKeyHash>
        freespace_evidence;
    if (options.freespace_carving) {
        if (options.freespace_mode == "directional") {
            subphase_started = Clock::now();
            freespace_evidence = navvis_recon::DirectionalFreespaceCarver::compute(
                ray_observations, options.freespace);
            result.seconds_freespace = secondsSince(subphase_started);
        } else {
            subphase_started = Clock::now();
            std::vector<navvis_recon::FreespaceCandidate> candidates;
            candidates.reserve(raw_ray_records.size() + freespace_candidate_records.size());
            const auto append_candidates = [&](const auto& candidate_records) {
                for (const auto& candidate : candidate_records) {
                    if (candidate.count == 0U) {
                        continue;
                    }
                    const float inverse = 1.0F / static_cast<float>(candidate.count);
                    candidates.push_back(
                        {VoxelKey{candidate.key_x, candidate.key_y, candidate.key_z},
                         Vec3f(inverse * candidate.xyz_x, inverse * candidate.xyz_y,
                               inverse * candidate.xyz_z),
                         normalizedOr(
                             Vec3f(candidate.normal_x, candidate.normal_y, candidate.normal_z),
                             Vec3f::Zero()),
                         candidate.count});
                }
            };
            append_candidates(raw_ray_records);
            if (use_global_sparse_rays) {
                append_candidates(freespace_candidate_records);
            }
            result.seconds_candidate_build = secondsSince(subphase_started);

            subphase_started = Clock::now();
            freespace_evidence = navvis_recon::SparseFreespaceCarver::compute(
                ray_observations, candidates, runtime_options.sparse_freespace,
                emit_freespace_diagnostics ? &result.leaf_diagnostics : nullptr, nullptr, false,
                &sparse_raw_removed, &result.sparse_timing);
            std::unordered_set<VoxelKey, VoxelKeyHash> kept_leaves;
            kept_leaves.reserve(raw_ray_records.size() / 4U + 1U);
            for (std::size_t index = 0; index < raw_ray_records.size(); ++index) {
                if (index < sparse_raw_removed.size() && sparse_raw_removed[index] != 0U) {
                    ++result.rejected_freespace;
                    continue;
                }
                const auto& record = raw_ray_records[index];
                const float inverse = 1.0F / static_cast<float>(record.count);
                const Vec3f point(inverse * record.xyz_x, inverse * record.xyz_y,
                                  inverse * record.xyz_z);
                kept_leaves.insert(
                    sparseFreespaceLeafKey(point, runtime_options.sparse_freespace));
            }
            result.kept_freespace_leaves.assign(kept_leaves.begin(), kept_leaves.end());
            // Sparse removal has already been applied point by point.
            freespace_evidence.clear();
            result.seconds_freespace = secondsSince(subphase_started);
        }
    }

    subphase_started = Clock::now();
    if (has_ray_history) {
        // Preserve raw rays until clean-occlusions, then aggregate at the requested resolution.
        result.surface_inputs.reserve(raw_ray_records.size());
        for (std::size_t index = 0U; index < raw_ray_records.size(); ++index) {
            const auto& record = raw_ray_records[index];
            if (record.count < options.minimum_hits) {
                ++result.rejected_hits;
                continue;
            }
            // CompactOccupancy is applied after the helper's 2 cm aggregation,
            // not to the raw helper source.  Keep sparse-mode raw points here;
            // processSurfaceTile filters the helper centroids and classifier
            // rays independently in the original order.
            if (options.freespace_mode != "sparse" && index < sparse_raw_removed.size() &&
                sparse_raw_removed[index] != 0U) {
                continue;
            }
            const VoxelKey endpoint{record.key_x, record.key_y, record.key_z};
            const auto directional = freespace_evidence.find(endpoint);
            if (directional != freespace_evidence.end() && directional->second.removed) {
                ++result.rejected_freespace;
                continue;
            }
            const float inverse = 1.0F / static_cast<float>(record.count);
            result.surface_inputs.push_back(
                {inverse * Vec3f(record.xyz_x, record.xyz_y, record.xyz_z),
                 inverse * Vec3f(record.origin_x, record.origin_y, record.origin_z),
                 std::clamp(inverse * record.intensity, 0.0F, 1.0F),
                 static_cast<float>(record.count)});
        }
    } else {
        result.surface_inputs.reserve(records.size());
        for (const auto& record : records) {
            if (record.count < options.minimum_hits) {
                ++result.rejected_hits;
                continue;
            }
            const float inverse = 1.0F / static_cast<float>(record.count);
            const Vec3f point = inverse * Vec3f(record.xyz_x, record.xyz_y, record.xyz_z);
            const Vec3f ordered_normal =
                normalizedOr(Vec3f(record.normal_x, record.normal_y, record.normal_z));
            result.surface_inputs.push_back({point, point - ordered_normal,
                                             std::clamp(inverse * record.intensity, 0.0F, 1.0F),
                                             static_cast<float>(record.count)});
        }
    }
    result.seconds_surface_input = secondsSince(subphase_started);
    return result;
}

navvis_recon::BinarySurfaceOptions makeSurfaceOptions(const Options& options) {
    navvis_recon::BinarySurfaceOptions surface_options;
    surface_options.output_resolution = options.output_cell;
    surface_options.output_merge_distance = options.output_cell;
    return surface_options;
}

SurfaceTileResult processSurfaceTile(const fs::path& surface_tile,
                                     std::vector<navvis_recon::BinarySurfaceInput> halo_rays,
                                     bool has_ray_history, const Options& options,
                                     const fs::path& after_freespace_record_directory,
                                     const fs::path& before_surface_record_directory,
                                     const std::unordered_set<VoxelKey, VoxelKeyHash>*
                                         kept_freespace_leaves,
                                     const navvis_recon::SparseFreespaceOptions&
                                         sparse_freespace_options) {
    SurfaceTileResult result;
    result.processed = true;
    const VoxelKey owner = binarySurfaceTileKey(surface_tile);
    result.owner = owner;
    const Vec3f core_minimum =
        kBinarySurfaceTileSize * Vec3f(static_cast<float>(owner.x), static_cast<float>(owner.y),
                                       static_cast<float>(owner.z));
    const Vec3f core_maximum = core_minimum + Vec3f::Constant(kBinarySurfaceTileSize);

    const auto write_inspection_points = [&](const fs::path& directory,
                                             const std::string& stage,
                                             const std::vector<navvis_recon::BinarySurfaceInput>&
                                                 input,
                                             fs::path& output_path) {
        output_path = directory /
                      ("tile_" + std::to_string(owner.x) + "_" + std::to_string(owner.y) +
                       "_" + std::to_string(owner.z) + ".inspection");
        std::ofstream output(output_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create " + stage + " inspection records " +
                                     output_path.string());
        }
        std::uint64_t count = 0U;
        for (const auto& point : input) {
            if ((point.xyz.array() < core_minimum.array()).any() ||
                (point.xyz.array() >= core_maximum.array()).any()) {
                continue;
            }
            const InspectionPoint record{point.xyz.x(), point.xyz.y(), point.xyz.z(),
                                         point.intensity};
            output.write(reinterpret_cast<const char*>(&record), sizeof(record));
            ++count;
        }
        if (!output) {
            throw std::runtime_error("cannot write " + stage + " inspection records " +
                                     output_path.string());
        }
        return count;
    };

    const bool diagnose_tile =
        !options.surface_diagnostics.empty() &&
        (!options.has_surface_diagnostic_tile || owner == options.surface_diagnostic_tile);
    const auto dump_inputs = [&](const std::string& stage,
                                 const std::vector<navvis_recon::BinarySurfaceInput>& input) {
        if (!diagnose_tile) {
            return;
        }
        const fs::path output_path =
            options.surface_diagnostics /
            ("tile_" + std::to_string(owner.x) + "_" + std::to_string(owner.y) + "_" +
             std::to_string(owner.z) + "_" + stage + "48.bin");
        std::ofstream output(output_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create surface diagnostic " +
                                     output_path.string());
        }
        for (const auto& point : input) {
            const CapturedSurfaceInputDisk record{{
                point.origin.x(), point.origin.y(), point.origin.z(), 0.0F,
                point.xyz.x(), point.xyz.y(), point.xyz.z(), 0.0F,
                point.intensity, point.weight, 0.0F, 0.0F,
            }};
            output.write(reinterpret_cast<const char*>(&record), sizeof(record));
        }
    };

    std::vector<navvis_recon::BinarySurfaceInput> halo_input;
    if (has_ray_history) {
        auto helper_input = navvis_recon::applyBinaryOcclusionHelperInputAggregation(
            halo_rays, 0.02F);
        std::vector<navvis_recon::BinarySurfaceInput> classifier_rays;
        if (kept_freespace_leaves != nullptr) {
            helper_input.erase(
                std::remove_if(helper_input.begin(), helper_input.end(), [&](const auto& point) {
                    return kept_freespace_leaves->find(
                               sparseFreespaceLeafKey(point.xyz, sparse_freespace_options)) ==
                           kept_freespace_leaves->end();
                }),
                helper_input.end());
            classifier_rays.reserve(halo_rays.size());
            for (const auto& ray : halo_rays) {
                if (kept_freespace_leaves->find(
                        sparseFreespaceLeafKey(ray.xyz, sparse_freespace_options)) !=
                    kept_freespace_leaves->end()) {
                    classifier_rays.push_back(ray);
                }
            }
        } else {
            classifier_rays = std::move(halo_rays);
        }
        {
            // Free-space decisions use the captured 2 cm CompactOccupancy grid, but the
            // inspection cloud must remain directly comparable with the requested Surface
            // resolution.  Aggregate the retained classifier rays at that resolution before
            // occlusion cleaning instead of exposing the coarser helper surface.
            const auto after_freespace_input =
                navvis_recon::applyBinaryOcclusionHelperInputAggregation(classifier_rays,
                                                                          options.resolution);
            result.after_freespace_count = write_inspection_points(
                after_freespace_record_directory, "after-free-space", after_freespace_input,
                result.after_freespace_records);
        }
        dump_inputs("raw", classifier_rays);
        dump_inputs("helper_input", helper_input);
        const auto cleaned_rays = navvis_recon::applyBinaryOcclusionCleaningFromHelperInput(
            classifier_rays, helper_input, nullptr, {}, {}, &result.occlusion_counts);
        dump_inputs("kept", cleaned_rays);
        halo_input = navvis_recon::applyBinaryOcclusionHelperInputAggregation(cleaned_rays,
                                                                              options.resolution);
    } else {
        halo_input = std::move(halo_rays);
        const auto after_freespace_input =
            navvis_recon::applyBinaryOcclusionHelperInputAggregation(halo_input,
                                                                      options.resolution);
        result.after_freespace_count = write_inspection_points(
            after_freespace_record_directory, "after-free-space", after_freespace_input,
            result.after_freespace_records);
    }

    dump_inputs("main", halo_input);
    result.before_surface_count = write_inspection_points(
        before_surface_record_directory, "before-surface", halo_input,
        result.before_surface_records);
    if (options.inspection_only) {
        return result;
    }

    const auto points = navvis_recon::runBinarySurfacePipeline(
        halo_input, makeSurfaceOptions(options), &result.counts);
    result.points.reserve(points.size());
    for (const auto& point : points) {
        if ((point.xyz.array() < core_minimum.array()).any() ||
            (point.xyz.array() >= core_maximum.array()).any()) {
            continue;
        }
        result.points.push_back({point.xyz.x(), point.xyz.y(), point.xyz.z(), point.intensity,
                                 point.normal.x(), point.normal.y(), point.normal.z(),
                                 point.curvature});
    }
    return result;
}

void writePly(const fs::path& output_path, const fs::path& records, std::uint64_t count) {
    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write " + output_path.string());
    }
    output << "ply\nformat binary_little_endian 1.0\n"
           << "comment navvis_recon G11 standard surface reconstruction\n"
           << "obj_info num_cols ";
    output.width(10);
    output.fill('0');
    output << count << "\nobj_info num_rows 0000000001\n"
           << "element vertex ";
    output.width(10);
    output.fill('0');
    output << count << "\n"
           << "property float x\nproperty float y\nproperty float z\n"
           << "property float intensity\nproperty float nx\nproperty float ny\n"
           << "property float nz\nproperty float curvature\nend_header\n";
    std::ifstream input(records, std::ios::binary);
    output << input.rdbuf();
}

void writeInspectionPly(const fs::path& output_path, const std::string& stage,
                        const std::vector<SurfaceTileResult>& tile_results,
                        bool after_freespace, std::uint64_t count) {
    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write " + output_path.string());
    }
    output << "ply\nformat binary_little_endian 1.0\n"
           << "comment navvis_recon inspection point cloud: " << stage << '\n'
           << "element vertex " << count << '\n'
           << "property float x\nproperty float y\nproperty float z\n"
           << "property float intensity\nend_header\n";
    std::uint64_t copied = 0U;
    for (const auto& result : tile_results) {
        if (!result.processed) {
            continue;
        }
        const fs::path& records =
            after_freespace ? result.after_freespace_records : result.before_surface_records;
        const std::uint64_t record_count =
            after_freespace ? result.after_freespace_count : result.before_surface_count;
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(record_count) * sizeof(InspectionPoint);
        if (fs::file_size(records) != expected_bytes) {
            throw std::runtime_error("inspection record size mismatch " + records.string());
        }
        if (expected_bytes == 0U) {
            continue;
        }
        std::ifstream input(records, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot read inspection records " + records.string());
        }
        output << input.rdbuf();
        if (!output) {
            throw std::runtime_error("cannot append inspection records " + records.string());
        }
        copied += record_count;
    }
    if (copied != count) {
        throw std::runtime_error("inspection record count mismatch for " + output_path.string());
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto total_started = Clock::now();
        auto phase_started = total_started;
        const Options options = parseArguments(argc, argv);
        if (!options.output.parent_path().empty()) {
            fs::create_directories(options.output.parent_path());
        }
        if (!options.after_freespace_output.parent_path().empty()) {
            fs::create_directories(options.after_freespace_output.parent_path());
        }
        if (!options.before_surface_output.parent_path().empty()) {
            fs::create_directories(options.before_surface_output.parent_path());
        }
        fs::create_directories(options.work_directory);
        if (!options.surface_diagnostics.empty()) {
            fs::create_directories(options.surface_diagnostics);
        }
        const fs::path merged_directory = options.work_directory / "merged";
        fs::create_directories(merged_directory);

        const PreparedShards prepared = prepareInputShards(options, merged_directory);
        const bool has_ray_history = prepared.has_ray_history;
        const auto& merged_files = prepared.files;
        Options runtime_options = loadRuntimeOptions(options, has_ray_history);
        std::cerr << "Phase timing: merge/input preparation " << secondsSince(phase_started)
                  << " s\n";
        phase_started = Clock::now();

        const fs::path ray_bucket_directory = options.work_directory / "global_ray_buckets";
        const fs::path candidate_halo_directory =
            options.work_directory / "freespace_candidate_halos";
        const bool use_global_sparse_rays = options.freespace_carving && has_ray_history &&
                                            options.freespace_mode == "sparse" &&
                                            options.global_ray_history;
        if (use_global_sparse_rays) {
            buildGlobalRayBuckets(merged_files, ray_bucket_directory, candidate_halo_directory,
                                  runtime_options.sparse_freespace);
        }
        std::cerr << "Phase timing: global ray index " << secondsSince(phase_started) << " s\n";
        phase_started = Clock::now();

        const fs::path surface_input_directory = options.work_directory / "binary_surface_inputs";
        fs::remove_all(surface_input_directory);
        fs::create_directories(surface_input_directory);
        std::ofstream freespace_diagnostics =
            openFreespaceDiagnostics(options.freespace_diagnostics);
        const bool emit_freespace_diagnostics = freespace_diagnostics.is_open();
        std::vector<ShardPreprocessResult> preprocess_results(merged_files.size());
        std::vector<std::size_t> preprocess_order(merged_files.size());
        for (std::size_t index = 0U; index < preprocess_order.size(); ++index) {
            preprocess_order[index] = index;
        }
        std::stable_sort(preprocess_order.begin(), preprocess_order.end(),
                         [&](std::size_t first, std::size_t second) {
                             return fs::file_size(merged_files[first]) >
                                    fs::file_size(merged_files[second]);
                         });
        std::atomic<std::size_t> next_preprocess{0U};
        std::atomic<std::size_t> completed_preprocess{0U};
        std::mutex preprocess_mutex;
        std::exception_ptr preprocess_error;
        const std::size_t preprocess_worker_count = std::min<std::size_t>(
            options.preprocess_threads, std::max<std::size_t>(merged_files.size(), 1U));
        auto preprocess_worker = [&]() {
            try {
                while (true) {
                    const std::size_t work_index = next_preprocess.fetch_add(1U);
                    if (work_index >= preprocess_order.size()) {
                        break;
                    }
                    const std::size_t shard_index = preprocess_order[work_index];
                    preprocess_results[shard_index] = preprocessShard(
                        merged_files[shard_index], has_ray_history, use_global_sparse_rays,
                        ray_bucket_directory, candidate_halo_directory, options, runtime_options,
                        emit_freespace_diagnostics);
                    const std::size_t completed = completed_preprocess.fetch_add(1U) + 1U;
                    std::lock_guard<std::mutex> lock(preprocess_mutex);
                    std::cerr << "Filtered " << completed << '/' << merged_files.size()
                              << " shards (" << preprocess_worker_count << " workers)"
                              << (options.freespace_carving
                                      ? " [" + options.freespace_mode + " ray carving]"
                                      : "")
                              << "\r";
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(preprocess_mutex);
                if (!preprocess_error) {
                    preprocess_error = std::current_exception();
                }
            }
        };
        std::vector<std::thread> preprocess_workers;
        preprocess_workers.reserve(preprocess_worker_count);
        for (std::size_t worker = 0; worker < preprocess_worker_count; ++worker) {
            preprocess_workers.emplace_back(preprocess_worker);
        }
        for (auto& worker : preprocess_workers) {
            worker.join();
        }
        if (preprocess_error) {
            std::rethrow_exception(preprocess_error);
        }
        const double preprocess_wall_seconds = secondsSince(phase_started);

        std::uint64_t rejected_hits = 0U;
        std::uint64_t rejected_freespace = 0U;
        double accumulated_read_seconds = 0.0;
        double accumulated_candidate_seconds = 0.0;
        double accumulated_freespace_seconds = 0.0;
        double accumulated_surface_input_seconds = 0.0;
        navvis_recon::SparseFreespaceTiming accumulated_sparse_timing;
        std::unordered_set<VoxelKey, VoxelKeyHash> kept_freespace_leaves;
        const auto partition_started = Clock::now();
        for (auto& result : preprocess_results) {
            rejected_hits += result.rejected_hits;
            rejected_freespace += result.rejected_freespace;
            accumulated_read_seconds += result.seconds_read;
            accumulated_candidate_seconds += result.seconds_candidate_build;
            accumulated_freespace_seconds += result.seconds_freespace;
            accumulated_surface_input_seconds += result.seconds_surface_input;
            accumulated_sparse_timing.occupancy_seconds += result.sparse_timing.occupancy_seconds;
            accumulated_sparse_timing.normal_seconds += result.sparse_timing.normal_seconds;
            accumulated_sparse_timing.trace_seconds += result.sparse_timing.trace_seconds;
            accumulated_sparse_timing.intersection_seconds +=
                result.sparse_timing.intersection_seconds;
            accumulated_sparse_timing.finalize_seconds += result.sparse_timing.finalize_seconds;
            kept_freespace_leaves.insert(result.kept_freespace_leaves.begin(),
                                         result.kept_freespace_leaves.end());
            std::vector<VoxelKey>().swap(result.kept_freespace_leaves);
            appendSurfaceInputs(surface_input_directory, result.surface_inputs);
            std::vector<navvis_recon::BinarySurfaceInput>().swap(result.surface_inputs);
            for (const auto& leaf : result.leaf_diagnostics) {
                // Halo leaves are context; emit each global leaf from its centroid shard.
                if (!(spatialShardKey(leaf.centroid) == result.diagnostic_shard)) {
                    continue;
                }
                freespace_diagnostics << leaf.voxel.x << ',' << leaf.voxel.y << ',' << leaf.voxel.z
                                      << ',' << leaf.centroid.x() << ',' << leaf.centroid.y() << ','
                                      << leaf.centroid.z() << ',' << leaf.normal.x() << ','
                                      << leaf.normal.y() << ',' << leaf.normal.z() << ','
                                      << leaf.hits << ',' << leaf.intersections << ','
                                      << static_cast<unsigned>(leaf.removed) << '\n';
            }
        }
        const double partition_wall_seconds = secondsSince(partition_started);
        std::cerr << '\n';
        freespace_diagnostics.close();
        std::cerr << "Preprocess timing: wall " << preprocess_wall_seconds
                  << " s; accumulated shard read " << accumulated_read_seconds
                  << " s, candidate build " << accumulated_candidate_seconds
                  << " s, sparse intersection " << accumulated_freespace_seconds
                  << " s, accepted-input materialization " << accumulated_surface_input_seconds
                  << " s; ordered tile partition wall " << partition_wall_seconds << " s\n";
        if (options.freespace_mode == "sparse") {
            std::cerr << "Sparse timing accumulated: occupancy "
                      << accumulated_sparse_timing.occupancy_seconds << " s, normals "
                      << accumulated_sparse_timing.normal_seconds << " s, trace "
                      << accumulated_sparse_timing.trace_seconds << " s, intersections "
                      << accumulated_sparse_timing.intersection_seconds << " s, finalize "
                      << accumulated_sparse_timing.finalize_seconds << " s\n";
        }
        std::cerr << "Phase timing: free-space and surface-input construction "
                  << secondsSince(phase_started) << " s\n";
        if (!kept_freespace_leaves.empty()) {
            std::cerr << "Retained " << kept_freespace_leaves.size()
                      << " CompactOccupancy leaves for post-aggregation Surface filtering\n";
        }
        phase_started = Clock::now();

        const fs::path output_records = options.work_directory / "surface_points.bin";
        const fs::path after_freespace_record_directory =
            options.work_directory / "after_freespace_inspection";
        const fs::path before_surface_record_directory =
            options.work_directory / "before_surface_inspection";
        fs::create_directories(after_freespace_record_directory);
        fs::create_directories(before_surface_record_directory);
        std::ofstream output(output_records, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create surface point records");
        }
        const auto surface_tiles = shardFiles(surface_input_directory, ".surface_input");
        auto surface_tile_inputs =
            loadAllSurfaceTilesWithHalo(surface_input_directory, surface_tiles);
        std::cerr << "Phase timing: ordered halo preparation " << secondsSince(phase_started)
                  << " s\n";
        phase_started = Clock::now();
        std::uint64_t kept = 0U;
        std::uint64_t after_freespace_kept = 0U;
        std::uint64_t before_surface_kept = 0U;
        navvis_recon::BinarySurfaceStageCounts totals;
        navvis_recon::BinaryOcclusionStageCounts occlusion_totals;
        std::ofstream surface_stage_counts;
        if (!options.surface_stage_counts.empty()) {
            surface_stage_counts.open(options.surface_stage_counts);
            if (!surface_stage_counts) {
                throw std::runtime_error("cannot create surface stage counts " +
                                         options.surface_stage_counts.string());
            }
            surface_stage_counts
                << "tile_x,tile_y,tile_z,raw,helper_input,helper_output,kept,main,valid,voxel,"
                   "density,sor,post,core\n";
        }
        std::vector<SurfaceTileResult> tile_results(surface_tiles.size());
        std::vector<std::size_t> surface_work_order;
        surface_work_order.reserve(surface_tiles.size());
        for (std::size_t index = 0U; index < surface_tiles.size(); ++index) {
            if (!options.has_surface_diagnostic_tile ||
                binarySurfaceTileKey(surface_tiles[index]) == options.surface_diagnostic_tile) {
                surface_work_order.push_back(index);
            }
        }
        if (surface_work_order.empty()) {
            throw std::runtime_error("surface diagnostic tile is not present in the input");
        }
        std::stable_sort(surface_work_order.begin(), surface_work_order.end(),
                         [&](std::size_t first, std::size_t second) {
                             return surface_tile_inputs[first].size() >
                                    surface_tile_inputs[second].size();
                         });
        std::atomic<std::size_t> next_surface_tile{0U};
        std::atomic<std::size_t> completed_surface_tiles{0U};
        std::atomic<bool> surface_failed{false};
        std::mutex surface_progress_mutex;
        std::mutex surface_error_mutex;
        std::exception_ptr surface_error;
        const auto process_surface_tiles = [&]() {
            try {
                while (!surface_failed.load(std::memory_order_relaxed)) {
                    const std::size_t work_index =
                        next_surface_tile.fetch_add(1U, std::memory_order_relaxed);
                    if (work_index >= surface_work_order.size()) {
                        break;
                    }
                    const std::size_t tile_index = surface_work_order[work_index];
                    tile_results[tile_index] = processSurfaceTile(
                        surface_tiles[tile_index], std::move(surface_tile_inputs[tile_index]),
                        has_ray_history, options, after_freespace_record_directory,
                        before_surface_record_directory,
                        kept_freespace_leaves.empty() ? nullptr : &kept_freespace_leaves,
                        runtime_options.sparse_freespace);
                    const std::size_t completed =
                        completed_surface_tiles.fetch_add(1U, std::memory_order_relaxed) + 1U;
                    std::lock_guard<std::mutex> progress_lock(surface_progress_mutex);
                    std::cerr << (options.inspection_only ? "Occlusion inspection " : "Surface ")
                              << completed << '/' << surface_tiles.size()
                              << " 5 m tiles (" << options.tile_threads << " workers)\r";
                }
            } catch (...) {
                surface_failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> error_lock(surface_error_mutex);
                if (surface_error == nullptr) {
                    surface_error = std::current_exception();
                }
            }
        };
        const std::size_t tile_thread_count = std::min<std::size_t>(
            options.tile_threads, std::max<std::size_t>(1U, surface_tiles.size()));
        std::vector<std::thread> tile_workers;
        tile_workers.reserve(tile_thread_count > 0U ? tile_thread_count - 1U : 0U);
        for (std::size_t index = 1U; index < tile_thread_count; ++index) {
            tile_workers.emplace_back(process_surface_tiles);
        }
        process_surface_tiles();
        for (auto& worker : tile_workers) {
            worker.join();
        }
        if (surface_error != nullptr) {
            std::rethrow_exception(surface_error);
        }
        std::cerr << "\nPhase timing: "
                  << (options.inspection_only ? "occlusion inspection tile processing "
                                              : "binary surface tile processing ")
                  << secondsSince(phase_started) << " s\n";
        phase_started = Clock::now();

        // Tile completion is intentionally allowed out of order, but counts
        // and records are reduced in the original sorted tile order.  Every
        // per-point calculation therefore keeps its former floating-point
        // order and the final PLY remains deterministic across thread counts.
        for (const auto& result : tile_results) {
            if (!result.processed) {
                continue;
            }
            accumulateSurfaceCounts(totals, result.counts);
            accumulateOcclusionCounts(occlusion_totals, result.occlusion_counts);
            after_freespace_kept += result.after_freespace_count;
            before_surface_kept += result.before_surface_count;
            if (surface_stage_counts.is_open()) {
                surface_stage_counts << result.owner.x << ',' << result.owner.y << ','
                                     << result.owner.z << ','
                                     << result.occlusion_counts.raw_input << ','
                                     << result.occlusion_counts.helper_input << ','
                                     << result.occlusion_counts.helper_output << ','
                                     << result.occlusion_counts.kept << ',' << result.counts.input
                                     << ',' << result.counts.valid << ','
                                     << result.counts.output_voxels << ',' << result.counts.density
                                     << ',' << result.counts.adaptive_sor << ','
                                     << result.counts.post << ',' << result.points.size() << '\n';
            }
            if (!options.surface_diagnostics.empty()) {
                std::cerr << "Surface diagnostic tile " << result.owner.x << ','
                          << result.owner.y << ',' << result.owner.z
                          << " raw/helper-in/helper-out/kept/main/valid/voxel/density/SOR/post/"
                          << "core: "
                          << result.occlusion_counts.raw_input << '/'
                          << result.occlusion_counts.helper_input << '/'
                          << result.occlusion_counts.helper_output << '/'
                          << result.occlusion_counts.kept << '/' << result.counts.input << '/'
                          << result.counts.valid << '/' << result.counts.output_voxels << '/'
                          << result.counts.density << '/' << result.counts.adaptive_sor << '/'
                          << result.counts.post << '/' << result.points.size() << '\n';
            }
            if (!result.points.empty()) {
                output.write(reinterpret_cast<const char*>(result.points.data()),
                             static_cast<std::streamsize>(result.points.size() *
                                                          sizeof(NavVisSurfacePoint)));
                kept += result.points.size();
            }
        }
        if (!options.inspection_only) {
            std::cerr << "\nSurface kernel accumulated timing: normals " << totals.seconds_normals
                      << " s, selection " << totals.seconds_selection << " s, output voxels "
                      << totals.seconds_output_voxels << " s, density " << totals.seconds_density
                      << " s, adaptive SOR " << totals.seconds_adaptive_sor
                      << " s, support pruning " << totals.seconds_support_pruning << " s, post "
                      << totals.seconds_post << " s\n";
        }
        output.close();
        if (!options.inspection_only) {
            writePly(options.output, output_records, kept);
        }
        writeInspectionPly(options.after_freespace_output,
                           "after free-space carving, aggregated at requested resolution",
                           tile_results, true, after_freespace_kept);
        writeInspectionPly(options.before_surface_output,
                           "after occlusion cleaning, before Surface", tile_results, false,
                           before_surface_kept);
        std::cerr << "Phase timing: deterministic reduction and PLY write "
                  << secondsSince(phase_started) << " s\n"
                  << "Phase timing: total " << secondsSince(total_started) << " s\n";
        std::cerr << "Rejected by hits: " << rejected_hits
                  << "; rejected by free space: " << rejected_freespace
                  << "; binary occlusion raw/kept/main: " << occlusion_totals.raw_input << '/'
                  << occlusion_totals.kept << '/' << totals.input
                  << "; binary surface stages input/valid/voxel/density/SOR/post: " << totals.input
                  << '/' << totals.valid << '/' << totals.output_voxels << '/' << totals.density
                  << '/' << totals.adaptive_sor << '/' << totals.post
                  << "; intermediate after-free-space/before-surface: "
                  << after_freespace_kept << '/' << before_surface_kept
                  << "; wrote after 5 m core crop: " << kept
                  << (options.inspection_only ? " [inspection-only; final Surface preserved]"
                                              : "")
                  << '\n';
        if (!options.retain_merged) {
            fs::remove_all(options.work_directory);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "navvis_recon_shard_surface_filter: " << error.what() << '\n';
        return 1;
    }
}
