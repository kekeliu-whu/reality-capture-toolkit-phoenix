#pragma once

#include "navvis_recon/types.hpp"

#include <unordered_map>

namespace navvis_recon {

namespace detail {
// Exposed for the bit-pattern regression that guards the binary-compatible
// uint32 Spherical Fibonacci decoder used by global free-space normals.
Vec3f decodeSphericalFibonacciNormal(std::uint32_t index);
}

struct VoxelKey {
    int x = 0;
    int y = 0;
    int z = 0;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& key) const noexcept;
};

struct SurfacePoint {
    Vec3f xyz = Vec3f::Zero();
    Vec3f origin = Vec3f::Zero();
    Vec3f normal = Vec3f::Zero();
    float intensity = 0.0F;
    float weight = 0.0F;
    bool has_normal = false;
};

struct SurfaceFilterOptions {
    float resolution = 0.01F;
    float minimum_normal_radius = 0.025F;
    float maximum_normal_radius = 0.15F;
    int number_of_normal_levels = 6;
    float search_radius_smoothing = 0.04F;
    float normal_smoothing_radius = 0.02F;
    int density_filter_k_neighbors = 16;
    float maximum_effective_planar_resolution = 0.05F;
    int statistical_k_neighbors = 16;
    float statistical_stddev = 2.0F;
    bool regularize_grid = false;
    bool clean_freespace = true;
    int freespace_minimum_intersections = 3;
    float freespace_intersection_hit_ratio = 1.5F;
    float freespace_minimum_origin_distance = 0.15F;
    float freespace_maximum_origin_distance = 50.0F;
    float freespace_endpoint_margin = 0.04F;
    float freespace_maximum_incidence_angle_deg = 80.0F;

    static SurfaceFilterOptions g11Standard(float resolution = 0.01F);
};

// Ray evidence used by the streaming surface worker.  Exact version-2 shards
// carry one observation per raw return; return_count remains for backward
// compatibility with older clustered shards.
struct FreespaceRayObservation {
    VoxelKey endpoint_voxel;
    Vec3f endpoint = Vec3f::Zero();
    Vec3f origin = Vec3f::Zero();
    std::uint32_t return_count = 1U;
};

struct DirectionalFreespaceOptions {
    float origin_cell = 0.50F;
    float angular_bin_degrees = 0.12F;
    float minimum_origin_distance = 0.40F;
    float maximum_origin_distance = 30.0F;
    float endpoint_margin = 0.08F;
    std::uint32_t minimum_intersections = 3U;
    float intersection_hit_ratio = 1.5F;
};

struct DirectionalFreespaceEvidence {
    std::uint32_t hit_viewpoints = 0U;
    std::uint32_t intersections = 0U;
    bool removed = false;
};

class DirectionalFreespaceCarver {
  public:
    // Uses two half-shifted origin/direction grids and accepts only evidence
    // present in both.  This is deliberately conservative around angular and
    // origin-cell boundaries.
    static std::unordered_map<VoxelKey, DirectionalFreespaceEvidence, VoxelKeyHash>
    compute(const std::vector<FreespaceRayObservation>& observations,
            const DirectionalFreespaceOptions& options);
};

struct FreespaceCandidate {
    VoxelKey voxel;
    Vec3f point = Vec3f::Zero();
    Vec3f normal = Vec3f::Zero();
    std::uint32_t hit_count = 1U;
};

struct SparseFreespaceOptions {
    // Captured from the G11/standard FreespaceOctreeOptions constructor in
    // liblibpointcloud_octree.so.4.85.  Keep these as algorithm constants;
    // they are not calibrated from a desired output point count.
    float traversal_resolution = 0.02F;
    float ray_radius = 0.006F;
    float minimum_origin_distance = 0.50F;
    float maximum_origin_distance = 15.0F;
    float endpoint_margin = 0.05F;
    float maximum_incidence_angle_degrees = 85.0F;
    std::uint32_t ray_stride = 1U;
    std::uint32_t minimum_intersections = 1U;
    float intersection_hit_ratio = 1.0F;
    // CompactOctree phase-aligns its global bounding cube to the first point
    // in the lowest occupied Tilecloud Morton tile.  Both the phase and the
    // root partition must be shared by every spatial worker.
    Vec3f grid_anchor = Vec3f::Zero();
    bool has_grid_anchor = false;
    VoxelKey grid_root_lower{};
    int grid_root_size = 0;
    bool has_grid_root = false;
};

struct SparseFreespaceLeafDiagnostic {
    VoxelKey voxel;
    Vec3f centroid = Vec3f::Zero();
    Vec3f normal = Vec3f::Zero();
    std::uint32_t hits = 0U;
    std::uint16_t intersections = 0U;
    bool removed = false;
};

struct SparseFreespaceTiming {
    double occupancy_seconds = 0.0;
    double normal_seconds = 0.0;
    double trace_seconds = 0.0;
    double intersection_seconds = 0.0;
    double finalize_seconds = 0.0;
};

class SparseFreespaceCarver {
  public:
    // Reproduces the original ordering: compact 2 cm occupancy with raw hit
    // counts, shortened/clipped rays, ray-to-centroid and incidence tests,
    // saturating intersection counters, then intersection/hit removal.
    static std::unordered_map<VoxelKey, DirectionalFreespaceEvidence, VoxelKeyHash>
    compute(const std::vector<FreespaceRayObservation>& observations,
            const std::vector<FreespaceCandidate>& candidates,
            const SparseFreespaceOptions& options,
            std::vector<SparseFreespaceLeafDiagnostic>* leaf_diagnostics = nullptr,
            std::vector<DirectionalFreespaceEvidence>* candidate_evidence = nullptr,
            bool build_evidence_map = true, std::vector<std::uint8_t>* candidate_removed = nullptr,
            SparseFreespaceTiming* timing = nullptr);
};

class VoxelAggregator {
  public:
    static std::vector<SurfacePoint> aggregate(const std::vector<LaserPoint>& input,
                                               float resolution, bool regularize_grid);
};

class MultiScaleNormalEstimator {
  public:
    static void compute(std::vector<SurfacePoint>& cloud, const SurfaceFilterOptions& options);
};

class FreespaceOctree {
  public:
    explicit FreespaceOctree(float resolution);
    void addPoint(const SurfacePoint& point);
    void computeCentroidNormals();
    void computeIntersections(const SurfaceFilterOptions& options);
    void removeFreespaceVoxels(const SurfaceFilterOptions& options);
    [[nodiscard]] std::vector<SurfacePoint> toCompactOccupancyOctree() const;

  private:
    struct Node {
        std::vector<SurfacePoint> points;
        SurfacePoint centroid;
        int hits = 0;
        int intersections = 0;
        bool removed = false;
    };
    float resolution_;
    std::unordered_map<VoxelKey, Node, VoxelKeyHash> nodes_;
};

class AdaptiveStatisticalOutlierRemoval {
  public:
    static std::vector<SurfacePoint> filter(const std::vector<SurfacePoint>& cloud, int k_neighbors,
                                            float standard_deviations);
};

struct SurfaceFilterResult {
    std::vector<SurfacePoint> smoothed;
    std::vector<SurfacePoint> raw_filtered;
};

class CloudSurfaceFilter {
  public:
    explicit CloudSurfaceFilter(SurfaceFilterOptions options);
    [[nodiscard]] SurfaceFilterResult filter(const std::vector<LaserPoint>& input) const;

  private:
    [[nodiscard]] std::vector<SurfacePoint> densityFilter(std::vector<SurfacePoint> cloud) const;
    [[nodiscard]] std::vector<SurfacePoint> smooth(const std::vector<SurfacePoint>& cloud) const;
    SurfaceFilterOptions options_;
};

} // namespace navvis_recon
