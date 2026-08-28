#pragma once

#include "navvis_recon/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace navvis_recon {

// Point layout used between the captured G11 surface-processing stages.  The
// sensor origin is retained only for orienting the multi-scale PCA normal.
struct BinarySurfaceInput {
    Vec3f xyz = Vec3f::Zero();
    Vec3f origin = Vec3f::Zero();
    float intensity = 0.0F;
    float weight = 1.0F;
};

struct BinarySurfacePoint {
    Vec3f xyz = Vec3f::Zero();
    Vec3f normal = Vec3f::Zero();
    float intensity = 0.0F;
    float curvature = 0.0F;
    float weight = 1.0F;
};

struct BinarySurfaceOptions {
    // Captured multi_scale_normal_estimation arguments.
    // Captured bits 0x3ccccccc (one ULP below the nearest 0.025f literal).
    float normal_radius_minimum = 0.02499999850988388F;
    float normal_radius_maximum = 0.15F;
    int normal_radius_levels = 6;
    std::size_t normal_minimum_support = 16U;
    float normal_minimum_in_plane_spread = 0.04F;
    float normal_minimum_planarity = 0.30F;
    float normal_maximum_direction_spread = 0.20F;
    float normal_minimum_incidence_cosine = 0.1736482233F; // sin(10 degrees)

    // Captured surface_point_selection arguments.
    float selection_radius = 0.04F;
    float normal_cylinder_radius = 0.02F;
    float minimum_cylinder_radius = 0.0070710676F;
    std::size_t selection_minimum_points = 5U;
    float selection_target_weight = 16.0F;

    // Captured output octree and filters.
    float output_resolution = 0.01F;
    float output_merge_distance = 0.01F;
    float output_normal_cosine = 0.93969261646F; // cos(20 degrees)
    std::size_t density_neighbors = 9U;
    float minimum_planar_density = 399.99997F;
    float distance_epsilon = 1.0e-5F;
    std::size_t sor_neighbors = 10U;
    float sor_sigma = 1.7F;
    // Captured bits 0x3ccccccc, shared with the minimum normal radius.
    float post_radius = 0.02499999850988388F;
    float post_plane_distance = 0.015F;
    float post_normal_cosine = 0.9848077297F; // cos(10 degrees)
};

struct BinarySurfaceStageCounts {
    std::size_t input = 0U;
    std::size_t normals = 0U;
    std::size_t selected = 0U;
    std::size_t valid = 0U;
    std::size_t output_voxels = 0U;
    std::size_t density = 0U;
    std::size_t adaptive_sor = 0U;
    std::size_t post = 0U;
    double seconds_normals = 0.0;
    double seconds_selection = 0.0;
    double seconds_output_voxels = 0.0;
    double seconds_density = 0.0;
    double seconds_adaptive_sor = 0.0;
    double seconds_support_pruning = 0.0;
    double seconds_post = 0.0;
};

struct BinaryOcclusionOptions {
    float helper_input_resolution = 0.02F;
    float helper_output_resolution = 0.02F;
    float helper_minimum_cylinder_radius = 0.0141421352F;
    float index_resolution = 0.02F;
    float surfel_disc_radius = 0.0282842703F;
    float normal_tolerance_radians = 0.5235987902F;
    float ray_hit_tolerance = 0.03F;
    float pair_distance_floor = 0.03F;
};

struct BinaryOcclusionStageCounts {
    std::size_t raw_input = 0U;
    std::size_t helper_input = 0U;
    std::size_t helper_normals = 0U;
    std::size_t helper_selected = 0U;
    std::size_t helper_valid = 0U;
    std::size_t helper_initial_voxels = 0U;
    std::size_t helper_output = 0U;
    std::size_t kept = 0U;
};

struct BinaryOcclusionRayDiagnostic {
    std::uint8_t status = 0U;
    std::uint8_t first_non_miss_primitive = 6U;
    std::vector<std::uint32_t> first_leaf_candidates;
    std::uint32_t endpoint_neighbor = UINT32_MAX;
    bool pair_consistent = false;
};

// Clean-room implementation of the non-SLAM surface stages recovered from
// the installed G11 binary.  A tile must include at least a 15 cm halo; the
// caller performs the final half-open 5 m core crop.
std::vector<BinarySurfacePoint>
runBinarySurfacePipeline(const std::vector<BinarySurfaceInput>& input,
                         const BinarySurfaceOptions& options = {},
                         BinarySurfaceStageCounts* counts = nullptr);

// Stage-level entry points used by the captured-intermediate acceptance tool.
// Production callers should normally use runBinarySurfacePipeline().
std::vector<BinarySurfacePoint>
applyBinaryOutputVoxelAggregation(const std::vector<BinarySurfacePoint>& input,
                                   const BinarySurfaceOptions& options = {});
std::vector<BinarySurfacePoint>
applyBinaryOutputVoxelPrimaryAggregation(const std::vector<BinarySurfacePoint>& input,
                                          const BinarySurfaceOptions& options = {});
std::vector<std::uint32_t>
inspectBinaryOutputVoxelMergeChoices(const std::vector<BinarySurfacePoint>& input,
                                      const BinarySurfaceOptions& options = {});
std::vector<BinarySurfacePoint>
applyBinaryMultiScaleNormalEstimation(const std::vector<BinarySurfaceInput>& input,
                                      const BinarySurfaceOptions& options = {});
std::vector<BinarySurfaceInput>
applyBinaryOcclusionHelperInputAggregation(const std::vector<BinarySurfaceInput>& raw_rays,
                                           float resolution = 0.02F);
std::vector<BinarySurfacePoint>
applyBinaryOcclusionHelperSurface(const std::vector<BinarySurfaceInput>& raw_rays,
                                  const BinarySurfaceOptions& surface_options = {},
                                  const BinaryOcclusionOptions& occlusion_options = {},
                                  BinaryOcclusionStageCounts* counts = nullptr);
std::vector<BinarySurfacePoint> applyBinaryOcclusionHelperSurfaceFromInput(
    const std::vector<BinarySurfaceInput>& helper_input,
    const BinarySurfaceOptions& surface_options = {},
    const BinaryOcclusionOptions& occlusion_options = {},
    BinaryOcclusionStageCounts* counts = nullptr);
// Diagnostic/classifier entry point for validating a captured helper surface
// independently from helper construction.
std::vector<std::uint8_t>
classifyBinaryOcclusionRays(const std::vector<BinarySurfaceInput>& raw_rays,
                            const std::vector<BinarySurfacePoint>& helper,
                            const BinaryOcclusionOptions& occlusion_options = {},
                            std::vector<BinaryOcclusionRayDiagnostic>* diagnostics = nullptr);
// G11 clean-occlusions stage. Input records are raw rays before 1 cm
// aggregation; the returned records retain the original raw weights. When
// requested, statuses use the binary's 0..6 classifier values.
std::vector<BinarySurfaceInput>
applyBinaryOcclusionCleaning(const std::vector<BinarySurfaceInput>& raw_rays,
                             std::vector<std::uint8_t>* statuses = nullptr,
                             const BinarySurfaceOptions& surface_options = {},
                             const BinaryOcclusionOptions& occlusion_options = {},
                             BinaryOcclusionStageCounts* counts = nullptr);
// Equivalent cleaning path when the caller has already built and filtered
// the 2 cm helper input.  The installed pipeline applies CompactOccupancy
// after helper aggregation, while the raw rays are filtered separately
// before classification; keeping the two inputs distinct preserves that
// observable ordering.
std::vector<BinarySurfaceInput> applyBinaryOcclusionCleaningFromHelperInput(
    const std::vector<BinarySurfaceInput>& raw_rays,
    const std::vector<BinarySurfaceInput>& helper_input,
    std::vector<std::uint8_t>* statuses = nullptr,
    const BinarySurfaceOptions& surface_options = {},
    const BinaryOcclusionOptions& occlusion_options = {},
    BinaryOcclusionStageCounts* counts = nullptr);
std::vector<BinarySurfacePoint>
applyBinarySurfacePointSelection(const std::vector<BinarySurfacePoint>& input,
                                 const BinarySurfaceOptions& options = {});
std::vector<BinarySurfacePoint>
applyBinaryDensityFilter(const std::vector<BinarySurfacePoint>& input,
                         const BinarySurfaceOptions& options = {});
std::vector<BinarySurfacePoint> applyBinaryAdaptiveSor(const std::vector<BinarySurfacePoint>& input,
                                                       const BinarySurfaceOptions& options = {});
std::vector<BinarySurfacePoint>
applyBinarySurfaceSupportPruning(const std::vector<BinarySurfacePoint>& support,
                                 const std::vector<BinarySurfacePoint>& target,
                                 float radius = 0.03F);
std::vector<BinarySurfacePoint>
applyBinaryPostSmoothingFilter(const std::vector<BinarySurfacePoint>& input,
                               const BinarySurfaceOptions& options = {});

} // namespace navvis_recon
