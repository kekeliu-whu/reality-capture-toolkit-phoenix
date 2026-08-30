#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace navvis_recon {

struct MappedSpaceQualityOptions {
    double voxel_size = 0.16666666666666669;
    int minimum_rays_per_voxel = 36;
    int use_every_nth_point = 1;
    int brotli_quality = 5;
};

struct MappedSpaceQualityRay {
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d endpoint = Eigen::Vector3d::Zero();
};

#pragma pack(push, 1)
struct CompactQualityVoxel {
    std::uint64_t spatial_key = 0U;
    std::uint16_t directional_diversity = 0U;
    std::uint16_t ray_count = 0U;
    std::uint8_t minimum_range = 255U;
};
#pragma pack(pop)
static_assert(sizeof(CompactQualityVoxel) == 13U);

class MappedSpaceQualityGrid {
public:
    explicit MappedSpaceQualityGrid(MappedSpaceQualityOptions options = {});
    ~MappedSpaceQualityGrid();

    MappedSpaceQualityGrid(const MappedSpaceQualityGrid&) = delete;
    MappedSpaceQualityGrid& operator=(const MappedSpaceQualityGrid&) = delete;
    MappedSpaceQualityGrid(MappedSpaceQualityGrid&&) noexcept;
    MappedSpaceQualityGrid& operator=(MappedSpaceQualityGrid&&) noexcept;

    void addRay(const MappedSpaceQualityRay& ray);
    // Adds a ray using its position in the original, globally ordered input.
    // This keeps use_every_nth_point stable when contiguous input chunks are
    // accumulated independently.
    void addRayAtInputIndex(const MappedSpaceQualityRay& ray, std::size_t input_index);
    // Merge a later contiguous input chunk.  Voxel insertion order and the
    // first-seen order of direction bins are retained, so ordered chunk merges
    // reproduce the serial accumulator.
    void mergeLaterChunk(MappedSpaceQualityGrid&& later);
    [[nodiscard]] std::vector<CompactQualityVoxel> compact() const;
    [[nodiscard]] std::size_t inputRayCount() const noexcept;
    [[nodiscard]] const MappedSpaceQualityOptions& options() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Writes quality_voxels.bin, quality_voxels_sidecar.json and mapped_space.pcd.
// The binary payload is the version-2 packed <QHHB> format used by the mapped
// space consumer; no installed NavVis component is required at runtime.
void writeMappedSpaceQuality(
    const std::filesystem::path& output_directory,
    const std::vector<CompactQualityVoxel>& voxels,
    const MappedSpaceQualityOptions& options);

}  // namespace navvis_recon
