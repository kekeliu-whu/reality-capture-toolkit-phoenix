#include "navvis_recon/mapped_space_quality.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

void assertEqual(
    const std::vector<navvis_recon::CompactQualityVoxel>& expected,
    const std::vector<navvis_recon::CompactQualityVoxel>& actual) {
    assert(expected.size() == actual.size());
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        assert(expected[index].spatial_key == actual[index].spatial_key);
        assert(expected[index].directional_diversity == actual[index].directional_diversity);
        assert(expected[index].ray_count == actual[index].ray_count);
        assert(expected[index].minimum_range == actual[index].minimum_range);
    }
}

void testOrderedChunkMerge() {
    navvis_recon::MappedSpaceQualityOptions options;
    options.minimum_rays_per_voxel = 1;
    options.use_every_nth_point = 3;

    std::vector<navvis_recon::MappedSpaceQualityRay> rays;
    for (int index = 0; index < 41; ++index) {
        navvis_recon::MappedSpaceQualityRay ray;
        ray.origin = Eigen::Vector3d(
            -1.25 + 0.013 * index, 2.75 - 0.007 * index, 1.05 + 0.003 * index);
        ray.endpoint = Eigen::Vector3d(
            -3.0 + 0.17 * (index % 7), -4.0 + 0.11 * index, 0.5 + 0.09 * (index % 5));
        rays.push_back(ray);
    }

    navvis_recon::MappedSpaceQualityGrid serial(options);
    for (const auto& ray : rays) {
        serial.addRay(ray);
    }

    navvis_recon::MappedSpaceQualityGrid merged(options);
    constexpr std::size_t chunk_size = 7U;
    for (std::size_t begin = 0U; begin < rays.size(); begin += chunk_size) {
        navvis_recon::MappedSpaceQualityGrid partial(options);
        const std::size_t end = std::min(rays.size(), begin + chunk_size);
        for (std::size_t index = begin; index < end; ++index) {
            partial.addRayAtInputIndex(rays[index], index);
        }
        merged.mergeLaterChunk(std::move(partial));
    }

    assert(serial.inputRayCount() == rays.size());
    assert(merged.inputRayCount() == rays.size());
    assertEqual(serial.compact(), merged.compact());
}

}  // namespace

int main() {
    navvis_recon::MappedSpaceQualityOptions options;
    options.minimum_rays_per_voxel = 1;
    navvis_recon::MappedSpaceQualityGrid grid(options);
    navvis_recon::MappedSpaceQualityRay ray;
    ray.origin = Eigen::Vector3d(
        static_cast<double>(-0.61929274F),
        static_cast<double>(3.0111508F),
        static_cast<double>(1.0272332F));
    ray.endpoint = Eigen::Vector3d(
        static_cast<double>(-0.6422162F),
        static_cast<double>(-5.015498F),
        static_cast<double>(1.644624F));
    grid.addRay(ray);
    std::vector<navvis_recon::CompactQualityVoxel> voxels = grid.compact();

    assert(voxels.size() == 50U);
    const std::uint64_t ray_count_sum = std::accumulate(
        voxels.begin(),
        voxels.end(),
        std::uint64_t{0},
        [](const std::uint64_t sum, const auto& voxel) { return sum + voxel.ray_count; });
    assert(ray_count_sum == 53U);
    assert(std::count_if(voxels.begin(), voxels.end(), [](const auto& voxel) {
               return voxel.ray_count == 2U;
           }) == 3);
    assert(std::all_of(voxels.begin(), voxels.end(), [](const auto& voxel) {
        return voxel.directional_diversity >= 202U && voxel.directional_diversity <= 257U &&
               voxel.minimum_range >= 1U && voxel.minimum_range <= 65U;
    }));

    const auto first = std::find_if(voxels.begin(), voxels.end(), [](const auto& voxel) {
        return voxel.spatial_key == 0x624924924924b370ULL;
    });
    const auto last = std::find_if(voxels.begin(), voxels.end(), [](const auto& voxel) {
        return voxel.spatial_key == 0x46db6db6db6d9a46ULL;
    });
    assert(first != voxels.end());
    assert(first->directional_diversity == 257U && first->ray_count == 1U &&
           first->minimum_range == 1U);
    assert(last != voxels.end());
    assert(last->directional_diversity == 202U && last->ray_count == 1U &&
           last->minimum_range == 65U);

    testOrderedChunkMerge();
    std::cout << "mapped-space quality semantic and ordered-merge oracles passed\n";
    return 0;
}
