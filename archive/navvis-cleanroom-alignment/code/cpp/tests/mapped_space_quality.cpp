#include "navvis_recon/mapped_space_quality.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

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

    std::cout << "mapped-space quality one-ray semantic oracle passed\n";
    return 0;
}
