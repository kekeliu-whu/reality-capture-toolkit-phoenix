#include "navvis_recon/cloud_builder.hpp"
#include "navvis_recon/cloud_surface_filter.hpp"
#include "navvis_recon/image_postprocessing.hpp"
#include "navvis_recon/panorama_rendering.hpp"
#include "navvis_recon/pointcloud_coloring.hpp"

#include <cassert>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace navvis_recon;

namespace {

Pose pose(double time, float x = 0.0F) {
    return Pose{time, Vec3f(x, 0.0F, 0.0F), Eigen::Quaternionf::Identity()};
}

std::uint32_t floatBits(float value) {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void testSphericalFibonacciDecode() {
    struct Case {
        std::uint32_t index;
        std::array<std::uint32_t, 3> expected;
    };
    const std::array<Case, 3> cases{{
        {2219185727U, {0xbf29a0f2U, 0xbf3f8b39U, 0xbd08c2c8U}},
        {1317944245U, {0xbeb538e8U, 0xbf5a0d48U, 0x3ec5c711U}},
        {3132314826U, {0x3f4fc14bU, 0xbeb95fa1U, 0xbeeacd53U}},
    }};
    for (const auto& item : cases) {
        const Vec3f decoded = detail::decodeSphericalFibonacciNormal(item.index);
        for (int axis = 0; axis < 3; ++axis) {
            assert(floatBits(decoded[axis]) == item.expected[static_cast<std::size_t>(axis)]);
        }
    }
}

void testCloudBuilder() {
    LaserPoint point;
    point.xyz = Vec3f(0.0F, 0.0F, 1.0F);
    point.timestamp = 0.5;
    point.intensity = 1.0F;
    CloudBuilderOptions options;
    options.filters.push_back(std::make_shared<NonFiniteXYZFilter>());
    const auto cloud = CloudBuilder(options).build({{point}}, {pose(0.0, 0.0F), pose(1.0, 2.0F)});
    assert(cloud.size() == 1U);
    assert((cloud.front().xyz - Vec3f(1.0F, 0.0F, 1.0F)).norm() < 1.0e-4F);
}

std::vector<LaserPoint> makePlane() {
    std::vector<LaserPoint> cloud;
    for (int y = -5; y <= 5; ++y) {
        for (int x = -5; x <= 5; ++x) {
            LaserPoint point;
            point.xyz = Vec3f(0.02F * x, 0.02F * y, 1.0F);
            point.origin = Vec3f::Zero();
            point.intensity = 1.0F;
            cloud.push_back(point);
        }
    }
    return cloud;
}

void testSurfaceFilter() {
    auto options = SurfaceFilterOptions::g11Standard(0.01F);
    options.clean_freespace = false;
    options.maximum_effective_planar_resolution = 0.2F;
    const auto result = CloudSurfaceFilter(options).filter(makePlane());
    assert(result.raw_filtered.size() > 50U);
    float median_alignment = 0.0F;
    for (const auto& point : result.raw_filtered) {
        median_alignment += std::abs(point.normal.z());
    }
    median_alignment /= static_cast<float>(result.raw_filtered.size());
    assert(median_alignment > 0.95F);
}

void testDirectionalFreespaceCarving() {
    std::vector<FreespaceRayObservation> rays;
    const VoxelKey transient{200, 0, 0};
    rays.push_back({transient, Vec3f(2.0F, 0.0F, 0.0F), Vec3f::Zero(), 1U});
    for (int offset = 0; offset < 5; ++offset) {
        const float range = 4.0F + 0.01F * static_cast<float>(offset);
        rays.push_back({VoxelKey{400 + offset, 0, 0}, Vec3f(range, 0.0F, 0.0F),
                        Vec3f::Zero(), 1U});
    }
    DirectionalFreespaceOptions options;
    options.origin_cell = 0.5F;
    options.angular_bin_degrees = 0.1F;
    options.endpoint_margin = 0.08F;
    options.minimum_intersections = 3U;
    options.intersection_hit_ratio = 1.5F;
    const auto evidence = DirectionalFreespaceCarver::compute(rays, options);
    assert(evidence.at(transient).removed);
    assert(evidence.at(transient).intersections == 5U);
    for (int offset = 0; offset < 5; ++offset) {
        assert(!evidence.at(VoxelKey{400 + offset, 0, 0}).removed);
    }
}

void testSparseFreespaceCarving() {
    const VoxelKey transient{200, 0, 0};
    std::vector<FreespaceCandidate> candidates{{transient, Vec3f(2.0F, 0.0F, 0.0F)}};
    std::vector<FreespaceRayObservation> rays;
    rays.push_back({transient, Vec3f(2.0F, 0.0F, 0.0F), Vec3f::Zero(), 1U});
    for (int offset = 0; offset < 5; ++offset) {
        rays.push_back({VoxelKey{400 + offset, 0, 0},
                        Vec3f(4.0F + 0.01F * offset, 0.0F, 0.0F),
                        Vec3f::Zero(), 1U});
    }
    SparseFreespaceOptions options;
    options.traversal_resolution = 0.02F;
    options.ray_radius = 0.012F;
    options.ray_stride = 1U;
    options.minimum_intersections = 3U;
    const auto evidence = SparseFreespaceCarver::compute(rays, candidates, options);
    assert(evidence.at(transient).removed);
    assert(evidence.at(transient).intersections == 5U);
}

Camera testCamera() {
    Camera camera;
    camera.width = 21;
    camera.height = 21;
    camera.fx = 10.0F;
    camera.fy = 10.0F;
    camera.cx = 10.0F;
    camera.cy = 10.0F;
    camera.world_from_camera = pose(0.0);
    return camera;
}

void testColorizer() {
    LaserPoint point;
    point.xyz = Vec3f(0.0F, 0.0F, 2.0F);
    point.normal = Vec3f(0.0F, 0.0F, -1.0F);
    point.has_normal = true;
    ColoringView view;
    view.camera = testCamera();
    view.image_bgr = cv::Mat3b(21, 21, cv::Vec3b(0, 0, 255));
    ColoringOptions options;
    options.global_exposure = false;
    options.patch_radius = 1;
    const auto colored = Colorizer(options).colorize({point}, {view});
    assert(colored.size() == 1U && colored.front().has_color);
    assert(colored.front().rgb[0] > 250 && colored.front().rgb[1] == 0 && colored.front().rgb[2] == 0);
}

void testImagePostprocessing() {
    cv::Mat3f image(32, 32, cv::Vec3f(0.25F, 0.5F, 0.75F));
    ImagePostprocessingOptions options;
    options.preset = ImagePostprocessingOptions::Preset::Fast;
    options.denoise = ImagePostprocessingOptions::Denoise::Off;
    const ImagePostprocessor processor(options);
    const cv::Mat3f output = processor.processLinearRgb(image);
    const auto jpeg = processor.encodeJpeg(output);
    assert(output.size() == image.size());
    assert(jpeg.size() > 100U && jpeg[0] == 0xffU && jpeg[1] == 0xd8U);
}

void testPanoramaAndSurfelRenderer() {
    Camera camera = testCamera();
    cv::Mat3b image(21, 21, cv::Vec3b(0, 255, 0));
    PanoramaOptions pano_options;
    pano_options.width = 64;
    pano_options.height = 32;
    pano_options.multiband_levels = 3;
    const cv::Mat3f panorama = ImageStitcher(pano_options).stitch({image}, {camera}, pose(0.0));
    assert(panorama.cols == 64 && panorama.rows == 32);

    ColoredPoint point;
    point.xyz = Vec3f(0.0F, 0.0F, 2.0F);
    point.rgb << 255, 0, 0;
    point.has_color = true;
    SurfelRenderingOptions surfel_options;
    surfel_options.width = 64;
    surfel_options.height = 32;
    surfel_options.surfel_radius = 0.2F;
    const auto rendered = PointCloudRenderer(surfel_options).render({point}, pose(0.0));
    assert(rendered.color.cols == 64 && rendered.depth.rows == 32);
    assert(std::isfinite(rendered.depth(16, 32)));
}

void testSoftExposureCompensator() {
    const std::array<cv::Vec3b, 4> colors{{
        {100, 110, 120},
        {140, 130, 120},
        {130, 125, 120},
        {110, 115, 120},
    }};
    std::vector<cv::Mat3b> images;
    std::vector<cv::Mat1b> masks;
    for (const cv::Vec3b& color : colors) {
        images.emplace_back(16, 16, color);
        masks.emplace_back(16, 16, static_cast<std::uint8_t>(255));
    }
    const auto gains =
        ExposureCompensatorSoftConstraint::estimateGains(images, masks);
    const std::array<cv::Vec3f, 4> expected{{
        {1.131435275F, 1.069522858F, 1.0F},
        {0.858073354F, 0.930413306F, 1.0F},
        {0.915091515F, 0.960329413F, 1.0F},
        {1.048426867F, 1.027716637F, 1.0F},
    }};
    assert(gains.size() == expected.size());
    for (std::size_t camera = 0; camera < gains.size(); ++camera) {
        for (int channel = 0; channel < 3; ++channel) {
            assert(std::abs(gains[camera][channel] - expected[camera][channel]) < 1.0e-5F);
        }
    }
}

}  // namespace

int main() {
    testSphericalFibonacciDecode();
    testCloudBuilder();
    testSurfaceFilter();
    testDirectionalFreespaceCarving();
    testSparseFreespaceCarving();
    testColorizer();
    testImagePostprocessing();
    testPanoramaAndSurfelRenderer();
    testSoftExposureCompensator();
    std::cout << "PASS: 9 C++ reconstruction smoke tests\n";
    return 0;
}
