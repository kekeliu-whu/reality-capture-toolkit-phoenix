#include "navvis_recon/panorama_rendering.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace navvis_recon;

int main() {
    Camera camera;
    camera.width = 21;
    camera.height = 21;
    camera.fx = 10.0F;
    camera.fy = 10.0F;
    camera.cx = 10.0F;
    camera.cy = 10.0F;
    camera.world_from_camera = {
        0.0, Vec3f::Zero(), Eigen::Quaternionf::Identity()};

    cv::Mat3b image(21, 21, cv::Vec3b(0, 0, 0));
    image(10, 10) = cv::Vec3b(17, 113, 241);
    const Pose panorama_pose{
        0.0, Vec3f::Zero(), Eigen::Quaternionf::Identity()};

    // Odd panorama dimensions put one equirectangular sample exactly on the
    // forward ray only when pixel centres, rather than pixel corners, are
    // used to construct the spherical direction.
    const WarpedImage warped =
        ImageStitcher::warp(image, camera, panorama_pose, 65, 33);
    const cv::Vec3f centre = warped.image(16, 32);
    assert(std::abs(centre[0] - 17.0F / 255.0F) < 1.0e-6F);
    assert(std::abs(centre[1] - 113.0F / 255.0F) < 1.0e-6F);
    assert(std::abs(centre[2] - 241.0F / 255.0F) < 1.0e-6F);
    assert(warped.mask(16, 32) == 1.0F);

    const cv::Mat1f depth = PanoramaDepthRenderer::render(
        {Vec3f(4.0F, 0.0F, 0.0F), Vec3f(2.0F, 0.0F, 0.0F)},
        panorama_pose, 0.0, 65, 33);
    assert(std::abs(depth(16, 32) - 2.0F) < 1.0e-6F);
    assert(cv::checkRange(depth));

    cv::Mat1f sparse_depth = cv::Mat1f::zeros(8, 16);
    cv::Mat1b sparse_valid = cv::Mat1b::zeros(8, 16);
    sparse_depth(4, 2) = 1.0F;
    sparse_depth(4, 10) = 3.0F;
    sparse_valid(4, 2) = 255;
    sparse_valid(4, 10) = 255;
    const cv::Mat1f optimized =
        GaussNewtonDepthMapOptimizer::optimize(sparse_depth, sparse_valid, 1.0F);
    assert(cv::checkRange(optimized));
    assert(optimized(4, 2) < optimized(4, 10));
    assert(optimized(4, 6) > optimized(4, 2));
    assert(optimized(4, 6) < optimized(4, 10));
    assert(std::abs(optimized(4, 0) - optimized(4, 15)) < 0.15F);

    std::cout << "PASS: pixel-centre sampling and multilevel radial depth rendering\n";
    return 0;
}
