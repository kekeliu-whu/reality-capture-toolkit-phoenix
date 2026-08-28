#pragma once

#include "navvis_recon/types.hpp"

#include <opencv2/core.hpp>

namespace navvis_recon {

struct PanoramaOptions {
    int width = 8192;
    int height = 4096;
    enum class SeamFinder { DynamicProgramming, GraphCut, Center } seam_finder = SeamFinder::GraphCut;
    enum class SeamCost { Color, ColorGradient } seam_cost = SeamCost::Color;
    int multiband_levels = 7;
    float weight_ray = 1.0F;
    bool floor_filling = true;
};

struct SurfelRenderingOptions {
    int width = 8192;
    int height = 4096;
    float surfel_radius = 0.01F;
    float near_distance = 0.2F;
    float far_distance = 35.0F;
    float inpainting_max_area_fraction = 0.0003F;
    float inpainting_depth_threshold = 0.01F;
    float inpainting_max_compactness = 0.7F;
};

struct WarpedImage {
    cv::Mat3f image;
    cv::Mat1f mask;
};

class GaussNewtonDepthMapOptimizer {
public:
    static cv::Mat1d optimizeDouble(
        const cv::Mat1f& measured_depth, const cv::Mat1b& valid_mask,
        double ray_weight, int iterations = 10000);
    static cv::Mat1f optimize(
        const cv::Mat1f& measured_depth, const cv::Mat1b& valid_mask,
        float ray_weight, int iterations = 10000);
};

class ExposureCompensatorSoftConstraint {
public:
    static std::vector<cv::Vec3f> estimateGains(
        const std::vector<cv::Mat3b>& images, const std::vector<cv::Mat1b>& masks);
};

class SeamMaskPreparer {
public:
    static std::vector<cv::Mat1f> prepare(
        const std::vector<cv::Mat1f>& masks, PanoramaOptions::SeamFinder finder);
};

class MultiBandBlender {
public:
    static cv::Mat3f blend(
        const std::vector<cv::Mat3f>& images, const std::vector<cv::Mat1f>& masks,
        int levels);
};

class PyramidInpainting {
public:
    static cv::Mat3f fillImage(const cv::Mat3f& image, const cv::Mat1b& valid_mask);
};

class FloorFiller {
public:
    static cv::Mat3f fill(const cv::Mat3f& panorama, const cv::Mat1b& valid_mask);
};

class PanoramaDepthRenderer {
public:
    // Reproduce nv_sparse-depthmap-renderer's PCL-octree ray query and
    // millimetre depth encoding, then run the exact four-level depth solver.
    static cv::Mat1d render(
        const std::vector<Vec3f>& points_world, const Pose& world_from_head,
        double near_distance, int width = 1024, int height = 512);
};

class ImageStitcher {
public:
    explicit ImageStitcher(PanoramaOptions options);
    [[nodiscard]] cv::Mat3f stitch(
        const std::vector<cv::Mat3b>& images,
        const std::vector<Camera>& cameras,
        const Pose& panorama_pose) const;
    static WarpedImage warp(
        const cv::Mat3b& image, const Camera& camera, const Pose& panorama_pose,
        int width, int height);

private:
    PanoramaOptions options_;
};

struct PointCloudRenderResult {
    cv::Mat3f color;
    cv::Mat1f depth;
};

class PointCloudRenderer {
public:
    explicit PointCloudRenderer(SurfelRenderingOptions options);
    [[nodiscard]] PointCloudRenderResult render(
        const std::vector<ColoredPoint>& cloud, const Pose& panorama_pose) const;

private:
    SurfelRenderingOptions options_;
};

}  // namespace navvis_recon
