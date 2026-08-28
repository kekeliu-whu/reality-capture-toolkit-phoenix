#pragma once

#include "navvis_recon/types.hpp"

#include <opencv2/core.hpp>

namespace navvis_recon {

struct ColoringOptions {
    float maximum_view_distance = 35.0F;
    int maximum_views = 5;
    float depth_tolerance = 0.04F;
    int patch_radius = 2;
    bool global_exposure = true;
    bool grayscale = false;
    enum class Extrapolation { None, Paint, Discard, Fill } extrapolation = Extrapolation::Paint;
    float fill_maximum_radius = 0.20F;
};

class DepthMap {
public:
    DepthMap() = default;
    explicit DepthMap(cv::Mat1f depth);
    static DepthMap render(const std::vector<LaserPoint>& cloud, const Camera& camera);
    [[nodiscard]] bool isVisibleInCamera(
        const LaserPoint& point, const Camera& camera, float tolerance,
        Eigen::Vector3f* projection = nullptr) const;
    [[nodiscard]] const cv::Mat1f& image() const { return depth_; }

private:
    cv::Mat1f depth_;
};

struct ColoringView {
    Camera camera;
    cv::Mat3b image_bgr;
    cv::Mat1b mask;
    DepthMap depth;
    float exposure_gain = 1.0F;
};

class RollingShutterProjector {
public:
    static Eigen::Vector3f projectCCS2ICSRollingShutter(
        const Vec3f& point_world, const Camera& camera,
        const Pose& exposure_start, const Pose& exposure_end);
};

class PatchProjector {
public:
    static std::vector<cv::Point2f> projectGridSamples(
        const Camera& camera, const Vec3f& point_world, int radius);
};

struct PatchColor {
    Vec3f rgb = Vec3f::Zero();
    float confidence = 0.0F;
};

class DirectPatchColorExtractor {
public:
    static std::optional<PatchColor> extract(
        const cv::Mat3b& image_bgr, const std::vector<cv::Point2f>& samples);
};

struct RankedView {
    std::size_t view_index = 0;
    Eigen::Vector3f projection = Eigen::Vector3f::Zero();
    float score = 0.0F;
};

class VoxelRanking {
public:
    static std::vector<RankedView> rankVoxels(
        const LaserPoint& point, const std::vector<ColoringView>& views,
        const ColoringOptions& options);
};

struct ExposureObservation {
    int first_view = 0;
    int second_view = 0;
    float first_brightness = 0.0F;
    float second_brightness = 0.0F;
};

class GlobalExposureOptimizer {
public:
    static std::vector<float> optimize(
        const std::vector<ExposureObservation>& observations, std::size_t number_of_views);
};

class MultiViewColorBlending {
public:
    static Vec3f blendColors(const std::vector<std::pair<float, Vec3f>>& weighted_colors);
};

class Colorizer {
public:
    explicit Colorizer(ColoringOptions options);
    [[nodiscard]] std::vector<ColoredPoint> colorize(
        const std::vector<LaserPoint>& cloud, std::vector<ColoringView> views) const;

private:
    static float boundaryWeight(const Camera& camera, float u, float v);
    static void paintUncolored(std::vector<ColoredPoint>& cloud, const ColoringOptions& options);
    [[nodiscard]] std::vector<ExposureObservation> collectExposureObservations(
        const std::vector<LaserPoint>& cloud, const std::vector<ColoringView>& views) const;
    ColoringOptions options_;
};

}  // namespace navvis_recon
