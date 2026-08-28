#include "navvis_recon/pointcloud_coloring.hpp"

#include <opencv2/imgproc.hpp>

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace navvis_recon {

DepthMap::DepthMap(cv::Mat1f depth) : depth_(std::move(depth)) {}

DepthMap DepthMap::render(const std::vector<LaserPoint>& cloud, const Camera& camera) {
    cv::Mat1f depth(camera.height, camera.width, std::numeric_limits<float>::infinity());
    for (const auto& point : cloud) {
        const auto projection = camera.project(point.xyz);
        if (!camera.inside(projection.x(), projection.y()) || projection.z() <= 0.0F) {
            continue;
        }
        const int x = static_cast<int>(std::lround(projection.x()));
        const int y = static_cast<int>(std::lround(projection.y()));
        if (x >= 0 && y >= 0 && x < camera.width && y < camera.height) {
            depth(y, x) = std::min(depth(y, x), projection.z());
        }
    }
    // Two rounds of nearest-depth splatting reproduce the sparse raycaster's
    // conservative visibility support without requiring the internal GPU path.
    for (int iteration = 0; iteration < 2; ++iteration) {
        cv::Mat1f expanded = depth.clone();
        for (int y = 1; y + 1 < depth.rows; ++y) {
            for (int x = 1; x + 1 < depth.cols; ++x) {
                if (std::isfinite(depth(y, x))) {
                    continue;
                }
                float nearest = std::numeric_limits<float>::infinity();
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        nearest = std::min(nearest, depth(y + dy, x + dx));
                    }
                }
                expanded(y, x) = nearest;
            }
        }
        depth = std::move(expanded);
    }
    return DepthMap(std::move(depth));
}

bool DepthMap::isVisibleInCamera(
    const LaserPoint& point, const Camera& camera, float tolerance,
    Eigen::Vector3f* projection_out) const {
    const Eigen::Vector3f projection = camera.project(point.xyz);
    if (projection_out != nullptr) {
        *projection_out = projection;
    }
    if (!camera.inside(projection.x(), projection.y(), 1.0F) || projection.z() <= 0.0F) {
        return false;
    }
    if (depth_.empty()) {
        return true;
    }
    const float observed = depth_(static_cast<int>(projection.y()), static_cast<int>(projection.x()));
    return !std::isfinite(observed) || projection.z() <= observed + tolerance;
}

Eigen::Vector3f RollingShutterProjector::projectCCS2ICSRollingShutter(
    const Vec3f& point_world, const Camera& camera,
    const Pose& exposure_start, const Pose& exposure_end) {
    Eigen::Vector3f projection = camera.project(point_world);
    for (int iteration = 0; iteration < 3; ++iteration) {
        if (!std::isfinite(projection.y())) {
            return projection;
        }
        const float row_fraction = std::clamp(
            projection.y() / std::max(1.0F, static_cast<float>(camera.height - 1)), 0.0F, 1.0F);
        Camera row_camera = camera;
        const double row_time = exposure_start.timestamp +
                                row_fraction * (exposure_end.timestamp - exposure_start.timestamp);
        row_camera.world_from_camera = Pose::interpolate(exposure_start, exposure_end, row_time);
        projection = row_camera.project(point_world);
    }
    return projection;
}

std::vector<cv::Point2f> PatchProjector::projectGridSamples(
    const Camera& camera, const Vec3f& point_world, int radius) {
    const auto center = camera.project(point_world);
    std::vector<cv::Point2f> samples;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            samples.emplace_back(center.x() + static_cast<float>(x), center.y() + static_cast<float>(y));
        }
    }
    return samples;
}

std::optional<PatchColor> DirectPatchColorExtractor::extract(
    const cv::Mat3b& image_bgr, const std::vector<cv::Point2f>& samples) {
    if (image_bgr.empty() || samples.empty()) {
        return std::nullopt;
    }
    std::vector<float> red;
    std::vector<float> green;
    std::vector<float> blue;
    for (const auto& sample : samples) {
        const int x = static_cast<int>(std::lround(sample.x));
        const int y = static_cast<int>(std::lround(sample.y));
        if (x < 0 || y < 0 || x >= image_bgr.cols || y >= image_bgr.rows) {
            return std::nullopt;
        }
        const cv::Vec3b bgr = image_bgr(y, x);
        red.push_back(static_cast<float>(bgr[2]) / 255.0F);
        green.push_back(static_cast<float>(bgr[1]) / 255.0F);
        blue.push_back(static_cast<float>(bgr[0]) / 255.0F);
    }
    const auto component_median = [](std::vector<float> values) {
        auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
        std::nth_element(values.begin(), middle, values.end());
        return *middle;
    };
    const Vec3f color(component_median(red), component_median(green), component_median(blue));
    float variance = 0.0F;
    for (std::size_t i = 0; i < red.size(); ++i) {
        variance += (Vec3f(red[i], green[i], blue[i]) - color).squaredNorm();
    }
    variance /= static_cast<float>(red.size() * 3U);
    return PatchColor{color, 1.0F / (1.0F + 20.0F * variance)};
}

std::vector<RankedView> VoxelRanking::rankVoxels(
    const LaserPoint& point, const std::vector<ColoringView>& views,
    const ColoringOptions& options) {
    std::vector<RankedView> ranked;
    for (std::size_t i = 0; i < views.size(); ++i) {
        Eigen::Vector3f projection;
        if (!views[i].depth.isVisibleInCamera(point, views[i].camera, options.depth_tolerance, &projection) ||
            projection.z() > options.maximum_view_distance) {
            continue;
        }
        if (!views[i].mask.empty() &&
            views[i].mask(static_cast<int>(projection.y()), static_cast<int>(projection.x())) == 0) {
            continue;
        }
        const Vec3f toward_camera = normalizedOr(views[i].camera.world_from_camera.translation - point.xyz);
        const float incidence = point.has_normal ? std::max(0.0F, point.normal.dot(toward_camera)) : 1.0F;
        const float distance_weight = std::exp(-std::pow(projection.z() / options.maximum_view_distance, 2.0F));
        const float edge = std::min({projection.x(), projection.y(),
                                     static_cast<float>(views[i].camera.width - 1) - projection.x(),
                                     static_cast<float>(views[i].camera.height - 1) - projection.y()});
        const float boundary = std::clamp(
            edge / std::max(1.0F, 0.08F * std::min(views[i].camera.width, views[i].camera.height)),
            0.0F, 1.0F);
        ranked.push_back({i, projection, incidence * incidence * distance_weight * boundary});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const RankedView& first, const RankedView& second) { return first.score > second.score; });
    if (ranked.size() > static_cast<std::size_t>(options.maximum_views)) {
        ranked.resize(static_cast<std::size_t>(options.maximum_views));
    }
    return ranked;
}

std::vector<float> GlobalExposureOptimizer::optimize(
    const std::vector<ExposureObservation>& observations, std::size_t number_of_views) {
    if (number_of_views == 0U) {
        return {};
    }
    Eigen::MatrixXf normal = Eigen::MatrixXf::Zero(number_of_views, number_of_views);
    Eigen::VectorXf rhs = Eigen::VectorXf::Zero(number_of_views);
    for (const auto& observation : observations) {
        if (observation.first_brightness <= 1.0e-4F || observation.second_brightness <= 1.0e-4F) {
            continue;
        }
        Eigen::VectorXf row = Eigen::VectorXf::Zero(number_of_views);
        row[observation.first_view] = 1.0F;
        row[observation.second_view] = -1.0F;
        const float target = std::log(observation.second_brightness) -
                             std::log(observation.first_brightness);
        normal.noalias() += row * row.transpose();
        rhs.noalias() += row * target;
    }
    // Parameter residual: anchor the first camera's log gain at zero.
    normal(0, 0) += 100.0F;
    normal.diagonal().array() += 1.0e-5F;
    const Eigen::VectorXf log_gain = normal.ldlt().solve(rhs);
    std::vector<float> gains(number_of_views);
    for (std::size_t i = 0; i < number_of_views; ++i) {
        gains[i] = std::exp(std::clamp(log_gain[static_cast<Eigen::Index>(i)], -1.5F, 1.5F));
    }
    return gains;
}

Vec3f MultiViewColorBlending::blendColors(
    const std::vector<std::pair<float, Vec3f>>& weighted_colors) {
    Vec3f color = Vec3f::Zero();
    float weight = 0.0F;
    for (const auto& sample : weighted_colors) {
        color += sample.first * sample.second;
        weight += sample.first;
    }
    if (weight > 1.0e-8F) {
        return color / weight;
    }
    return Vec3f::Zero();
}

Colorizer::Colorizer(ColoringOptions options) : options_(std::move(options)) {}

float Colorizer::boundaryWeight(const Camera& camera, float u, float v) {
    const float edge = std::min({u, v, static_cast<float>(camera.width - 1) - u,
                                 static_cast<float>(camera.height - 1) - v});
    return std::clamp(edge / std::max(1.0F, 0.08F * std::min(camera.width, camera.height)), 0.0F, 1.0F);
}

std::vector<ExposureObservation> Colorizer::collectExposureObservations(
    const std::vector<LaserPoint>& cloud, const std::vector<ColoringView>& views) const {
    std::vector<ExposureObservation> observations;
    const std::size_t stride = std::max<std::size_t>(1U, cloud.size() / 5000U);
    for (std::size_t point_index = 0; point_index < cloud.size(); point_index += stride) {
        std::vector<std::pair<int, float>> brightness;
        for (std::size_t view_index = 0; view_index < views.size(); ++view_index) {
            Eigen::Vector3f projection;
            if (!views[view_index].depth.isVisibleInCamera(
                    cloud[point_index], views[view_index].camera, options_.depth_tolerance, &projection)) {
                continue;
            }
            const auto patch = DirectPatchColorExtractor::extract(
                views[view_index].image_bgr,
                PatchProjector::projectGridSamples(views[view_index].camera, cloud[point_index].xyz, 1));
            if (patch) {
                brightness.emplace_back(static_cast<int>(view_index), patch->rgb.mean());
            }
        }
        for (std::size_t first = 0; first < brightness.size(); ++first) {
            for (std::size_t second = first + 1; second < brightness.size(); ++second) {
                observations.push_back({brightness[first].first, brightness[second].first,
                                        brightness[first].second, brightness[second].second});
            }
        }
    }
    return observations;
}

std::vector<ColoredPoint> Colorizer::colorize(
    const std::vector<LaserPoint>& cloud, std::vector<ColoringView> views) const {
    for (auto& view : views) {
        if (view.depth.image().empty()) {
            view.depth = DepthMap::render(cloud, view.camera);
        }
    }
    if (options_.global_exposure && !views.empty()) {
        const auto gains = GlobalExposureOptimizer::optimize(
            collectExposureObservations(cloud, views), views.size());
        for (std::size_t i = 0; i < views.size(); ++i) {
            views[i].exposure_gain = gains[i];
        }
    }
    std::vector<ColoredPoint> result;
    result.reserve(cloud.size());
    for (const auto& point : cloud) {
        ColoredPoint colored;
        static_cast<LaserPoint&>(colored) = point;
        std::vector<std::pair<float, Vec3f>> weighted_colors;
        for (const auto& ranked : VoxelRanking::rankVoxels(point, views, options_)) {
            const auto& view = views[ranked.view_index];
            const auto patch = DirectPatchColorExtractor::extract(
                view.image_bgr,
                PatchProjector::projectGridSamples(view.camera, point.xyz, options_.patch_radius));
            if (patch && ranked.score > 0.0F) {
                weighted_colors.emplace_back(
                    ranked.score * patch->confidence,
                    (view.exposure_gain * patch->rgb).cwiseMax(0.0F).cwiseMin(1.0F));
            }
        }
        if (!weighted_colors.empty()) {
            Vec3f rgb = MultiViewColorBlending::blendColors(weighted_colors);
            if (options_.grayscale) {
                rgb = Vec3f::Constant(rgb.dot(Vec3f(0.299F, 0.587F, 0.114F)));
            }
            for (int channel = 0; channel < 3; ++channel) {
                colored.rgb[channel] = static_cast<std::uint8_t>(
                    std::lround(255.0F * std::clamp(rgb[channel], 0.0F, 1.0F)));
            }
            colored.has_color = true;
        }
        result.push_back(colored);
    }
    paintUncolored(result, options_);
    return result;
}

void Colorizer::paintUncolored(std::vector<ColoredPoint>& cloud, const ColoringOptions& options) {
    if (options.extrapolation == ColoringOptions::Extrapolation::None) {
        return;
    }
    if (options.extrapolation == ColoringOptions::Extrapolation::Discard) {
        cloud.erase(std::remove_if(cloud.begin(), cloud.end(),
                                   [](const ColoredPoint& point) { return !point.has_color; }),
                    cloud.end());
        return;
    }
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        if (cloud[i].has_color) {
            continue;
        }
        std::vector<std::pair<float, std::size_t>> neighbors;
        for (std::size_t j = 0; j < cloud.size(); ++j) {
            if (cloud[j].has_color) {
                neighbors.emplace_back((cloud[j].xyz - cloud[i].xyz).norm(), j);
            }
        }
        std::sort(neighbors.begin(), neighbors.end());
        if (neighbors.empty() ||
            (options.extrapolation == ColoringOptions::Extrapolation::Fill &&
             neighbors.front().first > options.fill_maximum_radius)) {
            continue;
        }
        const std::size_t count = std::min<std::size_t>(5U, neighbors.size());
        Vec3f color = Vec3f::Zero();
        float total = 0.0F;
        for (std::size_t n = 0; n < count; ++n) {
            const float weight = 1.0F / std::max(neighbors[n].first, 1.0e-4F);
            color += weight * cloud[neighbors[n].second].rgb.cast<float>();
            total += weight;
        }
        color /= total;
        cloud[i].rgb = color.array().round().cast<std::uint8_t>();
        cloud[i].has_color = true;
    }
}

}  // namespace navvis_recon
