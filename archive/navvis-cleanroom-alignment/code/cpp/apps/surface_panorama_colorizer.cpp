#include "navvis_recon/cloud_surface_filter.hpp"
#include "navvis_recon/types.hpp"

#include <Eigen/Dense>

#ifdef NAVVIS_RECON_HAVE_CERES
#include <ceres/ceres.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using navvis_recon::Mat3f;
using navvis_recon::Pose;
using navvis_recon::Vec3f;
using navvis_recon::VoxelKey;
using navvis_recon::VoxelKeyHash;

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr std::size_t kCandidateViews = 32U;
constexpr std::size_t kBlendedViews = 24U;
constexpr std::size_t kSelectedViews = 5U;
constexpr int kG11ImageWidth = 5472;
constexpr int kG11ImageHeight = 3648;
// The installed extractor keeps reducing through level 7.  Exposure
// centroids can be less than half a metre from a camera, where stopping at
// level 5 samples a substantially sharper image than the footprint requires.
constexpr std::size_t kDirectPatchPyramidLevels = 8U;
// Runtime value read from the G11 SDK projector used by all four cameras.
constexpr float kDirectPatchFocalScale = 1368.4115F;
// OCamProjectionModel's runtime field-of-view threshold is cos(pi/2).
constexpr double kOCamFrontHemisphereThreshold = 6.123233995736766e-17;

#pragma pack(push, 1)
struct SurfacePoint {
    float x;
    float y;
    float z;
    float intensity;
    float nx;
    float ny;
    float nz;
    float curvature;
};

struct ColoredSurfacePoint {
    float x;
    float y;
    float z;
    std::uint8_t red;
    std::uint8_t green;
    std::uint8_t blue;
    std::uint8_t alpha;
    float intensity;
    float nx;
    float ny;
    float nz;
    float curvature;
};
#pragma pack(pop)
static_assert(sizeof(SurfacePoint) == 32U);
static_assert(sizeof(ColoredSurfacePoint) == 36U);

struct Options {
    fs::path input;
    fs::path output;
    fs::path panorama_directory;
    fs::path info_directory;
    fs::path camera_directory;
    fs::path camera_mask_directory;
    fs::path sensor_frame;
    fs::path direct_mask;
    fs::path exposure_models;
    fs::path exposure_model_output;
    fs::path exposure_ovs_output;
    fs::path exposure_ovs_binary_output;
    // Regression-only input for isolating the optimizer from view selection.
    // The file contains five packed 8-byte OVS entries per 0.1 m centroid.
    fs::path exposure_ovs_binary_input;
    // Regression dump matching nv_colorcloud's five packed 8-byte OVS
    // observations per final point.
    fs::path color_ovs_output;
    // Regression-only input for isolating Gamma/ABS/KNN from final view
    // selection. The layout is the same packed five-by-eight-byte OVS stream.
    fs::path color_ovs_input;
    fs::path exposure_cloud_output;
    fs::path exposure_projection_output;
    // Regression hook for comparing the clean-room visibility and scoring
    // paths against captured PCT depth maps. Empty in normal operation.
    fs::path depth_map_input_directory;
    fs::path depth_map_output_directory;
    // Raw 8-float surfels captured at the PCT raycaster boundary. This is a
    // regression-only input used to separate renderer error from PCT build
    // error; normal clean-room runs leave it empty.
    fs::path pct_surfel_input;
    // Regression dump of the independently generated 8-float PCT surfels.
    fs::path pct_surfel_output;
    // nv_colorcloud constructs the G11 PCT depth maps at 1/8 camera
    // resolution: 5472 x 3648 -> 684 x 456.
    int depth_width = 684;
    int depth_views = 24;
    int depth_splat_radius = 1;
    float input_resolution = 0.01F;
    int color_views = 24;
    float visibility_tolerance = 0.50F;
    int visibility_patch_radius = 1;
    float visibility_min_fraction = 0.56F;
    float view_max_distance = 30.0F;
    std::size_t chunk_points = 2'000'000U;
    std::size_t image_cache = 48U;
    std::size_t camera_cache = 12U;
    int camera_index = -1;
    int exposure_solver_threads = 32;
    // Kept configurable for deterministic solver-trajectory regression. The
    // production default remains the binary's 50 nonlinear iterations.
    int exposure_solver_max_iterations = 50;
    double exposure_solver_initial_trust_region_radius = 1.0e4;
    float score_incidence_power = 0.25F;
    float score_radius_power = 5.0F;
    float score_distance_power = 2.0F;
    bool average_direct_views = false;
    bool median_direct_views = false;
    bool robust_direct_views = true;
    // The binary's reported voxel count is its spatial work partition.  The
    // final five-view ranking itself is per point.
    bool voxel_view_selection = false;
    bool global_exposure = true;
    bool exposure_only = false;
    bool exposure_capture_only = false;
    bool exposure_source_consistency = true;
    // The installed optimizer omits black V=0 observations from both
    // exposure histogram families while retaining them in packed OVS.
    bool exposure_histogram_exclude_zero = true;
    bool knn_extrapolation = true;
    bool standard_histogram = false;
};

struct PlyInput {
    std::uint64_t count = 0U;
    std::streamoff data_offset = 0;
    bool colored_records = false;
};

struct CandidateList {
    std::array<int, kCandidateViews> indices{};
    std::size_t count = 0U;
};

using ConservativeCandidateCache = std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash>;

struct CapturePoses {
    Pose head;
    std::array<Pose, 4> cameras;
};

struct OCamCamera {
    double c = 1.0;
    double d = 0.0;
    double e = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::vector<double> world_to_camera;
    std::vector<double> camera_to_world;
    int transverse_orientation = 6;
};

struct OCamProjection {
    double image_x = 0.0;
    double image_y = 0.0;
    double radius = 0.0;
    float range = 0.0F;
};

struct VisibleViews {
    std::array<int, kBlendedViews> indices{};
    std::uint8_t count = 0U;
    int fallback = 0;
};

struct CameraMasks {
    std::array<cv::Mat, 4> valid;
    std::array<cv::Mat, 4> point_weight;
    int width = kG11ImageWidth;
    int height = kG11ImageHeight;
};

struct DirectSample {
    cv::Vec3b rgb{};
    cv::Vec3b raw_rgb{};
    float score = -1.0F;
    int view = -1;
    float image_x = 0.0F;
    float image_y = 0.0F;
    std::uint16_t normalized_quality = 0U;
    bool has_normalized_quality = false;
};

struct GammaModel {
    double gain = 1.0;
    double exponent = 1.0;
};

struct SelectedSamples {
    std::array<DirectSample, kSelectedViews> values{};
    std::uint8_t count = 0U;
};

struct VoxelViewAccumulator {
    explicit VoxelViewAccumulator(std::size_t view_count)
        : score_sum(view_count, 0.0), visible_count(view_count, 0U) {}

    std::vector<double> score_sum;
    std::vector<std::uint32_t> visible_count;
    std::uint32_t samples = 0U;
};

struct VoxelSelectedViews {
    std::array<int, kSelectedViews> indices{};
    std::uint8_t count = 0U;
};

// Exposure voxel ranking must consider every capture-camera view before the
// per-point Top-5 reduction.  kBlendedViews is only the final-coloring
// candidate budget; using it here silently restricted datasets with more
// than six four-camera captures to their first 24 views.
struct ExposureVoxelSelectedViews {
    std::vector<int> indices;
};

struct ExposureVoxelAccumulator {
    // OctreeCentroidContainer accumulates PointT fields in float, in input
    // order.  Keeping these sums in double moves the resulting centroids by
    // several ULPs and changes projection/visibility decisions at boundaries.
    std::array<float, 8> surface_sum{};
    std::array<double, 3> color_sum{};
    std::uint32_t count = 0U;
};

struct ExposurePoint {
    SurfacePoint surface{};
    cv::Vec3b source_rgb{};
    bool has_source_rgb = false;
};

struct ExposureObservation {
    int view = -1;
    std::uint8_t intensity = 0U;
    float weight = 0.0F;
};

struct ExposureJointSample {
    std::array<ExposureObservation, kSelectedViews> observations{};
    std::uint8_t count = 0U;
};

struct ExposureDynamicRange {
    int view = -1;
    std::uint8_t low = 0U;
    std::uint8_t high = 0U;
    double normalized_weight = 0.0;
    double loss_scale = 0.0;
};

struct ExposureSceneRange {
    int view = -1;
    std::uint8_t low = 0U;
    std::uint8_t high = 0U;
    std::uint8_t median = 0U;
    // SceneBrightness's binary item stores this value as float.  The residual
    // promotes the stored float (cvtss2sd in the double instantiation).
    float normalized_weight = 0.0F;
};

struct ExposureProblem {
    std::vector<ExposureJointSample> joint_samples;
    std::vector<ExposureDynamicRange> dynamic_ranges;
    std::vector<ExposureSceneRange> scene_ranges;
    std::vector<int> active_views;
};

void normalizeDynamicRanges(std::vector<ExposureDynamicRange>& ranges, double joint_count) {
    if (ranges.empty()) {
        return;
    }

    // The vendor's DynamicRange helper normalizes the already globally
    // normalized view weights once more.  Its loss scale is then formed as
    // (total_scale / range_count) * (range_count * normalized_weight).  Keep
    // both operations explicit: their rounding is observable in Ceres's
    // robustified residual and Jacobian.
    const double weight_sum = std::accumulate(
        ranges.begin(), ranges.end(), 0.0,
        [](double sum, const ExposureDynamicRange& range) {
            return sum + std::abs(range.normalized_weight);
        });
    if (weight_sum <= 0.0) {
        return;
    }

    const double range_count = static_cast<double>(ranges.size());
    const double average_loss_scale = (joint_count * 1.0e-4) / range_count;
    for (ExposureDynamicRange& range : ranges) {
        range.normalized_weight /= weight_sum;
        range.loss_scale =
            average_loss_scale * (range_count * range.normalized_weight);
    }
}

Options parseArguments(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument("missing value after " + argument);
            }
            return argv[i];
        };
        if (argument == "--input") {
            options.input = value();
        } else if (argument == "--output") {
            options.output = value();
        } else if (argument == "--panorama-dir") {
            options.panorama_directory = value();
        } else if (argument == "--info-dir") {
            options.info_directory = value();
        } else if (argument == "--camera-dir") {
            options.camera_directory = value();
        } else if (argument == "--camera-mask-dir") {
            options.camera_mask_directory = value();
        } else if (argument == "--sensor-frame") {
            options.sensor_frame = value();
        } else if (argument == "--direct-mask") {
            options.direct_mask = value();
        } else if (argument == "--exposure-models") {
            options.exposure_models = value();
        } else if (argument == "--exposure-model-output") {
            options.exposure_model_output = value();
        } else if (argument == "--exposure-ovs-output") {
            options.exposure_ovs_output = value();
        } else if (argument == "--exposure-ovs-binary-output") {
            options.exposure_ovs_binary_output = value();
        } else if (argument == "--exposure-ovs-binary-input") {
            options.exposure_ovs_binary_input = value();
        } else if (argument == "--color-ovs-output") {
            options.color_ovs_output = value();
        } else if (argument == "--color-ovs-input") {
            options.color_ovs_input = value();
        } else if (argument == "--exposure-cloud-output") {
            options.exposure_cloud_output = value();
        } else if (argument == "--exposure-projection-output") {
            options.exposure_projection_output = value();
        } else if (argument == "--depth-map-input-dir") {
            options.depth_map_input_directory = value();
        } else if (argument == "--depth-map-output-dir") {
            options.depth_map_output_directory = value();
        } else if (argument == "--pct-surfel-input") {
            options.pct_surfel_input = value();
        } else if (argument == "--pct-surfel-output") {
            options.pct_surfel_output = value();
        } else if (argument == "--input-resolution") {
            options.input_resolution = std::stof(value());
        } else if (argument == "--depth-width") {
            options.depth_width = std::stoi(value());
        } else if (argument == "--depth-views") {
            options.depth_views = std::stoi(value());
        } else if (argument == "--depth-splat-radius") {
            options.depth_splat_radius = std::stoi(value());
        } else if (argument == "--color-views") {
            options.color_views = std::stoi(value());
        } else if (argument == "--visibility-tolerance") {
            options.visibility_tolerance = std::stof(value());
        } else if (argument == "--visibility-patch-radius") {
            options.visibility_patch_radius = std::stoi(value());
        } else if (argument == "--visibility-min-fraction") {
            options.visibility_min_fraction = std::stof(value());
        } else if (argument == "--view-max-dist") {
            options.view_max_distance = std::stof(value());
        } else if (argument == "--chunk-points") {
            options.chunk_points = std::stoull(value());
        } else if (argument == "--image-cache") {
            options.image_cache = std::stoull(value());
        } else if (argument == "--camera-cache") {
            options.camera_cache = std::stoull(value());
        } else if (argument == "--camera-index") {
            options.camera_index = std::stoi(value());
        } else if (argument == "--exposure-solver-threads") {
            options.exposure_solver_threads = std::stoi(value());
        } else if (argument == "--exposure-solver-max-iterations") {
            options.exposure_solver_max_iterations = std::stoi(value());
        } else if (argument == "--exposure-solver-initial-trust-region-radius") {
            options.exposure_solver_initial_trust_region_radius = std::stod(value());
        } else if (argument == "--score-incidence-power") {
            options.score_incidence_power = std::stof(value());
        } else if (argument == "--score-radius-power") {
            options.score_radius_power = std::stof(value());
        } else if (argument == "--score-distance-power") {
            options.score_distance_power = std::stof(value());
        } else if (argument == "--direct-blend") {
            const std::string mode = value();
            options.average_direct_views = mode == "average";
            options.median_direct_views = mode == "median";
            options.robust_direct_views = mode == "robust";
            if (mode != "average" && mode != "median" && mode != "incidence" && mode != "robust") {
                throw std::invalid_argument(
                    "--direct-blend must be robust, average, median, or incidence");
            }
        } else if (argument == "--exposure") {
            const std::string mode = value();
            if (mode == "global") {
                options.global_exposure = true;
            } else if (mode == "none") {
                options.global_exposure = false;
            } else {
                throw std::invalid_argument("--exposure must be global or none");
            }
        } else if (argument == "--exposure-only") {
            options.exposure_only = true;
        } else if (argument == "--exposure-capture-only") {
            options.exposure_capture_only = true;
        } else if (argument == "--ignore-exposure-source-color") {
            options.exposure_source_consistency = false;
        } else if (argument == "--exposure-histogram-exclude-zero") {
            options.exposure_histogram_exclude_zero = true;
        } else if (argument == "--color-extrapolation") {
            const std::string mode = value();
            if (mode == "fill") {
                options.knn_extrapolation = true;
            } else if (mode == "panorama") {
                options.knn_extrapolation = false;
            } else {
                throw std::invalid_argument("--color-extrapolation must be fill or panorama");
            }
        } else if (argument == "--pointwise-view-selection") {
            options.voxel_view_selection = false;
        } else if (argument == "--voxel-view-selection") {
            options.voxel_view_selection = true;
        } else if (argument == "--standard-histogram") {
            options.standard_histogram = true;
        } else if (argument == "--no-standard-histogram") {
            options.standard_histogram = false;
        } else if (argument == "--help") {
            std::cout << "Usage: navvis_recon_surface_colorizer --input surface-or-colored.ply "
                         "--output pointcloud.ply --panorama-dir DIR --info-dir DIR "
                         "[--camera-dir DIR --sensor-frame sensor_frame.xml] "
                         "[--camera-mask-dir DIR] "
                         "[--direct-mask direct_mask.u8] "
                         "[--exposure-models view_gain_exponent.txt] "
                         "[--exposure-model-output FILE] [--exposure-only] "
                         "[--exposure-capture-only] "
                         "[--exposure-ovs-output FILE] "
                         "[--exposure-ovs-binary-output RAW_5X8] "
                         "[--exposure-ovs-binary-input RAW_5X8] "
                         "[--color-ovs-input RAW_5X8] "
                         "[--color-ovs-output RAW_5X8] "
                         "[--exposure-cloud-output FILE] "
                         "[--exposure-projection-output FILE] "
                         "[--depth-map-input-dir DIR] "
                         "[--depth-map-output-dir DIR] "
                         "[--pct-surfel-input RAW_F32] [--pct-surfel-output RAW_F32] "
                         "[--input-resolution 0.01] "
                         "[--exposure-solver-threads 32] "
                         "[--exposure-solver-max-iterations 50] "
                         "[--exposure-solver-initial-trust-region-radius 10000] "
                         "[--ignore-exposure-source-color] "
                         "[--exposure-histogram-exclude-zero] "
                         "[--direct-blend robust|incidence|average|median] "
                         "[--exposure global|none] [--color-extrapolation fill|panorama] "
                         "[--voxel-view-selection] "
                         "[--standard-histogram] "
                         "[--depth-width 684] [--depth-views 24] [--depth-splat-radius 1] "
                         "[--color-views 24] "
                         "[--camera-index 0..3] "
                         "[--exposure-solver-threads 32] "
                         "[--score-incidence-power 0.25 --score-radius-power 5 "
                         "--score-distance-power 2] "
                         "[--visibility-tolerance 0.50] [--visibility-patch-radius 1] "
                         "[--visibility-min-fraction 0.56] [--view-max-dist 30] "
                         "[--chunk-points 2000000]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.input.empty() || options.output.empty() || options.panorama_directory.empty() ||
        options.info_directory.empty()) {
        throw std::invalid_argument(
            "--input, --output, --panorama-dir, and --info-dir are required");
    }
    if (options.depth_width < 64 || options.depth_width % 2 != 0 || options.depth_views < 1 ||
        options.color_views < 1 || options.depth_views > static_cast<int>(kCandidateViews) ||
        options.color_views > static_cast<int>(kBlendedViews) || options.depth_splat_radius < 0 ||
        options.depth_splat_radius > 4 || options.chunk_points == 0U || options.image_cache == 0U ||
        options.camera_cache == 0U || options.exposure_solver_threads < 1 ||
        options.exposure_solver_max_iterations < 1 ||
        !(options.exposure_solver_initial_trust_region_radius > 0.0) ||
        !std::isfinite(options.exposure_solver_initial_trust_region_radius) ||
        options.camera_index < -1 || options.camera_index > 3 ||
        options.view_max_distance <= 0.0F || options.visibility_patch_radius < 0 ||
        options.visibility_patch_radius > 3 || options.visibility_min_fraction <= 0.0F ||
        options.visibility_min_fraction > 1.0F || options.score_incidence_power < 0.0F ||
        options.score_radius_power < 0.0F || options.score_distance_power < 0.0F) {
        throw std::invalid_argument("invalid depth/view/chunk/cache option");
    }
    if (!(options.input_resolution > 0.0F) || !std::isfinite(options.input_resolution)) {
        throw std::invalid_argument("invalid input resolution");
    }
    if (options.camera_directory.empty() != options.sensor_frame.empty()) {
        throw std::invalid_argument("--camera-dir and --sensor-frame must be supplied together");
    }
    if (!options.direct_mask.empty() && options.camera_directory.empty()) {
        throw std::invalid_argument("--direct-mask requires --camera-dir and --sensor-frame");
    }
    return options;
}

std::vector<GammaModel> loadExposureModels(const fs::path& path, std::size_t view_count) {
    std::vector<GammaModel> models(view_count);
    if (path.empty()) {
        return models;
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open exposure model file " + path.string());
    }
    std::vector<bool> seen(view_count, false);
    std::string line;
    std::size_t line_number = 0U;
    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        std::istringstream fields(line);
        std::size_t view = 0U;
        GammaModel model;
        if (!(fields >> view)) {
            continue;
        }
        if (!(fields >> model.gain >> model.exponent)) {
            throw std::runtime_error("malformed exposure model at line " +
                                     std::to_string(line_number));
        }
        if (view >= view_count || !std::isfinite(model.gain) || !std::isfinite(model.exponent) ||
            model.gain <= 0.0 || model.exponent <= 0.0) {
            throw std::runtime_error("invalid exposure model entry in " + path.string());
        }
        models[view] = model;
        seen[view] = true;
    }
    const std::size_t count = static_cast<std::size_t>(std::count(seen.begin(), seen.end(), true));
    std::cerr << "Loaded " << count << '/' << view_count << " per-view GammaModel entries from "
              << path << '\n';
    return models;
}

PlyInput readPlyHeader(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::string line;
    std::uint64_t count = 0U;
    std::vector<std::pair<std::string, std::string>> properties;
    while (std::getline(input, line)) {
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(^element vertex ([0-9]+)$)"))) {
            count = std::stoull(match[1].str());
        } else if (std::regex_match(line, match, std::regex(R"(^property ([^ ]+) ([^ ]+)$)"))) {
            properties.emplace_back(match[1].str(), match[2].str());
        } else if (line == "end_header") {
            break;
        }
    }
    const std::vector<std::pair<std::string, std::string>> expected_surface{
        {"float", "x"},  {"float", "y"},  {"float", "z"},  {"float", "intensity"},
        {"float", "nx"}, {"float", "ny"}, {"float", "nz"}, {"float", "curvature"}};
    const std::vector<std::pair<std::string, std::string>> expected_colored{
        {"float", "x"},     {"float", "y"},    {"float", "z"},     {"uchar", "red"},
        {"uchar", "green"}, {"uchar", "blue"}, {"uchar", "alpha"}, {"float", "intensity"},
        {"float", "nx"},    {"float", "ny"},   {"float", "nz"},    {"float", "curvature"}};
    const bool colored_records = properties == expected_colored;
    if (count == 0U || (properties != expected_surface && !colored_records)) {
        throw std::runtime_error(
            "input must use the 8-float surface or 36-byte colored PLY layout");
    }
    const auto offset = input.tellg();
    const std::uint64_t expected_size =
        static_cast<std::uint64_t>(offset) +
        count * (colored_records ? sizeof(ColoredSurfacePoint) : sizeof(SurfacePoint));
    if (fs::file_size(path) != expected_size) {
        throw std::runtime_error("input PLY is incomplete or has an unexpected record size");
    }
    return {count, offset, colored_records};
}

void readSurfacePoints(std::ifstream& input, bool colored_records,
                       std::vector<SurfacePoint>& points, std::size_t count,
                       std::vector<cv::Vec3b>* source_colors = nullptr) {
    points.resize(count);
    if (!colored_records) {
        if (source_colors != nullptr) {
            source_colors->clear();
        }
        input.read(reinterpret_cast<char*>(points.data()),
                   static_cast<std::streamsize>(count * sizeof(SurfacePoint)));
    } else {
        std::vector<ColoredSurfacePoint> colored(count);
        if (source_colors != nullptr) {
            source_colors->resize(count);
        }
        input.read(reinterpret_cast<char*>(colored.data()),
                   static_cast<std::streamsize>(count * sizeof(ColoredSurfacePoint)));
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
            const auto& source = colored[static_cast<std::size_t>(i)];
            points[static_cast<std::size_t>(i)] = {source.x,         source.y,        source.z,
                                                   source.intensity, source.nx,       source.ny,
                                                   source.nz,        source.curvature};
            if (source_colors != nullptr) {
                (*source_colors)[static_cast<std::size_t>(i)] =
                    cv::Vec3b(source.red, source.green, source.blue);
            }
        }
    }
    if (!input) {
        throw std::runtime_error("failed while reading input PLY records");
    }
}

std::vector<fs::path> sortedInfoFiles(const fs::path& directory) {
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() &&
            entry.path().filename().string().find("-info.json") != std::string::npos) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::array<double, 3> parseTriple(const std::string& text, const std::string& field) {
    const std::string number = R"(([-+0-9.eE]+))";
    const std::regex expression("\\\"" + field + "\\\"\\s*:\\s*\\[\\s*" + number + "\\s*,\\s*" +
                                number + "\\s*,\\s*" + number + "\\s*\\]");
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error("missing " + field + " in capture info");
    }
    return {std::stod(match[1].str()), std::stod(match[2].str()), std::stod(match[3].str())};
}

std::array<double, 4> parseQuaternion(const std::string& text) {
    const std::string number = R"(([-+0-9.eE]+))";
    const std::regex expression("\\\"quaternion\\\"\\s*:\\s*\\[\\s*" + number + "\\s*,\\s*" +
                                number + "\\s*,\\s*" + number + "\\s*,\\s*" + number + "\\s*\\]");
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error("missing quaternion in capture info");
    }
    return {std::stod(match[1].str()), std::stod(match[2].str()), std::stod(match[3].str()),
            std::stod(match[4].str())};
}

Pose poseFromJsonSection(const std::string& section) {
    const auto position = parseTriple(section, "position");
    const auto quaternion = parseQuaternion(section); // WXYZ in info JSON.
    Pose pose;
    pose.translation = Vec3f(static_cast<float>(position[0]), static_cast<float>(position[1]),
                             static_cast<float>(position[2]));
    pose.rotation =
        Eigen::Quaternionf(static_cast<float>(quaternion[0]), static_cast<float>(quaternion[1]),
                           static_cast<float>(quaternion[2]), static_cast<float>(quaternion[3]))
            .normalized();

    // Runtime tracing of nv_colorcloud's PCT renderer shows this exact pose
    // conversion path: the unit Quaterniond is converted to AngleAxisd, the
    // resulting rotation vector is cast component-wise to float, and only
    // then is its norm/axis used to rebuild the float matrix with Rodrigues.
    // Constructing Quaternionf directly changes several matrix entries by one
    // or two ULP and can move an oblique surfel intersection across a 1 mm
    // depth quantization boundary.
    const Eigen::Quaterniond quaternion_double(quaternion[0], quaternion[1], quaternion[2],
                                               quaternion[3]);
    const Eigen::AngleAxisd angle_axis_double(quaternion_double);
    const Eigen::Vector3d rotation_vector_double =
        angle_axis_double.angle() * angle_axis_double.axis();
    const Vec3f rotation_vector = rotation_vector_double.cast<float>();
    float squared_angle = rotation_vector.z() * rotation_vector.z();
    squared_angle += rotation_vector.y() * rotation_vector.y();
    squared_angle += rotation_vector.x() * rotation_vector.x();
    const float angle = std::sqrt(squared_angle);
    if (angle > 0.0F && std::isfinite(angle)) {
        // Pose<float>'s conversion computes one reciprocal with divss and
        // multiplies all three components by it.  Three independent vector
        // divisions produce a different x-axis bit on this capture.
        const float inverse_angle = 1.0F / angle;
        pose.rotation_matrix =
            Eigen::AngleAxisf(angle, rotation_vector * inverse_angle).toRotationMatrix();
    } else {
        pose.rotation_matrix = Mat3f::Identity();
    }
    pose.has_rotation_matrix = true;
    pose.translation_double = Eigen::Vector3d(position[0], position[1], position[2]);
    pose.rotation_matrix_double = quaternion_double.toRotationMatrix();
    pose.has_double_pose = true;
    return pose;
}

std::string jsonObjectSection(const std::string& json, const std::string& name) {
    const auto begin = json.find("\"" + name + "\"");
    if (begin == std::string::npos) {
        throw std::runtime_error("missing " + name + " in capture info");
    }
    const auto end = json.find("\n    }", begin);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated " + name + " in capture info");
    }
    return json.substr(begin, end - begin);
}

std::vector<CapturePoses> readCapturePoses(const fs::path& directory) {
    std::vector<CapturePoses> poses;
    for (const auto& path : sortedInfoFiles(directory)) {
        std::ifstream input(path);
        const std::string json((std::istreambuf_iterator<char>(input)), {});
        CapturePoses capture;
        capture.head = poseFromJsonSection(jsonObjectSection(json, "cam_head"));
        for (std::size_t camera = 0; camera < capture.cameras.size(); ++camera) {
            capture.cameras[camera] =
                poseFromJsonSection(jsonObjectSection(json, "cam" + std::to_string(camera)));
        }
        poses.push_back(capture);
    }
    if (poses.empty()) {
        throw std::runtime_error("no capture poses in " + directory.string());
    }
    return poses;
}

std::string readTextFile(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return std::string((std::istreambuf_iterator<char>(input)), {});
}

std::string xmlElement(const std::string& text, const std::string& tag) {
    const std::regex expression("<" + tag + R"((?:\s[^>]*)?>\s*([\s\S]*?)\s*</)" + tag + ">",
                                std::regex::icase);
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error("missing XML element " + tag);
    }
    return match[1].str();
}

double xmlDouble(const std::string& text, const std::string& tag) {
    return std::stod(xmlElement(text, tag));
}

std::array<OCamCamera, 4> readCameraModels(const fs::path& sensor_frame) {
    const std::string xml = readTextFile(sensor_frame);
    const std::regex expression(R"(<CameraModel>\s*([\s\S]*?)\s*</CameraModel>)");
    std::vector<std::pair<std::string, OCamCamera>> parsed;
    for (auto iterator = std::sregex_iterator(xml.begin(), xml.end(), expression);
         iterator != std::sregex_iterator(); ++iterator) {
        const std::string block = (*iterator)[1].str();
        OCamCamera camera;
        const std::string ocam = xmlElement(block, "OCamModel");
        camera.c = xmlDouble(ocam, "c");
        camera.d = xmlDouble(ocam, "d");
        camera.e = xmlDouble(ocam, "e");
        camera.cx = xmlDouble(ocam, "cx");
        camera.cy = xmlDouble(ocam, "cy");
        const std::string polynomial = xmlElement(ocam, "world2cam");
        const std::regex coefficients(R"(<coeff>\s*([^<]+?)\s*</coeff>)");
        for (auto coefficient =
                 std::sregex_iterator(polynomial.begin(), polynomial.end(), coefficients);
             coefficient != std::sregex_iterator(); ++coefficient) {
            camera.world_to_camera.push_back(std::stod((*coefficient)[1].str()));
        }
        const std::string inverse_polynomial = xmlElement(ocam, "cam2world");
        for (auto coefficient = std::sregex_iterator(inverse_polynomial.begin(),
                                                     inverse_polynomial.end(), coefficients);
             coefficient != std::sregex_iterator(); ++coefficient) {
            camera.camera_to_world.push_back(std::stod((*coefficient)[1].str()));
        }
        parsed.emplace_back(xmlElement(block, "SensorName"), std::move(camera));
    }
    std::sort(parsed.begin(), parsed.end(),
              [](const auto& first, const auto& second) { return first.first < second.first; });
    if (parsed.size() != 4U) {
        throw std::runtime_error("expected four OCam camera models");
    }
    std::array<OCamCamera, 4> result;
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = std::move(parsed[index].second);
        // Camera poses in the info JSON use the same OCam axis convention for
        // all four sensors.  JPEG EXIF differs for cam0, but raw decoding below
        // deliberately ignores EXIF, so every camera uses (Y, X, -Z).
        result[index].transverse_orientation = 6;
    }
    return result;
}

Vec3f xyz(const SurfacePoint& point) {
    return {point.x, point.y, point.z};
}

VoxelKey tileKey(const Vec3f& point) {
    return {static_cast<int>(std::floor(point.x() / 10.0F)),
            static_cast<int>(std::floor(point.y() / 10.0F)),
            static_cast<int>(std::floor(point.z() / 10.0F))};
}

VoxelKey ovsVoxelKey(const Vec3f& point) {
    return {static_cast<int>(std::floor(point.x())), static_cast<int>(std::floor(point.y())),
            static_cast<int>(std::floor(point.z()))};
}

CandidateList candidatesForTile(const VoxelKey& key, const std::vector<Pose>& poses) {
    const Vec3f center(10.0F * (static_cast<float>(key.x) + 0.5F),
                       10.0F * (static_cast<float>(key.y) + 0.5F),
                       10.0F * (static_cast<float>(key.z) + 0.5F));
    std::vector<std::pair<float, int>> distances;
    distances.reserve(poses.size());
    for (std::size_t index = 0; index < poses.size(); ++index) {
        distances.emplace_back((poses[index].translation - center).squaredNorm(),
                               static_cast<int>(index));
    }
    const std::size_t keep = std::min(kCandidateViews, distances.size());
    std::partial_sort(distances.begin(), distances.begin() + keep, distances.end());
    CandidateList result;
    result.count = keep;
    for (std::size_t index = 0; index < keep; ++index) {
        result.indices[index] = distances[index].second;
    }
    return result;
}

std::vector<std::uint16_t>
prepareCandidateSlots(const std::vector<SurfacePoint>& points, const std::vector<Pose>& poses,
                      std::unordered_map<VoxelKey, CandidateList, VoxelKeyHash>& cache,
                      std::vector<CandidateList>& local_lists) {
    std::unordered_map<VoxelKey, std::uint16_t, VoxelKeyHash> local_slots;
    std::vector<std::uint16_t> slots(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        const VoxelKey key = tileKey(xyz(points[index]));
        auto found = local_slots.find(key);
        if (found == local_slots.end()) {
            auto cached = cache.find(key);
            if (cached == cache.end()) {
                cached = cache.emplace(key, candidatesForTile(key, poses)).first;
            }
            if (local_lists.size() >= std::numeric_limits<std::uint16_t>::max()) {
                throw std::runtime_error("too many spatial tiles in one chunk");
            }
            const auto slot = static_cast<std::uint16_t>(local_lists.size());
            local_lists.push_back(cached->second);
            found = local_slots.emplace(key, slot).first;
        }
        slots[index] = found->second;
    }
    return slots;
}

std::vector<int> conservativeCandidatesForOvsVoxel(const VoxelKey& key,
                                                   const std::vector<Pose>& poses,
                                                   float maximum_distance, int camera_index) {
    // VoxelRanking in the installed worker rejects camera/voxel pairs before
    // running the point-level projector.  Reproduce only mathematically safe
    // rejections here: a camera is retained whenever any point in this
    // world-aligned one-metre cell could be in front of it and within range.
    // Retained view IDs stay ascending so SelectedViews_<5> tie behaviour is
    // identical to the former all-view loop.
    const Vec3f center(static_cast<float>(key.x) + 0.5F, static_cast<float>(key.y) + 0.5F,
                       static_cast<float>(key.z) + 0.5F);
    constexpr float voxel_half_diagonal = 0.8660254037844386F;
    constexpr float float_safety_margin = 1.0e-3F;
    std::vector<int> result;
    result.reserve(poses.size());
    for (std::size_t view = 0; view < poses.size(); ++view) {
        if (camera_index >= 0 && static_cast<int>(view % 4U) != camera_index) {
            continue;
        }
        const Pose& pose = poses[view];
        const Mat3f inverse_rotation = pose.rotationMatrix().transpose();
        const Vec3f local_center = inverse_rotation * (center - pose.translation);
        // ||R^T delta|| <= ||R^T||_F ||delta||.  This deliberately loose
        // Frobenius bound plus a float margin cannot reject a point that the
        // original float range test would accept.
        const float range_margin =
            inverse_rotation.norm() * voxel_half_diagonal + float_safety_margin;
        if (local_center.norm() - range_margin > maximum_distance) {
            continue;
        }

        double local_z = 0.0;
        double local_z_margin = 0.0;
        if (pose.has_double_pose) {
            const Eigen::Vector3d forward = pose.rotation_matrix_double.col(2);
            local_z = forward.dot(center.cast<double>() - pose.translation_double);
            local_z_margin = 0.5 * forward.cwiseAbs().sum();
        } else {
            const Vec3f forward = pose.rotationMatrix().col(2);
            local_z = static_cast<double>(forward.dot(center - pose.translation));
            local_z_margin = 0.5 * static_cast<double>(forward.cwiseAbs().sum());
        }
        // projectOCam requires strictly positive local Z (apart from the
        // cos(pi/2) round-off threshold).  Cull only with an outward epsilon.
        if (local_z + local_z_margin < -1.0e-9) {
            continue;
        }
        result.push_back(static_cast<int>(view));
    }
    return result;
}

std::vector<const std::vector<int>*>
prepareConservativeCandidateLists(const std::vector<SurfacePoint>& points,
                                  const std::vector<Pose>& poses, float maximum_distance,
                                  int camera_index, ConservativeCandidateCache& cache) {
    std::vector<const std::vector<int>*> lists(points.size(), nullptr);
    for (std::size_t index = 0; index < points.size(); ++index) {
        const VoxelKey key = ovsVoxelKey(xyz(points[index]));
        auto found = cache.find(key);
        if (found == cache.end()) {
            found = cache
                        .emplace(key, conservativeCandidatesForOvsVoxel(
                                          key, poses, maximum_distance, camera_index))
                        .first;
        }
        lists[index] = &found->second;
    }
    return lists;
}

std::array<int, kCandidateViews> nearestViews(const Vec3f& point, const CandidateList& candidates,
                                              const std::vector<Pose>& poses, int requested) {
    std::array<std::pair<float, int>, kCandidateViews> ranked{};
    for (std::size_t index = 0; index < candidates.count; ++index) {
        const int view = candidates.indices[index];
        ranked[index] = {(poses[view].translation - point).squaredNorm(), view};
    }
    const std::size_t keep = std::min<std::size_t>(requested, candidates.count);
    std::partial_sort(ranked.begin(), ranked.begin() + keep, ranked.begin() + candidates.count);
    std::array<int, kCandidateViews> result{};
    result.fill(-1);
    for (std::size_t index = 0; index < keep; ++index) {
        result[index] = ranked[index].second;
    }
    return result;
}

bool project(const Vec3f& point, const Pose& pose, int width, int height, int& pixel,
             float& range) {
    const Vec3f local = pose.inverseApply(point);
    range = local.norm();
    if (!(range > 1.0e-4F) || !std::isfinite(range)) {
        return false;
    }
    const float longitude = std::atan2(local.y(), local.x());
    const float latitude =
        std::atan2(local.z(), std::max(std::hypot(local.x(), local.y()), 1.0e-9F));
    int column = static_cast<int>(std::lround((longitude / (2.0F * kPi) + 0.5F) * width));
    column %= width;
    if (column < 0) {
        column += width;
    }
    const int row =
        std::clamp(static_cast<int>(std::lround((0.5F - latitude / kPi) * height)), 0, height - 1);
    pixel = row * width + column;
    return true;
}

std::uint32_t floatBits(float value) {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

float bitsFloat(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void atomicMinimum(std::atomic<std::uint32_t>& target, float value) {
    const std::uint32_t candidate = floatBits(value);
    std::uint32_t current = target.load(std::memory_order_relaxed);
    while (candidate < current &&
           !target.compare_exchange_weak(current, candidate, std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}

double evaluatePolynomial(const std::vector<double>& coefficients, double value) {
    double result = 0.0;
    for (auto coefficient = coefficients.rbegin(); coefficient != coefficients.rend();
         ++coefficient) {
        result = result * value + *coefficient;
    }
    return result;
}

std::pair<double, double> transverseCoordinates(int orientation, double x, double y) {
    if (orientation == 6) {
        return {y, x};
    }
    if (orientation == 7) {
        return {-y, -x};
    }
    throw std::runtime_error("unsupported camera transverse orientation");
}

CameraMasks readCameraMasks(const fs::path& directory) {
    CameraMasks result;
    if (directory.empty()) {
        return result;
    }
    const char* weight_debug_directory = std::getenv("NAVVIS_DEBUG_WEIGHT_MAP_DIR");
    const auto dumpWeightMap = [&](const cv::Mat& map, const std::string& name) {
        if (weight_debug_directory == nullptr || map.empty()) {
            return;
        }
        if (map.type() != CV_32F || !map.isContinuous()) {
            throw std::runtime_error("weight-map regression dump requires continuous CV_32F data");
        }
        const fs::path output_directory(weight_debug_directory);
        fs::create_directories(output_directory);
        std::ofstream output(output_directory / (name + ".f32"), std::ios::binary);
        output.write(reinterpret_cast<const char*>(map.ptr<float>()),
                     static_cast<std::streamsize>(map.total() * sizeof(float)));
        if (!output) {
            throw std::runtime_error("cannot write weight-map regression dump");
        }
    };
    for (std::size_t camera = 0; camera < result.valid.size(); ++camera) {
        const fs::path path = directory / ("mask-cam" + std::to_string(camera) + ".png");
        const cv::Mat source = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
        if (source.empty()) {
            throw std::runtime_error("cannot decode camera mask " + path.string());
        }
        cv::Mat channel;
        if (source.channels() == 1) {
            channel = source;
        } else {
            // The green channel is binary in all G11 masks.  In cam1/cam2 it
            // also ignores a transparent PNG encoder marker in the blue plane.
            cv::extractChannel(source, channel, std::min(1, source.channels() - 1));
        }
        cv::threshold(channel, result.valid[camera], 127.0, 255.0, cv::THRESH_BINARY);

        // nv_colorcloud's point scorer does not use the binary mask directly
        // as its image-space quality.  optimal_view_selection_weight_maps.cpp
        // combines two independently generated maps:
        //
        //  1. The mask is linearly resized to 0.5 scale, thresholded at 128,
        //     transformed with precise L2 distance, and feathered across 900
        //     pixels with 0.5 - 0.5*cos(pi*d/900).
        //  2. A centered Gaussian (sigma = image dimension) is multiplied by
        //     a separable 300-pixel sine boundary ramp.
        //
        // Both maps are then linearly resized by another factor of 0.5,
        // multiplied, and infinity-normalized.  These constants, operation
        // order, and interpolation modes are recovered from the binary's
        // computeMaskBoundaryWeightMap/WeightMap2D hot path.
        constexpr float first_scale = 0.5F;
        constexpr float second_scale = 0.5F;
        constexpr float mask_falloff = 900.0F;
        constexpr float image_boundary_falloff = 300.0F;
        const cv::Size half_size(cvRound(channel.cols * first_scale),
                                 cvRound(channel.rows * first_scale));
        cv::Mat resized_mask;
        cv::resize(channel, resized_mask, half_size, 0.0, 0.0, cv::INTER_LINEAR);
        cv::threshold(resized_mask, resized_mask, 128.0, 255.0, cv::THRESH_BINARY);
        cv::Mat mask_weight;
        cv::distanceTransform(resized_mask, mask_weight, cv::DIST_L2, cv::DIST_MASK_PRECISE,
                              CV_32F);
        for (int row = 0; row < mask_weight.rows; ++row) {
            float* values = mask_weight.ptr<float>(row);
            for (int column_index = 0; column_index < mask_weight.cols; ++column_index) {
                const float q = std::clamp(values[column_index] / mask_falloff, 0.0F, 1.0F);
                values[column_index] = 0.5F + 0.5F * std::sin(kPi * (q - 0.5F));
            }
        }

        // WeightMap2D constructs the Gaussian and image-boundary maps as two
        // independent CV_32F images and combineWeightMaps multiplies them.
        // Do not reassociate this as (gx*bx)*(gy*by): although algebraically
        // equivalent, that changes float rounding relative to the installed
        // worker's (gx*gy)*(bx*by) evaluation order.
        cv::Mat gaussian_weight(half_size, CV_32F);
        cv::Mat boundary_weight(half_size, CV_32F);
        cv::Mat initial_weight;
        std::vector<float> gaussian_horizontal(static_cast<std::size_t>(half_size.width));
        std::vector<float> boundary_horizontal(static_cast<std::size_t>(half_size.width));
        const float gaussian_width = static_cast<float>(half_size.width);
        const float gaussian_height = static_cast<float>(half_size.height);
        const float gaussian_center_x = 0.5F * gaussian_width;
        const float gaussian_center_y = 0.5F * gaussian_height;
        const float gaussian_factor_x = 1.0F / ((gaussian_width + gaussian_width) * gaussian_width);
        const float gaussian_factor_y =
            1.0F / ((gaussian_height + gaussian_height) * gaussian_height);
        for (int column_index = 0; column_index < half_size.width; ++column_index) {
            const float coordinate = static_cast<float>(column_index) + 0.5F;
            const float gaussian_delta = coordinate - gaussian_center_x;
            float gaussian_exponent = gaussian_delta * gaussian_delta;
            gaussian_exponent = -gaussian_exponent;
            gaussian_exponent *= gaussian_factor_x;
            const float boundary_distance =
                std::min(coordinate, static_cast<float>(half_size.width) - coordinate);
            gaussian_horizontal[static_cast<std::size_t>(column_index)] =
                std::exp(gaussian_exponent);
            boundary_horizontal[static_cast<std::size_t>(column_index)] =
                std::sin(0.5F * kPi * std::min(boundary_distance / image_boundary_falloff, 1.0F));
        }
        for (int row = 0; row < half_size.height; ++row) {
            const float coordinate = static_cast<float>(row) + 0.5F;
            const float gaussian_delta = coordinate - gaussian_center_y;
            float gaussian_exponent = gaussian_delta * gaussian_delta;
            gaussian_exponent = -gaussian_exponent;
            gaussian_exponent *= gaussian_factor_y;
            const float boundary_distance =
                std::min(coordinate, static_cast<float>(half_size.height) - coordinate);
            const float gaussian_vertical = std::exp(gaussian_exponent);
            const float boundary_vertical =
                std::sin(0.5F * kPi * std::min(boundary_distance / image_boundary_falloff, 1.0F));
            float* gaussian_values = gaussian_weight.ptr<float>(row);
            float* boundary_values = boundary_weight.ptr<float>(row);
            for (int column_index = 0; column_index < half_size.width; ++column_index) {
                gaussian_values[column_index] =
                    gaussian_vertical * gaussian_horizontal[static_cast<std::size_t>(column_index)];
                boundary_values[column_index] =
                    boundary_vertical * boundary_horizontal[static_cast<std::size_t>(column_index)];
            }
        }
        cv::multiply(gaussian_weight, boundary_weight, initial_weight);
        if (camera == 0U) {
            dumpWeightMap(gaussian_weight, "cam0_initial_contribution_a");
            dumpWeightMap(boundary_weight, "cam0_initial_contribution_b");
        }
        // renormalizeWeightMap maps the complete observed range to [0, 1],
        // rather than merely dividing by the maximum.  The distinction is
        // visible at the four image corners: the analytic Gaussian/sine map
        // has a small positive minimum there which the binary subtracts.
        // The installed worker routes all three renormalization sites through
        // cv::normalize.  Keeping that call (instead of spelling the affine
        // transform as min/subtract/divide) matters: OpenCV evaluates the
        // scale and offset in double precision and changes the last bit of a
        // sizeable fraction of the float map.  Captured G11 intermediate maps
        // are bit-exact with this operation order.
        cv::normalize(initial_weight, initial_weight, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

        if (camera == 0U) {
            dumpWeightMap(mask_weight, "cam0_mask_weight_half");
            dumpWeightMap(initial_weight, "cam0_initial_weight_half");
        }

        const cv::Size point_weight_size(cvRound(half_size.width * second_scale),
                                         cvRound(half_size.height * second_scale));
        cv::Mat mask_weight_quarter;
        cv::Mat initial_weight_quarter;
        cv::resize(mask_weight, mask_weight_quarter, point_weight_size, 0.0, 0.0, cv::INTER_LINEAR);
        cv::resize(initial_weight, initial_weight_quarter, point_weight_size, 0.0, 0.0,
                   cv::INTER_LINEAR);
        cv::normalize(initial_weight_quarter, initial_weight_quarter, 0.0, 1.0, cv::NORM_MINMAX,
                      CV_32F);
        cv::multiply(mask_weight_quarter, initial_weight_quarter, result.point_weight[camera]);
        cv::normalize(result.point_weight[camera], result.point_weight[camera], 0.0, 1.0,
                      cv::NORM_MINMAX, CV_32F);

        if (camera == 0U) {
            dumpWeightMap(mask_weight_quarter, "cam0_mask_weight_quarter");
            dumpWeightMap(initial_weight_quarter, "cam0_initial_weight_quarter");
            dumpWeightMap(result.point_weight[camera], "cam0_point_weight_quarter");
        }

        if (camera == 0U) {
            result.width = result.valid[camera].cols;
            result.height = result.valid[camera].rows;
        } else if (result.valid[camera].cols != result.width ||
                   result.valid[camera].rows != result.height) {
            throw std::runtime_error("camera masks do not share one image size");
        }
    }
    return result;
}

float cameraPointWeight(const CameraMasks& masks, int camera, double image_x, double image_y) {
    if (camera < 0 || camera >= static_cast<int>(masks.point_weight.size())) {
        return 0.0F;
    }
    const cv::Mat& map = masks.point_weight[static_cast<std::size_t>(camera)];
    if (map.empty()) {
        return 1.0F;
    }
    const float scale_x = static_cast<float>(map.cols) / masks.width;
    const float scale_y = static_cast<float>(map.rows) / masks.height;
    float x = static_cast<float>(image_x) * scale_x;
    float y = static_cast<float>(image_y) * scale_y;
    if (!(x >= 0.0F && y >= 0.0F && x < map.cols && y < map.rows)) {
        return 0.0F;
    }

    // WeightMap2D::eval's linear mode treats integer image samples as being
    // centred at n+0.5 and clamps to half-pixel image bounds.
    constexpr float half_pixel_epsilon = 0.500001013F;
    x = std::clamp(x, half_pixel_epsilon, map.cols - half_pixel_epsilon) - 0.5F;
    y = std::clamp(y, half_pixel_epsilon, map.rows - half_pixel_epsilon) - 0.5F;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, map.cols - 1);
    const int y1 = std::min(y0 + 1, map.rows - 1);
    const float fx = x - x0;
    const float fy = y - y0;
    const float top = (1.0F - fx) * map.at<float>(y0, x0) + fx * map.at<float>(y0, x1);
    const float bottom = (1.0F - fx) * map.at<float>(y1, x0) + fx * map.at<float>(y1, x1);
    return (1.0F - fy) * top + fy * bottom;
}

bool cameraMaskValid(const CameraMasks& masks, int camera, double x, double y) {
    if (camera < 0 || camera >= static_cast<int>(masks.valid.size())) {
        return false;
    }
    // The OCam sampler accepts the complete image domain.  Frozen original
    // OVS records include projections in both outer half-pixel strips (for
    // example y=0.014 and y=3647.826 in a 3648-row G11 image), so excluding a
    // one-pixel frame here drops valid observations before depth testing.
    if (x < 0.0 || y < 0.0 || x >= masks.width || y >= masks.height) {
        return false;
    }
    const cv::Mat& mask = masks.valid[static_cast<std::size_t>(camera)];
    const int column =
        std::clamp(static_cast<int>(std::lround(x)), 0, std::max(0, masks.width - 1));
    const int row =
        std::clamp(static_cast<int>(std::lround(y)), 0, std::max(0, masks.height - 1));
    return mask.empty() || mask.at<std::uint8_t>(row, column) != 0U;
}

std::optional<OCamProjection> projectOCamGeometry(const Vec3f& point, const Pose& pose,
                                                  const OCamCamera& model) {
    const Vec3f local_float = pose.inverseApply(point);
    // OVS/direct-camera projection receives Pose<double> in nv_colorcloud;
    // only PCTDepthMapRenderer explicitly converts its inverse to Pose<float>.
    // Keep the float local vector for the scorer's range, but use the original
    // double pose for OCam image coordinates and hemisphere classification.
    Eigen::Vector3d local;
    if (pose.has_double_pose) {
        local = pose.rotation_matrix_double.transpose() *
                (point.cast<double>() - pose.translation_double);
    } else {
        local = local_float.cast<double>();
    }
    const auto [x, y] = transverseCoordinates(model.transverse_orientation, local.x(), local.y());
    const double norm = std::hypot(x, y);
    const double theta = std::atan2(-local.z(), std::max(norm, 1.0e-12));
    const double radius = evaluatePolynomial(model.world_to_camera, theta);
    // The installed point selector computes range before applying the camera
    // rotation.  Its scalar SSE sequence is dy*dy, + dz*dz, + dx*dx, sqrt.
    // Rotating first is mathematically equivalent, but moves visibility and
    // quality decisions by a few ULPs at their strict boundaries.
    const Vec3f world_delta = point - pose.translation;
    float squared_range = world_delta.y() * world_delta.y();
    squared_range += world_delta.z() * world_delta.z();
    squared_range += world_delta.x() * world_delta.x();
    const float range = std::sqrt(squared_range);
    if (!(range > 1.0e-4F) || !std::isfinite(range) ||
        local.z() / static_cast<double>(range) <= kOCamFrontHemisphereThreshold) {
        return std::nullopt;
    }
    const double inverse_norm = 1.0 / std::max(norm, 1.0e-12);
    const double xp = x * inverse_norm * radius;
    const double yp = y * inverse_norm * radius;
    OCamProjection projection;
    projection.image_y = xp * model.c + yp * model.d + model.cx;
    projection.image_x = xp * model.e + yp + model.cy;
    projection.radius = radius;
    projection.range = range;
    return projection;
}

std::optional<OCamProjection> projectOCam(const Vec3f& point, const Pose& pose,
                                          const OCamCamera& model, const CameraMasks& masks,
                                          int camera) {
    const auto projection = projectOCamGeometry(point, pose, model);
    if (!projection ||
        !cameraMaskValid(masks, camera, projection->image_x, projection->image_y)) {
        return std::nullopt;
    }
    return projection;
}

bool projectOCamDepth(const Vec3f& point, const Pose& pose, const OCamCamera& model,
                      const CameraMasks& masks, int camera, int width, int height, int& pixel,
                      float& range, float* depth_x = nullptr, float* depth_y = nullptr,
                      OCamProjection* projected = nullptr) {
    const auto projection = projectOCam(point, pose, model, masks, camera);
    if (!projection) {
        return false;
    }
    // Image2D::eval converts its double scale field to float, then multiplies
    // float OCam coordinates (cvtsd2ss followed by mulss at 0x239bd2..0x239bef).
    // Scaling in double before the cast changes depth-edge decisions by one ULP.
    const float scale_x = static_cast<float>(static_cast<double>(width) / masks.width);
    const float scale_y = static_cast<float>(static_cast<double>(height) / masks.height);
    const float x = static_cast<float>(projection->image_x) * scale_x;
    const float y = static_cast<float>(projection->image_y) * scale_y;
    const int column = std::clamp(static_cast<int>(std::floor(x)), 0, width - 1);
    const int row = std::clamp(static_cast<int>(std::floor(y)), 0, height - 1);
    pixel = row * width + column;
    range = projection->range;
    if (depth_x != nullptr) {
        *depth_x = x;
    }
    if (depth_y != nullptr) {
        *depth_y = y;
    }
    if (projected != nullptr) {
        *projected = *projection;
    }
    return true;
}

std::optional<Vec3f> unprojectOCamPixel(const OCamCamera& model, double image_x, double image_y) {
    // OCamModel::cam2world as used by nv_colorcloud's ray image renderer.
    // The affine inverse is written in the same (row, column) convention as
    // projectOCam above. Camera poses use the (Y, X, -Z) OCam axis mapping.
    const double affine_x = image_y - model.cx;
    const double affine_y = image_x - model.cy;
    const double determinant = model.c - model.d * model.e;
    if (!(std::abs(determinant) > 1.0e-15)) {
        return std::nullopt;
    }
    const double xp = (affine_x - model.d * affine_y) / determinant;
    const double yp = (-model.e * affine_x + model.c * affine_y) / determinant;
    const double radius = std::hypot(xp, yp);
    const double polynomial_z = evaluatePolynomial(model.camera_to_world, radius);

    Eigen::Vector3d local;
    if (model.transverse_orientation == 6) {
        local = Eigen::Vector3d(yp, xp, -polynomial_z);
    } else if (model.transverse_orientation == 7) {
        local = Eigen::Vector3d(-yp, -xp, -polynomial_z);
    } else {
        return std::nullopt;
    }
    const double norm = local.norm();
    if (!(norm > 1.0e-12) || !std::isfinite(norm)) {
        return std::nullopt;
    }
    local /= norm;
    // The binary rejects rays outside the front hemisphere. On the G11 cam0
    // top row this makes column 200 the first ray; the recovered direction is
    // bit-close to the runtime raycaster argument.
    if (!(local.z() > 0.0)) {
        return std::nullopt;
    }
    return local.cast<float>();
}

struct PctVoxelAccumulator {
    std::array<float, 8> sum{};
    std::uint32_t count = 0U;
};

std::vector<SurfacePoint> loadCapturedPctSurfels(const fs::path& path) {
    const std::uintmax_t bytes = fs::file_size(path);
    if (bytes == 0U || bytes % sizeof(SurfacePoint) != 0U) {
        throw std::runtime_error("captured PCT surfel file has an invalid size");
    }
    std::vector<SurfacePoint> surfels(static_cast<std::size_t>(bytes / sizeof(SurfacePoint)));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(surfels.data()), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("cannot read captured PCT surfels");
    }
    return surfels;
}

std::vector<SurfacePoint> buildCleanRoomPctSurfels(const Options& options, const PlyInput& layout) {
    // Binary-grounded PCT input reduction. downsampleCloud receives
    // max(0.06 m, input resolution) and uses OctreeVoxelGrid. Its PCL bounds
    // are a power-of-two cube centered on [min, max + 2^-14], and the leaf
    // key is trunc((point - cube_min) / resolution). The fixed epsilon and
    // all bounds below were observed for both 0.06 m and 0.12 m runtime
    // probes. Octree field aggregators use float sums in input order.
    const double cell_size = std::max(0.06, static_cast<double>(options.input_resolution));
    constexpr double bounding_box_epsilon = 6.103515625e-05; // 2^-14

    Vec3f cloud_minimum = Vec3f::Constant(std::numeric_limits<float>::infinity());
    Vec3f cloud_maximum = Vec3f::Constant(-std::numeric_limits<float>::infinity());
    {
        std::ifstream input(options.input, std::ios::binary);
        input.seekg(layout.data_offset);
        std::vector<SurfacePoint> points(options.chunk_points);
        std::uint64_t processed = 0U;
        while (processed < layout.count) {
            const std::size_t chunk = static_cast<std::size_t>(
                std::min<std::uint64_t>(options.chunk_points, layout.count - processed));
            readSurfacePoints(input, layout.colored_records, points, chunk);
            for (const SurfacePoint& point : points) {
                cloud_minimum = cloud_minimum.cwiseMin(xyz(point));
                cloud_maximum = cloud_maximum.cwiseMax(xyz(point));
            }
            processed += chunk;
        }
    }
    const Eigen::Vector3d minimum = cloud_minimum.cast<double>();
    const Eigen::Vector3d maximum = cloud_maximum.cast<double>();
    const double maximum_extent = (maximum - minimum).maxCoeff() + bounding_box_epsilon;
    const int tree_depth =
        std::max(0, static_cast<int>(std::ceil(std::log2(maximum_extent / cell_size))));
    const double cube_side = std::ldexp(cell_size, tree_depth);
    const Eigen::Vector3d cube_minimum =
        0.5 * (minimum + maximum + Eigen::Vector3d::Constant(bounding_box_epsilon) -
               Eigen::Vector3d::Constant(cube_side));

    std::unordered_map<VoxelKey, PctVoxelAccumulator, VoxelKeyHash> voxels;
    voxels.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(layout.count, 4'000'000U)));
    std::ifstream input(options.input, std::ios::binary);
    input.seekg(layout.data_offset);
    std::vector<SurfacePoint> points(options.chunk_points);
    std::uint64_t processed = 0U;
    while (processed < layout.count) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(options.chunk_points, layout.count - processed));
        readSurfacePoints(input, layout.colored_records, points, chunk);
        for (const SurfacePoint& point : points) {
            const VoxelKey key{
                static_cast<int>((static_cast<double>(point.x) - cube_minimum.x()) / cell_size),
                static_cast<int>((static_cast<double>(point.y) - cube_minimum.y()) / cell_size),
                static_cast<int>((static_cast<double>(point.z) - cube_minimum.z()) / cell_size)};
            auto& voxel = voxels[key];

            // The PCL adapter normalizes every source normal before the
            // octree's NormalFieldAggregator sees it.  This is separate from
            // the representative normal normalization below.  The explicit
            // scalar order mirrors the adapter/library SSE path and matters
            // for cancellation-heavy leaves at millimetre depth boundaries.
            float input_normal_squared = point.nz * point.nz;
            input_normal_squared += point.ny * point.ny;
            input_normal_squared += point.nx * point.nx;
            float input_nx = point.nx;
            float input_ny = point.ny;
            float input_nz = point.nz;
            if (input_normal_squared > 0.0F && std::isfinite(input_normal_squared)) {
                const float input_normal_norm = std::sqrt(input_normal_squared);
                input_nx /= input_normal_norm;
                input_ny /= input_normal_norm;
                input_nz /= input_normal_norm;
            }
            const float values[8]{point.x,  point.y,  point.z,  point.intensity,
                                  input_nx, input_ny, input_nz, point.curvature};
            for (std::size_t field = 0; field < voxel.sum.size(); ++field) {
                voxel.sum[field] += values[field];
            }
            ++voxel.count;
        }
        processed += chunk;
        points.resize(options.chunk_points);
    }

    struct KeyedSurfel {
        VoxelKey key;
        SurfacePoint point;
    };
    std::vector<KeyedSurfel> keyed_surfels;
    keyed_surfels.reserve(voxels.size());
    for (const auto& [key, voxel] : voxels) {
        const float count = static_cast<float>(voxel.count);
        SurfacePoint surfel{
            voxel.sum[0] / count, voxel.sum[1] / count, voxel.sum[2] / count, 1.0F,
            voxel.sum[4] / count, voxel.sum[5] / count, voxel.sum[6] / count, voxel.sum[7] / count};
        // NormalFieldAggregator::getValue in libpointcloud_octree performs
        // scalar SSE operations in this exact order: (nz*nz + ny*ny) + nx*nx.
        // Preserve it here because changing the association moves normalized
        // components by one float ULP for a measurable subset of leaves.
        const float normal_x_squared = surfel.nx * surfel.nx;
        const float normal_y_squared = surfel.ny * surfel.ny;
        float squared_norm = surfel.nz * surfel.nz;
        squared_norm += normal_y_squared;
        squared_norm += normal_x_squared;
        if (squared_norm < 1.0e-6F || !std::isfinite(squared_norm)) {
            surfel.nx = 0.0F;
            surfel.ny = 0.0F;
            surfel.nz = 0.0F;
        } else if (std::abs(squared_norm - 1.0F) > 1.0e-6F) {
            const float normal_norm = std::sqrt(squared_norm);
            surfel.nx /= normal_norm;
            surfel.ny /= normal_norm;
            surfel.nz /= normal_norm;
        }
        keyed_surfels.push_back({key, surfel});
    }

    // PCL's depth-first octree iterator emits leaves in ascending Morton
    // order, with X/Y/Z as child bits 2/1/0 at every level.
    std::sort(keyed_surfels.begin(), keyed_surfels.end(),
              [tree_depth](const KeyedSurfel& first, const KeyedSurfel& second) {
                  for (int bit = tree_depth - 1; bit >= 0; --bit) {
                      const int first_values[3]{first.key.x, first.key.y, first.key.z};
                      const int second_values[3]{second.key.x, second.key.y, second.key.z};
                      for (int axis = 0; axis < 3; ++axis) {
                          const int first_bit = (first_values[axis] >> bit) & 1;
                          const int second_bit = (second_values[axis] >> bit) & 1;
                          if (first_bit != second_bit) {
                              return first_bit < second_bit;
                          }
                      }
                  }
                  return false;
              });
    std::vector<SurfacePoint> surfels;
    surfels.reserve(keyed_surfels.size());
    for (const KeyedSurfel& keyed : keyed_surfels) {
        surfels.push_back(keyed.point);
    }
    return surfels;
}

void renderPctDepthMaps(const std::vector<SurfacePoint>& surfels,
                        const std::vector<Pose>& view_poses,
                        const std::array<OCamCamera, 4>& camera_models, int full_width,
                        int full_height, int depth_width, int depth_height,
                        std::atomic<std::uint32_t>* depth) {
    const std::size_t pixels_per_view = static_cast<std::size_t>(depth_width) * depth_height;
    std::array<std::vector<Vec3f>, 4> local_rays;
    for (std::size_t camera = 0; camera < local_rays.size(); ++camera) {
        auto& rays = local_rays[camera];
        rays.assign(pixels_per_view, Vec3f::Zero());
        for (int row = 0; row < depth_height; ++row) {
            for (int column = 0; column < depth_width; ++column) {
                const double image_x =
                    (static_cast<double>(column) + 0.5) * full_width / depth_width;
                const double image_y =
                    (static_cast<double>(row) + 0.5) * full_height / depth_height;
                const auto ray = unprojectOCamPixel(camera_models[camera], image_x, image_y);
                if (ray) {
                    rays[static_cast<std::size_t>(row * depth_width + column)] = *ray;
                }
            }
        }
    }

    std::vector<std::vector<Vec3f>> world_rays(view_poses.size());
#pragma omp parallel for schedule(static)
    for (std::int64_t view = 0; view < static_cast<std::int64_t>(view_poses.size()); ++view) {
        auto& rays = world_rays[static_cast<std::size_t>(view)];
        rays.resize(pixels_per_view);
        const auto& source = local_rays[static_cast<std::size_t>(view % 4)];
        const Pose& pose = view_poses[static_cast<std::size_t>(view)];
        const Eigen::Matrix3f rotation = pose.rotationMatrix();
        for (std::size_t pixel = 0; pixel < pixels_per_view; ++pixel) {
            const Vec3f& local = source[pixel];
            Vec3f world;
            // The binary routes a local ray through transformPoint and then
            // subtracts the camera origin. Its generated SSE evaluates each
            // rotation row as (y term + z term) + x term, then performs the
            // otherwise-cancelling translation add/subtract. Both details can
            // move a component by one ULP at disc boundaries.
            for (int axis = 0; axis < 3; ++axis) {
                float component = rotation(axis, 1) * local.y();
                component += rotation(axis, 2) * local.z();
                component += rotation(axis, 0) * local.x();
                component += pose.translation[axis];
                component -= pose.translation[axis];
                world[axis] = component;
            }
            rays[pixel] = world;
        }
    }

    // Runtime PCT raycaster field +0x30 is 0.0107999993 for the G11 path.
    // It is (0.12 * sqrt(3) / 2)^2: the circumscribed radius of a 12 cm cell.
    constexpr float disc_radius_squared = 0.0107999993F;

    // The ray index is a cubic 6 cm grid. The binary expands the float point
    // maximum by 2^-14, rounds the longest cell count up to a power of two,
    // then symmetrically pads all axes. Candidate surfels are tested only when
    // the ray traverses the cell containing their centre; this is materially
    // different from treating every disc whose support touches the ray as a
    // candidate.
    constexpr double ray_cell_size = 0.06;
    constexpr float float_bbox_epsilon = 0.00006103515625F;
    Vec3f raw_minimum = Vec3f::Constant(std::numeric_limits<float>::infinity());
    Vec3f raw_maximum = Vec3f::Constant(-std::numeric_limits<float>::infinity());
    for (const SurfacePoint& surfel : surfels) {
        raw_minimum = raw_minimum.cwiseMin(xyz(surfel));
        raw_maximum = raw_maximum.cwiseMax(xyz(surfel));
    }
    raw_maximum.array() += float_bbox_epsilon;
    const Eigen::Vector3d raw_minimum_d = raw_minimum.cast<double>();
    const Eigen::Vector3d raw_maximum_d = raw_maximum.cast<double>();
    const double maximum_extent = (raw_maximum_d - raw_minimum_d).maxCoeff();
    std::uint32_t cells_per_side = 1U;
    const std::uint32_t required_cells = static_cast<std::uint32_t>(std::max(
        1.0, std::ceil((maximum_extent - std::numeric_limits<float>::epsilon()) / ray_cell_size)));
    while (cells_per_side < required_cells) {
        cells_per_side <<= 1U;
    }
    const double grid_side = ray_cell_size * cells_per_side;
    const Eigen::Vector3d padding =
        0.5 * (Eigen::Vector3d::Constant(grid_side) - (raw_maximum_d - raw_minimum_d));
    const Eigen::Vector3d grid_minimum = raw_minimum_d - padding;
    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> ray_cells;
    ray_cells.reserve(surfels.size());
    for (std::size_t index = 0; index < surfels.size(); ++index) {
        const Eigen::Vector3d point = xyz(surfels[index]).cast<double>();
        const Eigen::Array3i cell =
            ((point - grid_minimum) / ray_cell_size).array().floor().cast<int>();
        ray_cells[VoxelKey{cell.x(), cell.y(), cell.z()}].push_back(static_cast<int>(index));
    }

    const Eigen::Vector3d grid_maximum = grid_minimum + Eigen::Vector3d::Constant(grid_side);
    struct RayCandidateQuery {
        std::size_t candidate_count = 0U;
        std::size_t occupied_leaf_count = 0U;
    };
    constexpr std::size_t maximum_ray_entries = 512U;
    using RayCandidateArray = std::array<int, maximum_ray_entries>;
    int hierarchy_bits = 0;
    for (std::uint32_t side = cells_per_side; side > 1U; side >>= 1U) {
        ++hierarchy_bits;
    }
    struct RayOctreeNode {
        std::array<int, 8> children{};
        const std::vector<int>* indices = nullptr;

        RayOctreeNode() { children.fill(-1); }
    };
    std::vector<RayOctreeNode> ray_octree(1U);
    ray_octree.reserve(ray_cells.size() * 2U);
    for (const auto& [key, indices] : ray_cells) {
        int node = 0;
        for (int bit = hierarchy_bits - 1; bit >= 0; --bit) {
            const int child_id = (((key.x >> bit) & 1) << 2) |
                                 (((key.y >> bit) & 1) << 1) | ((key.z >> bit) & 1);
            int child = ray_octree[static_cast<std::size_t>(node)].children[child_id];
            if (child < 0) {
                child = static_cast<int>(ray_octree.size());
                ray_octree[static_cast<std::size_t>(node)].children[child_id] = child;
                ray_octree.emplace_back();
            }
            node = child;
        }
        ray_octree[static_cast<std::size_t>(node)].indices = &indices;
    }

    // nv_colorcloud uses the Revelles parametric octree traversal.  It is
    // important to recurse on the tx/ty/tz intervals directly: enumerating
    // voxels with DDA differs at shared edges and corners, which changes both
    // the leaf set and the candidate order at visibility boundaries.
    const auto collect_ray_candidates = [&ray_octree, &grid_minimum, &grid_maximum](
                                            const Vec3f& origin_f, const Vec3f& direction_f,
                                            RayCandidateArray& candidates,
                                            std::size_t occupied_leaf_limit) {
        RayCandidateQuery query;
        Vec3f origin = origin_f;
        Vec3f direction = direction_f;
        constexpr float zero_direction = 1.000000013351432e-10F;
        int reflection_mask = 0;
        constexpr std::array<int, 3> reflection_bits{4, 2, 1};
        for (int axis = 0; axis < 3; ++axis) {
            if (direction[axis] == 0.0F) {
                direction[axis] = zero_direction;
            }
            if (direction[axis] < 0.0F) {
                float reflected = static_cast<float>(grid_minimum[axis]);
                reflected += static_cast<float>(grid_maximum[axis]);
                reflected -= origin[axis];
                origin[axis] = reflected;
                direction[axis] = -direction[axis];
                reflection_mask |= reflection_bits[static_cast<std::size_t>(axis)];
            }
        }

        const double tx0 = (grid_minimum.x() - static_cast<double>(origin.x())) /
                           static_cast<double>(direction.x());
        const double tx1 = (grid_maximum.x() - static_cast<double>(origin.x())) /
                           static_cast<double>(direction.x());
        const double ty0 = (grid_minimum.y() - static_cast<double>(origin.y())) /
                           static_cast<double>(direction.y());
        const double ty1 = (grid_maximum.y() - static_cast<double>(origin.y())) /
                           static_cast<double>(direction.y());
        const double tz0 = (grid_minimum.z() - static_cast<double>(origin.z())) /
                           static_cast<double>(direction.z());
        const double tz1 = (grid_maximum.z() - static_cast<double>(origin.z())) /
                           static_cast<double>(direction.z());
        if (!(std::min({tx1, ty1, tz1}) > std::max({tx0, ty0, tz0}))) {
            return query;
        }

        const auto first_node = [](double x0, double y0, double z0, double xm, double ym,
                                   double zm) {
            int node = 0;
            if (x0 > y0) {
                if (x0 > z0) {
                    if (x0 > ym) {
                        node |= 2;
                    }
                    if (x0 > zm) {
                        node |= 1;
                    }
                    return node;
                }
            } else if (y0 > z0) {
                if (y0 > xm) {
                    node |= 4;
                }
                if (y0 > zm) {
                    node |= 1;
                }
                return node;
            }
            if (z0 > xm) {
                node |= 4;
            }
            if (z0 > ym) {
                node |= 2;
            }
            return node;
        };
        const auto next_node = [](double tx, int x, double ty, int y, double tz, int z) {
            if (tx < ty) {
                return tx < tz ? x : z;
            }
            return ty < tz ? y : z;
        };

        const auto traverse = [&](auto&& self, int node_index, double x0, double y0, double z0,
                                  double x1, double y1, double z1) -> std::size_t {
            if (node_index < 0 || x1 < 0.0 || y1 < 0.0 || z1 < 0.0) {
                return 0U;
            }
            const RayOctreeNode& node = ray_octree[static_cast<std::size_t>(node_index)];
            if (node.indices != nullptr) {
                for (const int index : *node.indices) {
                    if (query.candidate_count < candidates.size()) {
                        candidates[query.candidate_count++] = index;
                    }
                }
                return 1U;
            }

            const double xm = (x0 + x1) * 0.5;
            const double ym = (y0 + y1) * 0.5;
            const double zm = (z0 + z1) * 0.5;
            std::size_t leaves = 0U;
            int child = first_node(x0, y0, z0, xm, ym, zm);
            while (child < 8 && leaves < occupied_leaf_limit) {
                const int physical_child = child ^ reflection_mask;
                const int child_index = node.children[static_cast<std::size_t>(physical_child)];
                switch (child) {
                    case 0:
                        leaves += self(self, child_index, x0, y0, z0, xm, ym, zm);
                        child = next_node(xm, 4, ym, 2, zm, 1);
                        break;
                    case 1:
                        leaves += self(self, child_index, x0, y0, zm, xm, ym, z1);
                        child = next_node(xm, 5, ym, 3, z1, 8);
                        break;
                    case 2:
                        leaves += self(self, child_index, x0, ym, z0, xm, y1, zm);
                        child = next_node(xm, 6, y1, 8, zm, 3);
                        break;
                    case 3:
                        leaves += self(self, child_index, x0, ym, zm, xm, y1, z1);
                        child = next_node(xm, 7, y1, 8, z1, 8);
                        break;
                    case 4:
                        leaves += self(self, child_index, xm, y0, z0, x1, ym, zm);
                        child = next_node(x1, 8, ym, 6, zm, 5);
                        break;
                    case 5:
                        leaves += self(self, child_index, xm, y0, zm, x1, ym, z1);
                        child = next_node(x1, 8, ym, 7, z1, 8);
                        break;
                    case 6:
                        leaves += self(self, child_index, xm, ym, z0, x1, y1, zm);
                        child = next_node(x1, 8, y1, 8, zm, 7);
                        break;
                    case 7:
                        leaves += self(self, child_index, xm, ym, zm, x1, y1, z1);
                        child = 8;
                        break;
                    default:
                        child = 8;
                        break;
                }
            }
            return leaves;
        };
        query.occupied_leaf_count = traverse(traverse, 0, tx0, ty0, tz0, tx1, ty1, tz1);
        return query;
    };

    const auto intersect_candidates =
        [&surfels, disc_radius_squared](const Vec3f& origin, const Vec3f& ray,
                                        const RayCandidateArray& candidates,
                                        std::size_t count) {
        float best_range_squared = std::numeric_limits<float>::max();
        bool found = false;
        for (std::size_t candidate = 0; candidate < count; ++candidate) {
            const SurfacePoint& surfel = surfels[static_cast<std::size_t>(candidates[candidate])];
            const Vec3f center(surfel.x, surfel.y, surfel.z);
            const Vec3f normal(surfel.nx, surfel.ny, surfel.nz);

            float denominator = ray.y() * normal.y();
            denominator += ray.z() * normal.z();
            denominator += ray.x() * normal.x();
            if (denominator == 0.0F || !std::isfinite(denominator)) {
                continue;
            }

            float origin_dot_yz = normal.y() * origin.y();
            origin_dot_yz += normal.z() * origin.z();
            float origin_dot = normal.x() * origin.x();
            origin_dot += origin_dot_yz;
            float point_dot_yz = normal.y() * center.y();
            point_dot_yz += normal.z() * center.z();
            float point_dot = normal.x() * center.x();
            point_dot += point_dot_yz;
            const float distance = (point_dot - origin_dot) / denominator;
            if (!std::isfinite(distance)) {
                continue;
            }

            Vec3f intersection;
            intersection.x() = ray.x() * distance + origin.x();
            intersection.y() = ray.y() * distance + origin.y();
            intersection.z() = ray.z() * distance + origin.z();
            const float center_delta_x = center.x() - intersection.x();
            const float center_delta_y = center.y() - intersection.y();
            const float center_delta_z = center.z() - intersection.z();
            const float center_delta_x_squared = center_delta_x * center_delta_x;
            const float center_delta_y_squared = center_delta_y * center_delta_y;
            const float center_delta_z_squared = center_delta_z * center_delta_z;
            float center_distance_squared = center_delta_y_squared + center_delta_z_squared;
            center_distance_squared += center_delta_x_squared;
            if (center_distance_squared > disc_radius_squared) {
                continue;
            }

            const float range_delta_x = origin.x() - intersection.x();
            const float range_delta_y = origin.y() - intersection.y();
            const float range_delta_z = origin.z() - intersection.z();
            const float range_delta_x_squared = range_delta_x * range_delta_x;
            const float range_delta_y_squared = range_delta_y * range_delta_y;
            const float range_delta_z_squared = range_delta_z * range_delta_z;
            float range_squared = range_delta_y_squared + range_delta_z_squared;
            range_squared += range_delta_x_squared;
            if (range_squared < best_range_squared) {
                best_range_squared = range_squared;
                found = true;
            }
        }
        return found ? best_range_squared : std::numeric_limits<float>::max();
    };

    // Regression probe for matching the first octree-intersecting ray against
    // a runtime capture from nv_colorcloud. It is inert unless explicitly
    // enabled in the environment.
    if (std::getenv("NAVVIS_DEBUG_PCT_RAY") != nullptr && !world_rays.empty()) {
        if (pixels_per_view > 200U) {
            const Vec3f& local = local_rays[0][200U];
            const Eigen::Matrix3f rotation = view_poses[0].rotationMatrix();
            std::cerr << std::setprecision(17) << "PCT local ray probe=(" << local.x() << ' '
                      << local.y() << ' ' << local.z() << ") rotation=" << rotation(0, 0) << ' '
                      << rotation(0, 1) << ' ' << rotation(0, 2) << ' ' << rotation(1, 0) << ' '
                      << rotation(1, 1) << ' ' << rotation(1, 2) << ' ' << rotation(2, 0) << ' '
                      << rotation(2, 1) << ' ' << rotation(2, 2) << '\n';
        }
        for (std::size_t pixel = 0; pixel < pixels_per_view; ++pixel) {
            Vec3f ray = world_rays[0][pixel];
            const float x2 = ray.x() * ray.x();
            const float y2 = ray.y() * ray.y();
            float norm2 = ray.z() * ray.z();
            norm2 += y2;
            norm2 += x2;
            if (!(norm2 > 0.0F)) {
                continue;
            }
            const float norm = std::sqrt(norm2);
            ray.x() /= norm;
            ray.y() /= norm;
            ray.z() /= norm;
            RayCandidateArray candidates{};
            const RayCandidateQuery query =
                collect_ray_candidates(view_poses[0].translation, ray, candidates, 4U);
            if (query.candidate_count == 0U) {
                continue;
            }
            std::cerr << std::setprecision(9) << "PCT ray probe pixel=" << pixel
                      << " row=" << pixel / static_cast<std::size_t>(depth_width)
                      << " column=" << pixel % static_cast<std::size_t>(depth_width) << " origin=("
                      << view_poses[0].translation.x() << ' ' << view_poses[0].translation.y()
                      << ' ' << view_poses[0].translation.z() << ") ray=(" << ray.x() << ' '
                      << ray.y() << ' ' << ray.z() << ") candidates=" << query.candidate_count
                      << " occupied_leaves=" << query.occupied_leaf_count << ':';
            for (std::size_t index = 0; index < std::min<std::size_t>(query.candidate_count, 16U);
                 ++index) {
                std::cerr << ' ' << candidates[index];
            }
            std::cerr << '\n';
            break;
        }
    }

    // Targeted regression probe in "view:linear_pixel" form.  This keeps
    // mismatching depth pixels inspectable without changing normal runs.
    if (const char* probe = std::getenv("NAVVIS_DEBUG_PCT_PIXEL")) {
        const std::string specification(probe);
        const std::size_t separator = specification.find(':');
        if (separator != std::string::npos) {
            const std::size_t view = std::stoull(specification.substr(0, separator));
            const std::size_t pixel = std::stoull(specification.substr(separator + 1U));
            if (view < world_rays.size() && pixel < pixels_per_view) {
                Vec3f ray = world_rays[view][pixel];
                float squared_norm = ray.z() * ray.z();
                squared_norm += ray.y() * ray.y();
                squared_norm += ray.x() * ray.x();
                const float norm = std::sqrt(squared_norm);
                ray.x() /= norm;
                ray.y() /= norm;
                ray.z() /= norm;
                RayCandidateArray candidates{};
                RayCandidateQuery query =
                    collect_ray_candidates(view_poses[view].translation, ray, candidates, 4U);
                std::size_t evaluated_count =
                    std::min(query.occupied_leaf_count, query.candidate_count);
                float range_squared = intersect_candidates(view_poses[view].translation, ray,
                                                           candidates, evaluated_count);
                if (range_squared == std::numeric_limits<float>::max()) {
                    query =
                        collect_ray_candidates(view_poses[view].translation, ray, candidates, 128U);
                    evaluated_count = std::min(query.occupied_leaf_count, query.candidate_count);
                    range_squared = intersect_candidates(view_poses[view].translation, ray,
                                                         candidates, evaluated_count);
                }
                std::cerr << std::setprecision(17) << "PCT targeted probe view=" << view
                          << " pixel=" << pixel << " local=(" << local_rays[view % 4U][pixel].x()
                          << ' ' << local_rays[view % 4U][pixel].y() << ' '
                          << local_rays[view % 4U][pixel].z() << ") raw_world=("
                          << world_rays[view][pixel].x() << ' ' << world_rays[view][pixel].y()
                          << ' ' << world_rays[view][pixel].z() << ") ray=(" << ray.x() << ' '
                          << ray.y() << ' ' << ray.z() << ") origin=("
                          << view_poses[view].translation.x() << ' '
                          << view_poses[view].translation.y() << ' '
                          << view_poses[view].translation.z() << ") candidates="
                          << query.candidate_count
                          << " occupied_leaves=" << query.occupied_leaf_count
                          << " evaluated=" << evaluated_count << ':';
                for (std::size_t index = 0; index < query.candidate_count; ++index) {
                    std::cerr << ' ' << candidates[index];
                }
                if (evaluated_count > 0U) {
                    const SurfacePoint& surfel = surfels[candidates[0]];
                    const Vec3f center(surfel.x, surfel.y, surfel.z);
                    const Vec3f normal(surfel.nx, surfel.ny, surfel.nz);
                    float denominator = ray.y() * normal.y();
                    denominator += ray.z() * normal.z();
                    denominator += ray.x() * normal.x();
                    float origin_dot_yz = normal.y() * view_poses[view].translation.y();
                    origin_dot_yz += normal.z() * view_poses[view].translation.z();
                    float origin_dot = normal.x() * view_poses[view].translation.x();
                    origin_dot += origin_dot_yz;
                    float point_dot_yz = normal.y() * center.y();
                    point_dot_yz += normal.z() * center.z();
                    float point_dot = normal.x() * center.x();
                    point_dot += point_dot_yz;
                    const float distance = (point_dot - origin_dot) / denominator;
                    const Vec3f intersection(
                        ray.x() * distance + view_poses[view].translation.x(),
                        ray.y() * distance + view_poses[view].translation.y(),
                        ray.z() * distance + view_poses[view].translation.z());
                    const Vec3f delta = view_poses[view].translation - intersection;
                    float debug_range_squared = delta.y() * delta.y();
                    debug_range_squared += delta.z() * delta.z();
                    debug_range_squared += delta.x() * delta.x();
                    std::cerr << " surfel=(" << surfel.x << ' ' << surfel.y << ' ' << surfel.z
                              << "; " << surfel.nx << ' ' << surfel.ny << ' ' << surfel.nz
                              << ") denominator=" << denominator << " origin_dot=" << origin_dot
                              << " point_dot=" << point_dot << " distance=" << distance
                              << " intersection=(" << intersection.x() << ' ' << intersection.y()
                              << ' ' << intersection.z() << ") debug_range_squared="
                              << debug_range_squared;
                }
                for (std::size_t candidate = 0; candidate < evaluated_count; ++candidate) {
                    const SurfacePoint& surfel = surfels[candidates[candidate]];
                    const Vec3f center(surfel.x, surfel.y, surfel.z);
                    const Vec3f normal(surfel.nx, surfel.ny, surfel.nz);
                    float denominator = ray.y() * normal.y();
                    denominator += ray.z() * normal.z();
                    denominator += ray.x() * normal.x();
                    float origin_dot = normal.y() * view_poses[view].translation.y();
                    origin_dot += normal.z() * view_poses[view].translation.z();
                    origin_dot += normal.x() * view_poses[view].translation.x();
                    float point_dot = normal.y() * center.y();
                    point_dot += normal.z() * center.z();
                    point_dot += normal.x() * center.x();
                    const float distance = (point_dot - origin_dot) / denominator;
                    const Vec3f intersection(
                        ray.x() * distance + view_poses[view].translation.x(),
                        ray.y() * distance + view_poses[view].translation.y(),
                        ray.z() * distance + view_poses[view].translation.z());
                    const Vec3f center_delta = center - intersection;
                    float center_distance_squared = center_delta.y() * center_delta.y();
                    center_distance_squared += center_delta.z() * center_delta.z();
                    center_distance_squared += center_delta.x() * center_delta.x();
                    const Vec3f range_delta = view_poses[view].translation - intersection;
                    float candidate_range_squared = range_delta.y() * range_delta.y();
                    candidate_range_squared += range_delta.z() * range_delta.z();
                    candidate_range_squared += range_delta.x() * range_delta.x();
                    std::cerr << " candidate[" << candidate << "]=" << candidates[candidate]
                              << " center_distance_squared=" << center_distance_squared
                              << " accepted=" << (center_distance_squared <= disc_radius_squared)
                              << " range_squared=" << candidate_range_squared;
                }
                if (range_squared != std::numeric_limits<float>::max()) {
                    std::cerr << " range_squared=" << range_squared
                              << " raw_range=" << std::sqrt(range_squared);
                } else {
                    std::cerr << " no_hit";
                }
                std::cerr << '\n';
            }
        }
    }

    std::atomic<std::size_t> completed_views{0U};
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t signed_view = 0; signed_view < static_cast<std::int64_t>(view_poses.size());
         ++signed_view) {
        const std::size_t view = static_cast<std::size_t>(signed_view);
        const Pose& pose = view_poses[view];
        const auto& rays = world_rays[view];
        const std::size_t view_offset = view * pixels_per_view;
        for (std::size_t pixel = 0; pixel < pixels_per_view; ++pixel) {
            Vec3f ray = rays[pixel];
            const float ray_x_squared = ray.x() * ray.x();
            const float ray_y_squared = ray.y() * ray.y();
            float ray_squared_norm = ray.z() * ray.z();
            ray_squared_norm += ray_y_squared;
            ray_squared_norm += ray_x_squared;
            if (!(ray_squared_norm > 0.0F) || !std::isfinite(ray_squared_norm)) {
                continue;
            }
            const float ray_norm = std::sqrt(ray_squared_norm);
            ray.x() /= ray_norm;
            ray.y() /= ray_norm;
            ray.z() /= ray_norm;

            RayCandidateArray candidates{};
            RayCandidateQuery query = collect_ray_candidates(pose.translation, ray, candidates, 4U);
            std::size_t evaluated_count =
                std::min(query.occupied_leaf_count, query.candidate_count);
            float range_squared =
                intersect_candidates(pose.translation, ray, candidates, evaluated_count);
            if (range_squared == std::numeric_limits<float>::max()) {
                query = collect_ray_candidates(pose.translation, ray, candidates, 128U);
                evaluated_count = std::min(query.occupied_leaf_count, query.candidate_count);
                range_squared =
                    intersect_candidates(pose.translation, ray, candidates, evaluated_count);
            }
            if (range_squared != std::numeric_limits<float>::max()) {
                // The ray renderer first produces the raw sqrtss range.  The
                // downstream depth-map representation used by coloring floors
                // it to millimetres; that final representation is what the
                // visibility code and captured reference maps consume.
                const float range = std::sqrt(range_squared);
                const float quantized = std::floor(range * 1000.0F) * 0.001F;
                depth[view_offset + pixel].store(floatBits(quantized), std::memory_order_relaxed);
            }
        }
        const std::size_t done = completed_views.fetch_add(1U, std::memory_order_relaxed) + 1U;
#pragma omp critical(navvis_pct_depth_progress)
        std::cerr << "PCT depth view " << done << '/' << view_poses.size() << '\r';
    }
    std::cerr << '\n';
}

cv::Vec3f samplePixelCenteredBilinearBgr(const cv::Mat& image, float image_x, float image_y) {
    // PixelColorExtractorBilinear treats integer+0.5 as pixel centers.  Its
    // recovered clamp constants are 0.500001013 and size-0.500001013.
    constexpr float edge = 0.500001013F;
    const float x = std::clamp(image_x, edge, static_cast<float>(image.cols) - edge);
    const float y = std::clamp(image_y, edge, static_cast<float>(image.rows) - edge);

    const int center_x = static_cast<int>(x);
    const int center_y = static_cast<int>(y);
    const float signed_x = x - (static_cast<float>(center_x) + 0.5F);
    const float signed_y = y - (static_cast<float>(center_y) + 0.5F);
    const float fraction_x = std::abs(signed_x);
    const float fraction_y = std::abs(signed_y);
    const int neighbor_x = center_x + (signed_x > 0.0F ? 1 : -1);
    const int neighbor_y = center_y + (signed_y > 0.0F ? 1 : -1);
    const float center_weight_x = 1.0F - fraction_x;
    const float center_weight_y = 1.0F - fraction_y;

    cv::Vec3f result;
    const cv::Vec3b& center = image.at<cv::Vec3b>(center_y, center_x);
    const cv::Vec3b& horizontal = image.at<cv::Vec3b>(center_y, neighbor_x);
    const cv::Vec3b& vertical = image.at<cv::Vec3b>(neighbor_y, center_x);
    const cv::Vec3b& diagonal = image.at<cv::Vec3b>(neighbor_y, neighbor_x);
    for (int channel = 0; channel < 3; ++channel) {
        const float center_row = static_cast<float>(center[channel]) * center_weight_x +
                                 static_cast<float>(horizontal[channel]) * fraction_x;
        const float neighbor_row = static_cast<float>(vertical[channel]) * center_weight_x +
                                   static_cast<float>(diagonal[channel]) * fraction_x;
        result[channel] = center_row * center_weight_y + neighbor_row * fraction_y;
    }
    return result;
}

fs::path panoramaPath(const fs::path& directory, int index) {
    std::ostringstream plain;
    plain << std::setw(5) << std::setfill('0') << index << ".jpg";
    fs::path result = directory / plain.str();
    if (fs::exists(result)) {
        return result;
    }
    std::ostringstream navvis;
    navvis << std::setw(5) << std::setfill('0') << index << "-pano.jpg";
    result = directory / navvis.str();
    if (!fs::exists(result)) {
        throw std::runtime_error("missing panorama for capture " + std::to_string(index));
    }
    return result;
}

class ImageCache {
  public:
    ImageCache(fs::path directory, std::size_t capacity)
        : directory_(std::move(directory)), capacity_(capacity) {}

    const cv::Mat& get(int index) {
        auto found = images_.find(index);
        if (found != images_.end()) {
            order_.erase(found->second.order);
            order_.push_front(index);
            found->second.order = order_.begin();
            return found->second.image;
        }
        cv::Mat image = cv::imread(panoramaPath(directory_, index).string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("cannot decode panorama " + std::to_string(index));
        }
        order_.push_front(index);
        auto inserted = images_.emplace(index, Entry{std::move(image), order_.begin()}).first;
        while (images_.size() > capacity_) {
            const int evicted = order_.back();
            order_.pop_back();
            images_.erase(evicted);
        }
        return inserted->second.image;
    }

  private:
    struct Entry {
        cv::Mat image;
        std::list<int>::iterator order;
    };
    fs::path directory_;
    std::size_t capacity_;
    std::list<int> order_;
    std::unordered_map<int, Entry> images_;
};

class CameraImageCache {
  public:
    CameraImageCache(fs::path directory, std::size_t capacity)
        : directory_(std::move(directory)), capacity_(capacity) {}

    struct Pyramid {
        std::array<cv::Mat, kDirectPatchPyramidLevels> levels;
    };

    void preload(int capture_count) {
        if (capture_count <= 0 || capacity_ < static_cast<std::size_t>(capture_count)) {
            return;
        }
        using PreloadClock = std::chrono::steady_clock;
        const auto started = PreloadClock::now();
        std::vector<std::array<Pyramid, 4>> decoded(static_cast<std::size_t>(capture_count));
        std::vector<std::string> errors(static_cast<std::size_t>(capture_count));
        const int worker_count = std::min(4, capture_count);
#pragma omp parallel for schedule(dynamic, 1) num_threads(worker_count)
        for (int capture = 0; capture < capture_count; ++capture) {
            try {
                decoded[static_cast<std::size_t>(capture)] = decodeCapture(capture);
            } catch (const std::exception& error) {
                errors[static_cast<std::size_t>(capture)] = error.what();
            }
        }
        for (int capture = 0; capture < capture_count; ++capture) {
            const std::string& error = errors[static_cast<std::size_t>(capture)];
            if (!error.empty()) {
                throw std::runtime_error(error);
            }
            order_.push_front(capture);
            images_.emplace(capture, Entry{std::move(decoded[static_cast<std::size_t>(capture)]),
                                           order_.begin()});
        }
        std::cerr << "Preloaded " << capture_count << " camera captures with " << worker_count
                  << " workers in "
                  << std::chrono::duration<double>(PreloadClock::now() - started).count() << " s\n";
    }

    const std::array<Pyramid, 4>& get(int capture) {
        auto found = images_.find(capture);
        if (found != images_.end()) {
            order_.erase(found->second.order);
            order_.push_front(capture);
            found->second.order = order_.begin();
            return found->second.images;
        }
        std::array<Pyramid, 4> decoded = decodeCapture(capture);
        order_.push_front(capture);
        auto inserted = images_.emplace(capture, Entry{std::move(decoded), order_.begin()}).first;
        while (images_.size() > capacity_) {
            const int evicted = order_.back();
            order_.pop_back();
            images_.erase(evicted);
        }
        return inserted->second.images;
    }

  private:
    std::array<Pyramid, 4> decodeCapture(int capture) const {
        std::array<Pyramid, 4> decoded;
        for (std::size_t camera = 0; camera < decoded.size(); ++camera) {
            std::ostringstream name;
            name << std::setw(5) << std::setfill('0') << capture << "-cam" << camera << ".jpg";
            cv::Mat raw = cv::imread((directory_ / name.str()).string(),
                                     cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
            if (raw.empty()) {
                throw std::runtime_error("cannot decode " + (directory_ / name.str()).string());
            }

            // DirectPatchColorExtractor::setImage performs BGR->HSV, applies
            // gamma 0.8 to V in float [0,1], then converts back to BGR.
            cv::Mat hsv;
            cv::cvtColor(raw, hsv, cv::COLOR_BGR2HSV);
            std::vector<cv::Mat> channels;
            cv::split(hsv, channels);
            cv::Mat value_float;
            channels[2].convertTo(value_float, CV_32F, 1.0 / 255.0);
            cv::pow(value_float, static_cast<double>(0.8F), value_float);
            value_float.convertTo(channels[2], CV_8U, 255.0);
            cv::merge(channels, hsv);
            cv::cvtColor(hsv, decoded[camera].levels[0], cv::COLOR_HSV2BGR);
            for (std::size_t level = 1; level < kDirectPatchPyramidLevels; ++level) {
                cv::pyrDown(decoded[camera].levels[level - 1U], decoded[camera].levels[level]);
            }
        }
        return decoded;
    }

    struct Entry {
        std::array<Pyramid, 4> images;
        std::list<int>::iterator order;
    };
    fs::path directory_;
    std::size_t capacity_;
    std::list<int> order_;
    std::unordered_map<int, Entry> images_;
};

float binaryDistanceWeight(float distance) {
    // HybridDistanceWeightFunc recovered from nv_colorcloud.  The apparent
    // normalizing factor makes the maximum of the Gaussian-plus-reciprocal
    // distance response one.
    constexpr float gaussian_normalizer = 0.398942292F;
    constexpr float distance_offset = 6.0F;
    constexpr float response_normalizer = 1.84389675F;
    const float centered_distance = distance - 1.0F;
    return response_normalizer *
           (gaussian_normalizer * std::exp(-0.5F * centered_distance * centered_distance) +
            1.0F / (distance + distance_offset));
}

const cv::Mat& fibonacciNormalDirections() {
    // PclPointCloudAdapter stores each input normal as the nearest member of
    // this 65,535-direction Fibonacci sphere, using uint16_t value 65535 as
    // the invalid sentinel.  Its decoder was recovered instruction-for-
    // instruction from nv_colorcloud's 14-byte adapter record accessor.
    static const cv::Mat directions = [] {
        cv::Mat result(65535, 3, CV_32F);
        constexpr float golden_fraction = 0.6180340051651001F;
        constexpr float two_pi = 6.2831854820251465F;
        constexpr float code_scale = 65535.0F;
        for (int code = 0; code < result.rows; ++code) {
            const float value = static_cast<float>(code);
            const float turns = ::fmaf(value, golden_fraction, -::truncf(value * golden_fraction));
            const float angle = turns * two_pi;
            float sine = 0.0F;
            float cosine = 0.0F;
#if defined(_MSC_VER)
            sine = ::sinf(angle);
            cosine = ::cosf(angle);
#else
            ::sincosf(angle, &sine, &cosine);
#endif
            const float z = 1.0F - ((value + value) + 1.0F) / code_scale;
            const float radius = ::sqrtf(std::max(0.0F, 1.0F - z * z));
            result.at<float>(code, 0) = radius * cosine;
            result.at<float>(code, 1) = radius * sine;
            result.at<float>(code, 2) = z;
        }
        return result;
    }();
    return directions;
}

class FibonacciNormalKdTree {
  public:
    FibonacciNormalKdTree() {
        std::vector<int> codes(65535);
        std::iota(codes.begin(), codes.end(), 0);
        nodes_.reserve(codes.size());
        root_ = build(codes, 0U, codes.size(), 0);
    }

    [[nodiscard]] std::uint16_t nearest(const Vec3f& query) const {
        if (!query.allFinite()) {
            return std::numeric_limits<std::uint16_t>::max();
        }
        int best_code = -1;
        double best_squared_distance = std::numeric_limits<double>::infinity();
        search(root_, query, best_code, best_squared_distance);
        return best_code >= 0 ? static_cast<std::uint16_t>(best_code)
                              : std::numeric_limits<std::uint16_t>::max();
    }

  private:
    struct Node {
        int code = -1;
        int left = -1;
        int right = -1;
        int axis = 0;
    };

    int build(std::vector<int>& codes, std::size_t begin, std::size_t end, int depth) {
        if (begin >= end) {
            return -1;
        }
        const int axis = depth % 3;
        const std::size_t middle = begin + (end - begin) / 2U;
        const cv::Mat& directions = fibonacciNormalDirections();
        std::nth_element(codes.begin() + static_cast<std::ptrdiff_t>(begin),
                         codes.begin() + static_cast<std::ptrdiff_t>(middle),
                         codes.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](int first, int second) {
                             const float first_value = directions.at<float>(first, axis);
                             const float second_value = directions.at<float>(second, axis);
                             return first_value < second_value ||
                                    (first_value == second_value && first < second);
                         });
        const int node_index = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{codes[middle], -1, -1, axis});
        const int left = build(codes, begin, middle, depth + 1);
        const int right = build(codes, middle + 1U, end, depth + 1);
        nodes_[static_cast<std::size_t>(node_index)].left = left;
        nodes_[static_cast<std::size_t>(node_index)].right = right;
        return node_index;
    }

    void search(int node_index, const Vec3f& query, int& best_code,
                double& best_squared_distance) const {
        if (node_index < 0) {
            return;
        }
        const Node& node = nodes_[static_cast<std::size_t>(node_index)];
        const float* candidate = fibonacciNormalDirections().ptr<float>(node.code);
        const double dx = static_cast<double>(query.x()) - candidate[0];
        const double dy = static_cast<double>(query.y()) - candidate[1];
        const double dz = static_cast<double>(query.z()) - candidate[2];
        const double squared_distance = dx * dx + dy * dy + dz * dz;
        if (squared_distance < best_squared_distance ||
            (squared_distance == best_squared_distance && node.code < best_code)) {
            best_squared_distance = squared_distance;
            best_code = node.code;
        }

        const double split_difference =
            static_cast<double>(query[node.axis]) - candidate[node.axis];
        const int near_node = split_difference < 0.0 ? node.left : node.right;
        const int far_node = split_difference < 0.0 ? node.right : node.left;
        search(near_node, query, best_code, best_squared_distance);
        if (split_difference * split_difference <= best_squared_distance) {
            search(far_node, query, best_code, best_squared_distance);
        }
    }

    std::vector<Node> nodes_;
    int root_ = -1;
};

const FibonacciNormalKdTree& fibonacciNormalIndex() {
    static const FibonacciNormalKdTree index;
    return index;
}

struct QuantizedNormals {
    std::vector<Vec3f> values;
    std::vector<std::uint16_t> codes;
};

QuantizedNormals quantizeFibonacciNormals(const std::vector<SurfacePoint>& points,
                                          std::size_t count) {
    QuantizedNormals result;
    result.values.resize(count, Vec3f::Constant(std::numeric_limits<float>::quiet_NaN()));
    result.codes.resize(count, std::numeric_limits<std::uint16_t>::max());
    if (count == 0U) {
        return result;
    }

    const cv::Mat& directions = fibonacciNormalDirections();
#pragma omp parallel for schedule(static)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(count); ++index) {
        const SurfacePoint& point = points[static_cast<std::size_t>(index)];
        const std::uint16_t code =
            fibonacciNormalIndex().nearest(Vec3f(point.nx, point.ny, point.nz));
        if (code == std::numeric_limits<std::uint16_t>::max()) {
            continue;
        }
        result.codes[static_cast<std::size_t>(index)] = code;
        const float* normal = directions.ptr<float>(static_cast<int>(code));
        result.values[static_cast<std::size_t>(index)] = Vec3f(normal[0], normal[1], normal[2]);
    }
    return result;
}

Vec3f quantizeFibonacciNormal(const SurfacePoint& point) {
    const std::vector<SurfacePoint> single{point};
    return quantizeFibonacciNormals(single, 1U).values[0];
}

float cameraViewScore(const SurfacePoint& point, const Pose& pose, const OCamCamera& model,
                      const OCamProjection& projection, const CameraMasks& masks, int camera,
                      const Vec3f& scoring_normal, float incidence_power, float radius_power,
                      float distance_power) {
    // Image radius and the old power parameters are deliberately absent: the
    // reference point scorer does not use them.
    (void)model;
    (void)incidence_power;
    (void)radius_power;
    (void)distance_power;
    const Vec3f ray_to_camera = navvis_recon::normalizedOr(pose.translation - xyz(point));
    const float incidence = scoring_normal.dot(ray_to_camera);
    return incidence * binaryDistanceWeight(projection.range) *
           cameraPointWeight(masks, camera, projection.image_x, projection.image_y);
}

std::optional<DirectSample> directCameraSampleFromProjection(
    const SurfacePoint& point, const Pose& pose, const OCamCamera& model,
    const CameraImageCache::Pyramid& pyramid, const CameraMasks& masks, int camera,
    const Vec3f& scoring_normal, float incidence_power, float radius_power, float distance_power,
    float view_max_distance, int view, float local_radius, const OCamProjection& projection);

std::optional<DirectSample>
directCameraSample(const SurfacePoint& point, const Pose& pose, const OCamCamera& model,
                   const CameraImageCache::Pyramid& pyramid, const CameraMasks& masks, int camera,
                   const Vec3f& scoring_normal, float incidence_power, float radius_power,
                   float distance_power, float view_max_distance, int view, float local_radius) {
    const auto projection = projectOCam(xyz(point), pose, model, masks, camera);
    if (!projection) {
        return std::nullopt;
    }
    return directCameraSampleFromProjection(
        point, pose, model, pyramid, masks, camera, scoring_normal, incidence_power, radius_power,
        distance_power, view_max_distance, view, local_radius, *projection);
}

std::optional<DirectSample> directCameraSampleFromProjection(
    const SurfacePoint& point, const Pose& pose, const OCamCamera& model,
    const CameraImageCache::Pyramid& pyramid, const CameraMasks& masks, int camera,
    const Vec3f& scoring_normal, float incidence_power, float radius_power, float distance_power,
    float view_max_distance, int view, float local_radius, const OCamProjection& projection) {
    const cv::Mat& full_resolution = pyramid.levels[0];
    if (projection.range > view_max_distance || projection.image_x < 0.0 ||
        projection.image_y < 0.0 || projection.image_x >= full_resolution.cols ||
        projection.image_y >= full_resolution.rows) {
        return std::nullopt;
    }

    const float footprint = kDirectPatchFocalScale * local_radius / projection.range;
    float squared_footprint = footprint * footprint;
    std::size_t reductions = 0U;
    while (squared_footprint > 4.0F) {
        squared_footprint *= 0.25F;
        ++reductions;
    }
    const std::size_t level =
        std::min(reductions == 0U ? 0U : reductions - 1U, kDirectPatchPyramidLevels - 1U);
    const float coordinate_scale = std::ldexp(1.0F, -static_cast<int>(level));
    const cv::Vec3f raw_bgr = samplePixelCenteredBilinearBgr(
        pyramid.levels[level], static_cast<float>(projection.image_x) * coordinate_scale,
        static_cast<float>(projection.image_y) * coordinate_scale);
    DirectSample sample;
    for (int channel = 0; channel < 3; ++channel) {
        sample.raw_rgb[channel] = cv::saturate_cast<std::uint8_t>(raw_bgr[2 - channel]);
        sample.rgb[channel] = sample.raw_rgb[channel];
    }
    sample.score = cameraViewScore(point, pose, model, projection, masks, camera, scoring_normal,
                                   incidence_power, radius_power, distance_power);
    sample.view = view;
    sample.image_x = static_cast<float>(projection.image_x);
    sample.image_y = static_cast<float>(projection.image_y);
    return sample;
}

std::uint16_t quantizeUnitScore(float score);

void insertSelectedSample(SelectedSamples& selected, const DirectSample& sample) {
    // The binary uses two distinct comparisons.  Once all five slots are
    // occupied, 0x1d2515..0x1d252e first compares the incoming raw float with
    // the decoded uint16 quality of the current last slot.  A candidate which
    // passes that gate replaces the last slot.  The unrolled insertion at
    // 0x1d256c..0x1d2772 then bubbles the packed record towards the front only
    // while its uint16 quality is strictly greater than the preceding slot.
    // Equal packed qualities consequently retain candidate arrival order,
    // except that a passing candidate always replaces the occupied last slot.
    const std::uint16_t packed_score = quantizeUnitScore(sample.score);
    if (packed_score == 0U) {
        return;
    }

    std::size_t position = selected.count;
    if (selected.count < kSelectedViews) {
        ++selected.count;
    } else {
        position = kSelectedViews - 1U;
        const std::uint16_t worst_score =
            quantizeUnitScore(selected.values[position].score);
        if (sample.score <= static_cast<float>(worst_score) / 65535.0F) {
            return;
        }
    }

    selected.values[position] = sample;
    while (position > 0U &&
           quantizeUnitScore(selected.values[position - 1U].score) < packed_score) {
        std::swap(selected.values[position - 1U], selected.values[position]);
        --position;
    }
}

float channelMedian(const std::array<cv::Vec3f, kSelectedViews>& colors, std::size_t count,
                    int channel) {
    std::array<std::uint8_t, kSelectedViews> values{};
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = cv::saturate_cast<std::uint8_t>(colors[index][channel]);
    }
    const std::size_t middle = count / 2U;
    std::nth_element(values.begin(), values.begin() + middle, values.begin() + count);
    return static_cast<float>(values[middle]);
}

std::uint16_t quantizeUnitScore(float score) {
    // float16_01_t::encode in nv_colorcloud clamps and truncates.  In
    // particular, normalized five-view records sum to 65531..65535; rounding
    // would also produce sums above 65535, which the reference never does.
    return static_cast<std::uint16_t>(std::clamp(score, 0.0F, 1.0F) * 65535.0F);
}

std::array<std::uint16_t, kSelectedViews>
normalizeSelectedQualities(const SelectedSamples& selected) {
    std::array<float, kSelectedViews> decoded{};
    std::array<std::uint16_t, kSelectedViews> encoded{};
    bool supplied_normalized = selected.count > 0U;
    for (std::size_t index = 0; index < selected.count; ++index) {
        supplied_normalized &= selected.values[index].has_normalized_quality;
    }
    if (supplied_normalized) {
        for (std::size_t index = 0; index < selected.count; ++index) {
            encoded[index] = selected.values[index].normalized_quality;
        }
        return encoded;
    }
    float sum = 0.0F;
    for (std::size_t index = 0; index < selected.count; ++index) {
        decoded[index] =
            static_cast<float>(quantizeUnitScore(selected.values[index].score)) / 65535.0F;
        sum += decoded[index];
    }
    if (sum > 0.0F) {
        for (std::size_t index = 0; index < selected.count; ++index) {
            encoded[index] = quantizeUnitScore(decoded[index] / sum);
        }
    }
    return encoded;
}

cv::Vec3f applyExposureGammaRgb(const cv::Vec3f& rgb, const GammaModel& model) {
    // GammaModel::apply uses the installed color helpers verbatim.  Their
    // wrappers normalize RGB with three float divisions, scale HSV back to
    // 0..255, then perform the inverse H*360/255 and S,V /255 operations.
    // Do not fold these constants: the intermediate rounding is observable at
    // the final nearest-even byte conversion.
    constexpr float scale_255 = 255.0F;
    constexpr float degrees_per_turn = 360.0F;
    constexpr float degrees_per_sector = 60.0F;
    const float red = rgb[0] / scale_255;
    const float green = rgb[1] / scale_255;
    const float blue = rgb[2] / scale_255;
    const float maximum = std::max({red, green, blue});
    const float minimum = std::min({red, green, blue});
    const float delta = maximum - minimum;

    float hue = 0.0F;
    float saturation = 0.0F;
    if (maximum > 0.0F) {
        saturation = delta / maximum;
    }
    if (delta > 0.0F) {
        float degrees = 0.0F;
        if (maximum == red) {
            degrees = (green - blue) / delta;
            if (blue > green) {
                degrees += 6.0F;
            }
        } else if (maximum == green) {
            degrees = (blue - red) / delta + 2.0F;
        } else {
            degrees = (red - green) / delta + 4.0F;
        }
        degrees *= degrees_per_sector;
        hue = degrees * (scale_255 / degrees_per_turn);
    }
    saturation *= scale_255;
    const float original_value = maximum * scale_255;

    const float normalized_value = original_value / scale_255;
    const float corrected_normalized = static_cast<float>(
        model.gain * std::pow(static_cast<double>(normalized_value), model.exponent));
    const float value = corrected_normalized * scale_255;
    if (saturation <= 0.0F) {
        return cv::Vec3f(value, value, value);
    }

    float restored_hue_degrees = hue * degrees_per_turn;
    restored_hue_degrees /= scale_255;
    const float restored_saturation = saturation / scale_255;
    const float restored_value = value / scale_255;
    const float chroma = restored_saturation * restored_value;
    const float minimum_value = restored_value - chroma;
    const float hue_turn = restored_hue_degrees / degrees_per_turn;
    const float wrapped_hue_degrees =
        restored_hue_degrees - std::floor(hue_turn) * degrees_per_turn;
    const float hue_sector = wrapped_hue_degrees / degrees_per_sector;
    const float twice_floor_half = std::floor(0.5F * hue_sector) * 2.0F;
    const float triangle = 1.0F - std::fabs(hue_sector - twice_floor_half - 1.0F);
    const float x = triangle * chroma;
    const int sector = static_cast<int>(hue_sector);
    cv::Vec3f normalized_rgb;
    switch (sector % 6) {
        case 0:
            normalized_rgb = cv::Vec3f(chroma + minimum_value, x + minimum_value, minimum_value);
            break;
        case 1:
            normalized_rgb = cv::Vec3f(x + minimum_value, chroma + minimum_value, minimum_value);
            break;
        case 2:
            normalized_rgb = cv::Vec3f(minimum_value, chroma + minimum_value, x + minimum_value);
            break;
        case 3:
            normalized_rgb = cv::Vec3f(minimum_value, x + minimum_value, chroma + minimum_value);
            break;
        case 4:
            normalized_rgb = cv::Vec3f(x + minimum_value, minimum_value, chroma + minimum_value);
            break;
        default:
            normalized_rgb = cv::Vec3f(chroma + minimum_value, minimum_value, x + minimum_value);
            break;
    }
    for (int channel = 0; channel < 3; ++channel) {
        normalized_rgb[channel] *= scale_255;
    }
    return normalized_rgb;
}

std::optional<cv::Vec3f>
blendSelectedSamples(const SelectedSamples& selected, bool average, bool median, bool robust,
                     const std::vector<GammaModel>* exposure_models) {
    if (selected.count == 0U) {
        return std::nullopt;
    }

    // OptimalViewSelection keeps valid black observations in its packed
    // Top-5 output, but Gamma/AdaptiveBestScore does not consume them.  Keep
    // the original records (and their normalized qualities) intact for OVS
    // dumps and direct-mask semantics; compact only the blend inputs here.
    std::array<std::size_t, kSelectedViews> sample_indices{};
    std::size_t sample_count = 0U;
    for (std::size_t index = 0; index < selected.count; ++index) {
        const cv::Vec3b& rgb = selected.values[index].rgb;
        if (rgb[0] != 0U || rgb[1] != 0U || rgb[2] != 0U) {
            sample_indices[sample_count++] = index;
        }
    }
    // The binary also rejects a formerly multi-view selection when black
    // filtering leaves only one color. A point that was genuinely selected
    // from one non-black view remains valid; the distinction is observable
    // in the exact KNN colored/uncolored partition.
    if (sample_count == 0U || (selected.count > 1U && sample_count < 2U)) {
        return std::nullopt;
    }

    std::array<cv::Vec3f, kSelectedViews> colors{};
    for (std::size_t index = 0; index < sample_count; ++index) {
        const DirectSample& sample = selected.values[sample_indices[index]];
        colors[index] = cv::Vec3f(sample.rgb);
        if (exposure_models == nullptr) {
            continue;
        }
        const GammaModel& model = exposure_models->at(static_cast<std::size_t>(sample.view));
        colors[index] = applyExposureGammaRgb(colors[index], model);
    }
    if (!average && !median && !robust) {
        return colors[0];
    }
    if (median) {
        return cv::Vec3f(channelMedian(colors, sample_count, 0),
                         channelMedian(colors, sample_count, 1),
                         channelMedian(colors, sample_count, 2));
    }
    std::array<float, kSelectedViews> abs_weights{};
    if (robust) {
        // AdaptiveBestScore (ABS) recovered from the reference implementation.
        // OVS first stores each raw score as float16_01_t, normalizes the
        // decoded scores, stores the normalized values as float16_01_t again,
        // and only then applies ABS to costs (1 - normalized_score).
        std::array<float, kSelectedViews> original_normalized_scores{};
        bool supplied_normalized = selected.count > 0U;
        for (std::size_t index = 0; index < selected.count; ++index) {
            supplied_normalized &= selected.values[index].has_normalized_quality;
        }
        if (supplied_normalized) {
            for (std::size_t index = 0; index < selected.count; ++index) {
                original_normalized_scores[index] =
                    static_cast<float>(selected.values[index].normalized_quality) / 65535.0F;
            }
        } else {
            float score_sum = 0.0F;
            for (std::size_t index = 0; index < selected.count; ++index) {
                original_normalized_scores[index] =
                    static_cast<float>(quantizeUnitScore(selected.values[index].score)) / 65535.0F;
                score_sum += original_normalized_scores[index];
            }
            for (std::size_t index = 0; index < selected.count; ++index) {
                original_normalized_scores[index] =
                    static_cast<float>(quantizeUnitScore(
                        original_normalized_scores[index] / std::max(score_sum, 1.0e-12F))) /
                    65535.0F;
            }
        }
        std::array<float, kSelectedViews> normalized_scores{};
        for (std::size_t index = 0; index < sample_count; ++index) {
            normalized_scores[index] = original_normalized_scores[sample_indices[index]];
        }
        float minimum_cost = 1.0F;
        for (std::size_t index = 0; index < sample_count; ++index) {
            const float cost = 1.0F - normalized_scores[index];
            abs_weights[index] = cost;
            minimum_cost = std::min(minimum_cost, cost);
        }
        const float bandwidth = std::max(1.0e-6F, 0.1F * minimum_cost);
        for (std::size_t index = 0; index < sample_count; ++index) {
            abs_weights[index] = std::exp(-abs_weights[index] / bandwidth);
        }
        // AdaptiveBestScore normalizes once internally and its caller
        // normalizes the returned vector again.  Both reductions accumulate
        // absolute float weights in double, divide in double, then cast each
        // result back to float.  The two apparently redundant passes affect
        // nearest-even byte boundaries.
        double normalizer = 0.0;
        for (std::size_t index = 0; index < sample_count; ++index) {
            normalizer += std::fabs(static_cast<double>(abs_weights[index]));
        }
        if (normalizer > 0.0F) {
            for (std::size_t index = 0; index < sample_count; ++index) {
                abs_weights[index] = static_cast<float>(
                    static_cast<double>(abs_weights[index]) / normalizer);
            }
        }
        double caller_normalizer = 0.0;
        for (std::size_t index = 0; index < sample_count; ++index) {
            caller_normalizer += std::fabs(static_cast<double>(abs_weights[index]));
        }
        if (caller_normalizer > 0.0) {
            for (std::size_t index = 0; index < sample_count; ++index) {
                abs_weights[index] = static_cast<float>(
                    static_cast<double>(abs_weights[index]) / caller_normalizer);
            }
        }
    }
    cv::Vec3f result(0.0F, 0.0F, 0.0F);
    float total_weight = 0.0F;
    for (std::size_t index = 0; index < sample_count; ++index) {
        float weight =
            average ? 1.0F
                    : std::max(selected.values[sample_indices[index]].score, 1.0e-6F);
        if (robust) {
            weight = abs_weights[index];
        }
        result += weight * colors[index];
        total_weight += weight;
    }
    // ColorBlend divides each accumulated float channel by the accumulated
    // weight with scalar/divps IEEE division.  cv::Vec's operator/ is allowed
    // to form a rounded reciprocal first and multiply, which differs at a few
    // nearest-even byte boundaries.
    const float denominator = std::max(total_weight, 1.0e-6F);
    result[0] /= denominator;
    result[1] /= denominator;
    result[2] /= denominator;
    return result;
}

std::vector<ExposurePoint> downsampleExposureCloud(const Options& options, const PlyInput& layout,
                                                   const Vec3f& cloud_minimum,
                                                   const Vec3f& cloud_maximum) {
    // nv_colorcloud uses its OctreeVoxelGrid at 0.1 m for the exposure pass.
    // PCL's octree is a power-of-two cube centered around the input bounds;
    // reproducing that anchor is significant (the crop has exactly 1881
    // occupied leaves, while a world-zero grid has 1962).
    constexpr double resolution = 0.1;
    const Eigen::Vector3d minimum = cloud_minimum.cast<double>();
    const Eigen::Vector3d maximum = cloud_maximum.cast<double>();
    // PCL 1.12's parameterless OctreePointCloud::defineBoundingBox() adds
    // exactly 2^-14 to each maximum float coordinate before fitting the
    // power-of-two cube.  Its centre (and therefore the fitted minimum) is
    // consequently shifted by 2^-15.  Both constants and the max-only add
    // are visible in defineBoundingBox() and were verified against the live
    // nv_colorcloud OctreeVoxelGrid object.
    constexpr double pcl_maximum_padding = 1.0 / 16384.0;
    const Eigen::Vector3d padded_maximum = maximum + Eigen::Vector3d::Constant(pcl_maximum_padding);
    const Eigen::Vector3d extent = padded_maximum - minimum;
    const double maximum_extent = extent.maxCoeff();
    const double levels = std::ceil(std::log2(std::max(maximum_extent / resolution, 1.0)));
    const double side = resolution * std::ldexp(1.0, static_cast<int>(levels));
    const Eigen::Vector3d octree_minimum =
        minimum - 0.5 * (Eigen::Vector3d::Constant(side) - extent);

    std::unordered_map<VoxelKey, ExposureVoxelAccumulator, VoxelKeyHash> voxels;
    voxels.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(layout.count, 2'000'000U)));
    std::ifstream input(options.input, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot reopen input for exposure downsampling");
    }
    input.seekg(layout.data_offset);
    std::vector<SurfacePoint> points(options.chunk_points);
    std::vector<cv::Vec3b> source_colors;
    std::uint64_t processed = 0U;
    while (processed < layout.count) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(options.chunk_points, layout.count - processed));
        readSurfacePoints(input, layout.colored_records, points, chunk, &source_colors);
        for (std::size_t index = 0; index < chunk; ++index) {
            const SurfacePoint& point = points[index];
            const Eigen::Vector3d shifted = xyz(point).cast<double>() - octree_minimum;
            const VoxelKey key{static_cast<int>(std::floor(shifted.x() / resolution)),
                               static_cast<int>(std::floor(shifted.y() / resolution)),
                               static_cast<int>(std::floor(shifted.z() / resolution))};
            auto& accumulator = voxels[key];

            // PointXYZRGBINormal enters the octree through the same adapter as
            // the PCT cloud.  The adapter normalizes every source normal before
            // NormalFieldAggregator accumulates it.  Normalizing only the final
            // centroid changes cancellation-heavy leaves and, consequently,
            // exposure visibility decisions near angular boundaries.
            float input_normal_squared = point.nz * point.nz;
            input_normal_squared += point.ny * point.ny;
            input_normal_squared += point.nx * point.nx;
            float input_nx = point.nx;
            float input_ny = point.ny;
            float input_nz = point.nz;
            if (input_normal_squared > 0.0F && std::isfinite(input_normal_squared)) {
                const float input_normal_norm = std::sqrt(input_normal_squared);
                input_nx /= input_normal_norm;
                input_ny /= input_normal_norm;
                input_nz /= input_normal_norm;
            }

            const std::array<float, 8> values{point.x,  point.y,  point.z,  point.intensity,
                                              input_nx, input_ny, input_nz, point.curvature};
            for (std::size_t field = 0; field < values.size(); ++field) {
                accumulator.surface_sum[field] += values[field];
            }
            if (!source_colors.empty()) {
                for (std::size_t channel = 0; channel < 3U; ++channel) {
                    accumulator.color_sum[channel] += source_colors[index][channel];
                }
            }
            ++accumulator.count;
        }
        processed += chunk;
        points.resize(options.chunk_points);
    }

    struct KeyedExposurePoint {
        VoxelKey key;
        ExposurePoint point;
    };
    std::vector<KeyedExposurePoint> keyed_points;
    keyed_points.reserve(voxels.size());
    for (const auto& [key, accumulator] : voxels) {
        const float count = static_cast<float>(std::max<std::uint32_t>(accumulator.count, 1U));
        ExposurePoint point;
        point.surface = {accumulator.surface_sum[0] / count, accumulator.surface_sum[1] / count,
                         accumulator.surface_sum[2] / count, accumulator.surface_sum[3] / count,
                         accumulator.surface_sum[4] / count, accumulator.surface_sum[5] / count,
                         accumulator.surface_sum[6] / count, accumulator.surface_sum[7] / count};
        // NormalFieldAggregator::getValue uses this scalar SSE association and
        // clears near-cancelled normals instead of amplifying them.  Both the
        // 1e-6 thresholds and the conditional renormalization are observable
        // in the installed implementation.
        const float normal_x_squared = point.surface.nx * point.surface.nx;
        const float normal_y_squared = point.surface.ny * point.surface.ny;
        float squared_norm = point.surface.nz * point.surface.nz;
        squared_norm += normal_y_squared;
        squared_norm += normal_x_squared;
        if (squared_norm < 1.0e-6F || !std::isfinite(squared_norm)) {
            point.surface.nx = 0.0F;
            point.surface.ny = 0.0F;
            point.surface.nz = 0.0F;
        } else if (std::abs(squared_norm - 1.0F) > 1.0e-6F) {
            const float normal_norm = std::sqrt(squared_norm);
            point.surface.nx /= normal_norm;
            point.surface.ny /= normal_norm;
            point.surface.nz /= normal_norm;
        }
        point.has_source_rgb = layout.colored_records;
        if (point.has_source_rgb) {
            for (std::size_t channel = 0; channel < 3U; ++channel) {
                point.source_rgb[channel] = cv::saturate_cast<std::uint8_t>(
                    accumulator.color_sum[channel] * (1.0 / static_cast<double>(count)));
            }
        }
        keyed_points.push_back({key, point});
    }

    // OctreePointCloud's depth-first leaf iterator emits X/Y/Z child bits in
    // ascending Morton order.  VoxelRanking consumes this order directly;
    // retaining unordered_map iteration here changes the global PRNG sequence
    // used for visibility probes in one-metre voxels farther than 10 metres.
    const int tree_depth = static_cast<int>(levels);
    std::sort(keyed_points.begin(), keyed_points.end(),
              [tree_depth](const KeyedExposurePoint& first,
                           const KeyedExposurePoint& second) {
                  for (int bit = tree_depth - 1; bit >= 0; --bit) {
                      const int first_values[3]{first.key.x, first.key.y, first.key.z};
                      const int second_values[3]{second.key.x, second.key.y, second.key.z};
                      for (int axis = 0; axis < 3; ++axis) {
                          const int first_bit = (first_values[axis] >> bit) & 1;
                          const int second_bit = (second_values[axis] >> bit) & 1;
                          if (first_bit != second_bit) {
                              return first_bit < second_bit;
                          }
                      }
                  }
                  return false;
              });

    std::vector<ExposurePoint> result;
    result.reserve(keyed_points.size());
    for (const KeyedExposurePoint& keyed : keyed_points) {
        result.push_back(keyed.point);
    }
    return result;
}

std::uint8_t histogram8uPercentile(const std::array<std::uint32_t, 256>& histogram,
                                   float probability) {
    const std::uint32_t total =
        std::accumulate(histogram.begin(), histogram.end(), std::uint32_t{0U});
    // Histogram8U::percentile in nv_colorcloud computes
    // lround(double(float(total) * probability + 0.5f)), clamps that rank to
    // at least one, then returns the first bin whose cumulative count reaches
    // it.  The seemingly redundant +0.5 is observable at integer quantiles.
    const std::uint64_t target = std::max<std::uint64_t>(
        1U, static_cast<std::uint64_t>(
                std::lround(static_cast<double>(static_cast<float>(total) * probability + 0.5F))));
    std::uint64_t cumulative = 0U;
    for (std::size_t value = 0; value < histogram.size(); ++value) {
        cumulative += histogram[value];
        if (cumulative >= target) {
            return static_cast<std::uint8_t>(value);
        }
    }
    return 255U;
}

double exposureNormalizedByte(std::uint8_t value) {
    // nv_colorcloud normalizes exposure intensities in float32 and only then
    // promotes them to double for GammaModel evaluation.  Keep the multiply
    // (rather than a double /255) because it is visible in residual probes.
    constexpr float inverse_255 = 1.0F / 255.0F;
    return static_cast<double>(static_cast<float>(value) * inverse_255);
}

ExposureProblem buildExposureProblem(
    const Options& options, const std::vector<ExposurePoint>& points,
    const std::vector<Pose>& view_poses, const std::array<OCamCamera, 4>& camera_models,
    const CameraMasks& camera_masks, int depth_height,
    const std::function<bool(int, float, float, float)>& depth_visible,
    const std::function<bool(int, float, float, float, float)>& ranking_depth_visible,
    CameraImageCache& camera_images, std::size_t camera_cache_capacity) {
    using ExposureClock = std::chrono::steady_clock;
    const auto exposure_ranking_started = ExposureClock::now();
    // VoxelRanking::rankVoxels does not average surface-normal scores over a
    // one-metre voxel.  It evaluates the world-zero-aligned voxel centre using
    // the camera optical-axis cosine.  The half diagonal is admitted behind
    // the focal plane because part of that voxel can still be in front.  These
    // details and the 10 m / 100 point visibility probe below are recovered
    // from the rankVoxels machine code and runtime traces.
    std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelKeyHash> voxel_points;
    voxel_points.reserve(points.size() / 32U + 1U);
    for (std::size_t index = 0; index < points.size(); ++index) {
        voxel_points[ovsVoxelKey(xyz(points[index].surface))].push_back(index);
    }
    std::unordered_map<VoxelKey, ExposureVoxelSelectedViews, VoxelKeyHash> rankings;
    rankings.reserve(voxel_points.size());
    std::minstd_rand ranking_random(42U);
    constexpr float voxel_resolution = 1.0F;
    constexpr float voxel_half_diagonal = 0.866025388F;
    constexpr float visibility_probe_distance = 10.0F;
    constexpr std::size_t visibility_probe_count = 100U;
    for (const auto& [key, point_indices] : voxel_points) {
        (void)point_indices;
        rankings.emplace(key, ExposureVoxelSelectedViews{});
    }
    struct RankedVoxel {
        VoxelKey key{};
        float score = 0.0F;
    };
    for (std::size_t view = 0; view < view_poses.size(); ++view) {
        const int camera = static_cast<int>(view % 4U);
        if (options.camera_index >= 0 && camera != options.camera_index) {
            continue;
        }
        std::vector<RankedVoxel> ranked_voxels;
        ranked_voxels.reserve(voxel_points.size());
        for (const auto& [key, point_indices] : voxel_points) {
            const Vec3f center(voxel_resolution * (static_cast<float>(key.x) + 0.5F),
                               voxel_resolution * (static_cast<float>(key.y) + 0.5F),
                               voxel_resolution * (static_cast<float>(key.z) + 0.5F));
            const Vec3f local_center = view_poses[view].inverseApply(center);
            const float range = local_center.norm();
            if (!(range > 1.0e-4F) || !std::isfinite(range) ||
                local_center.z() < -voxel_half_diagonal) {
                continue;
            }

            if (range > visibility_probe_distance) {
                std::vector<std::size_t> probes;
                probes.reserve(std::min(visibility_probe_count, point_indices.size()));
                std::sample(point_indices.begin(), point_indices.end(), std::back_inserter(probes),
                            std::min(visibility_probe_count, point_indices.size()), ranking_random);
                bool visible = false;
                for (const std::size_t point_index : probes) {
                    int pixel = 0;
                    float point_range = 0.0F;
                    float depth_x = 0.0F;
                    float depth_y = 0.0F;
                    if (projectOCamDepth(xyz(points[point_index].surface), view_poses[view],
                                         camera_models[view % 4U], camera_masks, camera,
                                         options.depth_width, depth_height, pixel, point_range,
                                         &depth_x, &depth_y) &&
                        ranking_depth_visible(static_cast<int>(view), depth_x, depth_y, point_range,
                                              2.0F * voxel_half_diagonal + 0.5F)) {
                        visible = true;
                        break;
                    }
                }
                if (!visible) {
                    continue;
                }
            }

            ranked_voxels.push_back(
                {key, (local_center.z() / range) * binaryDistanceWeight(range)});
        }
        std::sort(ranked_voxels.begin(), ranked_voxels.end(),
                  [](const RankedVoxel& first, const RankedVoxel& second) {
                      return first.score > second.score;
                  });
        // Do not truncate this list per view.  The installed implementation's
        // full-cloud exposure pass colors 261,325 of 416,789 0.1 m centroids
        // spread over 5,450 one-metre voxels.  A previously inferred limit of
        // 16 was applied to *voxels per view*, leaving only about 25k exposure
        // points and producing severely underconstrained GammaModels outside
        // the small 12-voxel crop.  The bounded cardinality is the five
        // observations stored per point below, not a global spatial cap.
        for (const RankedVoxel& ranked : ranked_voxels) {
            rankings.at(ranked.key).indices.push_back(static_cast<int>(view));
        }
    }
    const auto exposure_ranking_finished = ExposureClock::now();

    using ViewHistogram = std::array<std::uint32_t, 256>;
    // The binary constructs two independent histogram families from the same
    // OVS records.  Dynamic-range regularization admits every positive
    // float16 quality.  Scene brightness uses only qualities > 0.01.  Both
    // histograms are unweighted counts; decoded quality is accumulated only
    // for their Ceres loss weights.
    std::vector<ViewHistogram> dynamic_histograms(view_poses.size());
    std::vector<std::uint32_t> dynamic_counts(view_poses.size(), 0U);
    std::vector<double> dynamic_quality_totals(view_poses.size(), 0.0);
    std::vector<ViewHistogram> scene_histograms(view_poses.size());
    std::vector<std::uint32_t> scene_counts(view_poses.size(), 0U);
    std::vector<double> scene_quality_totals(view_poses.size(), 0.0);
    std::vector<std::uint32_t> ranking_counts(view_poses.size(), 0U);
    for (const auto& [key, ranking] : rankings) {
        (void)key;
        for (const int view : ranking.indices) {
            ++ranking_counts[static_cast<std::size_t>(view)];
        }
    }
    ExposureProblem problem;
    std::ofstream ovs_output;
    if (!options.exposure_ovs_output.empty()) {
        if (!options.exposure_ovs_output.parent_path().empty()) {
            fs::create_directories(options.exposure_ovs_output.parent_path());
        }
        ovs_output.open(options.exposure_ovs_output);
        if (!ovs_output) {
            throw std::runtime_error("cannot create exposure OVS dump " +
                                     options.exposure_ovs_output.string());
        }
    }
    std::ofstream ovs_binary_output;
    if (!options.exposure_ovs_binary_output.empty()) {
        if (!options.exposure_ovs_binary_output.parent_path().empty()) {
            fs::create_directories(options.exposure_ovs_binary_output.parent_path());
        }
        ovs_binary_output.open(options.exposure_ovs_binary_output,
                               std::ios::binary | std::ios::trunc);
        if (!ovs_binary_output) {
            throw std::runtime_error("cannot create binary exposure OVS dump " +
                                     options.exposure_ovs_binary_output.string());
        }
    }
    std::size_t colored_points = 0U;
    std::ofstream projection_output;
    if (!options.exposure_projection_output.empty()) {
        projection_output.open(options.exposure_projection_output);
        if (!projection_output) {
            throw std::runtime_error("cannot create exposure projection dump " +
                                     options.exposure_projection_output.string());
        }
        projection_output << std::setprecision(9);
    }

    // The installed OptimalViewSelection worker is view-major, which ensures
    // that each camera image is preprocessed once.  On this implementation's
    // unordered voxel map, however, a parallel barrier per view is slower than
    // point-major traversal.  The caller therefore keeps all exposure
    // captures resident; the view-major branch remains a bounded-memory
    // fallback.  Both branches visit each point's views in ascending order, so
    // Top-5 insertion and floating-point operation order remain unchanged.
    const auto exposure_sampling_started = ExposureClock::now();
    std::vector<SelectedSamples> selected_by_point(points.size());
    std::vector<Vec3f> scoring_normals(points.size());
#pragma omp parallel for schedule(static)
    for (std::int64_t signed_point_index = 0;
         signed_point_index < static_cast<std::int64_t>(points.size()); ++signed_point_index) {
        const std::size_t point_index = static_cast<std::size_t>(signed_point_index);
        const SurfacePoint& point = points[point_index].surface;
        const Vec3f normal(point.nx, point.ny, point.nz);
        if (!normal.allFinite()) {
            scoring_normals[point_index] = Vec3f::Constant(std::numeric_limits<float>::quiet_NaN());
            continue;
        }
        // The exposure selector wraps a 48-byte PCL PointXYZRGBINormal cloud
        // and reads its float normal directly.  Fibonacci uint16 coding is a
        // storage detail of the later 14-byte final-cloud adapter only.
        scoring_normals[point_index] = normal;
    }
    const auto sample_point_view =
        [&points, &scoring_normals, &view_poses, &camera_models, &camera_masks, &options,
         depth_height, &depth_visible,
         &projection_output](std::size_t point_index, int view,
                             const std::array<CameraImageCache::Pyramid, 4>& images,
                             bool emit_projection) -> std::optional<DirectSample> {
        const ExposurePoint& point = points[point_index];
        const int camera = view % 4;
        int pixel = 0;
        float range = 0.0F;
        float depth_x = 0.0F;
        float depth_y = 0.0F;
        if (!projectOCamDepth(xyz(point.surface), view_poses[static_cast<std::size_t>(view)],
                              camera_models[static_cast<std::size_t>(camera)], camera_masks, camera,
                              options.depth_width, depth_height, pixel, range, &depth_x,
                              &depth_y)) {
            return std::nullopt;
        }
        if (emit_projection) {
            projection_output << "PROJ " << view << ' ' << point_index << ' ' << depth_x << ' '
                              << depth_y << ' ' << range << '\n';
        }
        if (!depth_visible(view, depth_x, depth_y, range)) {
            return std::nullopt;
        }
        const auto sample = directCameraSample(
            point.surface, view_poses[static_cast<std::size_t>(view)],
            camera_models[static_cast<std::size_t>(camera)],
            images[static_cast<std::size_t>(camera)], camera_masks, camera,
            scoring_normals[point_index], options.score_incidence_power, options.score_radius_power,
            options.score_distance_power, options.view_max_distance, view, 0.1F);
        if (emit_projection && sample) {
            projection_output << "SAMPLE " << view << ' ' << point_index << ' ' << sample->score
                              << ' ' << quantizeUnitScore(sample->score) << '\n';
        }
        return sample;
    };

    const std::size_t capture_count = (view_poses.size() + 3U) / 4U;
    const bool use_view_major = !projection_output && capture_count > camera_cache_capacity;
    if (!use_view_major) {
        // Debug dumps have a documented point-major order.  Point-major is
        // also faster when every capture fits in the LRU, because it avoids a
        // parallel-region barrier per view without causing image re-decodes.
        for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
            const auto ranking = rankings.find(ovsVoxelKey(xyz(points[point_index].surface)));
            if (ranking == rankings.end()) {
                continue;
            }
            for (const int view : ranking->second.indices) {
                const auto& images = camera_images.get(view / 4);
                const auto sample = sample_point_view(point_index, view, images,
                                                      static_cast<bool>(projection_output));
                if (sample) {
                    insertSelectedSample(selected_by_point[point_index], *sample);
                }
            }
        }
    } else {
        const auto& const_rankings = rankings;
        for (std::size_t view_index = 0; view_index < view_poses.size(); ++view_index) {
            if (ranking_counts[view_index] == 0U) {
                continue;
            }
            const int view = static_cast<int>(view_index);
            const auto& images = camera_images.get(view / 4);
#pragma omp parallel for schedule(static)
            for (std::int64_t signed_point_index = 0;
                 signed_point_index < static_cast<std::int64_t>(points.size());
                 ++signed_point_index) {
                const std::size_t point_index = static_cast<std::size_t>(signed_point_index);
                const auto ranking =
                    const_rankings.find(ovsVoxelKey(xyz(points[point_index].surface)));
                if (ranking == const_rankings.end() ||
                    !std::binary_search(ranking->second.indices.begin(),
                                        ranking->second.indices.end(), view)) {
                    continue;
                }
                const auto sample = sample_point_view(point_index, view, images, false);
                if (sample) {
                    insertSelectedSample(selected_by_point[point_index], *sample);
                }
            }
        }
    }
    const auto exposure_sampling_finished = ExposureClock::now();

    for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
        std::array<std::uint8_t, kSelectedViews * 8U> packed_ovs{};
        const SelectedSamples& selected_samples = selected_by_point[point_index];
        if (selected_samples.count == 0U) {
            if (ovs_binary_output) {
                ovs_binary_output.write(reinterpret_cast<const char*>(packed_ovs.data()),
                                        static_cast<std::streamsize>(packed_ovs.size()));
            }
            continue;
        }
        ++colored_points;
        ExposureJointSample joint;
        const cv::Vec3b reference_rgb = selected_samples.values[0].rgb;
        // The exposure OVS path does not use AdaptiveBestScore.  Runtime
        // records show that it first stores each raw score as float16_01_t,
        // divides those decoded scores by their per-point sum, then stores the
        // normalized values as float16_01_t again.  Their encoded sum is thus
        // 65535 minus only the truncation remainder.
        std::array<float, kSelectedViews> normalized_qualities{};
        float raw_quality_sum = 0.0F;
        for (std::uint8_t index = 0; index < selected_samples.count; ++index) {
            normalized_qualities[index] =
                static_cast<float>(quantizeUnitScore(selected_samples.values[index].score)) /
                65535.0F;
            raw_quality_sum += normalized_qualities[index];
        }
        if (raw_quality_sum > 0.0F) {
            for (std::uint8_t index = 0; index < selected_samples.count; ++index) {
                normalized_qualities[index] = static_cast<float>(quantizeUnitScore(
                                                  normalized_qualities[index] / raw_quality_sum)) /
                                              65535.0F;
            }
        }
        if (ovs_binary_output) {
            for (std::uint8_t index = 0; index < selected_samples.count; ++index) {
                const DirectSample& sample = selected_samples.values[index];
                const unsigned capture = static_cast<unsigned>(sample.view / 4);
                const std::uint16_t quality =
                    static_cast<std::uint16_t>(std::lround(normalized_qualities[index] * 65535.0F));
                const std::size_t offset = static_cast<std::size_t>(index) * 8U;
                packed_ovs[offset + 0U] = static_cast<std::uint8_t>((capture >> 8U) & 0xffU);
                packed_ovs[offset + 1U] = static_cast<std::uint8_t>(capture & 0xffU);
                packed_ovs[offset + 2U] = static_cast<std::uint8_t>(sample.view % 4);
                packed_ovs[offset + 3U] = sample.rgb[0];
                packed_ovs[offset + 4U] = sample.rgb[1];
                packed_ovs[offset + 5U] = sample.rgb[2];
                packed_ovs[offset + 6U] = static_cast<std::uint8_t>(quality & 0xffU);
                packed_ovs[offset + 7U] = static_cast<std::uint8_t>(quality >> 8U);
            }
            ovs_binary_output.write(reinterpret_cast<const char*>(packed_ovs.data()),
                                    static_cast<std::streamsize>(packed_ovs.size()));
        }
        if (ovs_output) {
            for (std::uint8_t index = 0; index < selected_samples.count; ++index) {
                const DirectSample& sample = selected_samples.values[index];
                ovs_output << "OBS " << point_index << ' ' << static_cast<int>(index) << ' '
                           << sample.view / 4 << ' ' << sample.view % 4 << ' '
                           << static_cast<int>(sample.rgb[0]) << ' '
                           << static_cast<int>(sample.rgb[1]) << ' '
                           << static_cast<int>(sample.rgb[2]) << ' '
                           << static_cast<int>(std::lround(normalized_qualities[index] * 65535.0F))
                           << ' ' << sample.image_x << ' ' << sample.image_y << ' '
                           << static_cast<int>(sample.raw_rgb[0]) << ' '
                           << static_cast<int>(sample.raw_rgb[1]) << ' '
                           << static_cast<int>(sample.raw_rgb[2]) << '\n';
            }
        }
        for (std::uint8_t index = 0; index < selected_samples.count; ++index) {
            const DirectSample& sample = selected_samples.values[index];
            const float quality = normalized_qualities[index];
            // Exposure observations use HSV value, i.e. max(B,G,R), not the
            // minimum channel.  This reproduces all 1577 original joint
            // residual blocks and their 5307 observation residuals.
            const std::uint8_t intensity = std::max({sample.rgb[0], sample.rgb[1], sample.rgb[2]});
            const bool histogram_sample =
                !options.exposure_histogram_exclude_zero || intensity != 0U;
            if (histogram_sample && quality > 0.0F) {
                ++dynamic_histograms[static_cast<std::size_t>(sample.view)][intensity];
                dynamic_quality_totals[static_cast<std::size_t>(sample.view)] += quality;
                ++dynamic_counts[static_cast<std::size_t>(sample.view)];
            }
            if (histogram_sample && quality > 0.01F) {
                ++scene_histograms[static_cast<std::size_t>(sample.view)][intensity];
                scene_quality_totals[static_cast<std::size_t>(sample.view)] += quality;
                ++scene_counts[static_cast<std::size_t>(sample.view)];
            }
            if (quality <= 0.0F) {
                continue;
            }
            if (intensity < 6U || intensity > 249U) {
                continue;
            }
            if (index > 0U && options.exposure_source_consistency) {
                int maximum_difference = 0;
                for (int channel = 0; channel < 3; ++channel) {
                    maximum_difference = std::max(
                        maximum_difference, std::abs(static_cast<int>(sample.rgb[channel]) -
                                                     static_cast<int>(reference_rgb[channel])));
                }
                if (maximum_difference > 50) {
                    continue;
                }
            }
            joint.observations[joint.count++] = {sample.view, intensity, quality};
        }
        if (joint.count >= 2U) {
            problem.joint_samples.push_back(joint);
        }
    }
    const auto exposure_reduction_finished = ExposureClock::now();

    if (ovs_binary_output.is_open()) {
        ovs_binary_output.flush();
        if (!ovs_binary_output) {
            throw std::runtime_error("failed while writing binary exposure OVS dump");
        }
    }

    const double total_dynamic_quality =
        std::accumulate(dynamic_quality_totals.begin(), dynamic_quality_totals.end(), 0.0);
    const double total_scene_quality =
        std::accumulate(scene_quality_totals.begin(), scene_quality_totals.end(), 0.0);
    const double joint_count = static_cast<double>(problem.joint_samples.size());
    for (std::size_t view = 0; view < view_poses.size(); ++view) {
        const std::size_t distinct = static_cast<std::size_t>(
            std::count_if(dynamic_histograms[view].begin(), dynamic_histograms[view].end(),
                          [](std::uint32_t value) { return value > 0U; }));
        if (distinct <= 1U || dynamic_quality_totals[view] <= 0.0 || total_dynamic_quality <= 0.0) {
            continue;
        }
        const std::uint8_t low = histogram8uPercentile(dynamic_histograms[view], 0.02F);
        const std::uint8_t high = histogram8uPercentile(dynamic_histograms[view], 0.98F);
        if (low >= high) {
            continue;
        }
        const double normalized_weight = dynamic_quality_totals[view] / total_dynamic_quality;
        problem.dynamic_ranges.push_back(
            {static_cast<int>(view), low, high, normalized_weight, 0.0});
    }
    normalizeDynamicRanges(problem.dynamic_ranges, joint_count);
    for (std::size_t view = 0; view < view_poses.size(); ++view) {
        const std::size_t distinct = static_cast<std::size_t>(
            std::count_if(scene_histograms[view].begin(), scene_histograms[view].end(),
                          [](std::uint32_t value) { return value > 0U; }));
        if (scene_counts[view] <= 9U || distinct <= 1U || scene_quality_totals[view] <= 0.0 ||
            total_scene_quality <= 0.0) {
            continue;
        }
        const std::uint8_t low = histogram8uPercentile(scene_histograms[view], 0.02F);
        const std::uint8_t high = histogram8uPercentile(scene_histograms[view], 0.98F);
        if (low >= high) {
            continue;
        }
        const double normalized_weight =
            scene_quality_totals[view] / total_scene_quality;
        problem.scene_ranges.push_back({static_cast<int>(view), low, high,
                                        histogram8uPercentile(scene_histograms[view], 0.5F),
                                        static_cast<float>(normalized_weight)});
    }

    std::vector<bool> active(view_poses.size(), false);
    for (const ExposureJointSample& joint : problem.joint_samples) {
        for (std::uint8_t index = 0; index < joint.count; ++index) {
            active[static_cast<std::size_t>(joint.observations[index].view)] = true;
        }
    }
    for (const ExposureDynamicRange& range : problem.dynamic_ranges) {
        active[static_cast<std::size_t>(range.view)] = true;
    }
    for (const ExposureSceneRange& range : problem.scene_ranges) {
        active[static_cast<std::size_t>(range.view)] = true;
    }
    for (std::size_t view = 0; view < active.size(); ++view) {
        if (active[view]) {
            problem.active_views.push_back(static_cast<int>(view));
        }
    }

    std::size_t joint_residuals = 0U;
    for (const auto& joint : problem.joint_samples) {
        joint_residuals += joint.count;
    }
    const auto seconds_between = [](const auto& begin, const auto& end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    std::cerr << "Exposure OVS: " << points.size() << " points, " << rankings.size()
              << " one-meter voxels; " << colored_points
              << " points with color\nExposure objective: " << problem.joint_samples.size()
              << " joint blocks / " << joint_residuals << " residuals, "
              << problem.dynamic_ranges.size() << " dynamic blocks, " << problem.scene_ranges.size()
              << " scene ranges, " << problem.active_views.size() << " active views\n"
              << "Exposure sampling strategy: "
              << (use_view_major ? "view-major parallel" : "point-major cache-resident") << " ("
              << capture_count << " captures, cache " << camera_cache_capacity << ")\n"
              << "Exposure timing: voxel/ranking "
              << seconds_between(exposure_ranking_started, exposure_ranking_finished)
              << " s, view sampling "
              << seconds_between(exposure_sampling_started, exposure_sampling_finished)
              << " s, ordered reduction "
              << seconds_between(exposure_sampling_finished, exposure_reduction_finished) << " s\n";
    for (std::size_t view = 0; view < view_poses.size(); ++view) {
        if (ranking_counts[view] == 0U && dynamic_counts[view] == 0U) {
            continue;
        }
        std::cerr << "  exposure view " << view << ": ranked in " << ranking_counts[view]
                  << " voxels, dynamic/scene samples " << dynamic_counts[view] << '/'
                  << scene_counts[view] << ", dynamic/scene quality sums "
                  << dynamic_quality_totals[view] << '/' << scene_quality_totals[view] << '\n';
    }
    return problem;
}

ExposureProblem loadExposureProblemFromPackedOvs(const Options& options, std::size_t view_count) {
    constexpr std::size_t record_size = kSelectedViews * 8U;
    const std::uintmax_t byte_count = fs::file_size(options.exposure_ovs_binary_input);
    if (byte_count % record_size != 0U) {
        throw std::runtime_error("packed exposure OVS size is not divisible by 40 bytes");
    }
    std::ifstream input(options.exposure_ovs_binary_input, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open packed exposure OVS input");
    }

    using ViewHistogram = std::array<std::uint32_t, 256>;
    std::vector<ViewHistogram> dynamic_histograms(view_count);
    std::vector<double> dynamic_quality_totals(view_count, 0.0);
    std::vector<ViewHistogram> scene_histograms(view_count);
    std::vector<std::uint32_t> scene_counts(view_count, 0U);
    std::vector<double> scene_quality_totals(view_count, 0.0);
    ExposureProblem problem;
    problem.joint_samples.reserve(static_cast<std::size_t>(byte_count / record_size / 2U));

    std::array<std::uint8_t, record_size> record{};
    while (input.read(reinterpret_cast<char*>(record.data()),
                      static_cast<std::streamsize>(record.size()))) {
        ExposureJointSample joint;
        const cv::Vec3b reference_rgb(record[3], record[4], record[5]);
        for (std::size_t rank = 0; rank < kSelectedViews; ++rank) {
            const std::size_t offset = rank * 8U;
            const std::uint16_t packed_quality =
                static_cast<std::uint16_t>(static_cast<std::uint16_t>(record[offset + 6U]) |
                                           (static_cast<std::uint16_t>(record[offset + 7U]) << 8U));
            if (packed_quality == 0U) {
                continue;
            }
            const std::uint16_t capture =
                static_cast<std::uint16_t>((static_cast<std::uint16_t>(record[offset]) << 8U) |
                                           static_cast<std::uint16_t>(record[offset + 1U]));
            const int view = 4 * static_cast<int>(capture) + static_cast<int>(record[offset + 2U]);
            if (view < 0 || static_cast<std::size_t>(view) >= view_count) {
                throw std::runtime_error("packed exposure OVS contains an invalid view");
            }
            const cv::Vec3b rgb(record[offset + 3U], record[offset + 4U], record[offset + 5U]);
            const std::uint8_t intensity = std::max({rgb[0], rgb[1], rgb[2]});
            const float quality = static_cast<float>(packed_quality) / 65535.0F;
            const bool histogram_sample =
                !options.exposure_histogram_exclude_zero || intensity != 0U;
            if (histogram_sample) {
                ++dynamic_histograms[static_cast<std::size_t>(view)][intensity];
                dynamic_quality_totals[static_cast<std::size_t>(view)] += quality;
                if (quality > 0.01F) {
                    ++scene_histograms[static_cast<std::size_t>(view)][intensity];
                    ++scene_counts[static_cast<std::size_t>(view)];
                    scene_quality_totals[static_cast<std::size_t>(view)] += quality;
                }
            }
            if (intensity < 6U || intensity > 249U) {
                continue;
            }
            if (rank > 0U && options.exposure_source_consistency) {
                int maximum_difference = 0;
                for (int channel = 0; channel < 3; ++channel) {
                    maximum_difference = std::max(
                        maximum_difference, std::abs(static_cast<int>(rgb[channel]) -
                                                     static_cast<int>(reference_rgb[channel])));
                }
                if (maximum_difference > 50) {
                    continue;
                }
            }
            joint.observations[joint.count++] = {view, intensity, quality};
        }
        if (joint.count >= 2U) {
            problem.joint_samples.push_back(joint);
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading packed exposure OVS");
    }

    const double total_dynamic_quality =
        std::accumulate(dynamic_quality_totals.begin(), dynamic_quality_totals.end(), 0.0);
    const double total_scene_quality =
        std::accumulate(scene_quality_totals.begin(), scene_quality_totals.end(), 0.0);
    const double joint_count = static_cast<double>(problem.joint_samples.size());
    for (std::size_t view = 0; view < view_count; ++view) {
        const std::size_t distinct = static_cast<std::size_t>(
            std::count_if(dynamic_histograms[view].begin(), dynamic_histograms[view].end(),
                          [](std::uint32_t value) { return value > 0U; }));
        if (distinct <= 1U || dynamic_quality_totals[view] <= 0.0 || total_dynamic_quality <= 0.0) {
            continue;
        }
        const std::uint8_t low = histogram8uPercentile(dynamic_histograms[view], 0.02F);
        const std::uint8_t high = histogram8uPercentile(dynamic_histograms[view], 0.98F);
        if (low >= high) {
            continue;
        }
        const double normalized_weight = dynamic_quality_totals[view] / total_dynamic_quality;
        problem.dynamic_ranges.push_back(
            {static_cast<int>(view), low, high, normalized_weight, 0.0});
    }
    normalizeDynamicRanges(problem.dynamic_ranges, joint_count);
    for (std::size_t view = 0; view < view_count; ++view) {
        const std::size_t distinct = static_cast<std::size_t>(
            std::count_if(scene_histograms[view].begin(), scene_histograms[view].end(),
                          [](std::uint32_t value) { return value > 0U; }));
        if (scene_counts[view] <= 9U || distinct <= 1U || scene_quality_totals[view] <= 0.0 ||
            total_scene_quality <= 0.0) {
            continue;
        }
        const std::uint8_t low = histogram8uPercentile(scene_histograms[view], 0.02F);
        const std::uint8_t high = histogram8uPercentile(scene_histograms[view], 0.98F);
        if (low >= high) {
            continue;
        }
        const double normalized_weight =
            scene_quality_totals[view] / total_scene_quality;
        problem.scene_ranges.push_back({static_cast<int>(view), low, high,
                                        histogram8uPercentile(scene_histograms[view], 0.5F),
                                        static_cast<float>(normalized_weight)});
    }

    std::vector<bool> active(view_count, false);
    for (const ExposureJointSample& joint : problem.joint_samples) {
        for (std::uint8_t index = 0; index < joint.count; ++index) {
            active[static_cast<std::size_t>(joint.observations[index].view)] = true;
        }
    }
    for (const ExposureDynamicRange& range : problem.dynamic_ranges) {
        active[static_cast<std::size_t>(range.view)] = true;
    }
    for (const ExposureSceneRange& range : problem.scene_ranges) {
        active[static_cast<std::size_t>(range.view)] = true;
    }
    for (std::size_t view = 0; view < active.size(); ++view) {
        if (active[view]) {
            problem.active_views.push_back(static_cast<int>(view));
        }
    }

    std::size_t residual_count = 0U;
    for (const ExposureJointSample& joint : problem.joint_samples) {
        residual_count += joint.count;
    }
    std::cerr << "Loaded exposure objective from packed OVS: " << problem.joint_samples.size()
              << " joint blocks / " << residual_count << " residuals, "
              << problem.dynamic_ranges.size() << " dynamic blocks, " << problem.scene_ranges.size()
              << " scene ranges\n";
    return problem;
}

Eigen::VectorXd evaluateExposureResiduals(const ExposureProblem& problem,
                                          const std::vector<GammaModel>& models) {
    std::size_t residual_count = 3U * problem.dynamic_ranges.size();
    for (const auto& joint : problem.joint_samples) {
        residual_count += joint.count;
    }
    double scene_weight = 0.0;
    for (const auto& range : problem.scene_ranges) {
        if (range.high > 203U) {
            scene_weight += range.normalized_weight;
        }
    }
    if (scene_weight > 0.0) {
        ++residual_count;
    }
    Eigen::VectorXd residuals(static_cast<Eigen::Index>(residual_count));
    Eigen::Index cursor = 0;
    // GlobalExposureOptimizer owns one CauchyLoss(0.005) and attaches it to
    // every JointVarianceResidual block.  The two doubles observed in that
    // loss object are a^2=2.5e-5 and 1/a^2=40000.  Convert each block to an
    // equivalent residual vector whose squared norm is rho(||r||^2); this
    // preserves the exact robust objective for the numerical LM below.
    constexpr double joint_cauchy_squared = 2.5e-5;
    for (const ExposureJointSample& joint : problem.joint_samples) {
        double total_weight = 0.0;
        double weighted_intensity = 0.0;
        std::array<double, kSelectedViews> corrected{};
        std::array<double, kSelectedViews> raw_residuals{};
        for (std::uint8_t index = 0; index < joint.count; ++index) {
            const ExposureObservation& observation = joint.observations[index];
            const GammaModel& model = models[static_cast<std::size_t>(observation.view)];
            const double input = exposureNormalizedByte(observation.intensity);
            corrected[index] = model.gain * std::pow(input, model.exponent);
            total_weight += observation.weight;
            weighted_intensity += observation.weight * corrected[index];
        }
        const double mean = total_weight > 0.0 ? weighted_intensity / total_weight : 0.0;
        double squared_norm = 0.0;
        for (std::uint8_t index = 0; index < joint.count; ++index) {
            const double difference = corrected[index] - mean;
            raw_residuals[index] = joint.observations[index].weight * difference * difference;
            squared_norm += raw_residuals[index] * raw_residuals[index];
        }
        const double robust_norm_squared =
            joint_cauchy_squared * std::log1p(squared_norm / joint_cauchy_squared);
        const double robust_scale =
            squared_norm > 0.0 ? std::sqrt(robust_norm_squared / squared_norm) : 1.0;
        for (std::uint8_t index = 0; index < joint.count; ++index) {
            residuals[cursor++] = robust_scale * raw_residuals[index];
        }
    }
    for (const ExposureDynamicRange& range : problem.dynamic_ranges) {
        const GammaModel& model = models[static_cast<std::size_t>(range.view)];
        const double low = exposureNormalizedByte(range.low);
        const double high = exposureNormalizedByte(range.high);
        const double corrected_low = model.gain * std::pow(low, model.exponent);
        const double corrected_high = model.gain * std::pow(high, model.exponent);
        std::array<double, 3> raw_residuals{1.1 - (corrected_high - corrected_low) / (high - low),
                                            std::min(corrected_low, 0.0),
                                            std::max(corrected_high - 1.0, 0.0)};
        // The installed binary wraps every three-residual dynamic block in
        // ScaledLoss(CauchyLoss(0.1), range.loss_scale).  Represent that
        // robust block by an equivalent residual norm for the numerical LM.
        constexpr double dynamic_cauchy_squared = 0.01;
        const double squared_norm = raw_residuals[0] * raw_residuals[0] +
                                    raw_residuals[1] * raw_residuals[1] +
                                    raw_residuals[2] * raw_residuals[2];
        const double robust_norm_squared =
            dynamic_cauchy_squared * std::log1p(squared_norm / dynamic_cauchy_squared);
        const double robust_scale =
            squared_norm > 0.0 ? std::sqrt(range.loss_scale * robust_norm_squared / squared_norm)
                               : std::sqrt(range.loss_scale);
        for (const double residual : raw_residuals) {
            residuals[cursor++] = robust_scale * residual;
        }
    }
    if (scene_weight > 0.0) {
        double corrected_high = 0.0;
        double original_high = 0.0;
        for (const ExposureSceneRange& range : problem.scene_ranges) {
            if (range.high <= 203U) {
                continue;
            }
            const GammaModel& model = models[static_cast<std::size_t>(range.view)];
            const double high = exposureNormalizedByte(range.high);
            corrected_high += range.normalized_weight * model.gain * std::pow(high, model.exponent);
            original_high += range.normalized_weight * high;
        }
        corrected_high /= scene_weight;
        original_high /= scene_weight;
        residuals[cursor++] = std::sqrt(static_cast<double>(problem.joint_samples.size())) *
                              (corrected_high - 1.1 * original_high);
    }
    return residuals;
}

#ifdef NAVVIS_RECON_HAVE_CERES
struct JointVarianceResidual {
  public:
    explicit JointVarianceResidual(const ExposureJointSample& sample) : sample_(sample) {}

    template <typename T> bool operator()(T const* const* parameters, T* residuals) const {
        using std::pow;
        std::array<T, kSelectedViews> corrected{};
        T total_weight = T(0.0);
        T weighted_intensity = T(0.0);
        for (std::uint8_t index = 0; index < sample_.count; ++index) {
            const ExposureObservation& observation = sample_.observations[index];
            const double input = exposureNormalizedByte(observation.intensity);
            const T powered = pow(T(input), parameters[index][1]);
            corrected[index] = parameters[index][0] * powered;
            total_weight += T(observation.weight);
            weighted_intensity += T(observation.weight) * corrected[index];
        }
        const T mean = total_weight > T(0.0) ? weighted_intensity / total_weight : T(0.0);
        for (std::uint8_t row = 0; row < sample_.count; ++row) {
            const T difference = corrected[row] - mean;
            // JointVarianceResidual squares the brightness difference before
            // applying its per-observation quality.  The grouping is visible
            // in the low bits of both the residual and its Jet derivatives.
            residuals[row] =
                T(sample_.observations[row].weight) * (difference * difference);
        }
        return true;
    }

  private:
    ExposureJointSample sample_;
};

struct DynamicRangeResidual {
  public:
    explicit DynamicRangeResidual(const ExposureDynamicRange& range) : range_(range) {}

    template <typename T> bool operator()(const T* parameters, T* residuals) const {
        using std::pow;
        const T gain = parameters[0];
        const T exponent = parameters[1];
        const double low = exposureNormalizedByte(range_.low);
        const double high = exposureNormalizedByte(range_.high);
        const T corrected_low = gain * pow(T(low), exponent);
        const T corrected_high = gain * pow(T(high), exponent);
        residuals[0] = T(1.1) - (corrected_high - corrected_low) / T(high - low);
        residuals[1] = corrected_low < T(0.0) ? corrected_low : T(0.0);
        residuals[2] = corrected_high > T(1.0) ? corrected_high - T(1.0) : T(0.0);
        return true;
    }

  private:
    ExposureDynamicRange range_;
};

struct SceneBrightnessResidual {
  public:
    explicit SceneBrightnessResidual(const std::vector<ExposureSceneRange>& ranges)
        : ranges_(ranges) {}

    template <typename T> bool operator()(T const* const* parameters, T* residuals) const {
        using std::pow;
        T selected_weight = T(0.0);
        T corrected_high = T(0.0);
        T original_high = T(0.0);
        for (std::size_t index = 0; index < ranges_.size(); ++index) {
            const ExposureSceneRange& range = ranges_[index];
            if (range.high <= 203U) {
                continue;
            }
            const double high = exposureNormalizedByte(range.high);
            const T corrected = parameters[index][0] * pow(T(high), parameters[index][1]);
            const T normalized_weight = T(range.normalized_weight);
            selected_weight += normalized_weight;
            corrected_high += normalized_weight * corrected;
            original_high += normalized_weight * T(high);
        }
        residuals[0] = selected_weight > T(0.0) ? corrected_high / selected_weight -
                                                      T(1.1) * original_high / selected_weight
                                                : T(0.0);
        return true;
    }

  private:
    std::vector<ExposureSceneRange> ranges_;
};
#endif

std::vector<GammaModel> solveExposureProblem(const ExposureProblem& problem,
                                             std::vector<GammaModel> models,
                                             int solver_threads, int max_iterations,
                                             double initial_trust_region_radius) {
    const std::size_t view_count = models.size();
    if (problem.joint_samples.empty() || problem.active_views.empty()) {
        return models;
    }
#ifdef NAVVIS_RECON_HAVE_CERES
    ceres::Problem optimizer;
    auto* joint_loss = new ceres::CauchyLoss(0.005);
    for (const ExposureJointSample& joint : problem.joint_samples) {
        auto* cost = new ceres::DynamicAutoDiffCostFunction<JointVarianceResidual, 4>(
            new JointVarianceResidual(joint));
        std::vector<double*> parameter_blocks;
        parameter_blocks.reserve(joint.count);
        for (std::uint8_t index = 0; index < joint.count; ++index) {
            cost->AddParameterBlock(2);
            parameter_blocks.push_back(
                &models[static_cast<std::size_t>(joint.observations[index].view)].gain);
        }
        cost->SetNumResiduals(static_cast<int>(joint.count));
        optimizer.AddResidualBlock(cost, joint_loss, parameter_blocks);
    }
    for (const ExposureDynamicRange& range : problem.dynamic_ranges) {
        auto* scaled_loss = new ceres::ScaledLoss(new ceres::CauchyLoss(0.1), range.loss_scale,
                                                  ceres::TAKE_OWNERSHIP);
        optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<DynamicRangeResidual, 3, 2>(
                                       new DynamicRangeResidual(range)),
                                   scaled_loss, &models[static_cast<std::size_t>(range.view)].gain);
    }
    double scene_weight = 0.0;
    std::vector<double*> scene_parameter_blocks;
    scene_parameter_blocks.reserve(problem.scene_ranges.size());
    for (const ExposureSceneRange& range : problem.scene_ranges) {
        if (range.high > 203U) {
            scene_weight += range.normalized_weight;
        }
        scene_parameter_blocks.push_back(&models[static_cast<std::size_t>(range.view)].gain);
    }
    if (scene_weight > 0.0) {
        auto* cost = new ceres::DynamicAutoDiffCostFunction<SceneBrightnessResidual, 10>(
            new SceneBrightnessResidual(problem.scene_ranges));
        for (std::size_t index = 0; index < problem.scene_ranges.size(); ++index) {
            cost->AddParameterBlock(2);
        }
        cost->SetNumResiduals(1);
        auto* scaled_loss = new ceres::ScaledLoss(
            nullptr, static_cast<double>(problem.joint_samples.size()), ceres::TAKE_OWNERSHIP);
        optimizer.AddResidualBlock(cost, scaled_loss, scene_parameter_blocks);
    }

    // nv_colorcloud lets AddResidualBlock register model parameter blocks on
    // first use.  CGNR/Jacobi is sensitive to that first-observation order;
    // pre-registering views in numeric order produces a measurably different
    // exposure solution even though the objective is otherwise identical.
    // Every active view has appeared by this point, so setting bounds now
    // preserves the binary's lazy parameter ordering.
    for (const int view : problem.active_views) {
        double* parameters = &models[static_cast<std::size_t>(view)].gain;
        optimizer.SetParameterLowerBound(parameters, 0, 0.2);
        optimizer.SetParameterUpperBound(parameters, 0, 5.0);
        optimizer.SetParameterLowerBound(parameters, 1, 0.66);
        optimizer.SetParameterUpperBound(parameters, 1, 1.5);
    }

    ceres::Solver::Options solver_options;
    solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options.linear_solver_type = ceres::CGNR;
    solver_options.preconditioner_type = ceres::JACOBI;
    solver_options.max_num_iterations = max_iterations;
    solver_options.function_tolerance = 1.0e-6;
    solver_options.initial_trust_region_radius = initial_trust_region_radius;
    // The installed worker passes 32 here.  Keep it explicit rather than
    // tying the nonlinear trajectory to the host's logical CPU count.
    solver_options.num_threads = solver_threads;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &optimizer, &summary);
    std::cerr << "Exposure Ceres: " << summary.BriefReport() << '\n';
    for (const int view : problem.active_views) {
        const GammaModel& model = models[static_cast<std::size_t>(view)];
        std::cerr << "  GammaModel view " << view << ": gain=" << model.gain
                  << " exponent=" << model.exponent << '\n';
    }
    return models;
#else
    const std::size_t parameter_count = 2U * problem.active_views.size();
    Eigen::VectorXd parameters = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(parameter_count));
    const auto models_from_parameters = [&](const Eigen::VectorXd& values) {
        std::vector<GammaModel> evaluated(view_count);
        for (std::size_t index = 0; index < problem.active_views.size(); ++index) {
            evaluated[static_cast<std::size_t>(problem.active_views[index])] = {
                values[static_cast<Eigen::Index>(2U * index)],
                values[static_cast<Eigen::Index>(2U * index + 1U)]};
        }
        return evaluated;
    };
    const auto clamp_parameters = [](Eigen::VectorXd& values) {
        // Bounds passed by the binary to ceres::Problem for GammaModel:
        // gain in [0.2, 5.0], exponent in [0.66, 1.5].
        for (Eigen::Index index = 0; index < values.size(); index += 2) {
            values[index] = std::clamp(values[index], 0.2, 5.0);
            values[index + 1] = std::clamp(values[index + 1], 0.66, 1.5);
        }
    };
    Eigen::VectorXd residuals =
        evaluateExposureResiduals(problem, models_from_parameters(parameters));
    double cost = 0.5 * residuals.squaredNorm();
    std::cerr << std::setprecision(12) << "Exposure LM: " << parameter_count << " parameters, "
              << residuals.size() << " residuals, initial cost " << cost << '\n';
    double lambda = 1.0e-3;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        Eigen::MatrixXd jacobian(residuals.size(), parameters.size());
        for (Eigen::Index column = 0; column < parameters.size(); ++column) {
            Eigen::VectorXd perturbed = parameters;
            const double step = 1.0e-6 * std::max(1.0, std::abs(parameters[column]));
            perturbed[column] += step;
            jacobian.col(column) =
                (evaluateExposureResiduals(problem, models_from_parameters(perturbed)) -
                 residuals) /
                step;
        }
        const Eigen::VectorXd gradient = jacobian.transpose() * residuals;
        if (gradient.lpNorm<Eigen::Infinity>() < 1.0e-12) {
            break;
        }
        const Eigen::MatrixXd normal = jacobian.transpose() * jacobian;
        Eigen::VectorXd diagonal = normal.diagonal().cwiseAbs().cwiseMax(1.0e-12);
        bool accepted = false;
        for (int trial = 0; trial < 12; ++trial) {
            Eigen::MatrixXd damped = normal;
            damped.diagonal() += lambda * diagonal;
            const Eigen::VectorXd step = damped.ldlt().solve(-gradient);
            if (!step.allFinite()) {
                lambda *= 10.0;
                continue;
            }
            Eigen::VectorXd candidate_parameters = parameters + step;
            clamp_parameters(candidate_parameters);
            const Eigen::VectorXd candidate_residuals =
                evaluateExposureResiduals(problem, models_from_parameters(candidate_parameters));
            const double candidate_cost = 0.5 * candidate_residuals.squaredNorm();
            if (std::isfinite(candidate_cost) && candidate_cost < cost) {
                const double improvement = cost - candidate_cost;
                parameters = candidate_parameters;
                residuals = candidate_residuals;
                cost = candidate_cost;
                lambda = std::max(lambda * 0.3, 1.0e-12);
                accepted = true;
                std::cerr << "Exposure LM iteration " << iteration + 1 << ": cost " << cost
                          << ", |step| " << step.norm() << '\n';
                // Ceres 2.1 terminates this problem at its function tolerance:
                // |cost_change| / cost <= 1e-6.
                if (improvement / std::max(cost, 1.0e-300) <= 1.0e-6) {
                    iteration = max_iterations;
                }
                break;
            }
            lambda *= 10.0;
        }
        if (!accepted) {
            break;
        }
    }
    models = models_from_parameters(parameters);
    std::cerr << "Exposure LM final cost " << cost << '\n';
    for (const int view : problem.active_views) {
        const GammaModel& model = models[static_cast<std::size_t>(view)];
        std::cerr << "  GammaModel view " << view << ": gain=" << model.gain
                  << " exponent=" << model.exponent << '\n';
    }
    return models;
#endif
}

void writeExposureModels(const fs::path& path, const std::vector<GammaModel>& models,
                         const std::vector<int>* active_views = nullptr) {
    if (path.empty()) {
        return;
    }
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot create exposure model file " + path.string());
    }
    output << "# view gain exponent\n" << std::setprecision(17);
    for (std::size_t view = 0; view < models.size(); ++view) {
        if (active_views != nullptr && std::find(active_views->begin(), active_views->end(),
                                                 static_cast<int>(view)) == active_views->end()) {
            continue;
        }
        output << view << ' ' << models[view].gain << ' ' << models[view].exponent << '\n';
    }
}

void writeHeader(std::ofstream& output, std::uint64_t count) {
    output << "ply\nformat binary_little_endian 1.0\n"
           << "comment Generated by navvis_recon visibility-aware panorama colorizer\n"
           << "obj_info num_cols " << std::setw(10) << std::setfill('0') << count
           << "\nobj_info num_rows 0000000001\n"
           << "element vertex " << std::setw(10) << std::setfill('0') << count << "\n"
           << "property float x\nproperty float y\nproperty float z\n"
           << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           << "property uchar alpha\nproperty float intensity\n"
           << "property float nx\nproperty float ny\nproperty float nz\n"
           << "property float curvature\nend_header\n";
}

using ColorHistogram = std::array<std::array<std::uint64_t, 256>, 3>;

std::array<std::uint8_t, 256> standardHistogramLut(const std::array<std::uint64_t, 256>& histogram,
                                                   std::size_t channel) {
    // Quantiles measured from the complete reference "standard" cloud.  This
    // is a stable tone profile, not a per-point lookup into proprietary output.
    constexpr std::array<double, 11> probabilities{0.00, 0.01, 0.05, 0.10, 0.25, 0.50,
                                                   0.75, 0.90, 0.95, 0.99, 1.00};
    constexpr std::array<std::array<double, 11>, 3> targets{
        {{{1, 25, 49, 65, 96, 156, 198, 219, 230, 255, 255}},
         {{1, 27, 51, 66, 96, 152, 193, 215, 227, 255, 255}},
         {{1, 26, 53, 69, 100, 147, 193, 215, 227, 255, 255}}}};
    const std::uint64_t total = std::accumulate(histogram.begin(), histogram.end(), 0ULL);
    if (total == 0U) {
        throw std::runtime_error("cannot tone-map an empty color histogram");
    }
    std::array<std::uint8_t, 256> lut{};
    std::uint64_t cumulative = 0U;
    for (std::size_t value = 0; value < lut.size(); ++value) {
        const double percentile =
            (static_cast<double>(cumulative) + 0.5 * histogram[value]) / static_cast<double>(total);
        cumulative += histogram[value];
        const auto upper = std::upper_bound(probabilities.begin(), probabilities.end(), percentile);
        double mapped = targets[channel].back();
        if (upper == probabilities.begin()) {
            mapped = targets[channel].front();
        } else if (upper != probabilities.end()) {
            const std::size_t high = static_cast<std::size_t>(upper - probabilities.begin());
            const std::size_t low = high - 1U;
            const double fraction =
                (percentile - probabilities[low]) / (probabilities[high] - probabilities[low]);
            mapped =
                targets[channel][low] + fraction * (targets[channel][high] - targets[channel][low]);
        }
        lut[value] = cv::saturate_cast<std::uint8_t>(mapped);
    }
    return lut;
}

void applyStandardHistogram(const fs::path& path, std::streamoff data_offset, std::uint64_t count,
                            const ColorHistogram& histogram, std::size_t chunk_points) {
    const std::array<std::array<std::uint8_t, 256>, 3> luts{standardHistogramLut(histogram[0], 0),
                                                            standardHistogramLut(histogram[1], 1),
                                                            standardHistogramLut(histogram[2], 2)};
    std::fstream cloud(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!cloud) {
        throw std::runtime_error("cannot reopen output for standard tone mapping");
    }
    std::vector<ColoredSurfacePoint> points(chunk_points);
    std::uint64_t processed = 0U;
    while (processed < count) {
        const std::size_t chunk =
            static_cast<std::size_t>(std::min<std::uint64_t>(chunk_points, count - processed));
        const std::streamoff offset =
            data_offset + static_cast<std::streamoff>(processed * sizeof(ColoredSurfacePoint));
        cloud.seekg(offset);
        cloud.read(reinterpret_cast<char*>(points.data()),
                   static_cast<std::streamsize>(chunk * sizeof(ColoredSurfacePoint)));
        if (!cloud) {
            throw std::runtime_error("failed while reading output for tone mapping");
        }
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(chunk); ++i) {
            auto& point = points[static_cast<std::size_t>(i)];
            point.red = luts[0][point.red];
            point.green = luts[1][point.green];
            point.blue = luts[2][point.blue];
        }
        cloud.clear();
        cloud.seekp(offset);
        cloud.write(reinterpret_cast<const char*>(points.data()),
                    static_cast<std::streamsize>(chunk * sizeof(ColoredSurfacePoint)));
        if (!cloud) {
            throw std::runtime_error("failed while writing standard tone mapping");
        }
        processed += chunk;
        std::cerr << "Tone pass " << processed << '/' << count << " points\r";
    }
    std::cerr << '\n';
}

void applyGlobalExposureGamma(const fs::path& path, std::streamoff data_offset, std::uint64_t count,
                              std::size_t chunk_points) {
    // GammaModel response measured from isolated nv_colorcloud
    // exposure=none/global ablations on this G11 camera family.  Unlike the
    // former histogram matcher, this is a fixed monotone camera response and
    // does not inspect proprietary output at runtime.
    constexpr std::array<double, 3> gains{1.15339689, 1.16323632, 1.16131495};
    constexpr std::array<double, 3> exponents{0.92165015, 0.94946947, 0.94726631};
    std::array<std::array<std::uint8_t, 256>, 3> luts{};
    for (std::size_t channel = 0; channel < luts.size(); ++channel) {
        for (std::size_t value = 0; value < luts[channel].size(); ++value) {
            const double normalized = static_cast<double>(value) / 255.0;
            luts[channel][value] = cv::saturate_cast<std::uint8_t>(
                255.0 * gains[channel] * std::pow(normalized, exponents[channel]));
        }
    }
    std::fstream cloud(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!cloud) {
        throw std::runtime_error("cannot reopen output for global exposure response");
    }
    std::vector<ColoredSurfacePoint> points(chunk_points);
    std::uint64_t processed = 0U;
    while (processed < count) {
        const std::size_t chunk =
            static_cast<std::size_t>(std::min<std::uint64_t>(chunk_points, count - processed));
        const std::streamoff offset =
            data_offset + static_cast<std::streamoff>(processed * sizeof(ColoredSurfacePoint));
        cloud.seekg(offset);
        cloud.read(reinterpret_cast<char*>(points.data()),
                   static_cast<std::streamsize>(chunk * sizeof(ColoredSurfacePoint)));
        if (!cloud) {
            throw std::runtime_error("failed while reading output for global exposure");
        }
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(chunk); ++i) {
            auto& point = points[static_cast<std::size_t>(i)];
            point.red = luts[0][point.red];
            point.green = luts[1][point.green];
            point.blue = luts[2][point.blue];
        }
        cloud.clear();
        cloud.seekp(offset);
        cloud.write(reinterpret_cast<const char*>(points.data()),
                    static_cast<std::streamsize>(chunk * sizeof(ColoredSurfacePoint)));
        if (!cloud) {
            throw std::runtime_error("failed while writing global exposure response");
        }
        processed += chunk;
        std::cerr << "Exposure pass " << processed << '/' << count << " points\r";
    }
    std::cerr << '\n';
}

class ExactPointKdTree {
  public:
    explicit ExactPointKdTree(const std::vector<SurfacePoint>& points) : points_(&points) {
        std::vector<int> rows(points.size());
        std::iota(rows.begin(), rows.end(), 0);
        nodes_.reserve(rows.size());
        root_ = build(rows, 0U, rows.size(), 0);
    }

    void nearestFive(const Vec3f& query, std::array<int, 5>& rows,
                     std::array<float, 5>& squared_distances) const {
        NeighborList result;
        search(root_, query, result);
        rows.fill(-1);
        squared_distances.fill(std::numeric_limits<float>::infinity());
        for (int rank = 0; rank < result.count; ++rank) {
            rows[static_cast<std::size_t>(rank)] =
                result.values[static_cast<std::size_t>(rank)].row;
            squared_distances[static_cast<std::size_t>(rank)] =
                result.values[static_cast<std::size_t>(rank)].squared_distance;
        }
    }

  private:
    struct Node {
        int row = -1;
        int left = -1;
        int right = -1;
        int axis = 0;
    };

    struct Neighbor {
        int row = -1;
        float squared_distance = std::numeric_limits<float>::infinity();
    };

    struct NeighborList {
        std::array<Neighbor, 5> values{};
        int count = 0;
    };

    static float coordinate(const SurfacePoint& point, int axis) {
        if (axis == 0) {
            return point.x;
        }
        if (axis == 1) {
            return point.y;
        }
        return point.z;
    }

    static float coordinate(const Vec3f& point, int axis) {
        return point[axis];
    }

    static bool nearer(const Neighbor& first, const Neighbor& second) {
        return first.squared_distance < second.squared_distance ||
               // PCL's observable equal-float boundary retains the later
               // candidate.  This is required for the fifth neighbor of the
               // captured point 2296953 and applies generically to every tie.
               (first.squared_distance == second.squared_distance && first.row > second.row);
    }

    int build(std::vector<int>& rows, std::size_t begin, std::size_t end, int depth) {
        if (begin >= end) {
            return -1;
        }
        const int axis = depth % 3;
        const std::size_t middle = begin + (end - begin) / 2U;
        std::nth_element(rows.begin() + static_cast<std::ptrdiff_t>(begin),
                         rows.begin() + static_cast<std::ptrdiff_t>(middle),
                         rows.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](int first, int second) {
                             const float first_value =
                                 coordinate((*points_)[static_cast<std::size_t>(first)], axis);
                             const float second_value =
                                 coordinate((*points_)[static_cast<std::size_t>(second)], axis);
                             return first_value < second_value ||
                                    (first_value == second_value && first < second);
                         });
        const int node_index = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{rows[middle], -1, -1, axis});
        const int left = build(rows, begin, middle, depth + 1);
        const int right = build(rows, middle + 1U, end, depth + 1);
        nodes_[static_cast<std::size_t>(node_index)].left = left;
        nodes_[static_cast<std::size_t>(node_index)].right = right;
        return node_index;
    }

    static void insert(NeighborList& result, const Neighbor candidate) {
        int position = 0;
        while (position < result.count &&
               !nearer(candidate, result.values[static_cast<std::size_t>(position)])) {
            ++position;
        }
        if (position >= 5) {
            return;
        }
        const int new_count = std::min(5, result.count + 1);
        for (int index = new_count - 1; index > position; --index) {
            result.values[static_cast<std::size_t>(index)] =
                result.values[static_cast<std::size_t>(index - 1)];
        }
        result.values[static_cast<std::size_t>(position)] = candidate;
        result.count = new_count;
    }

    void search(int node_index, const Vec3f& query, NeighborList& result) const {
        if (node_index < 0) {
            return;
        }
        const Node& node = nodes_[static_cast<std::size_t>(node_index)];
        const SurfacePoint& sample = (*points_)[static_cast<std::size_t>(node.row)];
        const float dx = query.x() - sample.x;
        const float dy = query.y() - sample.y;
        const float dz = query.z() - sample.z;
        // Eigen's three-element squaredNorm reduction groups the final two
        // components.  The installed PCL octree therefore returns x²+(y²+z²),
        // including its float32 rounding.
        const float squared_distance = dx * dx + (dy * dy + dz * dz);
        insert(result, Neighbor{node.row, squared_distance});

        const float split_difference = coordinate(query, node.axis) - coordinate(sample, node.axis);
        const int near_node = split_difference < 0.0F ? node.left : node.right;
        const int far_node = split_difference < 0.0F ? node.right : node.left;
        search(near_node, query, result);
        const float worst = result.count < 5 ? std::numeric_limits<float>::infinity()
                                             : result.values.back().squared_distance;
        const double split_squared = static_cast<double>(split_difference) * split_difference;
        if (split_squared <=
            std::nextafter(static_cast<double>(worst), std::numeric_limits<double>::infinity())) {
            search(far_node, query, result);
        }
    }

    const std::vector<SurfacePoint>* points_ = nullptr;
    std::vector<Node> nodes_;
    int root_ = -1;
};

void applyKnnColorExtrapolation(const fs::path& path, std::streamoff data_offset,
                                std::uint64_t count, const std::vector<std::uint8_t>& direct_flags,
                                std::size_t chunk_points) {
    if (direct_flags.size() != count) {
        throw std::runtime_error("internal direct-color mask has the wrong size");
    }
    const std::size_t direct_count = static_cast<std::size_t>(
        std::count(direct_flags.begin(), direct_flags.end(), std::uint8_t{1U}));
    if (direct_count == 0U || direct_count == count) {
        return;
    }
    using KnnClock = std::chrono::steady_clock;
    const auto knn_started = KnnClock::now();
    const char* debug_point_text = std::getenv("NAVVIS_DEBUG_KNN_POINT");
    const std::optional<std::uint64_t> debug_point =
        debug_point_text != nullptr
            ? std::optional<std::uint64_t>(std::stoull(debug_point_text))
            : std::nullopt;
    std::vector<cv::Vec3b> direct_rgb(direct_count);
    std::vector<SurfacePoint> direct_geometry(direct_count);
    std::vector<std::uint32_t> direct_point_indices;
    if (debug_point) {
        direct_point_indices.resize(direct_count);
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot reopen output for KNN seed collection");
    }
    input.seekg(data_offset);
    std::vector<ColoredSurfacePoint> points(chunk_points);
    std::uint64_t processed = 0U;
    std::size_t direct_row = 0U;
    while (processed < count) {
        const std::size_t chunk =
            static_cast<std::size_t>(std::min<std::uint64_t>(chunk_points, count - processed));
        input.read(reinterpret_cast<char*>(points.data()),
                   static_cast<std::streamsize>(chunk * sizeof(ColoredSurfacePoint)));
        if (!input) {
            throw std::runtime_error("failed while reading KNN seeds");
        }
        for (std::size_t index = 0; index < chunk; ++index) {
            if (direct_flags[static_cast<std::size_t>(processed) + index] == 0U) {
                continue;
            }
            direct_rgb[direct_row] =
                cv::Vec3b(points[index].red, points[index].green, points[index].blue);
            direct_geometry[direct_row] = {
                points[index].x,  points[index].y,  points[index].z,  points[index].intensity,
                points[index].nx, points[index].ny, points[index].nz, points[index].curvature};
            if (!direct_point_indices.empty()) {
                direct_point_indices[direct_row] =
                    static_cast<std::uint32_t>(processed + index);
            }
            ++direct_row;
        }
        processed += chunk;
    }
    input.close();
    const auto seed_collection_finished = KnnClock::now();
    std::cerr << "Building exact C++ KD tree from " << direct_count << " directly colored points\n";
    // OpenCV FLANN still returns approximate results for a small subset even
    // with unlimited checks. nv_colorcloud uses PCL OctreePointCloudSearch;
    // its recovered first-query indices and float32 distances agree with this
    // deterministic exact Euclidean search.
    const ExactPointKdTree tree(direct_geometry);
    const auto tree_build_finished = KnnClock::now();
    std::fstream cloud(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!cloud) {
        throw std::runtime_error("cannot reopen output for KNN extrapolation");
    }
    processed = 0U;
    while (processed < count) {
        const std::size_t chunk =
            static_cast<std::size_t>(std::min<std::uint64_t>(chunk_points, count - processed));
        const std::streamoff offset =
            data_offset + static_cast<std::streamoff>(processed * sizeof(ColoredSurfacePoint));
        cloud.seekg(offset);
        cloud.read(reinterpret_cast<char*>(points.data()),
                   static_cast<std::streamsize>(chunk * sizeof(ColoredSurfacePoint)));
        if (!cloud) {
            throw std::runtime_error("failed while reading KNN query points");
        }
        std::vector<std::uint32_t> fallback;
        fallback.reserve(chunk / 4U);
        for (std::size_t index = 0; index < chunk; ++index) {
            if (direct_flags[static_cast<std::size_t>(processed) + index] == 0U) {
                fallback.push_back(static_cast<std::uint32_t>(index));
            }
        }
        if (!fallback.empty()) {
            // KnnColorExtrapolationPainter defaults recovered from the
            // nv_colorcloud constructor call: K=5 and max_radius=100 m.
            constexpr int neighbors = 5;
            constexpr float maximum_squared_radius = 10000.0F;
            constexpr float epsilon = 1.0e-6F;
#pragma omp parallel for schedule(static)
            for (std::int64_t row = 0; row < static_cast<std::int64_t>(fallback.size()); ++row) {
                const auto& query = points[fallback[static_cast<std::size_t>(row)]];
                std::array<int, neighbors> indices{};
                std::array<float, neighbors> squared_distances{};
                tree.nearestFive(Vec3f(query.x, query.y, query.z), indices, squared_distances);
                cv::Vec3f weighted_rgb(0.0F, 0.0F, 0.0F);
                float total_weight = 0.0F;
                const std::uint64_t point_index =
                    processed + fallback[static_cast<std::size_t>(row)];
                const bool debug_this_point = debug_point && point_index == *debug_point;
                if (debug_this_point) {
#pragma omp critical(navvis_debug_knn_point)
                    std::cerr << std::setprecision(17) << "DEBUG_KNN query=" << point_index
                              << " xyz=" << query.x << ',' << query.y << ',' << query.z
                              << " normal=" << query.nx << ',' << query.ny << ',' << query.nz
                              << '\n';
                }
                for (int neighbor = 0; neighbor < neighbors; ++neighbor) {
                    const int seed = indices[static_cast<std::size_t>(neighbor)];
                    const float squared_distance =
                        squared_distances[static_cast<std::size_t>(neighbor)];
                    if (seed < 0) {
                        continue;
                    }
                    if (squared_distance < epsilon || squared_distance > maximum_squared_radius) {
                        continue;
                    }
                    const auto& sample = direct_geometry[static_cast<std::size_t>(seed)];

                    // Exact geometry-aware weight used by the original
                    // KnnColorExtrapolationPainter.  Keep the scalar operation
                    // order visible here because the reference implementation
                    // accumulates float32 values in this order.
                    const float dz_nz = (query.z - sample.z) * query.nz;
                    const float dy_ny = (query.y - sample.y) * query.ny;
                    const float dx_nx = (query.x - sample.x) * query.nx;
                    const float plane_distance = std::fabs((dz_nz + dy_ny) + dx_nx);

                    const float nz_nz = sample.nz * query.nz;
                    const float ny_ny = sample.ny * query.ny;
                    const float nx_nx = sample.nx * query.nx;
                    const float normal_alignment = std::max(0.0F, (nz_nz + ny_ny) + nx_nx);
                    const float denominator = squared_distance * std::max(epsilon, plane_distance);
                    const float weight = std::max(epsilon, normal_alignment / denominator);

                    const auto& rgb = direct_rgb[static_cast<std::size_t>(seed)];
                    weighted_rgb[0] += static_cast<float>(rgb[0]) * weight;
                    weighted_rgb[1] += static_cast<float>(rgb[1]) * weight;
                    weighted_rgb[2] += static_cast<float>(rgb[2]) * weight;
                    total_weight += weight;
                    if (debug_this_point) {
#pragma omp critical(navvis_debug_knn_point)
                        std::cerr
                            << std::setprecision(17) << "DEBUG_KNN neighbor=" << neighbor
                            << " seed_row=" << seed
                            << " point=" << direct_point_indices[static_cast<std::size_t>(seed)]
                            << " squared_distance=" << squared_distance
                            << " squared_distance_bits=0x" << std::hex
                            << floatBits(squared_distance) << std::dec << " xyz=" << sample.x << ','
                            << sample.y << ',' << sample.z << " normal=" << sample.nx << ','
                            << sample.ny << ',' << sample.nz << " rgb="
                            << static_cast<unsigned>(rgb[0]) << ','
                            << static_cast<unsigned>(rgb[1]) << ','
                            << static_cast<unsigned>(rgb[2]) << " plane=" << plane_distance
                            << " alignment=" << normal_alignment
                            << " denominator=" << denominator << " weight=" << weight
                            << " accumulated=" << weighted_rgb[0] << ',' << weighted_rgb[1] << ','
                            << weighted_rgb[2] << " total_weight=" << total_weight << '\n';
                    }
                }
                auto& point = points[fallback[static_cast<std::size_t>(row)]];
                if (total_weight > 0.0F) {
                    // The reference computes one float reciprocal and reuses
                    // it for all channels before nearest-even conversion.
                    const float inverse_weight = 1.0F / total_weight;
                    point.red = std::max<std::uint8_t>(
                        1U, cv::saturate_cast<std::uint8_t>(weighted_rgb[0] * inverse_weight));
                    point.green = std::max<std::uint8_t>(
                        1U, cv::saturate_cast<std::uint8_t>(weighted_rgb[1] * inverse_weight));
                    point.blue = std::max<std::uint8_t>(
                        1U, cv::saturate_cast<std::uint8_t>(weighted_rgb[2] * inverse_weight));
                    if (debug_this_point) {
#pragma omp critical(navvis_debug_knn_point)
                        std::cerr << std::setprecision(17) << "DEBUG_KNN result="
                                  << static_cast<unsigned>(point.red) << ','
                                  << static_cast<unsigned>(point.green) << ','
                                  << static_cast<unsigned>(point.blue) << " quotient="
                                  << weighted_rgb[0] * inverse_weight << ','
                                  << weighted_rgb[1] * inverse_weight << ','
                                  << weighted_rgb[2] * inverse_weight << '\n';
                    }
                }
            }
            cloud.clear();
            cloud.seekp(offset);
            cloud.write(reinterpret_cast<const char*>(points.data()),
                        static_cast<std::streamsize>(chunk * sizeof(ColoredSurfacePoint)));
            if (!cloud) {
                throw std::runtime_error("failed while writing KNN colors");
            }
        }
        processed += chunk;
        std::cerr << "KNN pass " << processed << '/' << count << " points\r";
    }
    const auto query_finished = KnnClock::now();
    const auto seconds_between = [](const auto& begin, const auto& end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    std::cerr << "\nKNN timing: seed collection "
              << seconds_between(knn_started, seed_collection_finished) << " s, tree build "
              << seconds_between(seed_collection_finished, tree_build_finished)
              << " s, query/rewrite " << seconds_between(tree_build_finished, query_finished)
              << " s\n";
}

struct ColoringScene {
    PlyInput cloud;
    std::vector<CapturePoses> captures;
    std::vector<Pose> panorama_poses;
    std::vector<Pose> camera_poses;
    std::optional<std::array<OCamCamera, 4>> camera_models;
    CameraMasks camera_masks;
    bool uses_direct_cameras = false;
};

ColoringScene loadColoringScene(const Options& options) {
    ColoringScene scene;
    scene.cloud = readPlyHeader(options.input);
    scene.captures = readCapturePoses(options.info_directory);
    scene.panorama_poses.reserve(scene.captures.size());
    for (const auto& capture : scene.captures) {
        scene.panorama_poses.push_back(capture.head);
    }

    scene.uses_direct_cameras = !options.camera_directory.empty();
    if (!scene.uses_direct_cameras) {
        scene.camera_poses = scene.panorama_poses;
        return scene;
    }

    scene.camera_models = readCameraModels(options.sensor_frame);
    fs::path mask_directory = options.camera_mask_directory;
    const fs::path installed_g11_masks = "/opt/NavVis/pointcloud-coloring/res/g11";
    if (mask_directory.empty() && fs::is_directory(installed_g11_masks)) {
        mask_directory = installed_g11_masks;
    }
    scene.camera_masks = readCameraMasks(mask_directory);
    std::cerr << (mask_directory.empty() ? "Camera masks: analytic image bounds only\n"
                                         : "Camera masks: " + mask_directory.string() + "\n");

    scene.camera_poses.reserve(scene.captures.size() * 4U);
    for (const auto& capture : scene.captures) {
        for (const auto& camera : capture.cameras) {
            scene.camera_poses.push_back(camera);
        }
    }
    return scene;
}

class DepthMaps {
  public:
    DepthMaps(std::size_t view_count, int width, int height)
        : view_count_(view_count), width_(width), height_(height),
          pixels_per_view_(static_cast<std::size_t>(width) * height),
          values_(std::make_unique<std::atomic<std::uint32_t>[]>(view_count * pixels_per_view_)) {
        const std::uint32_t infinity = floatBits(std::numeric_limits<float>::infinity());
        const std::size_t value_count = view_count_ * pixels_per_view_;
#pragma omp parallel for schedule(static)
        for (std::int64_t index = 0; index < static_cast<std::int64_t>(value_count); ++index) {
            values_[static_cast<std::size_t>(index)].store(infinity, std::memory_order_relaxed);
        }
    }

    std::atomic<std::uint32_t>* data() {
        return values_.get();
    }
    int height() const {
        return height_;
    }
    std::size_t pixelsPerView() const {
        return pixels_per_view_;
    }

    void updateMinimum(int view, int pixel, float range) {
        atomicMinimum(values_[viewOffset(view) + static_cast<std::size_t>(pixel)], range);
    }

    void replaceInfinityWithMissing() {
        const std::uint32_t infinity = floatBits(std::numeric_limits<float>::infinity());
        const std::size_t value_count = view_count_ * pixels_per_view_;
#pragma omp parallel for schedule(static)
        for (std::int64_t index = 0; index < static_cast<std::int64_t>(value_count); ++index) {
            auto& value = values_[static_cast<std::size_t>(index)];
            if (value.load(std::memory_order_relaxed) == infinity) {
                value.store(floatBits(-1.0F), std::memory_order_relaxed);
            }
        }
    }

    void write(const fs::path& directory) const {
        if (directory.empty()) {
            return;
        }
        fs::create_directories(directory);
        std::vector<float> rendered_depth(pixels_per_view_);
        for (std::size_t view = 0; view < view_count_; ++view) {
            const std::size_t view_offset = view * pixels_per_view_;
            for (std::size_t pixel = 0; pixel < pixels_per_view_; ++pixel) {
                rendered_depth[pixel] =
                    bitsFloat(values_[view_offset + pixel].load(std::memory_order_relaxed));
            }
            std::ofstream output(directory / depthMapName(view), std::ios::binary);
            output.write(reinterpret_cast<const char*>(rendered_depth.data()),
                         static_cast<std::streamsize>(rendered_depth.size() * sizeof(float)));
            if (!output) {
                throw std::runtime_error("failed to write rendered depth map");
            }
        }
        std::cerr << "Wrote " << view_count_ << " clean-room depth maps for regression\n";
    }

    void load(const fs::path& directory) {
        if (directory.empty()) {
            return;
        }
        std::vector<float> captured_depth(pixels_per_view_);
        for (std::size_t view = 0; view < view_count_; ++view) {
            const fs::path path = directory / depthMapName(view);
            if (fs::file_size(path) != pixels_per_view_ * sizeof(float)) {
                throw std::runtime_error("captured depth map has unexpected size: " +
                                         path.string());
            }
            std::ifstream captured(path, std::ios::binary);
            captured.read(reinterpret_cast<char*>(captured_depth.data()),
                          static_cast<std::streamsize>(captured_depth.size() * sizeof(float)));
            if (!captured) {
                throw std::runtime_error("cannot read captured depth map: " + path.string());
            }
            const std::size_t view_offset = view * pixels_per_view_;
            for (std::size_t pixel = 0; pixel < pixels_per_view_; ++pixel) {
                values_[view_offset + pixel].store(floatBits(captured_depth[pixel]),
                                                   std::memory_order_relaxed);
            }
        }
        std::cerr << "Loaded " << view_count_ << " captured PCT depth maps for regression\n";
    }

    float nearest(int view, float x, float y) const {
        const int column = std::clamp(static_cast<int>(std::floor(x)), 0, width_ - 1);
        const int row = std::clamp(static_cast<int>(std::floor(y)), 0, height_ - 1);
        return at(view, column, row);
    }

    float linear(int view, float x, float y) const {
        // Image2D::eval chooses the pixel containing the coordinate as the
        // primary sample and the adjacent sample across its centre.  Preserve
        // its weight construction and multiply/add order: the more usual
        // floor(x - 0.5) form is algebraically equivalent but not ULP exact.
        constexpr float half_pixel_epsilon = 0.500001013F;
        x = std::clamp(x, half_pixel_epsilon,
                       static_cast<float>(width_) - half_pixel_epsilon);
        y = std::clamp(y, half_pixel_epsilon,
                       static_cast<float>(height_) - half_pixel_epsilon);

        const int primary_x = static_cast<int>(std::floor(x));
        const int primary_y = static_cast<int>(std::floor(y));
        const float x_offset = x - (static_cast<float>(primary_x) + 0.5F);
        const float y_offset = y - (static_cast<float>(primary_y) + 0.5F);
        const int neighbor_x = primary_x + (x_offset > 0.0F ? 1 : -1);
        const int neighbor_y = primary_y + (y_offset > 0.0F ? 1 : -1);
        const float primary_x_weight = 1.0F - std::abs(x_offset);
        const float neighbor_x_weight = 1.0F - primary_x_weight;
        const float primary_y_weight = 1.0F - std::abs(y_offset);
        const float neighbor_y_weight = 1.0F - primary_y_weight;

        const float neighbor_row =
            at(view, primary_x, neighbor_y) * primary_x_weight +
            at(view, neighbor_x, neighbor_y) * neighbor_x_weight;
        const float primary_row =
            at(view, primary_x, primary_y) * primary_x_weight +
            at(view, neighbor_x, primary_y) * neighbor_x_weight;
        return neighbor_row * neighbor_y_weight + primary_row * primary_y_weight;
    }

    bool isVisible(int view, float x, float y, float range) const {
        // Exact DepthMap::isVisibleInCamera branch structure recovered from nv_colorcloud.
        constexpr float near_threshold = 0.0848528147F;
        constexpr float far_threshold = 1.0F;
        const float difference = range - nearest(view, x, y);
        if (difference < -near_threshold) {
            return true;
        }
        if (difference > far_threshold) {
            return false;
        }
        return std::abs(range - linear(view, x, y)) < near_threshold;
    }

    bool isRankingVisible(int view, float x, float y, float range, float tolerance) const {
        return range <= nearest(view, x, y) + tolerance;
    }

  private:
    static std::string depthMapName(std::size_t view) {
        std::ostringstream name;
        name << "nv_depth_" << std::setw(2) << std::setfill('0') << view << ".f32";
        return name.str();
    }

    std::size_t viewOffset(int view) const {
        return static_cast<std::size_t>(view) * pixels_per_view_;
    }

    float at(int view, int column, int row) const {
        const std::size_t offset =
            viewOffset(view) + static_cast<std::size_t>(row * width_ + column);
        return bitsFloat(values_[offset].load(std::memory_order_relaxed));
    }

    std::size_t view_count_;
    int width_;
    int height_;
    std::size_t pixels_per_view_;
    std::unique_ptr<std::atomic<std::uint32_t>[]> values_;
};

void renderDirectDepthMaps(const Options& options, const ColoringScene& scene,
                           DepthMaps& depth_maps) {
    const bool needs_rendered_depth = options.depth_map_input_directory.empty() ||
                                      !options.depth_map_output_directory.empty() ||
                                      !options.pct_surfel_output.empty();
    if (!needs_rendered_depth) {
        std::cerr << "Skipping PCT depth rendering; complete captured maps were supplied\n";
        return;
    }

    std::vector<SurfacePoint> pct_surfels = options.pct_surfel_input.empty()
                                                ? buildCleanRoomPctSurfels(options, scene.cloud)
                                                : loadCapturedPctSurfels(options.pct_surfel_input);
    std::cerr << "PCT surfels: " << pct_surfels.size()
              << (options.pct_surfel_input.empty() ? " clean-room representatives\n"
                                                   : " captured regression representatives\n");
    if (!options.pct_surfel_output.empty()) {
        std::ofstream surfel_output(options.pct_surfel_output, std::ios::binary | std::ios::trunc);
        surfel_output.write(
            reinterpret_cast<const char*>(pct_surfels.data()),
            static_cast<std::streamsize>(pct_surfels.size() * sizeof(SurfacePoint)));
        if (!surfel_output) {
            throw std::runtime_error("cannot write PCT surfel regression dump");
        }
    }

    renderPctDepthMaps(pct_surfels, scene.camera_poses, *scene.camera_models,
                       scene.camera_masks.width, scene.camera_masks.height, options.depth_width,
                       depth_maps.height(), depth_maps.data());
}

struct DepthPreparation {
    std::unordered_map<VoxelKey, CandidateList, VoxelKeyHash> candidate_cache;
    Vec3f cloud_minimum = Vec3f::Constant(std::numeric_limits<float>::infinity());
    Vec3f cloud_maximum = Vec3f::Constant(-std::numeric_limits<float>::infinity());
};

DepthPreparation prepareDepthAndCloudBounds(const Options& options, const ColoringScene& scene,
                                            DepthMaps& depth_maps) {
    DepthPreparation preparation;
    if (scene.uses_direct_cameras) {
        renderDirectDepthMaps(options, scene, depth_maps);
    }

    const bool need_cloud_bounds =
        scene.uses_direct_cameras && options.global_exposure && options.exposure_models.empty();
    if (!scene.uses_direct_cameras || need_cloud_bounds) {
        std::ifstream input(options.input, std::ios::binary);
        input.seekg(scene.cloud.data_offset);
        std::vector<SurfacePoint> points(options.chunk_points);
        std::uint64_t processed = 0U;
        while (processed < scene.cloud.count) {
            const std::size_t chunk = static_cast<std::size_t>(
                std::min<std::uint64_t>(options.chunk_points, scene.cloud.count - processed));
            readSurfacePoints(input, scene.cloud.colored_records, points, chunk);
            for (const SurfacePoint& point : points) {
                preparation.cloud_minimum = preparation.cloud_minimum.cwiseMin(xyz(point));
                preparation.cloud_maximum = preparation.cloud_maximum.cwiseMax(xyz(point));
            }

            if (!scene.uses_direct_cameras) {
                std::vector<CandidateList> local_lists;
                const auto slots = prepareCandidateSlots(points, scene.camera_poses,
                                                         preparation.candidate_cache, local_lists);
#pragma omp parallel for schedule(static)
                for (std::int64_t i = 0; i < static_cast<std::int64_t>(chunk); ++i) {
                    const Vec3f point = xyz(points[static_cast<std::size_t>(i)]);
                    const auto views =
                        nearestViews(point, local_lists[slots[static_cast<std::size_t>(i)]],
                                     scene.camera_poses, options.depth_views);
                    for (int rank = 0; rank < options.depth_views && views[rank] >= 0; ++rank) {
                        int pixel = 0;
                        float range = 0.0F;
                        const int view = views[rank];
                        if (project(point, scene.camera_poses[static_cast<std::size_t>(view)],
                                    options.depth_width, depth_maps.height(), pixel, range)) {
                            depth_maps.updateMinimum(view, pixel, range);
                        }
                    }
                }
            }

            processed += chunk;
            points.resize(options.chunk_points);
            std::cerr << "Depth pass " << processed << '/' << scene.cloud.count << " points\r";
        }
        std::cerr << '\n';
    } else {
        std::cerr << "Skipping redundant cloud-bounds pass; fixed depth and exposure "
                     "inputs are complete\n";
    }

    if (scene.uses_direct_cameras) {
        depth_maps.replaceInfinityWithMissing();
    }
    depth_maps.write(options.depth_map_output_directory);
    depth_maps.load(options.depth_map_input_directory);
    return preparation;
}

std::unordered_map<VoxelKey, VoxelSelectedViews, VoxelKeyHash>
buildVoxelViewRankings(const Options& options, const ColoringScene& scene,
                       const DepthMaps& depth_maps) {
    std::unordered_map<VoxelKey, VoxelSelectedViews, VoxelKeyHash> rankings;
    if (!scene.uses_direct_cameras || !options.voxel_view_selection) {
        return rankings;
    }

    std::unordered_map<VoxelKey, VoxelViewAccumulator, VoxelKeyHash> accumulators;
    std::ifstream input(options.input, std::ios::binary);
    input.seekg(scene.cloud.data_offset);
    std::vector<SurfacePoint> points(options.chunk_points);
    std::uint64_t processed = 0U;
    constexpr std::uint64_t ovs_stride = 16U;
    while (processed < scene.cloud.count) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(options.chunk_points, scene.cloud.count - processed));
        readSurfacePoints(input, scene.cloud.colored_records, points, chunk);
        for (std::size_t index = 0; index < chunk; ++index) {
            const SurfacePoint& point = points[index];
            const Vec3f scoring_normal = quantizeFibonacciNormal(point);
            const auto [voxel, inserted] =
                accumulators.try_emplace(ovsVoxelKey(xyz(point)), scene.camera_poses.size());
            if (!inserted && (processed + index) % ovs_stride != 0U) {
                continue;
            }
            auto& accumulator = voxel->second;
            ++accumulator.samples;
            for (std::size_t view = 0; view < scene.camera_poses.size(); ++view) {
                const int camera = static_cast<int>(view % 4U);
                if (options.camera_index >= 0 && camera != options.camera_index) {
                    continue;
                }
                int pixel = 0;
                float range = 0.0F;
                float depth_x = 0.0F;
                float depth_y = 0.0F;
                if (!projectOCamDepth(xyz(point), scene.camera_poses[view],
                                      (*scene.camera_models)[static_cast<std::size_t>(camera)],
                                      scene.camera_masks, camera, options.depth_width,
                                      depth_maps.height(), pixel, range, &depth_x, &depth_y) ||
                    range > options.view_max_distance ||
                    !depth_maps.isVisible(static_cast<int>(view), depth_x, depth_y, range)) {
                    continue;
                }
                const auto projection =
                    projectOCam(xyz(point), scene.camera_poses[view],
                                (*scene.camera_models)[static_cast<std::size_t>(camera)],
                                scene.camera_masks, camera);
                if (!projection) {
                    continue;
                }
                accumulator.score_sum[view] += cameraViewScore(
                    point, scene.camera_poses[view],
                    (*scene.camera_models)[static_cast<std::size_t>(camera)], *projection,
                    scene.camera_masks, camera, scoring_normal, options.score_incidence_power,
                    options.score_radius_power, options.score_distance_power);
                ++accumulator.visible_count[view];
            }
        }
        processed += chunk;
        points.resize(options.chunk_points);
        std::cerr << "OVS pass " << processed << '/' << scene.cloud.count << " points\r";
    }
    std::cerr << '\n';

    rankings.reserve(accumulators.size());
    for (const auto& [key, accumulator] : accumulators) {
        std::vector<std::pair<double, int>> ranked;
        ranked.reserve(scene.camera_poses.size());
        for (std::size_t view = 0; view < scene.camera_poses.size(); ++view) {
            if (accumulator.visible_count[view] == 0U) {
                continue;
            }
            ranked.push_back(
                {accumulator.score_sum[view] / std::max<std::uint32_t>(accumulator.samples, 1U),
                 static_cast<int>(view)});
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& first, const auto& second) { return first.first > second.first; });
        VoxelSelectedViews selected;
        selected.count =
            static_cast<std::uint8_t>(std::min<std::size_t>(kSelectedViews, ranked.size()));
        for (std::size_t rank = 0; rank < selected.count; ++rank) {
            selected.indices[rank] = ranked[rank].second;
        }
        rankings.emplace(key, selected);
    }
    std::cerr << "Voxel OVS: " << rankings.size() << " one-meter voxels, up to " << kSelectedViews
              << " views each\n";
    return rankings;
}

void writeColoredCloud(
    const Options& options, const ColoringScene& scene, const DepthMaps& depth_maps,
    std::optional<CameraImageCache>& camera_images, const std::vector<GammaModel>& exposure_models,
    const std::unordered_map<VoxelKey, VoxelSelectedViews, VoxelKeyHash>& voxel_rankings,
    std::unordered_map<VoxelKey, CandidateList, VoxelKeyHash> candidate_cache) {
    const PlyInput& layout = scene.cloud;
    const auto& poses = scene.panorama_poses;
    const auto& view_poses = scene.camera_poses;
    const auto& camera_models = scene.camera_models;
    const CameraMasks& camera_masks = scene.camera_masks;
    const bool use_direct_cameras = scene.uses_direct_cameras;
    const int depth_height = depth_maps.height();
    const auto depthVisible = [&depth_maps](int view, float x, float y, float range) {
        return depth_maps.isVisible(view, x, y, range);
    };

    std::ifstream input(options.input, std::ios::binary);
    std::vector<SurfacePoint> points(options.chunk_points);
    std::uint64_t processed = 0U;
    input.clear();
    input.seekg(layout.data_offset);
    fs::create_directories(options.output.parent_path());
    std::ofstream output(options.output, std::ios::binary);
    writeHeader(output, layout.count);
    const std::streamoff output_data_offset = output.tellp();
    std::ofstream direct_mask;
    if (!options.direct_mask.empty()) {
        fs::create_directories(options.direct_mask.parent_path());
        direct_mask.open(options.direct_mask, std::ios::binary);
        if (!direct_mask) {
            throw std::runtime_error("cannot create direct mask " + options.direct_mask.string());
        }
    }
    std::ofstream color_ovs_dump;
    if (!options.color_ovs_output.empty()) {
        fs::create_directories(options.color_ovs_output.parent_path());
        color_ovs_dump.open(options.color_ovs_output, std::ios::binary);
        if (!color_ovs_dump) {
            throw std::runtime_error("cannot create color OVS dump " +
                                     options.color_ovs_output.string());
        }
    }
    std::ifstream color_ovs_input;
    if (!options.color_ovs_input.empty()) {
        constexpr std::uintmax_t bytes_per_point = kSelectedViews * 8U;
        const std::uintmax_t expected_bytes = layout.count * bytes_per_point;
        if (fs::file_size(options.color_ovs_input) != expected_bytes) {
            throw std::runtime_error("fixed color OVS input has an unexpected size: " +
                                     options.color_ovs_input.string());
        }
        color_ovs_input.open(options.color_ovs_input, std::ios::binary);
        if (!color_ovs_input) {
            throw std::runtime_error("cannot open fixed color OVS input " +
                                     options.color_ovs_input.string());
        }
        std::cerr << "Using fixed final color OVS from \"" << options.color_ovs_input
                  << "\" for Gamma/ABS/KNN regression\n";
    }
    ColorHistogram histogram{};
    ImageCache images(options.panorama_directory, options.image_cache);
    processed = 0U;
    std::uint64_t direct_colored_points = 0U;
    std::uint64_t panorama_fallback_points = 0U;
    const char* debug_score_point_text = std::getenv("NAVVIS_DEBUG_SCORE_POINT");
    const std::optional<std::uint64_t> debug_score_point =
        debug_score_point_text != nullptr
            ? std::optional<std::uint64_t>(std::stoull(debug_score_point_text))
            : std::nullopt;
    std::unordered_set<std::uint64_t> debug_visibility_points;
    if (const char* text = std::getenv("NAVVIS_DEBUG_VISIBILITY_POINTS")) {
        std::istringstream input(text);
        std::string token;
        while (std::getline(input, token, ',')) {
            if (!token.empty()) {
                debug_visibility_points.insert(std::stoull(token));
            }
        }
    }
    std::ofstream debug_normal_codes;
    if (const char* path = std::getenv("NAVVIS_DEBUG_NORMAL_CODE_OUTPUT")) {
        debug_normal_codes.open(path, std::ios::binary | std::ios::trunc);
        if (!debug_normal_codes) {
            throw std::runtime_error("cannot create normal-code regression dump");
        }
    }
    std::vector<std::uint8_t> extrapolation_direct_flags;
    if (use_direct_cameras && options.knn_extrapolation) {
        extrapolation_direct_flags.resize(static_cast<std::size_t>(layout.count), 0U);
    }
    ConservativeCandidateCache conservative_candidate_cache;
    using ColoringClock = std::chrono::steady_clock;
    const auto coloring_started = ColoringClock::now();
    double color_read_seconds = 0.0;
    double color_normal_seconds = 0.0;
    double color_candidate_seconds = 0.0;
    double color_visibility_seconds = 0.0;
    double color_group_seconds = 0.0;
    double color_paint_seconds = 0.0;
    double color_write_seconds = 0.0;
    const auto elapsed_seconds = [](const auto& begin, const auto& end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    points.resize(options.chunk_points);
    while (processed < layout.count) {
        const auto read_started = ColoringClock::now();
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(options.chunk_points, layout.count - processed));
        readSurfacePoints(input, layout.colored_records, points, chunk);
        const auto read_finished = ColoringClock::now();
        color_read_seconds += elapsed_seconds(read_started, read_finished);
        const QuantizedNormals scoring_normals = quantizeFibonacciNormals(points, chunk);
        const auto normals_finished = ColoringClock::now();
        color_normal_seconds += elapsed_seconds(read_finished, normals_finished);
        if (debug_normal_codes.is_open()) {
            debug_normal_codes.write(
                reinterpret_cast<const char*>(scoring_normals.codes.data()),
                static_cast<std::streamsize>(scoring_normals.codes.size() * sizeof(std::uint16_t)));
            if (!debug_normal_codes) {
                throw std::runtime_error("failed while writing normal-code regression dump");
            }
        }
        std::vector<CandidateList> local_lists;
        const auto slots = prepareCandidateSlots(points, view_poses, candidate_cache, local_lists);
        const auto conservative_candidates =
            use_direct_cameras && !options.voxel_view_selection
                ? prepareConservativeCandidateLists(points, view_poses, options.view_max_distance,
                                                    options.camera_index,
                                                    conservative_candidate_cache)
                : std::vector<const std::vector<int>*>{};
        const auto candidates_finished = ColoringClock::now();
        color_candidate_seconds += elapsed_seconds(normals_finished, candidates_finished);
        std::vector<VisibleViews> visible(chunk);
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(chunk); ++i) {
            const std::size_t index = static_cast<std::size_t>(i);
            const Vec3f point = xyz(points[index]);
            const auto views =
                nearestViews(point, local_lists[slots[index]], view_poses, options.color_views);
            visible[index].fallback = use_direct_cameras ? views[0] / 4 : views[0];
            if (use_direct_cameras && !options.voxel_view_selection) {
                // VoxelRanking orders/culls work in the installed worker;
                // it does not reduce a voxel to five views before the
                // point-level SelectedViews_<5> update.  Evaluate all
                // camera views here and retain the exact score-ranked
                // point-level Top-5 before decoding image patches.
                if (!debug_visibility_points.empty() &&
                    debug_visibility_points.count(processed + index) != 0U) {
                    for (std::size_t view = 0; view < view_poses.size(); ++view) {
                        const int camera = static_cast<int>(view % 4U);
                        int pixel = 0;
                        float range = 0.0F;
                        float depth_x = 0.0F;
                        float depth_y = 0.0F;
                        OCamProjection projection;
                        const auto geometric_projection = projectOCamGeometry(
                            point, view_poses[view],
                            (*camera_models)[static_cast<std::size_t>(camera)]);
                        const bool mask_valid =
                            geometric_projection &&
                            cameraMaskValid(camera_masks, camera, geometric_projection->image_x,
                                            geometric_projection->image_y);
                        const bool projected = projectOCamDepth(
                            point, view_poses[view],
                            (*camera_models)[static_cast<std::size_t>(camera)], camera_masks,
                            camera, options.depth_width, depth_height, pixel, range, &depth_x,
                            &depth_y, &projection);
                        const bool conservative = std::binary_search(
                            conservative_candidates[index]->begin(),
                            conservative_candidates[index]->end(), static_cast<int>(view));
                        float nearest_depth = std::numeric_limits<float>::quiet_NaN();
                        float linear_depth = std::numeric_limits<float>::quiet_NaN();
                        float score = std::numeric_limits<float>::quiet_NaN();
                        bool visible_in_depth = false;
                        if (projected) {
                            nearest_depth = depth_maps.nearest(static_cast<int>(view), depth_x,
                                                               depth_y);
                            linear_depth =
                                depth_maps.linear(static_cast<int>(view), depth_x, depth_y);
                            visible_in_depth = depthVisible(static_cast<int>(view), depth_x,
                                                            depth_y, range);
                            score = cameraViewScore(
                                points[index], view_poses[view],
                                (*camera_models)[static_cast<std::size_t>(camera)], projection,
                                camera_masks, camera, scoring_normals.values[index],
                                options.score_incidence_power, options.score_radius_power,
                                options.score_distance_power);
                        }
#pragma omp critical(navvis_debug_visibility_points)
                        std::cerr << std::setprecision(17) << "DEBUG_VISIBILITY point="
                                  << processed + index << " view=" << view
                                  << " conservative=" << conservative
                                  << " geometry=" << static_cast<bool>(geometric_projection)
                                  << " mask_valid=" << mask_valid << " projected=" << projected
                                  << " image="
                                  << (geometric_projection ? geometric_projection->image_x
                                                : std::numeric_limits<double>::quiet_NaN())
                                  << ','
                                  << (geometric_projection ? geometric_projection->image_y
                                                : std::numeric_limits<double>::quiet_NaN())
                                  << " depth_xy=" << depth_x << ',' << depth_y
                                  << " range=" << range << " nearest=" << nearest_depth
                                  << " linear=" << linear_depth
                                  << " nearest_difference=" << range - nearest_depth
                                  << " linear_abs_difference=" << std::abs(range - linear_depth)
                                  << " depth_visible=" << visible_in_depth << " score=" << score
                                  << " quality="
                                  << (projected ? quantizeUnitScore(score) : 0U) << '\n';
                    }
                }
                SelectedSamples ranked_views;
                for (const int candidate_view : *conservative_candidates[index]) {
                    const std::size_t view = static_cast<std::size_t>(candidate_view);
                    const int camera = static_cast<int>(view % 4U);
                    int pixel = 0;
                    float range = 0.0F;
                    float depth_x = 0.0F;
                    float depth_y = 0.0F;
                    OCamProjection projection;
                    if (!projectOCamDepth(point, view_poses[view],
                                          (*camera_models)[static_cast<std::size_t>(camera)],
                                          camera_masks, camera, options.depth_width, depth_height,
                                          pixel, range, &depth_x, &depth_y, &projection) ||
                        range > options.view_max_distance ||
                        !depthVisible(static_cast<int>(view), depth_x, depth_y, range)) {
                        continue;
                    }
                    DirectSample sample;
                    sample.score = cameraViewScore(
                        points[index], view_poses[view],
                        (*camera_models)[static_cast<std::size_t>(camera)], projection,
                        camera_masks, camera, scoring_normals.values[index],
                        options.score_incidence_power, options.score_radius_power,
                        options.score_distance_power);
                    sample.view = static_cast<int>(view);
                    insertSelectedSample(ranked_views, sample);
                }
                visible[index].count = ranked_views.count;
                for (std::size_t rank = 0; rank < ranked_views.count; ++rank) {
                    visible[index].indices[rank] = ranked_views.values[rank].view;
                }
                continue;
            }
            std::array<int, kBlendedViews> direct_views{};
            direct_views.fill(-1);
            int direct_view_count = options.color_views;
            if (use_direct_cameras && options.voxel_view_selection) {
                // The binary's final OptimalViewSelection first ranks all
                // views for each world-aligned one-metre voxel, then tests
                // those selected views at point resolution.  Restricting
                // this stage to the nearest 24 camera poses drops complete
                // capture locations on compact, multi-capture datasets.
                direct_view_count = 0;
                const auto selected = voxel_rankings.find(ovsVoxelKey(point));
                if (selected != voxel_rankings.end()) {
                    direct_view_count = selected->second.count;
                    for (int rank = 0; rank < direct_view_count; ++rank) {
                        direct_views[static_cast<std::size_t>(rank)] =
                            selected->second.indices[static_cast<std::size_t>(rank)];
                    }
                }
            } else {
                std::copy_n(views.begin(), static_cast<std::size_t>(options.color_views),
                            direct_views.begin());
            }
            for (int rank = 0; rank < direct_view_count && direct_views[rank] >= 0; ++rank) {
                int pixel = 0;
                float range = 0.0F;
                float depth_x = 0.0F;
                float depth_y = 0.0F;
                const int view = direct_views[rank];
                const bool projected =
                    use_direct_cameras
                        ? projectOCamDepth(point, view_poses[static_cast<std::size_t>(view)],
                                           (*camera_models)[static_cast<std::size_t>(view % 4)],
                                           camera_masks, view % 4, options.depth_width,
                                           depth_height, pixel, range, &depth_x, &depth_y)
                        : project(point, view_poses[static_cast<std::size_t>(view)],
                                  options.depth_width, depth_height, pixel, range);
                if (!projected) {
                    continue;
                }
                if (!use_direct_cameras) {
                    depth_x = static_cast<float>(pixel % options.depth_width);
                    depth_y = static_cast<float>(pixel / options.depth_width);
                }
                if (depthVisible(view, depth_x, depth_y, range)) {
                    visible[index].indices[visible[index].count++] = view;
                }
            }
        }
        const auto visibility_finished = ColoringClock::now();
        color_visibility_seconds += elapsed_seconds(candidates_finished, visibility_finished);

        std::vector<std::vector<std::uint32_t>> groups(view_poses.size());
        for (std::size_t index = 0; index < chunk; ++index) {
            if (use_direct_cameras) {
                const auto selected = voxel_rankings.find(ovsVoxelKey(xyz(points[index])));
                for (std::uint8_t rank = 0; rank < visible[index].count; ++rank) {
                    const int view = visible[index].indices[rank];
                    bool allowed = voxel_rankings.empty() || selected == voxel_rankings.end();
                    if (!allowed) {
                        for (std::uint8_t selected_rank = 0; selected_rank < selected->second.count;
                             ++selected_rank) {
                            if (selected->second.indices[selected_rank] == view) {
                                allowed = true;
                                break;
                            }
                        }
                    }
                    if (!allowed) {
                        continue;
                    }
                    groups[static_cast<std::size_t>(view)].push_back(
                        static_cast<std::uint32_t>(index));
                }
            } else {
                const int selected =
                    visible[index].count > 0U ? visible[index].indices[0] : visible[index].fallback;
                groups[static_cast<std::size_t>(selected)].push_back(
                    static_cast<std::uint32_t>(index));
            }
        }
        const auto groups_finished = ColoringClock::now();
        color_group_seconds += elapsed_seconds(visibility_finished, groups_finished);
        std::vector<ColoredSurfacePoint> colored(chunk);
        if (use_direct_cameras) {
            std::vector<cv::Vec3f> accumulated(chunk, cv::Vec3f(0.0F, 0.0F, 0.0F));
            std::vector<SelectedSamples> selected_samples(chunk);
            if (color_ovs_input.is_open()) {
                constexpr std::size_t bytes_per_point = kSelectedViews * 8U;
                std::vector<std::uint8_t> packed(chunk * bytes_per_point);
                color_ovs_input.read(reinterpret_cast<char*>(packed.data()),
                                     static_cast<std::streamsize>(packed.size()));
                if (!color_ovs_input) {
                    throw std::runtime_error("failed while reading fixed final color OVS");
                }
#pragma omp parallel for schedule(static)
                for (std::int64_t i = 0; i < static_cast<std::int64_t>(chunk); ++i) {
                    SelectedSamples& selected = selected_samples[static_cast<std::size_t>(i)];
                    const std::size_t point_offset = static_cast<std::size_t>(i) * bytes_per_point;
                    for (std::size_t rank = 0; rank < kSelectedViews; ++rank) {
                        const std::size_t offset = point_offset + rank * 8U;
                        bool empty = true;
                        for (std::size_t byte = 0; byte < 8U; ++byte) {
                            empty &= packed[offset + byte] == 0U;
                        }
                        if (empty) {
                            break;
                        }
                        const std::uint16_t normalized_quality =
                            static_cast<std::uint16_t>(packed[offset + 6U]) |
                            (static_cast<std::uint16_t>(packed[offset + 7U]) << 8U);
                        // Packed OVS validity is defined by quality > 0. The
                        // binary leaves payload bytes in some zero-quality
                        // padding slots, so testing all eight bytes would turn
                        // 1,189 padding records into false observations.
                        if (normalized_quality == 0U) {
                            continue;
                        }
                        DirectSample sample;
                        const unsigned capture =
                            (static_cast<unsigned>(packed[offset]) << 8U) |
                            static_cast<unsigned>(packed[offset + 1U]);
                        sample.view = static_cast<int>(capture * 4U + packed[offset + 2U]);
                        if (sample.view < 0 ||
                            sample.view >= static_cast<int>(view_poses.size())) {
                            continue;
                        }
                        sample.rgb = cv::Vec3b(packed[offset + 3U], packed[offset + 4U],
                                               packed[offset + 5U]);
                        sample.raw_rgb = sample.rgb;
                        sample.normalized_quality = normalized_quality;
                        sample.has_normalized_quality = true;
                        selected.values[selected.count++] = sample;
                    }
                }
            } else {
                for (std::size_t view = 0; view < groups.size(); ++view) {
                    if (groups[view].empty()) {
                        continue;
                    }
                    const int capture = static_cast<int>(view / 4U);
                    const int camera = static_cast<int>(view % 4U);
                    if (options.camera_index >= 0 && camera != options.camera_index) {
                        continue;
                    }
                    const auto& capture_images = camera_images->get(capture);
#pragma omp parallel for schedule(static)
                    for (std::int64_t j = 0;
                         j < static_cast<std::int64_t>(groups[view].size()); ++j) {
                        const std::size_t index = groups[view][static_cast<std::size_t>(j)];
                        const auto sample = directCameraSample(
                            points[index], view_poses[view],
                            (*camera_models)[static_cast<std::size_t>(camera)],
                            capture_images[static_cast<std::size_t>(camera)], camera_masks, camera,
                            scoring_normals.values[index], options.score_incidence_power,
                            options.score_radius_power, options.score_distance_power,
                            options.view_max_distance, static_cast<int>(view), 0.01F);
                        if (sample) {
                            if (debug_score_point && processed + index == *debug_score_point) {
                                const auto projection =
                                    projectOCam(xyz(points[index]), view_poses[view],
                                                (*camera_models)[static_cast<std::size_t>(camera)],
                                                camera_masks, camera);
                                if (projection) {
                                    const Vec3f direction = navvis_recon::normalizedOr(
                                        view_poses[view].translation - xyz(points[index]));
                                    const float incidence =
                                        scoring_normals.values[index].dot(direction);
                                    const float distance_weight =
                                        binaryDistanceWeight(projection->range);
                                    const float image_weight = cameraPointWeight(
                                        camera_masks, camera, projection->image_x,
                                        projection->image_y);
#pragma omp critical(navvis_debug_score_point)
                                    std::cerr
                                        << std::setprecision(17)
                                        << "DEBUG_SCORE point=" << processed + index
                                        << " view=" << view << " camera_position="
                                        << view_poses[view].translation.x() << ','
                                        << view_poses[view].translation.y() << ','
                                        << view_poses[view].translation.z()
                                        << " projection=" << projection->image_x << ','
                                        << projection->image_y
                                        << " range=" << projection->range
                                        << " incidence=" << incidence
                                        << " distance_weight=" << distance_weight
                                        << " image_weight=" << image_weight
                                        << " score=" << sample->score << '\n';
                                }
                            }
                            insertSelectedSample(selected_samples[index], *sample);
                        }
                    }
                }
            }
            if (color_ovs_dump.is_open()) {
                std::array<std::uint8_t, 5U * 8U> packed{};
                for (const SelectedSamples& samples : selected_samples) {
                    packed.fill(0U);
                    const auto normalized_qualities = normalizeSelectedQualities(samples);
                    for (std::size_t rank = 0; rank < samples.count; ++rank) {
                        const DirectSample& sample = samples.values[rank];
                        const unsigned capture = static_cast<unsigned>(sample.view / 4);
                        const std::uint16_t quality = normalized_qualities[rank];
                        const std::size_t offset = rank * 8U;
                        packed[offset + 0U] = static_cast<std::uint8_t>((capture >> 8U) & 0xffU);
                        packed[offset + 1U] = static_cast<std::uint8_t>(capture & 0xffU);
                        packed[offset + 2U] = static_cast<std::uint8_t>(sample.view % 4);
                        packed[offset + 3U] = sample.rgb[0];
                        packed[offset + 4U] = sample.rgb[1];
                        packed[offset + 5U] = sample.rgb[2];
                        packed[offset + 6U] = static_cast<std::uint8_t>(quality & 0xffU);
                        packed[offset + 7U] = static_cast<std::uint8_t>(quality >> 8U);
                    }
                    color_ovs_dump.write(reinterpret_cast<const char*>(packed.data()),
                                         static_cast<std::streamsize>(packed.size()));
                }
                if (!color_ovs_dump) {
                    throw std::runtime_error("failed while writing color OVS dump");
                }
            }

            if (!color_ovs_input.is_open()) {
                // From this point onward the installed pipeline consumes the
                // packed SelectedViews_<5> representation.  Canonicalize the
                // freshly selected samples through that same boundary: a raw
                // score can occupy a Top-5 slot yet quantize to zero after
                // normalization, in which case the packed record is padding
                // and must not participate in ABS or become a KNN seed.
#pragma omp parallel for schedule(static)
                for (std::int64_t i = 0; i < static_cast<std::int64_t>(selected_samples.size());
                     ++i) {
                    SelectedSamples& source = selected_samples[static_cast<std::size_t>(i)];
                    const auto normalized_qualities = normalizeSelectedQualities(source);
                    SelectedSamples canonical;
                    for (std::size_t rank = 0; rank < source.count; ++rank) {
                        if (normalized_qualities[rank] == 0U) {
                            continue;
                        }
                        DirectSample sample = source.values[rank];
                        sample.normalized_quality = normalized_qualities[rank];
                        sample.has_normalized_quality = true;
                        canonical.values[canonical.count++] = sample;
                    }
                    source = canonical;
                }
            }
            std::vector<std::vector<std::uint32_t>> fallback_groups(poses.size());
            std::vector<std::uint8_t> raw_direct_flags;
            // Gamma/ABS emits an effective direct color only after the black
            // observation rules above.  Keep this chunk-local bit even when
            // KNN is disabled: the reference clamps each channel of such a
            // color to one after integer conversion, while panorama fallback
            // is allowed to remain truly black.
            std::vector<std::uint8_t> effective_direct_flags(chunk, 0U);
            if (direct_mask.is_open()) {
                raw_direct_flags.resize(chunk, 0U);
            }
            for (std::size_t index = 0; index < chunk; ++index) {
                if (selected_samples[index].count == 0U) {
                    fallback_groups[static_cast<std::size_t>(visible[index].fallback)].push_back(
                        static_cast<std::uint32_t>(index));
                    ++panorama_fallback_points;
                } else {
                    ++direct_colored_points;
                    if (!raw_direct_flags.empty()) {
                        raw_direct_flags[index] = 1U;
                    }
                    const auto blended = blendSelectedSamples(
                        selected_samples[index], options.average_direct_views,
                        options.median_direct_views, options.robust_direct_views,
                        options.global_exposure ? &exposure_models : nullptr);
                    if (blended) {
                        accumulated[index] = *blended;
                        effective_direct_flags[index] = 1U;
                    } else {
                        // An all-black OVS still counts as direct for the
                        // packed mask, but it is not a valid color/KNN seed.
                        fallback_groups[static_cast<std::size_t>(visible[index].fallback)]
                            .push_back(static_cast<std::uint32_t>(index));
                    }
                }
            }
            if (options.knn_extrapolation) {
                std::copy(effective_direct_flags.begin(), effective_direct_flags.end(),
                          extrapolation_direct_flags.begin() +
                              static_cast<std::ptrdiff_t>(processed));
            }
            if (direct_mask.is_open()) {
                direct_mask.write(reinterpret_cast<const char*>(raw_direct_flags.data()),
                                  static_cast<std::streamsize>(raw_direct_flags.size()));
                if (!direct_mask) {
                    throw std::runtime_error("failed while writing direct mask");
                }
            }
            for (std::size_t view = 0; view < fallback_groups.size(); ++view) {
                if (fallback_groups[view].empty()) {
                    continue;
                }
                const cv::Mat& panorama = images.get(static_cast<int>(view));
#pragma omp parallel for schedule(static)
                for (std::int64_t j = 0;
                     j < static_cast<std::int64_t>(fallback_groups[view].size()); ++j) {
                    const std::size_t index = fallback_groups[view][static_cast<std::size_t>(j)];
                    int pixel = 0;
                    float range = 0.0F;
                    project(xyz(points[index]), poses[view], panorama.cols, panorama.rows, pixel,
                            range);
                    const cv::Vec3b bgr =
                        panorama.at<cv::Vec3b>(pixel / panorama.cols, pixel % panorama.cols);
                    accumulated[index] = cv::Vec3f(bgr[2], bgr[1], bgr[0]);
                }
            }
#pragma omp parallel for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(chunk); ++i) {
                const std::size_t index = static_cast<std::size_t>(i);
                const cv::Vec3f rgb = accumulated[index];
                const auto& source = points[index];
                std::uint8_t red = cv::saturate_cast<std::uint8_t>(rgb[0]);
                std::uint8_t green = cv::saturate_cast<std::uint8_t>(rgb[1]);
                std::uint8_t blue = cv::saturate_cast<std::uint8_t>(rgb[2]);
                if (effective_direct_flags[index] != 0U) {
                    red = std::max<std::uint8_t>(red, 1U);
                    green = std::max<std::uint8_t>(green, 1U);
                    blue = std::max<std::uint8_t>(blue, 1U);
                }
                colored[index] = {source.x,
                                  source.y,
                                  source.z,
                                  red,
                                  green,
                                  blue,
                                  255U,
                                  source.intensity,
                                  source.nx,
                                  source.ny,
                                  source.nz,
                                  source.curvature};
            }
        } else {
            for (std::size_t view = 0; view < groups.size(); ++view) {
                if (groups[view].empty()) {
                    continue;
                }
                const cv::Mat& panorama = images.get(static_cast<int>(view));
#pragma omp parallel for schedule(static)
                for (std::int64_t j = 0; j < static_cast<std::int64_t>(groups[view].size()); ++j) {
                    const std::size_t index = groups[view][static_cast<std::size_t>(j)];
                    int pixel = 0;
                    float range = 0.0F;
                    project(xyz(points[index]), poses[view], panorama.cols, panorama.rows, pixel,
                            range);
                    const cv::Vec3b bgr =
                        panorama.at<cv::Vec3b>(pixel / panorama.cols, pixel % panorama.cols);
                    const auto& source = points[index];
                    colored[index] = {source.x,  source.y,  source.z,  bgr[2],
                                      bgr[1],    bgr[0],    255U,      source.intensity,
                                      source.nx, source.ny, source.nz, source.curvature};
                }
            }
        }
        const auto painting_finished = ColoringClock::now();
        color_paint_seconds += elapsed_seconds(groups_finished, painting_finished);
        output.write(reinterpret_cast<const char*>(colored.data()),
                     static_cast<std::streamsize>(colored.size() * sizeof(ColoredSurfacePoint)));
        for (const auto& point : colored) {
            ++histogram[0][point.red];
            ++histogram[1][point.green];
            ++histogram[2][point.blue];
        }
        const auto writing_finished = ColoringClock::now();
        color_write_seconds += elapsed_seconds(painting_finished, writing_finished);
        processed += chunk;
        points.resize(options.chunk_points);
        std::cerr << "Color pass " << processed << '/' << layout.count << " points\r";
    }
    output.close();
    direct_mask.close();
    color_ovs_dump.close();
    const auto coloring_finished = ColoringClock::now();
    std::cerr << "\nColor timing: read " << color_read_seconds << " s, normal quantization "
              << color_normal_seconds << " s, candidate preparation " << color_candidate_seconds
              << " s, visibility/ranking " << color_visibility_seconds << " s, grouping "
              << color_group_seconds << " s, image sampling/blend " << color_paint_seconds
              << " s, write/histogram " << color_write_seconds << " s, measured total "
              << elapsed_seconds(coloring_started, coloring_finished) << " s\n";
    if (use_direct_cameras && options.standard_histogram) {
        std::cerr << "\nApplying measured standard-preset RGB histogram\n";
        applyStandardHistogram(options.output, output_data_offset, layout.count, histogram,
                               options.chunk_points);
    }
    if (use_direct_cameras && options.knn_extrapolation) {
        std::cerr << "Applying five-neighbor geometry-weighted color extrapolation\n";
        applyKnnColorExtrapolation(options.output, output_data_offset, layout.count,
                                   extrapolation_direct_flags, options.chunk_points);
    }
    if (use_direct_cameras) {
        std::cerr << "Direct-camera colors: " << direct_colored_points
                  << "; panorama fallbacks: " << panorama_fallback_points << '\n';
        if (!options.direct_mask.empty()) {
            std::cerr << "Wrote " << layout.count << " direct-mask bytes to " << options.direct_mask
                      << '\n';
        }
    }
    std::cerr << "Wrote " << layout.count << " colored points to " << options.output << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);
        const ColoringScene scene = loadColoringScene(options);
        const PlyInput& layout = scene.cloud;
        const auto& capture_poses = scene.captures;
        const auto& view_poses = scene.camera_poses;
        const auto& camera_models = scene.camera_models;
        const CameraMasks& camera_masks = scene.camera_masks;
        const bool use_direct_cameras = scene.uses_direct_cameras;
        std::vector<GammaModel> exposure_models =
            loadExposureModels(options.exposure_models, view_poses.size());
        if (!options.exposure_ovs_binary_input.empty()) {
            if (!use_direct_cameras || !options.global_exposure) {
                throw std::runtime_error(
                    "--exposure-ovs-binary-input requires direct cameras and global exposure");
            }
            const ExposureProblem exposure_problem =
                loadExposureProblemFromPackedOvs(options, view_poses.size());
            exposure_models = solveExposureProblem(exposure_problem, std::move(exposure_models),
                                                   options.exposure_solver_threads,
                                                   options.exposure_solver_max_iterations,
                                                   options.exposure_solver_initial_trust_region_radius);
            writeExposureModels(options.exposure_model_output, exposure_models,
                                &exposure_problem.active_views);
            return 0;
        }
        const int depth_height =
            use_direct_cameras
                ? std::max(1,
                           static_cast<int>(std::lround(static_cast<double>(options.depth_width) *
                                                        camera_masks.height / camera_masks.width)))
                : options.depth_width / 2;
        DepthMaps depth_maps(view_poses.size(), options.depth_width, depth_height);

        DepthPreparation depth_preparation = prepareDepthAndCloudBounds(options, scene, depth_maps);
        auto candidate_cache = std::move(depth_preparation.candidate_cache);
        const Vec3f& cloud_minimum = depth_preparation.cloud_minimum;
        const Vec3f& cloud_maximum = depth_preparation.cloud_maximum;

        const auto depthVisible = [&depth_maps](int view, float x, float y, float range) {
            return depth_maps.isVisible(view, x, y, range);
        };
        const auto rankingDepthVisible = [&depth_maps](int view, float x, float y, float range,
                                                       float tolerance) {
            return depth_maps.isRankingVisible(view, x, y, range, tolerance);
        };

        std::optional<CameraImageCache> camera_images;
        std::size_t camera_cache_capacity = options.camera_cache;
        if (use_direct_cameras) {
            if (options.global_exposure && options.exposure_models.empty()) {
                camera_cache_capacity =
                    std::max(camera_cache_capacity, (view_poses.size() + 3U) / 4U);
            }
            camera_images.emplace(options.camera_directory, camera_cache_capacity);
            camera_images->preload(static_cast<int>(capture_poses.size()));
        }

        if (use_direct_cameras && options.global_exposure && options.exposure_models.empty()) {
            std::cerr << "Building binary-aligned 0.1 m exposure sample cloud\n";
            const std::vector<ExposurePoint> exposure_points =
                downsampleExposureCloud(options, layout, cloud_minimum, cloud_maximum);
            if (!options.exposure_cloud_output.empty()) {
                std::ofstream cloud_dump(options.exposure_cloud_output);
                if (!cloud_dump) {
                    throw std::runtime_error("cannot write exposure cloud " +
                                             options.exposure_cloud_output.string());
                }
                cloud_dump << std::setprecision(9);
                for (std::size_t point_index = 0; point_index < exposure_points.size();
                     ++point_index) {
                    const SurfacePoint& point = exposure_points[point_index].surface;
                    cloud_dump << "POINT " << point_index << ' ' << point.x << ' ' << point.y << ' '
                               << point.z << ' ' << point.nx << ' ' << point.ny << ' ' << point.nz
                               << '\n';
                }
            }
            // Exposure revisits views for every 0.1 m point.  The installed
            // worker preprocesses each camera image once; letting the LRU be
            // smaller than the capture set instead re-decodes full-resolution
            // JPEG pyramids for every point.  Keep all exposure captures
            // resident.  The final-color pass retains its independently
            // bounded cache below.
            const ExposureProblem exposure_problem = buildExposureProblem(
                options, exposure_points, view_poses, *camera_models, camera_masks, depth_height,
                depthVisible, rankingDepthVisible, *camera_images, camera_cache_capacity);
            if (options.exposure_capture_only) {
                std::cerr << "Exposure capture-only run complete\n";
                return 0;
            }
            exposure_models = solveExposureProblem(exposure_problem, std::move(exposure_models),
                                                   options.exposure_solver_threads,
                                                   options.exposure_solver_max_iterations,
                                                   options.exposure_solver_initial_trust_region_radius);
            writeExposureModels(options.exposure_model_output, exposure_models,
                                &exposure_problem.active_views);
            if (options.exposure_only) {
                std::cerr << "Exposure-only run complete\n";
                return 0;
            }
        } else if (use_direct_cameras && options.global_exposure) {
            writeExposureModels(options.exposure_model_output, exposure_models);
            if (options.exposure_only) {
                std::cerr << "Exposure-only run complete\n";
                return 0;
            }
        } else if (options.exposure_only || options.exposure_capture_only) {
            throw std::runtime_error("--exposure-only/--exposure-capture-only require direct "
                                     "cameras and global exposure");
        }

        const auto voxel_rankings = buildVoxelViewRankings(options, scene, depth_maps);

        writeColoredCloud(options, scene, depth_maps, camera_images, exposure_models,
                          voxel_rankings, std::move(candidate_cache));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "navvis_recon_surface_colorizer: " << error.what() << '\n';
        return 1;
    }
}
