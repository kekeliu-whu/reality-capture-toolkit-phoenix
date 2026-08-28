#include <libraw/libraw.h>

#include <dlfcn.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "navvis_recon/panorama_rendering.hpp"

namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMinimumOCamNormalizedZ = -0.13917310096006535;

struct Options {
    fs::path sensor_frame;
    fs::path input_directory;
    fs::path metadata_directory;
    fs::path processed_camera_directory;
    fs::path projected_camera_directory;
    fs::path blended_input;
    fs::path valid_mask;
    fs::path output;
    fs::path decoded_directory;
    fs::path camera_mask_directory;
    fs::path operator_mask;
    fs::path depth_map;
    fs::path world_map;
    fs::path seam_world_map;
    fs::path surface_cloud;
    fs::path panorama_info;
    fs::path debug_directory;
    std::string capture = "00000";
    std::string nadir_mode = "pyramid";
    std::string depth_translation_mode = "head-minus";
    std::string exposure_mode = "soft";
    std::string seam_mode = "pairwise";
    int width = 2048;
    int jpeg_quality = 95;
    int opencv_threads = -1;
    bool camera_only = false;
    std::string selected_camera;
};

struct OCamCamera {
    std::string name;
    Eigen::Matrix3d head_from_camera = Eigen::Matrix3d::Identity();
    Eigen::Vector3d head_from_camera_translation = Eigen::Vector3d::Zero();
    double c = 1.0;
    double d = 0.0;
    double e = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::vector<double> world_to_camera;
};

struct WarpedCamera {
    cv::Mat image;
    cv::Mat mask;
};

struct ProcessedDng {
    cv::Mat sensor_raster;
    int libraw_flip = 0;
};

ProcessedDng loadProcessedCameraJpeg(const fs::path& path) {
    // The vendor JPEG stores the unrotated sensor raster and relies on EXIF
    // Orientation for display. OCam projection consumes that encoded raster.
    const cv::Mat image = cv::imread(
        path.string(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::runtime_error(
            "Cannot read processed sensor-raster JPEG: " + path.string());
    }
    return {image, 0};
}

cv::Mat loadPanoramaDepthMetres(const fs::path& path) {
    const cv::Mat encoded = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (encoded.empty()) {
        throw std::runtime_error("Cannot read panorama depth map: " + path.string());
    }
    cv::Mat depth(encoded.rows, encoded.cols, CV_32F);
    if (encoded.type() == CV_8UC4) {
        // NavVis DepthMap PNG stores millimetres as a 16-bit big-endian value
        // in PNG R/G. OpenCV exposes PNG bytes as BGRA, hence G is the high
        // byte and R is the low byte.
        cv::parallel_for_(cv::Range(0, encoded.rows), [&](const cv::Range& rows) {
            for (int row = rows.start; row < rows.end; ++row) {
                for (int column = 0; column < encoded.cols; ++column) {
                    const cv::Vec4b value = encoded.at<cv::Vec4b>(row, column);
                    const std::uint16_t millimetres =
                        (static_cast<std::uint16_t>(value[1]) << 8U) |
                        static_cast<std::uint16_t>(value[2]);
                    depth.at<float>(row, column) = static_cast<float>(
                        static_cast<double>(millimetres) * 0.001);
                }
            }
        });
    } else if (encoded.type() == CV_16U) {
        encoded.convertTo(depth, CV_32F, 0.001);
    } else if (encoded.type() == CV_32F) {
        depth = encoded.clone();
    } else {
        throw std::runtime_error(
            "Unsupported panorama depth encoding in " + path.string());
    }
    return depth;
}

cv::Mat loadPanoramaWorldMap(
    const fs::path& path, int width, int height) {
    const std::uintmax_t expected_size =
        static_cast<std::uintmax_t>(width) *
        static_cast<std::uintmax_t>(height) * 3U * sizeof(float);
    if (fs::file_size(path) != expected_size) {
        throw std::runtime_error(
            "Panorama world map size does not match " +
            std::to_string(width) + "x" + std::to_string(height) +
            " CV_32FC3: " + path.string());
    }
    cv::Mat world_map(height, width, CV_32FC3);
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(world_map.data),
        static_cast<std::streamsize>(expected_size));
    if (!input) {
        throw std::runtime_error(
            "Cannot read panorama world map: " + path.string());
    }
    return world_map;
}

cv::Mat preparePanoramaProjectionDepth(
    const cv::Mat& native_depth, int width, bool quantize_to_millimetres) {
    if (native_depth.empty() ||
        (native_depth.type() != CV_32F && native_depth.type() != CV_64F)) {
        throw std::runtime_error(
            "Panorama world-map construction requires CV_32F or CV_64F depth");
    }

    cv::Mat native_float;
    if (quantize_to_millimetres) {
        cv::Mat1w millimetres(native_depth.size());
        cv::parallel_for_(cv::Range(0, native_depth.rows), [&](const cv::Range& rows) {
            for (int row = rows.start; row < rows.end; ++row) {
                std::uint16_t* output = millimetres.ptr<std::uint16_t>(row);
                for (int column = 0; column < native_depth.cols; ++column) {
                    const float value = native_depth.type() == CV_64F
                        ? static_cast<float>(native_depth.at<double>(row, column))
                        : native_depth.at<float>(row, column);
                    output[column] = static_cast<std::uint16_t>(std::clamp(
                        value * 1000.0F, 0.0F, 65535.0F));
                }
            }
        });
        // Step 2 reloads the millimetre depth PNG before its 8K projection.
        // The decoder multiplies in double and narrows once. OpenCV's
        // convertTo SIMD path multiplies uint16 by a float32 scale instead;
        // that one-ULP difference changes a few OCam mask-boundary samples.
        native_float.create(native_depth.size(), CV_32F);
        cv::parallel_for_(cv::Range(0, native_depth.rows), [&](const cv::Range& rows) {
            for (int row = rows.start; row < rows.end; ++row) {
                const std::uint16_t* input =
                    millimetres.ptr<std::uint16_t>(row);
                float* output = native_float.ptr<float>(row);
                for (int column = 0; column < native_depth.cols; ++column) {
                    output[column] = static_cast<float>(
                        static_cast<double>(input[column]) * 0.001);
                }
            }
        });
    } else {
        native_depth.convertTo(native_float, CV_32F);
    }

    cv::Mat projection_depth;
    cv::resize(
        native_float, projection_depth, cv::Size(width, width / 2),
        0.0, 0.0, cv::INTER_LINEAR);
    return projection_depth;
}

cv::Mat buildPanoramaWorldMap(const cv::Mat& projection_depth) {
    if (projection_depth.type() != CV_32F) {
        throw std::runtime_error(
            "Panorama world-map construction requires CV_32F projection depth");
    }
    cv::Mat world_map(
        projection_depth.size(), CV_32FC3, cv::Scalar::all(0));
    const int width = projection_depth.cols;
    const int height = projection_depth.rows;
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& rows) {
        for (int row = rows.start; row < rows.end; ++row) {
            const double latitude =
                (0.5 - (static_cast<double>(row) + 0.5) /
                           static_cast<double>(height)) *
                kPi;
            const double cos_latitude = std::cos(latitude);
            const float ray_z = static_cast<float>(std::sin(latitude));
            const float* depth = projection_depth.ptr<float>(row);
            cv::Vec3f* output = world_map.ptr<cv::Vec3f>(row);
            for (int column = 0; column < width; ++column) {
                const double longitude =
                    ((static_cast<double>(column) + 0.5) /
                         static_cast<double>(width) -
                     0.5) *
                    (2.0 * kPi);
                const cv::Vec3f ray(
                    static_cast<float>(
                        cos_latitude * std::cos(longitude)),
                    static_cast<float>(
                        -cos_latitude * std::sin(longitude)),
                    ray_z);
                output[column] = ray * depth[column];
            }
        }
    });
    return world_map;
}

void writeDebugDepthMap(
    const cv::Mat& depth_metres, const fs::path& directory,
    const std::string& filename) {
    if (directory.empty()) {
        return;
    }
    fs::create_directories(directory);
    cv::Mat1w millimetres(depth_metres.size());
    for (int row = 0; row < depth_metres.rows; ++row) {
        std::uint16_t* destination = millimetres.ptr<std::uint16_t>(row);
        for (int column = 0; column < depth_metres.cols; ++column) {
            const float depth = depth_metres.type() == CV_64F
                                    ? static_cast<float>(
                                          depth_metres.at<double>(row, column))
                                    : depth_metres.at<float>(row, column);
            const float scaled = depth * 1000.0F;
            destination[column] = static_cast<std::uint16_t>(
                std::clamp(scaled, 0.0F, 65535.0F));
        }
    }
    if (!cv::imwrite((directory / filename).string(), millimetres)) {
        throw std::runtime_error("Cannot write debug depth map: " +
                                 (directory / filename).string());
    }
}

void writeDebugDepthRaw(
    const cv::Mat& depth_metres, const fs::path& directory,
    const std::string& filename) {
    if (directory.empty()) {
        return;
    }
    if (depth_metres.type() != CV_64F) {
        throw std::runtime_error("Raw debug depth output requires CV_64F input");
    }
    fs::create_directories(directory);
    const fs::path path = directory / filename;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    for (int row = 0; row < depth_metres.rows; ++row) {
        output.write(
            reinterpret_cast<const char*>(depth_metres.ptr<double>(row)),
            static_cast<std::streamsize>(depth_metres.cols * sizeof(double)));
    }
    if (!output) {
        throw std::runtime_error("Cannot write raw debug depth map: " + path.string());
    }
}

std::string readTextFile(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Cannot read: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::vector<navvis_recon::Vec3f> readSurfacePositions(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open surface cloud: " + path.string());
    }
    std::string line;
    std::uint64_t count = 0U;
    std::vector<std::pair<std::string, std::string>> properties;
    while (std::getline(input, line)) {
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(^element vertex ([0-9]+)$)"))) {
            count = std::stoull(match[1].str());
        } else if (std::regex_match(
                       line, match, std::regex(R"(^property ([^ ]+) ([^ ]+)$)"))) {
            properties.emplace_back(match[1].str(), match[2].str());
        } else if (line == "end_header") {
            break;
        }
    }
    const std::vector<std::pair<std::string, std::string>> surface{
        {"float", "x"}, {"float", "y"}, {"float", "z"}, {"float", "intensity"},
        {"float", "nx"}, {"float", "ny"}, {"float", "nz"}, {"float", "curvature"}};
    const std::vector<std::pair<std::string, std::string>> colored{
        {"float", "x"}, {"float", "y"}, {"float", "z"},
        {"uchar", "red"}, {"uchar", "green"}, {"uchar", "blue"}, {"uchar", "alpha"},
        {"float", "intensity"}, {"float", "nx"}, {"float", "ny"}, {"float", "nz"},
        {"float", "curvature"}};
    const std::size_t stride = properties == surface ? 32U : properties == colored ? 36U : 0U;
    const std::streamoff data_offset = input.tellg();
    if (count == 0U || stride == 0U || data_offset <= 0) {
        throw std::runtime_error(
            "Surface cloud must use the 8-float or 36-byte colored PLY layout");
    }
    const std::uint64_t expected_size =
        static_cast<std::uint64_t>(data_offset) + count * stride;
    if (fs::file_size(path) != expected_size) {
        throw std::runtime_error("Surface cloud is incomplete: " + path.string());
    }

    constexpr std::size_t chunk_points = 262144U;
    std::vector<char> bytes(chunk_points * stride);
    std::vector<navvis_recon::Vec3f> positions;
    positions.reserve(static_cast<std::size_t>(count));
    std::uint64_t read_points = 0U;
    while (read_points < count) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(chunk_points, count - read_points));
        input.read(bytes.data(), static_cast<std::streamsize>(chunk * stride));
        if (!input) {
            throw std::runtime_error("Failed while reading surface cloud: " + path.string());
        }
        for (std::size_t index = 0; index < chunk; ++index) {
            float xyz[3];
            std::memcpy(xyz, bytes.data() + index * stride, sizeof(xyz));
            positions.emplace_back(xyz[0], xyz[1], xyz[2]);
        }
        read_points += chunk;
    }
    return positions;
}

std::string panoramaPoseSection(const std::string& json, const std::string& name) {
    const auto begin = json.find('"' + name + '"');
    if (begin == std::string::npos) {
        throw std::runtime_error("Missing " + name + " in panorama info");
    }
    const auto object_begin = json.find('{', begin);
    if (object_begin == std::string::npos) {
        throw std::runtime_error("Invalid " + name + " object in panorama info");
    }
    int depth = 0;
    for (std::size_t index = object_begin; index < json.size(); ++index) {
        if (json[index] == '{') {
            ++depth;
        } else if (json[index] == '}' && --depth == 0) {
            return json.substr(object_begin, index - object_begin + 1U);
        }
    }
    throw std::runtime_error("Unterminated " + name + " in panorama info");
}

Eigen::Vector3d panoramaPosePosition(
    const std::string& json, const std::string& name) {
    const std::string section = panoramaPoseSection(json, name);
    const std::string number = R"(([-+0-9.eE]+))";
    const std::regex position_expression(
        "\\\"position\\\"\\s*:\\s*\\[\\s*" + number +
        "\\s*,\\s*" + number + "\\s*,\\s*" + number + "\\s*\\]");
    std::smatch position;
    if (!std::regex_search(section, position, position_expression)) {
        throw std::runtime_error("Invalid " + name + " position in panorama info");
    }
    return Eigen::Vector3d(
        std::stod(position[1].str()), std::stod(position[2].str()),
        std::stod(position[3].str()));
}

navvis_recon::Pose readPanoramaHeadPose(const fs::path& path) {
    const std::string json = readTextFile(path);
    const std::string section = panoramaPoseSection(json, "cam_head");
    const std::string number = R"(([-+0-9.eE]+))";
    const std::regex quaternion_expression(
        "\\\"quaternion\\\"\\s*:\\s*\\[\\s*" + number +
        "\\s*,\\s*" + number + "\\s*,\\s*" + number +
        "\\s*,\\s*" + number + "\\s*\\]");
    std::smatch quaternion;
    if (!std::regex_search(section, quaternion, quaternion_expression)) {
        throw std::runtime_error("Invalid cam_head pose in panorama info: " + path.string());
    }
    const Eigen::Vector3d position = panoramaPosePosition(json, "cam_head");
    Eigen::Quaterniond rotation(
        std::stod(quaternion[1].str()), std::stod(quaternion[2].str()),
        std::stod(quaternion[3].str()), std::stod(quaternion[4].str()));
    rotation.normalize();
    navvis_recon::Pose pose;
    pose.translation_double = position;
    pose.rotation_double = rotation;
    pose.rotation_matrix_double = rotation.toRotationMatrix();
    pose.has_double_pose = true;
    pose.translation = position.cast<float>();
    pose.rotation = rotation.cast<float>();
    return pose;
}

double readPanoramaNearDistance(
    const fs::path& path, const navvis_recon::Pose& head_pose) {
    const std::string json = readTextFile(path);
    const Eigen::Vector3d head = head_pose.has_double_pose
                                     ? head_pose.translation_double
                                     : head_pose.translation.cast<double>();
    double near_distance = 0.0;
    for (int camera = 0; camera < 4; ++camera) {
        near_distance = std::max(
            near_distance,
            (panoramaPosePosition(json, "cam" + std::to_string(camera)) - head).norm());
    }
    return near_distance;
}

std::string xmlElement(const std::string& text, const std::string& tag) {
    const std::regex expression(
        "<" + tag + R"((?:\s[^>]*)?>\s*([\s\S]*?)\s*</)" + tag + ">",
        std::regex::icase);
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error("Missing XML element: " + tag);
    }
    return match[1].str();
}

double xmlDouble(const std::string& text, const std::string& tag) {
    return std::stod(xmlElement(text, tag));
}

std::vector<OCamCamera> readCameras(const fs::path& sensor_frame) {
    const std::string xml = readTextFile(sensor_frame);
    const std::regex camera_expression(R"(<CameraModel>\s*([\s\S]*?)\s*</CameraModel>)");
    std::vector<OCamCamera> cameras;
    for (auto iterator = std::sregex_iterator(xml.begin(), xml.end(), camera_expression);
         iterator != std::sregex_iterator(); ++iterator) {
        const std::string block = (*iterator)[1].str();
        OCamCamera camera;
        camera.name = xmlElement(block, "SensorName");
        const std::string pose = xmlElement(block, "Pose");
        const std::string position = xmlElement(pose, "position");
        camera.head_from_camera_translation = Eigen::Vector3d(
            xmlDouble(position, "x"), xmlDouble(position, "y"),
            xmlDouble(position, "z"));
        const std::string orientation = xmlElement(pose, "orientation");
        Eigen::Quaterniond quaternion(
            xmlDouble(orientation, "w"), xmlDouble(orientation, "x"),
            xmlDouble(orientation, "y"), xmlDouble(orientation, "z"));
        camera.head_from_camera = quaternion.normalized().toRotationMatrix();

        const std::string ocam = xmlElement(block, "OCamModel");
        camera.c = xmlDouble(ocam, "c");
        camera.d = xmlDouble(ocam, "d");
        camera.e = xmlDouble(ocam, "e");
        camera.cx = xmlDouble(ocam, "cx");
        camera.cy = xmlDouble(ocam, "cy");
        const std::string polynomial = xmlElement(ocam, "world2cam");
        const std::regex coefficient_expression(R"(<coeff>\s*([^<]+?)\s*</coeff>)");
        for (auto coefficient = std::sregex_iterator(
                 polynomial.begin(), polynomial.end(), coefficient_expression);
             coefficient != std::sregex_iterator(); ++coefficient) {
            camera.world_to_camera.push_back(std::stod((*coefficient)[1].str()));
        }
        if (camera.world_to_camera.empty()) {
            throw std::runtime_error(camera.name + " has no OCam world2cam coefficients");
        }
        cameras.push_back(std::move(camera));
    }
    std::sort(cameras.begin(), cameras.end(), [](const OCamCamera& first, const OCamCamera& second) {
        return first.name < second.name;
    });
    if (cameras.size() != 4 || cameras[0].name != "cam0" || cameras[3].name != "cam3") {
        throw std::runtime_error("Expected cam0..cam3 in sensor_frame.xml");
    }

    return cameras;
}

void checkLibRaw(int status, const fs::path& path, const char* operation) {
    if (status != LIBRAW_SUCCESS) {
        throw std::runtime_error(
            path.string() + ": " + operation + ": " + libraw_strerror(status));
    }
}

class DngHost {
public:
    DngHost() {
        // The LibRaw 0.22 build installed with NavVis delegates this camera's
        // 8-bit linear DNG to Adobe's DNG SDK. dng_host is 0x68 bytes in the
        // installed ABI (also visible in the original stack layout). Its
        // constructor/destructor are exported by liblibnavvis_dng_sdk.
        constructor_ = resolve<Constructor>(
            "_ZN8dng_hostC1EP20dng_memory_allocatorP17dng_abort_sniffer");
        destructor_ = resolve<Destructor>("_ZN8dng_hostD1Ev");
        constructor_(storage_.data(), nullptr, nullptr);
        live_ = true;
    }

    DngHost(const DngHost&) = delete;
    DngHost& operator=(const DngHost&) = delete;

    ~DngHost() { destroy(); }

    void* get() { return storage_.data(); }

    void destroy() {
        if (live_) {
            destructor_(storage_.data());
            live_ = false;
        }
    }

private:
    template <typename Function>
    static Function resolve(const char* symbol) {
        dlerror();
        void* address = dlsym(RTLD_DEFAULT, symbol);
        const char* error = dlerror();
        if (error || !address) {
            throw std::runtime_error(
                std::string("Cannot resolve DNG SDK symbol ") + symbol +
                (error ? std::string(": ") + error : std::string()));
        }
        return reinterpret_cast<Function>(address);
    }

    using Constructor = void (*)(void*, void*, void*);
    using Destructor = void (*)(void*);
    alignas(16) std::array<std::byte, 0x68> storage_{};
    Constructor constructor_ = nullptr;
    Destructor destructor_ = nullptr;
    bool live_ = false;
};

void configureCommonRawParameters(libraw_data_t* raw) {
    raw->params.gamm[0] = 0.45;
    raw->params.gamm[1] = 4.5;
    raw->params.use_camera_matrix = 1;
    raw->params.output_color = 1;
    raw->params.user_flip = 0;
    raw->params.user_qual = 3;
    raw->params.no_auto_bright = 1;
}

cv::Mat renderRawExposure(
    libraw_data_t* raw, const fs::path& path, double exposure_shift,
    bool exposure_correction) {
    raw->params.exp_correc = exposure_correction ? 1 : 0;
    raw->params.exp_shift = static_cast<float>(exposure_shift);
    raw->params.exp_preser = exposure_correction ? 1.0F : 0.0F;
    checkLibRaw(libraw_dcraw_process(raw), path, "dcraw_process");

    int memory_status = LIBRAW_SUCCESS;
    libraw_processed_image_t* output = libraw_dcraw_make_mem_image(raw, &memory_status);
    if (!output || memory_status != LIBRAW_SUCCESS || output->type != LIBRAW_IMAGE_BITMAP ||
        output->colors != 3 || output->bits != 16) {
        if (output) {
            libraw_dcraw_clear_mem(output);
        }
        throw std::runtime_error("Unexpected 16-bit LibRaw output for " + path.string());
    }

    const cv::Mat rgb16(
        static_cast<int>(output->height), static_cast<int>(output->width),
        CV_16UC3, output->data);
    cv::Mat bgr16;
    cv::cvtColor(rgb16, bgr16, cv::COLOR_RGB2BGR);
    cv::Mat bgr32;
    // The binary hands MergeMertens float BGR samples in [0,255]. Captured
    // matrices show exact uint16/257 conversion rather than normalization to
    // [0,1].
    bgr16.convertTo(bgr32, CV_32FC3, 1.0 / 257.0);
    libraw_dcraw_clear_mem(output);
    return bgr32;
}

float fusedNlmStrength(float iso_sensitivity, double reference_exposure_stops = 1.5) {
    // Recovered from nv_image-postprocessing 6.0.7.1:
    //
    //   gain = max(0, 4.5 + 6 * log2(ISO / 100))
    //   noise = gain + 6 * reference_exposure_stops
    //   h = 1.75 * pow(7.5 / 1.75, noise / 27)
    //
    // The four constants below are the live high-quality NLM-Fused parameter
    // object, not a fit to output images.  ISO is read from the DNG metadata;
    // the high-quality HDR bracket has its reference exposure at +1.5 EV.
    if (!(iso_sensitivity > 0.0F) || !std::isfinite(iso_sensitivity)) {
        return 4.5F;
    }
    const double sensor_gain = std::max(
        0.0, 4.5 + 6.0 * std::log2(static_cast<double>(iso_sensitivity) / 100.0));
    const double noise_level = sensor_gain + 6.0 * reference_exposure_stops;
    return static_cast<float>(
        1.75 * std::pow(7.5 / 1.75, noise_level / 27.0));
}

ProcessedDng postprocessDngHighQuality(const fs::path& path) {
    DngHost dng_host;
    libraw_data_t* raw = libraw_init(0);
    if (!raw) {
        throw std::runtime_error("libraw_init failed");
    }
    try {
        static_cast<LibRaw*>(raw->parent_class)->set_dng_host(dng_host.get());
        configureCommonRawParameters(raw);
        checkLibRaw(libraw_open_file(raw, path.c_str()), path, "open");
        checkLibRaw(libraw_unpack(raw), path, "unpack");
        const float iso_sensitivity = raw->other.iso_speed;
        // Preserve the TIFF display orientation before user_flip=0 is applied
        // by LibRaw's processing pass.  The sensor raster itself stays
        // unrotated for the OCam projection path.
        const int libraw_flip = raw->sizes.flip;

        // First pass is not emitted. The original uses it solely to estimate
        // auto-WB over this fixed sensor ROI, then freezes color.pre_mul for
        // all three HDR renders from the same unpacked mosaic.
        raw->params.greybox[0] = 980U;
        raw->params.greybox[1] = 40U;
        raw->params.greybox[2] = 3500U;
        raw->params.greybox[3] = 3500U;
        raw->params.use_auto_wb = 1;
        raw->params.use_camera_wb = 0;
        raw->params.output_bps = 8;
        raw->params.no_interpolation = 1;
        raw->params.exp_correc = 0;
        checkLibRaw(libraw_dcraw_process(raw), path, "auto-WB discovery");

        std::array<float, 4> white_balance{};
        std::copy(std::begin(raw->color.pre_mul), std::end(raw->color.pre_mul),
                  white_balance.begin());
        std::fill(std::begin(raw->params.greybox), std::end(raw->params.greybox),
                  std::numeric_limits<unsigned>::max());
        std::copy(white_balance.begin(), white_balance.end(),
                  std::begin(raw->params.user_mul));
        raw->params.use_auto_wb = 0;
        raw->params.use_camera_wb = 0;
        raw->params.output_bps = 16;
        raw->params.no_interpolation = 0;

        std::vector<cv::Mat> exposures;
        exposures.reserve(3);
        exposures.push_back(renderRawExposure(raw, path, 1.0, false));
        exposures.push_back(renderRawExposure(raw, path, std::pow(2.0, 1.5), true));
        exposures.push_back(renderRawExposure(raw, path, 8.0, true));

        cv::Mat fused;
        cv::createMergeMertens(1.0F, 1.0F, 1.0F)->process(exposures, fused);
        exposures.clear();

        double global_minimum = 0.0;
        cv::minMaxLoc(fused, &global_minimum, nullptr);
        // Preserve the vendor MatExpr evaluation order.  Multiplication by
        // the reciprocal and subtraction of the independently scaled minimum
        // differ by one float ULP from `(fused - minimum) / range`; that ULP
        // is enough to flip a small number of later 8-bit tone-map samples.
        constexpr double dynamic_range = 1.333;
        const float inverse_range = static_cast<float>(1.0 / dynamic_range);
        const float scaled_minimum =
            static_cast<float>(global_minimum / dynamic_range);
        cv::Mat normalized(fused.size(), CV_32FC3);
        cv::parallel_for_(cv::Range(0, fused.rows), [&](const cv::Range& rows) {
            for (int row = rows.start; row < rows.end; ++row) {
                const cv::Vec3f* source = fused.ptr<cv::Vec3f>(row);
                cv::Vec3f* destination = normalized.ptr<cv::Vec3f>(row);
                for (int column = 0; column < fused.cols; ++column) {
                    for (int channel = 0; channel < 3; ++channel) {
                        const float scaled = source[column][channel] * inverse_range;
                        destination[column][channel] = scaled - scaled_minimum;
                    }
                }
            }
        });
        cv::threshold(normalized, normalized, 0.0, 0.0, cv::THRESH_TOZERO);
        cv::threshold(normalized, normalized, 1.0, 1.0, cv::THRESH_TRUNC);

        // Binary helper: cubic Hermite with zero endpoint slopes (smoothstep),
        // followed by its default contrast=1.1 around middle gray. Keep the
        // four MatExpr operations in the order observed in the installed
        // binary. Letting the compiler choose the operand evaluation order
        // changes a few float32 tie cases before the 8-bit conversion.
        const cv::MatExpr linear_term = 3.0 * normalized;
        const cv::MatExpr squared = normalized.mul(normalized);
        const cv::MatExpr quadratic_term = -2.0 * squared;
        cv::Mat tone = normalized.mul(linear_term + quadratic_term);
        const cv::Scalar middle_gray = cv::Scalar::all(0.5);
        tone = middle_gray + (tone - middle_gray) * 1.1;
        cv::Mat image8;
        tone.convertTo(image8, CV_8UC3, 255.0);

        cv::Mat denoised;
        cv::fastNlMeansDenoising(
            image8, denoised, fusedNlmStrength(iso_sensitivity), 7, 17);
        cv::Mat blurred;
        cv::GaussianBlur(denoised, blurred, cv::Size(), 3.0, 3.0);
        cv::Mat sharpened;
        cv::addWeighted(denoised, 1.5, blurred, -0.5, 0.0, sharpened);
        dng_host.destroy();
        libraw_close(raw);
        return {sharpened, libraw_flip};
    } catch (...) {
        dng_host.destroy();
        libraw_close(raw);
        throw;
    }
}

std::uint16_t tiffOrientation(int libraw_flip) {
    switch (libraw_flip) {
        case 0: return 1;
        case 3: return 3;
        case 5: return 8;
        case 6: return 6;
        default:
            throw std::runtime_error(
                "Unsupported LibRaw display orientation: " +
                std::to_string(libraw_flip));
    }
}

struct TiffValue {
    std::uint16_t type = 0;
    std::uint32_t count = 0;
    std::vector<std::uint8_t> data;
};

std::uint16_t readUnsigned16(
    const std::vector<std::uint8_t>& bytes, std::size_t offset,
    bool little_endian) {
    if (offset + 2U > bytes.size()) {
        throw std::runtime_error("Truncated TIFF uint16");
    }
    if (little_endian) {
        return static_cast<std::uint16_t>(bytes[offset]) |
               (static_cast<std::uint16_t>(bytes[offset + 1U]) << 8U);
    }
    return (static_cast<std::uint16_t>(bytes[offset]) << 8U) |
           static_cast<std::uint16_t>(bytes[offset + 1U]);
}

std::uint32_t readUnsigned32(
    const std::vector<std::uint8_t>& bytes, std::size_t offset,
    bool little_endian) {
    if (offset + 4U > bytes.size()) {
        throw std::runtime_error("Truncated TIFF uint32");
    }
    std::uint32_t value = 0U;
    if (little_endian) {
        for (int index = 3; index >= 0; --index) {
            value = (value << 8U) | bytes[offset + static_cast<std::size_t>(index)];
        }
    } else {
        for (std::size_t index = 0; index < 4U; ++index) {
            value = (value << 8U) | bytes[offset + index];
        }
    }
    return value;
}

void appendUnsigned16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xffU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
}

void appendUnsigned32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned shift = 0U; shift < 32U; shift += 8U) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

std::size_t tiffTypeSize(std::uint16_t type) {
    switch (type) {
        case 1:
        case 2:
        case 6:
        case 7: return 1U;
        case 3:
        case 8: return 2U;
        case 4:
        case 9:
        case 11: return 4U;
        case 5:
        case 10:
        case 12: return 8U;
        default: throw std::runtime_error("Unsupported TIFF field type");
    }
}

std::map<std::uint16_t, TiffValue> readTiffIfd(
    const std::vector<std::uint8_t>& bytes, std::uint32_t offset,
    bool little_endian) {
    const std::uint16_t count = readUnsigned16(bytes, offset, little_endian);
    const std::size_t entries_begin = static_cast<std::size_t>(offset) + 2U;
    if (entries_begin + static_cast<std::size_t>(count) * 12U + 4U > bytes.size()) {
        throw std::runtime_error("Truncated TIFF IFD");
    }

    std::map<std::uint16_t, TiffValue> result;
    for (std::uint16_t index = 0; index < count; ++index) {
        const std::size_t entry = entries_begin + static_cast<std::size_t>(index) * 12U;
        TiffValue value;
        const std::uint16_t tag = readUnsigned16(bytes, entry, little_endian);
        value.type = readUnsigned16(bytes, entry + 2U, little_endian);
        value.count = readUnsigned32(bytes, entry + 4U, little_endian);
        const std::size_t element_size = tiffTypeSize(value.type);
        if (value.count > std::numeric_limits<std::size_t>::max() / element_size) {
            throw std::runtime_error("Oversized TIFF field");
        }
        const std::size_t byte_count = static_cast<std::size_t>(value.count) * element_size;
        const std::size_t data_offset = byte_count <= 4U
                                            ? entry + 8U
                                            : readUnsigned32(bytes, entry + 8U, little_endian);
        if (data_offset + byte_count > bytes.size()) {
            throw std::runtime_error("TIFF field points outside the file");
        }
        value.data.assign(
            bytes.begin() + static_cast<std::ptrdiff_t>(data_offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(data_offset + byte_count));

        // The output JPEG uses a little-endian TIFF. Normalize numeric source
        // values if a big-endian metadata source is ever supplied.
        if (!little_endian && element_size > 1U) {
            const std::size_t component_size =
                (value.type == 3 || value.type == 8) ? 2U : 4U;
            for (std::size_t component = 0; component < value.data.size();
                 component += component_size) {
                std::reverse(
                    value.data.begin() + static_cast<std::ptrdiff_t>(component),
                    value.data.begin() + static_cast<std::ptrdiff_t>(component + component_size));
            }
        }
        result.emplace(tag, std::move(value));
    }
    return result;
}

const TiffValue& requireTiffValue(
    const std::map<std::uint16_t, TiffValue>& values, std::uint16_t tag,
    const fs::path& path) {
    const auto value = values.find(tag);
    if (value == values.end()) {
        std::ostringstream message;
        message << path << ": required TIFF/EXIF tag 0x" << std::hex << tag
                << " is missing";
        throw std::runtime_error(message.str());
    }
    return value->second;
}

std::uint32_t inlineUnsigned32(const TiffValue& value) {
    if (value.type != 4U || value.count != 1U || value.data.size() != 4U) {
        throw std::runtime_error("Expected one TIFF LONG value");
    }
    return static_cast<std::uint32_t>(value.data[0]) |
           (static_cast<std::uint32_t>(value.data[1]) << 8U) |
           (static_cast<std::uint32_t>(value.data[2]) << 16U) |
           (static_cast<std::uint32_t>(value.data[3]) << 24U);
}

using TaggedTiffValue = std::pair<std::uint16_t, TiffValue>;

std::size_t externalTiffSize(const std::vector<TaggedTiffValue>& values) {
    std::size_t size = 0U;
    for (const auto& [tag, value] : values) {
        static_cast<void>(tag);
        if (value.data.size() > 4U) {
            size += value.data.size();
            size += size & 1U;
        }
    }
    return size;
}

std::vector<std::uint8_t> encodeTiffIfd(
    const std::vector<TaggedTiffValue>& values, std::uint32_t ifd_offset,
    std::uint32_t external_offset) {
    std::vector<std::uint8_t> result;
    std::vector<std::uint8_t> external;
    appendUnsigned16(result, static_cast<std::uint16_t>(values.size()));
    for (const auto& [tag, value] : values) {
        appendUnsigned16(result, tag);
        appendUnsigned16(result, value.type);
        appendUnsigned32(result, value.count);
        if (value.data.size() <= 4U) {
            result.insert(result.end(), value.data.begin(), value.data.end());
            result.resize(result.size() + (4U - value.data.size()), 0U);
        } else {
            appendUnsigned32(
                result, external_offset + static_cast<std::uint32_t>(external.size()));
            external.insert(external.end(), value.data.begin(), value.data.end());
            if ((external.size() & 1U) != 0U) {
                external.push_back(0U);
            }
        }
    }
    appendUnsigned32(result, 0U);
    const std::size_t expected_header_size =
        2U + values.size() * 12U + 4U;
    if (result.size() != expected_header_size ||
        ifd_offset + result.size() != external_offset) {
        throw std::runtime_error("Internal TIFF layout error");
    }
    result.insert(result.end(), external.begin(), external.end());
    return result;
}

std::vector<std::uint8_t> buildProcessedExif(
    const fs::path& metadata_path, std::uint16_t orientation) {
    std::ifstream input(metadata_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot read DNG metadata: " + metadata_path.string());
    }
    std::vector<std::uint8_t> source(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (source.size() < 8U ||
        !((source[0] == 'I' && source[1] == 'I') ||
          (source[0] == 'M' && source[1] == 'M'))) {
        throw std::runtime_error("Metadata source is not a TIFF/DNG: " + metadata_path.string());
    }
    const bool little_endian = source[0] == 'I';
    if (readUnsigned16(source, 2U, little_endian) != 42U) {
        throw std::runtime_error("Invalid TIFF/DNG header: " + metadata_path.string());
    }
    const auto image_ifd = readTiffIfd(
        source, readUnsigned32(source, 4U, little_endian), little_endian);
    const TiffValue& exif_pointer = requireTiffValue(image_ifd, 0x8769U, metadata_path);
    const auto exif_ifd = readTiffIfd(
        source, inlineUnsigned32(exif_pointer), little_endian);

    constexpr std::array<std::uint16_t, 4> image_tags{
        0x010fU, 0x0110U, 0x0112U, 0x0132U};
    constexpr std::array<std::uint16_t, 10> exif_tags{
        0x829aU, 0x829dU, 0x8827U, 0x9202U, 0x9204U,
        0x9207U, 0x920aU, 0xa402U, 0xa403U, 0xa431U};

    std::vector<TaggedTiffValue> image_values;
    image_values.reserve(image_tags.size() + 1U);
    for (const std::uint16_t tag : image_tags) {
        image_values.emplace_back(tag, requireTiffValue(image_ifd, tag, metadata_path));
    }
    image_values[2].second = TiffValue{
        3U, 1U,
        {static_cast<std::uint8_t>(orientation & 0xffU),
         static_cast<std::uint8_t>((orientation >> 8U) & 0xffU)}};

    std::vector<TaggedTiffValue> exif_values;
    exif_values.reserve(exif_tags.size());
    for (const std::uint16_t tag : exif_tags) {
        exif_values.emplace_back(tag, requireTiffValue(exif_ifd, tag, metadata_path));
    }

    constexpr std::uint32_t image_ifd_offset = 8U;
    const std::uint32_t image_external_offset =
        image_ifd_offset + 2U +
        static_cast<std::uint32_t>((image_values.size() + 1U) * 12U) + 4U;
    const std::uint32_t exif_ifd_offset = image_external_offset +
        static_cast<std::uint32_t>(externalTiffSize(image_values));
    TiffValue output_exif_pointer{4U, 1U, {}};
    appendUnsigned32(output_exif_pointer.data, exif_ifd_offset);
    image_values.emplace_back(0x8769U, std::move(output_exif_pointer));

    const std::uint32_t exif_external_offset =
        exif_ifd_offset + 2U +
        static_cast<std::uint32_t>(exif_values.size() * 12U) + 4U;
    const std::vector<std::uint8_t> encoded_image_ifd = encodeTiffIfd(
        image_values, image_ifd_offset, image_external_offset);
    const std::vector<std::uint8_t> encoded_exif_ifd = encodeTiffIfd(
        exif_values, exif_ifd_offset, exif_external_offset);

    std::vector<std::uint8_t> tiff{'I', 'I', 0x2aU, 0x00U};
    appendUnsigned32(tiff, image_ifd_offset);
    tiff.insert(tiff.end(), encoded_image_ifd.begin(), encoded_image_ifd.end());
    tiff.insert(tiff.end(), encoded_exif_ifd.begin(), encoded_exif_ifd.end());

    std::vector<std::uint8_t> segment{
        0xffU, 0xe1U, 0x00U, 0x00U, 'E', 'x', 'i', 'f', 0x00U, 0x00U};
    segment.insert(segment.end(), tiff.begin(), tiff.end());
    if (segment.size() > 65537U) {
        throw std::runtime_error("EXIF metadata does not fit in one JPEG APP1 segment");
    }
    const std::uint16_t jpeg_length = static_cast<std::uint16_t>(segment.size() - 2U);
    segment[2] = static_cast<std::uint8_t>((jpeg_length >> 8U) & 0xffU);
    segment[3] = static_cast<std::uint8_t>(jpeg_length & 0xffU);
    return segment;
}

bool writeProcessedJpeg(
    const fs::path& path, const ProcessedDng& processed, int jpeg_quality,
    const fs::path& metadata_path) {
    // The vendor sends the unrotated sensor raster to OpenCV's JPEG encoder,
    // then copies TIFF Orientation into EXIF.  Rotating before compression
    // changes the 8x8 block layout and creates a measurable pixel error.
    if (!cv::imwrite(
            path.string(), processed.sensor_raster,
            {cv::IMWRITE_JPEG_QUALITY, std::clamp(jpeg_quality, 35, 100),
             cv::IMWRITE_JPEG_OPTIMIZE, 1})) {
        return false;
    }

    std::ifstream input(path, std::ios::binary);
    std::vector<std::uint8_t> jpeg(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (jpeg.size() < 2 || jpeg[0] != 0xff || jpeg[1] != 0xd8) {
        throw std::runtime_error("JPEG encoder produced an invalid stream: " + path.string());
    }

    const std::vector<std::uint8_t> exif = buildProcessedExif(
        metadata_path, tiffOrientation(processed.libraw_flip));
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    // OpenCV emits a 16-byte JFIF APP0 marker immediately after SOI.  The
    // reference stream keeps APP0 first and places EXIF directly after it.
    constexpr std::size_t jfif_marker_size = 18;
    if (jpeg.size() < 2 + jfif_marker_size || jpeg[2] != 0xff || jpeg[3] != 0xe0) {
        throw std::runtime_error("JPEG encoder did not emit the expected JFIF marker: " +
                                 path.string());
    }
    output.write(
        reinterpret_cast<const char*>(jpeg.data()),
        static_cast<std::streamsize>(2 + jfif_marker_size));
    output.write(reinterpret_cast<const char*>(exif.data()), exif.size());
    output.write(
        reinterpret_cast<const char*>(jpeg.data() + 2 + jfif_marker_size),
        static_cast<std::streamsize>(jpeg.size() - 2 - jfif_marker_size));
    if (!output) {
        throw std::runtime_error("Failed to attach EXIF orientation: " + path.string());
    }
    return true;
}

double evaluatePolynomial(const std::vector<double>& coefficients, double value) {
    double result = 0.0;
    for (auto coefficient = coefficients.rbegin(); coefficient != coefficients.rend(); ++coefficient) {
        result = result * value + *coefficient;
    }
    return result;
}

cv::Vec3b sampleVendorBilinearBgr(
    const cv::Mat& image, float image_x, float image_y) {
    // liblibnavvis_image's getColorSubpixb<Bilinear> truncates the float32
    // coordinate, samples four neighbours, performs the two interpolation
    // stages in float32, and rounds the result to uint8.  OpenCV remap uses a
    // 1/32-pixel interpolation table, so it is measurably different here.
    const int x0 = static_cast<int>(image_x);
    const int y0 = static_cast<int>(image_y);
    const int x1 = cv::borderInterpolate(x0 + 1, image.cols, cv::BORDER_REFLECT_101);
    const int y1 = cv::borderInterpolate(y0 + 1, image.rows, cv::BORDER_REFLECT_101);
    const int reflected_x0 = cv::borderInterpolate(x0, image.cols, cv::BORDER_REFLECT_101);
    const int reflected_y0 = cv::borderInterpolate(y0, image.rows, cv::BORDER_REFLECT_101);
    const float fraction_x = image_x - static_cast<float>(x0);
    const float fraction_y = image_y - static_cast<float>(y0);
    const float inverse_x = 1.0F - fraction_x;
    const float inverse_y = 1.0F - fraction_y;
    const cv::Vec3b& top_left = image.at<cv::Vec3b>(reflected_y0, reflected_x0);
    const cv::Vec3b& top_right = image.at<cv::Vec3b>(reflected_y0, x1);
    const cv::Vec3b& bottom_left = image.at<cv::Vec3b>(y1, reflected_x0);
    const cv::Vec3b& bottom_right = image.at<cv::Vec3b>(y1, x1);
    cv::Vec3b result;
    for (int channel = 0; channel < 3; ++channel) {
        const float top = static_cast<float>(top_left[channel]) * inverse_x +
                          static_cast<float>(top_right[channel]) * fraction_x;
        const float bottom = static_cast<float>(bottom_left[channel]) * inverse_x +
                             static_cast<float>(bottom_right[channel]) * fraction_x;
        result[channel] = cv::saturate_cast<std::uint8_t>(
            top * inverse_y + bottom * fraction_y);
    }
    return result;
}

float sampleVendorBilinearMask(
    const cv::Mat& image, float image_x, float image_y) {
    const int x0 = static_cast<int>(image_x);
    const int y0 = static_cast<int>(image_y);
    const int x1 = cv::borderInterpolate(
        x0 + 1, image.cols, cv::BORDER_REFLECT_101);
    const int y1 = cv::borderInterpolate(
        y0 + 1, image.rows, cv::BORDER_REFLECT_101);
    const int reflected_x0 = cv::borderInterpolate(
        x0, image.cols, cv::BORDER_REFLECT_101);
    const int reflected_y0 = cv::borderInterpolate(
        y0, image.rows, cv::BORDER_REFLECT_101);
    const float fraction_x = image_x - static_cast<float>(x0);
    const float fraction_y = image_y - static_cast<float>(y0);
    const float inverse_x = 1.0F - fraction_x;
    const float inverse_y = 1.0F - fraction_y;
    const float top =
        static_cast<float>(image.at<std::uint8_t>(reflected_y0, reflected_x0)) *
            inverse_x +
        static_cast<float>(image.at<std::uint8_t>(reflected_y0, x1)) * fraction_x;
    const float bottom =
        static_cast<float>(image.at<std::uint8_t>(y1, reflected_x0)) * inverse_x +
        static_cast<float>(image.at<std::uint8_t>(y1, x1)) * fraction_x;
    return top * inverse_y + bottom * fraction_y;
}

WarpedCamera warpCamera(
    const cv::Mat& source, const cv::Mat& source_mask,
    const cv::Mat& world_map_head, const cv::Mat& depth_m,
    const OCamCamera& camera, int width, int height,
    const std::string& depth_translation_mode) {
    cv::Mat map_x(height, width, CV_32F, cv::Scalar(-1.0F));
    cv::Mat map_y(height, width, CV_32F, cv::Scalar(-1.0F));
    cv::Mat mask(height, width, CV_8U, cv::Scalar(0));
    if (!source_mask.empty() &&
        (source_mask.type() != CV_8U || source_mask.size() != source.size())) {
        throw std::runtime_error(
            "Camera mask must be 8-bit grayscale and match the sensor raster");
    }
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& rows) {
        for (int row = rows.start; row < rows.end; ++row) {
            // Equirectangular pixels represent samples at their centres.  The
            // renderer in the standard pipeline constructs the spherical ray
            // from (column + 0.5, row + 0.5); using the pixel corner shifts
            // every projected camera by roughly half a panorama pixel.
            const double latitude =
                (0.5 - (static_cast<double>(row) + 0.5) / height) * kPi;
            const double cos_latitude = std::cos(latitude);
            for (int column = 0; column < width; ++column) {
                const double longitude =
                    ((static_cast<double>(column) + 0.5) / width - 0.5) *
                    (2.0 * kPi);
                // The vendor world map stores the positive CameraHead ray.
                // Its first sample and the runtime camera transform reproduce
                // the first projectCCS2ICS input to double precision.
                const Eigen::Vector3d ray_head(
                    cos_latitude * std::cos(longitude),
                    -cos_latitude * std::sin(longitude),
                    std::sin(latitude));
                Eigen::Vector3d ray_camera;
                if (!world_map_head.empty()) {
                    const cv::Vec3f point =
                        world_map_head.at<cv::Vec3f>(row, column);
                    if (std::isfinite(point[0]) && std::isfinite(point[1]) &&
                        std::isfinite(point[2]) && cv::norm(point) > 0.0) {
                        const Eigen::Vector3d point_head(
                            static_cast<double>(point[0]),
                            static_cast<double>(point[1]),
                            static_cast<double>(point[2]));
                        ray_camera = camera.head_from_camera.transpose() *
                            (point_head - camera.head_from_camera_translation);
                    } else {
                        ray_camera = camera.head_from_camera.transpose() * ray_head;
                    }
                } else if (!depth_m.empty()) {
                    const float depth = depth_m.at<float>(row, column);
                    if (std::isfinite(depth) && depth > 0.0F) {
                        const Eigen::Vector3d point_head =
                            ray_head * static_cast<double>(depth);
                        if (depth_translation_mode == "head-minus") {
                            ray_camera = camera.head_from_camera.transpose() *
                                (point_head - camera.head_from_camera_translation);
                        } else if (depth_translation_mode == "head-plus") {
                            ray_camera = camera.head_from_camera.transpose() *
                                (point_head + camera.head_from_camera_translation);
                        } else if (depth_translation_mode == "camera-minus") {
                            ray_camera = camera.head_from_camera.transpose() * point_head -
                                camera.head_from_camera_translation;
                        } else if (depth_translation_mode == "camera-plus") {
                            ray_camera = camera.head_from_camera.transpose() * point_head +
                                camera.head_from_camera_translation;
                        } else {
                            throw std::runtime_error(
                                "Unsupported depth translation mode: " +
                                depth_translation_mode);
                        }
                    } else {
                        ray_camera = camera.head_from_camera.transpose() * ray_head;
                    }
                } else {
                    ray_camera = camera.head_from_camera.transpose() * ray_head;
                }
                // OCamProjectionModel::projectCCS2ICSImpl consumes CCS
                // directly.  Its SIMD lane shuffles are equivalent to using
                // radial_u=(y/r)*rho and radial_v=(x/r)*rho, then returning
                // ICS in OpenCV's (image_x, image_y) order.
                const double x = ray_camera.x();
                const double y = ray_camera.y();
                const double z = ray_camera.z();
                const double radial = std::hypot(x, y);
                if (radial <= 1.0e-12) {
                    continue;
                }
                const double theta = std::atan(-z / radial);
                const double rho = evaluatePolynomial(camera.world_to_camera, theta);
                const double normalized_z = z / std::hypot(radial, z);
                // CameraModel's half-domain guard accepts normalized CCS z
                // above sin(-8 degrees); the within-image guard is applied by
                // the raster bounds check below and does not change ICS.
                if (rho < 0.0 || normalized_z <= kMinimumOCamNormalizedZ) {
                    continue;
                }
                const double inverse_radial = 1.0 / radial;
                const double radial_u = y * inverse_radial * rho;
                const double radial_v = x * inverse_radial * rho;
                const double image_x =
                    camera.cy + camera.e * radial_u + radial_v;
                const double image_y =
                    camera.cx + camera.c * radial_u + camera.d * radial_v;
                if (image_x < 0.0 || image_y < 0.0 ||
                    image_x >= source.cols || image_y >= source.rows) {
                    continue;
                }
                const float sample_x = static_cast<float>(image_x);
                const float sample_y = static_cast<float>(image_y);
                if (!source_mask.empty() &&
                    sampleVendorBilinearMask(source_mask, sample_x, sample_y) !=
                        255.0F) {
                    continue;
                }
                map_x.at<float>(row, column) = sample_x;
                map_y.at<float>(row, column) = sample_y;
                mask.at<std::uint8_t>(row, column) = 255;
            }
        }
    });
    cv::Mat warped(height, width, CV_8UC3, cv::Scalar::all(0));
    // nv_panorama-renderer samples projected cameras through
    // getColorSubpixb<InterpolationMode::Bilinear> (enum value 1).
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& rows) {
        for (int row = rows.start; row < rows.end; ++row) {
            for (int column = 0; column < width; ++column) {
                if (mask.at<std::uint8_t>(row, column) == 0U) {
                    continue;
                }
                warped.at<cv::Vec3b>(row, column) = sampleVendorBilinearBgr(
                    source, map_x.at<float>(row, column), map_y.at<float>(row, column));
            }
        }
    });
    warped.setTo(cv::Scalar::all(0), mask == 0);
    return {warped, mask};
}

cv::Mat stabilizeNadir(const cv::Mat& image) {
    // The camera rig has no usable observations directly below the head. The
    // reference renderer replaces that device/stand region with a smooth,
    // scene-colored cap. Telea inpainting alone creates conspicuous vertical
    // streaks in an equirectangular image, so derive a low-frequency cap from
    // the last well-observed floor band and blend it in with a smooth ramp.
    const int band_begin = std::clamp(static_cast<int>(image.rows * 0.765), 0, image.rows - 1);
    const int band_end = std::clamp(static_cast<int>(image.rows * 0.795), band_begin + 1, image.rows);
    cv::Mat band_float;
    image(cv::Range(band_begin, band_end), cv::Range::all()).convertTo(band_float, CV_32FC3);
    cv::Mat profile;
    cv::reduce(band_float, profile, 0, cv::REDUCE_AVG, CV_32FC3);

    // Blur across the longitude seam with explicit circular padding.
    cv::Mat circular;
    cv::hconcat(std::vector<cv::Mat>{profile, profile, profile}, circular);
    int kernel = std::max(33, (image.cols / 8) | 1);
    cv::GaussianBlur(circular, circular, cv::Size(kernel, 1), 0.0, 0.0, cv::BORDER_REPLICATE);
    profile = circular(cv::Rect(image.cols, 0, image.cols, 1)).clone() * 0.92F;

    cv::Mat cap;
    cv::repeat(profile, image.rows, 1, cap);
    cv::Mat output_float;
    image.convertTo(output_float, CV_32FC3);
    const int transition_begin = std::clamp(static_cast<int>(image.rows * 0.79), 0, image.rows - 1);
    const int transition_end = std::clamp(static_cast<int>(image.rows * 0.89), transition_begin + 1, image.rows);
    for (int row = transition_begin; row < image.rows; ++row) {
        float alpha = std::clamp(
            static_cast<float>(row - transition_begin) /
                static_cast<float>(transition_end - transition_begin),
            0.0F, 1.0F);
        alpha = alpha * alpha * (3.0F - 2.0F * alpha);
        cv::Mat destination = output_float.row(row);
        cv::addWeighted(destination, 1.0F - alpha, cap.row(row), alpha, 0.0, destination);
    }
    cv::Mat output;
    output_float.convertTo(output, CV_8UC3);
    return output;
}

cv::Mat circularShiftHorizontal(const cv::Mat& image, int shift) {
    if (image.empty()) {
        return image.clone();
    }
    shift %= image.cols;
    if (shift < 0) {
        shift += image.cols;
    }
    if (shift == 0) {
        return image.clone();
    }
    cv::Mat result(image.size(), image.type());
    image.colRange(0, image.cols - shift).copyTo(result.colRange(shift, image.cols));
    image.colRange(image.cols - shift, image.cols).copyTo(result.colRange(0, shift));
    return result;
}

void prepareBinaryPairwiseMasks(std::vector<cv::UMat>& masks) {
    if (masks.size() != 4U) {
        throw std::runtime_error("Pairwise mask preparation requires four cameras");
    }
    const int width = masks.front().cols;
    const int height = masks.front().rows;
    const int quarter = width / 4;
    const int half = width / 2;
    // MaskPreparerVlxAndMira builds a rectangular erosion kernel from one
    // percent of the seam canvas in each dimension.  At the fixed 2048x1024
    // seam resolution this is the asymmetric 20x10 kernel used by the
    // renderer (integer conversion truncates both products).
    const int erosion_width = std::max(
        1, static_cast<int>(static_cast<float>(width) * 0.01F));
    const int erosion_height = std::max(
        1, static_cast<int>(static_cast<float>(height) * 0.01F));
    const cv::Mat erosion_kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(erosion_width, erosion_height));
    for (std::size_t index = 0; index < masks.size(); ++index) {
        cv::Mat prepared = masks[index].getMat(cv::ACCESS_READ).clone();
        cv::erode(prepared, prepared, erosion_kernel);
        cv::Mat angular_window(height, width, CV_8U, cv::Scalar(0));
        // Recovered VLX/MIRA seam-preparer windows at 2048 px are
        // [0,1023], [492,1515], [1024,2047], and the wrapped complement.
        // The 20 px cam1/cam3 offset is calibration-specific and scales with
        // panorama width.
        const int calibration_offset = static_cast<int>(std::lround(
            20.0 * static_cast<double>(width) / 2048.0));
        if (index == 0U) {
            angular_window.colRange(0, half).setTo(255);
        } else if (index == 1U) {
            const int begin = quarter - calibration_offset;
            angular_window.colRange(begin, begin + half).setTo(255);
        } else if (index == 2U) {
            // The angular interval is open at the -X meridian. Camera 0 owns
            // column 1024's opposite boundary, so MIRA starts at 1025 on the
            // fixed 2K seam canvas.
            angular_window.colRange(half + 1, width).setTo(255);
        } else {
            const int begin = quarter * 3 - calibration_offset;
            angular_window.colRange(begin, width).setTo(255);
            angular_window.colRange(0, quarter - calibration_offset - 1).setTo(255);
        }
        cv::bitwise_and(prepared, angular_window, prepared);
        masks[index] = prepared.getUMat(cv::ACCESS_RW);
    }
}

void findPairwiseCircularSeams(
    const std::vector<cv::UMat>& images, std::vector<cv::UMat>& masks) {
    if (images.size() != 4U || masks.size() != 4U) {
        throw std::runtime_error("Pairwise circular seam mode requires four cameras");
    }
    struct Pair {
        std::size_t first;
        std::size_t second;
        double horizontal_shift;
    };
    // Recovered from the renderer's seam-pair records.  The shift is applied
    // to both equirectangular inputs so the selected overlap does not cross
    // the left/right image boundary.
    constexpr std::array<Pair, 4> pairs{{
        {0U, 1U, 0.0},
        {1U, 2U, 0.0},
        {2U, 3U, 0.6},
        {0U, 3U, 0.4},
    }};
    const std::vector<cv::Point> pair_corners(2, cv::Point(0, 0));
    for (const Pair& pair : pairs) {
        const int shift = static_cast<int>(
            pair.horizontal_shift * static_cast<double>(images.front().cols));
        const cv::Mat first_image = circularShiftHorizontal(
            images[pair.first].getMat(cv::ACCESS_READ), shift);
        const cv::Mat second_image = circularShiftHorizontal(
            images[pair.second].getMat(cv::ACCESS_READ), shift);
        cv::Mat first_mask = circularShiftHorizontal(
            masks[pair.first].getMat(cv::ACCESS_READ), shift);
        cv::Mat second_mask = circularShiftHorizontal(
            masks[pair.second].getMat(cv::ACCESS_READ), shift);
        std::vector<cv::UMat> pair_images{
            first_image.getUMat(cv::ACCESS_READ),
            second_image.getUMat(cv::ACCESS_READ),
        };
        std::vector<cv::UMat> pair_masks{
            first_mask.getUMat(cv::ACCESS_RW),
            second_mask.getUMat(cv::ACCESS_RW),
        };
        cv::Ptr<cv::detail::SeamFinder> finder =
            cv::makePtr<cv::detail::GraphCutSeamFinder>(
                cv::detail::GraphCutSeamFinderBase::COST_COLOR);
        finder->find(pair_images, pair_corners, pair_masks);
        masks[pair.first] = circularShiftHorizontal(
            pair_masks[0].getMat(cv::ACCESS_READ), -shift).getUMat(cv::ACCESS_RW);
        masks[pair.second] = circularShiftHorizontal(
            pair_masks[1].getMat(cv::ACCESS_READ), -shift).getUMat(cv::ACCESS_RW);
    }
}

void reduceValid2x2(
    const cv::Mat& source, const cv::Mat& source_mask,
    cv::Mat& reduced, cv::Mat& reduced_mask) {
    const int rows = source.rows / 2;
    const int columns = source.cols / 2;
    reduced = cv::Mat(rows, columns, CV_32FC3, cv::Scalar::all(0));
    reduced_mask = cv::Mat(rows, columns, CV_8UC1, cv::Scalar::all(0));

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            const cv::Vec3f* top = source.ptr<cv::Vec3f>(2 * row);
            const cv::Vec3f* bottom = source.ptr<cv::Vec3f>(2 * row + 1);
            const std::uint8_t* top_mask = source_mask.ptr<std::uint8_t>(2 * row);
            const std::uint8_t* bottom_mask =
                source_mask.ptr<std::uint8_t>(2 * row + 1);
            cv::Vec3f* destination = reduced.ptr<cv::Vec3f>(row);
            std::uint8_t* destination_mask = reduced_mask.ptr<std::uint8_t>(row);

            for (int column = 0; column < columns; ++column) {
                const int source_column = 2 * column;
                cv::Vec3f sum(0.0F, 0.0F, 0.0F);
                int count = 0;
                if (top_mask[source_column] != 0) {
                    sum += top[source_column];
                    ++count;
                }
                if (top_mask[source_column + 1] != 0) {
                    sum += top[source_column + 1];
                    ++count;
                }
                if (bottom_mask[source_column] != 0) {
                    sum += bottom[source_column];
                    ++count;
                }
                if (bottom_mask[source_column + 1] != 0) {
                    sum += bottom[source_column + 1];
                    ++count;
                }
                if (count != 0) {
                    destination[column] =
                        sum * (1.0F / static_cast<float>(count));
                    destination_mask[column] = 255;
                }
            }
        }
    });
}

cv::Mat pyramidInpaint(
    const cv::Mat& input, const cv::Mat& input_mask, bool horizontal_wrap) {
    if (input.type() != CV_8UC3 || input_mask.type() != CV_8UC1 ||
        input.size() != input_mask.size()) {
        throw std::runtime_error(
            "Pyramid inpainting expects equally sized CV_8UC3/CV_8UC1 inputs");
    }
    if (horizontal_wrap && (input.cols & 1) != 0) {
        throw std::runtime_error(
            "Wrapped pyramid inpainting requires an even panorama width");
    }

    cv::Mat image = input;
    cv::Mat mask = input_mask;
    int horizontal_padding = 0;
    if (horizontal_wrap) {
        // The floor call wraps half a panorama on each side before building
        // its valid-aware float pyramid.  There is no vertical padding.
        horizontal_padding = input.cols / 2;
        cv::copyMakeBorder(
            input, image, 0, 0, horizontal_padding, horizontal_padding,
            cv::BORDER_WRAP);
        cv::copyMakeBorder(
            input_mask, mask, 0, 0, horizontal_padding, horizontal_padding,
            cv::BORDER_WRAP);
    }

    std::vector<cv::Mat> image_pyramid;
    std::vector<cv::Mat> mask_pyramid;
    cv::Mat image_float;
    image.convertTo(image_float, CV_32FC3);
    image_pyramid.push_back(std::move(image_float));
    mask_pyramid.push_back(std::move(mask));

    // The full-source check is deliberately made before reduction, but the
    // child is still computed.  The first child of a fully valid source is
    // discarded, matching the recovered vendor call sequence.
    while (image_pyramid.back().rows > 1 && image_pyramid.back().cols > 1) {
        const bool source_fully_valid =
            static_cast<std::size_t>(cv::countNonZero(mask_pyramid.back())) ==
            mask_pyramid.back().total();
        cv::Mat reduced;
        cv::Mat reduced_mask;
        reduceValid2x2(
            image_pyramid.back(), mask_pyramid.back(), reduced, reduced_mask);
        if (source_fully_valid) {
            break;
        }
        image_pyramid.push_back(std::move(reduced));
        mask_pyramid.push_back(std::move(reduced_mask));
    }

    cv::Mat filled = image_pyramid.back().clone();
    for (std::size_t level = image_pyramid.size() - 1; level-- > 0;) {
        cv::Mat upsampled;
        cv::resize(
            filled, upsampled, image_pyramid[level].size(), 0.0, 0.0,
            cv::INTER_LINEAR);
        image_pyramid[level].copyTo(upsampled, mask_pyramid[level]);
        filled = std::move(upsampled);
    }

    cv::Mat quantized;
    filled.convertTo(quantized, CV_8UC3);
    if (!horizontal_wrap) {
        return quantized;
    }
    return quantized.colRange(
        horizontal_padding, horizontal_padding + input.cols).clone();
}

cv::Mat padEquirectangular(const cv::Mat& source, int padding = 2) {
    cv::Mat vertical;
    cv::copyMakeBorder(
        source, vertical, padding, padding, 0, 0, cv::BORDER_REFLECT);
    cv::Mat padded;
    cv::copyMakeBorder(
        vertical, padded, 0, 0, padding, padding, cv::BORDER_WRAP);
    return padded;
}

cv::Mat equirectangularPyrDown(const cv::Mat& source) {
    cv::Mat filtered;
    cv::pyrDown(padEquirectangular(source), filtered);

    // Keep this as an ROI instead of cloning it.  The next copyMakeBorder call
    // deliberately sees the pixels outside this ROI in the filtered parent.
    // That OpenCV submatrix behaviour is part of the renderer's polar-boundary
    // rule and becomes visible in the coarsest pyramid levels.
    return filtered(cv::Rect(1, 1, source.cols / 2, source.rows / 2));
}

cv::Mat equirectangularPyrUp(
    const cv::Mat& source, cv::Size destination_size) {
    cv::Mat filtered;
    cv::pyrUp(padEquirectangular(source), filtered);
    return filtered(cv::Rect(
        4, 4, destination_size.width, destination_size.height));
}

cv::Mat convertPyramidLevelPreservingParent(
    const cv::Mat& source, int destination_type) {
    cv::Size whole_size;
    cv::Point offset;
    source.locateROI(whole_size, offset);
    cv::Mat source_parent(
        whole_size, source.type(), const_cast<uchar*>(source.datastart),
        source.step[0]);
    cv::Mat converted_parent;
    source_parent.convertTo(converted_parent, destination_type);
    return converted_parent(cv::Rect(offset, source.size()));
}

struct BlendPyramids {
    std::vector<cv::Mat> image;
    std::vector<cv::Mat> weight;
};

BlendPyramids buildBlendPyramids(
    const cv::Mat& image, const cv::Mat& seam_mask, int number_of_bands) {
    cv::Mat weight_f32;
    seam_mask.convertTo(weight_f32, CV_32FC1, 1.0 / 255.0);

    // The image Gaussian pyramid is filtered in uint8.  Conversion to signed
    // 16-bit happens only after downsampling, and includes the hidden parent
    // border of every ROI.  Filtering a CV_16S pyramid is close visually but
    // does not reproduce the renderer's integer rounding.
    std::vector<cv::Mat> gaussian_images{image};
    std::vector<cv::Mat> weights{weight_f32};
    for (int level = 0; level < number_of_bands; ++level) {
        gaussian_images.push_back(equirectangularPyrDown(gaussian_images.back()));
        weights.push_back(equirectangularPyrDown(weights.back()));
    }

    std::vector<cv::Mat> laplacians;
    laplacians.reserve(number_of_bands + 1);
    for (int level = 0; level < number_of_bands; ++level) {
        const cv::Mat current = convertPyramidLevelPreservingParent(
            gaussian_images[level], CV_16SC3);
        const cv::Mat reduced = convertPyramidLevelPreservingParent(
            gaussian_images[level + 1], CV_16SC3);
        laplacians.push_back(
            current - equirectangularPyrUp(
                reduced, gaussian_images[level].size()));
    }
    laplacians.push_back(convertPyramidLevelPreservingParent(
        gaussian_images.back(), CV_16SC3));
    return {std::move(laplacians), std::move(weights)};
}

struct BlendResult {
    cv::Mat image;
    cv::Mat valid_mask;
};

BlendResult blendEquirectangular(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& seam_masks,
    const std::vector<cv::Mat>& projection_masks,
    int number_of_bands) {
    if (images.empty() || images.size() != seam_masks.size() ||
        images.size() != projection_masks.size()) {
        throw std::runtime_error("Multi-band blend inputs have inconsistent sizes");
    }

    std::vector<cv::Mat> accumulated_images;
    std::vector<cv::Mat> accumulated_weights;
    accumulated_images.reserve(number_of_bands + 1);
    accumulated_weights.reserve(number_of_bands + 1);

    // Camera order and OpenCV's float32 expression order are both observable
    // in the frozen intermediate tensors.  Build and consume one camera at a
    // time: retaining all four image/weight pyramids adds several GiB at 8K
    // without affecting the accumulation order or ROI-parent semantics.
    for (std::size_t camera = 0; camera < images.size(); ++camera) {
        const BlendPyramids pyramid = buildBlendPyramids(
            images[camera], seam_masks[camera], number_of_bands);
        if (camera == 0U) {
            for (int level = 0; level <= number_of_bands; ++level) {
                accumulated_images.emplace_back(
                    pyramid.image[level].size(), CV_32FC3,
                    cv::Scalar::all(0));
                accumulated_weights.emplace_back(
                    pyramid.weight[level].size(), CV_32FC1,
                    cv::Scalar::all(0));
            }
        }
        for (int level = 0; level <= number_of_bands; ++level) {
            cv::Mat image_float;
            pyramid.image[level].convertTo(image_float, CV_32FC3);
            cv::Mat weight_channels[] = {
                pyramid.weight[level], pyramid.weight[level],
                pyramid.weight[level]};
            cv::Mat weight3;
            cv::merge(weight_channels, 3, weight3);
            accumulated_images[level] += image_float.mul(weight3);
            accumulated_weights[level] += pyramid.weight[level];
        }
    }

    // The binary normalization worker computes one float32 reciprocal from
    // each scalar weight and multiplies the three float32 image channels by
    // it.  Do that directly in the accumulated storage.  Materializing a
    // three-channel denominator, its reciprocal, and a separate output would
    // add more than a GiB of simultaneous temporaries at level zero in 8K.
    for (int level = 0; level <= number_of_bands; ++level) {
        cv::Mat& image = accumulated_images[level];
        const cv::Mat& weight = accumulated_weights[level];
        cv::parallel_for_(cv::Range(0, image.rows), [&](const cv::Range& rows) {
            for (int row = rows.start; row < rows.end; ++row) {
                cv::Vec3f* image_row = image.ptr<cv::Vec3f>(row);
                const float* weight_row = weight.ptr<float>(row);
                for (int column = 0; column < image.cols; ++column) {
                    const float inverse_weight =
                        1.0F / (weight_row[column] + 1.0e-5F);
                    image_row[column][0] *= inverse_weight;
                    image_row[column][1] *= inverse_weight;
                    image_row[column][2] *= inverse_weight;
                }
            }
        });
        accumulated_weights[level].release();
    }
    accumulated_weights.clear();

    cv::Mat reconstructed = accumulated_images.back();
    for (int level = number_of_bands - 1; level >= 0; --level) {
        reconstructed = equirectangularPyrUp(
            reconstructed, accumulated_images[level].size());
        // Add into the ROI itself.  A Mat expression would allocate a compact
        // result and discard the four-pixel parent border needed by the next
        // level's non-isolated copyMakeBorder call.
        cv::parallel_for_(
            cv::Range(0, reconstructed.rows), [&](const cv::Range& rows) {
                for (int row = rows.start; row < rows.end; ++row) {
                    cv::Vec3f* destination =
                        reconstructed.ptr<cv::Vec3f>(row);
                    const cv::Vec3f* addition =
                        accumulated_images[level].ptr<cv::Vec3f>(row);
                    for (int column = 0; column < reconstructed.cols; ++column) {
                        destination[column] += addition[column];
                    }
                }
            });
    }

    cv::Mat blended;
    reconstructed.convertTo(blended, CV_8UC3);
    cv::Mat valid_mask(images.front().size(), CV_8UC1, cv::Scalar::all(0));
    for (const cv::Mat& projection_mask : projection_masks) {
        cv::bitwise_or(valid_mask, projection_mask, valid_mask);
    }
    blended.setTo(cv::Scalar::all(0), valid_mask == 0);
    return {std::move(blended), std::move(valid_mask)};
}

cv::Mat stitch(
    std::vector<WarpedCamera> cameras,
    std::vector<WarpedCamera> seam_control_cameras,
    int width, int height,
    const cv::Mat& operator_mask, const fs::path& debug_directory,
    const std::string& nadir_mode, const std::string& exposure_mode,
    const std::string& seam_mode, int jpeg_quality) {
    std::cerr << "Stitch: exposure compensation\n";
    const bool has_separate_seam_control = !seam_control_cameras.empty();
    if (has_separate_seam_control &&
        seam_control_cameras.size() != cameras.size()) {
        throw std::runtime_error(
            "Seam-control projections must match the camera count");
    }
    const std::vector<cv::Point> corners(cameras.size(), cv::Point(0, 0));
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> masks;
    for (std::size_t index = 0; index < cameras.size(); ++index) {
        if (!debug_directory.empty()) {
            fs::create_directories(debug_directory);
            cv::imwrite(
                (debug_directory / ("warped-cam" + std::to_string(index) + ".jpg")).string(),
                cameras[index].image,
                {cv::IMWRITE_JPEG_QUALITY, 95});
            cv::imwrite(
                (debug_directory / ("warped-cam" + std::to_string(index) + ".png")).string(),
                cameras[index].image);
            cv::imwrite(
                (debug_directory /
                 ("projection-mask-cam" + std::to_string(index) + ".png")).string(),
                cameras[index].mask);
        }
        images.push_back(std::move(cameras[index].image));
        masks.push_back(std::move(cameras[index].mask));
    }
    cameras.clear();

    std::vector<cv::Mat> seam_control_images;
    std::vector<cv::Mat> seam_control_masks;
    if (has_separate_seam_control) {
        seam_control_images.reserve(seam_control_cameras.size());
        seam_control_masks.reserve(seam_control_cameras.size());
        for (std::size_t index = 0; index < seam_control_cameras.size(); ++index) {
            if (!debug_directory.empty()) {
                cv::imwrite(
                    (debug_directory /
                     ("seam-control-warped-cam" + std::to_string(index) + ".png"))
                        .string(),
                    seam_control_cameras[index].image);
                cv::imwrite(
                    (debug_directory /
                     ("seam-control-projection-mask-cam" +
                      std::to_string(index) + ".png"))
                        .string(),
                    seam_control_cameras[index].mask);
            }
            seam_control_images.push_back(
                std::move(seam_control_cameras[index].image));
            seam_control_masks.push_back(
                std::move(seam_control_cameras[index].mask));
        }
        seam_control_cameras.clear();
    }

    if (exposure_mode == "soft") {
        std::vector<cv::Mat3b> exposure_images;
        std::vector<cv::Mat1b> exposure_masks;
        exposure_images.reserve(images.size());
        exposure_masks.reserve(images.size());
        for (std::size_t index = 0; index < images.size(); ++index) {
            const cv::Mat& source_image = has_separate_seam_control
                ? seam_control_images[index]
                : images[index];
            const cv::Mat& source_mask = has_separate_seam_control
                ? seam_control_masks[index]
                : masks[index];
            cv::Mat exposure_image;
            cv::Mat exposure_mask;
            cv::resize(
                source_image, exposure_image, cv::Size(2048, 1024),
                0.0, 0.0, cv::INTER_AREA);
            cv::resize(
                source_mask, exposure_mask, cv::Size(2048, 1024),
                0.0, 0.0, cv::INTER_NEAREST);
            cv::Mat3b reduced_image;
            cv::Mat1b reduced_mask;
            cv::pyrDown(exposure_image, reduced_image);
            cv::pyrDown(exposure_mask, reduced_mask);
            if (!debug_directory.empty()) {
                cv::imwrite(
                    (debug_directory /
                     ("exposure-input-image-cam" + std::to_string(index) + ".png")).string(),
                    reduced_image);
                cv::imwrite(
                    (debug_directory /
                     ("exposure-input-mask-cam" + std::to_string(index) + ".png")).string(),
                    reduced_mask);
            }
            exposure_images.push_back(std::move(reduced_image));
            exposure_masks.push_back(std::move(reduced_mask));
        }
        const std::vector<cv::Vec3f> gains =
            navvis_recon::ExposureCompensatorSoftConstraint::estimateGains(
                exposure_images, exposure_masks);
        for (std::size_t index = 0; index < images.size(); ++index) {
            std::cerr << "Exposure gain cam" << index << " BGR=["
                      << gains[index][0] << ", " << gains[index][1] << ", "
                      << gains[index][2] << "]\n";
            cv::multiply(images[index], cv::Scalar(
                gains[index][0], gains[index][1], gains[index][2]),
                images[index], 1.0, CV_8UC3);
            if (has_separate_seam_control) {
                cv::multiply(
                    seam_control_images[index],
                    cv::Scalar(
                        gains[index][0], gains[index][1], gains[index][2]),
                    seam_control_images[index], 1.0, CV_8UC3);
            }
            if (!debug_directory.empty()) {
                cv::imwrite(
                    (debug_directory /
                     ("exposure-cam" + std::to_string(index) + ".png")).string(),
                    images[index]);
                if (has_separate_seam_control) {
                    cv::imwrite(
                        (debug_directory /
                         ("seam-control-exposure-cam" +
                          std::to_string(index) + ".png"))
                            .string(),
                        seam_control_images[index]);
                }
            }
        }
    } else if (exposure_mode == "opencv") {
        if (has_separate_seam_control) {
            throw std::runtime_error(
                "Separate seam-control projections require soft or no exposure");
        }
        std::vector<cv::UMat> images_umat;
        std::vector<cv::UMat> masks_umat;
        for (std::size_t index = 0; index < images.size(); ++index) {
            images_umat.push_back(images[index].getUMat(cv::ACCESS_RW));
            masks_umat.push_back(masks[index].getUMat(cv::ACCESS_READ));
        }
        // OpenCV 4.5's block compensator allocates a pathological 128 GiB
        // matrix for an 8K canvas, so use one gain per camera at 8K.
        const int compensator_type = width > 2048
            ? cv::detail::ExposureCompensator::GAIN
            : cv::detail::ExposureCompensator::GAIN_BLOCKS;
        cv::Ptr<cv::detail::ExposureCompensator> compensator =
            cv::detail::ExposureCompensator::createDefault(compensator_type);
        compensator->feed(corners, images_umat, masks_umat);
        for (std::size_t index = 0; index < images.size(); ++index) {
            compensator->apply(
                static_cast<int>(index), corners[index], images[index], masks[index]);
        }
    } else if (exposure_mode != "none") {
        throw std::runtime_error("Unsupported --exposure-mode: " + exposure_mode);
    }

    // The vendor computes graph-cut seams from the exposure-compensated
    // projections.  Exposure overlap statistics still use the unmodified
    // projection masks above.
    std::cerr << "Stitch: seam estimation\n";
    std::vector<cv::UMat> seam_images;
    std::vector<cv::UMat> seam_masks;
    // Graph-cut storage grows much faster than the panorama pixel count in
    // OpenCV 4.5.  The renderer always uses a 2048x1024 seam canvas, lifting
    // its binary result to the native panorama only after seam estimation.
    const std::vector<cv::Mat>& seam_source_images = has_separate_seam_control
        ? seam_control_images
        : images;
    const std::vector<cv::Mat>& seam_source_masks = has_separate_seam_control
        ? seam_control_masks
        : masks;
    const double seam_scale = std::min(
        1.0, 2048.0 /
            static_cast<double>(seam_source_images.front().cols));
    for (std::size_t index = 0; index < images.size(); ++index) {
        cv::Mat image_float;
        seam_source_images[index].convertTo(image_float, CV_32FC3);
        cv::Mat seam_image;
        cv::resize(
            image_float, seam_image, cv::Size(), seam_scale, seam_scale,
            cv::INTER_AREA);
        seam_images.push_back(seam_image.getUMat(cv::ACCESS_READ));

        cv::Mat seam_mask;
        cv::resize(
            seam_source_masks[index], seam_mask, seam_image.size(), 0.0, 0.0,
            cv::INTER_NEAREST);
        seam_masks.push_back(seam_mask.getUMat(cv::ACCESS_RW));
    }
    if (seam_mode == "pairwise") {
        prepareBinaryPairwiseMasks(seam_masks);
        findPairwiseCircularSeams(seam_images, seam_masks);
    } else if (seam_mode == "global") {
        cv::Ptr<cv::detail::SeamFinder> seam_finder =
            cv::makePtr<cv::detail::GraphCutSeamFinder>(
                cv::detail::GraphCutSeamFinderBase::COST_COLOR);
        seam_finder->find(seam_images, corners, seam_masks);
    } else {
        throw std::runtime_error("Unsupported --seam-mode: " + seam_mode);
    }

    std::vector<cv::Mat> optimized_masks;
    optimized_masks.reserve(seam_masks.size());
    for (std::size_t index = 0; index < seam_masks.size(); ++index) {
        const cv::Mat small_mask = seam_masks[index].getMat(cv::ACCESS_READ);
        cv::Mat full_mask;
        // The renderer lifts the fixed-size graph-cut mask with linear
        // interpolation and then applies a strict binary threshold.
        cv::resize(
            small_mask, full_mask, cv::Size(width, height), 0.0, 0.0,
            cv::INTER_LINEAR);
        cv::threshold(full_mask, full_mask, 127.0, 255.0, cv::THRESH_BINARY);
        cv::bitwise_and(full_mask, masks[index], full_mask);
        optimized_masks.push_back(full_mask);
        if (!debug_directory.empty()) {
            cv::imwrite(
                (debug_directory /
                 ("seam-mask-cam" + std::to_string(index) + ".png"))
                    .string(),
                full_mask);
        }
    }

    std::cerr << "Stitch: multi-band blending\n";
    std::vector<cv::Mat> prepared_images;
    prepared_images.reserve(images.size());
    for (std::size_t index = 0; index < images.size(); ++index) {
        prepared_images.push_back(
            pyramidInpaint(images[index], masks[index], false));
        if (!debug_directory.empty()) {
            cv::imwrite(
                (debug_directory /
                 ("seam-prepared-cam" + std::to_string(index) + ".png"))
                    .string(),
                prepared_images.back());
        }
    }
    // The installed renderer uses ten levels at both 2K and 8K: nine
    // Laplacian bands followed by one residual image.
    BlendResult blend = blendEquirectangular(
        prepared_images, optimized_masks, masks, 9);
    cv::Mat blended = std::move(blend.image);
    cv::Mat blended_mask = std::move(blend.valid_mask);
    if (!debug_directory.empty()) {
        cv::imwrite(
            (debug_directory / "multiband-output.png").string(), blended);
        cv::imwrite(
            (debug_directory / "multiband-valid-mask.png").string(),
            blended_mask);
    }
    // The exact floor pyramid is the dominant 8K allocation.  None of the
    // camera, seam, or blend-input matrices are needed after this point, so
    // release their backing stores before entering the floor stage.
    prepared_images.clear();
    optimized_masks.clear();
    seam_images.clear();
    seam_masks.clear();
    seam_control_images.clear();
    seam_control_masks.clear();
    images.clear();
    masks.clear();
    if (!operator_mask.empty()) {
        cv::Mat resized_operator_mask;
        cv::resize(
            operator_mask, resized_operator_mask, cv::Size(width, height),
            0.0, 0.0, cv::INTER_NEAREST);
        if (resized_operator_mask.type() != CV_8U) {
            throw std::runtime_error("--operator-mask must be single-channel");
        }
        cv::bitwise_and(blended_mask, resized_operator_mask, blended_mask);
        blended.setTo(cv::Scalar::all(0), resized_operator_mask == 0);
    }
    cv::Mat floor_mask;
    cv::threshold(blended_mask, floor_mask, 0.0, 1.0, cv::THRESH_BINARY);

    const int output_quality = std::clamp(jpeg_quality, 35, 100);
    const std::vector<int> no_floor_jpeg_parameters{
        cv::IMWRITE_JPEG_QUALITY, output_quality,
        cv::IMWRITE_JPEG_OPTIMIZE, 0,
    };
    cv::Mat floor_input;
    if (!debug_directory.empty()) {
        fs::create_directories(debug_directory);
        const fs::path binary_mask_path =
            debug_directory / "binary-mask-before-floor.png";
        const fs::path no_floor_path =
            debug_directory / "panorama-before-floor.jpg";
        if (!cv::imwrite(
                binary_mask_path.string(), floor_mask,
                {cv::IMWRITE_PNG_COMPRESSION, 9})) {
            throw std::runtime_error(
                "Failed to write binary panorama mask: " +
                binary_mask_path.string());
        }
        if (!cv::imwrite(
                no_floor_path.string(), blended, no_floor_jpeg_parameters)) {
            throw std::runtime_error(
                "Failed to write no-floor panorama: " + no_floor_path.string());
        }
        floor_input = cv::imread(
            no_floor_path.string(),
            cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    } else {
        std::vector<std::uint8_t> encoded;
        if (!cv::imencode(
                ".jpg", blended, encoded, no_floor_jpeg_parameters)) {
            throw std::runtime_error("Failed to encode no-floor panorama");
        }
        floor_input = cv::imdecode(
            encoded, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    }
    if (floor_input.empty()) {
        throw std::runtime_error("Failed to decode no-floor panorama JPEG");
    }
    if (nadir_mode == "none") {
        return blended;
    }
    if (nadir_mode == "pyramid") {
        std::cerr << "Stitch: mask-driven pyramid nadir fill\n";
        if (!debug_directory.empty()) {
            cv::imwrite(
                (debug_directory / "floor-input.png").string(), floor_input);
        }
        cv::Mat floor_output = pyramidInpaint(floor_input, floor_mask, true);
        if (!debug_directory.empty()) {
            cv::imwrite(
                (debug_directory / "floor-output.png").string(), floor_output);
        }
        return floor_output;
    }
    if (nadir_mode != "stabilize") {
        throw std::runtime_error("Unsupported --nadir-mode: " + nadir_mode);
    }
    std::cerr << "Stitch: nadir fill\n";
    return stabilizeNadir(floor_input);
}

Options parseOptions(int argc, char** argv) {
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument.rfind("--", 0) != 0 || index + 1 >= argc) {
            throw std::runtime_error("Expected --key value, got: " + argument);
        }
        values[argument.substr(2)] = argv[++index];
    }
    Options options;
    const bool standalone_nadir = values.count("blended-input") != 0U;
    if (!values.count("output") ||
        (!standalone_nadir &&
         (!values.count("sensor-frame") ||
          (!values.count("input-dir") && !values.count("processed-camera-dir") &&
           !values.count("projected-camera-dir"))))) {
        throw std::runtime_error(
            "Usage: navvis_recon_ocam_panorama --sensor-frame FILE "
            "(--input-dir DIR | --processed-camera-dir DIR | --projected-camera-dir DIR) "
            "--output FILE [--capture 00000] [--width 2048] [--jpeg-quality 95] "
            "[--opencv-threads N] "
            "[--metadata-dir ORIGINAL_DNG_DIR] "
            "or --blended-input FILE --valid-mask FILE --output FILE "
            "[--camera-mask-dir DIR] [--depth-map FILE | --world-map F32C3] "
            "[--seam-world-map 2048x1024_F32C3] "
            "[--surface-cloud FILE --panorama-info FILE] "
            "[--depth-translation-mode head-minus|head-plus|camera-minus|camera-plus] "
            "[--operator-mask FILE] [--nadir-mode pyramid|stabilize|none] "
            "[--seam-mode pairwise|global] [--exposure-mode soft|opencv|none] "
            "[--debug-dir DIR] [--camera-only true] [--selected-camera cam0]");
    }
    options.sensor_frame = values["sensor-frame"];
    if (values.count("input-dir")) options.input_directory = values["input-dir"];
    if (values.count("metadata-dir")) options.metadata_directory = values["metadata-dir"];
    if (values.count("processed-camera-dir")) {
        options.processed_camera_directory = values["processed-camera-dir"];
    }
    if (values.count("projected-camera-dir")) {
        options.projected_camera_directory = values["projected-camera-dir"];
    }
    if (values.count("blended-input")) options.blended_input = values["blended-input"];
    if (values.count("valid-mask")) options.valid_mask = values["valid-mask"];
    options.output = values["output"];
    if (values.count("decoded-dir")) options.decoded_directory = values["decoded-dir"];
    if (values.count("camera-mask-dir")) {
        options.camera_mask_directory = values["camera-mask-dir"];
    }
    if (values.count("operator-mask")) options.operator_mask = values["operator-mask"];
    if (values.count("depth-map")) options.depth_map = values["depth-map"];
    if (values.count("world-map")) options.world_map = values["world-map"];
    if (values.count("seam-world-map")) {
        options.seam_world_map = values["seam-world-map"];
    }
    if (values.count("surface-cloud")) options.surface_cloud = values["surface-cloud"];
    if (values.count("panorama-info")) options.panorama_info = values["panorama-info"];
    if (values.count("debug-dir")) options.debug_directory = values["debug-dir"];
    if (values.count("capture")) options.capture = values["capture"];
    if (values.count("nadir-mode")) options.nadir_mode = values["nadir-mode"];
    if (values.count("exposure-mode")) options.exposure_mode = values["exposure-mode"];
    if (values.count("seam-mode")) options.seam_mode = values["seam-mode"];
    if (values.count("depth-translation-mode")) {
        options.depth_translation_mode = values["depth-translation-mode"];
    }
    if (values.count("width")) options.width = std::stoi(values["width"]);
    if (values.count("jpeg-quality")) options.jpeg_quality = std::stoi(values["jpeg-quality"]);
    if (values.count("opencv-threads")) {
        options.opencv_threads = std::stoi(values["opencv-threads"]);
    }
    if (values.count("camera-only")) {
        options.camera_only = values["camera-only"] == "true" || values["camera-only"] == "1";
    }
    if (values.count("selected-camera")) options.selected_camera = values["selected-camera"];
    if (!options.selected_camera.empty() && !options.camera_only) {
        throw std::runtime_error("--selected-camera requires --camera-only true");
    }
    if (!options.processed_camera_directory.empty() &&
        (!options.decoded_directory.empty() || options.camera_only ||
         !options.selected_camera.empty())) {
        throw std::runtime_error(
            "--processed-camera-dir cannot be combined with camera-output options");
    }
    if (!options.projected_camera_directory.empty() &&
        (!options.input_directory.empty() || !options.processed_camera_directory.empty() ||
         !options.decoded_directory.empty() || options.camera_only ||
         !options.selected_camera.empty())) {
        throw std::runtime_error(
            "--projected-camera-dir is an isolated stitching input");
    }
    if (!options.blended_input.empty() && options.valid_mask.empty()) {
        throw std::runtime_error("--blended-input requires --valid-mask");
    }
    if (options.surface_cloud.empty() != options.panorama_info.empty()) {
        throw std::runtime_error(
            "--surface-cloud and --panorama-info must be provided together");
    }
    const int depth_input_count =
        static_cast<int>(!options.depth_map.empty()) +
        static_cast<int>(!options.world_map.empty()) +
        static_cast<int>(!options.surface_cloud.empty());
    if (depth_input_count > 1) {
        throw std::runtime_error(
            "--depth-map, --world-map and --surface-cloud are mutually exclusive");
    }
    if (!options.world_map.empty() &&
        options.depth_translation_mode != "head-minus") {
        throw std::runtime_error(
            "--world-map requires --depth-translation-mode head-minus");
    }
    if (!options.seam_world_map.empty() && options.world_map.empty()) {
        throw std::runtime_error("--seam-world-map requires --world-map");
    }
    if (!options.seam_world_map.empty() && options.width <= 2048) {
        throw std::runtime_error("--seam-world-map is only used for panoramas wider than 2048");
    }
    if (options.nadir_mode != "pyramid" && options.nadir_mode != "stabilize" &&
        options.nadir_mode != "none") {
        throw std::runtime_error("--nadir-mode must be pyramid, stabilize or none");
    }
    if (options.exposure_mode != "soft" && options.exposure_mode != "opencv" &&
        options.exposure_mode != "none") {
        throw std::runtime_error("--exposure-mode must be soft, opencv or none");
    }
    if (options.seam_mode != "pairwise" && options.seam_mode != "global") {
        throw std::runtime_error("--seam-mode must be pairwise or global");
    }
    if (options.depth_translation_mode != "head-minus" &&
        options.depth_translation_mode != "head-plus" &&
        options.depth_translation_mode != "camera-minus" &&
        options.depth_translation_mode != "camera-plus") {
        throw std::runtime_error("Invalid --depth-translation-mode");
    }
    if (options.width < 256 || options.width % 2 != 0) {
        throw std::runtime_error("--width must be an even integer >= 256");
    }
    if (options.opencv_threads == 0 || options.opencv_threads < -1) {
        throw std::runtime_error("--opencv-threads must be positive when specified");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.opencv_threads > 0) {
            cv::setNumThreads(options.opencv_threads);
        }
        if (!options.blended_input.empty()) {
            const cv::Mat image = cv::imread(options.blended_input.string(), cv::IMREAD_COLOR);
            cv::Mat valid = cv::imread(options.valid_mask.string(), cv::IMREAD_GRAYSCALE);
            if (image.empty() || valid.empty()) {
                throw std::runtime_error("Cannot read standalone nadir image or mask");
            }
            if (valid.size() != image.size()) {
                cv::resize(valid, valid, image.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            const cv::Mat filled = pyramidInpaint(image, valid, true);
            fs::create_directories(options.output.parent_path());
            if (!cv::imwrite(
                    options.output.string(), filled,
                    {cv::IMWRITE_JPEG_QUALITY,
                     std::clamp(options.jpeg_quality, 35, 100),
                     cv::IMWRITE_JPEG_OPTIMIZE, 1})) {
                throw std::runtime_error("Failed to write: " + options.output.string());
            }
            std::cout << "Wrote " << options.output << " (standalone nadir fill)\n";
            return 0;
        }
        const std::vector<OCamCamera> cameras = readCameras(options.sensor_frame);
        if (!options.selected_camera.empty()) {
            const auto camera = std::find_if(
                cameras.begin(), cameras.end(), [&](const OCamCamera& candidate) {
                    return candidate.name == options.selected_camera;
                });
            if (camera == cameras.end()) {
                throw std::runtime_error("Unknown --selected-camera: " + options.selected_camera);
            }
            const fs::path path = options.input_directory /
                                  (options.capture + "-" + camera->name + ".dng");
            const ProcessedDng processed = postprocessDngHighQuality(path);
            if (!options.decoded_directory.empty()) {
                fs::create_directories(options.decoded_directory);
                const fs::path metadata_path =
                    (options.metadata_directory.empty() ? options.input_directory
                                                        : options.metadata_directory) /
                    (options.capture + "-" + camera->name + ".dng");
                if (!writeProcessedJpeg(
                        (options.decoded_directory /
                         (options.capture + "-" + camera->name + ".jpg")).string(),
                        processed, options.jpeg_quality, metadata_path)) {
                    throw std::runtime_error("Failed to write processed camera image");
                }
            }
            std::cout << "Wrote processed camera " << camera->name << " for capture "
                      << options.capture << '\n';
            return 0;
        }
        std::vector<WarpedCamera> warped;
        std::vector<WarpedCamera> seam_control_warped;
        if (!options.projected_camera_directory.empty()) {
            for (std::size_t index = 0; index < cameras.size(); ++index) {
                cv::Mat image = cv::imread(
                    (options.projected_camera_directory / "images" /
                     ("img_cam" + std::to_string(index) + ".png")).string(),
                    cv::IMREAD_COLOR);
                cv::Mat mask = cv::imread(
                    (options.projected_camera_directory / "masks" /
                     ("mask_cam" + std::to_string(index) + ".png")).string(),
                    cv::IMREAD_GRAYSCALE);
                if (image.empty() || mask.empty()) {
                    throw std::runtime_error(
                        "Cannot load projected seam input for camera " +
                        std::to_string(index));
                }
                if (image.size() != cv::Size(options.width, options.width / 2)) {
                    cv::resize(
                        image, image, cv::Size(options.width, options.width / 2),
                        0.0, 0.0, cv::INTER_AREA);
                    cv::resize(
                        mask, mask, cv::Size(options.width, options.width / 2),
                        0.0, 0.0, cv::INTER_NEAREST);
                }
                image.setTo(cv::Scalar::all(0), mask == 0);
                warped.push_back({image, mask});
            }
            std::cerr << "Loaded " << warped.size() << " projected camera images\n";
        } else {
        std::vector<ProcessedDng> images;
        if (!options.processed_camera_directory.empty()) {
            for (const auto& camera : cameras) {
                images.push_back(loadProcessedCameraJpeg(
                    options.processed_camera_directory /
                    (options.capture + "-" + camera.name + ".jpg")));
            }
        } else {
            // Warm the Adobe DNG SDK on one thread before constructing hosts
            // in parallel. Its process-wide tables are lazily initialized.
            images.resize(cameras.size());
            images.front() = postprocessDngHighQuality(
                options.input_directory /
                (options.capture + "-" + cameras.front().name + ".dng"));
            cv::parallel_for_(
                cv::Range(1, static_cast<int>(cameras.size())),
                [&](const cv::Range& range) {
                    for (int index = range.start; index < range.end; ++index) {
                        const auto camera_index = static_cast<std::size_t>(index);
                        const fs::path path =
                            options.input_directory /
                            (options.capture + "-" + cameras[camera_index].name + ".dng");
                        images[camera_index] = postprocessDngHighQuality(path);
                    }
                });
        }
        std::cerr << "Decoded " << images.size() << " camera images\n";
        if (!options.decoded_directory.empty()) {
            fs::create_directories(options.decoded_directory);
            for (std::size_t index = 0; index < images.size(); ++index) {
                const fs::path metadata_path =
                    (options.metadata_directory.empty() ? options.input_directory
                                                        : options.metadata_directory) /
                    (options.capture + "-" + cameras[index].name + ".dng");
                if (!writeProcessedJpeg(
                    (options.decoded_directory /
                     (options.capture + "-" + cameras[index].name + ".jpg")).string(),
                    images[index], options.jpeg_quality, metadata_path)) {
                    throw std::runtime_error("Failed to write processed camera image");
                }
            }
        }
        if (options.camera_only) {
            std::cout << "Wrote processed cameras for capture " << options.capture << '\n';
            return 0;
        }
        std::vector<cv::Mat> camera_masks(cameras.size());
        if (!options.camera_mask_directory.empty()) {
            for (std::size_t index = 0; index < cameras.size(); ++index) {
                const fs::path mask_path =
                    options.camera_mask_directory /
                    ("mask-" + cameras[index].name + ".png");
                camera_masks[index] = cv::imread(mask_path.string(), cv::IMREAD_GRAYSCALE);
                if (camera_masks[index].empty()) {
                    throw std::runtime_error(
                        "Cannot read camera mask: " + mask_path.string());
                }
            }
        }
        cv::Mat panorama_world_map;
        cv::Mat seam_world_map;
        cv::Mat panorama_depth;
        if (!options.world_map.empty()) {
            panorama_world_map = loadPanoramaWorldMap(
                options.world_map, options.width, options.width / 2);
            if (!options.seam_world_map.empty()) {
                seam_world_map = loadPanoramaWorldMap(
                    options.seam_world_map, 2048, 1024);
            }
        } else if (!options.depth_map.empty()) {
            panorama_depth = loadPanoramaDepthMetres(options.depth_map);
            const cv::Mat1b sparse_valid = panorama_depth > 0.0F;
            if (static_cast<std::size_t>(cv::countNonZero(sparse_valid)) !=
                panorama_depth.total()) {
                std::cerr << "Optimizing sparse panorama depth at "
                          << panorama_depth.cols << 'x' << panorama_depth.rows << '\n';
                panorama_depth =
                    navvis_recon::GaussNewtonDepthMapOptimizer::optimizeDouble(
                        panorama_depth, sparse_valid, 1.0F);
            }
        } else if (!options.surface_cloud.empty()) {
            std::cerr << "Rendering 1024x512 panorama depth from "
                      << options.surface_cloud << '\n';
            const std::vector<navvis_recon::Vec3f> surface_points =
                readSurfacePositions(options.surface_cloud);
            const navvis_recon::Pose head_pose =
                readPanoramaHeadPose(options.panorama_info);
            panorama_depth = navvis_recon::PanoramaDepthRenderer::render(
                surface_points, head_pose,
                readPanoramaNearDistance(options.panorama_info, head_pose),
                1024, 512);
        }
        if (!panorama_depth.empty()) {
            if (panorama_depth.type() == CV_64F) {
                writeDebugDepthRaw(
                    panorama_depth, options.debug_directory,
                    "panorama-depth-native.f64");
            }
            writeDebugDepthMap(
                panorama_depth, options.debug_directory,
                "panorama-depth-native-mm.png");
            if (options.depth_translation_mode == "head-minus") {
                const bool second_stage = options.width > 2048;
                cv::Mat projection_depth = preparePanoramaProjectionDepth(
                    panorama_depth, options.width, second_stage);
                writeDebugDepthMap(
                    projection_depth, options.debug_directory,
                    "panorama-depth-projection-mm.png");
                panorama_world_map = buildPanoramaWorldMap(projection_depth);
                projection_depth.release();

                if (second_stage) {
                    cv::Mat seam_depth = preparePanoramaProjectionDepth(
                        panorama_depth, 2048, false);
                    seam_world_map = buildPanoramaWorldMap(seam_depth);
                }
                panorama_depth.release();
            } else {
                if (panorama_depth.cols != options.width ||
                    panorama_depth.rows != options.width / 2) {
                    cv::resize(
                        panorama_depth, panorama_depth,
                        cv::Size(options.width, options.width / 2),
                        0.0, 0.0, cv::INTER_LINEAR);
                }
                writeDebugDepthMap(
                    panorama_depth, options.debug_directory,
                    "panorama-depth-projection-mm.png");
                if (panorama_depth.type() == CV_64F) {
                    panorama_depth.convertTo(panorama_depth, CV_32F);
                }
            }
        }
        warped.resize(cameras.size());
        cv::parallel_for_(
            cv::Range(0, static_cast<int>(cameras.size())),
            [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    const auto camera_index = static_cast<std::size_t>(index);
                    warped[camera_index] = warpCamera(
                        images[camera_index].sensor_raster,
                        camera_masks[camera_index], panorama_world_map,
                        panorama_depth, cameras[camera_index], options.width,
                        options.width / 2, options.depth_translation_mode);
                }
            });
        std::cerr << "Warped " << warped.size() << " camera images\n";

        if (!seam_world_map.empty()) {
            seam_control_warped.resize(cameras.size());
            cv::parallel_for_(
                cv::Range(0, static_cast<int>(cameras.size())),
                [&](const cv::Range& range) {
                    for (int index = range.start; index < range.end; ++index) {
                        const auto camera_index = static_cast<std::size_t>(index);
                        seam_control_warped[camera_index] = warpCamera(
                            images[camera_index].sensor_raster,
                            camera_masks[camera_index], seam_world_map, cv::Mat(),
                            cameras[camera_index], 2048, 1024,
                            options.depth_translation_mode);
                    }
                });
            std::cerr << "Warped " << seam_control_warped.size()
                      << " independent 2K seam-control camera images\n";
        }
        panorama_world_map.release();
        seam_world_map.release();
        panorama_depth.release();
        images.clear();
        camera_masks.clear();
        }
        cv::Mat operator_mask;
        if (!options.operator_mask.empty()) {
            operator_mask = cv::imread(options.operator_mask.string(), cv::IMREAD_GRAYSCALE);
            if (operator_mask.empty()) {
                throw std::runtime_error(
                    "Cannot read --operator-mask: " + options.operator_mask.string());
            }
        }
        const cv::Mat panorama = stitch(
            std::move(warped), std::move(seam_control_warped),
            options.width, options.width / 2, operator_mask,
            options.debug_directory, options.nadir_mode, options.exposure_mode,
            options.seam_mode, options.jpeg_quality);
        fs::create_directories(options.output.parent_path());
        if (!cv::imwrite(
                options.output.string(), panorama,
                {cv::IMWRITE_JPEG_QUALITY,
                 std::clamp(options.jpeg_quality, 35, 100),
                 cv::IMWRITE_JPEG_OPTIMIZE,
                 options.nadir_mode == "pyramid" ? 1 : 0})) {
            throw std::runtime_error("Failed to write: " + options.output.string());
        }
        std::cout << "Wrote " << options.output << " (" << panorama.cols << "x" << panorama.rows << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
