#define PCL_NO_PRECOMPILE

#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kVoxelResolution = 0.05;

struct PlyInput {
    std::ifstream stream;
    std::uint64_t point_count = 0;
    std::size_t stride = 0;
    std::streamoff data_offset = 0;
};

struct Pose {
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
};

struct Metrics {
    std::uint64_t clean_valid = 0;
    std::uint64_t vendor_valid = 0;
    std::uint64_t intersection = 0;
    std::uint64_t exact_depth = 0;
    std::uint64_t absolute_mm_sum = 0;
    int maximum_mm = 0;
    std::size_t first_mismatch = std::numeric_limits<std::size_t>::max();
};

std::string readText(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot read: " + path.string());
    }
    return std::string(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string jsonObjectSection(const std::string& json, const std::string& name) {
    const std::size_t name_position = json.find('"' + name + '"');
    if (name_position == std::string::npos) {
        throw std::runtime_error("Missing JSON object: " + name);
    }
    const std::size_t begin = json.find('{', name_position);
    if (begin == std::string::npos) {
        throw std::runtime_error("Malformed JSON object: " + name);
    }
    int depth = 0;
    for (std::size_t index = begin; index < json.size(); ++index) {
        if (json[index] == '{') {
            ++depth;
        } else if (json[index] == '}' && --depth == 0) {
            return json.substr(begin, index - begin + 1);
        }
    }
    throw std::runtime_error("Unterminated JSON object: " + name);
}

std::vector<double> jsonArray(
    const std::string& section, const std::string& name, int expected_size) {
    const std::string number = R"(([-+0-9.eE]+))";
    std::string expression = "\\\"" + name + "\\\"\\s*:\\s*\\[\\s*";
    for (int index = 0; index < expected_size; ++index) {
        if (index != 0) {
            expression += "\\s*,\\s*";
        }
        expression += number;
    }
    expression += "\\s*\\]";
    std::smatch match;
    if (!std::regex_search(section, match, std::regex(expression))) {
        throw std::runtime_error("Missing JSON array: " + name);
    }
    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(expected_size));
    for (int index = 0; index < expected_size; ++index) {
        result.push_back(std::stod(match[static_cast<std::size_t>(index + 1)].str()));
    }
    return result;
}

Eigen::Vector3d readPosition(const std::string& json, const std::string& name) {
    const auto values = jsonArray(jsonObjectSection(json, name), "position", 3);
    return {values[0], values[1], values[2]};
}

Pose readHeadPose(const std::string& json) {
    const std::string section = jsonObjectSection(json, "cam_head");
    const auto position = jsonArray(section, "position", 3);
    const auto quaternion = jsonArray(section, "quaternion", 4);
    Pose pose;
    pose.translation = {position[0], position[1], position[2]};
    pose.rotation = Eigen::Quaterniond(
        quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    pose.rotation.normalize();
    return pose;
}

double nearDistance(const std::string& json, const Eigen::Vector3d& head) {
    double result = 0.0;
    for (int camera = 0; camera < 4; ++camera) {
        const Eigen::Vector3d position =
            readPosition(json, "cam" + std::to_string(camera));
        result = std::max(result, (position - head).norm());
    }
    return result;
}

PlyInput openPly(const fs::path& path) {
    PlyInput input{std::ifstream(path, std::ios::binary)};
    if (!input.stream) {
        throw std::runtime_error("Cannot open PLY: " + path.string());
    }
    std::string line;
    std::vector<std::pair<std::string, std::string>> properties;
    while (std::getline(input.stream, line)) {
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(^element vertex ([0-9]+)$)"))) {
            input.point_count = std::stoull(match[1].str());
        } else if (std::regex_match(
                       line, match, std::regex(R"(^property ([^ ]+) ([^ ]+)$)"))) {
            properties.emplace_back(match[1].str(), match[2].str());
        } else if (line == "end_header") {
            break;
        }
    }
    const std::vector<std::pair<std::string, std::string>> surface{
        {"float", "x"}, {"float", "y"}, {"float", "z"},
        {"float", "intensity"}, {"float", "nx"}, {"float", "ny"},
        {"float", "nz"}, {"float", "curvature"}};
    const std::vector<std::pair<std::string, std::string>> colored{
        {"float", "x"}, {"float", "y"}, {"float", "z"},
        {"uchar", "red"}, {"uchar", "green"}, {"uchar", "blue"},
        {"uchar", "alpha"}, {"float", "intensity"}, {"float", "nx"},
        {"float", "ny"}, {"float", "nz"}, {"float", "curvature"}};
    input.stride = properties == surface ? 32U : properties == colored ? 36U : 0U;
    input.data_offset = input.stream.tellg();
    if (input.point_count == 0 || input.stride == 0 || input.data_offset <= 0) {
        throw std::runtime_error("Unsupported PLY layout: " + path.string());
    }
    const std::uint64_t expected = static_cast<std::uint64_t>(input.data_offset) +
                                   input.point_count * input.stride;
    if (fs::file_size(path) != expected) {
        throw std::runtime_error("Incomplete PLY: " + path.string());
    }
    return input;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr readCloud(const fs::path& path) {
    PlyInput input = openPly(path);
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->points.resize(static_cast<std::size_t>(input.point_count));
    cloud->width = static_cast<std::uint32_t>(input.point_count);
    cloud->height = 1;
    cloud->is_dense = true;

    constexpr std::size_t chunk_points = 262144;
    std::vector<char> bytes(chunk_points * input.stride);
    std::uint64_t consumed = 0;
    while (consumed < input.point_count) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(chunk_points, input.point_count - consumed));
        input.stream.read(bytes.data(), static_cast<std::streamsize>(chunk * input.stride));
        if (!input.stream) {
            throw std::runtime_error("Failed while reading PLY vertices");
        }
        for (std::size_t index = 0; index < chunk; ++index) {
            std::array<float, 3> xyz{};
            std::memcpy(xyz.data(), bytes.data() + index * input.stride, sizeof(xyz));
            pcl::PointXYZ& point = cloud->points[static_cast<std::size_t>(consumed) + index];
            point.x = xyz[0];
            point.y = xyz[1];
            point.z = xyz[2];
            point.data[3] = 1.0F;
            cloud->is_dense = cloud->is_dense && pcl::isFinite(point);
        }
        consumed += chunk;
    }
    return cloud;
}

Eigen::Vector3d panoramaRay(int column, int row, int width, int height) {
    const double latitude =
        (0.5 - (static_cast<double>(row) + 0.5) / height) * kPi;
    const double longitude =
        ((static_cast<double>(column) + 0.5) / width - 0.5) * (2.0 * kPi);
    const double cosine = std::cos(latitude);
    return {cosine * std::cos(longitude),
            -cosine * std::sin(longitude),
            std::sin(latitude)};
}

std::pair<double, double> worldToPanorama(
    const pcl::PointXYZ& point, const Pose& world_from_head,
    int width, int height) {
    const Eigen::Vector3d point_world(point.x, point.y, point.z);
    const Eigen::Vector3d local =
        world_from_head.rotation.conjugate() *
        (point_world - world_from_head.translation);
    const double longitude = std::atan2(-local.y(), local.x());
    const double latitude = std::atan2(local.z(), std::hypot(local.x(), local.y()));
    const double column =
        (longitude / (2.0 * kPi) + 0.5) * width - 0.5;
    const double row = (0.5 - latitude / kPi) * height - 0.5;
    return {column, row};
}

float vendorSquaredDistance(
    const pcl::PointXYZ& point, const pcl::PointXYZ& origin) {
    const float delta_z = origin.z - point.z;
    const float delta_y = origin.y - point.y;
    const float delta_x = origin.x - point.x;
    const float squared_z = delta_z * delta_z;
    const float squared_y = delta_y * delta_y;
    const float squared_x = delta_x * delta_x;
    const float squared_yz = squared_y + squared_z;
    return squared_x + squared_yz;
}

std::uint16_t quantizeRound(float depth) {
    if (!(depth > 0.0F) || !std::isfinite(depth)) {
        return 0;
    }
    return static_cast<std::uint16_t>(
        std::clamp(std::lround(static_cast<double>(depth) * 1000.0), 0L, 65535L));
}

std::uint16_t quantizeTruncate(float depth) {
    if (!(depth > 0.0F) || !std::isfinite(depth)) {
        return 0;
    }
    const float millimetres = depth * 1000.0F;
    return static_cast<std::uint16_t>(std::clamp(
        static_cast<long>(millimetres), 0L, 65535L));
}

Metrics compareDepth(
    const cv::Mat_<std::uint16_t>& clean,
    const cv::Mat_<std::uint16_t>& vendor) {
    Metrics metrics;
    for (std::size_t index = 0; index < clean.total(); ++index) {
        const int clean_value = clean.ptr<std::uint16_t>()[index];
        const int vendor_value = vendor.ptr<std::uint16_t>()[index];
        const bool clean_valid = clean_value != 0;
        const bool vendor_valid = vendor_value != 0;
        metrics.clean_valid += clean_valid;
        metrics.vendor_valid += vendor_valid;
        if (clean_valid && vendor_valid) {
            ++metrics.intersection;
            metrics.exact_depth += clean_value == vendor_value;
            const int difference = std::abs(clean_value - vendor_value);
            metrics.absolute_mm_sum += static_cast<std::uint64_t>(difference);
            metrics.maximum_mm = std::max(metrics.maximum_mm, difference);
        }
        if (metrics.first_mismatch == std::numeric_limits<std::size_t>::max() &&
            clean_value != vendor_value) {
            metrics.first_mismatch = index;
        }
    }
    return metrics;
}

void printMetrics(const char* label, const Metrics& metrics) {
    const std::uint64_t union_count =
        metrics.clean_valid + metrics.vendor_valid - metrics.intersection;
    const double iou = union_count == 0
                           ? 1.0
                           : static_cast<double>(metrics.intersection) / union_count;
    const double exact_fraction = metrics.intersection == 0
                                      ? 1.0
                                      : static_cast<double>(metrics.exact_depth) /
                                            metrics.intersection;
    const double mae = metrics.intersection == 0
                           ? 0.0
                           : static_cast<double>(metrics.absolute_mm_sum) /
                                 metrics.intersection;
    std::cout << label << " clean_valid=" << metrics.clean_valid
              << " vendor_valid=" << metrics.vendor_valid
              << " intersection=" << metrics.intersection
              << " union=" << union_count
              << " mask_iou=" << iou
              << " common_exact_mm=" << metrics.exact_depth << '/'
              << metrics.intersection
              << " exact_fraction=" << exact_fraction
              << " common_mae_mm=" << mae
              << " common_max_mm=" << metrics.maximum_mm << '\n';
}

void writeRaw(const fs::path& path, const cv::Mat& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot write: " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(values.data),
        static_cast<std::streamsize>(values.total() * values.elemSize()));
}

cv::Mat4b encodeDepth(const cv::Mat_<std::uint16_t>& millimetres) {
    cv::Mat4b encoded(millimetres.size(), cv::Vec4b(0, 0, 0, 0));
    for (int row = 0; row < millimetres.rows; ++row) {
        for (int column = 0; column < millimetres.cols; ++column) {
            const std::uint16_t value = millimetres(row, column);
            encoded(row, column)[1] = static_cast<std::uint8_t>(value >> 8U);
            encoded(row, column)[2] = static_cast<std::uint8_t>(value & 0xffU);
        }
    }
    return encoded;
}

std::uint32_t floatBits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7 && argc != 8) {
            std::cerr <<
                "Usage: panorama_world_map_sparse_octree_probe SURFACE.ply INFO.json "
                "VENDOR.png OUTPUT_DIR WIDTH HEIGHT [--compact]\n";
            return 2;
        }
        const bool compact = argc == 8 && std::string(argv[7]) == "--compact";
        if (argc == 8 && !compact) {
            throw std::invalid_argument("The only optional argument is --compact");
        }
        const int width = std::stoi(argv[5]);
        const int height = std::stoi(argv[6]);
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("Width and height must be positive");
        }

        const auto begin = Clock::now();
        const auto cloud = readCloud(argv[1]);
        const auto loaded = Clock::now();
        std::uint64_t nonfinite_points = 0;
        Eigen::Vector3f raw_min = Eigen::Vector3f::Constant(
            std::numeric_limits<float>::max());
        Eigen::Vector3f raw_max = Eigen::Vector3f::Constant(
            std::numeric_limits<float>::lowest());
        for (const auto& point : cloud->points) {
            nonfinite_points += !pcl::isFinite(point);
            raw_min = raw_min.cwiseMin(point.getVector3fMap());
            raw_max = raw_max.cwiseMax(point.getVector3fMap());
        }
        const std::string info_json = readText(argv[2]);
        const Pose pose = readHeadPose(info_json);
        const double near_distance = nearDistance(info_json, pose.translation);
        const double near_squared = near_distance * near_distance;

        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(kVoxelResolution);
        octree.setInputCloud(cloud);
        octree.defineBoundingBox();
        double defined_min_x = 0.0;
        double defined_min_y = 0.0;
        double defined_min_z = 0.0;
        double defined_max_x = 0.0;
        double defined_max_y = 0.0;
        double defined_max_z = 0.0;
        octree.getBoundingBox(
            defined_min_x, defined_min_y, defined_min_z,
            defined_max_x, defined_max_y, defined_max_z);
        octree.addPointsFromInputCloud();
        double octree_min_x = 0.0;
        double octree_min_y = 0.0;
        double octree_min_z = 0.0;
        double octree_max_x = 0.0;
        double octree_max_y = 0.0;
        double octree_max_z = 0.0;
        octree.getBoundingBox(
            octree_min_x, octree_min_y, octree_min_z,
            octree_max_x, octree_max_y, octree_max_z);
        const auto indexed = Clock::now();

        cv::Mat1f ray_distance(height, width, -1.0F);
        cv::Mat1i ray_owner(height, width, -1);
        cv::Mat1i ray_candidate_count(height, width, 0);
        cv::Mat1f reprojected_depth(height, width, -1.0F);
        cv::Mat1i reprojected_owner(height, width, -1);

        const pcl::PointXYZ origin(
            static_cast<float>(pose.translation.x()),
            static_cast<float>(pose.translation.y()),
            static_cast<float>(pose.translation.z()));
        const Eigen::Vector3d first_world =
            pose.rotation * panoramaRay(0, 0, width, height) + pose.translation;
        const std::array<float, 3> first_direction{
            static_cast<float>(first_world.x()) - origin.x,
            static_cast<float>(first_world.y()) - origin.y,
            static_cast<float>(first_world.z()) - origin.z};
        std::cout << std::hex << std::setfill('0')
                  << "first_origin_bits=0x" << std::setw(8) << floatBits(origin.x)
                  << ",0x" << std::setw(8) << floatBits(origin.y)
                  << ",0x" << std::setw(8) << floatBits(origin.z)
                  << " first_direction_bits=0x" << std::setw(8)
                  << floatBits(first_direction[0]) << ",0x" << std::setw(8)
                  << floatBits(first_direction[1]) << ",0x" << std::setw(8)
                  << floatBits(first_direction[2]) << std::dec << std::setfill(' ')
                  << '\n';
        const auto& first_point = cloud->points.front();
        std::cout << std::setprecision(9) << "cloud_point index=0 xyz=("
                  << first_point.x << ' ' << first_point.y << ' '
                  << first_point.z << ")\n";
        std::vector<int> candidates;
        std::vector<int> first_ray_candidates;
        std::uint64_t rays_with_voxels = 0;
        std::uint64_t rays_with_points = 0;
        std::uint64_t projected_hits = 0;
        std::uint64_t projection_collisions = 0;
        std::uint64_t projection_outside = 0;
        std::uint64_t candidate_indices = 0;

        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const Eigen::Vector3d point_world =
                    pose.rotation * panoramaRay(column, row, width, height) +
                    pose.translation;
                pcl::PointXYZ direction;
                direction.x = static_cast<float>(point_world.x()) - origin.x;
                direction.y = static_cast<float>(point_world.y()) - origin.y;
                direction.z = static_cast<float>(point_world.z()) - origin.z;
                direction.data[3] = 1.0F;

                const auto voxel_count = octree.getIntersectedVoxelIndices(
                    origin.getVector3fMap(), direction.getVector3fMap(), candidates);
                if (row == 0 && column == 0) {
                    first_ray_candidates = candidates;
                }
                rays_with_voxels += voxel_count != 0;
                candidate_indices += candidates.size();
                ray_candidate_count(row, column) =
                    static_cast<int>(candidates.size());

                int best_index = -1;
                double best_squared = std::numeric_limits<double>::infinity();
                for (const int point_index : candidates) {
                    const double squared = static_cast<double>(vendorSquaredDistance(
                        cloud->points[static_cast<std::size_t>(point_index)], origin));
                    if (squared > near_squared && squared < best_squared) {
                        best_squared = squared;
                        best_index = point_index;
                    }
                }
                if (best_index < 0) {
                    continue;
                }
                ++rays_with_points;
                const float distance = static_cast<float>(std::sqrt(best_squared));
                ray_distance(row, column) = distance;
                ray_owner(row, column) = best_index;

                const auto [projected_column, projected_row] = worldToPanorama(
                    cloud->points[static_cast<std::size_t>(best_index)],
                    pose, width, height);
                const int target_column = static_cast<int>(std::round(projected_column));
                const int target_row = static_cast<int>(std::round(projected_row));
                if (target_column < 0 || target_column >= width ||
                    target_row < 0 || target_row >= height) {
                    ++projection_outside;
                    continue;
                }
                float& current = reprojected_depth(target_row, target_column);
                if (current >= 0.0F && current <= distance) {
                    ++projection_collisions;
                    continue;
                }
                projection_collisions += current >= 0.0F;
                current = distance;
                reprojected_owner(target_row, target_column) = best_index;
                ++projected_hits;
            }
        }
        const auto rendered = Clock::now();

        cv::Mat_<std::uint16_t> clean_round(height, width, std::uint16_t{0});
        cv::Mat_<std::uint16_t> clean_truncate(height, width, std::uint16_t{0});
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                clean_round(row, column) = quantizeRound(reprojected_depth(row, column));
                clean_truncate(row, column) = quantizeTruncate(reprojected_depth(row, column));
            }
        }

        const cv::Mat4b vendor_png = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
        if (vendor_png.size() != cv::Size(width, height) ||
            vendor_png.type() != CV_8UC4) {
            throw std::runtime_error("Vendor PNG dimensions/type do not match");
        }
        cv::Mat_<std::uint16_t> vendor_mm(height, width);
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const cv::Vec4b pixel = vendor_png(row, column);
                vendor_mm(row, column) = static_cast<std::uint16_t>(
                    (static_cast<std::uint16_t>(pixel[1]) << 8U) | pixel[2]);
            }
        }
        const Metrics round_metrics = compareDepth(clean_round, vendor_mm);
        const Metrics truncate_metrics = compareDepth(clean_truncate, vendor_mm);

        const fs::path output_directory = argv[4];
        fs::create_directories(output_directory);
        if (!compact) {
            writeRaw(output_directory / "ray_candidate_count.i32", ray_candidate_count);
            if (!first_ray_candidates.empty()) {
                std::ofstream first_candidates(
                    output_directory / "first_ray_candidates.i32",
                    std::ios::binary | std::ios::trunc);
                first_candidates.write(
                    reinterpret_cast<const char*>(first_ray_candidates.data()),
                    static_cast<std::streamsize>(first_ray_candidates.size() *
                                                 sizeof(first_ray_candidates.front())));
            }
            writeRaw(output_directory / "ray_owner.i32", ray_owner);
            writeRaw(output_directory / "ray_distance.f32", ray_distance);
            writeRaw(output_directory / "reprojected_owner.i32", reprojected_owner);
            writeRaw(output_directory / "reprojected_depth.f32", reprojected_depth);
            cv::imwrite(
                (output_directory / "clean_sparse_round.png").string(),
                encodeDepth(clean_round));
        }
        cv::imwrite(
            (output_directory / "clean_sparse_truncate.png").string(),
            encodeDepth(clean_truncate));

        const auto milliseconds = [](Clock::time_point start, Clock::time_point end) {
            return std::chrono::duration<double, std::milli>(end - start).count();
        };
        std::cout << std::setprecision(17)
                  << "points=" << cloud->size()
                  << " is_dense=" << cloud->is_dense
                  << " nonfinite_points=" << nonfinite_points
                  << " width=" << width << " height=" << height
                  << " voxel_resolution_m=" << kVoxelResolution
                  << " near_distance_m=" << near_distance << '\n'
                  << "raw_point_bounds=[" << raw_min.x() << ',' << raw_min.y()
                  << ',' << raw_min.z() << "]..[" << raw_max.x() << ','
                  << raw_max.y() << ',' << raw_max.z() << "]\n"
                  << "defined_octree_bounds=[" << defined_min_x << ','
                  << defined_min_y << ',' << defined_min_z << "]..["
                  << defined_max_x << ',' << defined_max_y << ','
                  << defined_max_z << "]\n"
                  << "octree_depth=" << octree.getTreeDepth()
                  << " octree_leaves=" << octree.getLeafCount()
                  << " octree_branches=" << octree.getBranchCount() << '\n'
                  << "octree_bounds=[" << octree_min_x << ',' << octree_min_y
                  << ',' << octree_min_z << "]..[" << octree_max_x << ','
                  << octree_max_y << ',' << octree_max_z << "]\n"
                  << "stage_load_ms=" << milliseconds(begin, loaded)
                  << " stage_octree_ms=" << milliseconds(loaded, indexed)
                  << " stage_render_ms=" << milliseconds(indexed, rendered) << '\n'
                  << "rays=" << static_cast<std::uint64_t>(width) * height
                  << " first_ray_candidates=" << first_ray_candidates.size()
                  << " rays_with_voxels=" << rays_with_voxels
                  << " candidate_indices=" << candidate_indices
                  << " rays_with_points=" << rays_with_points << '\n'
                  << "projected_hits=" << projected_hits
                  << " projection_collisions=" << projection_collisions
                  << " projection_outside=" << projection_outside << '\n';
        printMetrics("round", round_metrics);
        printMetrics("truncate", truncate_metrics);

        const Metrics& best =
            round_metrics.exact_depth >= truncate_metrics.exact_depth
                ? round_metrics
                : truncate_metrics;
        if (best.first_mismatch == std::numeric_limits<std::size_t>::max()) {
            std::cout << "sparse_png=EXACT\n";
        } else {
            const std::size_t index = best.first_mismatch;
            const int row = static_cast<int>(index / static_cast<std::size_t>(width));
            const int column = static_cast<int>(index % static_cast<std::size_t>(width));
            const bool use_round = &best == &round_metrics;
            const auto& clean = use_round ? clean_round : clean_truncate;
            std::cout << "best_quantizer=" << (use_round ? "round" : "truncate")
                      << " first_mismatch_index=" << index
                      << " row=" << row << " column=" << column
                      << " clean_mm=" << clean(row, column)
                      << " vendor_mm=" << vendor_mm(row, column)
                      << " owner=" << reprojected_owner(row, column)
                      << " clean_depth_m=" << reprojected_depth(row, column) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
