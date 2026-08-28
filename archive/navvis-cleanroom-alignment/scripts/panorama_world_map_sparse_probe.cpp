#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
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
using Vec3f = Eigen::Vector3f;

struct PlyInput {
    std::ifstream stream;
    std::uint64_t point_count = 0;
    std::size_t stride = 0;
    std::streamoff data_offset = 0;
};

struct Pose {
    Vec3f translation = Vec3f::Zero();
    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
};

std::string readText(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot read: " + path.string());
    }
    return std::string(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

Pose readHeadPose(const fs::path& path) {
    const std::string json = readText(path);
    const std::size_t begin = json.find("\"cam_head\"");
    const std::size_t end = json.find("\n    }", begin);
    if (begin == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("Missing cam_head pose: " + path.string());
    }
    const std::string section = json.substr(begin, end - begin);
    const std::string number = R"(([-+0-9.eE]+))";
    const std::regex position_expression(
        "\\\"position\\\"\\s*:\\s*\\[\\s*" + number +
        "\\s*,\\s*" + number + "\\s*,\\s*" + number + "\\s*\\]");
    const std::regex quaternion_expression(
        "\\\"quaternion\\\"\\s*:\\s*\\[\\s*" + number +
        "\\s*,\\s*" + number + "\\s*,\\s*" + number +
        "\\s*,\\s*" + number + "\\s*\\]");
    std::smatch position;
    std::smatch quaternion;
    if (!std::regex_search(section, position, position_expression) ||
        !std::regex_search(section, quaternion, quaternion_expression)) {
        throw std::runtime_error("Invalid cam_head pose: " + path.string());
    }

    Pose pose;
    pose.translation = Vec3f(
        std::stof(position[1].str()), std::stof(position[2].str()),
        std::stof(position[3].str()));
    pose.rotation = Eigen::Quaternionf(
        std::stof(quaternion[1].str()), std::stof(quaternion[2].str()),
        std::stof(quaternion[3].str()), std::stof(quaternion[4].str()))
                        .normalized()
                        .toRotationMatrix();
    return pose;
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
        throw std::runtime_error("Unsupported PLY vertex layout: " + path.string());
    }
    const std::uint64_t expected = static_cast<std::uint64_t>(input.data_offset) +
                                   input.point_count * input.stride;
    if (fs::file_size(path) != expected) {
        throw std::runtime_error("Incomplete PLY: " + path.string());
    }
    return input;
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void writeRaw(const fs::path& path, const cv::Mat1f& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot write: " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(values.ptr<float>()),
        static_cast<std::streamsize>(values.total() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr <<
                "Usage: panorama_world_map_sparse_probe SURFACE.ply INFO.json "
                "VENDOR.tiff OUTPUT_DIR\n";
            return 2;
        }
        constexpr int width = 1024;
        constexpr int height = 512;
        constexpr float pi = 3.14159265358979323846F;

        PlyInput ply = openPly(argv[1]);
        const Pose pose = readHeadPose(argv[2]);
        const cv::Mat1f vendor = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
        if (vendor.size() != cv::Size(width, height) || vendor.type() != CV_32FC1) {
            throw std::runtime_error("Vendor sparse depth must be 1024x512 CV_32FC1");
        }

        cv::Mat1f clean(height, width, std::numeric_limits<float>::infinity());
        std::vector<std::uint64_t> owner(
            static_cast<std::size_t>(width) * height,
            std::numeric_limits<std::uint64_t>::max());
        std::uint64_t accepted = 0;
        constexpr std::size_t chunk_points = 262144;
        std::vector<char> bytes(chunk_points * ply.stride);
        std::uint64_t consumed = 0;
        while (consumed < ply.point_count) {
            const std::size_t chunk = static_cast<std::size_t>(
                std::min<std::uint64_t>(chunk_points, ply.point_count - consumed));
            ply.stream.read(
                bytes.data(), static_cast<std::streamsize>(chunk * ply.stride));
            if (!ply.stream) {
                throw std::runtime_error("Failed while reading PLY vertices");
            }
            for (std::size_t index = 0; index < chunk; ++index) {
                float xyz[3];
                std::memcpy(xyz, bytes.data() + index * ply.stride, sizeof(xyz));
                const Vec3f point(xyz[0], xyz[1], xyz[2]);
                if (!point.allFinite()) {
                    continue;
                }
                const Vec3f local = pose.rotation.transpose() * (point - pose.translation);
                const float distance = local.norm();
                if (!std::isfinite(distance) || distance < 0.2F || distance > 65.0F) {
                    continue;
                }
                const float longitude = std::atan2(-local.y(), local.x());
                const float latitude =
                    std::asin(std::clamp(local.z() / distance, -1.0F, 1.0F));
                int x = static_cast<int>(std::floor(
                    (longitude / (2.0F * pi) + 0.5F) * static_cast<float>(width)));
                const int y = std::clamp(
                    static_cast<int>(std::floor(
                        (0.5F - latitude / pi) * static_cast<float>(height))),
                    0, height - 1);
                x = (x % width + width) % width;
                if (distance < clean(y, x)) {
                    clean(y, x) = distance;
                    owner[static_cast<std::size_t>(y) * width + x] =
                        consumed + index;
                }
                ++accepted;
            }
            consumed += chunk;
        }
        clean.setTo(0.0F, clean == std::numeric_limits<float>::infinity());

        std::uint64_t clean_valid = 0;
        std::uint64_t vendor_valid = 0;
        std::uint64_t intersection = 0;
        std::uint64_t exact_common = 0;
        long double absolute_sum = 0.0;
        long double squared_sum = 0.0;
        double maximum_absolute = 0.0;
        std::size_t first_mismatch = clean.total();
        std::size_t first_clean_only = clean.total();
        std::size_t first_vendor_only = clean.total();
        std::size_t first_common_depth = clean.total();
        for (std::size_t index = 0; index < clean.total(); ++index) {
            const float clean_value = clean.ptr<float>()[index];
            const float vendor_value = vendor.ptr<float>()[index];
            const bool clean_has_value = clean_value != 0.0F;
            const bool vendor_has_value = vendor_value != 0.0F;
            clean_valid += clean_has_value;
            vendor_valid += vendor_has_value;
            if (clean_has_value && vendor_has_value) {
                ++intersection;
                exact_common += bits(clean_value) == bits(vendor_value);
                const double absolute = std::abs(
                    static_cast<double>(clean_value) - vendor_value);
                absolute_sum += absolute;
                squared_sum += absolute * absolute;
                maximum_absolute = std::max(maximum_absolute, absolute);
            }
            if (first_mismatch == clean.total() &&
                (clean_has_value != vendor_has_value ||
                 (clean_has_value && bits(clean_value) != bits(vendor_value)))) {
                first_mismatch = index;
            }
            if (first_clean_only == clean.total() &&
                clean_has_value && !vendor_has_value) {
                first_clean_only = index;
            }
            if (first_vendor_only == clean.total() &&
                !clean_has_value && vendor_has_value) {
                first_vendor_only = index;
            }
            if (first_common_depth == clean.total() && clean_has_value &&
                vendor_has_value && bits(clean_value) != bits(vendor_value)) {
                first_common_depth = index;
            }
        }
        const std::uint64_t union_count = clean_valid + vendor_valid - intersection;
        const double iou = union_count == 0
                               ? 1.0
                               : static_cast<double>(intersection) / union_count;
        const double mae = intersection == 0
                               ? 0.0
                               : static_cast<double>(absolute_sum / intersection);
        const double rmse = intersection == 0
                                ? 0.0
                                : std::sqrt(static_cast<double>(squared_sum / intersection));

        const fs::path output_directory = argv[4];
        fs::create_directories(output_directory);
        writeRaw(output_directory / "clean_sparse_measured.f32", clean);

        std::cout << std::setprecision(17)
                  << "points=" << ply.point_count << " accepted=" << accepted << '\n'
                  << "clean_valid=" << clean_valid << " vendor_valid=" << vendor_valid
                  << " intersection=" << intersection << " union=" << union_count
                  << " mask_iou=" << iou << '\n'
                  << "common_exact=" << exact_common << '/' << intersection
                  << " depth_mae_m=" << mae << " depth_rmse_m=" << rmse
                  << " depth_max_m=" << maximum_absolute << '\n';
        if (first_mismatch != clean.total()) {
            const int row = static_cast<int>(first_mismatch / width);
            const int column = static_cast<int>(first_mismatch % width);
            const float clean_value = clean(row, column);
            const float vendor_value = vendor(row, column);
            std::cout << "first_mismatch_index=" << first_mismatch
                      << " row=" << row << " column=" << column
                      << " clean=" << clean_value << " clean_bits=0x" << std::hex
                      << bits(clean_value) << " vendor=" << std::dec << vendor_value
                      << " vendor_bits=0x" << std::hex << bits(vendor_value)
                      << std::dec << '\n';
        } else {
            std::cout << "sparse_depth=EXACT\n";
        }
        const auto print_detail = [&](const char* label, std::size_t index) {
            if (index == clean.total()) {
                std::cout << label << "=none\n";
                return;
            }
            const int row = static_cast<int>(index / width);
            const int column = static_cast<int>(index % width);
            std::cout << label << "_index=" << index << " row=" << row
                      << " column=" << column << " clean=" << clean(row, column)
                      << " vendor=" << vendor(row, column);
            const std::uint64_t point_index = owner[index];
            if (point_index != std::numeric_limits<std::uint64_t>::max()) {
                std::cout << " clean_owner_point=" << point_index;
            }
            std::cout << '\n';
        };
        print_detail("first_clean_only", first_clean_only);
        print_detail("first_vendor_only", first_vendor_only);
        print_detail("first_common_depth", first_common_depth);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
