#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

struct ColoredPoint {
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
static_assert(sizeof(ColoredPoint) == 36);

struct SurfacePoint {
    std::array<float, 8> fields;
};
static_assert(sizeof(SurfacePoint) == 32);

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::vector<std::array<float, 3>> fibonacciDirections() {
    std::vector<std::array<float, 3>> result(65535);
    constexpr float golden_fraction = 0.6180340051651001F;
    constexpr float two_pi = 6.2831854820251465F;
    constexpr float code_scale = 65535.0F;
    for (int code = 0; code < 65535; ++code) {
        const float value = static_cast<float>(code);
        const float turns = ::fmaf(value, golden_fraction, -::truncf(value * golden_fraction));
        const float angle = turns * two_pi;
        float sine = 0.0F;
        float cosine = 0.0F;
        ::sincosf(angle, &sine, &cosine);
        const float z = 1.0F - ((value + value) + 1.0F) / code_scale;
        const float radius = ::sqrtf(std::max(0.0F, 1.0F - z * z));
        result[code] = {radius * cosine, radius * sine, z};
    }
    return result;
}

const std::array<float, 3>& nearestDirection(
    const std::array<float, 3>& query,
    const std::vector<std::array<float, 3>>& directions) {
    int best = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int code = 0; code < static_cast<int>(directions.size()); ++code) {
        const double dx = static_cast<double>(query[0]) - directions[code][0];
        const double dy = static_cast<double>(query[1]) - directions[code][1];
        const double dz = static_cast<double>(query[2]) - directions[code][2];
        const double distance = dx * dx + dy * dy + dz * dz;
        if (distance < best_distance) {
            best = code;
            best_distance = distance;
        }
    }
    return directions[best];
}

std::array<float, 3> finishNormal(const std::array<float, 3>& sum, int count) {
    const float divisor = static_cast<float>(count);
    std::array<float, 3> result{sum[0] / divisor, sum[1] / divisor, sum[2] / divisor};
    float squared_norm = result[2] * result[2];
    squared_norm += result[1] * result[1];
    squared_norm += result[0] * result[0];
    if (squared_norm < 1.0e-6F || !std::isfinite(squared_norm)) {
        return {0.0F, 0.0F, 0.0F};
    }
    if (std::abs(squared_norm - 1.0F) > 1.0e-6F) {
        const float norm = std::sqrt(squared_norm);
        result[0] /= norm;
        result[1] /= norm;
        result[2] /= norm;
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: pct_normal_quantization_probe INPUT.ply OFFICIAL_SURFELS "
                     "SURFEL_INDEX_A SURFEL_INDEX_B\n";
        return 2;
    }
    const std::string cloud_path = argv[1];
    const std::string surfel_path = argv[2];
    const std::array<std::size_t, 2> target_indices{
        static_cast<std::size_t>(std::stoull(argv[3])),
        static_cast<std::size_t>(std::stoull(argv[4]))};

    std::ifstream cloud(cloud_path, std::ios::binary);
    std::string line;
    std::uint64_t point_count = 0;
    while (std::getline(cloud, line)) {
        std::smatch match;
        if (std::regex_match(line, match, std::regex(R"(^element vertex ([0-9]+)$)"))) {
            point_count = std::stoull(match[1].str());
        }
        if (line == "end_header") {
            break;
        }
    }
    const std::streamoff cloud_offset = cloud.tellg();
    if (point_count == 0) {
        throw std::runtime_error("missing PLY vertex count");
    }

    std::array<float, 3> minimum{
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    std::array<float, 3> maximum{
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    ColoredPoint point{};
    for (std::uint64_t index = 0; index < point_count; ++index) {
        cloud.read(reinterpret_cast<char*>(&point), sizeof(point));
        const std::array<float, 3> xyz{point.x, point.y, point.z};
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], xyz[axis]);
            maximum[axis] = std::max(maximum[axis], xyz[axis]);
        }
    }

    constexpr double cell_size = 0.06;
    constexpr double epsilon = 6.103515625e-05;
    double maximum_extent = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        maximum_extent =
            std::max(maximum_extent, static_cast<double>(maximum[axis]) - minimum[axis]);
    }
    maximum_extent += epsilon;
    const int depth = static_cast<int>(std::ceil(std::log2(maximum_extent / cell_size)));
    const double side = std::ldexp(cell_size, depth);
    std::array<double, 3> cube_minimum{};
    for (int axis = 0; axis < 3; ++axis) {
        cube_minimum[axis] = 0.5 * (static_cast<double>(minimum[axis]) + maximum[axis] + epsilon - side);
    }

    std::ifstream surfels(surfel_path, std::ios::binary);
    std::array<SurfacePoint, 2> targets{};
    std::array<std::array<int, 3>, 2> keys{};
    for (int target = 0; target < 2; ++target) {
        surfels.seekg(static_cast<std::streamoff>(target_indices[target] * sizeof(SurfacePoint)));
        surfels.read(reinterpret_cast<char*>(&targets[target]), sizeof(SurfacePoint));
        for (int axis = 0; axis < 3; ++axis) {
            keys[target][axis] = static_cast<int>(
                (static_cast<double>(targets[target].fields[axis]) - cube_minimum[axis]) /
                cell_size);
        }
    }

    const auto directions = fibonacciDirections();
    std::array<std::array<float, 3>, 2> raw_sums{};
    std::array<std::array<float, 3>, 2> quantized_sums{};
    std::array<int, 2> counts{};
    cloud.clear();
    cloud.seekg(cloud_offset);
    for (std::uint64_t index = 0; index < point_count; ++index) {
        cloud.read(reinterpret_cast<char*>(&point), sizeof(point));
        std::array<int, 3> key{};
        const std::array<float, 3> xyz{point.x, point.y, point.z};
        for (int axis = 0; axis < 3; ++axis) {
            key[axis] = static_cast<int>(
                (static_cast<double>(xyz[axis]) - cube_minimum[axis]) / cell_size);
        }
        for (int target = 0; target < 2; ++target) {
            if (key != keys[target]) {
                continue;
            }
            const std::array<float, 3> normal{point.nx, point.ny, point.nz};
            const auto& quantized = nearestDirection(normal, directions);
            for (int axis = 0; axis < 3; ++axis) {
                raw_sums[target][axis] += normal[axis];
                quantized_sums[target][axis] += quantized[axis];
            }
            ++counts[target];
        }
    }

    for (int target = 0; target < 2; ++target) {
        const auto raw = finishNormal(raw_sums[target], counts[target]);
        const auto quantized = finishNormal(quantized_sums[target], counts[target]);
        std::cout << "surfel=" << target_indices[target] << " count=" << counts[target]
                  << " official=";
        for (int axis = 4; axis < 7; ++axis) {
            std::cout << " 0x" << std::hex << std::setw(8) << std::setfill('0')
                      << bits(targets[target].fields[axis]);
        }
        std::cout << " raw=";
        for (float value : raw) {
            std::cout << " 0x" << std::hex << std::setw(8) << std::setfill('0') << bits(value);
        }
        std::cout << " quantized=";
        for (float value : quantized) {
            std::cout << " 0x" << std::hex << std::setw(8) << std::setfill('0') << bits(value);
        }
        std::cout << std::dec << '\n';
    }
}
