#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#pragma pack(push, 1)
struct Point {
    float x, y, z;
    std::uint8_t red, green, blue, alpha;
    float intensity, nx, ny, nz, curvature;
};
#pragma pack(pop)
static_assert(sizeof(Point) == 36);

struct Neighbor {
    int row = -1;
    float squared_distance = std::numeric_limits<float>::infinity();
};

struct NeighborList {
    std::array<Neighbor, 5> values{};
    int count = 0;
};

class ExactKdTree {
  public:
    explicit ExactKdTree(const std::vector<Point>& points) : points_(&points) {
        std::vector<int> rows(points.size());
        std::iota(rows.begin(), rows.end(), 0);
        nodes_.reserve(points.size());
        root_ = build(rows, 0, rows.size(), 0);
    }

    NeighborList nearestFive(const Point& query) const {
        NeighborList result;
        search(root_, query, result);
        return result;
    }

  private:
    struct Node {
        int row = -1;
        int left = -1;
        int right = -1;
        int axis = 0;
    };

    static float coordinate(const Point& point, int axis) {
        return axis == 0 ? point.x : axis == 1 ? point.y : point.z;
    }

    static bool nearer(const Neighbor& first, const Neighbor& second) {
        return first.squared_distance < second.squared_distance ||
               // PCL's OctreePointCloudSearch retains the later candidate
               // at an equal float32 distance. The captured fifth-neighbor
               // tie at query 2296953 selects original row 2307256 rather
               // than the earlier equidistant row 2306411.
               (first.squared_distance == second.squared_distance && first.row > second.row);
    }

    int build(std::vector<int>& rows, std::size_t begin, std::size_t end, int depth) {
        if (begin >= end) {
            return -1;
        }
        const int axis = depth % 3;
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(rows.begin() + static_cast<std::ptrdiff_t>(begin),
                         rows.begin() + static_cast<std::ptrdiff_t>(middle),
                         rows.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](int first, int second) {
                             const float a = coordinate((*points_)[static_cast<std::size_t>(first)], axis);
                             const float b = coordinate((*points_)[static_cast<std::size_t>(second)], axis);
                             return a < b || (a == b && first < second);
                         });
        const int node = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{rows[middle], -1, -1, axis});
        const int left = build(rows, begin, middle, depth + 1);
        const int right = build(rows, middle + 1, end, depth + 1);
        nodes_[static_cast<std::size_t>(node)].left = left;
        nodes_[static_cast<std::size_t>(node)].right = right;
        return node;
    }

    static void insert(NeighborList& result, Neighbor candidate) {
        int position = 0;
        while (position < result.count &&
               !nearer(candidate, result.values[static_cast<std::size_t>(position)])) {
            ++position;
        }
        if (position >= 5) {
            return;
        }
        const int count = std::min(5, result.count + 1);
        for (int index = count - 1; index > position; --index) {
            result.values[static_cast<std::size_t>(index)] =
                result.values[static_cast<std::size_t>(index - 1)];
        }
        result.values[static_cast<std::size_t>(position)] = candidate;
        result.count = count;
    }

    void search(int node_index, const Point& query, NeighborList& result) const {
        if (node_index < 0) {
            return;
        }
        const Node& node = nodes_[static_cast<std::size_t>(node_index)];
        const Point& sample = (*points_)[static_cast<std::size_t>(node.row)];
        const float dx = query.x - sample.x;
        const float dy = query.y - sample.y;
        const float dz = query.z - sample.z;
        // Eigen/PCL reduces the final two components first.
        const float distance = dx * dx + (dy * dy + dz * dz);
        insert(result, Neighbor{node.row, distance});

        const float split = coordinate(query, node.axis) - coordinate(sample, node.axis);
        const int near_node = split < 0.0F ? node.left : node.right;
        const int far_node = split < 0.0F ? node.right : node.left;
        search(near_node, query, result);
        const float worst = result.count < 5 ? std::numeric_limits<float>::infinity()
                                              : result.values.back().squared_distance;
        const double split_squared = static_cast<double>(split) * split;
        if (split_squared <=
            std::nextafter(static_cast<double>(worst), std::numeric_limits<double>::infinity())) {
            search(far_node, query, result);
        }
    }

    const std::vector<Point>* points_;
    std::vector<Node> nodes_;
    int root_ = -1;
};

std::uint64_t readPly(const std::string& path, std::vector<Point>& points) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open PLY: " + path);
    }
    std::string line;
    std::uint64_t count = 0;
    while (std::getline(input, line)) {
        if (line.rfind("element vertex ", 0) == 0) {
            count = std::stoull(line.substr(std::strlen("element vertex ")));
        }
        if (line == "end_header") {
            break;
        }
    }
    if (count == 0) {
        throw std::runtime_error("invalid PLY header");
    }
    points.resize(static_cast<std::size_t>(count));
    input.read(reinterpret_cast<char*>(points.data()),
               static_cast<std::streamsize>(points.size() * sizeof(Point)));
    if (!input) {
        throw std::runtime_error("short PLY body");
    }
    return count;
}

std::vector<std::int32_t> readIndices(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open index file: " + path);
    }
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(std::int32_t)) != 0) {
        throw std::runtime_error("invalid index file size");
    }
    input.seekg(0);
    std::vector<std::int32_t> indices(
        static_cast<std::size_t>(bytes / static_cast<std::streamoff>(sizeof(std::int32_t))));
    input.read(reinterpret_cast<char*>(indices.data()), bytes);
    if (!input) {
        throw std::runtime_error("short index file");
    }
    return indices;
}

std::uint8_t byte(float value) {
    return static_cast<std::uint8_t>(
        std::clamp<long>(std::max<long>(1, std::lrintf(value)), 1, 255));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "usage: gamma_knn_exact_replay VENDOR_PLY COLORED_I32 UNCOLORED_I32 "
                         "[MISMATCH_CSV]\n";
            return 2;
        }
        std::fesetround(FE_TONEAREST);
        std::vector<Point> reference;
        const std::uint64_t count = readPly(argv[1], reference);
        const auto colored_indices = readIndices(argv[2]);
        const auto uncolored_indices = readIndices(argv[3]);
        if (colored_indices.size() + uncolored_indices.size() != count) {
            throw std::runtime_error("KNN partition size mismatch");
        }

        std::vector<Point> seeds;
        seeds.reserve(colored_indices.size());
        for (std::int32_t index : colored_indices) {
            if (index < 0 || static_cast<std::uint64_t>(index) >= count) {
                throw std::runtime_error("colored index out of range");
            }
            seeds.push_back(reference[static_cast<std::size_t>(index)]);
        }
        const ExactKdTree tree(seeds);

        std::uint64_t exact_points = 0;
        std::uint64_t within_one_points = 0;
        std::uint64_t absolute_sum = 0;
        std::uint64_t changed_channels = 0;
        int maximum_delta = 0;
        std::vector<std::string> mismatch_rows;

#pragma omp parallel for schedule(static) reduction(+ : exact_points, within_one_points, absolute_sum, changed_channels) reduction(max : maximum_delta)
        for (std::int64_t row = 0; row < static_cast<std::int64_t>(uncolored_indices.size()); ++row) {
            const Point& query = reference[static_cast<std::size_t>(
                uncolored_indices[static_cast<std::size_t>(row)])];
            const NeighborList neighbors = tree.nearestFive(query);
            std::array<float, 3> weighted{};
            std::array<float, 5> target_weights{};
            float total_weight = 0.0F;
            for (std::size_t rank = 0; rank < neighbors.values.size(); ++rank) {
                const Neighbor& neighbor = neighbors.values[rank];
                if (neighbor.row < 0 || neighbor.squared_distance < 1.0e-6F ||
                    neighbor.squared_distance > 10000.0F) {
                    continue;
                }
                const Point& sample = seeds[static_cast<std::size_t>(neighbor.row)];
                const float dz_nz = (query.z - sample.z) * query.nz;
                const float dy_ny = (query.y - sample.y) * query.ny;
                const float dx_nx = (query.x - sample.x) * query.nx;
                const float plane_distance = std::fabs((dz_nz + dy_ny) + dx_nx);
                const float nz_nz = sample.nz * query.nz;
                const float ny_ny = sample.ny * query.ny;
                const float nx_nx = sample.nx * query.nx;
                const float alignment = std::max(0.0F, (nz_nz + ny_ny) + nx_nx);
                const float denominator =
                    neighbor.squared_distance * std::max(1.0e-6F, plane_distance);
                const float weight = std::max(1.0e-6F, alignment / denominator);
                weighted[0] += static_cast<float>(sample.red) * weight;
                weighted[1] += static_cast<float>(sample.green) * weight;
                weighted[2] += static_cast<float>(sample.blue) * weight;
                total_weight += weight;
                if (uncolored_indices[static_cast<std::size_t>(row)] == 2336288) {
                    target_weights[rank] = weight;
                }
            }
            // nv_colorcloud computes one float reciprocal and reuses it for
            // all three channels (0x252dda..0x252dec), rather than issuing
            // three independent divisions. The two forms are algebraically
            // equivalent but differ by one ULP around a few uchar thresholds.
            const float inverse_weight = 1.0F / total_weight;
            if (uncolored_indices[static_cast<std::size_t>(row)] == 2336288) {
                std::array<std::uint32_t, 10> bits{};
                std::memcpy(&bits[0], &weighted[0], sizeof(float));
                std::memcpy(&bits[1], &weighted[1], sizeof(float));
                std::memcpy(&bits[2], &weighted[2], sizeof(float));
                std::memcpy(&bits[3], &total_weight, sizeof(float));
                std::memcpy(&bits[4], &inverse_weight, sizeof(float));
                for (std::size_t rank = 0; rank < target_weights.size(); ++rank) {
                    std::memcpy(&bits[5 + rank], &target_weights[rank], sizeof(float));
                }
#pragma omp critical(gamma_knn_target_trace)
                std::cerr << std::hex << "TARGET_2336288 weighted_rgb_bits=" << bits[0] << ','
                          << bits[1] << ',' << bits[2] << " total_weight_bits=" << bits[3]
                          << " inverse_weight_bits=" << bits[4] << " weight_bits=" << bits[5]
                          << ',' << bits[6] << ',' << bits[7] << ',' << bits[8] << ',' << bits[9]
                          << " neighbors=";
                for (std::size_t rank = 0; rank < neighbors.values.size(); ++rank) {
                    const Neighbor& neighbor = neighbors.values[rank];
                    std::uint32_t distance_bits = 0;
                    std::memcpy(&distance_bits, &neighbor.squared_distance, sizeof(float));
                    if (rank != 0) {
                        std::cerr << ';';
                    }
                    std::cerr << colored_indices[static_cast<std::size_t>(neighbor.row)] << ':'
                              << distance_bits;
                }
                std::cerr << std::dec << '\n';
            }
            const std::array<std::uint8_t, 3> candidate{
                byte(weighted[0] * inverse_weight), byte(weighted[1] * inverse_weight),
                byte(weighted[2] * inverse_weight)};
            const std::array<std::uint8_t, 3> expected{query.red, query.green, query.blue};
            bool exact = true;
            bool within_one = true;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const int delta = std::abs(static_cast<int>(candidate[channel]) -
                                           static_cast<int>(expected[channel]));
                absolute_sum += static_cast<std::uint64_t>(delta);
                changed_channels += delta != 0;
                maximum_delta = std::max(maximum_delta, delta);
                exact &= delta == 0;
                within_one &= delta <= 1;
            }
            exact_points += exact;
            within_one_points += within_one;
            if (!exact && argc == 5) {
                std::ostringstream line;
                line << uncolored_indices[static_cast<std::size_t>(row)] << ','
                     << static_cast<int>(expected[0]) << ':' << static_cast<int>(expected[1])
                     << ':' << static_cast<int>(expected[2]) << ','
                     << static_cast<int>(candidate[0]) << ':' << static_cast<int>(candidate[1])
                     << ':' << static_cast<int>(candidate[2]) << ',';
                for (std::size_t rank = 0; rank < neighbors.values.size(); ++rank) {
                    if (rank != 0) {
                        line << ';';
                    }
                    const Neighbor& neighbor = neighbors.values[rank];
                    line << colored_indices[static_cast<std::size_t>(neighbor.row)] << ':'
                         << std::hexfloat << neighbor.squared_distance << std::defaultfloat;
                }
#pragma omp critical(gamma_knn_mismatch_rows)
                mismatch_rows.push_back(line.str());
            }
        }

        if (argc == 5) {
            std::sort(mismatch_rows.begin(), mismatch_rows.end());
            std::ofstream mismatch_output(argv[4]);
            mismatch_output << "point,reference_rgb,candidate_rgb,neighbors_and_squared_distances\n";
            for (const std::string& row : mismatch_rows) {
                mismatch_output << row << '\n';
            }
        }

        const double values = static_cast<double>(uncolored_indices.size()) * 3.0;
        std::cout << "{\n"
                  << "  \"points\": " << count << ",\n"
                  << "  \"direct_seed_points\": " << colored_indices.size() << ",\n"
                  << "  \"fallback_points\": " << uncolored_indices.size() << ",\n"
                  << "  \"fallback_rgb_mae\": " << absolute_sum / values << ",\n"
                  << "  \"fallback_exact_point_fraction\": "
                  << static_cast<double>(exact_points) / uncolored_indices.size() << ",\n"
                  << "  \"fallback_max_channel_le_1_fraction\": "
                  << static_cast<double>(within_one_points) / uncolored_indices.size() << ",\n"
                  << "  \"changed_channels\": " << changed_channels << ",\n"
                  << "  \"maximum_delta\": " << maximum_delta << "\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gamma_knn_exact_replay: " << error.what() << '\n';
        return 1;
    }
}
