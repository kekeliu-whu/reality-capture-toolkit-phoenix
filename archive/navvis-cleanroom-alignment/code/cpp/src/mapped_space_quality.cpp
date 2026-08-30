#include "navvis_recon/mapped_space_quality.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace fs = std::filesystem;

extern "C" {
std::size_t BrotliEncoderMaxCompressedSize(std::size_t input_size);
int BrotliEncoderCompress(
    int quality,
    int lgwin,
    int mode,
    std::size_t input_size,
    const std::uint8_t* input_buffer,
    std::size_t* encoded_size,
    std::uint8_t* encoded_buffer);
}

namespace navvis_recon {
namespace {

constexpr float kRangeResolution = 0.01F;
constexpr float kRangePower = 1.6F;
constexpr float kInverseRangePower = 0.625F;
constexpr float kMinimumDirectionWeight = 0.01F;
constexpr float kDirectionWeightSlope = -0.035357143729925156F;
constexpr float kDirectionWeightIntercept = 1.0707142353057861F;
constexpr float kDirectionWeightNormalizer = 255.0F;
constexpr float kDirectionalDiversityScale = 65535.0F;
constexpr double kSamplingFactor = 1.0 / 3.0;
constexpr double kFirstSampleFactor = 0.001;
constexpr double kMinimumDirectionNorm = 1.0e-6;
constexpr int kSpatialOffset = 1 << 20;
constexpr int kSpatialLimit = 1 << 20;
constexpr char kQualityHeader[] =
    "#NavVis GmbH binary file format for compressed voxel quality data. version: 2\n";
static_assert(sizeof(kQualityHeader) - 1U == 78U);

using SpatialIndex = std::array<int, 3>;

struct SpatialIndexHash {
    std::size_t operator()(const SpatialIndex& key) const noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const int value : key) {
            const std::uint32_t bits = static_cast<std::uint32_t>(value);
            hash ^= bits;
            hash *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(hash);
    }
};

struct QualityVoxelState {
    SpatialIndex spatial_index{};
    // Pair order is observable: the original compact conversion accumulates
    // float weights in first-direction-seen order.
    std::vector<std::pair<std::uint8_t, std::uint8_t>> minimum_range_per_direction;
    std::uint16_t ray_count = 0U;
};

std::uint64_t mortonKey(const SpatialIndex& spatial_index) {
    std::array<std::uint32_t, 3> shifted{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (spatial_index[axis] < -kSpatialLimit || spatial_index[axis] >= kSpatialLimit) {
            throw std::out_of_range("mapped-space quality voxel exceeds signed 21-bit range");
        }
        shifted[axis] = static_cast<std::uint32_t>(spatial_index[axis] + kSpatialOffset);
    }
    std::uint64_t key = 0U;
    for (unsigned bit = 0; bit < 21U; ++bit) {
        key |= static_cast<std::uint64_t>((shifted[0] >> bit) & 1U) << (3U * bit);
        key |= static_cast<std::uint64_t>((shifted[1] >> bit) & 1U) << (3U * bit + 1U);
        key |= static_cast<std::uint64_t>((shifted[2] >> bit) & 1U) << (3U * bit + 2U);
    }
    return key;
}

SpatialIndex spatialIndex(const Eigen::Vector3d& point, const double voxel_size) {
    SpatialIndex result{};
    const double inverse_voxel_size = 1.0 / voxel_size;
    for (int axis = 0; axis < 3; ++axis) {
        // QualityVoxelGridAggregatorHash caches the reciprocal and multiplies
        // before its manual floor. A per-coordinate division changes exact
        // boundary cells on full recordings.
        const double scaled = point[axis] * inverse_voxel_size;
        const double floored = std::floor(scaled);
        if (!std::isfinite(floored) || floored < std::numeric_limits<int>::min() ||
            floored > std::numeric_limits<int>::max()) {
            throw std::out_of_range("non-finite or out-of-range mapped-space quality point");
        }
        result[static_cast<std::size_t>(axis)] = static_cast<int>(floored);
    }
    return result;
}

Eigen::Vector3d voxelCenter(const SpatialIndex& key, const double voxel_size) {
    return Eigen::Vector3d(
        (static_cast<double>(key[0]) + 0.5) * voxel_size,
        (static_cast<double>(key[1]) + 0.5) * voxel_size,
        (static_cast<double>(key[2]) + 0.5) * voxel_size);
}

const std::array<std::uint8_t, 7001>& compressRangeLookup() {
    static const std::array<std::uint8_t, 7001> lookup = [] {
        std::array<std::uint8_t, 7001> result{};
        for (std::size_t index = 0; index < result.size(); ++index) {
            const float encoded = std::pow(static_cast<float>(index), kInverseRangePower);
            result[index] = static_cast<std::uint8_t>(std::floor(encoded + 0.5F));
        }
        return result;
    }();
    return lookup;
}

std::uint8_t compressRange(const double range) {
    const float range_float = static_cast<float>(range);
    const float scaled = range_float * 100.0F;
    int index = static_cast<int>(scaled);
    index = std::clamp(index, 0, 7000);
    const std::uint8_t compressed = compressRangeLookup()[static_cast<std::size_t>(index)];
    return compressed == 0U ? 1U : compressed;
}

float decompressRange(const std::uint8_t compressed) {
    return std::pow(static_cast<float>(compressed), kRangePower) * kRangeResolution;
}

const std::array<Eigen::Vector3f, 255>& sphericalFibonacciDirections() {
    static const std::array<Eigen::Vector3f, 255> directions = [] {
        std::array<Eigen::Vector3f, 255> result{};
        // These float constants and operations are part of the uint8
        // SphericalFibonacci contract. Evaluating the mathematically
        // equivalent expression in double changes a boundary ray bin.
        constexpr float reciprocal_golden_ratio = 0.6180340051651001F;
        constexpr float two_pi = 6.2831854820251465F;
        constexpr float count = 255.0F;
        for (std::size_t index = 0; index < result.size(); ++index) {
            const float value = static_cast<float>(index);
            // decode() forms the fractional golden-ratio turn with fmaf, not
            // fmodf, and divides by 255 rather than multiplying by its cached
            // reciprocal. Those single-rounding details decide rare Voronoi
            // boundary directions on complete recordings.
            const float product = value * reciprocal_golden_ratio;
            const float turns = std::fma(
                value, reciprocal_golden_ratio, -std::trunc(product));
            const float z = 1.0F - (2.0F * value + 1.0F) / count;
            const float angle = two_pi * turns;
            const float radius = std::sqrt(std::max(0.0F, 1.0F - z * z));
            result[index] = Eigen::Vector3f(
                radius * std::cos(angle),
                radius * std::sin(angle),
                z);
        }
        return result;
    }();
    return directions;
}

std::uint8_t encodeDirection(const MappedSpaceQualityRay& ray) {
    Eigen::Vector3d sensor_direction = ray.origin - ray.endpoint;
    const double norm = sensor_direction.norm();
    sensor_direction /= std::max(norm, kMinimumDirectionNorm);
    const Eigen::Vector3f direction = sensor_direction.cast<float>();
    const auto& candidates = sphericalFibonacciDirections();
    // The vendor encoder selects the decoded point by float32 squared
    // Euclidean distance. Maximising a dot product is only mathematically
    // equivalent when every decoded point has exactly unit length; float32
    // decoding makes a few full-recording boundary rays choose differently.
    float best_distance = std::numeric_limits<float>::infinity();
    std::uint8_t best = 0U;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const float distance = (candidates[index] - direction).squaredNorm();
        if (distance < best_distance) {
            best_distance = distance;
            best = static_cast<std::uint8_t>(index);
        }
    }
    return best;
}

std::uint16_t compressDirectionalDiversity(const float diversity) {
    if (!(diversity < 1.0F)) {
        return std::numeric_limits<std::uint16_t>::max();
    }
    const float scaled = diversity * kDirectionalDiversityScale;
    if (!(scaled <= kDirectionalDiversityScale) || scaled < 0.0F) {
        throw std::runtime_error("invalid mapped-space directional diversity");
    }
    return static_cast<std::uint16_t>(static_cast<int>(scaled));
}

CompactQualityVoxel compactVoxel(const QualityVoxelState& state) {
    CompactQualityVoxel result;
    result.spatial_key = mortonKey(state.spatial_index);
    result.ray_count = state.ray_count;
    float sum = 0.0F;
    std::uint8_t minimum_range = 255U;
    for (const auto& [direction, compressed_range] : state.minimum_range_per_direction) {
        static_cast<void>(direction);
        float weight = decompressRange(compressed_range) * kDirectionWeightSlope;
        weight += kDirectionWeightIntercept;
        if (weight < kMinimumDirectionWeight) {
            weight = kMinimumDirectionWeight;
        } else {
            weight = std::min(1.0F, weight);
        }
        sum += weight;
        minimum_range = std::min(minimum_range, compressed_range);
    }
    result.directional_diversity =
        compressDirectionalDiversity(sum / kDirectionWeightNormalizer);
    result.minimum_range = minimum_range;
    return result;
}

SpatialIndex decodeMortonKey(const std::uint64_t key) {
    std::array<std::uint32_t, 3> shifted{};
    for (unsigned bit = 0; bit < 21U; ++bit) {
        shifted[0] |= static_cast<std::uint32_t>((key >> (3U * bit)) & 1U) << bit;
        shifted[1] |= static_cast<std::uint32_t>((key >> (3U * bit + 1U)) & 1U) << bit;
        shifted[2] |= static_cast<std::uint32_t>((key >> (3U * bit + 2U)) & 1U) << bit;
    }
    return SpatialIndex{
        static_cast<int>(shifted[0]) - kSpatialOffset,
        static_cast<int>(shifted[1]) - kSpatialOffset,
        static_cast<int>(shifted[2]) - kSpatialOffset};
}

}  // namespace

struct MappedSpaceQualityGrid::Impl {
    explicit Impl(MappedSpaceQualityOptions requested_options)
        : options(std::move(requested_options)) {}

    MappedSpaceQualityOptions options;
    std::unordered_map<SpatialIndex, std::size_t, SpatialIndexHash> lookup;
    std::vector<QualityVoxelState> voxels;
    std::size_t input_ray_count = 0U;

    void addPoint(
        const Eigen::Vector3d& point,
        const Eigen::Vector3d& origin,
        const std::uint8_t direction) {
        const SpatialIndex key = spatialIndex(point, options.voxel_size);
        auto [iterator, inserted] = lookup.try_emplace(key, voxels.size());
        if (inserted) {
            QualityVoxelState state;
            state.spatial_index = key;
            voxels.push_back(std::move(state));
        }
        QualityVoxelState& voxel = voxels[iterator->second];
        if (voxel.ray_count != std::numeric_limits<std::uint16_t>::max()) {
            ++voxel.ray_count;
        }
        const std::uint8_t compressed_range = compressRange((point - origin).norm());
        const auto existing = std::find_if(
            voxel.minimum_range_per_direction.begin(),
            voxel.minimum_range_per_direction.end(),
            [direction](const auto& entry) { return entry.first == direction; });
        if (existing == voxel.minimum_range_per_direction.end()) {
            voxel.minimum_range_per_direction.emplace_back(direction, compressed_range);
        } else {
            existing->second = std::min(existing->second, compressed_range);
        }
    }
};

MappedSpaceQualityGrid::MappedSpaceQualityGrid(MappedSpaceQualityOptions options) {
    if (!(options.voxel_size > 0.0) || !std::isfinite(options.voxel_size) ||
        options.minimum_rays_per_voxel < 0 || options.use_every_nth_point <= 0 ||
        options.brotli_quality < 0 || options.brotli_quality > 11) {
        throw std::invalid_argument("invalid mapped-space quality options");
    }
    impl_ = std::make_unique<Impl>(std::move(options));
}

MappedSpaceQualityGrid::~MappedSpaceQualityGrid() = default;
MappedSpaceQualityGrid::MappedSpaceQualityGrid(MappedSpaceQualityGrid&&) noexcept = default;
MappedSpaceQualityGrid& MappedSpaceQualityGrid::operator=(MappedSpaceQualityGrid&&) noexcept = default;

void MappedSpaceQualityGrid::addRay(const MappedSpaceQualityRay& ray) {
    const std::size_t input_index = impl_->input_ray_count;
    addRayAtInputIndex(ray, input_index);
}

void MappedSpaceQualityGrid::addRayAtInputIndex(
    const MappedSpaceQualityRay& ray, const std::size_t input_index) {
    ++impl_->input_ray_count;
    if (input_index % static_cast<std::size_t>(impl_->options.use_every_nth_point) != 0U) {
        return;
    }
    if (!ray.origin.allFinite() || !ray.endpoint.allFinite()) {
        throw std::invalid_argument("mapped-space quality ray contains non-finite coordinates");
    }
    const Eigen::Vector3d delta = ray.endpoint - ray.origin;
    const double length = delta.norm();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    if (length > 0.0) {
        direction = delta / length;
    }
    const std::uint8_t encoded_direction = encodeDirection(ray);
    const SpatialIndex endpoint_key = spatialIndex(ray.endpoint, impl_->options.voxel_size);
    SpatialIndex previous_sample_key{};
    bool has_previous_sample = false;

    const double step = kSamplingFactor * impl_->options.voxel_size;
    for (double distance = kFirstSampleFactor * impl_->options.voxel_size;
         distance <= length;
         distance += step) {
        const Eigen::Vector3d sample = ray.origin + direction * distance;
        const SpatialIndex sample_key = spatialIndex(sample, impl_->options.voxel_size);
        if (has_previous_sample && sample_key == previous_sample_key) {
            continue;
        }
        Eigen::Vector3d contribution = ray.endpoint;
        if (sample_key != endpoint_key) {
            const Eigen::Vector3d center = voxelCenter(sample_key, impl_->options.voxel_size);
            const double projection_distance = (center - ray.origin).dot(direction);
            const Eigen::Vector3d projection = ray.origin + direction * projection_distance;
            if ((projection - ray.origin).dot(delta) > 0.0) {
                contribution = projection;
            } else {
                contribution = sample;
            }
        }
        impl_->addPoint(contribution, ray.origin, encoded_direction);
        previous_sample_key = sample_key;
        has_previous_sample = true;
    }
    if (!has_previous_sample || previous_sample_key != endpoint_key) {
        impl_->addPoint(ray.endpoint, ray.origin, encoded_direction);
    }
}

void MappedSpaceQualityGrid::mergeLaterChunk(MappedSpaceQualityGrid&& later) {
    if (impl_.get() == later.impl_.get()) {
        throw std::invalid_argument("cannot merge a mapped-space grid into itself");
    }
    const auto& first_options = impl_->options;
    const auto& later_options = later.impl_->options;
    if (first_options.voxel_size != later_options.voxel_size ||
        first_options.minimum_rays_per_voxel != later_options.minimum_rays_per_voxel ||
        first_options.use_every_nth_point != later_options.use_every_nth_point ||
        first_options.brotli_quality != later_options.brotli_quality) {
        throw std::invalid_argument("cannot merge mapped-space grids with different options");
    }
    if (impl_->input_ray_count == 0U && impl_->voxels.empty()) {
        impl_ = std::move(later.impl_);
        return;
    }

    // Chunks are merged in original input order.  Appending newly observed
    // voxels and direction bins in the later chunk's insertion order therefore
    // preserves the serial accumulator's observable record and float-sum order.
    for (QualityVoxelState& source : later.impl_->voxels) {
        auto [iterator, inserted] =
            impl_->lookup.try_emplace(source.spatial_index, impl_->voxels.size());
        if (inserted) {
            impl_->voxels.push_back(std::move(source));
            continue;
        }

        QualityVoxelState& destination = impl_->voxels[iterator->second];
        const std::uint32_t combined_count =
            static_cast<std::uint32_t>(destination.ray_count) + source.ray_count;
        destination.ray_count = static_cast<std::uint16_t>(std::min<std::uint32_t>(
            combined_count, std::numeric_limits<std::uint16_t>::max()));
        for (const auto& [direction, compressed_range] : source.minimum_range_per_direction) {
            const auto existing = std::find_if(
                destination.minimum_range_per_direction.begin(),
                destination.minimum_range_per_direction.end(),
                [direction](const auto& entry) { return entry.first == direction; });
            if (existing == destination.minimum_range_per_direction.end()) {
                destination.minimum_range_per_direction.emplace_back(direction, compressed_range);
            } else {
                existing->second = std::min(existing->second, compressed_range);
            }
        }
    }
    impl_->input_ray_count += later.impl_->input_ray_count;
}

std::vector<CompactQualityVoxel> MappedSpaceQualityGrid::compact() const {
    std::vector<CompactQualityVoxel> result;
    result.reserve(impl_->voxels.size());
    const float scale = static_cast<float>(impl_->options.use_every_nth_point);
    for (const QualityVoxelState& source : impl_->voxels) {
        QualityVoxelState scaled = source;
        const float scaled_count = std::round(static_cast<float>(source.ray_count) * scale);
        scaled.ray_count = static_cast<std::uint16_t>(std::min(
            static_cast<int>(scaled_count),
            static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
        if (scaled.ray_count >= impl_->options.minimum_rays_per_voxel) {
            result.push_back(compactVoxel(scaled));
        }
    }
    return result;
}

std::size_t MappedSpaceQualityGrid::inputRayCount() const noexcept {
    return impl_->input_ray_count;
}

const MappedSpaceQualityOptions& MappedSpaceQualityGrid::options() const noexcept {
    return impl_->options;
}

void writeMappedSpaceQuality(
    const fs::path& output_directory,
    const std::vector<CompactQualityVoxel>& voxels,
    const MappedSpaceQualityOptions& options) {
    if (voxels.empty()) {
        throw std::runtime_error("mapped-space quality filter produced no voxels");
    }
    fs::create_directories(output_directory);

    const std::size_t raw_size = voxels.size() * sizeof(CompactQualityVoxel);
    const auto* raw = reinterpret_cast<const std::uint8_t*>(voxels.data());
    std::vector<std::uint8_t> compressed(BrotliEncoderMaxCompressedSize(raw_size));
    std::size_t compressed_size = compressed.size();
    constexpr int kBrotliDefaultWindow = 22;
    constexpr int kBrotliGenericMode = 0;
    if (!BrotliEncoderCompress(
            options.brotli_quality,
            kBrotliDefaultWindow,
            kBrotliGenericMode,
            raw_size,
            raw,
            &compressed_size,
            compressed.data())) {
        throw std::runtime_error("Brotli compression failed for mapped-space quality payload");
    }
    compressed.resize(compressed_size);
    const fs::path binary_path = output_directory / "quality_voxels.bin";
    std::ofstream binary(binary_path, std::ios::binary);
    if (!binary) {
        throw std::runtime_error("cannot create " + binary_path.string());
    }
    binary.write(kQualityHeader, static_cast<std::streamsize>(sizeof(kQualityHeader) - 1U));
    binary.write(
        reinterpret_cast<const char*>(compressed.data()),
        static_cast<std::streamsize>(compressed.size()));
    binary.close();
    if (!binary) {
        throw std::runtime_error("failed while writing " + binary_path.string());
    }

    SpatialIndex minimum = decodeMortonKey(voxels.front().spatial_key);
    SpatialIndex maximum = minimum;
    for (const CompactQualityVoxel& voxel : voxels) {
        const SpatialIndex key = decodeMortonKey(voxel.spatial_key);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], key[axis]);
            maximum[axis] = std::max(maximum[axis], key[axis]);
        }
    }
    const auto lower_corner = [&minimum, &options](const std::size_t axis) {
        const double center =
            (static_cast<double>(minimum[axis]) + 0.5) * options.voxel_size;
        return center - options.voxel_size;
    };
    const auto upper_corner = [&maximum, &options](const std::size_t axis) {
        const double center =
            (static_cast<double>(maximum[axis]) + 0.5) * options.voxel_size;
        return center + options.voxel_size;
    };
    const std::uintmax_t total_compressed_bytes = fs::file_size(binary_path);
    const fs::path sidecar_path = output_directory / "quality_voxels_sidecar.json";
    std::ofstream sidecar(sidecar_path);
    if (!sidecar) {
        throw std::runtime_error("cannot create " + sidecar_path.string());
    }
    sidecar << std::setprecision(17)
            << "{\n"
            << "  \"quality_grid_format_version\": 2,\n"
            << "  \"voxel_size\": " << options.voxel_size << ",\n"
            << "  \"num_voxels\": " << voxels.size() << ",\n"
            << "  \"min_corner_x\": " << lower_corner(0) << ",\n"
            << "  \"min_corner_y\": " << lower_corner(1) << ",\n"
            << "  \"min_corner_z\": " << lower_corner(2) << ",\n"
            << "  \"max_corner_x\": " << upper_corner(0) << ",\n"
            << "  \"max_corner_y\": " << upper_corner(1) << ",\n"
            << "  \"max_corner_z\": " << upper_corner(2) << ",\n"
            << "  \"dataset_T_quality\": {\n"
            << "    \"translation\": {\n"
            << "      \"x\": -0,\n"
            << "      \"y\": -0,\n"
            << "      \"z\": -0\n"
            << "    },\n"
            << "    \"rotation\": {\n"
            << "      \"w\": 1,\n"
            << "      \"x\": -0,\n"
            << "      \"y\": -0,\n"
            << "      \"z\": -0\n"
            << "    }\n"
            << "  },\n"
            << "  \"bytes_uncompressed\": " << raw_size << ",\n"
            << "  \"bytes_compressed\": " << total_compressed_bytes << "\n"
            << "}\n";
    sidecar.close();
    if (!sidecar) {
        throw std::runtime_error("failed while writing " + sidecar_path.string());
    }

    const fs::path pcd_path = output_directory / "mapped_space.pcd";
    std::ofstream pcd(pcd_path, std::ios::binary);
    if (!pcd) {
        throw std::runtime_error("cannot create " + pcd_path.string());
    }
    pcd << "# .PCD v0.7 - Point Cloud Data file format\n"
        << "VERSION 0.7\n"
        << "FIELDS x y z intensity normal_x normal_y normal_z curvature\n"
        << "SIZE 4 4 4 4 4 4 4 4\n"
        << "TYPE F F F F F F F F\n"
        << "COUNT 1 1 1 1 1 1 1 1\n"
        << "WIDTH " << voxels.size() << "\n"
        << "HEIGHT 1\n"
        << "VIEWPOINT 0 0 0 1 0 0 0\n"
        << "POINTS " << voxels.size() << "\n"
        << "DATA binary\n";
    for (const CompactQualityVoxel& voxel : voxels) {
        const SpatialIndex key = decodeMortonKey(voxel.spatial_key);
        const Eigen::Vector3d center = voxelCenter(key, options.voxel_size);
        const std::array<float, 8> point{
            static_cast<float>(center.x()),
            static_cast<float>(center.y()),
            static_cast<float>(center.z()),
            static_cast<float>(voxel.directional_diversity) / kDirectionalDiversityScale,
            0.0F,
            0.0F,
            0.0F,
            static_cast<float>(voxel.ray_count)};
        pcd.write(
            reinterpret_cast<const char*>(point.data()),
            static_cast<std::streamsize>(sizeof(point)));
    }
    pcd.close();
    if (!pcd) {
        throw std::runtime_error("failed while writing " + pcd_path.string());
    }
}

}  // namespace navvis_recon
