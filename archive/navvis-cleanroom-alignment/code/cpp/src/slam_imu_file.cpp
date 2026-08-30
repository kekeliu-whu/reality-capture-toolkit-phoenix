#include "navvis_recon/slam_imu_file.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace navvis_recon::slam {
namespace {

constexpr std::array<char, 16> kMagic{
    'N', 'V', 'C', 'R', 'R', 'A', 'W', 'I', 'M', 'U', '0', '1',
    '\0', '\0', '\0', '\0'};
constexpr std::uint64_t kMaximumSamples = 100'000'000ULL;

template <typename Value>
Value readScalar(std::ifstream& input, const std::filesystem::path& path,
                 const char* label) {
  static_assert(std::is_trivially_copyable_v<Value>);
  Value value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) {
    throw std::runtime_error("truncated IMU file while reading " +
                             std::string(label) + ": " + path.string());
  }
  return value;
}

}  // namespace

std::vector<ImuSample> loadRawImuFile(
    const std::filesystem::path& path,
    const std::optional<std::int64_t> end_timestamp_ns) {
  const std::uint16_t endian_probe = 1U;
  if (*reinterpret_cast<const std::uint8_t*>(&endian_probe) != 1U) {
    throw std::runtime_error("raw IMU handoff requires a little-endian host");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open raw IMU file " + path.string());
  }
  std::array<char, kMagic.size()> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != kMagic) {
    throw std::runtime_error("invalid raw IMU file magic: " + path.string());
  }
  const std::uint64_t count = readScalar<std::uint64_t>(input, path, "count");
  if (count > kMaximumSamples ||
      count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("invalid raw IMU sample count in " + path.string());
  }

  std::vector<ImuSample> samples;
  samples.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    const std::int64_t timestamp_ns =
        readScalar<std::int64_t>(input, path, "timestamp");
    std::array<double, 10> values{};
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(sizeof(values)));
    if (!input) {
      throw std::runtime_error("truncated raw IMU sample in " + path.string());
    }
    samples.push_back(ImuSample{
        timestamp_ns,
        Eigen::Vector3d(values[0], values[1], values[2]),
        Eigen::Vector3d(values[3], values[4], values[5]),
        Eigen::Quaterniond(values[9], values[6], values[7], values[8])});
    // Retain the first sample beyond the lidar endpoint as interpolation
    // bracket, matching the ROS1 reader exactly.
    if (end_timestamp_ns.has_value() && timestamp_ns > *end_timestamp_ns) {
      break;
    }
  }

  if (samples.size() < 2U) {
    throw std::runtime_error("raw IMU file contains fewer than two samples: " +
                             path.string());
  }
  for (std::size_t index = 1; index < samples.size(); ++index) {
    if (samples[index].timestamp_ns <= samples[index - 1U].timestamp_ns) {
      throw std::runtime_error("raw IMU samples are not strictly time ordered");
    }
  }
  return samples;
}

}  // namespace navvis_recon::slam
