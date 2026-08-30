#pragma once

#include "navvis_recon/slam_imu.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace navvis_recon::slam {

// Read the platform-neutral little-endian IMU handoff emitted by the Python
// runner.  Samples retain their ROS header timestamp and raw float64 values.
std::vector<ImuSample> loadRawImuFile(
    const std::filesystem::path& path,
    std::optional<std::int64_t> end_timestamp_ns = std::nullopt);

}  // namespace navvis_recon::slam
