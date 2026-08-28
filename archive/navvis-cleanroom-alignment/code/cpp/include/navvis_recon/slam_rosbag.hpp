#pragma once

#include "navvis_recon/slam_imu.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace navvis_recon::slam {

std::vector<ImuSample> loadRawImuRosbag(
    const std::filesystem::path& bag_path,
    const std::string& topic = "/imu/imu_raw/data",
    std::optional<std::int64_t> end_timestamp_ns = std::nullopt);

}  // namespace navvis_recon::slam
