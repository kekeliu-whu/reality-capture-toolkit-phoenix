#include "navvis_recon/slam_rosbag.hpp"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace navvis_recon::slam {

std::vector<ImuSample> loadRawImuRosbag(
    const std::filesystem::path& bag_path, const std::string& topic,
    const std::optional<std::int64_t> end_timestamp_ns) {
  rosbag::Bag bag;
  try {
    bag.open(bag_path.string(), rosbag::bagmode::Read);
  } catch (const rosbag::BagException& error) {
    throw std::runtime_error("cannot open IMU rosbag " + bag_path.string() +
                             ": " + error.what());
  }

  std::vector<ImuSample> samples;
  rosbag::View view(bag, rosbag::TopicQuery({topic}));
  samples.reserve(view.size());
  for (const rosbag::MessageInstance& message_instance : view) {
    const sensor_msgs::Imu::ConstPtr message =
        message_instance.instantiate<sensor_msgs::Imu>();
    if (!message) {
      continue;
    }
    const std::int64_t timestamp_ns =
        static_cast<std::int64_t>(message->header.stamp.sec) * 1'000'000'000LL +
        static_cast<std::int64_t>(message->header.stamp.nsec);
    samples.push_back(ImuSample{
        timestamp_ns,
        Eigen::Vector3d(message->linear_acceleration.x,
                        message->linear_acceleration.y,
                        message->linear_acceleration.z),
        Eigen::Vector3d(message->angular_velocity.x,
                        message->angular_velocity.y,
                        message->angular_velocity.z),
        Eigen::Quaterniond(message->orientation.w, message->orientation.x,
                           message->orientation.y, message->orientation.z)});
    // The first sample after the requested endpoint is needed as the final
    // interpolation bracket, exactly as in the Python regression reader.
    if (end_timestamp_ns.has_value() &&
        timestamp_ns > *end_timestamp_ns) {
      break;
    }
  }
  bag.close();

  if (samples.size() < 2U) {
    throw std::runtime_error("IMU rosbag contains fewer than two samples on " +
                             topic);
  }
  for (std::size_t index = 1; index < samples.size(); ++index) {
    if (samples[index].timestamp_ns <= samples[index - 1U].timestamp_ns) {
      throw std::runtime_error("IMU samples are not strictly time ordered");
    }
  }
  return samples;
}

}  // namespace navvis_recon::slam
