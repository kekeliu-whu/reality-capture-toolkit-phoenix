#include "navvis_recon/slam_rosbag.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: slam_rosbag_test IMU_BAG\n";
      return EXIT_FAILURE;
    }
    const auto samples = navvis_recon::slam::loadRawImuRosbag(argv[1]);
    std::cout << std::setprecision(17) << "PASS count=" << samples.size()
              << " first=" << samples.front().timestamp_ns
              << " last=" << samples.back().timestamp_ns
              << " first_accel="
              << samples.front().linear_acceleration.transpose()
              << " first_gyro=" << samples.front().angular_velocity.transpose()
              << " first_quat_xyzw=" << samples.front().orientation.x() << ' '
              << samples.front().orientation.y() << ' '
              << samples.front().orientation.z() << ' '
              << samples.front().orientation.w() << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
