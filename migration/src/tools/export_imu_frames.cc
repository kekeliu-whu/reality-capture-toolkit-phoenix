#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "migration/proto_io.h"

namespace fs = std::filesystem;

namespace {

struct PcdPoint {
  float x;
  float y;
  float z;
  float intensity;
  double timestamp;
};

static_assert(sizeof(PcdPoint) == 24, "Unexpected PCD point layout");

bool WriteBinaryPcd(const fs::path& filename, const proto::LidarMsg& scan) {
  std::ofstream output(filename, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    std::cerr << "Failed to create PCD: " << filename << '\n';
    return false;
  }

  const size_t point_count = static_cast<size_t>(scan.points_size());
  output << "# .PCD v0.7 - Point Cloud Data file format\n"
         << "VERSION 0.7\n"
         << "FIELDS x y z intensity timestamp\n"
         << "SIZE 4 4 4 4 8\n"
         << "TYPE F F F F F\n"
         << "COUNT 1 1 1 1 1\n"
         << "WIDTH " << point_count << "\n"
         << "HEIGHT 1\n"
         << "VIEWPOINT 0 0 0 1 0 0 0\n"
         << "POINTS " << point_count << "\n"
         << "DATA binary\n";

  std::vector<PcdPoint> points;
  points.reserve(point_count);
  for (const auto& point : scan.points()) {
    points.push_back(PcdPoint{point.x(), point.y(), point.z(),
                              static_cast<float>(point.intensity()),
                              point.timestamp()});
  }
  if (!points.empty()) {
    output.write(reinterpret_cast<const char*>(points.data()),
                 static_cast<std::streamsize>(points.size() * sizeof(PcdPoint)));
  }
  return output.good();
}

std::string FrameFilename(size_t index) {
  std::ostringstream stream;
  stream << std::setfill('0') << std::setw(6) << index << ".pcd";
  return stream.str();
}

int Run(const fs::path& project_dir, const fs::path& output_dir) {
  const fs::path pose_input = project_dir / "traj.dat";
  const fs::path lidar_input = project_dir / "lidar_undist.dat";
  if (!fs::is_regular_file(pose_input) || !fs::is_regular_file(lidar_input)) {
    std::cerr << "Expected traj.dat and lidar_undist.dat under: " << project_dir
              << '\n';
    return 1;
  }

  if (fs::exists(output_dir) && !fs::is_empty(output_dir)) {
    std::cerr << "Output directory must be absent or empty: " << output_dir
              << '\n';
    return 1;
  }
  const fs::path cloud_dir = output_dir / "clouds";
  fs::create_directories(cloud_dir);

  proto::PoseMsgList poses;
  if (!ReadPoseFile(pose_input.string(), poses) || poses.pose_msgs().empty()) {
    std::cerr << "Failed to load poses: " << pose_input << '\n';
    return 1;
  }

  std::ofstream pose_output(output_dir / "poses.txt", std::ios::trunc);
  if (!pose_output.is_open()) {
    std::cerr << "Failed to create poses.txt\n";
    return 1;
  }
  pose_output << std::setprecision(17);
  pose_output
      << "frame timestamp tx ty tz qx qy qz qw gx gy gz point_count cloud\n";

  SequentialLidarFileReader<proto::LidarMsg> reader;
  if (!reader.Open(lidar_input.string())) {
    return 1;
  }

  size_t frame_index = 0;
  std::shared_ptr<proto::LidarMsg> scan;
  while (reader.ReadNext(scan)) {
    if (frame_index >= static_cast<size_t>(poses.pose_msgs_size())) {
      std::cerr << "More LiDAR frames than IMU poses; first extra frame: "
                << frame_index << '\n';
      return 1;
    }
    const auto& pose = poses.pose_msgs(static_cast<int>(frame_index));
    const std::string cloud_name = FrameFilename(frame_index);
    if (!WriteBinaryPcd(cloud_dir / cloud_name, *scan)) {
      return 1;
    }
    pose_output << frame_index << ' ' << pose.timestamp() << ' ' << pose.tx()
                << ' ' << pose.ty() << ' ' << pose.tz() << ' ' << pose.rx()
                << ' ' << pose.ry() << ' ' << pose.rz() << ' ' << pose.rw()
                << ' ' << pose.gx() << ' ' << pose.gy() << ' ' << pose.gz()
                << ' ' << scan->points_size() << " clouds/" << cloud_name
                << '\n';
    ++frame_index;
    if (frame_index % 500 == 0) {
      std::cout << "Exported " << frame_index << "/" << poses.pose_msgs_size()
                << " frames (" << std::fixed << std::setprecision(1)
                << reader.getProgress() << "%)\n";
    }
  }
  reader.Close();

  if (frame_index != static_cast<size_t>(poses.pose_msgs_size())) {
    std::cerr << "Frame count mismatch: poses=" << poses.pose_msgs_size()
              << ", clouds=" << frame_index << '\n';
    return 1;
  }

  std::ofstream metadata(output_dir / "README.txt", std::ios::trunc);
  metadata << "Coordinate convention\n"
           << "=====================\n"
           << "poses.txt stores T_world_imu for every LiDAR frame.\n"
           << "Quaternion order is qx qy qz qw. Translation is in meters.\n"
           << "Each clouds/*.pcd is the motion-undistorted single-frame cloud "
              "in that frame's IMU/body coordinate system.\n"
           << "PCD fields are x y z (meters), intensity, timestamp (seconds).\n"
           << "Frame index provides the strict 1:1 pose/cloud correspondence.\n"
           << "Source poses: ../traj.dat\n"
           << "Source clouds: ../lidar_undist.dat\n"
           << "Frames: " << frame_index << "\n";

  std::cout << "Export complete: " << frame_index << " pose/cloud pairs -> "
            << output_dir << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: export_imu_frames <project_output_dir> <output_dir>\n";
    return 2;
  }
  try {
    return Run(fs::absolute(fs::path(argv[1])), fs::absolute(fs::path(argv[2])));
  } catch (const std::exception& error) {
    std::cerr << "Export failed: " << error.what() << '\n';
    return 1;
  }
}
