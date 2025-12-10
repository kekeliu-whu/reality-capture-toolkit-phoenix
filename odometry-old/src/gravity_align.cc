#include <gflags/gflags.h>
#include <omp.h>
#include <pcl/filters/bilateral.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <proj.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <Eigen/Eigen>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/BufferReader.hpp>
#include <pdal/io/LasWriter.hpp>
#include <sstream>
#include <vector>

#include "migration/logging.h"
#include "migration/string.h"

DEFINE_string(input_las_file, "C:\\4.indoor-big-slow\\hall\\out\\map.las", "Input trajectory file path");
DEFINE_string(input_pose_file, "C:\\4.indoor-big-slow\\hall\\out\\traj.txt", "Input pose file path");
DEFINE_string(output_las_file, "C:\\4.indoor-big-slow\\hall\\out\\map_aligned.las",
              "Output aligned trajectory file path");
DEFINE_string(output_pose_file, "C:\\4.indoor-big-slow\\hall\\out\\traj_aligned.txt", "Output aligned pose file path");

#pragma pack(push, 1)
struct LidarPoint {
  double  timestamp;
  float   x;
  float   y;
  float   z;
  uint8_t intensity;
};
#pragma pack(pop)

std::vector<LidarPoint> ReadLidarPoints(const std::string& filename) {
  spdlog::info("Reading lidar point cloud from {}", filename);

  pdal::StageFactory factory;
  pdal::Stage*       reader = factory.createStage("readers.las");
  pdal::Options      opts;
  opts.add(pdal::Option("filename", PlatformToUTF8(filename)));
  reader->setOptions(opts);

  pdal::PointTable table;
  reader->prepare(table);
  pdal::PointViewSet viewSet = reader->execute(table);
  pdal::PointViewPtr view    = *viewSet.begin();

  std::vector<LidarPoint> points;
  points.reserve(view->size());
  double last_timestamp = 0.0;
  for (size_t i = 0; i < view->size(); ++i) {
    LidarPoint p;
    p.x         = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
    p.y         = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
    p.z         = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
    p.timestamp = view->getFieldAs<double>(pdal::Dimension::Id::GpsTime, i);
    p.intensity = view->getFieldAs<float>(pdal::Dimension::Id::Intensity, i);

    if (p.timestamp < last_timestamp) {
      spdlog::warn("Timestamps are not in ascending order at point {}: current {:.6f}, last {:.6f}", i, p.timestamp,
                   last_timestamp);
      continue;
    }
    last_timestamp = p.timestamp;

    points.push_back(p);
  }

  spdlog::info("Loaded {} lidar points from {}", points.size(), filename);

  return points;
}

struct PoseStamped {
  double             timestamp;
  Eigen::Vector3d    position;
  Eigen::Quaterniond orientation;
};

// Read trajectory file
bool readTrajectory(const std::string& filename, std::vector<PoseStamped>& poses, Eigen::Vector3d& grav) {
  spdlog::info("Reading trajectory from {}", filename);

  std::ifstream file(filename);
  if (!file.is_open()) {
    spdlog::error("Failed to open trajectory file: {}", filename);
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    PoseStamped        pose;
    double             qx, qy, qz, qw;

    if (!(iss >> pose.timestamp >> pose.position[0] >> pose.position[1] >> pose.position[2] >> qx >> qy >> qz >> qw >>
          grav[0] >> grav[1] >> grav[2])) {
      continue;
    }

    pose.orientation = Eigen::Quaterniond(qw, qx, qy, qz);
    poses.push_back(pose);
  }

  spdlog::info("Loaded trajectory points: {}", poses.size());
  return true;
}

bool SaveTrajectory(const std::string& filename, const std::vector<PoseStamped>& poses, const Eigen::Vector3d& grav) {
  spdlog::info("Saving aligned trajectory to {}", filename);

  std::ofstream file(filename);
  if (!file.is_open()) {
    spdlog::error("Failed to open output pose file: {}", filename);
    return false;
  }

  // Write header
  file << "#timestamp tx ty tz qx qy qz qw\n";

  // Gravity alignment quaternion
  Eigen::Quaterniond grav_q = Eigen::Quaterniond::FromTwoVectors(grav.normalized(), Eigen::Vector3d(0, 0, -1));
  // Write aligned trajectory (identity rotation for simplicity)
  file << std::fixed << std::setprecision(6);

  for (const auto& pose : poses) {
    // Rotate position
    Eigen::Vector3d aligned_position = grav_q * pose.position;

    // For simplicity, we set the orientation to identity after gravity alignment
    Eigen::Quaterniond aligned_orientation = grav_q * pose.orientation;

    file << pose.timestamp << " " << aligned_position[0] << " " << aligned_position[1] << " " << aligned_position[2]
         << " " << aligned_orientation.x() << " " << aligned_orientation.y() << " " << aligned_orientation.z() << " "
         << aligned_orientation.w() << "\n";
  }

  spdlog::info("Saved aligned trajectory to {}", filename);
  return true;
}

void SaveColoredPointCloud(const std::string& filename, const std::vector<LidarPoint>& lidarPoints,
                           const Eigen::Vector3d& grav) {
  spdlog::info("Saving point cloud to {}", filename);

  pdal::PointTable     table;
  pdal::PointLayoutPtr layout = table.layout();

  layout->registerDim(pdal::Dimension::Id::X);
  layout->registerDim(pdal::Dimension::Id::Y);
  layout->registerDim(pdal::Dimension::Id::Z);
  layout->registerDim(pdal::Dimension::Id::Intensity);
  layout->registerDim(pdal::Dimension::Id::GpsTime);

  pdal::PointViewPtr view(new pdal::PointView(table));

  Eigen::Quaterniond grav_q = Eigen::Quaterniond::FromTwoVectors(grav.normalized(), Eigen::Vector3d(0, 0, -1));

  for (const auto& pt : lidarPoints) {
    // Rotate point to align gravity with -Z
    Eigen::Vector3d point(pt.x, pt.y, pt.z);
    point = grav_q * point;

    pdal::PointId idx = view->size();
    view->setField(pdal::Dimension::Id::X, idx, point(0));
    view->setField(pdal::Dimension::Id::Y, idx, point(1));
    view->setField(pdal::Dimension::Id::Z, idx, point(2));
    view->setField(pdal::Dimension::Id::Intensity, idx, static_cast<float>(pt.intensity));
    view->setField(pdal::Dimension::Id::GpsTime, idx, pt.timestamp);
  }

  pdal::BufferReader reader;
  reader.addView(view);

  pdal::Options options;
  options.add("filename", PlatformToUTF8(filename));
  options.add("scale_x", 1e-4);
  options.add("scale_y", 1e-4);
  options.add("scale_z", 1e-4);
  options.add("offset_x", "auto");
  options.add("offset_y", "auto");
  options.add("offset_z", "auto");

  pdal::LasWriter writer;
  writer.setOptions(options);
  writer.setInput(reader);
  writer.prepare(table);
  writer.execute(table);

  spdlog::info("Saved point cloud to {}", filename);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  InitSpdLog();

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);
  spdlog::info("Using {} / {} cores.", cores_used, cores);

  Eigen::Vector3d          grav;
  std::vector<PoseStamped> poses;
  if (!readTrajectory(FLAGS_input_pose_file, poses, grav)) {
    spdlog::error("Failed to read trajectory.");
    return -1;
  }

  std::vector<LidarPoint> lidarPoints = ReadLidarPoints(FLAGS_input_las_file);

  SaveColoredPointCloud(FLAGS_output_las_file, lidarPoints, grav);

  SaveTrajectory(FLAGS_output_pose_file, poses, grav);

  spdlog::info("Gravity alignment completed.");
  return 0;
}
