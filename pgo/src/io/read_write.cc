
#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <map>
#include <proj.h>
#include <sstream>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/io/BufferReader.hpp>
#include <pdal/io/LasReader.hpp>
#include <pdal/io/LasWriter.hpp>
#include <spdlog/spdlog.h>
#include <vector>

#include "map/utils.h"
#include "migration/proto_io.h"
#include "read_write.h"

namespace {

void LoadRawScans(
    const std::string &project_path,
    std::vector<TimestampedPointCloud> &scans) {
  const std::string traj_dat_filename = project_path + "/traj.dat";
  const std::string lidar_undist_filename = project_path + "/lidar_undist.dat";

  // Load trajectory from traj.dat (low-frequency, one pose per LiDAR frame, in order)
  proto::PoseMsgList traj_msg_list;
  if (!ReadPoseFile(traj_dat_filename, traj_msg_list)) {
    spdlog::error("Failed to read trajectory from: {}", traj_dat_filename);
    exit(1);
  }

  std::vector<Sophus::SE3d> poses;
  for (const auto &pose_msg : traj_msg_list.pose_msgs()) {
    Eigen::Quaterniond q(pose_msg.rw(), pose_msg.rx(), pose_msg.ry(), pose_msg.rz());
    Eigen::Vector3d pos(pose_msg.tx(), pose_msg.ty(), pose_msg.tz());
    poses.push_back(Sophus::SE3d(q.normalized().toRotationMatrix(), pos));
  }
  spdlog::info("Loaded {} poses from {}", poses.size(), traj_dat_filename);

  if (poses.empty()) {
    spdlog::error("No poses loaded from {}", traj_dat_filename);
    exit(1);
  }

  // Read lidar_undist.dat (proto::LidarMsg stream, body frame, undistorted)
  // Scans and poses have 1:1 correspondence
  SequentialLidarFileReader<proto::LidarMsg> lidar_reader;
  if (!lidar_reader.Open(lidar_undist_filename)) {
    spdlog::error("Failed to open: {}", lidar_undist_filename);
    exit(1);
  }

  std::shared_ptr<proto::LidarMsg> lidar_msg;
  size_t scan_count = 0;
  while (lidar_reader.ReadNext(lidar_msg)) {
    if (!lidar_msg || lidar_msg->points().empty()) {
      continue;
    }

    // Check if scan index is within bounds
    if (scan_count >= poses.size()) {
      spdlog::warn("More scans than poses, stopping at scan {}", scan_count);
      break;
    }

    double scan_timestamp = 0.0;
    if (lidar_msg->points().size() > 0) {
      scan_timestamp = lidar_msg->points(0).timestamp();
    }

    // Use corresponding pose directly (1:1 correspondence)
    const Sophus::SE3d &pose_w = poses[scan_count];
    TimestampedPointCloud scan;
    scan.timestamp = scan_timestamp;
    scan.pose      = pose_w;

    // Points in lidar_undist.dat are in body frame, keep them in body frame (no coordinate transform)
    for (const auto &pt_proto : lidar_msg->points()) {
      Eigen::Vector3d p_body(pt_proto.x(), pt_proto.y(), pt_proto.z());
      PointType pt;
      pt.getVector3fMap() = p_body.cast<float>();
      pt.intensity        = pt_proto.intensity();
      scan.cloud->push_back(pt);
    }

    scans.push_back(std::move(scan));
    scan_count++;
  }

  lidar_reader.Close();
  
  if (scan_count != poses.size()) {
    spdlog::warn("Loaded {} scans but have {} poses", scan_count, poses.size());
    exit(1);
  }
  spdlog::info("Loaded {} scans from {}", scan_count, lidar_undist_filename);
}

void BuildSubMapFromRawScans(const std::vector<TimestampedPointCloud> &scans,
                             double submap_duration_secs,
                             std::vector<TimestampedPointCloud> &submaps) {
  submaps.clear();
  if (scans.empty()) {
    return;
  }

  TimestampedPointCloud current_submap;
  current_submap.pose      = scans.front().pose;
  current_submap.timestamp = scans.front().timestamp;

  for (const auto &scan : scans) {
    if (scan.timestamp - current_submap.timestamp > submap_duration_secs) {
      submaps.push_back(current_submap);
      current_submap           = TimestampedPointCloud();
      current_submap.pose      = scan.pose;
      current_submap.timestamp = scan.timestamp;
    }

    Sophus::SE3d pose_scan_body_to_submap_body = 
        current_submap.pose.inverse() * scan.pose;
    
    for (const auto &p : scan.cloud->points) {
      auto np = p;
      np.getVector3fMap() =
          (pose_scan_body_to_submap_body * np.getVector3fMap().cast<double>())
              .cast<float>();
      current_submap.cloud->push_back(np);
    }
  }

  spdlog::info("Build {} submaps from {} scans done.", submaps.size(), scans.size());
}

}  // namespace

void LoadSubmapList(const std::string &project_path,
                    std::vector<TimestampedPointCloud> &submaps,
                    double submap_duration_secs) {
  CHECK_GT(submap_duration_secs, 0);
  std::vector<TimestampedPointCloud> scans;
  LoadRawScans(project_path, scans);
  BuildSubMapFromRawScans(scans, submap_duration_secs, submaps);
}

// todo kk to be tested
void SaveLasFile(const std::vector<TimestampedPointCloud> &submaps,
                 const std::string &output_filename,
                 const std::string &proj4_string) {
  pdal::PointTable table;
  pdal::PointLayoutPtr layout = table.layout();

  layout->registerDim(pdal::Dimension::Id::X);
  layout->registerDim(pdal::Dimension::Id::Y);
  layout->registerDim(pdal::Dimension::Id::Z);
  layout->registerDim(pdal::Dimension::Id::Intensity);
  layout->registerDim(pdal::Dimension::Id::GpsTime);

  pdal::PointViewPtr view(new pdal::PointView(table));

  for (const auto &submap : submaps) {
    for (const auto &p : submap.cloud->points) {
      auto point_transform = submap.pose * p.getVector3fMap().cast<double>();

      pdal::PointId id = view->size();
      view->setField(pdal::Dimension::Id::X, id, point_transform.x());
      view->setField(pdal::Dimension::Id::Y, id, point_transform.y());
      view->setField(pdal::Dimension::Id::Z, id, point_transform.z());
      view->setField(pdal::Dimension::Id::Intensity, id, p.intensity);
      view->setField(pdal::Dimension::Id::GpsTime, id, submap.timestamp);
    }
  }

  pdal::BufferReader reader;
  reader.addView(view);

  pdal::Options options;
  options.add("filename", output_filename);
  options.add("scale_x", 1e-4);
  options.add("scale_y", 1e-4);
  options.add("scale_z", 1e-4);
  options.add("offset_x", "auto");
  options.add("offset_y", "auto");
  options.add("offset_z", "auto");
  options.add("minor_version", 2);
  options.add("dataformat_id", 1);  // PointFormat = 1
  
  // Add coordinate system if provided
  if (!proj4_string.empty()) {
    options.add("a_srs", proj4_string);
  }

  pdal::LasWriter writer;
  writer.setOptions(options);
  writer.setInput(reader);
  writer.prepare(table);
  writer.execute(table);
}

namespace {
}  // namespace

bool LoadGnssData(const std::string &gnss_filename,
                  std::vector<GpsData> &gnss_data) {
  proto::GpsMsgList gnss_msg_list;
  if (!ReadGnssFile(gnss_filename, gnss_msg_list)) {
    spdlog::error("Failed to read GNSS file: {}", gnss_filename);
    return false;
  }

  if (gnss_msg_list.gps_msgs().empty()) {
    spdlog::warn("No GNSS measurements in file: {}", gnss_filename);
    return false;
  }

  // Create transformer using first GNSS point as origin
  const auto &first_msg = gnss_msg_list.gps_msgs(0);
  LocalENUTransformer transformer(first_msg.latitude(), first_msg.longitude());

  gnss_data.clear();
  
  for (const auto &msg : gnss_msg_list.gps_msgs()) {
    GpsData gps;
    gps.timestamp = msg.timestamp();
    gps.latitude  = msg.latitude();
    gps.longitude = msg.longitude();
    gps.altitude  = msg.altitude();
    gps.lat_std   = msg.lat_std();
    gps.lon_std   = msg.lon_std();
    gps.alt_std   = msg.alt_std();
    
    gnss_data.push_back(gps);
  }

  spdlog::info("Loaded {} GNSS measurements from {}", 
               gnss_data.size(), gnss_filename);
  return !gnss_data.empty();
}

bool LoadGnssDataFromProject(const std::string &project_path,
                             std::vector<GpsData> &gnss_data) {
  return LoadGnssData(project_path + "/gnss.dat", gnss_data);
}
