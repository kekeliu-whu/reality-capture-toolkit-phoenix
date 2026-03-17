
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

  // Load trajectory from traj.dat (low-frequency, one pose per LiDAR frame)
  proto::PoseMsgList traj_msg_list;
  if (!ReadPoseFile(traj_dat_filename, traj_msg_list)) {
    spdlog::error("Failed to read trajectory from: {}", traj_dat_filename);
    exit(1);
  }

  std::map<double, Sophus::SE3d> traj;
  for (const auto &pose_msg : traj_msg_list.pose_msgs()) {
    Eigen::Quaterniond q(pose_msg.rw(), pose_msg.rx(), pose_msg.ry(), pose_msg.rz());
    Eigen::Vector3d pos(pose_msg.tx(), pose_msg.ty(), pose_msg.tz());
    traj[pose_msg.timestamp()] = Sophus::SE3d(q.normalized().toRotationMatrix(), pos);
  }
  spdlog::info("Loaded {} poses from {}", traj.size(), traj_dat_filename);

  if (traj.empty()) {
    spdlog::error("No poses loaded from {}", traj_dat_filename);
    exit(1);
  }

  // Read lidar_undist.dat (proto::LidarMsg stream, body frame, undistorted)
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

    double scan_timestamp = 0.0;
    if (lidar_msg->points().size() > 0) {
      scan_timestamp = lidar_msg->points(0).timestamp();
    }

    // Find closest pose
    auto it = traj.lower_bound(scan_timestamp);
    if (it != traj.begin() && it != traj.end()) {
      auto prev = std::prev(it);
      if (std::abs(prev->first - scan_timestamp) < std::abs(it->first - scan_timestamp)) {
        it = prev;
      }
    } else if (it == traj.end()) {
      --it;
    }

    double time_diff = std::abs(it->first - scan_timestamp);
    if (time_diff > 1.0) {
      spdlog::warn("No matching pose for scan at t={:.6f} (closest diff={:.3f}s), skipping", scan_timestamp, time_diff);
      continue;
    }

    const Sophus::SE3d &pose_w = it->second;
    TimestampedPointCloud scan;
    scan.timestamp = scan_timestamp;
    scan.pose      = pose_w;

    // Points in lidar_undist.dat are in body frame, transform to world frame
    for (const auto &pt_proto : lidar_msg->points()) {
      Eigen::Vector3d p_body(pt_proto.x(), pt_proto.y(), pt_proto.z());
      Eigen::Vector3d p_world = pose_w * p_body;
      PointType pt;
      pt.getVector3fMap() = p_world.cast<float>();
      pt.intensity        = pt_proto.intensity();
      scan.cloud->push_back(pt);
    }

    scans.push_back(std::move(scan));
    scan_count++;
  }

  lidar_reader.Close();
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
    Sophus::SE3d pose_scan_to_cur_submap =
        current_submap.pose.inverse() * scan.pose;
    for (auto &p : scan.cloud->points) {
      auto np = p;
      np.getVector3fMap() =
          (pose_scan_to_cur_submap * np.getVector3fMap().cast<double>())
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
                 const std::string &output_filename) {
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

  pdal::LasWriter writer;
  writer.setOptions(options);
  writer.setInput(reader);
  writer.prepare(table);
  writer.execute(table);
}

namespace {
}  // namespace

// LocalENUTransformer implementation

LocalENUTransformer::LocalENUTransformer(double origin_lat_deg,
                                         double origin_lon_deg,
                                         double origin_alt_m)
    : origin_lat_(origin_lat_deg),
      origin_lon_(origin_lon_deg),
      origin_alt_(origin_alt_m) {
  ctx_ = proj_context_create();

  // Build target local ENU projection (tmerc centered at origin)
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(9)
      << "+proj=tmerc +lat_0=" << origin_lat_deg
      << " +lon_0=" << origin_lon_deg
      << " +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
  std::string target_proj_str = oss.str();

  // Source CRS: WGS84 geographic (EPSG:4326)
  transformer_ = proj_create_crs_to_crs(ctx_, "EPSG:4326", target_proj_str.c_str(), nullptr);
  if (!transformer_) {
    spdlog::error("LocalENUTransformer: failed to create PROJ transformer to {}", target_proj_str);
    exit(1);
  }
  spdlog::info("LocalENUTransformer initialized: origin lat={:.6f}deg, lon={:.6f}deg, alt={:.2f}m",
               origin_lat_deg, origin_lon_deg, origin_alt_m);
}

LocalENUTransformer::~LocalENUTransformer() {
  if (transformer_) {
    proj_destroy(transformer_);
    transformer_ = nullptr;
  }
  if (ctx_) {
    proj_context_destroy(ctx_);
    ctx_ = nullptr;
  }
}

Eigen::Vector3d LocalENUTransformer::Convert(double lat_deg, double lon_deg,
                                              double alt_m) const {
  // EPSG:4326 expects (latitude, longitude) order
  PJ_COORD coord_in = proj_coord(lat_deg, lon_deg, alt_m, 0);
  PJ_COORD coord_out = proj_trans(transformer_, PJ_FWD, coord_in);

  if (coord_out.xyz.x == HUGE_VAL || coord_out.xyz.y == HUGE_VAL) {
    spdlog::warn("LocalENUTransformer::Convert failed for lat={:.6f}, lon={:.6f}", lat_deg, lon_deg);
    return Eigen::Vector3d::Zero();
  }
  // tmerc output: x=East, y=North; altitude delta as Up
  return Eigen::Vector3d(coord_out.xyz.x, coord_out.xyz.y, alt_m - origin_alt_);
}

bool LoadGnssData(const std::string &gnss_filename,
                  std::vector<GpsData> &gnss_data) {
  proto::GpsMsgList gnss_msg_list;
  if (!ReadGnssFile(gnss_filename, gnss_msg_list)) {
    spdlog::error("Failed to read GNSS file: {}", gnss_filename);
    return false;
  }

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

  spdlog::info("Loaded {} GNSS measurements from {}", gnss_data.size(), gnss_filename);
  return !gnss_data.empty();
}

bool LoadGnssDataFromProject(const std::string &project_path,
                             std::vector<GpsData> &gnss_data) {
  return LoadGnssData(project_path + "/gnss.dat", gnss_data);
}
