
#include <glog/logging.h>
#include <Eigen/Dense>
#include <fstream>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/io/BufferReader.hpp>
#include <pdal/io/LasReader.hpp>
#include <pdal/io/LasWriter.hpp>
#include <vector>

#include "common/msg_conversions.h"
#include "map/utils.h"
#include "migration/proto_io.h"
#include "read_write.h"

namespace {

void LoadRawScans(
    const std::string &project_path,
    std::vector<TimestampedPointCloud> &scans) {
  std::string lidar_filename = project_path + "/lidar.dat";
  std::string poses_filename = project_path + "/poses.dat";

  // read undistorted lidar scans
  std::vector<PointCloud::Ptr> raw_scans;
  std::vector<TimestampedPose> raw_timestamped_poses;
  DLOG(INFO) << "Loading undistorted lidar scans from " << lidar_filename;
  auto ok = ReadUndistortedLidarFile(
      lidar_filename, [&raw_scans](const ConstPtr<proto::UndistoredLidarMsg> &msg) {
        raw_scans.push_back(FromProto(*msg));
      });
  CHECK(ok) << "Failed to load undistorted lidar scans from " << lidar_filename;
  DLOG(INFO) << "scans number loaded: " << raw_scans.size();

  proto::PoseMsgList pose_msg_list;
  DLOG(INFO) << "Loading poses from " << poses_filename;
  ok = ReadPoseFile(project_path + "/poses.dat", pose_msg_list);
  CHECK(ok) << "Failed to load poses from " << poses_filename;
  DLOG(INFO) << "poses number loaded: " << pose_msg_list.pose_msgs_size();

  CHECK_EQ(pose_msg_list.pose_msgs_size(), raw_scans.size());

  for (int i = 0; i < pose_msg_list.pose_msgs_size(); ++i) {
    scans.emplace_back();
    scans.back().timestamp = pose_msg_list.pose_msgs(i).timestamp();
    scans.back().pose      = FromProto(pose_msg_list.pose_msgs(i));
    scans.back().cloud     = raw_scans[i];
  }

  DLOG(INFO) << "All project data loaded done.";
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

  DLOG(INFO) << "Build " << submaps.size() << " submaps from " << scans.size()
             << " scans done.";
}

}  // namespace

void LoadSubmapList(const std::string &project_path,
                    std::vector<TimestampedPointCloud> &submaps,
                    double submap_duration_secs) {
  std::vector<TimestampedPointCloud> scans;
  LoadRawScans(project_path, scans);
  if (submap_duration_secs <= 0) {
    submaps = std::move(scans);
    return;
  }
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
