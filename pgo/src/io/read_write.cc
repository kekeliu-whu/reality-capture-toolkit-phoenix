
#include <glog/logging.h>
#include <liblas/liblas.hpp>

#include "common/msg_conversions.h"
#include "map/utils.h"
#include "migration/sensor_io.h"
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
      lidar_filename, [&raw_scans](const ConstPtr<UndistoredLidarMsg> &msg) {
        raw_scans.push_back(FromProto(*msg));
      });
  CHECK(ok) << "Failed to load undistorted lidar scans from " << lidar_filename;
  DLOG(INFO) << "scans number loaded: " << raw_scans.size();

  PoseMsgList pose_msg_list;
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

void SaveLasFile(const std::vector<TimestampedPointCloud> &submaps,
                 const std::string &output_filename) {
  int point_num = 0;
  for (auto submap : submaps) {
    point_num += submap.cloud->size();
  }

  std::ofstream ofs(output_filename, std::ios::out | std::ios::binary);
  CHECK(ofs) << "Failed to open output file: " << output_filename;

  DLOG(INFO) << "Saving " << point_num << " points to " << output_filename;

  liblas::Header header;
  header.SetDataFormatId(liblas::ePointFormat1);
  header.SetVersionMajor(1);
  header.SetVersionMinor(2);
  header.SetPointRecordsCount(point_num);
  header.SetScale(1e-4, 1e-4, 1e-4);
  header.SetCreationYear(2024);
  header.SetCreationDOY(1);

  liblas::Writer writer(ofs, header);

  bool is_first_point = true;
  Eigen::AlignedBox<double, 3> bounding_box;
  for (auto submap : submaps) {
    for (const auto &p : submap.cloud->points) {
      if (is_first_point) {
        is_first_point = false;
      }

      auto point_transform = submap.pose * p.getVector3fMap().cast<double>();

      bounding_box.extend(point_transform);

      liblas::Point point(&header);
      point.SetX(point_transform.x());
      point.SetY(point_transform.y());
      point.SetZ(point_transform.z());
      point.SetIntensity(p.intensity);
      point.SetTime(submap.timestamp);
      writer.WritePoint(point);
    }
  }

  header.SetMax(bounding_box.max().x(), bounding_box.max().y(),
                bounding_box.max().z());
  header.SetMin(bounding_box.min().x(), bounding_box.min().y(),
                bounding_box.min().z());
  writer.SetHeader(header);
  writer.WriteHeader();

  DLOG(INFO) << "Save " << point_num << " points to " << output_filename
            << " successfully.";
}
