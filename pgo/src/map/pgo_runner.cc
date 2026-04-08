
#include <spdlog/spdlog.h>
#include <fstream>
#include <iomanip>

#include "io/read_write.h"
#include "map/optimizer.h"
#include "migration/utils.h"
#include "pgo_runner.h"
#include "utils.h"

namespace {

void ReleaseMemoryAndDownsample(std::vector<TimestampedPointCloud>& submaps,
                                double downsample_resolution) {
  MallocTrim();
  PrintMemoryUsage();

  DownsampleSubmaps(submaps, downsample_resolution);

  MallocTrim();
  PrintMemoryUsage();
}

bool BuildFullResolutionScansForSave(
    const std::string& input_path,
    const std::vector<TimestampedPose>& optimized_scan_poses,
    std::vector<TimestampedPointCloud>& full_resolution_scans) {
  std::vector<TimestampedPose> full_resolution_scan_poses;
  LoadSubmapList(input_path,
                 full_resolution_scan_poses,
                 full_resolution_scans,
                 0.0);
  if (full_resolution_scans.size() != optimized_scan_poses.size()) {
    spdlog::error(
        "Scan count mismatch when saving optimized map: {} reloaded vs {} optimized",
        full_resolution_scans.size(),
        optimized_scan_poses.size());
    return false;
  }

  for (size_t i = 0; i < full_resolution_scans.size(); ++i) {
    full_resolution_scans[i].pose = optimized_scan_poses[i].pose;
  }
  return true;
}

void SaveOptimizedTrajectory(const std::string& output_path,
                             const std::vector<TimestampedPose>& scan_poses) {
  const std::string traj_filename = output_path + "/trajectory_opt.txt";
  std::ofstream traj_file(traj_filename);
  if (!traj_file.is_open()) {
    spdlog::error("Failed to open trajectory output file: {}", traj_filename);
    return;
  }

  traj_file << "# x y z roll pitch yaw qx qy qz qw timestamp\n";
  traj_file << std::fixed << std::setprecision(12);
  for (const auto& scan_pose : scan_poses) {
    const auto& pose     = *scan_pose.pose;
    Eigen::Vector3d t    = pose.translation();
    Eigen::Quaterniond q = pose.unit_quaternion();
    traj_file << t.x() << " " << t.y() << " " << t.z() << " " << 0.0 << " " << 0.0 << " " << 0.0 << " "
              << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " << scan_pose.timestamp << "\n";
  }

  spdlog::info("Saved optimized trajectory to {}", traj_filename);
}

}  // namespace

void PgoRunner::Run(const std::string& input_path,
                    const std::string& output_path) {
  std::vector<TimestampedPose> timestamped_scan_poses;
  std::vector<TimestampedPointCloud> submaps;
  LoadSubmapList(input_path,
                 timestamped_scan_poses,
                 submaps,
                 config_.submap_duration_secs());

  // spdlog::info("Saving raw map to {}", output_path + "/map_raw.las");
  // SaveLasFile(submaps, output_path + "/map_raw.las");

  ReleaseMemoryAndDownsample(submaps, config_.submap_downsample_resolution());

  // Try to load GNSS data for fusion
  std::vector<GpsData> gnss_data;
  bool has_gnss_data = LoadGnssDataFromProject(input_path + "/gnss.dat", gnss_data);

  // Use RTK constraints if both GNSS data is available and user enabled it
  bool use_rtk = has_gnss_data && use_rtk_constraint_;

  std::string proj4_string;
  OptimizeWithGnss(timestamped_scan_poses,
                   submaps,
                   gnss_data,
                   config_,
                   use_rtk,
                   proj4_string);

  std::vector<TimestampedPointCloud> full_resolution_scans;
  if (!BuildFullResolutionScansForSave(input_path,
                                       timestamped_scan_poses,
                                       full_resolution_scans)) {
    return;
  }

  spdlog::info("Saving optimized map to {}", output_path + "/map_opt.las");
  SaveLasFile(full_resolution_scans, output_path + "/map_opt.las", proj4_string);
  SaveOptimizedTrajectory(output_path, timestamped_scan_poses);
}
