
#include <spdlog/spdlog.h>

#include "io/read_write.h"
#include "map/optimizer.h"
#include "migration/utils.h"
#include "pgo_runner.h"
#include "utils.h"

void PgoRunner::Run(const std::string &input_path,
                    const std::string &output_path) {
  std::vector<TimestampedPointCloud> submaps;
  LoadSubmapList(input_path, submaps, config_.submap_duration_secs());
  SaveLasFile(submaps, output_path + "/map_raw.las");

  // release raw scans to save memory
  MallocTrim();
  PrintMemoryUsage();

  DownsampleSubmaps(submaps, config_.submap_downsample_resolution());
  MallocTrim();
  PrintMemoryUsage();

  // Try to load GNSS data for fusion
  std::vector<GpsData> gnss_data;
  bool has_gnss_data = LoadGnssDataFromProject(input_path, gnss_data);
  
  // Use RTK constraints if both GNSS data is available and user enabled it
  bool use_rtk = has_gnss_data && use_rtk_constraint_;
  
  if (use_rtk) {
    spdlog::info("GNSS data found ({}), using RTK fusion optimization", gnss_data.size());
  } else {
    spdlog::info("No GNSS data or RTK disabled, using standard PGO optimization");
  }
  
  OptimizeWithGnss(submaps, gnss_data, config_, use_rtk);

  // reload and save optimized submaps
  std::vector<TimestampedPointCloud> submaps_reload;
  LoadSubmapList(input_path, submaps_reload, config_.submap_duration_secs());
  // copy optimized poses to the reloaded submaps
  for (size_t i = 0; i < submaps_reload.size(); ++i) {
    submaps_reload[i].pose = submaps[i].pose;
  }
  SaveLasFile(submaps_reload, output_path + "/map_optimized.las");
}

void PgoRunner::RunWithGnss(const std::string &input_path,
                            const std::string &output_path,
                            bool use_gnss) {
  std::vector<TimestampedPointCloud> submaps;
  LoadSubmapList(input_path, submaps, config_.submap_duration_secs());
  SaveLasFile(submaps, output_path + "/map_raw.las");

  // release raw scans to save memory
  MallocTrim();
  PrintMemoryUsage();

  DownsampleSubmaps(submaps, config_.submap_downsample_resolution());
  MallocTrim();
  PrintMemoryUsage();

  // Load GNSS data
  std::vector<GpsData> gnss_data;
  if (use_gnss) {
    if (!LoadGnssDataFromProject(input_path, gnss_data)) {
      spdlog::error("GNSS data requested but not found, aborting RunWithGnss");
      return;
    }
    spdlog::info("Loaded {} GNSS measurements", gnss_data.size());
  }

  // Optimize with GNSS fusion - use RTK constraints if enabled
  OptimizeWithGnss(submaps, gnss_data, config_, use_rtk_constraint_);

  // reload and save optimized submaps
  std::vector<TimestampedPointCloud> submaps_reload;
  LoadSubmapList(input_path, submaps_reload, config_.submap_duration_secs());
  // copy optimized poses to the reloaded submaps
  for (size_t i = 0; i < submaps_reload.size(); ++i) {
    submaps_reload[i].pose = submaps[i].pose;
  }
  SaveLasFile(submaps_reload, output_path + "/map_optimized.las");
  
  spdlog::info("GNSS-fused PGO completed successfully");
}
