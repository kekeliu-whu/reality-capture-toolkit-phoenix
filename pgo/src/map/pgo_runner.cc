
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

  std::string proj4_string;
  OptimizeWithGnss(submaps, gnss_data, config_, use_rtk, proj4_string);

  // reload and save optimized submaps
  std::vector<TimestampedPointCloud> submaps_reload;
  LoadSubmapList(input_path, submaps_reload, config_.submap_duration_secs());
  // copy optimized poses to the reloaded submaps
  for (size_t i = 0; i < submaps_reload.size(); ++i) {
    submaps_reload[i].pose = submaps[i].pose;
  }
  spdlog::info("Saving optimized map to {}", output_path + "/map_opt.las");
  SaveLasFile(submaps_reload, output_path + "/map_opt.las", proj4_string);
}
