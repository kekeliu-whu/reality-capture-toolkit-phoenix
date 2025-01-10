#include <liblas/liblas.hpp>

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

  DownsampleSubmaps(submaps, 0.15);
  MallocTrim();
  PrintMemoryUsage();

  Optimize(submaps, this->config_);

  // reload and save optimized submaps
  std::vector<TimestampedPointCloud> submaps_reload;
  LoadSubmapList(input_path, submaps_reload, config_.submap_duration_secs());
  // copy optimized poses to the reloaded submaps
  for (size_t i = 0; i < submaps_reload.size(); ++i) {
    submaps_reload[i].pose = submaps[i].pose;
  }
  SaveLasFile(submaps_reload, output_path + "/map_optimized.las");
}
