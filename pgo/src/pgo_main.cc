
#include "map/pgo_runner.h"
#include "migration/proto_io.h"

DEFINE_string(project_input_path, "D:/output", "Input project path");
DEFINE_string(project_output_path, "D:/output", "Output project path");
DEFINE_string(config_filename, std::string(PROJECT_DIR) + "/../migration/config/pgo/pgo.json", "PGO configuration filename");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  proto::PgoConfig pgo_config;
  auto ok = ReadPgoConfigFile(FLAGS_config_filename, pgo_config);
  if (!ok) {
    spdlog::error("Check failed");
    exit(1);
  }
  spdlog::info("Read configuration success:\n{}", pgo_config.DebugString());

  PgoRunner runner(pgo_config);
  runner.SetUseRtkConstraint(true);  // Enable RTK constraints for better accuracy if GNSS data is available
  runner.Run(FLAGS_project_input_path, FLAGS_project_output_path);

  spdlog::info("done.");

  return 0;
}
