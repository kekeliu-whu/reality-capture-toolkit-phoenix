
#include <glog/logging.h>

#include "map/pgo_runner.h"
#include "migration/proto_io.h"

DEFINE_string(project_input_path, "D:/Users/rick/Desktop/slam_evaluation/l2pro/process", "Input project path");
DEFINE_string(project_output_path, "D:/Users/rick/Desktop/slam_evaluation/l2pro/process", "Output project path");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = 1;

  proto::PgoConfig pgo_config;
  auto ok = ReadPgoConfigFile(
      std::string(PROJECT_DIR) + "/../migration/config/pgo/pgo.json",
      pgo_config);
  DCHECK(ok);
  DLOG(INFO) << "Read configuration success:" << std::endl
             << pgo_config.DebugString();

  PgoRunner runner(pgo_config);
  runner.Run(FLAGS_project_input_path, FLAGS_project_output_path);

  return 0;
}
