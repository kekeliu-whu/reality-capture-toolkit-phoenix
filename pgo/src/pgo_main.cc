
#include <glog/logging.h>

#include "map/pgo_runner.h"
#include "migration/proto_io.h"

DEFINE_string(project_input_path, "D:/BaiduNetdiskDownload/2024-11-27_17-36-04-SHAREUAV-S20", "Input project path");
DEFINE_string(project_output_path, "D:/BaiduNetdiskDownload/2024-11-27_17-36-04-SHAREUAV-S20", "Output project path");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = 1;

  proto::PgoConfig pgo_config;
  auto ok = ReadPgoConfigFile(
      R"(D:\matrix641\migration\config\pgo\pgo.json)",
      pgo_config);
  CHECK(ok);
  DLOG(INFO) << "Read configuration success:\n"
            << pgo_config.DebugString();

  PgoRunner runner(pgo_config);
  runner.Run(FLAGS_project_input_path, FLAGS_project_output_path);

  return 0;
}
