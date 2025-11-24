#include <crashpad/client/crash_report_database.h>
#include <crashpad/client/crashpad_client.h>
#include <crashpad/client/settings.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <omp.h>

#include "common/msg_conversions.h"
#include "core/slam_core_lib.h"
#include "migration/proto_io.h"

DEFINE_string(project_dir, "/root/output_dir", "Project directory");

bool StartCrashpad(const base::FilePath::StringType& db_path, const base::FilePath::StringType& handler_path, const std::string& url) {
  using namespace crashpad;
  base::FilePath database(db_path);
  base::FilePath handler(handler_path);
  base::FilePath metrics_dir(db_path);

  std::map<std::string, std::string> annotations;
  std::vector<std::string> arguments;

  std::unique_ptr<CrashReportDatabase> database_ptr = CrashReportDatabase::Initialize(database);

  if (database_ptr && database_ptr->GetSettings()) database_ptr->GetSettings()->SetUploadsEnabled(true);

  CrashpadClient client;
  return client.StartHandler(handler, database, metrics_dir, url, annotations, arguments,
                             true,  // restartable
                             false  // asynchronous start
  );
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /////////////////////////////////// setup glog ///////////////////////////////////
  google::InitGoogleLogging(argv[0]);
  // EncryptedLogSink *sink = new EncryptedLogSink();
  // google::AddLogSink(sink);
  FLAGS_logtostderr = 1;

  std::shared_ptr<void> done{nullptr, [](void*) {
                               DLOG(INFO) << "Shutdown glog.";
                               google::ShutdownGoogleLogging();
                             }};

/////////////////////////////////// setup minidump ///////////////////////////////////
#ifdef __linux__
  base::FilePath::StringType db_path      = "dump";
  base::FilePath::StringType handler_path = "/buildspace/vcpkg/installed/x64-linux/tools/crashpad_handler";
#else
  base::FilePath::StringType db_path      = L"dump";
  base::FilePath::StringType handler_path = L"crashpad_handler.exe";
#endif
  std::string report_url = ".";
  StartCrashpad(db_path, handler_path, report_url);

  /////////////////////////////////// setup omp ///////////////////////////////////
  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  DLOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  /////////////////////////////////// setup odometry core ///////////////////////////////////
  proto::SensorCalib calib;
  proto::ImuMsgList imu_msg_list;
  proto::EncoderMsgList encoder_msg_list;
  ReadSensorCalibFile(FLAGS_project_dir + "/calibration.dat", calib);
  ReadImuFile(FLAGS_project_dir + "/imu.dat", imu_msg_list);
  ReadEncoderFile(FLAGS_project_dir + "/encoder.dat", encoder_msg_list);

  DLOG(INFO) << "Calibration: " << calib.DebugString();
  SlamCore core(FromProto(calib));
  for (auto& msg : imu_msg_list.imu_msgs()) {
    core.AddSensorData(FromProto(msg));
  }
  for (auto& msg : encoder_msg_list.encoder_msgs()) {
    core.AddSensorData(FromProto(msg));
  }
  ReadLidarFile(FLAGS_project_dir + "/lidar.dat", [&core](const ConstPtr<proto::LidarMsg>& msg) {
    core.AddSensorData(FromProto(msg));
    OdometryResult::Ptr result;
    core.TryEstimateState(result);
  });

  return 0;
}