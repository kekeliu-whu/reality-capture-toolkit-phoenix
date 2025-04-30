#include <crashpad/client/crash_report_database.h>
#include <crashpad/client/crashpad_client.h>
#include <crashpad/client/settings.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <omp.h>

DEFINE_string(point_cloud_filename, "D:/project_3d/data/sfm-share/output_dir/colorized.las_normals.pcd", "Point cloud filename");

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
// todo kk generate dump file with version
#ifdef __linux__
  base::FilePath::StringType db_path      = "dump";
  base::FilePath::StringType handler_path = "/buildspace/vcpkg/installed/x64-linux/tools/crashpad_handler";
#else
  base::FilePath::StringType db_path      = L"dump";
  base::FilePath::StringType handler_path = L"crashpad_handler.exe";
#endif
  std::string report_url = ".";
  StartCrashpad(db_path, handler_path, report_url);

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  DLOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  return 0;
}