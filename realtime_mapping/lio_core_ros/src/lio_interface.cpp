#include <omp.h>
#include <boost/filesystem.hpp>
#include <csignal>
#include <memory>
#include <string>

#define LIO_DLL_EXPORTS
#include "lio_core_ros/exception.h"
#include "lio_core_ros/lio_interface.h"
#include "lio_core_ros/parameters_reader.h"
#include "lio_core_ros/publisher.h"
#include "lio_core_ros/subscriber.h"
#include "log/lsLogger.h"

volatile sig_atomic_t stop_signal_recv = 0;  // flag to cancel LIO
LIOHANDLE handler = nullptr;                 // handler for API control flow
bool started_flag = false;                   // flag to prevent starting LIO multiple times
double progress = 0.0;                       // progress of LIO

namespace
{

namespace fs = boost::filesystem;

#ifdef _WIN32
#include <Windows.h>
std::string getDynamicLibraryPath()
{
  HMODULE hModule = GetModuleHandle(NULL);
  std::vector<char> path(40960, 0);
  GetModuleFileName(hModule, path.data(), MAX_PATH);
  return fs::path(path.data()).parent_path().string();
}
#elif __linux__
#include <dlfcn.h>
#include <unistd.h>
std::string getDynamicLibraryPath()
{
  Dl_info dlInfo;
  dladdr((void*)getDynamicLibraryPath, &dlInfo);
  char* path = realpath(dlInfo.dli_fname, NULL);
  std::string fullPath(path);
  free(path);
  size_t pos = fullPath.find_last_of("/");
  return fullPath.substr(0, pos);
}
#else
#error Unsupported platform
#endif

void RemoveAllYamlFilesInDir(const std::string& path)
{
  try
  {
    fs::path dirPath(path);
    if (fs::exists(dirPath) && fs::is_directory(dirPath))
    {
      fs::directory_iterator end_iter;
      for (fs::directory_iterator dir_itr(dirPath); dir_itr != end_iter; ++dir_itr)
      {
        if (fs::is_regular_file(dir_itr->status()))
        {
          std::string extension = dir_itr->path().extension().string();
          if (extension == ".yaml")
          {
            fs::remove(dir_itr->path());
            lslog(LSLOG_INFO) << "Remove: " << dir_itr->path() << std::endl;
          }
        }
      }
    }
  }
  catch (const fs::filesystem_error& e)
  {
    std::cerr << e.what() << std::endl;
  }
}

/**
 * @brief
 *
 * @param algorithm_config_filename
 * @param bag_filename
 * @param use_external_calib
 * @param external_calib_filepath
 * @param output_dir
 * @param output_dir_temp
 * @param calib_model =0 L2-like calibration =1 K1 calibration
 */
LioCoreRosErrorCode startImpl(
    const std::string& algorithm_config_filename,
    const std::string& bag_filename,
    bool use_external_calib,
    const std::string& external_calib_filepath,
    const std::string& output_dir,
    const std::string& output_dir_temp,
    CalibModel calib_model)
{
  int core_num = std::max((int)std::thread::hardware_concurrency() - 2, 1);
  omp_set_num_threads(core_num);
  lslog(LSLOG_INFO) << "hardware_concurrency: " << std::thread::hardware_concurrency()
                    << " omp_set_num_threads: " << core_num << " omp_get_num_threads: " << omp_get_num_threads();

  LioCoreRosErrorCode code;
  // init logging

  ///////////////////////////// 2. Read Parameters /////////////////////////////
  try
  {
    std::string real_calib_filepath = external_calib_filepath;
    if (!use_external_calib)
    {
      real_calib_filepath = output_dir_temp;
      lslog(LSLOG_INFO) << "Force to use HBC calibration data, real_calib_filepath will be set to output_dir: "
                        << real_calib_filepath;

      RemoveAllYamlFilesInDir(real_calib_filepath);

      int valid_yaml_filecount = ExtractYamlFileFromHbc(bag_filename, real_calib_filepath);
      if (valid_yaml_filecount == 0)
      {
        lslog(LSLOG_ERROR) << "Failed to extract yaml file from HBC, exit.";
        return LioCoreRosErrorCode::EC_LIO_CREATE_EXTRACTED_CALIB_FILE;
      }
    }

    // DCHECK(calib_model == CalibModel::K1 || calib_model == CalibModel::L2_LIKE)
    // << "Invalid device model, only support 0(L1/L2) or 1(K1)";
    FullParameters params;
    if ((code = ReadParameters(algorithm_config_filename, real_calib_filepath, params, calib_model)) !=
        LioCoreRosErrorCode::EC_LIO_OK)
    {
      lslog(LSLOG_ERROR) << "ReadParameters() failed, exit.";
      return code;
    }
    params.map_param.config_path = algorithm_config_filename;

    //////////////////// 3. Start LIO ////////////////////
    std::shared_ptr<lixel::LioCore> lio_core(new lixel::LioCore(params));
    Publisher publisher{params, output_dir, output_dir_temp};
    Subscriber subscriber{params, calib_model};

    publisher.registerCallbacks(lio_core);

    // DCHECK(!bag_filename.empty());
    lslog(LSLOG_INFO) << "Using offline mode ...";
    rosbag::Bag bag;
    bag.open(bag_filename);
    lslog(LSLOG_INFO) << "Reading bag file " << bag_filename << " from " << params.ros_param.offline_mode_start_frame_id
                      << " to " << params.ros_param.offline_mode_end_frame_id << " by speed_ratio "
                      << params.ros_param.offline_mode_speed_ratio;
    subscriber.setLioCore(lio_core);  // set lio_core because it will be called in callbacks
    lio_core->start();
    if ((code = subscriber.processBag(bag, stop_signal_recv, &progress)) != LioCoreRosErrorCode::EC_LIO_OK)
    {
      lslog(LSLOG_ERROR) << "ProcessBag() failed, exit with code " << static_cast<int>(code);
      return code;
    }

    bag.close();

    ///////////////////////////// 4. Exit /////////////////////////////
    {
      lslog(LSLOG_INFO) << "Waiting for all data processed done or CTRL-C ...";
      // wait for all data processed done or CTRL-C received
      while (lio_core->containsEnoughDataForSyncPackages() && !stop_signal_recv)
      {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      lslog(LSLOG_INFO) << "All data processed done, stop lio_core ...";
      lio_core->stop();
    }
    publisher.stop();  // stop publisher after lio_core stopped
  }
  catch (const rosbag::BagException& e)
  {
    lslog(LSLOG_ERROR) << "Failed to open bag file: " << bag_filename << ", error message: " << e.what();
    return LioCoreRosErrorCode::EC_LIO_PARSE_HBC_FILE;
  }

  return LioCoreRosErrorCode::EC_LIO_OK;
}

};  // namespace

LIOHANDLE LioInterface_Init()
{
  lslog(LSLOG_INFO) << "LioInterface_Init()";
  if (handler)
  {
    std::cerr << "LioInterface_Init can only be called once" << std::endl;
  }
  handler = malloc(1);
  started_flag = false;
  progress = 0;
  return handler;
}

int LioInterface_Cleanup(LIOHANDLE* handle_ptr)
{
  lslog(LSLOG_INFO) << "LioInterface_Cleanup(" << handle_ptr << ")";
  if (handle_ptr == nullptr)
  {
    lslog(LSLOG_ERROR) << "handle_ptr is nullptr";
    return (int)LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }
  if (*handle_ptr != handler)
  {
    lslog(LSLOG_ERROR) << "handle_ptr pointer mismatch";
    return (int)LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }

  free(handler);
  handler = nullptr;
  return (int)LioCoreRosErrorCode::EC_LIO_OK;
}

int LioInterface_Start(LIOHANDLE self, const LioStartParam* param)
{
  if (!fs::exists(param->output_dir) || !fs::is_directory(param->output_dir))
  {
    lslog(LSLOG_ERROR) << "output_dir: " << param->output_dir << " doesn't exist or not a directory";
    return (int)LioCoreRosErrorCode::EC_LIO_OUTPUT_DIR_NOT_EXISTS;
  }
  xgrids_lio::lsLogger::init(xgrids_lio::xgridsLogLevel::info, std::string(param->output_dir) + "/lio.log", true);

  progress = 0;
  std::shared_ptr<void> ptr_guard(nullptr, [](void*) { progress = 100.0; });
  ///////////////// check parameters /////////////////
  if (!fs::exists(param->output_dir_temp) || !fs::is_directory(param->output_dir_temp))
  {
    lslog(LSLOG_ERROR) << "output_dir_temp: " << param->output_dir_temp << " doesn't exist or not a directory";
    return (int)LioCoreRosErrorCode::EC_LIO_OUTPUT_DIR_TEMP_NOT_EXISTS;
  }
  if (!fs::exists(param->bag_filename) || !fs::is_regular_file(param->bag_filename))
  {
    lslog(LSLOG_ERROR) << "bag_filename: " << param->bag_filename << " doesn't exist or not a regular file";
    return (int)LioCoreRosErrorCode::EC_LIO_MISSING_HBC_FILE;
  }

  if (self == nullptr)
  {
    lslog(LSLOG_ERROR) << "self is nullptr";
    return (int)LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }
  if (self != handler)
  {
    lslog(LSLOG_ERROR) << "self pointer mismatch";
    return (int)LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }
  if (started_flag)
  {
    lslog(LSLOG_ERROR) << "LioInterface_Start can only be called once";
    return (int)LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }
  started_flag = true;

  std::string dll_path = getDynamicLibraryPath();
  lslog(LSLOG_INFO) << "getDynamicLibraryPath: " << dll_path;

  if (param->device_type == LioDeviceType::LIO_DT_LIXEL_K1)
  {
    lslog(LSLOG_INFO) << "start with K1, bag_filename: " << param->bag_filename << " output_dir: " << param->output_dir
                      << " output_dir_temp: " << param->output_dir_temp;
    return (int)startImpl(
        std::string(dll_path + "/config/K1.yaml"),
        std::string(param->bag_filename),
        false,
        "",
        std::string(param->output_dir),
        std::string(param->output_dir_temp),
        CalibModel::K1);
  }
  else if (param->device_type == LioDeviceType::LIO_DT_LIXEL_V2)
  {
    lslog(LSLOG_INFO) << "start with L2, bag_filename: " << param->bag_filename << " output_dir: " << param->output_dir
                      << " output_dir_temp: " << param->output_dir_temp;
    return (int)startImpl(
        std::string(dll_path + "/config/L2.yaml"),
        std::string(param->bag_filename),
        false,
        "",
        std::string(param->output_dir),
        std::string(param->output_dir_temp),
        CalibModel::L2_LIKE);
  }
  else if (param->device_type == LioDeviceType::LIO_DT_LIXEL_L2PRO)
  {
    lslog(LSLOG_INFO) << "start with L2, bag_filename: " << param->bag_filename << " output_dir: " << param->output_dir
                      << " output_dir_temp: " << param->output_dir_temp;
    return (int)startImpl(
        std::string(dll_path + "/config/L2PRO.yaml"),
        std::string(param->bag_filename),
        false,
        "",
        std::string(param->output_dir),
        std::string(param->output_dir_temp),
        CalibModel::L2_LIKE);
  }
  else if (param->device_type == LioDeviceType::LIO_DT_LIXEL_V1)
  {
    std::string external_calib_filepath = dll_path + "/L1CALIB";
    lslog(LSLOG_INFO) << "start with L1, bag_filename: " << param->bag_filename << " output_dir: " << param->output_dir
                      << " output_dir_temp: " << param->output_dir_temp
                      << " external_calib_filepath: " << external_calib_filepath;
    return (int)startImpl(
        std::string(dll_path + "/config/L1.yaml"),
        std::string(param->bag_filename),
        true,
        external_calib_filepath,
        std::string(param->output_dir),
        std::string(param->output_dir_temp),
        CalibModel::L2_LIKE);
  }
  else
  {
    return (int)LioCoreRosErrorCode::EC_LIO_UNSUPPORTED_DEVICE_MODEL;
  }

  return 0;
}

int LioInterface_Cancel(LIOHANDLE handle)
{
  lslog(LSLOG_INFO) << "LioInterface_Cancel(" << handle << ")";
  if (handle == nullptr)
  {
    lslog(LSLOG_ERROR) << "handle is nullptr";
    return (int)LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }
  if (handle != handler)
  {
    lslog(LSLOG_ERROR) << "handle pointer mismatch";
    return (int)LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }

  stop_signal_recv = true;
  return (int)LioCoreRosErrorCode::EC_LIO_OK;
}

int LioInterface_GetProgress(LIOHANDLE self, double* progress_out)
{
  lslog(LSLOG_INFO) << "LioInterface_GetProgress(" << self << "," << progress_out << ")";
  if (self == nullptr)
  {
    lslog(LSLOG_ERROR) << "self is nullptr";
    return (int)LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }
  if (self != handler)
  {
    lslog(LSLOG_ERROR) << "self pointer mismatch";
    return (int)LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }

  *progress_out = progress;
  return (int)LioCoreRosErrorCode::EC_LIO_OK;
}
