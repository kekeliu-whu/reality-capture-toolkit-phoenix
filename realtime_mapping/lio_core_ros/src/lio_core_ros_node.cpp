
#include <glog/logging.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <signal.h>
#include <stdlib.h>
#include <boost/filesystem.hpp>

#include "lio.h"
#include "lio_core_ros/parameters_reader.h"
#include "lio_core_ros/publisher.h"
#include "lio_core_ros/subscriber.h"
#include "log/lsLogger.h"

static volatile sig_atomic_t g_stop_signal_recv = 0;
DEFINE_string(algorithm_config_filename, "/empty", "Config file.");
DEFINE_int32(calib_model, 0, "Device model: l1/l2=0 k1=1, used for reading calibration data.");
DEFINE_bool(use_external_calib, false, "Force to use hbc calibration data.");
DEFINE_string(external_calib_filepath, "/empty", "Directory to store default calibration files.");
DEFINE_bool(
    offline_mode,
    false,
    "Runtime mode: online or offline, when offline mode is enabled, bag filename will be "
    "required.");
DEFINE_string(bag_filename, "", "Bag file to read in offline mode.");
DEFINE_string(output_dir, "", "Directory to store output files.");
DEFINE_string(output_dir_temp, "", "Directory to store temporary files.");

namespace
{

namespace fs = boost::filesystem;

void RemoveAllYamlFilesInDir(const std::string &path)
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
  catch (const fs::filesystem_error &e)
  {
    std::cerr << e.what() << std::endl;
  }
}

}  // namespace

static void SigIntHandler(int signo)
{
  g_stop_signal_recv = 1;
}

int main(int argc, char **argv)
{
  ///////////////////////////// 1. Initialize /////////////////////////////
  // init ROS
  ros::init(argc, argv, "lio_core_ros_node");
  ros::NodeHandle nh("lio_core_ros_node");

  // init logging
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // remove FLAGS_output_dir << "/lio.log" if it exists
  if (fs::exists(FLAGS_output_dir + "/lio.log"))
  {
    fs::remove(FLAGS_output_dir + "/lio.log");
  }
  std::cerr << "Logging to " << FLAGS_output_dir << "/lio.log" << std::endl;
  xgrids_lio::lsLogger::init(xgrids_lio::xgridsLogLevel::info, FLAGS_output_dir + "/lio.log");

  signal(SIGINT, SigIntHandler);

  ///////////////////////////// 2. Read Parameters /////////////////////////////
  std::string real_calib_filepath = FLAGS_external_calib_filepath;
  if (!FLAGS_use_external_calib)
  {
    real_calib_filepath = FLAGS_output_dir_temp;
    lslog(LSLOG_INFO) << "Force to use HBC calibration data, real_calib_filepath will be set to output_dir: "
                      << real_calib_filepath;

    RemoveAllYamlFilesInDir(real_calib_filepath);

    ExtractYamlFileFromHbc(FLAGS_bag_filename, real_calib_filepath);
  }

  CHECK(FLAGS_calib_model == 0 || FLAGS_calib_model == 1) << "Invalid device model, only support 0(L1/L2) or 1(K1)";
  FullParameters params;
  if (ReadParameters(FLAGS_algorithm_config_filename, real_calib_filepath, params, (CalibModel)FLAGS_calib_model) !=
      LioCoreRosErrorCode::EC_LIO_OK)
  {
    lslog(LSLOG_ERROR) << "ReadParameters() failed, exit.";
    return EXIT_FAILURE;
  }
  params.map_param.config_path = FLAGS_algorithm_config_filename;

  //////////////////// 3. Start LIO ////////////////////
  std::shared_ptr<lixel::LioCore> lio_core{new lixel::LioCore(params)};
  Publisher publisher{params, FLAGS_output_dir, FLAGS_output_dir_temp};
  Subscriber subscriber{params, (CalibModel)FLAGS_calib_model};
#ifdef __linux__
  publisher.setNodeHandle(nh);
  subscriber.setNodeHandle(nh);
#endif

  publisher.registerCallbacks(lio_core);
  if (FLAGS_offline_mode)
  {
    CHECK(!FLAGS_bag_filename.empty());
    lslog(LSLOG_INFO) << "Using offline mode ...";
    rosbag::Bag bag;
    bag.open(FLAGS_bag_filename);
    lslog(LSLOG_INFO) << "Reading bag file " << FLAGS_bag_filename << " from "
                      << params.ros_param.offline_mode_start_frame_id << " to "
                      << params.ros_param.offline_mode_end_frame_id << " by speed_ratio "
                      << params.ros_param.offline_mode_speed_ratio;
    subscriber.setLioCore(lio_core);  // set lio_core because it will be called in callbacks
    lio_core->start();
    subscriber.processBag(bag, g_stop_signal_recv);  // process messages until g_signal_recv is set

    bag.close();
  }
  else
  {
    subscriber.registerHandlers(lio_core);
    lio_core->start();
    // waiting for interrupt signal
    while (ros::ok() && !g_stop_signal_recv)
    {
      ros::spinOnce();
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }

  if (g_stop_signal_recv)
  {
    lslog(LSLOG_INFO) << "Received interrupt signal SIGINT";
  }
  else
  {
    lslog(LSLOG_INFO) << "ros::ok() returns false or other cases";
  }

  ///////////////////////////// 4. Exit /////////////////////////////
  if (!FLAGS_offline_mode || g_stop_signal_recv)
  {
    // if either of the two conditions is true, stop lio_core immediately
    // 1. online mode is on
    // 2. offline mode is on and CTRL-C is received
    lio_core->stop();
  }
  else
  {
    lslog(LSLOG_INFO) << "Waiting for all data processed done or CTRL-C ...";
    // wait for all data processed done or CTRL-C received
    while (lio_core->containsEnoughDataForSyncPackages() && !g_stop_signal_recv)
    {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    lslog(LSLOG_INFO) << "All data processed done, stop lio_core ...";
    lio_core->stop();
  }
  publisher.stop();  // stop publisher after lio_core stopped

  return EXIT_SUCCESS;
}
