#pragma once

#ifdef __linux__
#include <ros/ros.h>
#endif

#include "lio.h"
#include "lio_core_ros/data_io.h"
#include "lio_core_ros/ulog.h"
#include "parameters_reader.h"

#ifdef SAVE_LAS
#include <liblas/liblas.hpp>
#endif

class Publisher
{
 public:
  Publisher(const FullParameters &params, const std::string &output_dir, const std::string &output_dir_temp);

#ifdef __linux__
  void setNodeHandle(ros::NodeHandle &nh);
#endif

  void registerCallbacks(std::shared_ptr<lixel::LioCore> lio_core);

  void stop();

  virtual ~Publisher();

 private:
  void publishLioResultAsCallback(const lixel::LioResultMsg::Ptr &msg);

  void saveUlog(const lixel::LioResultMsg::Ptr &msg);

  void saveXbc(const lixel::LioResultMsg::Ptr &msg);

  void saveLas(const lixel::LioResultMsg::Ptr &msg);

  void publishToRviz(const lixel::LioResultMsg::Ptr &msg);

 private:
#ifdef __linux__
  ros::NodeHandle nh_;
  ros::Publisher pub_laser_cloud_full_;
  ros::Publisher pub_path_;
  ros::Publisher pub_odom_after_mapped_;
  ros::Publisher pub_imu_;
#endif

  std::mutex mtx_for_exit_;
  std::atomic<bool> exit_ = false;

  std::shared_ptr<middleware::ULogStorage> log_storage_;
  std::string output_dir_;
  std::string output_dir_temp_;
  DataIO data_io_;

  const FullParameters &params_;

#ifdef SAVE_LAS
  liblas::Header header;
  bool first_flag = true;
  std::ofstream m_output_las;
  int point_size = 0;
#endif
};
