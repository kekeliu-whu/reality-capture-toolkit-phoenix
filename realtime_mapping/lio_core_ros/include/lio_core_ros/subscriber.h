#pragma once

#ifdef __linux__
#include <ros/ros.h>
#endif

#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>

#include "lio.h"
#include "lixel_msgs/CustomMsg.h"
#include "parameters_reader.h"

class Subscriber
{
 public:
  Subscriber(const FullParameters &params, CalibModel calib_model);

#ifdef __linux__
  void setNodeHandle(ros::NodeHandle &nh);
#endif

  void registerHandlers(const std::shared_ptr<lixel::LioCore> &lio_core);
  LioCoreRosErrorCode processBag(const rosbag::Bag &bag, volatile int &stop_signal_recv, double *progress = nullptr);
  void setLioCore(const std::shared_ptr<lixel::LioCore> &lio_core);

  void imuMsgCallback(const sensor_msgs::ImuConstPtr &msg);
  void encoderMsgCallback(const nav_msgs::OdometryConstPtr &msg);
  void lidarMsgCallback(const sensor_msgs::PointCloud2ConstPtr &msg, int frame_id = 0);
  void lidarMsgCallback(const lixel_msgs::CustomMsgConstPtr &msg, int frame_id = 0);
  void gnssMsgCallback(const sensor_msgs::NavSatFixConstPtr &msg);

 private:
  /**
   * @brief wait if internal data queue size is too large
   *
   * @param msg
   */
  void lidarMsgCallback(const lixel::PointCloudMsg::Ptr &msg);

 private:
#ifdef __linux__
  ros::NodeHandle nh_;
  ros::Subscriber sub_imu_;
  ros::Subscriber sub_encoder_;
  ros::Subscriber sub_lidar_;
  ros::Subscriber sub_gnss_;
#endif

  const FullParameters &params_;
  CalibModel calib_model_;

  std::shared_ptr<lixel::LioCore> lio_core_;

 private:
  static constexpr int ARM_IMU_QUEUE_SIZE = 10000;
  static constexpr int ARM_ENCODER_QUEUE_SIZE = 10000;
  static constexpr int ARM_LIDAR_QUEUE_SIZE = 1000;
  static constexpr int ARM_GNSS_QUEUE_SIZE = 10000;

  static constexpr int MAX_CACHED_QUEUE_DURATION = 5;

  static constexpr double TIME_DATA_GAP_LIDAR = 0.25;
  static constexpr double TIME_DATA_GAP_IMU = 0.1;
  static constexpr double TIME_DATA_GAP_ENCODER = 0.05;

  static constexpr double TIME_DATA_NOT_ENOUGH_THRESHOLD = 20;
};
