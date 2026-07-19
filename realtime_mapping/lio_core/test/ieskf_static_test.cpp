//
// Created by youyuan on 23-12-5.
//

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ieskf/ieskf.h"
#include "sensor_fusion/static_fusion.h"
using namespace lixel;
float proc_gb;
std::string path = "/home/youyuan/Datasets/static/V2/2024-02-06-071100.hbc";
TEST(IESKF, StaticPredictAndUpdate)
{
  // 初始化 Google Log
  google::InitGoogleLogging("ieskf_test");
  // 设置日志输出目录
  google::SetLogDestination(google::GLOG_INFO, "logs/");
  // 设置日志文件名前缀
  google::SetLogFilenameExtension("log_");
  // 可选：设置日志级别
  FLAGS_minloglevel = google::INFO;

  IESKFParam ieskfParam;
  IESKF ieskf(ieskfParam);

  rosbag::Bag bag;
  bag.open(path, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back("/imu");  // 替换为你的IMU topic名称
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  MeaureGroup meaureGroup;
  StaticFusion staticFusion;

  std::vector<ImuMsg> imu_vec;
  StatePredict state_predict;
  bool is_init = false;
  for (rosbag::MessageInstance const& m : view)
  {
    sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
    V3D gyro(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    V3D acc(
        imu_msg->linear_acceleration.x,
        imu_msg->linear_acceleration.y,
        imu_msg->linear_acceleration.z);
    ImuMsg imuMsg(imu_msg->header.stamp.toSec(), acc, gyro);
    imu_vec.push_back(imuMsg);
    if (imu_vec.size() >= 20)
    {
      meaureGroup.imu_vec = imu_vec;
      meaureGroup.pcl_start_time = imu_vec.front().timestamp + 1e-6;
      meaureGroup.pcl_end_time = imu_vec.back().timestamp - 1e-6;

      if (!is_init)
      {
        is_init = ieskf.staticInit(meaureGroup.imu_vec, meaureGroup.pcl_end_time);
        staticFusion.setInitRT(ieskf.getStatesPtr()->rot, ieskf.getStatesPtr()->pos);
      }
      else
      {
        AttributeIterate attri_iter;
        AttributePredict attri_predict;
        ieskf.predict(
            meaureGroup.imu_vec, meaureGroup.pcl_end_time, 0, state_predict, attri_predict);
        ieskf.update(staticFusion, attri_iter);
        usleep(5000);
      }
      imu_vec.clear();
    }
  }
  bag.close();
}