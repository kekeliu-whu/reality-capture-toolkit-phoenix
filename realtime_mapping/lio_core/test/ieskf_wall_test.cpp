//
// Created by youyuan on 23-12-5.
//

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <iostream>
#include <random>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ieskf/ieskf.h"
#include "sensor_fusion/lidar_fusion.h"
#include "sensor_fusion/static_fusion.h"
#include "xmap.h"
using namespace lixel;
float proc_gb;
int map_size = 300;
std::string path = "/home/youyuan/Datasets/static/V2/2024-02-06-071100.hbc";
std::string config_path = "/home/youyuan/lixel-algorithm-l2/src/lixel_lio/xmap/configs/map.yaml";

std::random_device rd;                                    // 使用随机设备作为随机种子
std::mt19937 gen(rd());                                   // 梅森旋转算法作为随机数引擎
std::normal_distribution<float> distribution(0.0, 0.03);  // 均值为 0，标准差为 0.01 的高斯分布
double resolution = 0.1;
int counter = map_size / resolution;
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
  LiDARFusion liDarFusion;
  std::vector<ImuMsg> imu_vec;
  StatePredict state_predict;
  bool is_init = false;

  // Build The Wall ! ! !
  std::shared_ptr<xmap::Xmap> xmap_ptr = std::make_shared<xmap::Xmap>(config_path);
  {
    xmap::PointCloudPtr pointCloudPtr(new xmap::PointCloud);
    for (int i = -counter; i < counter; ++i)
    {
      for (int j = -counter; j < counter; ++j)
      {
        xmap::PointType pointType;
        pointType.x = 0.05 + i * resolution;
        pointType.y = 0.05 + j * resolution;
        pointType.z = 100.05;
        pointCloudPtr->push_back(pointType);
      }
    }

    for (int i = -counter; i < counter; ++i)
    {
      for (int j = -counter; j < counter; ++j)
      {
        xmap::PointType pointType;
        pointType.x = 0.05 + i * resolution;
        pointType.y = 100.05;
        pointType.z = 0.05 + j * resolution;
        pointCloudPtr->push_back(pointType);
      }
    }

    for (int i = -counter; i < counter; ++i)
    {
      for (int j = -counter; j < counter; ++j)
      {
        xmap::PointType pointType;
        pointType.x = 100.05;
        pointType.y = 0.05 + i * resolution;
        pointType.z = 0.05 + j * resolution;
        pointCloudPtr->push_back(pointType);
      }
    }

    xmap_ptr->mapIncremental(pointCloudPtr, xmap::V3F(0, 0, 0));
    liDarFusion.setXmap(xmap_ptr);
  }

  std::vector<V3F> lidar_meas;
  // z wall
  for (int i = -50; i < 50; ++i)
  {
    for (int j = -50; j < 50; ++j)
    {
      V3F p;
      p.x() = 0.05 + i + distribution(gen);
      p.y() = 0.05 + j + distribution(gen);
      p.z() = 100.05 + distribution(gen);
      lidar_meas.push_back(p);
    }
  }

  // y wall
  for (int i = -50; i < 50; ++i)
  {
    for (int j = -50; j < 50; ++j)
    {
      V3F p;
      p.x() = 0.05 + i + distribution(gen);
      p.y() = 100.05 + distribution(gen);
      p.z() = 0.05 + j + distribution(gen);
      lidar_meas.push_back(p);
    }
  }

  // x wall
  for (int i = -50; i < 50; ++i)
  {
    for (int j = -50; j < 50; ++j)
    {
      V3F p;
      p.x() = 100.05 + distribution(gen);
      p.y() = 0.05 + i + distribution(gen);
      p.z() = 0.05 + j + distribution(gen);
      lidar_meas.push_back(p);
    }
  }
  std::cout << "lidar_meas.size:" << lidar_meas.size() << std::endl;

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
      }
      else
      {
        auto start = std::chrono::high_resolution_clock::now();
        liDarFusion.setLidarMeas(lidar_meas);
        AttributeIterate attri_iter;
        AttributePredict attri_predict;
        ieskf.predict(
            meaureGroup.imu_vec, meaureGroup.pcl_end_time, 0, state_predict, attri_predict);
        ieskf.update(liDarFusion, attri_iter);
        ieskf.logState();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "耗时：" << duration << " 毫秒" << std::endl;
      }
      imu_vec.clear();
      usleep(5000);
    }
  }
  bag.close();
}