//
// Created by youyuan on 24-3-15.
//
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <fstream>
#include <liblas/header.hpp>
#include <liblas/liblas.hpp>
#include <liblas/point.hpp>
#include <liblas/writer.hpp>
#include "lixel_msgs/LioFullStates.h"

int main(int argc, char* argv[])
{
  if (argc < 3)
  {
    // 如果未提供足够的命令行参数，则输出提示信息并退出
    std::cerr << "Usage: " << argv[0] << " <input_string1: paht of xbc> <input_string2: path of las>" << std::endl;
    return 1;
  }
  // 从命令行参数中获取两个字符串
  std::string xbc_path = argv[1];
  // std::string las_path = argv[2];
  // 输出读取到的两个字符串
  std::cout << "Input string 1 from command line: " << xbc_path << std::endl;

  rosbag::Bag bag;
  bag.open(xbc_path, rosbag::bagmode::Read);

  // 创建一个topic列表，用于筛选需要读取的topic
  std::vector<std::string> topics;
  topics.emplace_back("/algorithm/l1f_pcbin");
  topics.emplace_back("/algorithm/fullstates_bin");
  // 创建一个view来读取bag中的消息
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  // 遍历bag中的消息
  std::deque<lixel_msgs::LioFullStates::ConstPtr> lio_full_states_deque;
  std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> point_cloud_deque;
  for (rosbag::MessageInstance const& m : view)
  {
    std::string topic = m.getTopic();
    if (topic == "/algorithm/l1f_pcbin")
    {
      sensor_msgs::PointCloud2::ConstPtr cloud = m.instantiate<sensor_msgs::PointCloud2>();
      pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
      // 将sensor_msgs::PointCloud2消息转换为PCL格式
      pcl::fromROSMsg(*cloud, *pcl_cloud);
      point_cloud_deque.push_back(pcl_cloud);
      // 处理sensor_msgs::PointCloud2类型的消息
    }
    else if (topic == "/algorithm/fullstates_bin")
    {
      lixel_msgs::LioFullStates::ConstPtr lioStates = m.instantiate<lixel_msgs::LioFullStates>();
      lio_full_states_deque.push_back(lioStates);
    }
  }
  // 关闭bag文件
  bag.close();

  std::ofstream file("/tmp/pose.csv");
  for (const lixel_msgs::LioFullStates::ConstPtr& pose : lio_full_states_deque)
  {
    file << std::fixed << pose->header.stamp.toSec() << " " << pose->p.x << " " << pose->p.y << " " << pose->p.z << " "
         << pose->q.w << " " << pose->q.x << " " << pose->q.y << " " << pose->q.z << "\n";
  }

  file.close();
}
