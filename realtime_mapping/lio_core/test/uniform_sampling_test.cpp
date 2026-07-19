//
// Created by youyuan on 23-12-5.
//
#include "interface/uniform_sampling.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "sensor_fusion/static_fusion.h"
#include "xmap.h"
using namespace lixel;
float proc_gb;
std::string path = "/home/youyuan/Datasets/static/V2/2024-02-06-071100.hbc";
std::string xmap_path = "/home/youyuan/realtime_mapping_ws/src/lio_core_ros/config/l2/l2.yaml";
constexpr int MAX_POINT_NUM_IN_ONE_AXIS = 3000;
constexpr int MAX_ITER_NUM = 100;
constexpr float RESOLUTION = 0.1;
TEST(UniformSampling, index_test)
{
  pcl::UniformSampling<PointXYZINormal> uniform_sampling_surf_;
  PointCloudXYZINormal::Ptr pcl_load(new PointCloudXYZINormal);
  PointCloudXYZINormal::Ptr pcl_map(new PointCloudXYZINormal);
  PointCloudXYZINormal::ConstPtr pcl_xmap(new PointCloudXYZINormal);

  pcl::io::loadPCDFile<PointXYZINormal>("/home/youyuan/tmp/init_pcl_down_body.pcd", *pcl_load);
  uniform_sampling_surf_.setRadiusSearch(0.1);
  uniform_sampling_surf_.setInputCloud(pcl_load);
  uniform_sampling_surf_.filter(*pcl_map);

  xmap::Xmap xmap(xmap_path);
  xmap.mapIncremental(pcl_map, V3F(0, 0, 0));
  pcl_xmap = xmap.getMapPointCloud();
  std::cout << "pcl_load.size:" << pcl_load->size() << std::endl;
  std::cout << "pcl_map.size:" << pcl_map->size() << std::endl;
  std::cout << "pcl_xmap.size:" << pcl_xmap->size() << std::endl;
}

TEST(UniformSampling, random_test)
{
  UniformSampling<PointXYZINormal> uniform_sampling_surf_;
  PointCloudXYZINormal::Ptr pcl_input(new PointCloudXYZINormal);
  PointCloudXYZINormal::Ptr pcl_output(new PointCloudXYZINormal);
  for (int i = 0; i < MAX_POINT_NUM_IN_ONE_AXIS; ++i)
  {
    for (int j = 0; j < MAX_POINT_NUM_IN_ONE_AXIS; ++j)
    {
      PointXYZINormal p;
      p.x = i * RESOLUTION;
      p.y = j * RESOLUTION;
      p.z = 0;
      pcl_input->push_back(p);
    }
  }

  uniform_sampling_surf_.setRandomSeed(1000);
  for (int i = 0; i < MAX_ITER_NUM; ++i)
  {
    uniform_sampling_surf_.setRadius(1.0);
    uniform_sampling_surf_.setInputCloud(pcl_input);
    uniform_sampling_surf_.filter(*pcl_output);
    pcl::io::savePCDFileASCII("/home/youyuan/tmp/temp" + std::to_string(i) + ".pcd", *pcl_output);
  }
}
