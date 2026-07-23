//
// Created by youyuan on 23-12-5.
//
#include <gtest/gtest.h>
#include <malloc.h>
#include <sys/resource.h>
#include <random>

#include "liblas/liblas.hpp"
#include "mem_monitor.h"
#include "xmap.h"

using namespace xmap;
using namespace std;

constexpr int FORGET_FRAME_NUM = 100;
constexpr int TEST_FRAME_NUM = 100;
std::string path = "/home/youyuan/lixel-algorithm-l2/src/lixel_lio/xmap/configs/map.yaml";

/*** 按TEST_FRAME_NUM插入点云，每组点云正好充满一个格子 ***/
/*** 遗忘FORGET_FRAME_NUM个时刻，则必定遗忘对应格子，以此check点数是否正确 ***/
TEST(ForgetTest, forget) {
  float proc_gb;
  Xmap xmap(path);
  int map_scale = xmap.getConfigs().small_scale;
  float map_size = xmap.getConfigs().small_voxel_size;
  int pcl_size = map_scale * map_scale * map_scale;
  std::cout << "pcl_size:" << pcl_size << std::endl;

  auto start_mapping = std::chrono::high_resolution_clock::now();
  for (int l = 0; l < TEST_FRAME_NUM; ++l) {
    PointCloudPtr pointCloudPtr(new PointCloud);
    pointCloudPtr->points.reserve(pcl_size);
    pointCloudPtr->header.stamp = l * 1e5;

    for (int i = 0; i < map_scale; ++i) {
      for (int j = 0; j < map_scale; ++j) {
        for (int k = 0; k < map_scale; ++k) {
          PointType pointType;
          pointType.x = 0.05 + i * xmap.getConfigs().resolution;
          pointType.y = 0.05 + j * xmap.getConfigs().resolution;
          pointType.z = 0.05 + k * xmap.getConfigs().resolution + l * map_size;
          pointCloudPtr->push_back(pointType);
        }
      }
    }
    xmap.mapIncremental(pointCloudPtr, V3F(0, 0, 100));
  }
  auto end_mapping = std::chrono::high_resolution_clock::now();
  double duration_mapping =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_mapping - start_mapping).count();
  std::cout << "建图耗时：" << duration_mapping << " 毫秒" << std::endl;

  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "遗忘前Memory usage:" << proc_gb << "GB" << endl;
  cout << "pointSize:" << xmap.pointSize() << endl;
  cout << "==================================" << endl;
  ASSERT_EQ(xmap.pointSize(), pcl_size * TEST_FRAME_NUM);

  auto start_forget = std::chrono::high_resolution_clock::now();
  xmap.forget(FORGET_FRAME_NUM);
  auto end_forget = std::chrono::high_resolution_clock::now();
  double duration_forget =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_forget - start_forget).count();
  std::cout << "遗忘耗时：" << duration_forget << " 毫秒" << std::endl;

  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "遗忘后Memory usage:" << proc_gb << "GB" << endl;
  cout << "pointSize:" << xmap.pointSize() << endl;
  cout << "==================================" << endl;
  ASSERT_EQ(xmap.pointSize(), pcl_size * (TEST_FRAME_NUM - FORGET_FRAME_NUM));
}