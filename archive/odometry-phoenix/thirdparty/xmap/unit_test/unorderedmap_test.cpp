//
// Created by youyuan on 23-12-5.
//
#include <gtest/gtest.h>
#include <malloc.h>
#include <sys/resource.h>
#include "mem_monitor.h"
// #include "xivt/xivt.hpp"
#include "xmap.h"
#include "xmap_util.h"

using namespace xmap;
using namespace std;

int map_size = 2700;
int h = 1;
double proc_gb;

TEST(UnordedMapTest, eraseTest) {
  unordered_map<int, PointCloud> map;
  {
    PointCloud pointCloud;
    pointCloud.points.reserve(h * 4 * map_size * map_size);
    cout << "pcl_size:" << h * 4 * map_size * map_size << endl;
    int counter = 0;
    for (int i = -map_size; i < map_size; ++i) {
      for (int j = -map_size; j < map_size; ++j) {
        for (int k = 0; k < h; ++k) {
          PointType pointType;
          pointType.x = 0.05 + i * 0.1;
          pointType.y = 0.05 + j * 0.1;
          pointType.z = 0.05 + k * 0.1;
          pointCloud.push_back(pointType);
          counter++;
        }
      }
    }
    map.insert(std::make_pair(0, pointCloud));
  }
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "finish PointCloud Memory usage:" << proc_gb << "GB" << endl;

  map.erase(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "finish PointCloud Memory usage:" << proc_gb << "GB" << endl;
}

TEST(UnordedMapTest, vecTest) {
  std::vector<std::pair<PointPtr, int>> vec;
  {
    cout << "pcl_size:" << h * 4 * map_size * map_size << endl;
    for (int i = -map_size; i < map_size; ++i) {
      for (int j = -map_size; j < map_size; ++j) {
        for (int k = 0; k < h; ++k) {
          PointPtr ptr = std::make_shared<PointType>();
          ptr->x = 0.05 + i * 0.1;
          ptr->y = 0.05 + j * 0.1;
          ptr->z = 0.05 + k * 0.1;
          vec.push_back(std::make_pair(ptr, 0));
        }
      }
    }
  }
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "before swap Memory usage:" << proc_gb << "GB" << endl;

  {
    std::vector<std::pair<PointPtr, int>> empty_vec;
    empty_vec.swap(vec);
  }

  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "after swap Memory usage:" << proc_gb << "GB" << endl;
}