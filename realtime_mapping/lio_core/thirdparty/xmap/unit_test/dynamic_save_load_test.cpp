//
// Created by youyuan on 23-12-5.
//

#include <gtest/gtest.h>
#include <malloc.h>
#include <random>
#include "liblas/liblas.hpp"
#include "mem_monitor.h"
#include "xmap.h"
#include "xmap_test.h"
#include "xmap_util.h"

using namespace xmap;
using namespace std;

float proc_gb;
std::string path = "/home/youyuan/lixel-algorithm-l2/src/lixel_lio/xmap/configs/map.yaml";

/*** 动态加载：落盘 ***/
TEST(Dynamic, Save) {
  Xmap xmap(path);
  XmapTest xmapTest;
  std::cout << "bin path:" << xmap.getConfigs().pcd_path << std::endl;
  // 读取点云数据并加载到xmap
  {
    PointCloudPtr pointCloudPtr(new PointCloud);
    liblas::ReaderFactory reader_factory;
    std::ifstream ifs;
    ifs.open(xmap.getConfigs().test_data_path, std::ios::in);
    liblas::Reader reader = reader_factory.CreateWithStream(ifs);
    // 读取点云数据
    while (reader.ReadNextPoint()) {
      liblas::Point const& p = reader.GetPoint();
      PointType pointType;
      pointType.x = p.GetX() - reader.GetHeader().GetOffsetX();
      pointType.y = p.GetY() - reader.GetHeader().GetOffsetY();
      pointType.z = p.GetZ() - reader.GetHeader().GetOffsetZ();
      pointCloudPtr->push_back(pointType);
    }
    std::cout << "size:" << pointCloudPtr->size() << std::endl;
    std::vector<PointType, Eigen::aligned_allocator<PointType>> pv;
    xmap::TicToc startMapping;
    xmap.mapIncremental(pointCloudPtr, V3F(0, 0, 100));
    std::cout << "Mapping time cost:" << startMapping.Toc() << "ms" << std::endl;
  }
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "size:" << xmap.pointSize() << endl;
  cout << "PointCloud + Structure Memory usage:" << proc_gb << "GB" << endl;

  xmap::TicToc startSave;
  xmapTest.dynamicSaveTest(xmap, V3F(0, 0, 100));

  std::cout << "DynamicSave time cost:" << startSave.Toc() << "ms" << std::endl;
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "After Save point size:" << xmap.pointSize() << endl;
  cout << "After Save Memory usage:" << proc_gb << "GB" << endl;
}

/*** 动态加载：加载 ***/
TEST(Dynamic, Load) {
  Xmap xmap(path);
  XmapTest xmapTest;
  std::cout << "bin path:" << xmap.getConfigs().pcd_path << std::endl;

  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "size:" << xmap.pointSize() << endl;
  cout << "PointCloud + Structure Memory usage:" << proc_gb << "GB" << endl;

  xmap::TicToc startLoad;
  xmapTest.dynamicLoadTest(xmap, V3F(250, -150, 100));
  std::cout << "dynamicLoad time cost:" << startLoad.Toc() << "ms" << std::endl;
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "After Load point size:" << xmap.pointSize() << endl;
  cout << "After Load Memory usage:" << proc_gb << "GB" << endl;
}

/*** 测试readPointCloudFromFile，writePointCloudToFile的正确性 ***/
TEST(Dynamic, IO) {
  // 创建一些测试数据
  std::vector<PointType, Eigen::aligned_allocator<PointType>> testData;
  pcl::PointXYZINormal point1;
  point1.x = 1;
  point1.y = 2;
  point1.z = 3;
  point1.normal_x = 1;
  point1.normal_y = 2;
  point1.normal_z = 3;

  pcl::PointXYZINormal point2;
  point2.x = 4;
  point2.y = 5;
  point2.z = 6;
  point2.normal_x = 4;
  point2.normal_y = 5;
  point2.normal_z = 6;

  testData.push_back(point1);
  testData.push_back(point2);

  // 定义测试文件名
  std::string filename_test = "test_output.bin";
  // 写入数据到文件
  writePointCloudToFile(testData, filename_test);
  // 从文件中读取数据
  std::vector<PointType, Eigen::aligned_allocator<PointType>> readData =
      readPointCloudFromFile(filename_test);

  // 检查读取的数据与原始数据是否一致
  ASSERT_EQ(readData.size(), testData.size());
  for (std::size_t i = 0; i < testData.size(); ++i) {
    std::cout << testData[i] << std::endl;
    ASSERT_EQ(readData[i].x, testData[i].x);
    ASSERT_EQ(readData[i].y, testData[i].y);
    ASSERT_EQ(readData[i].z, testData[i].z);
    ASSERT_EQ(readData[i].normal_x, testData[i].normal_x);
    ASSERT_EQ(readData[i].normal_y, testData[i].normal_y);
    ASSERT_EQ(readData[i].normal_z, testData[i].normal_z);
  }
  // 删除测试文件
  std::remove(filename_test.c_str());
}
