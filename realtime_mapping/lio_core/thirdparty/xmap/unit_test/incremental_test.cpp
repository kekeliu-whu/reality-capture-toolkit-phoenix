//
// Created by youyuan on 23-12-5.
// 本单元测试关于xivt的部分需要移动到父仓库中才能运行
//
#include <gtest/gtest.h>
#include <malloc.h>
#include <sys/resource.h>
#include <random>

#include "liblas/liblas.hpp"
#include "mem_monitor.h"
#include "xmap.h"
#include "xmap_test.h"
#include "xmap_util.h"
using namespace xmap;
using namespace std;

int map_size = 75;
int h = 1;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, 0.01);
std::string path = "/home/youyuan/lixel-algorithm-l2/src/lixel_lio/xmap/configs/map.yaml";
XmapTest xmap_test;
float proc_gb;

/*** 按仿真数据一次性对Xmap进行地图增广, 查看内存占用是否符合预期 ***/
TEST(IncrementalTest, mapIncremental) {
  float start_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  Xmap xmap(path);
  cout << "map_size:" << map_size << endl;
  {
    PointCloudPtr pointCloudPtr(new PointCloud);
    pointCloudPtr->points.reserve(h * 4 * map_size * map_size);
    cout << "pcl_size:" << h * 4 * map_size * map_size << endl;
    int counter = 0;
    for (int i = -map_size; i < map_size; ++i) {
      for (int j = -map_size; j < map_size; ++j) {
        for (int k = 0; k < h; ++k) {
          PointType pointType;
          pointType.x = 0.05 + i * 0.2;
          pointType.y = 0.05 + j * 0.2;
          pointType.z = 0.05 + k * 0.2;
          pointCloudPtr->push_back(pointType);
          counter++;
        }
      }
    }
    cout << "counter:" << counter << endl;
    proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
    cout << "finish PointCloud Memory usage:" << proc_gb - start_gb << "GB" << endl;

    TicToc start;
    xmap.mapIncremental(pointCloudPtr, V3F(0, 0, 100));
    // 输出耗时
    std::cout << "Xmap 耗时：" << start.Toc() << " 毫秒" << std::endl;
  }
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "xmap point_size:" << xmap.pointSize() << endl;
  cout << "PointCloud + Structure Memory usage:" << proc_gb - start_gb << "GB" << endl;

  for (auto& entry : xmap_test.getSmallVoxelMap(xmap)) {
    entry.second.kd_tree_ = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>);
  }
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "Remove KDTREE Memory usage:" << proc_gb - start_gb << "GB" << endl;

  for (auto& entry : xmap_test.getSmallVoxelMap(xmap)) {
    entry.second.index_map_.clear();
  }
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "Remove index_map + KDTREE Memory usage:" << proc_gb - start_gb << "GB" << endl;

  for (auto& entry : xmap_test.getSmallVoxelMap(xmap)) {
    entry.second.cloud_->points.shrink_to_fit();
  }
  for (auto& entry : xmap_test.getSmallVoxelMap(xmap)) {
    entry.second.no_normal_points_.clear();
    entry.second.no_normal_points_.shrink_to_fit();
  }
  xmap_test.getLargeVoxelMap(xmap).clear();
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "after shrink_to_fit Memory usage:" << proc_gb << "GB" << endl;

  //  float final_proc_gb;
  //  std::vector<PointCloudPtr> pcl_vec;
  //  for (auto& entry : xmap_test.getSmallVoxelMap(xmap)) {
  //    PointCloudPtr ptr;
  //    ptr.reset(new PointCloud(*entry.second.cloud_));
  //    pcl_vec.push_back(ptr);
  //  }
  //  final_proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  //  cout << "multi pointcloud Memory usage:" << final_proc_gb - proc_gb << "GB" << endl;

  for (auto& entry : xmap_test.getSmallVoxelMap(xmap)) {
    entry.second.cloud_.reset(new PointCloud());
  }
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "reinit SmallVoxelValue:" << proc_gb << "GB" << endl;
}

/*** 按真实数据一次性对Xmap进行地图增广 ***/
TEST(IncrementalTest, LasXMap) {
  Xmap xmap(path);

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
    {
      xmap::TicToc startMapping;
      xmap.mapIncremental(pointCloudPtr, V3F(0, 0, 100));
      std::cout << "startMapping time cost:" << startMapping.Toc() << "ms" << std::endl;
    }
  }
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "size:" << xmap.pointSize() << endl;
  cout << "PointCloud + Structure Memory usage:" << proc_gb << "GB" << endl;
}

/*** 按真实数据拆分后，多帧对Xmap进行地图增广 ***/
TEST(IncrementalTest, LasXMapEachPoint) {
  Xmap xmap(path);

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
    double ts = 0;
    {
      for (int i = 0; i < pointCloudPtr->size() / 10000; ++i) {
        PointCloudPtr pointCloudFlagPtr(new PointCloud);

        for (int j = i * 10000; j < i * 10000 + 10000; ++j) {
          pointCloudFlagPtr->push_back(pointCloudPtr->at(j));
        }
        auto start = std::chrono::high_resolution_clock::now();
        xmap.mapIncremental(pointCloudFlagPtr, V3F(0, 0, 100));
        auto end = std::chrono::high_resolution_clock::now();
        // 计算耗时（以秒为单位）
        double duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        ts += duration;
      }
    }
    cout << "10000点增广的单帧耗时：" << ts / (pointCloudPtr->size() / 10000) << endl;
  }
}

/*** 按真实数据一次性对Xivt进行地图增广 ***/
// TEST(IncrementalTest, LasXivt) {
//   XiVT::Options xivt_options;
//   xivt_options.m_inv_resolution = 10;
//   xivt_options.m_downsample_resolution = 0.1;
//
//   XiVT xiVt(xivt_options);
//
//   {
//     PointCloudPtr pointCloudPtr(new PointCloud);
//     liblas::ReaderFactory reader_factory;
//     std::ifstream ifs;
//     ifs.open(filename, std::ios::in);
//     liblas::Reader reader = reader_factory.CreateWithStream(ifs);
//     Point_vector p_v;
//     std::vector<XiVT_point_status> status_to_add;
//
//     // 读取点云数据
//     while (reader.ReadNextPoint()) {
//       liblas::Point const& p = reader.GetPoint();
//       PointType pointType;
//       pointType.x = p.GetX();
//       pointType.y = p.GetY();
//       pointType.z = p.GetZ();
//
//       XiVT_point_status xiVtPointStatus;
//       pointCloudPtr->push_back(pointType);
//       p_v.push_back(pointType);
//       status_to_add.push_back(xiVtPointStatus);
//     }
//
//     std::cout << "size:" << pointCloudPtr->size() << std::endl;
//     std::vector<PointType, Eigen::aligned_allocator<PointType>> pv;
//     {
//       xgrids::TicToc startMapping;
//       xiVt.add_points(p_v, 0, status_to_add, true);
//       std::cout << "startMapping time cost:" << startMapping.Toc() << "ms" << std::endl;
//     }
//   }
//   malloc_trim(0);
//   proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
//   cout << "PointCloud + Structure Memory usage:" << proc_gb << "GB" << endl;
// }
//
///*** 按真实数据拆分后，多帧对Xivt进行地图增广 ***/
// TEST(IncrementalTest, LasXivtEachPoint) {
//   XiVT::Options xivt_options;
//   xivt_options.m_inv_resolution = 10;
//   xivt_options.m_downsample_resolution = 0.1;
//
//   XiVT xiVt(xivt_options);
//
//   {
//     PointCloudPtr pointCloudPtr(new PointCloud);
//     liblas::ReaderFactory reader_factory;
//     std::ifstream ifs;
//     ifs.open(filename, std::ios::in);
//     liblas::Reader reader = reader_factory.CreateWithStream(ifs);
//     Point_vector p_v;
//     std::vector<XiVT_point_status> status_to_add;
//
//     // 读取点云数据
//     while (reader.ReadNextPoint()) {
//       liblas::Point const& p = reader.GetPoint();
//       PointType pointType;
//       pointType.x = p.GetX();
//       pointType.y = p.GetY();
//       pointType.z = p.GetZ();
//
//       XiVT_point_status xiVtPointStatus;
//       pointCloudPtr->push_back(pointType);
//       p_v.push_back(pointType);
//       status_to_add.push_back(xiVtPointStatus);
//     }
//
//     std::cout << "size:" << pointCloudPtr->size() << std::endl;
//     std::vector<PointType, Eigen::aligned_allocator<PointType>> pv;
//
//     {
//       for (int i = 0; i < pointCloudPtr->size() / 10000; ++i) {
//         Point_vector p_v_temp;
//         std::vector<XiVT_point_status> status_to_add_temp;
//         for (int j = i * 10000; j < i * 10000 + 10000; ++j) {
//           XiVT_point_status xiVtPointStatus;
//           p_v_temp.push_back(pointCloudPtr->at(j));
//           status_to_add_temp.push_back(xiVtPointStatus);
//         }
//         xgrids::TicToc startMapping;
//       }
//     }
//
//     {
//       xgrids::TicToc startMapping;
//       xiVt.add_points(p_v, 0, status_to_add, true);
//       std::cout << "startMapping time cost:" << startMapping.Toc() << "ms" << std::endl;
//     }
//   }
//   malloc_trim(0);
//   proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
//   cout << "PointCloud + Structure Memory usage:" << proc_gb << "GB" << endl;
// }

/*** 按仿真数据一次性对Xivt进行地图增广 ***/
// TEST(XivtTest, mapIncremental2) {
//   XiVT::Options xivt_options;
//   xivt_options.m_inv_resolution = 10;
//   xivt_options.m_downsample_resolution = 0.1;
//
//   XiVT xiVt(xivt_options);
//
//   cout << "map_size:" << map_size << endl;
//   float proc_gb;
//   {
//     Point_vector p_v;
//     std::vector<XiVT_point_status> status_to_add;
//     status_to_add.reserve(h * 4 * map_size * map_size);
//     p_v.reserve(h * 4 * map_size * map_size);
//     cout << "pcl_size:" << h * 4 * map_size * map_size << endl;
//
//     int counter = 0;
//     for (int i = -map_size; i < map_size; ++i) {
//       for (int j = -map_size; j < map_size; ++j) {
//         for (int k = 0; k < h; ++k) {
//           PointType pointType;
//           XiVT_point_status xiVtPointStatus;
//           pointType.x = 0.05 + i * 0.1;
//           pointType.y = 0.05 + j * 0.1;
//           pointType.z = 0.05 + k * 0.1;
//           p_v.push_back(pointType);
//           status_to_add.push_back(xiVtPointStatus);
//           counter++;
//         }
//       }
//     }
//     cout << "pv_size:" << p_v.size() << endl;
//     proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
//     cout << "finish PointCloud Memory usage:" << proc_gb << "GB" << endl;
//
//     auto start = std::chrono::high_resolution_clock::now();
//     xiVt.add_points(p_v, 0, status_to_add, true);
//     // 输出耗时
//     // xiVt.recompute_normal(1, 150, 15, 20, 5, 5, 1);
//     auto end = std::chrono::high_resolution_clock::now();
//     // 计算耗时（以秒为单位）
//     double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     std::cout << "Xivt 耗时：" << duration << " 毫秒" << std::endl;
//   }
//
//   malloc_trim(0);
//   cout << "xiVT.pointSize:" << xiVt.NumPoints() << endl;
//   proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
//   cout << "PointCloud + Structure Memory usage:" << proc_gb << "GB" << endl;
// }

/*** 单测pcl八叉树深度与resolution的关系 ***/
// TEST(OctoTreeTest, mapIncremental3) {
//   double resolution = 0.1;
//   for (int i = 1; i < 100; ++i) {
//     PointCloudPtr cloud(new PointCloud);
//     pcl::octree::OctreePointCloudSearch<PointType>::Ptr octo_tree(
//         new pcl::octree::OctreePointCloudSearch<PointType>(resolution));
//
//     octo_tree->defineBoundingBox(i * resolution, i * resolution, i * resolution);
//
//     cout << "size:" << resolution * i << ", depth:" << octo_tree->getTreeDepth() << endl;
//   }
// }
