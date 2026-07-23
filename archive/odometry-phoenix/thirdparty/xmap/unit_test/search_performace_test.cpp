//
// Created by youyuan on 23-12-5.
//
#include <gtest/gtest.h>
#include <malloc.h>
#include <sys/resource.h>
#include <iostream>
#include <liblas/liblas.hpp>

#include "mem_monitor.h"
// #include "xivt/xivt.hpp"
#include "xmap.h"
#include "xmap_util.h"

using namespace xmap;
using namespace std;

std::string path = "/home/youyuan/lixel-algorithm-l2/src/lixel_lio/xmap/configs/map.yaml";

float proc_gb;

// int map_size = 300;
// int h = 3;
// std::default_random_engine generator, generator2;
// std::normal_distribution<double> distribution1(0.0, 0.01);
// std::normal_distribution<double> distribution2(0.0, 0.02);
//
// TEST(SearchPerformanceTest, XmapSearch) {
//   std::string path = "/home/youyuan/lixel-algorithm-xmap/src/lixel_lio/config/map.yaml";
//   Xmap xmap(path);
//
//   std::vector<PointType, Eigen::aligned_allocator<PointType>> pv;
//   {
//     xgrids::TicToc startMapping;
//     PointCloudPtr pointCloudPtr(new PointCloud);
//     pointCloudPtr->points.reserve(h * 4 * map_size * map_size);
//     cout << "pcl_size:" << h * 4 * map_size * map_size << endl;
//     int counter = 0;
//     for (int i = -map_size; i < map_size; ++i) {
//       for (int j = -map_size; j < map_size; ++j) {
//         for (int k = 0; k < h; ++k) {
//           PointType pointType;
//           pointType.x = 0.05 + i * 0.1;
//           pointType.y = 0.05 + j * 0.1;
//           pointType.z = 0.05 + k * 0.1;
//
//           pointCloudPtr->push_back(pointType);
//           pv.push_back(pointType);
//           counter++;
//         }
//       }
//     }
//     xmap.mapIncremental(pointCloudPtr, V3F(0, 0, 100));
//     xmap.constructKDTree();
//   }
//
//   {
//     xgrids::TicToc startSearch;
//     V3F view_point(0, 0, 100);
//     V3F ref_normal(0, 0, 1);
//     double total_error;
//     int counter = 0;
//     for (int i = -map_size; i < map_size; ++i) {
//       for (int j = -map_size; j < map_size; ++j) {
//         for (int k = 0; k < h; ++k) {
//           std::vector<V3F> nearest_points;
//           V3F search_point(0.05 + i * 0.1, 0.05 + j * 0.1, 0.05 + k * 0.1);
//           xmap.radiusSearch(search_point, view_point, nearest_points, 1);
//           // PlanePtr plane = xmap.planeFitting(nearest_points);
//           counter++;
//         }
//       }
//     }
//     // 输出耗时
//     double ts = startSearch.Toc();
//     std::cout << "Xmap 半径搜索耗时：" << ts << " 毫秒" << std::endl;
//     std::cout << "Xmap 五千次半径搜索耗时：" << 1e4 * ts / counter / 2 << " 毫秒" << std::endl;
//   }
//
//   {
//     xgrids::TicToc startSearch;
//     V3F view_point(0, 0, 100);
//     V3F ref_normal(0, 0, 1);
//     double total_error;
//     int counter = 0;
//     for (int i = -map_size; i < map_size; ++i) {
//       for (int j = -map_size; j < map_size; ++j) {
//         for (int k = 0; k < h; ++k) {
//           std::vector<V3F> nearest_points;
//           V3F search_point(0.05 + i * 0.1, 0.05 + j * 0.1, 0.05 + k * 0.1);
//           xmap.knnSearch(search_point, view_point, nearest_points, 1);
//           // PlanePtr plane = xmap.planeFitting(nearest_points);
//           counter++;
//         }
//       }
//     }
//     // 输出耗时
//     double ts = startSearch.Toc();
//
//     std::cout << "Xmap KNN搜索耗时：" << ts << " 毫秒" << std::endl;
//     std::cout << "Xmap 五千次KNN搜索耗时：" << 1e4 * ts / counter / 2 << " 毫秒" << std::endl;
//   }
// }
//
// TEST(SearchPerformanceTest, XivtSearch2) {
//   XiVT::Options xivt_options;
//   xivt_options.m_inv_resolution = 10;
//   xivt_options.m_downsample_resolution = 0.1;
//
//   XiVT xiVt(xivt_options);
//
//   cout << "map_size:" << map_size << endl;
//
//   {
//     xgrids::TicToc startMapping;
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
//     xiVt.add_points(p_v, 0, status_to_add, true);
//     std::cout << "XiVT 地图增广耗时：" << startMapping.Toc() << " 毫秒" << std::endl;
//   }
//
//   {
//     xgrids::TicToc startSearch;
//     V3F view_point(0, 0, 100);
//     V3F ref_normal(0, 0, 1);
//     int counter = 0;
//     for (int i = -map_size; i < map_size; ++i) {
//       for (int j = -map_size; j < map_size; ++j) {
//         for (int k = 0; k < h; ++k) {
//           std::vector<V3F> nearest_points;
//           V3F search_point(0.05 + i * 0.1, 0.05 + j * 0.1, 0.05 + k * 0.1);
//           // 生成随机噪声并添加到对应的点坐标上
//
//           search_point.x() += distribution2(generator2);
//           search_point.y() += distribution2(generator2);
//           search_point.z() += distribution2(generator2);
//
//           XiVT_point_type point_xivt;
//           point_xivt.x = search_point.x();
//           point_xivt.y = search_point.y();
//           point_xivt.z = search_point.z();
//
//           XiVT_point_vector m_nearest_points_vectors;
//           PlaneT planeT;
//           xiVt.get_closest_point(
//               point_xivt, view_point, m_nearest_points_vectors, planeT, 5, 5, 1, false, 1);
//
//           counter++;
//         }
//       }
//     }
//     // 输出耗时
//     double ts = startSearch.Toc();
//     std::cout << "XiVT KNN搜索耗时：" << ts << " 毫秒" << std::endl;
//     std::cout << "XiVT 五千次KNN搜索耗时：" << 1e4 * ts / counter / 2 << " 毫秒" << std::endl;
//   }
// }

TEST(SearchPerformanceTest, LasXMapSearch) {
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
  cout << "PointCloud + Structure Memory usage:" << proc_gb << "GB" << endl;

  //  {
  //    xgrids::TicToc startSearch;
  //    V3F view_point(0, 0, 100);
  //    int counter = 0;
  //    for (auto& p : pointCloudPtr->points) {
  //      std::vector<V3F> nearest_points;
  //      V3F search_point(p.x, p.y, p.z);
  //      xmap.radiusSearch(search_point, view_point, nearest_points, 1);
  //      // PlanePtr plane = xmap.planeFitting(nearest_points);
  //      counter++;
  //    }
  //    // 输出耗时
  //    double ts = startSearch.Toc();
  //    std::cout << "Xmap 半径搜索耗时：" << ts << " 毫秒" << std::endl;
  //    std::cout << "Xmap 五千次半径搜索耗时：" << 1e4 * ts / counter / 2 << " 毫秒" << std::endl;
  //  }

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

    xmap::TicToc startSearch;
    V3F view_point(0, 0, 100);
    int counter = 0;
    for (auto& p : pointCloudPtr->points) {
      std::vector<V3F> nearest_points;
      V3F search_point(p.x, p.y, p.z);
      xmap.knnSearch(search_point, view_point, nearest_points, 1);
      // PlanePtr plane = xmap.planeFitting(nearest_points);
      counter++;
    }
    // 输出耗时
    double ts = startSearch.Toc();

    std::cout << "Xmap KNN搜索耗时：" << ts << " 毫秒" << std::endl;
    std::cout << "Xmap 五千次KNN搜索耗时：" << 1e4 * ts / counter / 2 << " 毫秒" << std::endl;
  }
  malloc_trim(0);
  proc_gb = Mem_monitor::get_instance().get_proc_mem_usage_in_gb();
  cout << "PointCloud + Structure Memory usage:" << proc_gb << "GB" << endl;
}

// TEST(SearchPerformanceTest, LasXivtSearch) {
//   XiVT::Options xivt_options;
//   xivt_options.m_inv_resolution = 10;
//   xivt_options.m_downsample_resolution = 0.1;
//
//   XiVT xiVt(xivt_options);
//   PointCloudPtr pointCloudPtr(new PointCloud);
//
//   {
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
//
//   {
//     xgrids::TicToc startSearch;
//     V3F view_point(0, 0, 100);
//     V3F ref_normal(0, 0, 1);
//     int counter = 0;
//
//     for (auto& p : pointCloudPtr->points) {
//       std::vector<V3F> nearest_points;
//       V3F search_point(p.x, p.y, p.z);
//
//       counter++;
//
//       XiVT_point_type point_xivt;
//       point_xivt.x = search_point.x();
//       point_xivt.y = search_point.y();
//       point_xivt.z = search_point.z();
//       XiVT_point_vector m_nearest_points_vectors;
//       PlaneT planeT;
//
//       xiVt.get_closest_point(
//           point_xivt, view_point, m_nearest_points_vectors, planeT, 5, 5, 1, false, 1);
//       // PlanePtr plane = xmap.planeFitting(nearest_points);
//     }
//     // 输出耗时
//     double ts = startSearch.Toc();
//     std::cout << "XiVT KNN搜索耗时：" << ts << " 毫秒" << std::endl;
//     std::cout << "XiVT 五千次KNN搜索耗时：" << 1e4 * ts / counter / 2 << " 毫秒" << std::endl;
//   }
// }