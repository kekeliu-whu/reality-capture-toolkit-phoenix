//
// Created by youyuan on 23-12-5.
//
#include <gtest/gtest.h>
#include <malloc.h>
#include <sys/resource.h>
#include <random>

#include "liblas/liblas.hpp"
// #include "xivt/xivt.hpp"
#include "mem_monitor.h"
#include "xmap.h"
#include "xmap_test.h"
#include "xmap_util.h"

using namespace xmap;
using namespace std;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, 0.01);

std::string path = "/home/youyuan/lixel-algorithm-l2/src/lixel_lio/xmap/configs/map.yaml";
float proc_gb;
XmapTest xmap_test;

/*** 法向拟合时，线状点云应该拟合失败 ***/
TEST(NormalFilterTest, PlaneFittingFailed) {
  Xmap xmap(path);
  Configs configs = xmap.getConfigs();
  configs.enable_normal_filter = true;
  xmap_test.setConfigs(xmap, configs);

  int l = 1000;
  PointCloudPtr pointCloudPtr(new PointCloud);
  for (int i = 0; i < l; ++i) {
    PointType pointType;
    pointType.x = 0.05;
    pointType.y = 0.05 + i * 0.2;
    pointType.z = 0.05;
    // 生成随机噪声并添加到对应的点坐标上
    pointCloudPtr->push_back(pointType);
  }
  xmap.mapIncremental(pointCloudPtr, V3F(-1, 0, 0.05));
  for (auto& entry : xmap_test.getSmallVoxelMap(xmap)) {
    auto& cloud = entry.second.cloud_;
    for (auto& p : cloud->points) {
      assert(p.normal_x == 0 && p.normal_y == 0 && p.normal_z == 0);
    }
  }
}

/*** 法向拟合时，面状点云应该拟合成功 ***/
TEST(NormalFilterTest, PlaneFittingSuccess) {
  Xmap xmap(path);
  Configs configs = xmap.getConfigs();
  configs.enable_normal_filter = true;
  xmap_test.setConfigs(xmap, configs);

  int l = 1000;
  PointCloudPtr pointCloudPtr(new PointCloud);
  for (int i = 0; i < l; ++i) {
    for (int j = 0; j < l; ++j) {
      PointType pointType;
      pointType.x = 0.05;
      pointType.y = 0.05 + i * 0.2;
      pointType.z = 0.05 + j * 0.2;
      // std::cout << "p:" << pointType.x << "," << pointType.y << "," << pointType.z <<
      // std::endl; 生成随机噪声并添加到对应的点坐标上
      pointCloudPtr->push_back(pointType);
    }
  }
  xmap.mapIncremental(pointCloudPtr, V3F(-1, 0, 0.05));
  for (auto& entry : xmap_test.getSmallVoxelMap(xmap)) {
    auto& cloud = entry.second.cloud_;
    for (auto& p : cloud->points) {
      assert(p.normal_x != 0 || p.normal_y != 0 || p.normal_z != 0);
    }
  }
}

TEST(NormalFilterTest, NormalFilterInSearch) {
  Xmap xmap(path);
  Configs configs = xmap.getConfigs();
  configs.enable_normal_filter = true;
  xmap_test.setConfigs(xmap, configs);

  // 构建左墙
  int l = 100;
  {
    PointCloudPtr pointCloudPtr(new PointCloud);
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < l; ++j) {
        PointType pointType;
        pointType.x = 0.05;
        pointType.y = 0.05 + i * 0.2;
        pointType.z = 0.05 + j * 0.2;
        // 生成随机噪声并添加到对应的点坐标上
        pointCloudPtr->push_back(pointType);
      }
    }
    xmap.mapIncremental(pointCloudPtr, V3F(-1, 0, 0.05));
  }
  for (auto& entry : xmap_test.getSmallVoxelMap(xmap)) {
    auto& cloud = entry.second.cloud_;
    for (auto& p : cloud->points) {
      assert(p.normal_x != 0 || p.normal_y != 0 || p.normal_z != 0);
    }
  }
  std::cout << "构建左墙完成" << std::endl;

  // 构建右墙
  {
    PointCloudPtr pointCloudPtr(new PointCloud);
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < l; ++j) {
        PointType pointType;
        pointType.x = 0.25;
        pointType.y = 0.05 + i * 0.2;
        pointType.z = 0.05 + j * 0.2;
        // 生成随机噪声并添加到对应的点坐标上
        pointCloudPtr->push_back(pointType);
      }
    }
    xmap.mapIncremental(pointCloudPtr, V3F(1, 0, 0.05));
  }

  std::cout << "构建右墙完成" << std::endl;
  xmap.reBuildKDTree();
  // 左墙搜索
  {
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < l; ++j) {
        PointType pointType;
        pointType.x = 0.05;
        pointType.y = 0.05 + i * 0.2;
        pointType.z = 0.05 + j * 0.2;
        std::vector<V3F> nearest_points;
        V3F search_point(pointType.x, pointType.y, pointType.z);
        std::cout << "search_point:" << search_point.transpose() << std::endl;
        xmap.knnSearch(search_point, V3F(-1, 0, 0.05), nearest_points, 1);
        for (auto& p : nearest_points) {
          std::cout << "p:" << p.transpose() << std::endl;
          ASSERT_EQ(p.x(), 0.05f);
        }
      }
    }
  }
  std::cout << "左墙搜索成功" << std::endl;

  // 右墙搜索
  {
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < l; ++j) {
        PointType pointType;
        pointType.x = 0.25;
        pointType.y = 0.05 + i * 0.2;
        pointType.z = 0.05 + j * 0.2;
        std::vector<V3F> nearest_points;
        V3F search_point(pointType.x, pointType.y, pointType.z);
        std::cout << "search_point:" << search_point.transpose() << std::endl;
        xmap.knnSearch(search_point, V3F(1, 0, 0.05), nearest_points, 1);
        for (auto& p : nearest_points) {
          std::cout << "p:" << p.transpose() << std::endl;
          ASSERT_EQ(p.x(), 0.25f);
        }
      }
    }
  }
  std::cout << "右墙搜索成功" << std::endl;
}