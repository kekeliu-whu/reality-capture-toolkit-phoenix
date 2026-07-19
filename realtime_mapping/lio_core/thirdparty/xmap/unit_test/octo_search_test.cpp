//
// Created by youyuan on 23-12-5.
//
#include <gtest/gtest.h>
#include <malloc.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_search.h>
#include <sys/resource.h>
#include <liblas/liblas.hpp>
#include "mem_monitor.h"

#include "xmap.h"
#include "xmap_util.h"

using namespace xmap;
using namespace std;
std::string filename =
    "/home/youyuan/lixel-algorithm-l2/src/lixel_lio/xmap/unit_test/test_data/L2_downsample.las";
std::string path = "/home/youyuan/lixel-algorithm-l2/src/lixel_lio/xmap/configs/map.yaml";

float proc_gb;

template <typename T>
bool isElementInVector(const std::vector<T>& vec, const T& target) {
  auto it = std::find(vec.begin(), vec.end(), target);
  return (it != vec.end());
}

float distCalculate(const PlanePtr& plane, const V3F& p) {
  float distance = (p - plane->center).dot(plane->normal) / plane->normal.norm();
  return distance;
}

V3F calculateMean(const std::vector<V3F>& vec) {
  V3F mean = V3F::Zero();

  if (!vec.empty()) {
    for (const V3F& v : vec) {
      mean += v;
    }
    mean /= vec.size();
  }

  return mean;
}

// 写入向量数据到 CSV 文件
void writeVectorToCSV(const std::string& filename, const std::vector<V3F>& data) {
  std::ofstream file(filename);
  if (file.is_open()) {
    for (const auto& v : data) {
      file << v.x() << "," << v.y() << "," << v.z() << "\n";
    }
    file.close();
    std::cout << "Data has been written to " << filename << std::endl;
  } else {
    std::cerr << "Unable to open file " << filename << " for writing" << std::endl;
  }
}

// 计算点之间的欧氏距离
inline float distance(const V3F& p1, const V3F& p2) {
  return std::sqrt(
      (p1.x() - p2.x()) * (p1.x() - p2.x()) + (p1.y() - p2.y()) * (p1.y() - p2.y()) +
      (p1.z() - p2.z()) * (p1.z() - p2.z()));
}

std::string convert(const std::vector<int>& numbers) {
  std::stringstream ss;
  for (size_t i = 0; i < numbers.size(); ++i) {
    ss << numbers[i];
    if (i != numbers.size() - 1) {
      ss << ",";
    }
  }
  return ss.str();
}

/*** 树搜索的精度验证 ***/
TEST(OctoTreeTest, SearchAcc) {
  Xmap xmap(path);
  PointCloudPtr pointCloudPtr(new PointCloud);
  PointCloudPtr filteredCloud(new PointCloud);
  PointCloudPtr octoCloud(new PointCloud);

  pcl::octree::OctreePointCloudSearch<PointType>::Ptr octo_tree(
      new pcl::octree::OctreePointCloudSearch<PointType>(0.2));
  pcl::KdTreeFLANN<PointType>::Ptr kd_tree(new pcl::KdTreeFLANN<PointType>);
  octo_tree->defineBoundingBox(-1000, -1000, -1000, 1000, 1000, 1000);
  octo_tree->setInputCloud(octoCloud);

  liblas::ReaderFactory reader_factory;
  std::ifstream ifs;
  ifs.open(filename, std::ios::in);
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

  // 创建体素滤波器对象
  pcl::VoxelGrid<PointType> sor;
  sor.setInputCloud(pointCloudPtr);
  sor.setLeafSize(0.2f, 0.2f, 0.2f);  // 设置体素大小
  sor.filter(*filteredCloud);

  for (auto p : filteredCloud->points) {
    octo_tree->addPointToCloud(p, octoCloud);
  }
  kd_tree->setInputCloud(filteredCloud);

  // 八叉树的搜索精度验证
  //  for (auto& p : filteredCloud->points) {
  //    std::vector<V3F> nearest_points_radius, nearest_points_knn, nearest_points_xivt,
  //        nearest_points_octotree;
  //    std::vector<int> pointIdxSearch;          // 存储k近邻搜索点的索引结果
  //    std::vector<float> pointSquaredDistance;  // 存储k近邻搜索的平方距离
  //    V3F search_point(p.x, p.y, p.z);
  //
  //    octo_tree->nearestKSearch(V3F2PointType(search_point), 5, pointIdxSearch,
  //    pointSquaredDistance);
  //
  //    float minDistance = std::numeric_limits<float>::max();
  //    // 执行暴力搜索，找到最近的5个邻居
  //    int k = 5;
  //    std::vector<int> nearestNeighbors(k, -1);
  //    std::vector<float> distances(k, std::numeric_limits<float>::max());
  //
  //    for (size_t i = 0; i < filteredCloud->size(); ++i) {
  //      const PointType& point = filteredCloud->at(i);
  //      V3F v3fPoint = {point.x, point.y, point.z};
  //      float dist = distance(search_point, v3fPoint);
  //
  //      // 更新最近邻居列表
  //      for (int j = 0; j < k; ++j) {
  //        if (dist < distances[j]) {
  //          for (int l = k - 1; l > j; --l) {
  //            nearestNeighbors[l] = nearestNeighbors[l - 1];
  //            distances[l] = distances[l - 1];
  //          }
  //          nearestNeighbors[j] = i;
  //          distances[j] = dist;
  //          break;
  //        }
  //      }
  //    }
  //
  //    std::cout << "idx octo:" << convert(pointIdxSearch) << std::endl;
  //    std::cout << "idx bru :" << convert(nearestNeighbors) << std::endl;
  //    std::cout << "------------------------" << std::endl;
  //  }

  // KD树的搜索精度验证
  for (auto& p : filteredCloud->points) {
    std::vector<V3F> nearest_points_kd, nearest_points_octo;
    std::vector<int> pointIdxSearchOcto, pointIdxSearchKD;  // 存储k近邻搜索点的索引结果
    std::vector<float> pointSquaredDistanceOcto, pointSquaredDistanceKD;  // 存储k近邻搜索的平方距离
    V3F search_point(p.x, p.y, p.z);

    octo_tree->nearestKSearch(
        V3F2PointType(search_point), 5, pointIdxSearchOcto, pointSquaredDistanceOcto);
    kd_tree->nearestKSearch(
        V3F2PointType(search_point), 5, pointIdxSearchKD, pointSquaredDistanceKD);

    for (int j = 0; j < pointIdxSearchOcto.size(); ++j) {
      PointType nearest_j_point = filteredCloud->points[pointIdxSearchOcto[j]];
      V3F p_temp(nearest_j_point.x, nearest_j_point.y, nearest_j_point.z);
      nearest_points_octo.push_back(p_temp);
    }
    for (int j = 0; j < pointIdxSearchOcto.size(); ++j) {
      PointType nearest_j_point = filteredCloud->points[pointIdxSearchKD[j]];
      V3F p_temp(nearest_j_point.x, nearest_j_point.y, nearest_j_point.z);
      nearest_points_kd.push_back(p_temp);
    }
    if (nearest_points_kd.size() < 5 || nearest_points_octo.size() < 5) continue;

    PlanePtr plane_kd = planeFitting(nearest_points_kd);
    PlanePtr plane_octo = planeFitting(nearest_points_octo);

    // 法向，质心误差统计
    V3F normal_diff_kd, center_diff_kd;
    normal_diff_kd = plane_kd->normal - plane_octo->normal;
    center_diff_kd = plane_kd->center - plane_octo->center;

    // 点面距离误差统计
    float dist_octo = distCalculate(plane_octo, search_point);
    float dist_kd = distCalculate(plane_kd, search_point);
    float dist_diff = dist_octo - dist_kd;
    if (dist_diff != 0) {
      std::cout << "dist_diff:" << dist_diff << std::endl;
      std::cout << "--------------------------------------------------" << std::endl;
    }
  }
}
