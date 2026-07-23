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
// #include "xivt/xivt.hpp"
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

/*** xivt与xmap的搜索精度验证 ***/
TEST(SearchAccTest, XmapSearch) {}

// TEST(SearchAccTest, XmapSearch) {
//   std::string path = "/home/youyuan/lixel-algorithm-xmap/src/lixel_lio/config/map.yaml";
//   Xmap xmap(path);
//   PointCloudPtr pointCloudPtr(new PointCloud);
//   PointCloudPtr filteredCloud(new PointCloud);
//   PointCloudPtr octoCloud(new PointCloud);
//
//   XiVT::Options xivt_options;
//   xivt_options.m_inv_resolution = 10;
//   xivt_options.m_downsample_resolution = 0.1;
//   XiVT xiVt(xivt_options);
//   pcl::octree::OctreePointCloudSearch<PointType>::Ptr octo_tree(
//       new pcl::octree::OctreePointCloudSearch<PointType>(0.1));
//   octo_tree->defineBoundingBox(-1000, -1000, -1000, 1000, 1000, 1000);
//   octo_tree->setInputCloud(octoCloud);
//   {
//     liblas::ReaderFactory reader_factory;
//     std::ifstream ifs;
//     ifs.open(filename_las, std::ios::in);
//     liblas::Reader reader = reader_factory.CreateWithStream(ifs);
//     Point_vector p_v;
//     std::vector<XiVT_point_status> status_to_add;
//
//     // 读取点云数据
//     while (reader.ReadNextPoint()) {
//       liblas::Point const& p = reader.GetPoint();
//       PointType pointType;
//       pointType.x = p.GetX() - reader.GetHeader().GetOffsetX();
//       pointType.y = p.GetY() - reader.GetHeader().GetOffsetY();
//       pointType.z = p.GetZ() - reader.GetHeader().GetOffsetZ();
//       pointCloudPtr->push_back(pointType);
//     }
//
//     // 创建体素滤波器对象
//     pcl::VoxelGrid<PointType> sor;
//     sor.setInputCloud(pointCloudPtr);
//     sor.setLeafSize(0.2f, 0.2f, 0.2f);  // 设置体素大小
//     sor.filter(*filteredCloud);
//
//     for (auto p : filteredCloud->points) {
//       XiVT_point_status xiVtPointStatus;
//       p_v.push_back(p);
//       status_to_add.push_back(xiVtPointStatus);
//       octo_tree->addPointToCloud(p, octoCloud);
//     }
//     filteredCloud->header.stamp = uint64_t(0);
//
//     std::cout << "size:" << filteredCloud->size() << std::endl;
//     std::vector<PointType, Eigen::aligned_allocator<PointType>> pv;
//     {
//       xgrids::TicToc startMapping;
//       xmap.mapIncremental(filteredCloud, V3F(0, 0, 100));
//       std::cout << "Xmap startMapping time cost:" << startMapping.Toc() << "ms" << std::endl;
//       std::cout << xmap.pointSize() << std::endl;
//
//       xgrids::TicToc startMapping1;
//       xiVt.add_points(p_v, 0, status_to_add, true);
//       std::cout << "Xivt startMapping time cost:" << startMapping.Toc() << "ms" << std::endl;
//       std::cout << xiVt.NumPoints() << std::endl;
//     }
//   }
//
//   // 精度验证，召回率，法向误差，质心误差
//   {
//     V3F view_point(0, 0, 100);
//
//     std::vector<V3F> recall_vec;
//     std::vector<V3F> err_normal_vec;
//     std::vector<V3F> err_center_vec;
//     std::vector<V3F> err_dist_vec;
//
//     for (auto& p : filteredCloud->points) {
//       std::vector<V3F> nearest_points_radius, nearest_points_knn, nearest_points_xivt,
//           nearest_points_octotree;
//       V3F search_point(p.x, p.y, p.z);
//
//       xmap.radiusSearch(search_point, view_point, nearest_points_radius, 1);
//       xmap.knnSearch(search_point, view_point, nearest_points_knn, 1);
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
//       for (int j = 0; j < m_nearest_points_vectors.size(); ++j) {
//         V3F p_temp(
//             m_nearest_points_vectors[j].x,
//             m_nearest_points_vectors[j].y,
//             m_nearest_points_vectors[j].z);
//         nearest_points_xivt.push_back(p_temp);
//       }
//       std::vector<int> pointIdxSearch;          // 存储k近邻搜索点的索引结果
//       std::vector<float> pointSquaredDistance;  // 存储k近邻搜索的平方距离
//
//                                                 //      std::cout << "4" << std::endl;
//       octo_tree->nearestKSearch(
//           V3F2PointType(search_point), 5, pointIdxSearch, pointSquaredDistance);
//       for (int j = 0; j < pointIdxSearch.size(); ++j) {
//         PointType nearest_j_point = filteredCloud->points[pointIdxSearch[j]];
//         V3F p_temp(nearest_j_point.x, nearest_j_point.y, nearest_j_point.z);
//         nearest_points_octotree.push_back(p_temp);
//       }
//
//       if (nearest_points_knn.size() < 5 || nearest_points_radius.size() < 5 ||
//           nearest_points_xivt.size() < 5)
//         continue;
//
//       float counter_radius = 0, counter_knn = 0, counter_xivt = 0;
//       // 召回率统计
//       for (int j = 0; j < 5; ++j) {
//         if (isElementInVector(nearest_points_octotree, nearest_points_xivt[j])) counter_xivt++;
//         if (isElementInVector(nearest_points_octotree, nearest_points_radius[j]))
//         counter_radius++; if (isElementInVector(nearest_points_octotree, nearest_points_knn[j]))
//         counter_knn++;
//       }
//       V3F recall(counter_xivt / 5.0, counter_knn / 5.0, counter_radius / 5.0);
//       recall_vec.push_back(recall);
//
//       // 法向误差统计
//       PlanePtr plane_radius = planeFitting(nearest_points_radius);
//       PlanePtr plane_knn = planeFitting(nearest_points_knn);
//       PlanePtr plane_xivt = planeFitting(nearest_points_xivt);
//       PlanePtr plane_true = planeFitting(nearest_points_octotree);
//       if (plane_xivt->normal.dot(plane_true->normal) < 0) {
//         plane_xivt->normal = -plane_xivt->normal;
//       }
//       if (plane_knn->normal.dot(plane_true->normal) < 0) {
//         plane_knn->normal = -plane_knn->normal;
//       }
//       if (plane_radius->normal.dot(plane_true->normal) < 0) {
//         plane_radius->normal = -plane_radius->normal;
//       }
//
//       if (!plane_true->is_plane) continue;
//
//       V3F normal_diff_radius, normal_diff_knn, normal_diff_xivt;
//
//       normal_diff_knn = plane_knn->normal - plane_true->normal;
//       normal_diff_radius = plane_radius->normal - plane_true->normal;
//       normal_diff_xivt = plane_xivt->normal - plane_true->normal;
//
//       V3F error_norm(normal_diff_xivt.norm(), normal_diff_knn.norm(), normal_diff_radius.norm());
//       err_normal_vec.push_back(error_norm);
//
//       // 质心误差统计
//       V3F center_diff_radius, center_diff_knn, center_diff_xivt;
//       center_diff_knn = plane_knn->center - plane_true->center;
//       center_diff_radius = plane_radius->center - plane_true->center;
//       center_diff_xivt = plane_xivt->center - plane_true->center;
//       V3F error_center(center_diff_xivt.norm(), center_diff_knn.norm(),
//       center_diff_radius.norm()); err_center_vec.push_back(error_center);
//
//       // 点面距离误差统计
//       float dist_true = distCalculate(plane_true, search_point);
//       float dist_radius = distCalculate(plane_radius, search_point);
//       float dist_knn = distCalculate(plane_knn, search_point);
//       float dist_xivt = distCalculate(plane_xivt, search_point);
//       V3F dist_diff(
//           abs(dist_xivt - dist_true), abs(dist_knn - dist_true), abs(dist_radius - dist_true));
//       err_dist_vec.push_back(dist_diff);
//
//       std::cout << "plane_true:" << plane_true->center.transpose() << "***"
//                 << plane_true->normal.transpose() << std::endl;
//       std::cout << "plane_xivt:" << plane_xivt->center.transpose() << "***"
//                 << plane_xivt->normal.transpose() << "***" << dist_xivt << "***" << error_norm(0)
//                 << "," << error_center(0) << "," << dist_diff(0) << std::endl;
//       std::cout << "plane_knn:" << plane_knn->center.transpose() << "***"
//                 << plane_knn->normal.transpose() << "***" << dist_knn << "***" << error_norm(1)
//                 << "," << error_center(1) << "," << dist_diff(1) << std::endl;
//       std::cout << "plane_radius:" << plane_radius->center.transpose() << "***"
//                 << plane_radius->normal.transpose() << "***" << dist_radius << "***"
//                 << error_norm(2) << "," << error_center(2) << "," << dist_diff(2) << std::endl;
//
//       std::cout << "--------------------------------------------------" << std::endl;
//     }
//
//     // 将数据写入 CSV 文件
//     writeVectorToCSV(path_save + "recall.csv", recall_vec);
//     writeVectorToCSV(path_save + "err_normal.csv", err_normal_vec);
//     writeVectorToCSV(path_save + "err_center.csv", err_center_vec);
//     writeVectorToCSV(path_save + "err_dist.csv", err_dist_vec);
//
//     std::cout << "recall xivt, knn, raiuds:" << calculateMean(recall_vec).transpose() <<
//     std::endl; std::cout << "normal xivt, knn, raiuds:" <<
//     calculateMean(err_normal_vec).transpose()
//               << std::endl;
//     std::cout << "center xivt, knn, raiuds:" << calculateMean(err_center_vec).transpose()
//               << std::endl;
//     std::cout << "dist xivt, knn, raiuds:" << calculateMean(err_dist_vec).transpose() <<
//     std::endl;
//   }
// }
