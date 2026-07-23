// 用于管理和维护Xmap的公共方法，不涉及Xmap的算法实现
// 涉及Xmap中各个类型的转换，如三维点到VoxelLoc，PointType类型转V3F等等
// 设计文档：https://pecivkvtit.feishu.cn/docx/Vq1OdiUmjoq3D5xa9IbczkMhnsb

#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include "voxel_loc.h"
#include "xmap_types.hpp"

namespace xmap {
/**
 * 越过VoxelLoc创建，通过点和size生成对应体素哈希值
 * 若后续需要进一步压缩内存，改Key为IntDataType则可使用
 * 优点：取消了临时对象的创建，速度更快，可以节约内存
 * 缺点：代码可读性可能会下降
 * 注意：这里需要用floor或者ceil，不能用round否则会导致大小格不一致
 * @param point 三维点
 * @param voxel_size 体素大小
 * @return 体素的哈希值
 */
inline IntDataType hashValue(const V3F& point, FloatDataType voxel_size) {
  V3F loc_xyz_d = point / voxel_size;
  Eigen::Matrix<int64_t, 3, 1> loc_xyz_i;  // 注意：这里必须是int64_t，不然会溢出
  loc_xyz_i << static_cast<int64_t>(std::floor(loc_xyz_d(0))),
      static_cast<int64_t>(std::floor(loc_xyz_d(1))),
      static_cast<int64_t>(std::floor(loc_xyz_d(2)));
  int64_t hashValue =
      ((loc_xyz_i.z() * HASH_PRIME % MAX_HASH_NUM + loc_xyz_i.y()) * HASH_PRIME) % MAX_HASH_NUM +
      loc_xyz_i.x();
  return static_cast<IntDataType>(hashValue);
}

// 类型转换函数
inline VoxelLoc pos2VoxelLoc(const PointType& point, FloatDataType voxel_size) {
  VoxelLoc voxelLoc;
  voxelLoc.x_ = static_cast<IntDataType>(std::floor(point.x / voxel_size));
  voxelLoc.y_ = static_cast<IntDataType>(std::floor(point.y / voxel_size));
  voxelLoc.z_ = static_cast<IntDataType>(std::floor(point.z / voxel_size));
  return voxelLoc;
}

// 类型转换函数
inline VoxelLoc pos2VoxelLoc(const V3F& point, FloatDataType voxel_size) {
  VoxelLoc voxelLoc;
  voxelLoc.x_ = static_cast<IntDataType>(std::floor(point(0) / voxel_size));
  voxelLoc.y_ = static_cast<IntDataType>(std::floor(point(1) / voxel_size));
  voxelLoc.z_ = static_cast<IntDataType>(std::floor(point(2) / voxel_size));
  return voxelLoc;
}

// 类型转换函数
inline PointType V3F2PointType(const V3F& point) {
  PointType p;
  p.x = point.x();
  p.y = point.y();
  p.z = point.z();
  return p;
}

// 类型转换函数
inline V3F pointType2V3F(const PointType& point) {
  V3F p;
  p.x() = point.x;
  p.y() = point.y;
  p.z() = point.z;
  return p;
}

// 类型转换函数
inline V3F voxelLoc2V3F(const VoxelLoc& voxelLoc, float voxel_size) {
  V3F p;
  p.x() = voxelLoc.x_ * voxel_size;
  p.y() = voxelLoc.y_ * voxel_size;
  p.z() = voxelLoc.z_ * voxel_size;
  return p;
}

// 计算体素的中心位置
inline V3F calVoxelCenter(const VoxelLoc& voxel, FloatDataType voxel_size) {
  return V3F{
      static_cast<FloatDataType>(voxel.x_ * voxel_size + 0.5 * voxel_size),
      static_cast<FloatDataType>(voxel.y_ * voxel_size + 0.5 * voxel_size),
      static_cast<FloatDataType>(voxel.z_ * voxel_size + 0.5 * voxel_size)};
}

// 计算体素的中心位置
inline V3F calVoxelCenter(const V3F& point, FloatDataType voxel_size) {
  VoxelLoc voxelLoc = pos2VoxelLoc(point, voxel_size);
  return calVoxelCenter(voxelLoc, voxel_size);
}

/**
 * 将点云按二进制写入磁盘，动态加载时会频繁调用，所以inline
 * @param data vector保存的点云
 * @param filename 磁盘中点云的文件名
 */
inline void writePointCloudToFile(
    const std::vector<PointType, Eigen::aligned_allocator<PointType>>& data,
    const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (file.is_open()) {
    file.write(
        reinterpret_cast<const char*>(data.data()), data.size() * sizeof(pcl::PointXYZINormal));
    file.close();
  }
}

/**
 * 从磁盘读取文件中的点云，动态加载时会频繁调用，所以inline
 * @param filename 磁盘中点云的文件名
 * @return vector保存的点云
 */
inline std::vector<PointType, Eigen::aligned_allocator<PointType>> readPointCloudFromFile(
    const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  std::vector<PointType, Eigen::aligned_allocator<PointType>> data;

  if (file.is_open()) {
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::size_t numElements = fileSize / sizeof(pcl::PointXYZINormal);
    data.resize(numElements);

    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();
  }

  return data;
}

/**
 * PCA根据点序列拟合平面，匹配时会频繁调用，所以inline
 * @param points_vec 点序列
 * @return 平面对象的指针
 */
inline PlanePtr planeFitting(std::vector<V3F> points_vec, const V3F& view_point) {
  PlanePtr plane = std::make_shared<Plane>();
  /** 1. SVD特征值分解求解法向量 **/
  int numPoints = points_vec.size();
  Eigen::MatrixXd dataMatrix(numPoints, 3);
  for (int i = 0; i < numPoints; i++) {
    dataMatrix.row(i) = points_vec[i].transpose().cast<double>();
  }
  Eigen::Vector3d mean = dataMatrix.colwise().mean();
  dataMatrix.rowwise() -= mean.transpose();
  // 使用奇异值分解求解平面法向量
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(dataMatrix, Eigen::ComputeThinU | Eigen::ComputeFullV);
  Eigen::VectorXd D = svd.singularValues().transpose();

  /** 2. 是否满足平面条件 **/
  double planarity = sqrt(D(2) * D(2) / (D(1) * D(0)));
  plane->is_plane = planarity < PLANARITY_THRESHOLD;
  plane->normal = svd.matrixV().col(2).cast<FloatDataType>();
  plane->center = mean.cast<FloatDataType>();
  plane->points_size = numPoints;
  plane->planarity = static_cast<FloatDataType>(planarity);

  /** 3. 确保法向与视锥方向相反 **/
  V3F view_vec = view_point - mean.cast<FloatDataType>();
  view_vec.normalize();
  double cos_theta = plane->normal.dot(view_vec);
  if (cos_theta < 0) plane->normal = -plane->normal;
  plane->d = -plane->normal.transpose() * plane->center;
  return plane;
}

/**
 * 递归创建文件夹
 * @param path 路径
 */
void createFolders(const std::string& path);

/**
 * 删除文件夹内的所有文件
 * @param folderPath 路径
 */
void deleteFolderContents(const std::string& folderPath);

// c构建字符串工具
std::string strprintf(const char* fmt, ...);

class TicToc {
 public:
  TicToc() { Tic(); }

  void Tic() { start_ = std::chrono::system_clock::now(); }

  double Toc() {
    end_ = std::chrono::system_clock::now();
    elapsed_seconds_ = end_ - start_;
    return elapsed_seconds_.count() * 1000;
  }

  double GetLastStop() { return elapsed_seconds_.count() * 1000; }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_, end_;
  std::chrono::duration<double> elapsed_seconds_{};
};

}  // namespace xmap