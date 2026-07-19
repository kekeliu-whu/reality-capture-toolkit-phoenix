//
// Created by youyuan on 23-12-5.
//
#include <gtest/gtest.h>

#include "xmap_types.hpp"
#include "xmap_util.h"
using namespace xmap;
using namespace std;

inline IntDataType hashValueOld(const V3F& point, double voxel_size) {
  V3F loc_xyz_d = point / voxel_size;
  Eigen::Matrix<int64_t, 3, 1> loc_xyz_i;  // 注意：这里必须是int64_t，不然会溢出
  loc_xyz_i << static_cast<int64_t>(std::round(loc_xyz_d(0))),
      static_cast<int64_t>(std::round(loc_xyz_d(1))),
      static_cast<int64_t>(std::round(loc_xyz_d(2)));
  return size_t(
      ((loc_xyz_i[0]) * 73856093) ^ ((loc_xyz_i[1]) * 471943) ^ ((loc_xyz_i[2]) * 83492791));
}

inline IntDataType hashValueIVOX(const V3F& point, double voxel_size) {
  V3F loc_xyz_d = point / voxel_size;
  Eigen::Matrix<int32_t, 3, 1> loc_xyz_i;  // 注意：这里必须是int64_t，不然会溢出
  loc_xyz_i << static_cast<int32_t>(std::round(loc_xyz_d(0))),
      static_cast<int32_t>(std::round(loc_xyz_d(1))),
      static_cast<int32_t>(std::round(loc_xyz_d(2)));
  return size_t(
      (loc_xyz_i[0] * 73856093) ^ (loc_xyz_i[1] * 471943) ^ (loc_xyz_i[2] * 83492791) % 10000000);
}

inline IntDataType hashValueVoxelMap(const V3F& point, double voxel_size) {
  V3F loc_xyz_d = point / voxel_size;
  Eigen::Matrix<int64_t, 3, 1> loc_xyz_i;  // 注意：这里必须是int64_t，不然会溢出
  loc_xyz_i << static_cast<int64_t>(std::round(loc_xyz_d(0))),
      static_cast<int64_t>(std::round(loc_xyz_d(1))),
      static_cast<int64_t>(std::round(loc_xyz_d(2)));
  return ((loc_xyz_i.z() * HASH_PRIME % MAX_HASH_NUM + loc_xyz_i.y()) * HASH_PRIME) % MAX_HASH_NUM +
         loc_xyz_i.x();
}

inline IntDataType hashValueVoxelMap2(const V3F& point, double voxel_size) {
  V3F loc_xyz_d = point / voxel_size;
  Eigen::Matrix<int64_t, 3, 1> loc_xyz_i;  // 注意：这里必须是int64_t，不然会溢出
  loc_xyz_i << static_cast<int64_t>(std::round(loc_xyz_d(0))),
      static_cast<int64_t>(std::round(loc_xyz_d(1))),
      static_cast<int64_t>(std::round(loc_xyz_d(2)));
  return ((loc_xyz_i.z() * HASH_PRIME + loc_xyz_i.y()) * HASH_PRIME) + loc_xyz_i.x();
}

/*** 测试0.1m的小格子在(0.2*point_size)^3空间下的哈希冲突率，正负取值范围都会覆盖 ***/
TEST(VoxelLocTest, HashConflicting) {
  IntDataType point_size = 100;
  double len = 0.1;
  int max_depth = 1;
  std::unordered_map<IntDataType, int> map;
  int counter = 0;
  for (int i = -point_size; i < point_size; ++i) {
    for (int j = -point_size; j < point_size; ++j) {
      for (int k = -point_size; k < point_size; ++k) {
        V3F point(0.05 + i * len, 0.05 + j * len, 0.05 + k * len);
        IntDataType key = hashValue(point, len);
        auto it = map.find(key);
        if (it != map.end()) {
          counter++;
          it->second++;
          if (it->second > max_depth) {
            max_depth = it->second;
          }
        } else {
          map[key] = 1;
        }
      }
    }
  }
  std::cout << "counter:" << counter << endl;
  std::cout << "rate:" << double(counter) / (point_size * point_size * point_size)
            << ", max_length:" << max_depth << std::endl;
}

/*** 测试求余是否会导致额外的开销，结论是不会 ***/
/*** 而且overflow在不同平台可能会有未定义行为，最好还是别用hashValueVoxelMap2 ***/
TEST(OverFlowTest, HashValue) {
  int32_t a = 0;
  int64_t b = (int64_t(1) << 32) + 1;
  cout << "int32_t b:" << int32_t(b) << std::endl;

  V3F point(0, 0, 0);
  double len = 1;
  auto start = std::chrono::high_resolution_clock::now();  // 记录开始时间
  for (int m = 0; m < (int)1e12; ++m) {
    IntDataType key = hashValueVoxelMap(point, len);
  }
  auto end = std::chrono::high_resolution_clock::now();  // 记录结束时间
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);  // 计算耗时
  std::cout << "Time taken by code: " << duration.count() << " milliseconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();  // 记录开始时间
  for (int m = 0; m < (int)1e12; ++m) {
    IntDataType key = hashValueVoxelMap2(point, len);
  }
  end = std::chrono::high_resolution_clock::now();  // 记录结束时间
  duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);  // 计算耗时
  std::cout << "Time taken by code: " << duration.count() << " milliseconds" << std::endl;
}
