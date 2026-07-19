//
// Created by youyuan on 24-1-3.
//

#ifndef SRC_VOXELLOC_H
#define SRC_VOXELLOC_H

#include "xmap_types.hpp"

namespace xmap {
class VoxelLoc {
 public:
  IntDataType x_, y_, z_;

  VoxelLoc();
  VoxelLoc(IntDataType vx, IntDataType vy, IntDataType vz);

  /**
   * 利用世界系的点云和体素size，换算成3维int的VoxelID
   * @param point 三维世界中的点
   * @param voxel_size 格子大小
   */
  VoxelLoc(const V3F& point, FloatDataType voxel_size);

  bool operator==(const VoxelLoc& other) const;

  bool operator!=(const VoxelLoc& other) const;

  VoxelLoc operator-(const VoxelLoc& other) const;

  friend std::ostream& operator<<(std::ostream& os, const VoxelLoc& obj);

  /*** 打印VoxelLoc使用 ***/
  std::string toString() const;
};
}  // namespace xmap

namespace std {
template <>
struct hash<xmap::VoxelLoc> {
  int32_t operator()(const xmap::VoxelLoc& s) const {
    return ((((s.z_) * xmap::HASH_PRIME) % xmap::MAX_HASH_NUM + (s.y_)) * xmap::HASH_PRIME) %
               xmap::MAX_HASH_NUM +
           (s.x_);
  }
};
}  // namespace std
#endif  // SRC_VOXELLOC_H
