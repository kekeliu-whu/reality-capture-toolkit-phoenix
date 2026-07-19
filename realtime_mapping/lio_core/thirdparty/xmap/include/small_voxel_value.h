//
// Created by youyuan on 24-1-3.
//

#ifndef SRC_SMALL_VOXEL_VALUE_H
#define SRC_SMALL_VOXEL_VALUE_H

#include <pcl/kdtree/kdtree_flann.h>
#include "voxel_loc.h"
#include "xmap_types.hpp"

namespace xmap {
class SmallVoxelValue {
 public:
  std::unordered_map<IntDataType, int> index_map_;
  pcl::KdTreeFLANN<PointType>::Ptr kd_tree_;
  PointCloudPtr cloud_;
  TimeMark time_mark_;
  int counter_;
  bool is_dynamic_IO;

  /**
   * SmallVoxelValue的初始化方法，会设置八叉树的bound，并初始化小格子的点云
   * @param configs Xmap的配置
   * @param voxel 本Value对应的voxel
   */
  SmallVoxelValue(const Configs& configs);
};
}  // namespace xmap

#endif  // SRC_SMALL_VOXEL_VALUE_H
