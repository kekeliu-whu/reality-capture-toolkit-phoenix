//
// Created by youyuan on 24-1-3.
//

#ifndef SRC_SMALL_VOXEL_VALUE_H
#define SRC_SMALL_VOXEL_VALUE_H

#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include "voxel_loc.h"
#include "xmap_types.hpp"

namespace xmap {
class SmallVoxelValue {
 public:
  // Dense resolution-cell -> point index table. The production XMap uses a
  // fixed small_scale^3 table and -1 for an empty cell; this avoids hash
  // collisions and the per-node overhead of unordered_map.
  std::vector<IntDataType> index_map_;
  pcl::KdTreeFLANN<PointType>::Ptr kd_tree_;
  PointCloudPtr cloud_;
  TimeMark time_mark_;
  int counter_ = 0;  // accepted new/replaced cells since rebuild

  /**
   * SmallVoxelValue的初始化方法，会设置八叉树的bound，并初始化小格子的点云
   * @param configs Xmap的配置
   * @param voxel 本Value对应的voxel
   */
  SmallVoxelValue(const Configs& configs);
};
}  // namespace xmap

#endif  // SRC_SMALL_VOXEL_VALUE_H
