//
// Created by youyuan on 24-1-3.
//
#include "small_voxel_value.h"
#include "xmap_util.h"
namespace xmap {

SmallVoxelValue::SmallVoxelValue(const Configs& configs)
    : index_map_(
          static_cast<std::size_t>(configs.small_scale) *
              static_cast<std::size_t>(configs.small_scale) *
              static_cast<std::size_t>(configs.small_scale),
          IntDataType{-1}),
      kd_tree_(new pcl::KdTreeFLANN<PointType>),
      cloud_(new PointCloud) {}

}  // namespace xmap
