//
// Created by youyuan on 24-1-3.
//
#include "small_voxel_value.h"
#include "xmap_util.h"
namespace xmap {

SmallVoxelValue::SmallVoxelValue(const Configs& configs) {
  counter_ = 0;
  cloud_.reset(new PointCloud);
  // TODO: 查明reserve()导致的内存增长问题
  // cloud_->reserve(std::pow(configs.small_voxel_size / configs.resolution, 3));
  // index_map_.reserve(std::pow(configs.small_voxel_size / configs.resolution, 3));
  kd_tree_ = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>);
  is_dynamic_IO = false;
}

}  // namespace xmap
