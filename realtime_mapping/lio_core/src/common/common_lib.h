//
// Created by youyuan on 24-3-16.
//
#pragma once

#include "common_struct.h"
#include "ieskf/ieskf.h"

namespace lixel
{
PointCloudXYZINormal::Ptr transformPCL(const PointCloud::ConstPtr &undistort_pcl);

void transformToWorld(
    const PointCloudXYZINormal::ConstPtr &feats_map_body,
    const KFState::ConstPtr &state,
    PointCloudXYZINormal::Ptr &feats_map_world);
}  // namespace lixel