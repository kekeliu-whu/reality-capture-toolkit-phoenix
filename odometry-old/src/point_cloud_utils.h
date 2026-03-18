#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <deque>

typedef pcl::PointXYZINormal       PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

void DownsamplePoints(const PointCloudXYZI &sweep, PointCloudXYZI &sweep_downsampled, int target_size);
