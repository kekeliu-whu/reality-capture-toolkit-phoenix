#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.hpp>

using PointType = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointType>;

struct TimestampedPose {
  double timestamp;
  Sophus::SE3d pose;
};

struct TimestampedPointCloud {
  double timestamp;
  Sophus::SE3d pose;
  PointCloud::Ptr cloud;

  TimestampedPointCloud() : cloud(new PointCloud()) {}
};
