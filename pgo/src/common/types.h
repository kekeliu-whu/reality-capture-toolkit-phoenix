#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

using PointType  = pcl::PointXYZI;
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

struct GpsData {
  double timestamp = 0.0;
  double latitude  = 0.0;  // degrees
  double longitude = 0.0;  // degrees
  double altitude  = 0.0;  // meters
  double lat_std   = 0.0;  // latitude standard deviation (meters)
  double lon_std   = 0.0;  // longitude standard deviation (meters)
  double alt_std   = 0.0;  // altitude standard deviation (meters)
};
