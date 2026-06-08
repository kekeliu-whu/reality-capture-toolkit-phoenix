#pragma once

#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

using PointType  = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointType>;

struct TimestampedPose {
  double timestamp = 0.0;
  std::shared_ptr<Sophus::SE3d> pose;
  Eigen::Vector3d gravity;

  TimestampedPose() : pose(std::make_shared<Sophus::SE3d>()) {}
};

struct TimestampedPointCloud {
  double timestamp = 0.0;
  std::shared_ptr<Sophus::SE3d> pose;
  size_t anchor_pose_index = 0;
  PointCloud::Ptr cloud;

  TimestampedPointCloud() : cloud(new PointCloud()) {}
};

struct GpsData {
  // Basic GPS data
  double timestamp = 0.0;
  double latitude  = 0.0;  // degrees
  double longitude = 0.0;  // degrees
  double altitude  = 0.0;  // meters
  double lat_std   = 0.0;  // latitude standard deviation (meters)
  double lon_std   = 0.0;  // longitude standard deviation (meters)
  double alt_std   = 0.0;  // altitude standard deviation (meters)

  // RTK-specific data
  int fix_type             = 0;      // 0=None, 1=SPP, 2=RTK, 3=RTK-FIX
  int num_satellites       = 0;      // Number of satellites used
  double gdop              = 0.0;    // Geometric Dilution of Precision
  double hdop              = 0.0;    // Horizontal Dilution of Precision
  double vdop              = 0.0;    // Vertical Dilution of Precision
  double baseline_length   = 0.0;    // RTK baseline length (meters)
  bool has_moving_baseline = false;  // Whether using moving baseline RTK
  double heading           = 0.0;    // Heading/yaw angle from RTK (degrees)
  double pitch             = 0.0;    // Pitch angle from RTK (degrees)
  double roll              = 0.0;    // Roll angle from RTK (degrees)
  double heading_std       = 0.0;    // Heading standard deviation (degrees)
};
