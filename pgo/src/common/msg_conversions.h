#pragma once

#include "proto/sensors.pb.h"
#include "types.h"

inline PointCloud::Ptr FromProto(const UndistoredLidarMsg &msg) {
  PointCloud::Ptr cloud{new PointCloud};
  cloud->reserve(msg.points_size());

  for (const auto &p : msg.points()) {
    PointType new_point;
    new_point.x = p.x();
    new_point.y = p.y();
    new_point.z = p.z();
    new_point.intensity = p.intensity();
    cloud->push_back(new_point);
  }

  return cloud;
}

inline Sophus::SE3d FromProto(const PoseMsg &pose_msg) {
  Sophus::SE3d pose;
  pose.translation().x() = pose_msg.tx();
  pose.translation().y() = pose_msg.ty();
  pose.translation().z() = pose_msg.tz();
  pose.setQuaternion(Eigen::Quaterniond(pose_msg.rw(), pose_msg.rx(),
                                        pose_msg.ry(), pose_msg.rz()));
  return pose;
}
