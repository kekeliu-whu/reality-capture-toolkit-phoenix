//
// Created by youyuan on 24-3-16.
//
#include "common_lib.h"

namespace lixel
{

PointCloudXYZINormal::Ptr transformPCL(const PointCloud::ConstPtr &undistort_pcl)
{
  PointCloudXYZINormal::Ptr transform_pcl(new PointCloudXYZINormal);
  transform_pcl->header = undistort_pcl->header;
  transform_pcl->width = undistort_pcl->width;
  transform_pcl->height = undistort_pcl->height;
  transform_pcl->sensor_origin_ = undistort_pcl->sensor_origin_;
  transform_pcl->sensor_orientation_ = undistort_pcl->sensor_orientation_;
  transform_pcl->reserve(undistort_pcl->size());
  for (const PointT &p : undistort_pcl->points)
  {
    PointXYZINormal p_transform;
    p_transform.x = p.x;
    p_transform.y = p.y;
    p_transform.z = p.z;
    p_transform.intensity = p.intensity;
    p_transform.curvature = static_cast<float>(p.timestamp - undistort_pcl->header.stamp * 1e-6);
    transform_pcl->push_back(p_transform);
  }
  return transform_pcl;
}

void transformToWorld(
    const PointCloudXYZINormal::ConstPtr &feats_map_body,
    const KFState::ConstPtr &state,
    PointCloudXYZINormal::Ptr &feats_map_world)
{
  Mat3 rot = state->sw_rot_.back();
  Vec3 pos = state->sw_pos_.back();
  feats_map_world->header = feats_map_body->header;
  feats_map_world->width = feats_map_body->width;
  feats_map_world->height = feats_map_body->height;
  feats_map_world->sensor_origin_ = feats_map_body->sensor_origin_;
  feats_map_world->sensor_orientation_ = feats_map_body->sensor_orientation_;
  feats_map_world->reserve(feats_map_body->size());
  for (const auto &p : feats_map_body->points)
  {
    Vec3 p_body(p.x, p.y, p.z);
    Vec3 p_world = rot * p_body + pos;
    PointXYZINormal p_transform;
    p_transform.x = p_world.x();
    p_transform.y = p_world.y();
    p_transform.z = p_world.z();
    p_transform.intensity = p.intensity;
    p_transform.normal_x = p.normal_x;
    p_transform.normal_y = p.normal_y;
    p_transform.normal_z = p.normal_z;
    feats_map_world->push_back(p_transform);
  }
}

}  // namespace lixel
