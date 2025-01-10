#pragma once

#include "common/types.h"

void DownsampleSubmaps(std::vector<TimestampedPointCloud> &submaps,
                       double voxel_size);

void PcaEstimateNormal(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud,
                       const std::vector<Eigen::Vector3f> &centers,
                       int k,
                       double downsample_voxel_size,
                       std::vector<Eigen::Vector3f> &normals);

void DownsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud_in,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out,
                          double voxel_size);

void LoadFullPointCloud(const std::string &project_input_path,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                        std::vector<Eigen::Vector3f> &centers);
