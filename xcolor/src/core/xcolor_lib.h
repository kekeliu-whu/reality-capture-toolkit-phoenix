#pragma once

#include <colmap/scene/database.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>

void PerformXColor(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::vector<Image> &images, std::string output_path);
