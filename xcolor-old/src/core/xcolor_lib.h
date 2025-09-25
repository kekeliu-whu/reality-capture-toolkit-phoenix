#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace xcolor {

void PerformXColor(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::vector<Image> &images, std::string output_path);

}  // namespace xcolor
