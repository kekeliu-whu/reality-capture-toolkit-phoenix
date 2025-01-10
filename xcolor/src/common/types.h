#pragma once

#include <colmap/scene/database.h>
#include <opencv2/opencv.hpp>

struct Image {
  std::string name;
  cv::Mat image;
  colmap::Rigid3d pose;
  colmap::Camera camera;
};
