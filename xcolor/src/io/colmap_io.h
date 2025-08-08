#pragma once

#include <colmap/scene/point3d.h>

#include "common/types.h"

namespace xcolor {

void ReadCamerasBinary(const std::string& filename, std::vector<colmap::Camera>& cameras);

// msvc seems to have an internal bug in ifstream::read
// which causes an error with errno == 2 midway
//
// example:
//
// colmap::Reconstruction reconstruction;
// reconstruction.ReadBinary(FLAGS_sfm_result_path);
// auto images = reconstruction.Images();
// DLOG(INFO) << images.size();
//
// msvc seems to have an internal bug in ifstream::read
// which causes an error with errno == 2 midway
//
// example:
//
// colmap::Reconstruction reconstruction;
// reconstruction.ReadBinary(FLAGS_sfm_result_path);
// auto images = reconstruction.Images();
// DLOG(INFO) << images.size();
//
void ReadImagesBinary(const std::string& filename, std::vector<colmap::Image>& images);

void ReadImages(const std::string& sfm_path, const std::string& images_path, std::vector<Image>& images);

void WriteCamerasBinary(const std::string& filename, const std::vector<colmap::Camera>& cameras);

void WriteImagesBinary(const std::string& filename, const std::vector<colmap::Image>& images);

void WritePoints3DBinary(const std::string& filename, const std::vector<colmap::Point3D>& points3D);

}  // namespace xcolor
