#pragma once

#include <colmap/estimators/cost_functions.h>
#include <colmap/scene/database.h>

#include "core/xsfm_lib.h"

namespace xcolor {

void SaveXml(const std::string &filename, const std::unordered_map<colmap::image_t, colmap::Image> &images,
             const std::unordered_map<colmap::camera_t, colmap::Camera> &cameras, std::vector<MatchTrack> &match_tracks, double longitude,
             double latitude, const std::string &images_path);

}
