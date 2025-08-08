#pragma once

#include "core/xsfm_lib.h"

void SaveTriangulatedPoints(const std::vector<xcolor::MatchTrack> &match_tracks, const std::string &filename);

void SaveImagePoses(const std::string &filename_txt, const std::string &filename_bin, const std::unordered_set<colmap::image_t> &optimized_image_ids,
                    const std::unordered_map<colmap::image_t, colmap::Image> &images,
                    const std::unordered_map<colmap::camera_t, colmap::Rigid3d> &pose_priors);

void SaveCameraParams(const std::string &filename_txt, const std::string &filename_bin,
                      const std::unordered_map<colmap::camera_t, colmap::Camera> &cameras);