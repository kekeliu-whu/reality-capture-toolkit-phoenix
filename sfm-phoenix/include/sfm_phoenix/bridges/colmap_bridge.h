#pragma once

#include "sfm_phoenix/extractors/aliked.h"
#include "sfm_phoenix/matchers/lightglue.h"

#include <colmap/feature/types.h>
#include <colmap/scene/camera.h>

#include <vector>

namespace sfm_phoenix {

colmap::FeatureKeypoints ToColmapKeypoints(const AlikedResult& result);

colmap::FeatureDescriptors ToColmapDescriptors(const AlikedResult& result);

std::vector<float> ToMatcherKeypoints(const colmap::FeatureKeypoints& keypoints,
                                      float scale);

std::vector<float> ToMatcherDescriptors(
    const colmap::FeatureDescriptors& descriptors);

int DescriptorDim(const colmap::FeatureDescriptors& descriptors);

colmap::FeatureMatches ToColmapMatches(const MatchResult& result);

float ComputeExtractionScale(const colmap::Camera& camera, int max_edge);

}  // namespace sfm_phoenix