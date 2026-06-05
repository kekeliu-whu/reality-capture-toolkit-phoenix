#include "sfm_phoenix/bridges/colmap_bridge.h"
#include "sfm_phoenix/internal/runtime_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace sfm_phoenix {

colmap::FeatureKeypoints ToColmapKeypoints(const AlikedResult& result) {
  colmap::FeatureKeypoints keypoints(result.keypoints.size());
  const float inv_scale = result.scale > 0.0f ? 1.0f / result.scale : 1.0f;
  for (size_t index = 0; index < result.keypoints.size(); ++index) {
    keypoints[index] = colmap::FeatureKeypoint(
        result.keypoints[index].x * inv_scale,
        result.keypoints[index].y * inv_scale);
  }
  return keypoints;
}

colmap::FeatureDescriptors ToColmapDescriptors(const AlikedResult& result) {
  colmap::FeatureDescriptors descriptors;
  descriptors.resize(result.num_keypoints,
                     result.descriptor_dim * sizeof(float));
  if (!result.descriptors.empty()) {
    std::memcpy(descriptors.data(),
                result.descriptors.data(),
                result.descriptors.size() * sizeof(float));
  }
  return descriptors;
}

std::vector<float> ToMatcherKeypoints(const colmap::FeatureKeypoints& keypoints,
                                      const float scale) {
  std::vector<float> flattened(keypoints.size() * 2);
  for (size_t index = 0; index < keypoints.size(); ++index) {
    flattened[index * 2 + 0] = keypoints[index].x * scale;
    flattened[index * 2 + 1] = keypoints[index].y * scale;
  }
  return flattened;
}

std::vector<float> ToMatcherDescriptors(
    const colmap::FeatureDescriptors& descriptors) {
  const int descriptor_dim = DescriptorDim(descriptors);
  std::vector<float> flattened(static_cast<size_t>(descriptors.rows()) *
                               descriptor_dim);
  if (!flattened.empty()) {
    std::memcpy(flattened.data(), descriptors.data(),
                flattened.size() * sizeof(float));
  }
  return flattened;
}

int DescriptorDim(const colmap::FeatureDescriptors& descriptors) {
  const int64_t cols = descriptors.cols();
  if (cols % static_cast<int64_t>(sizeof(float)) != 0) {
    throw std::runtime_error(
        "Descriptor byte width is not divisible by sizeof(float)");
  }
  return internal::CheckedIntCast(
      cols / static_cast<int64_t>(sizeof(float)), "Descriptor dimension");
}

colmap::FeatureMatches ToColmapMatches(const MatchResult& result) {
  colmap::FeatureMatches matches;
  matches.reserve(result.matches.size());
  for (const auto& [index0, index1] : result.matches) {
    colmap::FeatureMatch match;
    match.point2D_idx1 = static_cast<colmap::point2D_t>(index0);
    match.point2D_idx2 = static_cast<colmap::point2D_t>(index1);
    matches.push_back(match);
  }
  return matches;
}

float ComputeExtractionScale(const colmap::Camera& camera, const int max_edge) {
  if (max_edge <= 0) {
    return 1.0f;
  }
  const size_t longest_edge = std::max(camera.width, camera.height);
  if (longest_edge == 0) {
    return 1.0f;
  }
  return std::min(1.0f,
                  static_cast<float>(max_edge) /
                      static_cast<float>(longest_edge));
}

}  // namespace sfm_phoenix