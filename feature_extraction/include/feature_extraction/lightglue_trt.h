#pragma once

#include "feature_extraction/trt_engine.h"

#include <vector>
#include <string>

namespace feature_extraction {

/// Result of LightGlue matching between two keypoint sets.
struct MatchResult {
    std::vector<std::pair<int, int>> matches;  // (idx0, idx1) pairs
    std::vector<float> scores;                 // Match confidence
    int num_matches = 0;
};

/// Configuration for LightGlue TensorRT matcher.
struct LightGlueConfig {
    std::string engine_path;   // Path to lightglue.engine
    int descriptor_dim = 128;
};

/// LightGlue feature matcher using TensorRT.
///
/// Input:  two sets of (keypoints, descriptors) from ALIKED
/// Output: matched keypoint index pairs with confidence scores
class LightGlueMatcher {
public:
    LightGlueMatcher() = default;
    ~LightGlueMatcher();

    /// Initialise with engine path.
    bool init(const LightGlueConfig& config);

    /// Match two sets of features.
    ///
    /// @param kpts0   Keypoints of image 0 [N, 2] in pixel coords
    /// @param desc0   Descriptors of image 0 [N, D] row-major
    /// @param N0      Number of keypoints in image 0
    /// @param kpts1   Keypoints of image 1 [M, 2] in pixel coords
    /// @param desc1   Descriptors of image 1 [M, D] row-major
    /// @param N1      Number of keypoints in image 1
    MatchResult match(const float* kpts0, const float* desc0, int N0,
                      const float* kpts1, const float* desc1, int N1);

private:
    LightGlueConfig config_;
    TrtEngine engine_;
    cudaStream_t stream_ = nullptr;
};

}  // namespace feature_extraction
