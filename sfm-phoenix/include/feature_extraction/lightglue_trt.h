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

    /// Match two sets of features (host pointers, includes H2D copy).
    MatchResult match(const float* kpts0, const float* desc0, int N0,
                      const float* kpts1, const float* desc1, int N1);

    /// Match two sets of features already on GPU (no H2D copy).
    /// All device pointers must remain valid until this call returns.
    MatchResult match_gpu(const void* d_kpts0, const void* d_desc0, int N0,
                          const void* d_kpts1, const void* d_desc1, int N1);

private:
    LightGlueConfig config_;
    TrtEngine engine_;
    cudaStream_t stream_ = nullptr;

    /// Internal: run inference + decode output. Assumes inputs are set.
    MatchResult run_and_decode(int N0, int N1);
};

}  // namespace feature_extraction
