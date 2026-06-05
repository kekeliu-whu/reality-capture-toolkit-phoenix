#pragma once

#include "sfm_phoenix/internal/trt_engine.h"

#include <vector>
#include <string>

namespace sfm_phoenix {

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
    int max_matches = 4000;
};

/// One pair's GPU feature data for batched inference.
struct BatchMatchInput {
    const void* d_kpts0;  ///< Device ptr: [N0, 2] float (from staging slot)
    const void* d_desc0;  ///< Device ptr: [N0, D] float
    int N0;
    const void* d_kpts1;  ///< Device ptr: [N1, 2] float
    const void* d_desc1;  ///< Device ptr: [N1, D] float
    int N1;
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

    /// Match a batch of pairs in ONE TRT forward pass.
    /// Pairs within the batch are zero-padded to the maximum N among them.
    /// Returns one MatchResult per input pair in the same order.
    /// Requires the engine to have been built with a dynamic batch profile
    /// (max_batch_size() > 1). Falls back to serial match_gpu when B == 1.
    std::vector<MatchResult> match_gpu_batch(
        const std::vector<BatchMatchInput>& inputs);

    /// Maximum keypoints per image supported by the loaded engine profile.
    /// Returns 0 if the engine has not been loaded yet.
    int max_keypoints() const;

    /// Maximum batch size supported by the loaded engine profile.
    /// Returns 1 for engines built without a batch dimension profile.
    int max_batch_size() const;

private:
    LightGlueConfig config_;
    TrtEngine engine_;
    cudaStream_t stream_ = nullptr;

    // Scratch buffers for match_gpu_batch: [B, N_max, *] packed on GPU.
    CudaBuffer batch_kpts0_;
    CudaBuffer batch_desc0_;
    CudaBuffer batch_kpts1_;
    CudaBuffer batch_desc1_;

    /// Internal: run inference + decode single-pair output.
    MatchResult run_and_decode(int N0, int N1);

    /// Internal: run inference + decode batch output (B pairs).
    /// Returns true on success, false on TRT failure (e.g. OOM).
    bool run_and_decode_batch(
        int B, const int* N0_arr, int N_max0,
        const int* N1_arr, int N_max1,
        std::vector<MatchResult>& out);
};

}  // namespace sfm_phoenix
