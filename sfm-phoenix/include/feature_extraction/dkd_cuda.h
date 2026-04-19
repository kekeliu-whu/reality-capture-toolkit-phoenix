#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace feature_extraction {
namespace dkd {

/// Parameters for the DKD (Differentiable Keypoint Detector) post-processing.
struct DKDParams {
    int nms_radius = 2;          // NMS kernel = 2*radius+1 = 5
    int top_k = 5000;            // Max keypoints (TopK mode)
    float scores_th = 0.2f;      // Score threshold (used when top_k <= 0)
    int n_limit = 5000;          // Hard cap on keypoint count
    float temperature = 0.1f;    // Soft-argmax temperature
    int border = 2;              // Border exclusion radius
};

/// Pre-allocated workspace for DKD, eliminating per-frame cudaMalloc/Free.
///
/// Call init() once with the maximum expected image dimensions.
/// All temporary buffers used by NMS, TopK, and CUB sort are allocated
/// up front and reused across frames.
struct DKDWorkspace {
    // 4 general-purpose buffers, each >= max_HW * sizeof(float).
    // Reused across NMS (pooled, max_mask, supp_scores, tmp_pool)
    // and TopK (cand_scores, cand_indices, sorted_scores, sorted_indices).
    void* buf[4] = {};
    void* count_buf = nullptr;  // int[1] for candidate count
    void* cub_temp = nullptr;   // CUB radix sort scratch
    size_t cub_temp_bytes = 0;
    size_t buf_elem_bytes = 0;  // bytes per buf element (= max_HW * 4)
    bool ready = false;

    DKDWorkspace() = default;
    ~DKDWorkspace();
    DKDWorkspace(const DKDWorkspace&) = delete;
    DKDWorkspace& operator=(const DKDWorkspace&) = delete;

    /// Allocate all internal buffers for images up to max_H × max_W.
    void init(int max_H, int max_W);
};

/// Run NMS (non-maximum suppression) on the score map.
///
/// Uses pre-allocated buffers from workspace instead of per-call allocation.
void nms_cuda(const float* score_map, float* nms_out,
              int H, int W, int radius,
              DKDWorkspace& workspace, cudaStream_t stream);

/// TopK selection on flattened NMS scores.
///
/// Uses pre-allocated buffers from workspace instead of per-call allocation.
void topk_cuda(const float* nms_scores, int H, int W,
               int top_k, int border,
               float* kpts_xy, float* scores, int* count,
               DKDWorkspace& workspace, cudaStream_t stream);

/// Sub-pixel refinement via soft-argmax.
void subpixel_refine_cuda(const float* score_map, float* kpts_xy,
                          int N, int H, int W,
                          int radius, float temperature,
                          cudaStream_t stream);

/// Full DKD pipeline: NMS → TopK → SubPixel → output keypoints.
///
/// @param score_map   Device pointer, float [1, 1, H, W]
/// @param H, W        Dimensions
/// @param params      DKD parameters
/// @param kpts_xy     Output: device float [max_k, 2] — pixel coords
/// @param kpt_scores  Output: device float [max_k]
/// @param num_kpts    Output: device int [1] — actual keypoint count
/// @param nms_buf     Device buffer for NMS output (>= H*W*sizeof(float))
/// @param workspace   Pre-allocated DKD workspace
/// @param stream      CUDA stream
void detect_keypoints(const float* score_map, int H, int W,
                      const DKDParams& params,
                      float* kpts_xy, float* kpt_scores, int* num_kpts,
                      void* nms_buf, DKDWorkspace& workspace,
                      cudaStream_t stream);

/// Returns the required NMS output buffer size in bytes.
size_t nms_buf_size(int H, int W);

}  // namespace dkd
}  // namespace feature_extraction
