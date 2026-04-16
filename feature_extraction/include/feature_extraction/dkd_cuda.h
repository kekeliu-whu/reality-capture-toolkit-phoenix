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

/// Run NMS (non-maximum suppression) on the score map.
///
/// Implements the iterative suppression from ALIKED's simple_nms():
///   1. max_pool2d → local max mask
///   2. 2 iterations of suppression & re-detection
///
/// @param score_map  Device pointer, float [1, 1, H, W]
/// @param nms_out    Device pointer, float [1, 1, H, W] — output
/// @param H, W       Score map dimensions
/// @param radius     NMS radius (kernel = 2*radius + 1)
/// @param stream     CUDA stream
void nms_cuda(const float* score_map, float* nms_out,
              int H, int W, int radius, cudaStream_t stream);

/// TopK selection on flattened NMS scores.
///
/// Selects the top K scoring positions and returns their pixel coordinates.
///
/// @param nms_scores  Device pointer, float [H * W]
/// @param H, W        Score map dimensions
/// @param top_k       Number of keypoints to select
/// @param border      Exclude keypoints within this many pixels of the border
/// @param kpts_xy     Output device pointer, float [top_k, 2] — pixel (x, y)
/// @param scores      Output device pointer, float [top_k] — scores
/// @param count       Output device pointer, int [1] — actual count (≤ top_k)
/// @param stream      CUDA stream
void topk_cuda(const float* nms_scores, int H, int W,
               int top_k, int border,
               float* kpts_xy, float* scores, int* count,
               cudaStream_t stream);

/// Sub-pixel refinement via soft-argmax.
///
/// For each keypoint, unfolds a (2*radius+1)² patch from the score map and
/// computes the temperature-scaled soft-argmax offset.
///
/// @param score_map   Device pointer, float [H * W]
/// @param kpts_xy     Device pointer, float [N, 2] — integer pixel coords, in-place refined
/// @param N           Number of keypoints
/// @param H, W        Score map dimensions
/// @param radius      Kernel radius for soft-argmax
/// @param temperature Temperature for softmax
/// @param stream      CUDA stream
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
/// @param workspace   Device buffer for intermediate results (must be pre-allocated)
/// @param workspace_bytes  Size of workspace buffer
/// @param stream      CUDA stream
void detect_keypoints(const float* score_map, int H, int W,
                      const DKDParams& params,
                      float* kpts_xy, float* kpt_scores, int* num_kpts,
                      void* workspace, size_t workspace_bytes,
                      cudaStream_t stream);

/// Returns the required workspace size in bytes for detect_keypoints().
size_t workspace_size(int H, int W);

}  // namespace dkd
}  // namespace feature_extraction
