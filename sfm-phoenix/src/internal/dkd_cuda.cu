#include "sfm_phoenix/internal/dkd_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cmath>
#include <algorithm>

namespace sfm_phoenix {
namespace dkd {

// --------------------------------------------------------------------------
// DKDWorkspace — pre-allocated buffers
// --------------------------------------------------------------------------

DKDWorkspace::~DKDWorkspace() {
    for (int i = 0; i < 4; ++i) {
        if (buf[i]) cudaFree(buf[i]);
    }
    if (count_buf) cudaFree(count_buf);
    if (cub_temp) cudaFree(cub_temp);
}

void DKDWorkspace::init(int max_H, int max_W) {
    size_t max_HW = static_cast<size_t>(max_H) * max_W;
    buf_elem_bytes = max_HW * sizeof(float);

    for (int i = 0; i < 4; ++i) {
        cudaMalloc(&buf[i], buf_elem_bytes);
    }
    cudaMalloc(&count_buf, sizeof(int));

    // Query CUB radix sort temporary storage for max element count
    cub_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, cub_temp_bytes,
        static_cast<float*>(nullptr), static_cast<float*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        static_cast<int>(max_HW), 0, sizeof(float) * 8,
        static_cast<cudaStream_t>(nullptr));
    if (cub_temp_bytes > 0) {
        cudaMalloc(&cub_temp, cub_temp_bytes);
    }
    ready = true;
}

// --------------------------------------------------------------------------
// NMS kernel: max_pool2d-based non-maximum suppression
// --------------------------------------------------------------------------

__global__ void max_pool2d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int H, int W, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float max_val = -FLT_MAX;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int ny = y + dy;
            int nx = x + dx;
            if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                max_val = fmaxf(max_val, input[ny * W + nx]);
            }
        }
    }
    output[y * W + x] = max_val;
}

__global__ void local_max_mask_kernel(const float* __restrict__ scores,
                                      const float* __restrict__ pooled,
                                      float* __restrict__ output,
                                      int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HW) return;
    output[idx] = (scores[idx] == pooled[idx]) ? scores[idx] : 0.0f;
}

__global__ void nms_suppress_kernel(const float* __restrict__ scores,
                                    const float* __restrict__ max_mask_pooled,
                                    const float* __restrict__ supp_scores_pooled,
                                    float* __restrict__ max_mask,
                                    float* __restrict__ supp_scores,
                                    int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HW) return;

    bool is_max = max_mask[idx] > 0.0f;
    bool is_supp = max_mask_pooled[idx] > 0.0f;
    float ss = is_supp ? 0.0f : scores[idx];
    supp_scores[idx] = ss;

    bool new_max = (ss > 0.0f) && (ss == supp_scores_pooled[idx]);
    if (new_max && !is_supp) {
        max_mask[idx] = scores[idx];
    }
}

void nms_cuda(const float* score_map, float* nms_out,
              int H, int W, int radius,
              DKDWorkspace& ws, cudaStream_t stream) {
    int HW = H * W;
    dim3 block2d(16, 16);
    dim3 grid2d((W + 15) / 16, (H + 15) / 16);
    int block1d = 256;
    int grid1d = (HW + block1d - 1) / block1d;

    // Use pre-allocated workspace buffers
    float* pooled     = static_cast<float*>(ws.buf[0]);
    float* max_mask   = static_cast<float*>(ws.buf[1]);
    float* supp_scores = static_cast<float*>(ws.buf[2]);
    float* tmp_pool   = static_cast<float*>(ws.buf[3]);

    // Step 1: initial max_mask = (scores == max_pool(scores)) ? scores : 0
    max_pool2d_kernel<<<grid2d, block2d, 0, stream>>>(
        score_map, pooled, H, W, radius);
    local_max_mask_kernel<<<grid1d, block1d, 0, stream>>>(
        score_map, pooled, max_mask, HW);

    // Step 2-3: two iterations of suppression
    for (int iter = 0; iter < 2; ++iter) {
        max_pool2d_kernel<<<grid2d, block2d, 0, stream>>>(
            max_mask, pooled, H, W, radius);

        nms_suppress_kernel<<<grid1d, block1d, 0, stream>>>(
            score_map, pooled, tmp_pool, max_mask, supp_scores, HW);

        max_pool2d_kernel<<<grid2d, block2d, 0, stream>>>(
            supp_scores, tmp_pool, H, W, radius);

        nms_suppress_kernel<<<grid1d, block1d, 0, stream>>>(
            score_map, pooled, tmp_pool, max_mask, supp_scores, HW);
    }

    cudaMemcpyAsync(nms_out, max_mask, HW * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}

// --------------------------------------------------------------------------
// TopK selection (CUB radix sort) with pre-allocated workspace
// --------------------------------------------------------------------------

__global__ void gather_candidates_kernel(
    const float* __restrict__ nms_scores,
    int H, int W, int border, float threshold,
    float* __restrict__ cand_scores,
    int* __restrict__ cand_indices,
    int* __restrict__ count,
    int max_candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    if (idx >= HW) return;

    float score = nms_scores[idx];
    if (score <= threshold) return;

    int y = idx / W;
    int x = idx % W;
    if (x < border || x >= W - border || y < border || y >= H - border) return;

    int pos = atomicAdd(count, 1);
    if (pos < max_candidates) {
        cand_scores[pos] = score;
        cand_indices[pos] = idx;
    }
}

__global__ void extract_topk_xy_kernel(
    const float* __restrict__ sorted_scores,
    const int* __restrict__ sorted_indices,
    int K, int W,
    float* __restrict__ kpts_xy,
    float* __restrict__ kpt_scores) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;

    int flat = sorted_indices[i];
    kpts_xy[i * 2 + 0] = static_cast<float>(flat % W);
    kpts_xy[i * 2 + 1] = static_cast<float>(flat / W);
    kpt_scores[i] = sorted_scores[i];
}

void topk_cuda(const float* nms_scores, int H, int W,
               int top_k, int border,
               float* kpts_xy, float* scores, int* count,
               DKDWorkspace& ws, cudaStream_t stream) {
    int HW = H * W;
    int block = 256;
    int grid = (HW + block - 1) / block;
    float threshold = 0.001f;

    // Reuse workspace buffers (NMS is done, these are now free)
    float* d_cand_scores   = static_cast<float*>(ws.buf[0]);
    int*   d_cand_indices  = static_cast<int*>(ws.buf[1]);
    float* d_sorted_scores = static_cast<float*>(ws.buf[2]);
    int*   d_sorted_indices = static_cast<int*>(ws.buf[3]);
    int*   d_count         = static_cast<int*>(ws.count_buf);

    cudaMemsetAsync(d_count, 0, sizeof(int), stream);

    gather_candidates_kernel<<<grid, block, 0, stream>>>(
        nms_scores, H, W, border, threshold,
        d_cand_scores, d_cand_indices, d_count, HW);

    int h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    h_count = std::min(h_count, HW);

    int K = 0;
    if (h_count > 0) {
        // CUB radix sort using pre-allocated temp storage
        size_t needed = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, needed,
            d_cand_scores, d_sorted_scores,
            d_cand_indices, d_sorted_indices,
            h_count, 0, sizeof(float) * 8, stream);

        // Use pre-allocated cub_temp (sized for max elements)
        cub::DeviceRadixSort::SortPairsDescending(
            ws.cub_temp, ws.cub_temp_bytes,
            d_cand_scores, d_sorted_scores,
            d_cand_indices, d_sorted_indices,
            h_count, 0, sizeof(float) * 8, stream);

        K = std::min(h_count, top_k);
        int ext_grid = (K + 255) / 256;
        extract_topk_xy_kernel<<<ext_grid, 256, 0, stream>>>(
            d_sorted_scores, d_sorted_indices, K, W,
            kpts_xy, scores);
    }

    cudaMemcpyAsync(count, &K, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
}

// --------------------------------------------------------------------------
// Sub-pixel refinement via soft-argmax
// --------------------------------------------------------------------------
__global__ void subpixel_refine_kernel(
    const float* __restrict__ score_map,
    float* __restrict__ kpts_xy,
    int N, int H, int W,
    int radius, float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float cx = kpts_xy[idx * 2 + 0];
    float cy = kpts_xy[idx * 2 + 1];
    int ix = __float2int_rn(cx);
    int iy = __float2int_rn(cy);

    float max_val = -FLT_MAX;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int ny = iy + dy;
            int nx = ix + dx;
            if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                max_val = fmaxf(max_val, score_map[ny * W + nx]);
            }
        }
    }

    float sum_exp = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int ny = iy + dy;
            int nx = ix + dx;
            float val = 0.0f;
            if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                val = score_map[ny * W + nx];
            }
            float w = expf((val - max_val) / temperature);
            sum_exp += w;
            sum_x += w * static_cast<float>(dx);
            sum_y += w * static_cast<float>(dy);
        }
    }

    if (sum_exp > 0.0f) {
        kpts_xy[idx * 2 + 0] = cx + sum_x / sum_exp;
        kpts_xy[idx * 2 + 1] = cy + sum_y / sum_exp;
    }
}

void subpixel_refine_cuda(const float* score_map, float* kpts_xy,
                          int N, int H, int W,
                          int radius, float temperature,
                          cudaStream_t stream) {
    if (N <= 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    subpixel_refine_kernel<<<grid, block, 0, stream>>>(
        score_map, kpts_xy, N, H, W, radius, temperature);
}

// --------------------------------------------------------------------------
// Full DKD pipeline
// --------------------------------------------------------------------------
void detect_keypoints(const float* score_map, int H, int W,
                      const DKDParams& params,
                      float* kpts_xy, float* kpt_scores, int* num_kpts,
                      void* nms_buf, DKDWorkspace& workspace,
                      cudaStream_t stream) {
    float* nms_out = static_cast<float*>(nms_buf);

    // 1. NMS
    nms_cuda(score_map, nms_out, H, W, params.nms_radius, workspace, stream);

    // 2. TopK / threshold selection
    topk_cuda(nms_out, H, W, params.top_k, params.border,
              kpts_xy, kpt_scores, num_kpts, workspace, stream);

    // 3. Sub-pixel refinement
    // topk_cuda already synced to get h_count; read num_kpts from device
    int h_count = 0;
    cudaMemcpyAsync(&h_count, num_kpts, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    h_count = std::min(h_count, params.top_k);

    if (h_count > 0) {
        subpixel_refine_cuda(score_map, kpts_xy, h_count, H, W,
                             params.nms_radius, params.temperature, stream);
    }

    // Update count on device
    cudaMemcpyAsync(num_kpts, &h_count, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
}

size_t nms_buf_size(int H, int W) {
    return static_cast<size_t>(H) * W * sizeof(float);
}

}  // namespace dkd
}  // namespace sfm_phoenix
