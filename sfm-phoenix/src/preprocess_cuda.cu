#include "feature_extraction/preprocess_cuda.h"

namespace feature_extraction {

__global__ void bgr_hwc_to_rgb_chw_kernel(const unsigned char* __restrict__ src,
                                           float* __restrict__ dst,
                                           int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hw = H * W;
    if (idx >= hw) return;

    unsigned char b = src[idx * 3 + 0];
    unsigned char g = src[idx * 3 + 1];
    unsigned char r = src[idx * 3 + 2];

    constexpr float inv255 = 1.0f / 255.0f;
    dst[0 * hw + idx] = r * inv255;  // R channel
    dst[1 * hw + idx] = g * inv255;  // G channel
    dst[2 * hw + idx] = b * inv255;  // B channel
}

void bgr_hwc_to_rgb_chw_gpu(const unsigned char* src_bgr_hwc,
                             float* dst_chw_gpu,
                             int H, int W,
                             cudaStream_t stream) {
    int hw = H * W;
    int threads = 256;
    int blocks = (hw + threads - 1) / threads;
    bgr_hwc_to_rgb_chw_kernel<<<blocks, threads, 0, stream>>>(
        src_bgr_hwc, dst_chw_gpu, H, W);
}

}  // namespace feature_extraction
