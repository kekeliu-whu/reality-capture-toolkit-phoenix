#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace sfm_phoenix {

/// Upload a BGR uint8 image to GPU as CHW float [0,1] RGB.
/// src_bgr_hwc: host pointer to HWC BGR uint8 data (H * W * 3 bytes)
/// dst_chw_gpu: device pointer for CHW RGB float output (3 * H * W floats)
/// H, W: image dimensions
/// stream: CUDA stream
void bgr_hwc_to_rgb_chw_gpu(const unsigned char* src_bgr_hwc,
                             float* dst_chw_gpu,
                             int H, int W,
                             cudaStream_t stream);

}  // namespace sfm_phoenix
