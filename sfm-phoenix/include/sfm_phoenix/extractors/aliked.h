#pragma once

#include "sfm_phoenix/internal/dkd_cuda.h"
#include "sfm_phoenix/internal/trt_engine.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace sfm_phoenix {

/// Result of ALIKED feature extraction for a single image.
struct AlikedResult {
    std::vector<cv::Point2f> keypoints;      // Pixel coordinates (padded image space)
    std::vector<float> scores;               // Keypoint confidence
    std::vector<float> descriptors;          // Row-major [N, 128]
    int descriptor_dim = 128;
    int num_keypoints = 0;
    float scale = 1.0f;                      // resize scale (padded→original: x/scale)
};

/// GPU-resident detection result.  Keeps keypoints/descriptors on device.
struct GpuDetectResult {
    CudaBuffer kpts;         // [N, 2] float on device
    CudaBuffer descs;        // [N, D] float on device
    CudaBuffer scores;       // [N] float on device
    int num_keypoints = 0;
    int descriptor_dim = 128;
    float scale = 1.0f;
};

/// Configuration for the ALIKED TensorRT pipeline.
struct AlikedConfig {
    std::string backbone_engine;   // Path to aliked_backbone.engine
    std::string sddh_engine;      // Path to aliked_sddh.engine
    dkd::DKDParams dkd;
    int max_edge = 1600;           // Resize longest edge to this (0 = no resize)
    int descriptor_dim = 128;
};

/// ALIKED feature detector using TensorRT.
///
/// Pipeline:
///   Image → [TRT backbone] → feature_map + score_map
///       → [CUDA DKD] → keypoints
///       → [TRT SDDH] → descriptors
class AlikedDetector {
public:
    AlikedDetector() = default;
    ~AlikedDetector();

    /// Initialise with engine paths and parameters.
    bool init(const AlikedConfig& config);

    /// Extract features from a BGR image (copies results to host).
    AlikedResult detect(const cv::Mat& image_bgr);

    /// Extract features, keeping results on GPU (no D2H copy).
    /// Much faster when results will be passed directly to LightGlue.
    GpuDetectResult detect_gpu(const cv::Mat& image_bgr);

private:
    AlikedConfig config_;
    TrtEngine backbone_;
    TrtEngine sddh_;
    cudaStream_t stream_ = nullptr;

    // Pre-allocated GPU workspace for DKD
    CudaBuffer dkd_nms_buf_;     // NMS output buffer
    CudaBuffer dkd_kpts_;        // [max_kpts, 2]
    CudaBuffer dkd_scores_;      // [max_kpts]
    CudaBuffer dkd_count_;       // [1]
    dkd::DKDWorkspace dkd_workspace_;  // Pre-allocated DKD temp buffers
    CudaBuffer preprocess_buf_;  // GPU staging for BGR uint8 upload

    // Padding bookkeeping
    int pad_h_ = 0, pad_w_ = 0;  // Padded dimensions (divisible by 32)
    int orig_h_ = 0, orig_w_ = 0;
    float scale_ = 1.0f;

    /// Pre-process: BGR → RGB float, resize, pad to 32-multiple.
    void preprocess(const cv::Mat& image_bgr,
                    int& padded_h, int& padded_w);

    /// Run backbone + DKD + SDDH, leaving results on GPU.
    /// Returns the actual keypoint count.
    int run_pipeline(const cv::Mat& image_bgr);
};

}  // namespace sfm_phoenix
