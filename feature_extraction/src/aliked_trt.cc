#include "feature_extraction/aliked_trt.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace feature_extraction {

AlikedDetector::~AlikedDetector() {
    if (stream_) cudaStreamDestroy(stream_);
}

bool AlikedDetector::init(const AlikedConfig& config) {
    config_ = config;

    if (!backbone_.load(config.backbone_engine)) {
        std::cerr << "Failed to load backbone engine" << std::endl;
        return false;
    }
    if (!sddh_.load(config.sddh_engine)) {
        std::cerr << "Failed to load SDDH engine" << std::endl;
        return false;
    }

    cudaStreamCreate(&stream_);

    // Pre-allocate DKD buffers for max keypoints
    int max_kpts = config_.dkd.top_k > 0 ? config_.dkd.top_k : config_.dkd.n_limit;
    dkd_kpts_.resize(max_kpts * 2 * sizeof(float));
    dkd_scores_.resize(max_kpts * sizeof(float));
    dkd_count_.resize(sizeof(int));

    return true;
}

void AlikedDetector::preprocess(const cv::Mat& image_bgr,
                                std::vector<float>& out,
                                int& padded_h, int& padded_w) {
    cv::Mat img = image_bgr;

    // Resize by max edge
    if (config_.max_edge > 0) {
        int h = img.rows, w = img.cols;
        scale_ = std::min(1.0f,
                          static_cast<float>(config_.max_edge) /
                              static_cast<float>(std::max(h, w)));
        if (scale_ < 1.0f) {
            cv::resize(img, img,
                       cv::Size(static_cast<int>(std::round(w * scale_)),
                                static_cast<int>(std::round(h * scale_))),
                       0, 0, cv::INTER_AREA);
        }
    } else {
        scale_ = 1.0f;
    }

    orig_h_ = img.rows;
    orig_w_ = img.cols;

    // Pad to multiple of 32
    padded_h = ((orig_h_ + 31) / 32) * 32;
    padded_w = ((orig_w_ + 31) / 32) * 32;
    pad_h_ = padded_h;
    pad_w_ = padded_w;

    // BGR → RGB, float32, [0, 1]
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgb_f;
    rgb.convertTo(rgb_f, CV_32FC3, 1.0 / 255.0);

    // Pad with border replication (matches PyTorch's replicate padding)
    int pad_bottom = padded_h - orig_h_;
    int pad_right = padded_w - orig_w_;
    if (pad_bottom > 0 || pad_right > 0) {
        cv::copyMakeBorder(rgb_f, rgb_f, 0, pad_bottom, 0, pad_right,
                           cv::BORDER_REPLICATE);
    }

    // HWC → CHW (NCHW with N=1)
    out.resize(3 * padded_h * padded_w);
    int hw = padded_h * padded_w;
    const float* src = reinterpret_cast<const float*>(rgb_f.data);
    for (int i = 0; i < hw; ++i) {
        out[0 * hw + i] = src[i * 3 + 0];  // R
        out[1 * hw + i] = src[i * 3 + 1];  // G
        out[2 * hw + i] = src[i * 3 + 2];  // B
    }
}

int AlikedDetector::run_dkd(int H, int W) {
    // Allocate workspace
    size_t ws_bytes = dkd::workspace_size(H, W);
    dkd_workspace_.resize(ws_bytes);

    // score_map is on GPU in backbone output buffer
    const float* score_map_gpu =
        static_cast<const float*>(backbone_.device_ptr("score_map"));

    dkd::detect_keypoints(
        score_map_gpu, H, W, config_.dkd,
        static_cast<float*>(dkd_kpts_.ptr),
        static_cast<float*>(dkd_scores_.ptr),
        static_cast<int*>(dkd_count_.ptr),
        dkd_workspace_.ptr, ws_bytes, stream_);

    cudaStreamSynchronize(stream_);

    int count = 0;
    cudaMemcpy(&count, dkd_count_.ptr, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
}

AlikedResult AlikedDetector::detect(const cv::Mat& image_bgr) {
    AlikedResult result;
    result.descriptor_dim = config_.descriptor_dim;

    // 1. Preprocess
    std::vector<float> input_data;
    int padded_h, padded_w;
    preprocess(image_bgr, input_data, padded_h, padded_w);

    // 2. Run backbone
    backbone_.set_input_shape("image", {1, 3, padded_h, padded_w});
    backbone_.set_input("image", input_data.data(),
                        input_data.size() * sizeof(float));
    backbone_.infer(stream_);

    // 3. Run DKD on score_map (still on GPU)
    int num_kpts = run_dkd(padded_h, padded_w);
    if (num_kpts <= 0) return result;

    // Clamp to actual limit
    int max_kpts = config_.dkd.top_k > 0 ? config_.dkd.top_k : config_.dkd.n_limit;
    num_kpts = std::min(num_kpts, max_kpts);
    result.num_keypoints = num_kpts;

    // 4. Run SDDH: feature_map (on GPU) + keypoints → descriptors
    // Set SDDH inputs
    sddh_.set_input_shape("feature_map",
                          {1, config_.descriptor_dim, padded_h, padded_w});
    sddh_.set_input_shape("keypoints_wh", {num_kpts, 2});

    // feature_map: copy GPU→GPU from backbone output to SDDH input
    size_t fm_bytes = static_cast<size_t>(config_.descriptor_dim) * padded_h *
                      padded_w * sizeof(float);
    cudaMemcpyAsync(sddh_.device_ptr("feature_map"),
                    backbone_.device_ptr("feature_map"), fm_bytes,
                    cudaMemcpyDeviceToDevice, stream_);

    // keypoints: already on GPU in dkd_kpts_
    cudaMemcpyAsync(sddh_.device_ptr("keypoints_wh"), dkd_kpts_.ptr,
                    num_kpts * 2 * sizeof(float), cudaMemcpyDeviceToDevice,
                    stream_);

    sddh_.infer(stream_);
    cudaStreamSynchronize(stream_);

    // 5. Copy results to host
    // Keypoints
    std::vector<float> kpts_flat(num_kpts * 2);
    cudaMemcpy(kpts_flat.data(), dkd_kpts_.ptr,
               num_kpts * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    result.keypoints.resize(num_kpts);
    for (int i = 0; i < num_kpts; ++i) {
        float x = kpts_flat[i * 2 + 0];
        float y = kpts_flat[i * 2 + 1];
        // Unpad and unscale back to original image coordinates
        result.keypoints[i] = cv::Point2f(x / scale_, y / scale_);
    }

    // Scores
    result.scores.resize(num_kpts);
    cudaMemcpy(result.scores.data(), dkd_scores_.ptr,
               num_kpts * sizeof(float), cudaMemcpyDeviceToHost);

    // Descriptors
    int D = config_.descriptor_dim;
    result.descriptors.resize(num_kpts * D);
    sddh_.get_output("descriptors", result.descriptors.data(),
                     num_kpts * D * sizeof(float));

    return result;
}

}  // namespace feature_extraction
