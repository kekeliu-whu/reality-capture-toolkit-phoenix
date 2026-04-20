#include "sfm_phoenix/extractors/aliked.h"
#include "sfm_phoenix/internal/preprocess_cuda.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <vector>

namespace sfm_phoenix {

AlikedDetector::~AlikedDetector() {
    if (stream_) cudaStreamDestroy(stream_);
}

bool AlikedDetector::init(const AlikedConfig& config) {
    config_ = config;

    if (!backbone_.load(config.backbone_engine)) {
        spdlog::error("Failed to load backbone engine: {}",
                      config.backbone_engine);
        return false;
    }
    if (!sddh_.load(config.sddh_engine)) {
        spdlog::error("Failed to load SDDH engine: {}", config.sddh_engine);
        return false;
    }

    int max_kpts = config_.dkd.top_k > 0 ? config_.dkd.top_k : config_.dkd.n_limit;
    const auto sddh_max_shape = sddh_.max_profile_shape("keypoints_wh");
    if (!sddh_max_shape.empty() && max_kpts > sddh_max_shape[0]) {
        spdlog::error(
            "Configured Phoenix top_k={} exceeds SDDH engine profile max {}. "
            "Rebuild {} with a larger keypoints_wh profile.",
            max_kpts,
            sddh_max_shape[0],
            config.sddh_engine);
        return false;
    }

    cudaStreamCreate(&stream_);

    // Pre-allocate DKD buffers for max keypoints
    dkd_kpts_.resize(max_kpts * 2 * sizeof(float));
    dkd_scores_.resize(max_kpts * sizeof(float));
    dkd_count_.resize(sizeof(int));

    // Pre-allocate NMS output buffer for max image size
    int max_dim = config_.max_edge > 0 ? config_.max_edge : 1600;
    int max_padded = ((max_dim + 31) / 32) * 32;
    dkd_nms_buf_.resize(dkd::nms_buf_size(max_padded, max_padded));

    // Pre-allocate DKD workspace (eliminates per-frame cudaMalloc/cudaFree)
    dkd_workspace_.init(max_padded, max_padded);

    // Pre-allocate GPU buffer for preprocess upload (max BGR uint8)
    size_t max_img_bytes = static_cast<size_t>(max_dim) * max_dim * 3;
    preprocess_buf_.resize(max_img_bytes);

    return true;
}

void AlikedDetector::preprocess(const cv::Mat& image,
                                const bool input_is_rgb,
                                int& padded_h, int& padded_w) {
    cv::Mat img = image;

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

    // Pad with border replication (on CPU, BGR uint8)
    int pad_bottom = padded_h - orig_h_;
    int pad_right = padded_w - orig_w_;
    if (pad_bottom > 0 || pad_right > 0) {
        cv::copyMakeBorder(img, img, 0, pad_bottom, 0, pad_right,
                           cv::BORDER_REPLICATE);
    }

    if (!img.isContinuous()) img = img.clone();

    // Upload uint8 HWC image to GPU and convert to CHW RGB float [0,1].
    size_t img_bytes = static_cast<size_t>(padded_h) * padded_w * 3;
    preprocess_buf_.resize(img_bytes);
    cudaMemcpyAsync(preprocess_buf_.ptr, img.data, img_bytes,
                    cudaMemcpyHostToDevice, stream_);

    backbone_.set_input_shape("image", {1, 3, padded_h, padded_w});
    float* dst_gpu = static_cast<float*>(backbone_.device_ptr("image"));

    if (input_is_rgb) {
        rgb_hwc_to_rgb_chw_gpu(
            static_cast<const unsigned char*>(preprocess_buf_.ptr),
            dst_gpu, padded_h, padded_w, stream_);
    } else {
        bgr_hwc_to_rgb_chw_gpu(
            static_cast<const unsigned char*>(preprocess_buf_.ptr),
            dst_gpu, padded_h, padded_w, stream_);
    }
}

int AlikedDetector::run_pipeline(const cv::Mat& image,
                                 const bool input_is_rgb) {
    static bool profile = (std::getenv("PROFILE") != nullptr);
    std::chrono::high_resolution_clock::time_point tp[5];

    if (profile) tp[0] = std::chrono::high_resolution_clock::now();

    // 1. Preprocess
    int padded_h, padded_w;
    preprocess(image, input_is_rgb, padded_h, padded_w);

    if (profile) {
        cudaStreamSynchronize(stream_);
        tp[1] = std::chrono::high_resolution_clock::now();
    }

    // 2. Run backbone (async)
    backbone_.infer(stream_);

    if (profile) {
        cudaStreamSynchronize(stream_);
        tp[2] = std::chrono::high_resolution_clock::now();
    }

    // 3. Run DKD on score_map (still on GPU)
    const float* score_map_gpu =
        static_cast<const float*>(backbone_.device_ptr("score_map"));

    dkd::detect_keypoints(
        score_map_gpu, padded_h, padded_w, config_.dkd,
        static_cast<float*>(dkd_kpts_.ptr),
        static_cast<float*>(dkd_scores_.ptr),
        static_cast<int*>(dkd_count_.ptr),
        dkd_nms_buf_.ptr, dkd_workspace_, stream_);

    // detect_keypoints already synced internally to get keypoint count.
    // Read the count from device.
    int num_kpts = 0;
    cudaMemcpyAsync(&num_kpts, dkd_count_.ptr, sizeof(int),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    if (profile) tp[3] = std::chrono::high_resolution_clock::now();

    int max_kpts = config_.dkd.top_k > 0 ? config_.dkd.top_k : config_.dkd.n_limit;
    num_kpts = std::min(num_kpts, max_kpts);
    if (num_kpts <= 0) return 0;

    // 4. Run SDDH: use pointer aliasing for feature_map (no D2D copy!)
    sddh_.set_input_shape("feature_map",
                          {1, config_.descriptor_dim, padded_h, padded_w});
    sddh_.set_input_shape("keypoints_wh", {num_kpts, 2});
    sddh_.set_input_shape("feature_map_hw", {2});

    // Alias SDDH feature_map input directly to backbone feature_map output
    sddh_.set_device_input("feature_map",
                           backbone_.device_ptr("feature_map"));

    // Alias SDDH keypoints input directly to DKD output
    sddh_.set_device_input("keypoints_wh", dkd_kpts_.ptr);

    // feature_map_hw: small H2D for [H, W]
    float fm_hw[2] = {static_cast<float>(padded_h),
                      static_cast<float>(padded_w)};
    sddh_.set_input("feature_map_hw", fm_hw, 2 * sizeof(float), stream_);

    sddh_.infer(stream_);

    if (profile) {
        cudaStreamSynchronize(stream_);
        tp[4] = std::chrono::high_resolution_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        spdlog::info("[PROFILE] preproc={}ms backbone={}ms dkd={}ms sddh={}ms",
                 ms(tp[0], tp[1]),
                 ms(tp[1], tp[2]),
                 ms(tp[2], tp[3]),
                 ms(tp[3], tp[4]));
    }
    // Don't sync here — let the caller decide when to sync.

    return num_kpts;
}

GpuDetectResult AlikedDetector::detect_gpu(const cv::Mat& image_bgr) {
    GpuDetectResult result;
    result.descriptor_dim = config_.descriptor_dim;

    int num_kpts = run_pipeline(image_bgr, false);
    if (num_kpts <= 0) return result;

    result.num_keypoints = num_kpts;
    result.scale = scale_;
    int D = config_.descriptor_dim;

    // D2D copies from internal buffers to result buffers (very fast, ~3μs)
    result.kpts.resize(num_kpts * 2 * sizeof(float));
    result.scores.resize(num_kpts * sizeof(float));
    result.descs.resize(num_kpts * D * sizeof(float));

    cudaMemcpyAsync(result.kpts.ptr, dkd_kpts_.ptr,
                    num_kpts * 2 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(result.scores.ptr, dkd_scores_.ptr,
                    num_kpts * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(result.descs.ptr, sddh_.device_ptr("descriptors"),
                    num_kpts * D * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // Clear external pointer aliases before next call
    sddh_.clear_device_inputs();

    cudaStreamSynchronize(stream_);
    return result;
}

AlikedResult AlikedDetector::detect(const cv::Mat& image_bgr) {
    AlikedResult result;
    result.descriptor_dim = config_.descriptor_dim;

    int num_kpts = run_pipeline(image_bgr, false);
    if (num_kpts <= 0) return result;

    result.num_keypoints = num_kpts;
    result.scale = scale_;
    int D = config_.descriptor_dim;

    // Sync to ensure SDDH is done before D2H copies
    cudaStreamSynchronize(stream_);

    // D2H copies
    std::vector<float> kpts_flat(num_kpts * 2);
    cudaMemcpyAsync(kpts_flat.data(), dkd_kpts_.ptr,
                    num_kpts * 2 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);

    result.scores.resize(num_kpts);
    cudaMemcpyAsync(result.scores.data(), dkd_scores_.ptr,
                    num_kpts * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);

    result.descriptors.resize(num_kpts * D);
    sddh_.get_output("descriptors", result.descriptors.data(),
                     num_kpts * D * sizeof(float), stream_);

    // Clear external pointer aliases
    sddh_.clear_device_inputs();

    cudaStreamSynchronize(stream_);

    result.keypoints.resize(num_kpts);
    for (int i = 0; i < num_kpts; ++i) {
        result.keypoints[i] = cv::Point2f(kpts_flat[i * 2 + 0],
                                          kpts_flat[i * 2 + 1]);
    }

    // Sanitize NaN descriptors
    for (int i = 0; i < num_kpts * D; ++i) {
        if (std::isnan(result.descriptors[i])) {
            result.descriptors[i] = 0.0f;
        }
    }

    return result;
}

AlikedResult AlikedDetector::detect_rgb(const cv::Mat& image_rgb) {
    AlikedResult result;
    result.descriptor_dim = config_.descriptor_dim;

    int num_kpts = run_pipeline(image_rgb, true);
    if (num_kpts <= 0) return result;

    result.num_keypoints = num_kpts;
    result.scale = scale_;
    int D = config_.descriptor_dim;

    cudaStreamSynchronize(stream_);

    std::vector<float> kpts_flat(num_kpts * 2);
    cudaMemcpyAsync(kpts_flat.data(), dkd_kpts_.ptr,
                    num_kpts * 2 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);

    result.scores.resize(num_kpts);
    cudaMemcpyAsync(result.scores.data(), dkd_scores_.ptr,
                    num_kpts * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);

    result.descriptors.resize(num_kpts * D);
    sddh_.get_output("descriptors", result.descriptors.data(),
                     num_kpts * D * sizeof(float), stream_);

    sddh_.clear_device_inputs();

    cudaStreamSynchronize(stream_);

    result.keypoints.resize(num_kpts);
    for (int i = 0; i < num_kpts; ++i) {
        result.keypoints[i] = cv::Point2f(kpts_flat[i * 2 + 0],
                                          kpts_flat[i * 2 + 1]);
    }

    for (int i = 0; i < num_kpts * D; ++i) {
        if (std::isnan(result.descriptors[i])) {
            result.descriptors[i] = 0.0f;
        }
    }

    return result;
}

}  // namespace sfm_phoenix
