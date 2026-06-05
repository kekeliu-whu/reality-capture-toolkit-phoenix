#include "sfm_phoenix/extractors/aliked.h"
#include "sfm_phoenix/internal/preprocess_cuda.h"
#include "sfm_phoenix/internal/runtime_utils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <vector>

namespace sfm_phoenix {

namespace {

std::filesystem::path ResolveFullModelEnginePath(
    const std::string& model_path) {
    const std::filesystem::path path(model_path);
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Full ALIKED model not found: " +
                                 path.string());
    }

    if (path.extension() == ".engine") {
        return path;
    }
    if (path.extension() != ".onnx") {
        throw std::runtime_error(
            "Unsupported full ALIKED model extension: " + path.string());
    }

    auto engine_path = path;
    engine_path.replace_extension(".engine");
    if (std::filesystem::exists(engine_path)) {
        return engine_path;
    }

    TrtBuildOptions build_options;
    build_options.enable_fp16 = false;
    build_options.builder_optimization_level = 3;
    spdlog::info("Building full ALIKED engine from {}", path.string());
    if (!BuildSerializedEngine(path, engine_path, build_options)) {
        throw std::runtime_error("Failed to build full ALIKED engine from: " +
                                 path.string());
    }
    return engine_path;
}

bool HasPositiveDims(const std::vector<int>& shape) {
    return !shape.empty() &&
           std::all_of(shape.begin(), shape.end(),
                       [](const int dim) { return dim > 0; });
}

bool KeypointsUseNormalizedCoordinates(
    const std::vector<float>& keypoints) {
    float max_abs_value = 0.0f;
    for (const float value : keypoints) {
        if (!std::isfinite(value)) continue;
        max_abs_value = std::max(max_abs_value, std::abs(value));
    }
    return max_abs_value <= 2.0f;
}

void PopulateFullModelHostResult(const int raw_num_keypoints,
                                 const int descriptor_dim,
                                 const int input_width,
                                 const int input_height,
                                 const int valid_width,
                                 const int valid_height,
                                 const float scale,
                                 const std::vector<float>& raw_keypoints,
                                 const std::vector<float>& raw_scores,
                                 const std::vector<float>& raw_descriptors,
                                 AlikedResult* result) {
    result->descriptor_dim = descriptor_dim;
    result->scale = scale;
    result->keypoints.clear();
    result->scores.clear();
    result->descriptors.clear();
    result->keypoints.reserve(raw_num_keypoints);
    result->scores.reserve(raw_num_keypoints);
    result->descriptors.reserve(
        static_cast<size_t>(raw_num_keypoints) * descriptor_dim);

    const bool normalized = KeypointsUseNormalizedCoordinates(raw_keypoints);
    for (int index = 0; index < raw_num_keypoints; ++index) {
        float keypoint_x = raw_keypoints[index * 2 + 0];
        float keypoint_y = raw_keypoints[index * 2 + 1];
        if (normalized) {
            keypoint_x = (keypoint_x + 1.0f) * 0.5f *
                         static_cast<float>(input_width - 1);
            keypoint_y = (keypoint_y + 1.0f) * 0.5f *
                         static_cast<float>(input_height - 1);
        }
        if (!std::isfinite(keypoint_x) || !std::isfinite(keypoint_y) ||
            keypoint_x < 0.0f || keypoint_y < 0.0f ||
            keypoint_x >= static_cast<float>(valid_width) ||
            keypoint_y >= static_cast<float>(valid_height)) {
            continue;
        }

        result->keypoints.emplace_back(keypoint_x, keypoint_y);
        result->scores.push_back(
            index < static_cast<int>(raw_scores.size()) ? raw_scores[index]
                                                        : 0.0f);

        const size_t descriptor_begin =
            static_cast<size_t>(index) * descriptor_dim;
        for (int dim = 0; dim < descriptor_dim; ++dim) {
            const size_t descriptor_index = descriptor_begin + dim;
            float descriptor_value =
                descriptor_index < raw_descriptors.size()
                    ? raw_descriptors[descriptor_index]
                    : 0.0f;
            if (std::isnan(descriptor_value)) descriptor_value = 0.0f;
            result->descriptors.push_back(descriptor_value);
        }
    }
    result->num_keypoints = static_cast<int>(result->keypoints.size());
}

}  // namespace

AlikedDetector::~AlikedDetector() {
    if (stream_) cudaStreamDestroy(stream_);
}

bool AlikedDetector::init(const AlikedConfig& config) {
    config_ = config;
    if (config.full_model_path.empty()) {
        spdlog::error("ALIKED full model path must be set");
        return false;
    }

    const auto engine_path = ResolveFullModelEnginePath(config.full_model_path);
    if (!full_model_.load(engine_path.string())) {
        spdlog::error("Failed to load full ALIKED engine: {}",
                      engine_path.string());
        return false;
    }

    auto input_shape = full_model_.shape("image");
    if (!HasPositiveDims(input_shape)) {
        input_shape = full_model_.max_profile_shape("image");
    }
    if (input_shape.size() != 4 || input_shape[2] <= 0 ||
        input_shape[3] <= 0) {
        spdlog::error("Unsupported full ALIKED input shape for image");
        return false;
    }

    full_input_h_ = input_shape[2];
    full_input_w_ = input_shape[3];
    cudaStreamCreate(&stream_);
    preprocess_buf_.resize(
        static_cast<size_t>(full_input_h_) * full_input_w_ * 3);
    use_full_model_ = true;
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

void AlikedDetector::preprocess_full(const cv::Mat& image,
                                     const bool input_is_rgb) {
    cv::Mat img = image;

    const float scale_h = static_cast<float>(full_input_h_) /
                          static_cast<float>(img.rows);
    const float scale_w = static_cast<float>(full_input_w_) /
                          static_cast<float>(img.cols);
    scale_ = std::min(scale_h, scale_w);

    const int resized_h = std::min(
        full_input_h_,
        std::max(1, static_cast<int>(std::round(img.rows * scale_))));
    const int resized_w = std::min(
        full_input_w_,
        std::max(1, static_cast<int>(std::round(img.cols * scale_))));
    if (resized_h != img.rows || resized_w != img.cols) {
        const int interpolation =
            scale_ < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(img, img, cv::Size(resized_w, resized_h), 0, 0,
                   interpolation);
    }

    orig_h_ = img.rows;
    orig_w_ = img.cols;
    pad_h_ = full_input_h_;
    pad_w_ = full_input_w_;

    const int pad_bottom = full_input_h_ - orig_h_;
    const int pad_right = full_input_w_ - orig_w_;
    if (pad_bottom > 0 || pad_right > 0) {
        cv::copyMakeBorder(img, img, 0, pad_bottom, 0, pad_right,
                           cv::BORDER_REPLICATE);
    }

    if (!img.isContinuous()) img = img.clone();

    const size_t img_bytes =
        static_cast<size_t>(full_input_h_) * full_input_w_ * 3;
    preprocess_buf_.resize(img_bytes);
    cudaMemcpyAsync(preprocess_buf_.ptr, img.data, img_bytes,
                    cudaMemcpyHostToDevice, stream_);

    full_model_.set_input_shape("image", {1, 3, full_input_h_, full_input_w_});
    float* dst_gpu = static_cast<float*>(full_model_.device_ptr("image"));
    if (input_is_rgb) {
        rgb_hwc_to_rgb_chw_gpu(
            static_cast<const unsigned char*>(preprocess_buf_.ptr),
            dst_gpu, full_input_h_, full_input_w_, stream_);
    } else {
        bgr_hwc_to_rgb_chw_gpu(
            static_cast<const unsigned char*>(preprocess_buf_.ptr),
            dst_gpu, full_input_h_, full_input_w_, stream_);
    }
}

int AlikedDetector::run_pipeline(const cv::Mat& image,
                                 const bool input_is_rgb) {
    static bool profile = internal::EnvVarEnabled("PROFILE");
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

int AlikedDetector::run_full_model(const cv::Mat& image,
                                   const bool input_is_rgb) {
    preprocess_full(image, input_is_rgb);
    full_model_.infer(stream_);
    cudaStreamSynchronize(stream_);

    const auto keypoints_shape = full_model_.shape("keypoints");
    if (keypoints_shape.size() != 2 || keypoints_shape[1] != 2) {
        throw std::runtime_error(
            "Unexpected full ALIKED keypoints output shape");
    }
    return std::max(0, keypoints_shape[0]);
}

int AlikedDetector::backbone_max_batch() const {
    const auto shape = full_model_.max_profile_shape("image");
    return shape.empty() ? 1 : shape[0];
}

GpuDetectResult AlikedDetector::detect_gpu(const cv::Mat& image_bgr) {
    GpuDetectResult result;
    result.descriptor_dim = config_.descriptor_dim;

    if (use_full_model_) {
        const int num_kpts = run_full_model(image_bgr, false);
        if (num_kpts <= 0) return result;

        const auto descriptors_shape = full_model_.shape("descriptors");
        const int descriptor_dim =
            descriptors_shape.size() >= 2 ? descriptors_shape[1]
                                          : config_.descriptor_dim;

        result.num_keypoints = num_kpts;
        result.descriptor_dim = descriptor_dim;
        result.scale = scale_;

        result.kpts.resize(num_kpts * 2 * sizeof(float));
        result.scores.resize(num_kpts * sizeof(float));
        result.descs.resize(static_cast<size_t>(num_kpts) * descriptor_dim *
                            sizeof(float));

        cudaMemcpyAsync(result.kpts.ptr, full_model_.device_ptr("keypoints"),
                        num_kpts * 2 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(result.scores.ptr, full_model_.device_ptr("scores"),
                        num_kpts * sizeof(float), cudaMemcpyDeviceToDevice,
                        stream_);
        cudaMemcpyAsync(result.descs.ptr,
                        full_model_.device_ptr("descriptors"),
                        static_cast<size_t>(num_kpts) * descriptor_dim *
                            sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaStreamSynchronize(stream_);
        return result;
    }

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

    if (use_full_model_) {
        const int num_kpts = run_full_model(image_bgr, false);
        if (num_kpts <= 0) return result;

        const auto descriptors_shape = full_model_.shape("descriptors");
        const int descriptor_dim =
            descriptors_shape.size() >= 2 ? descriptors_shape[1]
                                          : config_.descriptor_dim;

        std::vector<float> kpts_flat(num_kpts * 2);
        std::vector<float> scores(num_kpts);
        std::vector<float> descriptors(static_cast<size_t>(num_kpts) *
                                       descriptor_dim);

        full_model_.get_output("keypoints", kpts_flat.data(),
                               num_kpts * 2 * sizeof(float), stream_);
        full_model_.get_output("scores", scores.data(),
                               num_kpts * sizeof(float), stream_);
        full_model_.get_output(
            "descriptors", descriptors.data(),
            static_cast<size_t>(num_kpts) * descriptor_dim * sizeof(float),
            stream_);
        cudaStreamSynchronize(stream_);

        PopulateFullModelHostResult(num_kpts, descriptor_dim,
                                    full_input_w_, full_input_h_,
                                    orig_w_, orig_h_, scale_,
                                    kpts_flat, scores, descriptors,
                                    &result);
        return result;
    }

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

    if (use_full_model_) {
        const int num_kpts = run_full_model(image_rgb, true);
        if (num_kpts <= 0) return result;

        const auto descriptors_shape = full_model_.shape("descriptors");
        const int descriptor_dim =
            descriptors_shape.size() >= 2 ? descriptors_shape[1]
                                          : config_.descriptor_dim;

        std::vector<float> kpts_flat(num_kpts * 2);
        std::vector<float> scores(num_kpts);
        std::vector<float> descriptors(static_cast<size_t>(num_kpts) *
                                       descriptor_dim);

        full_model_.get_output("keypoints", kpts_flat.data(),
                               num_kpts * 2 * sizeof(float), stream_);
        full_model_.get_output("scores", scores.data(),
                               num_kpts * sizeof(float), stream_);
        full_model_.get_output(
            "descriptors", descriptors.data(),
            static_cast<size_t>(num_kpts) * descriptor_dim * sizeof(float),
            stream_);
        cudaStreamSynchronize(stream_);

        PopulateFullModelHostResult(num_kpts, descriptor_dim,
                                    full_input_w_, full_input_h_,
                                    orig_w_, orig_h_, scale_,
                                    kpts_flat, scores, descriptors,
                                    &result);
        return result;
    }

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

std::vector<AlikedResult> AlikedDetector::detect_batch(
    const std::vector<cv::Mat>& images_bgr) {
    if (images_bgr.empty()) return {};

    if (use_full_model_) {
        std::vector<AlikedResult> results;
        results.reserve(images_bgr.size());
        for (const auto& image : images_bgr) {
            results.push_back(detect(image));
        }
        return results;
    }

    // Check engine supports batch > 1
    const auto img_max = backbone_.max_profile_shape("image");
    const int max_batch = img_max.empty() ? 1 : img_max[0];
    if (max_batch <= 1 || static_cast<int>(images_bgr.size()) == 1) {
        std::vector<AlikedResult> results;
        results.reserve(images_bgr.size());
        for (const auto& img : images_bgr) results.push_back(detect(img));
        return results;
    }

    const int B = static_cast<int>(images_bgr.size());

    // ------------------------------------------------------------------
    // 1. CPU preprocessing for each image
    // ------------------------------------------------------------------
    struct FrameInfo {
        cv::Mat padded;
        int pad_h = 0;
        int pad_w = 0;
        float scale = 1.0f;
    };
    std::vector<FrameInfo> frames(B);
    for (int i = 0; i < B; ++i) {
        cv::Mat img = images_bgr[i];
        float scale = 1.0f;
        if (config_.max_edge > 0) {
            const int h = img.rows, w = img.cols;
            scale = std::min(1.0f,
                             static_cast<float>(config_.max_edge) /
                                 static_cast<float>(std::max(h, w)));
            if (scale < 1.0f) {
                cv::resize(img, img,
                           cv::Size(static_cast<int>(std::round(w * scale)),
                                    static_cast<int>(std::round(h * scale))),
                           0, 0, cv::INTER_AREA);
            }
        }
        FrameInfo& f = frames[i];
        f.pad_h = ((img.rows + 31) / 32) * 32;
        f.pad_w = ((img.cols + 31) / 32) * 32;
        f.scale = scale;
        const int pb = f.pad_h - img.rows;
        const int pr = f.pad_w - img.cols;
        if (pb > 0 || pr > 0) {
            cv::copyMakeBorder(img, img, 0, pb, 0, pr,
                               cv::BORDER_REPLICATE);
        }
        if (!img.isContinuous()) img = img.clone();
        f.padded = std::move(img);
    }

    // All images must share the same padded size; otherwise fall back.
    const int pad_h = frames[0].pad_h;
    const int pad_w = frames[0].pad_w;
    for (int i = 1; i < B; ++i) {
        if (frames[i].pad_h != pad_h || frames[i].pad_w != pad_w) {
            std::vector<AlikedResult> results;
            results.reserve(B);
            for (const auto& img : images_bgr) results.push_back(detect(img));
            return results;
        }
    }

    // ------------------------------------------------------------------
    // 2. Upload all images to backbone input buffer [B, 3, H, W]
    // ------------------------------------------------------------------
    backbone_.set_input_shape("image", {B, 3, pad_h, pad_w});
    float* backbone_input =
        static_cast<float*>(backbone_.device_ptr("image"));

    const size_t img_bytes =
        static_cast<size_t>(pad_h) * pad_w * 3;   // uint8 bytes per image
    const size_t chw_floats =
        static_cast<size_t>(3) * pad_h * pad_w;   // float elems per CHW image

    preprocess_buf_.resize(img_bytes);
    for (int i = 0; i < B; ++i) {
        cudaMemcpyAsync(preprocess_buf_.ptr, frames[i].padded.data, img_bytes,
                        cudaMemcpyHostToDevice, stream_);
        bgr_hwc_to_rgb_chw_gpu(
            static_cast<const unsigned char*>(preprocess_buf_.ptr),
            backbone_input + static_cast<ptrdiff_t>(i) * chw_floats,
            pad_h, pad_w, stream_);
    }

    // ------------------------------------------------------------------
    // 3. Single backbone pass for the whole batch
    // ------------------------------------------------------------------
    backbone_.infer(stream_);

    const float* score_map_gpu =
        static_cast<const float*>(backbone_.device_ptr("score_map"));
    float* feature_map_gpu =
        static_cast<float*>(backbone_.device_ptr("feature_map"));

    const int C = config_.descriptor_dim;
    const size_t sm_per_image =
        static_cast<size_t>(pad_h) * pad_w;        // [1, H, W] floats
    const size_t fm_per_image =
        static_cast<size_t>(C) * pad_h * pad_w;    // [C, H, W] floats

    // ------------------------------------------------------------------
    // 4. Per-image DKD + SDDH (reuses pre-allocated workspace)
    // ------------------------------------------------------------------
    const int max_kpts =
        config_.dkd.top_k > 0 ? config_.dkd.top_k : config_.dkd.n_limit;

    std::vector<AlikedResult> results(B);
    for (int i = 0; i < B; ++i) {
        results[i].scale = frames[i].scale;
        results[i].descriptor_dim = C;

        const float* sm_i =
            score_map_gpu + static_cast<ptrdiff_t>(i) * sm_per_image;
        dkd::detect_keypoints(
            sm_i, pad_h, pad_w, config_.dkd,
            static_cast<float*>(dkd_kpts_.ptr),
            static_cast<float*>(dkd_scores_.ptr),
            static_cast<int*>(dkd_count_.ptr),
            dkd_nms_buf_.ptr, dkd_workspace_, stream_);

        int num_kpts = 0;
        cudaMemcpyAsync(&num_kpts, dkd_count_.ptr, sizeof(int),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        num_kpts = std::min(num_kpts, max_kpts);
        if (num_kpts <= 0) continue;

        results[i].num_keypoints = num_kpts;

        float* fm_i =
            feature_map_gpu + static_cast<ptrdiff_t>(i) * fm_per_image;
        sddh_.set_input_shape("feature_map", {1, C, pad_h, pad_w});
        sddh_.set_input_shape("keypoints_wh", {num_kpts, 2});
        sddh_.set_input_shape("feature_map_hw", {2});
        sddh_.set_device_input("feature_map", fm_i);
        sddh_.set_device_input("keypoints_wh", dkd_kpts_.ptr);
        float fm_hw[2] = {static_cast<float>(pad_h),
                          static_cast<float>(pad_w)};
        sddh_.set_input("feature_map_hw", fm_hw, 2 * sizeof(float), stream_);
        sddh_.infer(stream_);
        cudaStreamSynchronize(stream_);

        std::vector<float> kpts_flat(num_kpts * 2);
        cudaMemcpy(kpts_flat.data(), dkd_kpts_.ptr,
                   num_kpts * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        results[i].keypoints.resize(num_kpts);
        for (int k = 0; k < num_kpts; ++k) {
            results[i].keypoints[k] =
                cv::Point2f(kpts_flat[k * 2], kpts_flat[k * 2 + 1]);
        }

        results[i].scores.resize(num_kpts);
        cudaMemcpy(results[i].scores.data(), dkd_scores_.ptr,
                   num_kpts * sizeof(float), cudaMemcpyDeviceToHost);

        results[i].descriptors.resize(num_kpts * C);
        sddh_.get_output("descriptors", results[i].descriptors.data(),
                         static_cast<size_t>(num_kpts) * C * sizeof(float),
                         stream_);
        cudaStreamSynchronize(stream_);

        for (int j = 0; j < num_kpts * C; ++j) {
            if (std::isnan(results[i].descriptors[j])) {
                results[i].descriptors[j] = 0.0f;
            }
        }

        sddh_.clear_device_inputs();
    }

    return results;
}

}  // namespace sfm_phoenix
