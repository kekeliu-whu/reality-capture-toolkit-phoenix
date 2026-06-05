#include "sfm_phoenix/pipelines/feature_pipeline.h"
#include "sfm_phoenix/internal/runtime_utils.h"

#include <chrono>
#include <iostream>

namespace sfm_phoenix {

bool FeaturePipeline::init(const PipelineConfig& config) {
    if (!detector_.init(config.aliked)) {
        std::cerr << "Failed to init ALIKED detector" << std::endl;
        return false;
    }
    if (!matcher_.init(config.lightglue)) {
        std::cerr << "Failed to init LightGlue matcher" << std::endl;
        return false;
    }
    return true;
}

AlikedResult FeaturePipeline::detect(const cv::Mat& image_bgr) {
    return detector_.detect(image_bgr);
}

MatchResult FeaturePipeline::match(const AlikedResult& r0,
                                   const AlikedResult& r1) {
    if (r0.num_keypoints <= 0 || r1.num_keypoints <= 0) {
        return {};
    }

    // Flatten keypoints to [N, 2] float arrays
    std::vector<float> kpts0(r0.num_keypoints * 2);
    std::vector<float> kpts1(r1.num_keypoints * 2);
    for (int i = 0; i < r0.num_keypoints; ++i) {
        kpts0[i * 2 + 0] = r0.keypoints[i].x;
        kpts0[i * 2 + 1] = r0.keypoints[i].y;
    }
    for (int i = 0; i < r1.num_keypoints; ++i) {
        kpts1[i * 2 + 0] = r1.keypoints[i].x;
        kpts1[i * 2 + 1] = r1.keypoints[i].y;
    }

    return matcher_.match(kpts0.data(), r0.descriptors.data(),
                          r0.num_keypoints, kpts1.data(),
                          r1.descriptors.data(), r1.num_keypoints);
}

PairResult FeaturePipeline::detect_and_match_gpu(const cv::Mat& image0_bgr,
                                                  const cv::Mat& image1_bgr) {
    PairResult result;
    static bool profile = internal::EnvVarEnabled("PROFILE");

    std::chrono::high_resolution_clock::time_point t[4];
    if (profile) t[0] = std::chrono::high_resolution_clock::now();

    // Detect both images, keeping data on GPU
    result.det0 = detector_.detect_gpu(image0_bgr);

    if (profile) t[1] = std::chrono::high_resolution_clock::now();

    result.det1 = detector_.detect_gpu(image1_bgr);

    if (result.det0.num_keypoints <= 0 || result.det1.num_keypoints <= 0) {
        return result;
    }

    if (profile) t[2] = std::chrono::high_resolution_clock::now();

    // Match using GPU device pointers — no H2D copy!
    result.matches = matcher_.match_gpu(
        result.det0.kpts.ptr, result.det0.descs.ptr,
        result.det0.num_keypoints,
        result.det1.kpts.ptr, result.det1.descs.ptr,
        result.det1.num_keypoints);

    if (profile) {
        t[3] = std::chrono::high_resolution_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::cerr << "[PROFILE] det0=" << ms(t[0], t[1])
                  << "ms det1=" << ms(t[1], t[2])
                  << "ms match=" << ms(t[2], t[3])
                  << "ms total=" << ms(t[0], t[3]) << "ms" << std::endl;
    }

    return result;
}

PairResult FeaturePipeline::detect_and_match_gpu(
    GpuDetectResult&& cached_det0, const cv::Mat& image1_bgr) {
    PairResult result;
    static bool profile = internal::EnvVarEnabled("PROFILE");

    std::chrono::high_resolution_clock::time_point t[3];
    if (profile) t[0] = std::chrono::high_resolution_clock::now();

    result.det0 = std::move(cached_det0);
    result.det1 = detector_.detect_gpu(image1_bgr);

    if (result.det0.num_keypoints <= 0 || result.det1.num_keypoints <= 0)
        return result;

    if (profile) t[1] = std::chrono::high_resolution_clock::now();

    result.matches = matcher_.match_gpu(
        result.det0.kpts.ptr, result.det0.descs.ptr,
        result.det0.num_keypoints,
        result.det1.kpts.ptr, result.det1.descs.ptr,
        result.det1.num_keypoints);

    if (profile) {
        t[2] = std::chrono::high_resolution_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::cerr << "[PROFILE] det0=cached det1=" << ms(t[0], t[1])
                  << "ms match=" << ms(t[1], t[2])
                  << "ms total=" << ms(t[0], t[2]) << "ms" << std::endl;
    }

    return result;
}

MatchResult FeaturePipeline::detect_and_match(const cv::Mat& image0_bgr,
                                              const cv::Mat& image1_bgr) {
    auto r0 = detect(image0_bgr);
    auto r1 = detect(image1_bgr);
    return match(r0, r1);
}

}  // namespace sfm_phoenix
