#include "feature_extraction/feature_pipeline.h"

#include <iostream>

namespace feature_extraction {

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

MatchResult FeaturePipeline::detect_and_match(const cv::Mat& image0_bgr,
                                              const cv::Mat& image1_bgr) {
    auto r0 = detect(image0_bgr);
    auto r1 = detect(image1_bgr);
    return match(r0, r1);
}

}  // namespace feature_extraction
