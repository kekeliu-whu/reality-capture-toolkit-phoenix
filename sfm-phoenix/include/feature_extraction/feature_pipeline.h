#pragma once

#include "feature_extraction/aliked_trt.h"
#include "feature_extraction/lightglue_trt.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace feature_extraction {

/// Complete feature extraction + matching pipeline.
///
/// Usage:
///   FeaturePipeline pipe;
///   pipe.init(config);
///   auto result0 = pipe.detect(image0);
///   auto result1 = pipe.detect(image1);
///   auto matches = pipe.match(result0, result1);
struct PipelineConfig {
    AlikedConfig aliked;
    LightGlueConfig lightglue;
};

/// Result of the GPU-optimized detect-and-match pipeline.
struct PairResult {
    GpuDetectResult det0;
    GpuDetectResult det1;
    MatchResult matches;
};

class FeaturePipeline {
public:
    FeaturePipeline() = default;

    /// Initialise all engines.
    bool init(const PipelineConfig& config);

    /// Extract ALIKED features from a single image.
    AlikedResult detect(const cv::Mat& image_bgr);

    /// Match features between two images.
    MatchResult match(const AlikedResult& result0,
                      const AlikedResult& result1);

    /// GPU-optimized: detect both images and match without host round-trip.
    PairResult detect_and_match_gpu(const cv::Mat& image0_bgr,
                                    const cv::Mat& image1_bgr);

    /// GPU-optimized with detection caching: reuse pre-computed det0.
    PairResult detect_and_match_gpu(GpuDetectResult&& cached_det0,
                                    const cv::Mat& image1_bgr);

    /// Convenience: detect + match a pair of images.
    MatchResult detect_and_match(const cv::Mat& image0_bgr,
                                 const cv::Mat& image1_bgr);

private:
    AlikedDetector detector_;
    LightGlueMatcher matcher_;
};

}  // namespace feature_extraction
