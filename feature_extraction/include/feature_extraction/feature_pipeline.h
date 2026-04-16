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

    /// Convenience: detect + match a pair of images.
    MatchResult detect_and_match(const cv::Mat& image0_bgr,
                                 const cv::Mat& image1_bgr);

private:
    AlikedDetector detector_;
    LightGlueMatcher matcher_;
};

}  // namespace feature_extraction
