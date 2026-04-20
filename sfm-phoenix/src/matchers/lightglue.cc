#include "sfm_phoenix/matchers/lightglue.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <vector>

namespace sfm_phoenix {

LightGlueMatcher::~LightGlueMatcher() {
    if (stream_) cudaStreamDestroy(stream_);
}

bool LightGlueMatcher::init(const LightGlueConfig& config) {
    config_ = config;

    if (!engine_.load(config.engine_path)) {
        spdlog::error("Failed to load LightGlue engine: {}",
                      config.engine_path);
        return false;
    }

    const auto kpts0_max_shape = engine_.max_profile_shape("kpts0");
    const auto kpts1_max_shape = engine_.max_profile_shape("kpts1");
    const int supported_keypoints =
        (kpts0_max_shape.size() >= 2 && kpts1_max_shape.size() >= 2)
            ? std::min(kpts0_max_shape[1], kpts1_max_shape[1])
            : 0;
    if (supported_keypoints > 0 && config_.max_matches > supported_keypoints) {
        spdlog::error(
            "Configured Phoenix max_matches={} exceeds LightGlue engine "
            "profile max {}. Rebuild {} with a larger keypoint profile.",
            config_.max_matches,
            supported_keypoints,
            config.engine_path);
        return false;
    }

    cudaStreamCreate(&stream_);
    return true;
}

MatchResult LightGlueMatcher::run_and_decode(int N0, int N1) {
    MatchResult result;

    if (!engine_.infer(stream_)) {
        spdlog::error("LightGlue inference failed");
        return result;
    }
    cudaStreamSynchronize(stream_);

    // Read outputs
    std::vector<int64_t> matches0_i64(N0);
    std::vector<float> mscores0_raw(N0);

    engine_.get_output("matches0", matches0_i64.data(),
                       N0 * sizeof(int64_t));
    engine_.get_output("mscores0", mscores0_raw.data(),
                       N0 * sizeof(float));

    struct ScoredMatch {
        int index0;
        int index1;
        float score;
    };
    std::vector<ScoredMatch> valid_matches;
    valid_matches.reserve(N0);

    // Extract valid matches
    for (int i = 0; i < N0; ++i) {
        int j = static_cast<int>(matches0_i64[i]);
        if (j >= 0 && j < N1) {
            valid_matches.push_back({i, j, mscores0_raw[i]});
        }
    }

    if (config_.max_matches > 0 &&
        static_cast<int>(valid_matches.size()) > config_.max_matches) {
        auto by_score_desc = [](const ScoredMatch& lhs, const ScoredMatch& rhs) {
            return lhs.score > rhs.score;
        };
        auto cutoff = valid_matches.begin() + config_.max_matches;
        std::nth_element(valid_matches.begin(),
                         cutoff,
                         valid_matches.end(),
                         by_score_desc);
        valid_matches.resize(config_.max_matches);
        std::sort(valid_matches.begin(), valid_matches.end(), by_score_desc);
    }

    result.matches.reserve(valid_matches.size());
    result.scores.reserve(valid_matches.size());
    for (const auto& match : valid_matches) {
        result.matches.emplace_back(match.index0, match.index1);
        result.scores.push_back(match.score);
    }

    result.num_matches = static_cast<int>(result.matches.size());
    return result;
}

MatchResult LightGlueMatcher::match(const float* kpts0, const float* desc0,
                                    int N0, const float* kpts1,
                                    const float* desc1, int N1) {
    if (N0 <= 0 || N1 <= 0) return {};

    int D = config_.descriptor_dim;

    engine_.set_input_shape("kpts0", {1, N0, 2});
    engine_.set_input_shape("desc0", {1, N0, D});
    engine_.set_input_shape("kpts1", {1, N1, 2});
    engine_.set_input_shape("desc1", {1, N1, D});

    // H2D upload
    engine_.set_input("kpts0", kpts0, N0 * 2 * sizeof(float), stream_);
    engine_.set_input("desc0", desc0, N0 * D * sizeof(float), stream_);
    engine_.set_input("kpts1", kpts1, N1 * 2 * sizeof(float), stream_);
    engine_.set_input("desc1", desc1, N1 * D * sizeof(float), stream_);

    return run_and_decode(N0, N1);
}

MatchResult LightGlueMatcher::match_gpu(const void* d_kpts0,
                                        const void* d_desc0, int N0,
                                        const void* d_kpts1,
                                        const void* d_desc1, int N1) {
    if (N0 <= 0 || N1 <= 0) return {};

    int D = config_.descriptor_dim;

    engine_.set_input_shape("kpts0", {1, N0, 2});
    engine_.set_input_shape("desc0", {1, N0, D});
    engine_.set_input_shape("kpts1", {1, N1, 2});
    engine_.set_input_shape("desc1", {1, N1, D});

    // Use device pointers directly — no H2D copy!
    engine_.set_device_input("kpts0", const_cast<void*>(d_kpts0));
    engine_.set_device_input("desc0", const_cast<void*>(d_desc0));
    engine_.set_device_input("kpts1", const_cast<void*>(d_kpts1));
    engine_.set_device_input("desc1", const_cast<void*>(d_desc1));

    auto result = run_and_decode(N0, N1);
    engine_.clear_device_inputs();
    return result;
}

}  // namespace sfm_phoenix
