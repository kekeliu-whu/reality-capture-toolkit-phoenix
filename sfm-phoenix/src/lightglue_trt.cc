#include "feature_extraction/lightglue_trt.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace feature_extraction {

LightGlueMatcher::~LightGlueMatcher() {
    if (stream_) cudaStreamDestroy(stream_);
}

bool LightGlueMatcher::init(const LightGlueConfig& config) {
    config_ = config;

    if (!engine_.load(config.engine_path)) {
        std::cerr << "Failed to load LightGlue engine" << std::endl;
        return false;
    }

    cudaStreamCreate(&stream_);
    return true;
}

MatchResult LightGlueMatcher::run_and_decode(int N0, int N1) {
    MatchResult result;

    if (!engine_.infer(stream_)) {
        std::cerr << "LightGlue inference failed" << std::endl;
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

    // Extract valid matches
    for (int i = 0; i < N0; ++i) {
        int j = static_cast<int>(matches0_i64[i]);
        if (j >= 0 && j < N1) {
            result.matches.emplace_back(i, j);
            result.scores.push_back(mscores0_raw[i]);
        }
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

}  // namespace feature_extraction
