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

int LightGlueMatcher::max_keypoints() const {
    const auto shape = engine_.max_profile_shape("kpts0");
    return (shape.size() >= 2) ? shape[1] : 0;
}

int LightGlueMatcher::max_batch_size() const {
    const auto shape = engine_.max_profile_shape("kpts0");
    return (shape.size() >= 1) ? shape[0] : 1;
}

std::vector<MatchResult> LightGlueMatcher::match_gpu_batch(
    const std::vector<BatchMatchInput>& inputs) {
    const int B = static_cast<int>(inputs.size());
    if (B == 0) return {};

    // Fast path: skip padding overhead for a single pair.
    if (B == 1) {
        auto r = match_gpu(inputs[0].d_kpts0, inputs[0].d_desc0, inputs[0].N0,
                           inputs[0].d_kpts1, inputs[0].d_desc1, inputs[0].N1);
        return {std::move(r)};
    }

    const int D = config_.descriptor_dim;

    // Compute per-batch max N (determines padding extent).
    int N_max0 = 0, N_max1 = 0;
    std::vector<int> N0_arr(B), N1_arr(B);
    for (int b = 0; b < B; ++b) {
        N0_arr[b] = inputs[b].N0;
        N1_arr[b] = inputs[b].N1;
        N_max0 = std::max(N_max0, inputs[b].N0);
        N_max1 = std::max(N_max1, inputs[b].N1);
    }

    // Resize scratch buffers (no-op if already large enough).
    const size_t kpts0_bytes = static_cast<size_t>(B) * N_max0 * 2 * sizeof(float);
    const size_t desc0_bytes = static_cast<size_t>(B) * N_max0 * D * sizeof(float);
    const size_t kpts1_bytes = static_cast<size_t>(B) * N_max1 * 2 * sizeof(float);
    const size_t desc1_bytes = static_cast<size_t>(B) * N_max1 * D * sizeof(float);
    batch_kpts0_.resize(kpts0_bytes);
    batch_desc0_.resize(desc0_bytes);
    batch_kpts1_.resize(kpts1_bytes);
    batch_desc1_.resize(desc1_bytes);

    // Zero-fill so padding positions don't carry garbage values.
    cudaMemsetAsync(batch_kpts0_.ptr, 0, kpts0_bytes, stream_);
    cudaMemsetAsync(batch_desc0_.ptr, 0, desc0_bytes, stream_);
    cudaMemsetAsync(batch_kpts1_.ptr, 0, kpts1_bytes, stream_);
    cudaMemsetAsync(batch_desc1_.ptr, 0, desc1_bytes, stream_);

    // D2D copy each pair's features into its row in the packed batch buffer.
    for (int b = 0; b < B; ++b) {
        auto* kp0 = static_cast<float*>(batch_kpts0_.ptr) + b * N_max0 * 2;
        auto* dc0 = static_cast<float*>(batch_desc0_.ptr) + b * N_max0 * D;
        auto* kp1 = static_cast<float*>(batch_kpts1_.ptr) + b * N_max1 * 2;
        auto* dc1 = static_cast<float*>(batch_desc1_.ptr) + b * N_max1 * D;
        cudaMemcpyAsync(kp0, inputs[b].d_kpts0,
                        N0_arr[b] * 2 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(dc0, inputs[b].d_desc0,
                        N0_arr[b] * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(kp1, inputs[b].d_kpts1,
                        N1_arr[b] * 2 * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(dc1, inputs[b].d_desc1,
                        N1_arr[b] * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    engine_.set_input_shape("kpts0", {B, N_max0, 2});
    engine_.set_input_shape("desc0", {B, N_max0, D});
    engine_.set_input_shape("kpts1", {B, N_max1, 2});
    engine_.set_input_shape("desc1", {B, N_max1, D});

    engine_.set_device_input("kpts0", batch_kpts0_.ptr);
    engine_.set_device_input("desc0", batch_desc0_.ptr);
    engine_.set_device_input("kpts1", batch_kpts1_.ptr);
    engine_.set_device_input("desc1", batch_desc1_.ptr);

    std::vector<MatchResult> results(B);
    if (run_and_decode_batch(
            B, N0_arr.data(), N_max0, N1_arr.data(), N_max1, results)) {
        engine_.clear_device_inputs();
        return results;
    }
    engine_.clear_device_inputs();

    // Batch inference failed (likely GPU OOM at large B).
    // Fall back to serial single-pair inference which uses far less
    // activation memory (proportional to B=1 instead of B_max).
    spdlog::warn(
        "LightGlue batch B={} failed, falling back to serial inference", B);
    for (int i = 0; i < B; ++i) {
        results[i] = match_gpu(
            inputs[i].d_kpts0, inputs[i].d_desc0, inputs[i].N0,
            inputs[i].d_kpts1, inputs[i].d_desc1, inputs[i].N1);
    }
    return results;
}

bool LightGlueMatcher::run_and_decode_batch(
    int B, const int* N0_arr, int N_max0,
    const int* N1_arr, int N_max1,
    std::vector<MatchResult>& results) {
    results.assign(B, MatchResult{});

    if (!engine_.infer(stream_)) {
        spdlog::error("LightGlue batch inference failed (B={})", B);
        return false;
    }
    cudaStreamSynchronize(stream_);

    // Flat host buffers for [B, N_max0] outputs.
    const int total0 = B * N_max0;
    std::vector<int64_t> matches0_flat(total0);
    std::vector<float>   mscores0_flat(total0);
    engine_.get_output("matches0", matches0_flat.data(),
                       total0 * sizeof(int64_t));
    engine_.get_output("mscores0", mscores0_flat.data(),
                       total0 * sizeof(float));

    struct ScoredMatch { int i, j; float score; };

    for (int b = 0; b < B; ++b) {
        const int N0b  = N0_arr[b];
        const int N1b  = N1_arr[b];
        const int base = b * N_max0;

        std::vector<ScoredMatch> valid;
        valid.reserve(N0b);
        for (int i = 0; i < N0b; ++i) {
            int j = static_cast<int>(matches0_flat[base + i]);
            if (j >= 0 && j < N1b) {
                valid.push_back({i, j, mscores0_flat[base + i]});
            }
        }

        if (config_.max_matches > 0 &&
            static_cast<int>(valid.size()) > config_.max_matches) {
            auto cmp = [](const ScoredMatch& a, const ScoredMatch& b) {
                return a.score > b.score;
            };
            auto cut = valid.begin() + config_.max_matches;
            std::nth_element(valid.begin(), cut, valid.end(), cmp);
            valid.resize(config_.max_matches);
            std::sort(valid.begin(), valid.end(), cmp);
        }

        results[b].matches.reserve(valid.size());
        results[b].scores.reserve(valid.size());
        for (const auto& m : valid) {
            results[b].matches.emplace_back(m.i, m.j);
            results[b].scores.push_back(m.score);
        }
        results[b].num_matches = static_cast<int>(results[b].matches.size());
    }

    return true;
}

}  // namespace sfm_phoenix
