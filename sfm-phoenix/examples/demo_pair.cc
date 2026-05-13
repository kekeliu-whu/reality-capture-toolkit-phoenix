/**
 * Demo: ALIKED feature extraction + LightGlue matching (TensorRT).
 *
 * Usage:
 *   demo_feature_matching \
 *       --aliked D:/Users/rick/Downloads/aliked.onnx \
 *       --lightglue engines/lightglue.engine \
 *       --image0 path/to/image0.jpg \
 *       --image1 path/to/image1.jpg \
 *       --output matches.jpg
 */

#include "sfm_phoenix/pipelines/feature_pipeline.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace sfm_phoenix;

static cv::Mat draw_matches(const cv::Mat& img0, const cv::Mat& img1,
                            const AlikedResult& r0, const AlikedResult& r1,
                            const MatchResult& matches) {
    int H = std::max(img0.rows, img1.rows);
    int W = img0.cols + img1.cols;
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    img0.copyTo(canvas(cv::Rect(0, 0, img0.cols, img0.rows)));
    img1.copyTo(canvas(cv::Rect(img0.cols, 0, img1.cols, img1.rows)));

    // Scale factor to convert padded-space keypoints to original image coords
    float inv_s0 = 1.0f / r0.scale;
    float inv_s1 = 1.0f / r1.scale;

    // Draw keypoints
    for (auto& kp : r0.keypoints) {
        cv::Point2f p(kp.x * inv_s0, kp.y * inv_s0);
        cv::circle(canvas, p, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
    for (auto& kp : r1.keypoints) {
        cv::Point2f p(kp.x * inv_s1 + img0.cols, kp.y * inv_s1);
        cv::circle(canvas, p, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }

    // Draw matches
    for (int i = 0; i < matches.num_matches; ++i) {
        auto [i0, i1] = matches.matches[i];
        cv::Point2f p0(r0.keypoints[i0].x * inv_s0, r0.keypoints[i0].y * inv_s0);
        cv::Point2f p1(r1.keypoints[i1].x * inv_s1 + img0.cols,
                       r1.keypoints[i1].y * inv_s1);
        cv::line(canvas, p0, p1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    return canvas;
}

// Run detection + matching + optional dump for one pair.
static void process_pair(FeaturePipeline& pipeline,
                         const std::string& image0_path,
                         const std::string& image1_path,
                         const std::string& output_path,
                         const std::string& dump_dir) {
    cv::Mat img0 = cv::imread(image0_path);
    cv::Mat img1 = cv::imread(image1_path);
    if (img0.empty() || img1.empty()) {
        std::cerr << "Cannot read images: " << image0_path << ", "
                  << image1_path << std::endl;
        return;
    }

    bool need_dump = !dump_dir.empty();

    if (need_dump) {
        // Use the original CPU-path for dump (needs host-side data)
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r0 = pipeline.detect(img0);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto r1 = pipeline.detect(img1);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto mr = pipeline.match(r0, r1);
        auto t3 = std::chrono::high_resolution_clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        std::cout << "Image 0: " << r0.num_keypoints << " keypoints ("
                  << ms(t0, t1) << " ms)" << std::endl;
        std::cout << "Image 1: " << r1.num_keypoints << " keypoints ("
                  << ms(t1, t2) << " ms)" << std::endl;
        std::cout << "Matches: " << mr.num_matches << " (" << ms(t2, t3)
                  << " ms)" << std::endl;

        // Dump binary data
        auto write_bin = [](const std::string& path, const void* data,
                            size_t bytes) {
            std::ofstream f(path, std::ios::binary);
            f.write(static_cast<const char*>(data), bytes);
        };
        auto dump_kpts = [](const AlikedResult& r) {
            std::vector<float> kpts(r.num_keypoints * 2);
            for (int i = 0; i < r.num_keypoints; ++i) {
                kpts[i * 2 + 0] = r.keypoints[i].x;
                kpts[i * 2 + 1] = r.keypoints[i].y;
            }
            return kpts;
        };
        auto kpts0 = dump_kpts(r0);
        auto kpts1 = dump_kpts(r1);
        write_bin(dump_dir + "/kpts0.bin", kpts0.data(),
                  kpts0.size() * sizeof(float));
        write_bin(dump_dir + "/kpts1.bin", kpts1.data(),
                  kpts1.size() * sizeof(float));
        write_bin(dump_dir + "/scores0.bin", r0.scores.data(),
                  r0.scores.size() * sizeof(float));
        write_bin(dump_dir + "/scores1.bin", r1.scores.data(),
                  r1.scores.size() * sizeof(float));
        write_bin(dump_dir + "/desc0.bin", r0.descriptors.data(),
                  r0.descriptors.size() * sizeof(float));
        write_bin(dump_dir + "/desc1.bin", r1.descriptors.data(),
                  r1.descriptors.size() * sizeof(float));
        std::vector<int32_t> match_pairs(mr.num_matches * 2);
        for (int i = 0; i < mr.num_matches; ++i) {
            match_pairs[i * 2 + 0] = mr.matches[i].first;
            match_pairs[i * 2 + 1] = mr.matches[i].second;
        }
        write_bin(dump_dir + "/matches.bin", match_pairs.data(),
                  match_pairs.size() * sizeof(int32_t));
        write_bin(dump_dir + "/mscores.bin", mr.scores.data(),
                  mr.scores.size() * sizeof(float));
        {
            std::ofstream f(dump_dir + "/meta.txt");
            f << "n0=" << r0.num_keypoints << "\n"
              << "n1=" << r1.num_keypoints << "\n"
              << "matches=" << mr.num_matches << "\n"
              << "scale0=" << r0.scale << "\n"
              << "scale1=" << r1.scale << "\n";
        }

        if (!output_path.empty()) {
            cv::Mat vis = draw_matches(img0, img1, r0, r1, mr);
            cv::imwrite(output_path, vis);
            std::cout << "Saved: " << output_path << std::endl;
        }
    } else {
        // GPU-optimized path: no host round-trip between detect and match
        auto t0 = std::chrono::high_resolution_clock::now();
        auto pr = pipeline.detect_and_match_gpu(img0, img1);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        std::cout << "Image 0: " << pr.det0.num_keypoints << " keypoints"
                  << std::endl;
        std::cout << "Image 1: " << pr.det1.num_keypoints << " keypoints"
                  << std::endl;
        std::cout << "Matches: " << pr.matches.num_matches << " ("
                  << ms(t0, t1) << " ms total)" << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string aliked_path = R"(D:\Users\rick\Downloads\aliked.onnx)";
    std::string lightglue_path;
    std::string image0_path, image1_path, output_path = "matches.jpg";
    std::string dump_dir;
    std::string pair_list;  // batch mode: file with lines "img0 img1 dump_dir"
    int max_edge = 1600;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--aliked" && i + 1 < argc)
            aliked_path = argv[++i];
        else if (arg == "--lightglue" && i + 1 < argc)
            lightglue_path = argv[++i];
        else if (arg == "--image0" && i + 1 < argc)
            image0_path = argv[++i];
        else if (arg == "--image1" && i + 1 < argc)
            image1_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            output_path = argv[++i];
        else if (arg == "--dump" && i + 1 < argc)
            dump_dir = argv[++i];
        else if (arg == "--pair-list" && i + 1 < argc)
            pair_list = argv[++i];
        else if (arg == "--max-edge" && i + 1 < argc)
            max_edge = std::stoi(argv[++i]);
    }

    bool single_mode = !image0_path.empty() && !image1_path.empty();
    bool batch_mode = !pair_list.empty();

    if (aliked_path.empty() || lightglue_path.empty() ||
        (!single_mode && !batch_mode)) {
        std::cerr
            << "Usage:\n"
            << "  Single pair:\n"
            << "    demo_feature_matching --aliked <.onnx/.engine>\n"
            << "      --lightglue <.engine> --image0 <path> --image1 <path>\n"
            << "      [--output <path>] [--dump <dir>]\n"
            << "  Batch mode:\n"
            << "    demo_feature_matching --aliked <.onnx/.engine>\n"
            << "      --lightglue <.engine> --pair-list <file>\n"
            << "    pair-list format: image0_path image1_path dump_dir\n"
            << std::endl;
        return 1;
    }

    // Init pipeline
    PipelineConfig config;
    config.aliked.full_model_path = aliked_path;
    config.aliked.dkd.top_k = 5000;
    config.aliked.dkd.scores_th = 0.2f;
    config.aliked.max_edge = max_edge;
    config.lightglue.engine_path = lightglue_path;

    FeaturePipeline pipeline;
    if (!pipeline.init(config)) {
        std::cerr << "Pipeline init failed" << std::endl;
        return 1;
    }

    if (batch_mode) {
        // Read pair list
        std::ifstream pf(pair_list);
        if (!pf.is_open()) {
            std::cerr << "Cannot open pair list: " << pair_list << std::endl;
            return 1;
        }

        // Parse all pairs first for detection caching
        struct PairEntry {
            std::string p0, p1, dd;
        };
        std::vector<PairEntry> entries;
        std::string line;
        while (std::getline(pf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            PairEntry e;
            iss >> e.p0 >> e.p1;
            iss >> e.dd;
            entries.push_back(std::move(e));
        }

        // Batch processing with detection caching.
        // If consecutive pairs share an image (pair[i].p1 == pair[i+1].p0),
        // reuse the detection result instead of recomputing.
        GpuDetectResult cached_det;
        std::string cached_path;
        for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
            auto& e = entries[i];
            std::cout << "--- Pair " << i << ": " << e.p0 << " <-> "
                      << e.p1 << " ---" << std::endl;

            bool use_dump = !e.dd.empty();
            if (use_dump) {
                // Dump path: no caching (needs CPU-side AlikedResult)
                process_pair(pipeline, e.p0, e.p1, "", e.dd);
                cached_path.clear();
            } else {
                // GPU path with detection caching
                cv::Mat img1 = cv::imread(e.p1);
                if (img1.empty()) {
                    std::cerr << "Cannot read: " << e.p1 << std::endl;
                    cached_path.clear();
                    continue;
                }

                PairResult pr;
                auto t0 = std::chrono::high_resolution_clock::now();
                if (!cached_path.empty() && cached_path == e.p0) {
                    // Reuse cached detection for image0
                    pr = pipeline.detect_and_match_gpu(
                        std::move(cached_det), img1);
                } else {
                    cv::Mat img0 = cv::imread(e.p0);
                    if (img0.empty()) {
                        std::cerr << "Cannot read: " << e.p0 << std::endl;
                        cached_path.clear();
                        continue;
                    }
                    pr = pipeline.detect_and_match_gpu(img0, img1);
                }
                auto t1 = std::chrono::high_resolution_clock::now();

                auto ms = [](auto a, auto b) {
                    return std::chrono::duration<double, std::milli>(
                               b - a)
                        .count();
                };
                std::cout << "Image 0: " << pr.det0.num_keypoints
                          << " keypoints" << std::endl;
                std::cout << "Image 1: " << pr.det1.num_keypoints
                          << " keypoints" << std::endl;
                std::cout << "Matches: " << pr.matches.num_matches << " ("
                          << ms(t0, t1) << " ms total)" << std::endl;

                // Cache det1 for potential reuse as det0 of the next pair
                cached_det = std::move(pr.det1);
                cached_path = e.p1;
            }
        }
    } else {
        process_pair(pipeline, image0_path, image1_path, output_path, dump_dir);
    }

    return 0;
}
