/**
 * Demo: ALIKED feature extraction + LightGlue matching (TensorRT).
 *
 * Usage:
 *   demo_feature_matching \
 *       --backbone engines/aliked_backbone.engine \
 *       --sddh engines/aliked_sddh.engine \
 *       --lightglue engines/lightglue.engine \
 *       --image0 path/to/image0.jpg \
 *       --image1 path/to/image1.jpg \
 *       --output matches.jpg
 */

#include "feature_extraction/feature_pipeline.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <string>

using namespace feature_extraction;

static cv::Mat draw_matches(const cv::Mat& img0, const cv::Mat& img1,
                            const AlikedResult& r0, const AlikedResult& r1,
                            const MatchResult& matches) {
    int H = std::max(img0.rows, img1.rows);
    int W = img0.cols + img1.cols;
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    img0.copyTo(canvas(cv::Rect(0, 0, img0.cols, img0.rows)));
    img1.copyTo(canvas(cv::Rect(img0.cols, 0, img1.cols, img1.rows)));

    // Draw keypoints
    for (auto& kp : r0.keypoints)
        cv::circle(canvas, kp, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    for (auto& kp : r1.keypoints) {
        cv::Point2f shifted(kp.x + img0.cols, kp.y);
        cv::circle(canvas, shifted, 2, cv::Scalar(0, 0, 255), -1,
                   cv::LINE_AA);
    }

    // Draw matches
    for (int i = 0; i < matches.num_matches; ++i) {
        auto [i0, i1] = matches.matches[i];
        cv::Point2f p0 = r0.keypoints[i0];
        cv::Point2f p1(r1.keypoints[i1].x + img0.cols, r1.keypoints[i1].y);
        cv::line(canvas, p0, p1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    return canvas;
}

int main(int argc, char** argv) {
    // Simple argument parsing
    std::string backbone_path, sddh_path, lightglue_path;
    std::string image0_path, image1_path, output_path = "matches.jpg";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backbone" && i + 1 < argc)
            backbone_path = argv[++i];
        else if (arg == "--sddh" && i + 1 < argc)
            sddh_path = argv[++i];
        else if (arg == "--lightglue" && i + 1 < argc)
            lightglue_path = argv[++i];
        else if (arg == "--image0" && i + 1 < argc)
            image0_path = argv[++i];
        else if (arg == "--image1" && i + 1 < argc)
            image1_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            output_path = argv[++i];
    }

    if (backbone_path.empty() || sddh_path.empty() || lightglue_path.empty() ||
        image0_path.empty() || image1_path.empty()) {
        std::cerr
            << "Usage: demo_feature_matching --backbone <.engine> --sddh "
               "<.engine> --lightglue <.engine> --image0 <path> --image1 "
               "<path> [--output <path>]"
            << std::endl;
        return 1;
    }

    // Init pipeline
    PipelineConfig config;
    config.aliked.backbone_engine = backbone_path;
    config.aliked.sddh_engine = sddh_path;
    config.aliked.dkd.top_k = 5000;
    config.aliked.dkd.scores_th = 0.2f;
    config.aliked.max_edge = 1600;
    config.lightglue.engine_path = lightglue_path;

    FeaturePipeline pipeline;
    if (!pipeline.init(config)) {
        std::cerr << "Pipeline init failed" << std::endl;
        return 1;
    }

    // Load images
    cv::Mat img0 = cv::imread(image0_path);
    cv::Mat img1 = cv::imread(image1_path);
    if (img0.empty() || img1.empty()) {
        std::cerr << "Cannot read images" << std::endl;
        return 1;
    }

    // Detect features
    auto t0 = std::chrono::high_resolution_clock::now();
    auto r0 = pipeline.detect(img0);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto r1 = pipeline.detect(img1);
    auto t2 = std::chrono::high_resolution_clock::now();

    // Match
    auto mr = pipeline.match(r0, r1);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    std::cout << "Image 0: " << r0.num_keypoints << " keypoints ("
              << ms(t0, t1) << " ms)" << std::endl;
    std::cout << "Image 1: " << r1.num_keypoints << " keypoints ("
              << ms(t1, t2) << " ms)" << std::endl;
    std::cout << "Matches: " << mr.num_matches << " (" << ms(t2, t3) << " ms)"
              << std::endl;

    // Visualise
    cv::Mat vis = draw_matches(img0, img1, r0, r1, mr);
    cv::imwrite(output_path, vis);
    std::cout << "Saved: " << output_path << std::endl;

    return 0;
}
