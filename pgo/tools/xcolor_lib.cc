
#include "xcolor_lib.h"

#include "migration/string.h"
#include "migration/utils.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/BufferReader.hpp>
#include <spdlog/spdlog.h>

namespace xcolor {

namespace {

void PrintProgress(double progress) {
  static int last_progress = -1;
  const int progress_percent = std::clamp(
      static_cast<int>(std::lround(progress)), 0, 100);
  if (progress_percent == last_progress) {
    return;
  }
  last_progress = progress_percent;
  std::cout << "Progress " << progress_percent << "%" << std::endl;
}

cv::Mat DecodeDepthMeters(const cv::Mat& depth_png) {
  if (depth_png.empty()) {
    return {};
  }
  if (depth_png.type() != CV_8UC4) {
    throw std::runtime_error("Depth PNG must be CV_8UC4 RGBA/BGRA encoded");
  }

  cv::Mat depth_rgba;
  cv::cvtColor(depth_png, depth_rgba, cv::COLOR_BGRA2RGBA);
  cv::Mat depth_meters(depth_png.rows, depth_png.cols, CV_32FC1, 0.0f);

  for (int y = 0; y < depth_rgba.rows; ++y) {
    for (int x = 0; x < depth_rgba.cols; ++x) {
      const cv::Vec4b rgba = depth_rgba.at<cv::Vec4b>(y, x);
      const uint32_t depth_mm = static_cast<uint32_t>(rgba[0]) |
                                (static_cast<uint32_t>(rgba[1]) << 8) |
                                (static_cast<uint32_t>(rgba[2]) << 16) |
                                (static_cast<uint32_t>(rgba[3]) << 24);
      depth_meters.at<float>(y, x) = static_cast<float>(depth_mm) * 0.001f;
    }
  }

  return depth_meters;
}

std::optional<cv::Mat> LoadDepthMeters(const Image& image) {
  if (!image.has_depth) {
    return std::nullopt;
  }

  cv::Mat depth_png = cv::imread(image.depth_filename, cv::IMREAD_UNCHANGED);
  if (depth_png.empty()) {
    spdlog::warn("Failed to read depth image: {}", image.depth_filename);
    return std::nullopt;
  }

  return DecodeDepthMeters(depth_png);
}

cv::Mat LoadMask(const Image& image) {
  if (!image.has_mask) {
    return {};
  }

  cv::Mat mask = cv::imread(image.mask_filename, cv::IMREAD_GRAYSCALE);
  if (mask.empty()) {
    spdlog::warn("Failed to read mask image: {}", image.mask_filename);
  }
  return mask;
}

std::optional<Eigen::Vector2d> ProjectDepthPixel(const DepthIntrinsics& intrinsics,
                                                 const Eigen::Vector3d& pt_in_cam) {
  if (pt_in_cam.z() <= 0.0) {
    return std::nullopt;
  }

  const double pixel_x = intrinsics.fx * pt_in_cam.x() / pt_in_cam.z() +
                         intrinsics.cx;
  const double pixel_y = intrinsics.fy * pt_in_cam.y() / pt_in_cam.z() +
                         intrinsics.cy;
  if (pixel_x < 0.0 || pixel_x >= intrinsics.width || pixel_y < 0.0 ||
      pixel_y >= intrinsics.height) {
    return std::nullopt;
  }

  return Eigen::Vector2d(pixel_x, pixel_y);
}

bool IsDepthVisible(const DepthIntrinsics& depth_intrinsics,
                    const Eigen::Vector3d& pt_in_cam,
                    const cv::Mat& depth_meters) {
  const auto depth_pixel_opt = ProjectDepthPixel(depth_intrinsics, pt_in_cam);
  if (!depth_pixel_opt.has_value()) {
    return false;
  }

  const Eigen::Vector2d depth_pixel = depth_pixel_opt.value();
  const int pixel_x = std::clamp(
      static_cast<int>(std::llround(depth_pixel.x() - 0.5)),
      0,
      depth_meters.cols - 1);
  const int pixel_y = std::clamp(
      static_cast<int>(std::llround(depth_pixel.y() - 0.5)),
      0,
      depth_meters.rows - 1);
  const float depth_value = depth_meters.at<float>(pixel_y, pixel_x);
  if (!(depth_value > 0.0f)) {
    return false;
  }

  return std::abs(static_cast<double>(depth_value) - pt_in_cam.z()) <=
         kDepthVisibilityTolerance;
}

DepthIntrinsics ScaleDepthIntrinsicsToSize(const DepthIntrinsics& intrinsics,
                                           int width,
                                           int height) {
  if (intrinsics.width <= 0 || intrinsics.height <= 0 || width <= 0 ||
      height <= 0) {
    throw std::invalid_argument("Depth intrinsics sizes must be positive");
  }

  const double scale_x =
      static_cast<double>(width) / static_cast<double>(intrinsics.width);
  const double scale_y =
      static_cast<double>(height) / static_cast<double>(intrinsics.height);

  DepthIntrinsics scaled = intrinsics;
  scaled.width = width;
  scaled.height = height;
  scaled.fx *= scale_x;
  scaled.fy *= scale_y;
  scaled.cx *= scale_x;
  scaled.cy *= scale_y;
  return scaled;
}

}  // namespace

bool IsMaskPixelAllowed(const cv::Mat& mask, const Eigen::Vector2d& pixel) {
  if (mask.empty()) {
    return true;
  }

  const int pixel_x = static_cast<int>(pixel.x());
  const int pixel_y = static_cast<int>(pixel.y());
  if (pixel_x < 0 || pixel_x >= mask.cols || pixel_y < 0 ||
      pixel_y >= mask.rows) {
    return false;
  }

  return mask.at<uint8_t>(pixel_y, pixel_x) == 0;
}

DepthIntrinsics DepthIntrinsicsFromCamera(const colmap::Camera& camera) {
  DepthIntrinsics intrinsics;
  intrinsics.width = static_cast<int>(camera.width);
  intrinsics.height = static_cast<int>(camera.height);
  intrinsics.fx = camera.FocalLengthX();
  intrinsics.fy = camera.FocalLengthY();
  intrinsics.cx = camera.PrincipalPointX();
  intrinsics.cy = camera.PrincipalPointY();
  return intrinsics;
}

std::optional<std::pair<double, cv::Vec3b>> ComputeColorCandidates(
    const pcl::PointXYZRGB& point,
    const cv::Mat& image,
    const cv::Mat& mask,
    const cv::Mat& depth_meters,
    const DepthIntrinsics& depth_intrinsics,
    const colmap::Camera& camera,
    const colmap::Rigid3d& pose) {
  Eigen::Vector3d pt_in_cam = pose * point.getVector3fMap().cast<double>();
  if (pt_in_cam.z() < 0) {
    return std::nullopt;  // Point is behind camera
  }

  auto pixel_opt = camera.ImgFromCam(pt_in_cam / pt_in_cam.z());
  if (!pixel_opt.has_value()) {
    return std::nullopt;
  }

  auto pixel = pixel_opt.value();
  if (pixel.x() < 0 || pixel.x() >= image.cols || pixel.y() < 0 ||
      pixel.y() >= image.rows) {
    return std::nullopt;  // Pixel out of image bounds
  }

  if (!IsMaskPixelAllowed(mask, pixel)) {
    return std::nullopt;
  }

  if (!IsDepthVisible(depth_intrinsics, pt_in_cam, depth_meters)) {
    return std::nullopt;
  }

  cv::Vec3b color = image.at<cv::Vec3b>(pixel.y(), pixel.x());
  double distance = pt_in_cam.norm();
  return std::make_pair(distance, color);
}

cv::Vec3b ComputeMedianColor(
    const std::vector<std::pair<double, cv::Vec3b>>& color_candidates,
    int inlier_threshold) {
  cv::Vec3b result = {0, 0, 0};

  for (int k = 0; k < 3; ++k) {
    std::vector<uint8_t> color_channel_candidates;
    color_channel_candidates.reserve(color_candidates.size());

    for (const auto& candidate : color_candidates) {
      color_channel_candidates.push_back(candidate.second[k]);
    }

    int mid_idx = color_channel_candidates.size() / 2;
    std::nth_element(color_channel_candidates.begin(),
                     color_channel_candidates.begin() + mid_idx,
                     color_channel_candidates.end());
    uint8_t mid_color = color_channel_candidates[mid_idx];

    int color_sum = 0;
    int color_num = 0;
    for (uint8_t color_val : color_channel_candidates) {
      if (std::abs((int)color_val - (int)mid_color) < inlier_threshold) {
        color_sum += color_val;
        color_num++;
      }
    }

    result[k] = color_num > 0 ? color_sum / color_num : mid_color;
  }

  return result;
}

cv::Vec3b ComputeMeanColor(
    const std::vector<std::pair<double, cv::Vec3b>>& color_candidates) {
  cv::Vec3i color_sum = {0, 0, 0};
  for (const auto& candidate : color_candidates) {
    color_sum += candidate.second;
  }
  return {static_cast<uint8_t>(color_sum[0] / (int)color_candidates.size()),
          static_cast<uint8_t>(color_sum[1] / (int)color_candidates.size()),
          static_cast<uint8_t>(color_sum[2] / (int)color_candidates.size())};
}

void WritePointCloudToLAS(const std::string& filename,
                          const pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
  spdlog::info("Saving point cloud to {}", filename);

  pdal::PointTable table;
  table.layout()->registerDim(pdal::Dimension::Id::X);
  table.layout()->registerDim(pdal::Dimension::Id::Y);
  table.layout()->registerDim(pdal::Dimension::Id::Z);
  table.layout()->registerDim(pdal::Dimension::Id::Red);
  table.layout()->registerDim(pdal::Dimension::Id::Green);
  table.layout()->registerDim(pdal::Dimension::Id::Blue);

  pdal::PointViewPtr view = std::make_shared<pdal::PointView>(table);
  for (size_t i = 0; i < cloud.size(); ++i) {
    view->setField(pdal::Dimension::Id::X, i, cloud[i].x);
    view->setField(pdal::Dimension::Id::Y, i, cloud[i].y);
    view->setField(pdal::Dimension::Id::Z, i, cloud[i].z);
    view->setField(pdal::Dimension::Id::Red,   i, cloud[i].r);
    view->setField(pdal::Dimension::Id::Green, i, cloud[i].g);
    view->setField(pdal::Dimension::Id::Blue,  i, cloud[i].b);
  }

  pdal::BufferReader reader;
  reader.addView(view);

  pdal::StageFactory factory;
  pdal::Stage* writer = factory.createStage("writers.las");
  pdal::Options opts;
  opts.add(pdal::Option("filename", PlatformToUTF8(filename)));
  opts.add(pdal::Option("scale_x", 0.0001));
  opts.add(pdal::Option("scale_y", 0.0001));
  opts.add(pdal::Option("scale_z", 0.0001));
  writer->setOptions(opts);
  writer->setInput(reader);
  writer->prepare(table);
  writer->execute(table);
}

void PerformXColor(const std::vector<Image>& images,
                   pcl::PointCloud<pcl::PointXYZRGB>& cloud_rgb,
                   std::string output_path,
                   int color_candidate_limit) {
  const int candidate_limit = std::max(1, color_candidate_limit);
  PrintProgress(0.0);
  const std::filesystem::path output_dir(output_path);
  std::error_code error_code;
  if (!std::filesystem::exists(output_dir, error_code) &&
      !std::filesystem::create_directories(output_dir, error_code)) {
    throw std::runtime_error("Failed to create output directory: " +
                             output_dir.string() +
                             ", error: " + error_code.message());
  }
  PrintMemoryUsage();
  for (auto& p : cloud_rgb) {
    p.getBGRVector3cMap().setZero();
  }

  PrintMemoryUsage();

  // Step 1: 为每个点准备颜色候选值存储
  std::vector<PointColorCandidates> point_color_candidates(cloud_rgb.size());

  // Step 2: 逐图片处理，实时读取深度图并对整片点云做深度可见性过滤
  spdlog::info("Collecting color candidates from images...");
  for (int image_idx = 0; image_idx < images.size(); ++image_idx) {
    const auto print_image_progress = [&images, image_idx]() {
      PrintProgress(95.0 * static_cast<double>(image_idx + 1) /
                    static_cast<double>(images.size()));
    };
    const Image& image = images[image_idx];
    if (!image.has_depth) {
      spdlog::warn("Skipping image without depth image: {}", image.filename);
      print_image_progress();
      continue;
    }

    cv::Mat img_cv = cv::imread(image.filename);
    if (img_cv.empty()) {
      spdlog::warn("Failed to read image: {}", image.filename);
      print_image_progress();
      continue;
    }

    const cv::Mat mask = LoadMask(image);
    const auto depth_meters_opt = LoadDepthMeters(image);
    if (!depth_meters_opt.has_value()) {
      print_image_progress();
      continue;
    }
    const cv::Mat& depth_meters = depth_meters_opt.value();
    const DepthIntrinsics depth_intrinsics = ScaleDepthIntrinsicsToSize(
        image.depth_intrinsics, depth_meters.cols, depth_meters.rows);
    if (depth_intrinsics.width != image.depth_intrinsics.width ||
        depth_intrinsics.height != image.depth_intrinsics.height) {
      spdlog::info("Scaled depth intrinsics for {} from {}x{} to {}x{}",
                   image.filename,
                   image.depth_intrinsics.width,
                   image.depth_intrinsics.height,
                   depth_intrinsics.width,
                   depth_intrinsics.height);
    }
    const int point_count = static_cast<int>(cloud_rgb.size());

    spdlog::info("Rendering image {} / {} against {} points",
                 image_idx,
                 images.size(),
                 point_count);

#pragma omp parallel for
    for (int point_idx = 0; point_idx < point_count; ++point_idx) {
      auto color_candidate_opt = ComputeColorCandidates(
          cloud_rgb[point_idx],
          img_cv,
          mask,
          depth_meters,
          depth_intrinsics,
          image.camera,
          image.pose);

      if (color_candidate_opt) {
        auto& candidates = point_color_candidates[point_idx].candidates;
        const auto& candidate = color_candidate_opt.value();
        // 二分查找找到插入位置
        auto it = std::lower_bound(
            candidates.begin(),
            candidates.end(),
            candidate,
            [](const auto& a, const auto& b) { return a.first < b.first; });
        candidates.insert(it, candidate);

        // 如果超过限制，删除最后一个（距离最远的）
        if (static_cast<int>(candidates.size()) > candidate_limit) {
          candidates.pop_back();
        }
      }
    }
    print_image_progress();
  }

  // Step 3: 统一计算每个点的最终颜色
  spdlog::info("Computing final colors...");
  PrintProgress(96.0);
#pragma omp parallel for
  for (int point_idx = 0; point_idx < cloud_rgb.size(); ++point_idx) {
    auto& candidates = point_color_candidates[point_idx].candidates;

    if (candidates.empty()) {
      continue;
    }

    // Candidates were limited to the nearest samples while processing images.
    cv::Vec3b final_color;
    if (candidates.size() >= kMinCandidateNum) {
      final_color = ComputeMedianColor(candidates, kColorInlierThreshold);
    } else {
      final_color = ComputeMeanColor(candidates);
    }

    cloud_rgb[point_idx].b = final_color[0];
    cloud_rgb[point_idx].g = final_color[1];
    cloud_rgb[point_idx].r = final_color[2];
  }

  pcl::PointCloud<pcl::PointXYZRGB> final_colored_cloud;
  final_colored_cloud.reserve(cloud_rgb.size());
  for (int point_idx = 0; point_idx < cloud_rgb.size(); ++point_idx) {
    if (!point_color_candidates[point_idx].candidates.empty()) {
      final_colored_cloud.push_back(cloud_rgb[point_idx]);
    }
  }
  PrintProgress(98.0);
  spdlog::info("Final colored point cloud contains {} / {} points",
               final_colored_cloud.size(),
               cloud_rgb.size());
  PrintProgress(99.0);
  WritePointCloudToLAS((output_dir / "xcolor.las").string(),
                       final_colored_cloud);
  PrintProgress(100.0);
}

}  // namespace xcolor
