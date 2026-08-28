
#include "xcolor_lib.h"

#include "migration/inc_las_writer.h"
#include "migration/string.h"
#include "migration/utils.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/BufferReader.hpp>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/mman.h>
#endif

namespace xcolor {

namespace {

constexpr uint32_t kInvalidCandidateBlock =
    std::numeric_limits<uint32_t>::max();
constexpr size_t kCandidateBlocksPerCommit = 65536;
constexpr size_t kOutputPointChunkSize = 250000;

struct PackedColorCandidate {
  float distance = 0.0f;
  uint32_t color = 0;
};
static_assert(sizeof(PackedColorCandidate) == 8);

struct CandidateBlock {
  std::array<PackedColorCandidate, kColorInlierMaxNum> candidates;
};

struct PointCandidateState {
  uint32_t block_index = kInvalidCandidateBlock;
  uint8_t count = 0;
  uint8_t padding[3] = {0, 0, 0};
};
static_assert(sizeof(PointCandidateState) == 8);

uint32_t PackColor(const cv::Vec3b& color) {
  return static_cast<uint32_t>(color[0]) |
         (static_cast<uint32_t>(color[1]) << 8) |
         (static_cast<uint32_t>(color[2]) << 16);
}

cv::Vec3b UnpackColor(uint32_t color) {
  return {static_cast<uint8_t>(color & 0xff),
          static_cast<uint8_t>((color >> 8) & 0xff),
          static_cast<uint8_t>((color >> 16) & 0xff)};
}

class CandidateBlockPool {
 public:
  explicit CandidateBlockPool(size_t max_blocks) : max_blocks_(max_blocks) {
    if (max_blocks_ > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("Point cloud is too large for candidate indexing");
    }
    reserved_bytes_ = max_blocks_ * sizeof(CandidateBlock);
#ifdef _WIN32
    blocks_ = static_cast<CandidateBlock*>(
        VirtualAlloc(nullptr, reserved_bytes_, MEM_RESERVE, PAGE_NOACCESS));
    if (blocks_ == nullptr) {
      throw std::bad_alloc();
    }
#else
    blocks_ = static_cast<CandidateBlock*>(
        mmap(nullptr, reserved_bytes_, PROT_NONE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (blocks_ == MAP_FAILED) {
      blocks_ = nullptr;
      throw std::bad_alloc();
    }
#endif
    spdlog::info(
        "Reserved {:.2f} GB virtual address space for up to {} compact "
        "top-{} candidate blocks",
        static_cast<double>(reserved_bytes_) / (1024.0 * 1024.0 * 1024.0),
        max_blocks_,
        kColorInlierMaxNum);
  }

  CandidateBlockPool(const CandidateBlockPool&) = delete;
  CandidateBlockPool& operator=(const CandidateBlockPool&) = delete;

  ~CandidateBlockPool() {
    if (blocks_ == nullptr) {
      return;
    }
#ifdef _WIN32
    VirtualFree(blocks_, 0, MEM_RELEASE);
#else
    munmap(blocks_, reserved_bytes_);
#endif
  }

  uint32_t Allocate() {
    const uint32_t index = next_block_.fetch_add(1, std::memory_order_relaxed);
    if (index >= max_blocks_) {
      throw std::runtime_error("Candidate block pool exhausted");
    }
    EnsureCommitted(index);
    blocks_[index] = CandidateBlock{};
    return index;
  }

  CandidateBlock& Get(uint32_t index) { return blocks_[index]; }
  const CandidateBlock& Get(uint32_t index) const { return blocks_[index]; }

  size_t size() const {
    return std::min<size_t>(next_block_.load(std::memory_order_relaxed),
                            max_blocks_);
  }

 private:
  void EnsureCommitted(uint32_t index) {
    if (index < committed_blocks_.load(std::memory_order_acquire)) {
      return;
    }
    std::lock_guard<std::mutex> lock(commit_mutex_);
    const size_t committed =
        committed_blocks_.load(std::memory_order_relaxed);
    if (index < committed) {
      return;
    }
    const size_t desired = std::min(
        max_blocks_,
        ((static_cast<size_t>(index) / kCandidateBlocksPerCommit) + 1) *
            kCandidateBlocksPerCommit);
    const size_t block_count = desired - committed;
    void* address = blocks_ + committed;
    const size_t bytes = block_count * sizeof(CandidateBlock);
#ifdef _WIN32
    if (VirtualAlloc(address, bytes, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
      throw std::bad_alloc();
    }
#else
    if (mprotect(address, bytes, PROT_READ | PROT_WRITE) != 0) {
      throw std::bad_alloc();
    }
#endif
    committed_blocks_.store(desired, std::memory_order_release);
  }

  CandidateBlock* blocks_ = nullptr;
  size_t max_blocks_ = 0;
  size_t reserved_bytes_ = 0;
  std::atomic<uint32_t> next_block_{0};
  std::atomic<size_t> committed_blocks_{0};
  std::mutex commit_mutex_;
};

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

  // Source and cubemap masks use the conventional validity encoding:
  // non-zero pixels are valid, while zero pixels must be excluded.
  return mask.at<uint8_t>(pixel_y, pixel_x) != 0;
}

bool IsNormalFacingCamera(const Eigen::Vector3d& point_in_camera,
                          const Eigen::Vector3d& normal_in_camera) {
  if (!normal_in_camera.allFinite() || normal_in_camera.squaredNorm() < 1e-12) {
    return true;
  }
  return normal_in_camera.dot(-point_in_camera) > 0.0;
}

double ComputeCandidateDistance(const Eigen::Vector3d& point_in_camera) {
  return point_in_camera.norm();
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
    const pcl::PointXYZRGBNormal& point,
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

  const Eigen::Vector3d normal_in_cam =
      pose.rotation() * point.getNormalVector3fMap().cast<double>();
  if (!IsNormalFacingCamera(pt_in_cam, normal_in_cam)) {
    return std::nullopt;  // Camera sees the back side of the surface
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
  const double distance = ComputeCandidateDistance(pt_in_cam);
  return std::make_pair(distance, color);
}

cv::Vec3b ComputeMedianColor(const CandidateBlock& block,
                             int candidate_count,
                             int inlier_threshold) {
  cv::Vec3b result = {0, 0, 0};

  for (int k = 0; k < 3; ++k) {
    std::array<uint8_t, kColorInlierMaxNum> color_channel_candidates{};
    for (int i = 0; i < candidate_count; ++i) {
      color_channel_candidates[i] =
          UnpackColor(block.candidates[i].color)[k];
    }

    const int mid_idx = candidate_count / 2;
    std::nth_element(color_channel_candidates.begin(),
                     color_channel_candidates.begin() + mid_idx,
                     color_channel_candidates.begin() + candidate_count);
    const uint8_t mid_color = color_channel_candidates[mid_idx];

    int color_sum = 0;
    int color_num = 0;
    for (int i = 0; i < candidate_count; ++i) {
      const uint8_t color_val = color_channel_candidates[i];
      if (std::abs((int)color_val - (int)mid_color) < inlier_threshold) {
        color_sum += color_val;
        color_num++;
      }
    }

    result[k] = color_num > 0 ? color_sum / color_num : mid_color;
  }

  return result;
}

cv::Vec3b ComputeMeanColor(const CandidateBlock& block, int candidate_count) {
  cv::Vec3i color_sum = {0, 0, 0};
  for (int i = 0; i < candidate_count; ++i) {
    color_sum += UnpackColor(block.candidates[i].color);
  }
  return {static_cast<uint8_t>(color_sum[0] / candidate_count),
          static_cast<uint8_t>(color_sum[1] / candidate_count),
          static_cast<uint8_t>(color_sum[2] / candidate_count)};
}

void WritePointCloudToLAS(
    const std::string& filename,
    const pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud,
    const std::vector<PointCandidateState>& point_states) {
  spdlog::info("Saving point cloud to {}", filename);

  pdal::PointTable table;
  table.layout()->registerDim(pdal::Dimension::Id::X);
  table.layout()->registerDim(pdal::Dimension::Id::Y);
  table.layout()->registerDim(pdal::Dimension::Id::Z);
  table.layout()->registerDim(pdal::Dimension::Id::Red);
  table.layout()->registerDim(pdal::Dimension::Id::Green);
  table.layout()->registerDim(pdal::Dimension::Id::Blue);
  table.layout()->registerDim(pdal::Dimension::Id::ReturnNumber);
  table.layout()->registerDim(pdal::Dimension::Id::NumberOfReturns);

  migration::IncrementalLasWriter writer;
  writer.initialize(filename, table);

  pdal::PointViewPtr view = std::make_shared<pdal::PointView>(table);
  size_t output_count = 0;
  for (size_t i = 0; i < cloud.size(); ++i) {
    if (point_states[i].count == 0) {
      continue;
    }
    const size_t chunk_index = view->size();
    view->setField(pdal::Dimension::Id::X, chunk_index, cloud[i].x);
    view->setField(pdal::Dimension::Id::Y, chunk_index, cloud[i].y);
    view->setField(pdal::Dimension::Id::Z, chunk_index, cloud[i].z);
    view->setField(pdal::Dimension::Id::Red, chunk_index, cloud[i].r);
    view->setField(pdal::Dimension::Id::Green, chunk_index, cloud[i].g);
    view->setField(pdal::Dimension::Id::Blue, chunk_index, cloud[i].b);
    view->setField(pdal::Dimension::Id::ReturnNumber, chunk_index, uint8_t{1});
    view->setField(pdal::Dimension::Id::NumberOfReturns,
                   chunk_index,
                   point_states[i].count);
    ++output_count;

    if (view->size() >= kOutputPointChunkSize) {
      writer.writeView(view);
      view = std::make_shared<pdal::PointView>(table);
    }
  }
  writer.writeView(view);
  writer.finalize(table);
  spdlog::info("Wrote {} colored points in chunks of at most {}",
               output_count,
               kOutputPointChunkSize);
}

void PerformXColor(const std::vector<Image>& images,
                   pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud_rgb,
                   std::string output_path,
                   int color_candidate_limit) {
  const int candidate_limit =
      std::clamp(color_candidate_limit, 1, kColorInlierMaxNum);
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

  // Keep only a compact 8-byte state per input point. Candidate blocks are
  // committed lazily for points that are actually visible in at least one
  // image, avoiding one heap-allocated std::vector per point.
  std::vector<PointCandidateState> point_candidate_states(cloud_rgb.size());
  CandidateBlockPool candidate_pool(cloud_rgb.size());

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
        auto& state = point_candidate_states[point_idx];
        if (state.block_index == kInvalidCandidateBlock) {
          state.block_index = candidate_pool.Allocate();
        }
        auto& block = candidate_pool.Get(state.block_index);
        const auto& candidate = color_candidate_opt.value();
        const PackedColorCandidate packed = {
            static_cast<float>(candidate.first), PackColor(candidate.second)};

        int insert_at = 0;
        while (insert_at < state.count &&
               block.candidates[insert_at].distance < packed.distance) {
          ++insert_at;
        }
        if (insert_at < candidate_limit) {
          const int new_count =
              std::min<int>(state.count + 1, candidate_limit);
          for (int i = new_count - 1; i > insert_at; --i) {
            block.candidates[i] = block.candidates[i - 1];
          }
          block.candidates[insert_at] = packed;
          state.count = static_cast<uint8_t>(new_count);
        }
      }
    }
    if ((image_idx + 1) % 100 == 0 || image_idx + 1 == images.size()) {
      spdlog::info("Candidate pool contains {} visible points after {} images",
                   candidate_pool.size(),
                   image_idx + 1);
    }
    print_image_progress();
  }

  // Step 3: 统一计算每个点的最终颜色
  spdlog::info("Computing final colors...");
  PrintProgress(96.0);
#pragma omp parallel for
  for (int point_idx = 0; point_idx < cloud_rgb.size(); ++point_idx) {
    const auto& state = point_candidate_states[point_idx];
    if (state.count == 0) {
      continue;
    }
    const auto& block = candidate_pool.Get(state.block_index);

    // Candidates were limited to the nearest samples while processing images.
    cv::Vec3b final_color;
    if (state.count >= kMinCandidateNum) {
      final_color =
          ComputeMedianColor(block, state.count, kColorInlierThreshold);
    } else {
      final_color = ComputeMeanColor(block, state.count);
    }

    cloud_rgb[point_idx].b = final_color[0];
    cloud_rgb[point_idx].g = final_color[1];
    cloud_rgb[point_idx].r = final_color[2];
  }

  PrintProgress(98.0);
  spdlog::info("Final colored point cloud contains {} / {} points",
               candidate_pool.size(),
               cloud_rgb.size());
  PrintProgress(99.0);
  WritePointCloudToLAS((output_dir / "xcolor.las").string(),
                       cloud_rgb,
                       point_candidate_states);
  PrintProgress(100.0);
}

}  // namespace xcolor
