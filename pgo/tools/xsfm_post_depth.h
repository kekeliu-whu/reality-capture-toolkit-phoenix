#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace xsfm_post {

struct FaceIntrinsics;

struct DepthRenderResult {
  std::vector<float> depth;
  int width              = 0;
  int height             = 0;
  int contributing_count = 0;
  int auxiliary_count    = 0;
};

struct DepthWorldPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct DepthWorldNormal {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

// A point that projects onto the rendered surface within the requested depth
// tolerance. Pixel coordinates refer to the source image rather than the
// potentially downscaled depth buffer.
struct VisiblePointProjection {
  uint32_t point_index = 0;
  int32_t pixel_x      = 0;
  int32_t pixel_y      = 0;
  float distance       = 0.0f;
};
static_assert(sizeof(VisiblePointProjection) == 16);

struct FisheyeVisibilityResult {
  std::vector<float> depth;
  std::vector<VisiblePointProjection> visible_points;
  int width              = 0;
  int height             = 0;
  int contributing_count = 0;
};

// Final per-point color selected by the CUDA fisheye path. Channels are kept
// in OpenCV's BGR order. view_count is saturated at the caller supplied
// maximum so it remains suitable for the LAS NumberOfReturns field.
struct FusedPointColor {
  uint8_t blue       = 0;
  uint8_t green      = 0;
  uint8_t red        = 0;
  uint8_t view_count = 0;
};
static_assert(sizeof(FusedPointColor) == 4);

struct FisheyeColoringResult {
  std::vector<float> depth;
  int width              = 0;
  int height             = 0;
  int contributing_count = 0;
};

// Parameters follow COLMAP's OPENCV_FISHEYE model:
// fx, fy, cx, cy, k1, k2, k3, k4.
struct OpenCVFisheyeIntrinsics {
  int width  = 0;
  int height = 0;
  double fx  = 0.0;
  double fy  = 0.0;
  double cx  = 0.0;
  double cy  = 0.0;
  double k1  = 0.0;
  double k2  = 0.0;
  double k3  = 0.0;
  double k4  = 0.0;
};

class CudaDepthRenderer {
 public:
  explicit CudaDepthRenderer(const std::vector<DepthWorldPoint>& world_points);
  CudaDepthRenderer(const std::vector<DepthWorldPoint>& world_points,
                    const std::vector<DepthWorldNormal>& world_normals);
  ~CudaDepthRenderer();

  CudaDepthRenderer(const CudaDepthRenderer&)            = delete;
  CudaDepthRenderer& operator=(const CudaDepthRenderer&) = delete;

  DepthRenderResult Render(const std::array<double, 9>& rotation,
                           const std::array<double, 3>& translation,
                           const FaceIntrinsics& intrinsics,
                           float voxel_size,
                           int gpu_chunk_points,
                           float max_distance,
                           bool sparse_mode) const;

  DepthRenderResult RenderFisheye(
      const std::array<double, 9>& rotation,
      const std::array<double, 3>& translation,
      const OpenCVFisheyeIntrinsics& intrinsics,
      float voxel_size,
      int gpu_chunk_points,
      float max_distance,
      bool sparse_mode) const;

  FisheyeVisibilityResult RenderFisheyeVisibility(
      const std::array<double, 9>& rotation,
      const std::array<double, 3>& translation,
      const OpenCVFisheyeIntrinsics& intrinsics,
      int source_width,
      int source_height,
      float voxel_size,
      int gpu_chunk_points,
      float max_distance,
      float visibility_tolerance,
      bool copy_depth) const;

  // Render a fisheye depth image and fuse all depth-visible samples directly
  // on the GPU. No per-image visible-point list is copied back to the CPU.
  FisheyeColoringResult RenderFisheyeColor(
      const std::array<double, 9>& rotation,
      const std::array<double, 3>& translation,
      const OpenCVFisheyeIntrinsics& intrinsics,
      const uint8_t* bgr_data,
      int source_width,
      int source_height,
      size_t source_row_stride,
      float voxel_size,
      int gpu_chunk_points,
      float max_distance,
      float visibility_tolerance,
      int max_view_count,
      int view_index,
      bool copy_depth) const;

  void ResetFusedColors() const;
  std::vector<FusedPointColor> DownloadFusedColors(
      bool smooth_fusion = false) const;

  void SetVisibilityMask(const uint8_t* data,
                         int width,
                         int height,
                         size_t row_stride);
  void ClearVisibilityMask();

  size_t PointCount() const;

 private:
  DepthRenderResult RenderProjection(
      const std::array<double, 9>& rotation,
      const std::array<double, 3>& translation,
      int width,
      int height,
      const std::array<double, 8>& projection_params,
      bool fisheye,
      float voxel_size,
      int gpu_chunk_points,
      float max_distance,
      bool sparse_mode,
      int visibility_source_width,
      int visibility_source_height,
      float visibility_tolerance,
      std::vector<VisiblePointProjection>* visible_points,
      bool copy_depth,
      const uint8_t* color_image,
      int color_image_width,
      int color_image_height,
      size_t color_image_row_stride,
      int max_view_count,
      int color_view_index) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool HasCudaDevice();

}  // namespace xsfm_post
