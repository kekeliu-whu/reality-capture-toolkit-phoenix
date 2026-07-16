#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace xsfm_post {

struct FaceIntrinsics;

struct DepthRenderResult {
  std::vector<float> depth;
  int width = 0;
  int height = 0;
  int contributing_count = 0;
  int auxiliary_count = 0;
};

struct DepthWorldPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

class CudaDepthRenderer {
 public:
  explicit CudaDepthRenderer(const std::vector<DepthWorldPoint>& world_points);
  ~CudaDepthRenderer();

  CudaDepthRenderer(const CudaDepthRenderer&) = delete;
  CudaDepthRenderer& operator=(const CudaDepthRenderer&) = delete;

  DepthRenderResult Render(const std::array<double, 9>& rotation,
                           const std::array<double, 3>& translation,
                           const FaceIntrinsics& intrinsics,
                           float voxel_size,
                           int gpu_chunk_points,
                           float max_distance,
                           bool sparse_mode) const;

  size_t PointCount() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool HasCudaDevice();

}  // namespace xsfm_post
