#include "xsfm_post_depth.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xsfm_post {

struct FaceIntrinsics {
  int width    = 0;
  int height   = 0;
  double focal = 0.0;
  double cx    = 0.0;
  double cy    = 0.0;
};

namespace {

void CheckCuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) {
    if (count > 0) {
      CheckCuda(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc(buffer)");
    }
  }

  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer(const DeviceBuffer&)            = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const { return data_; }

 private:
  T* data_ = nullptr;
};

struct FaceTransform {
  double rotation[9];
  double translation[3];
};

enum class ProjectionModel : int { kPerspective   = 0,
                                   kOpenCVFisheye = 1 };

struct ProjectionIntrinsics {
  ProjectionModel model = ProjectionModel::kPerspective;
  int width             = 0;
  int height            = 0;
  double fx             = 0.0;
  double fy             = 0.0;
  double cx             = 0.0;
  double cy             = 0.0;
  double k1             = 0.0;
  double k2             = 0.0;
  double k3             = 0.0;
  double k4             = 0.0;
};

__device__ bool ProjectPoint(const ProjectionIntrinsics& intrinsics,
                             double x,
                             double y,
                             double z,
                             double voxel_circumradius,
                             double* pixel_x,
                             double* pixel_y,
                             double* pixel_radius) {
  if (z <= 1e-12) {
    return false;
  }
  if (intrinsics.model == ProjectionModel::kPerspective) {
    *pixel_x      = intrinsics.fx * x / z + intrinsics.cx;
    *pixel_y      = intrinsics.fy * y / z + intrinsics.cy;
    *pixel_radius = fmax(intrinsics.fx, intrinsics.fy) *
                    voxel_circumradius / z;
    return true;
  }

  // Match COLMAP's OPENCV_FISHEYE projection exactly: normalized pinhole
  // coordinates are mapped to theta, followed by the four radial terms.
  const double radial_xy = hypot(x, y);
  const double theta     = atan2(radial_xy, z);
  double unit_x          = 0.0;
  double unit_y          = 0.0;
  if (radial_xy > 1e-12) {
    unit_x = x / radial_xy;
    unit_y = y / radial_xy;
  }
  const double theta2          = theta * theta;
  const double theta4          = theta2 * theta2;
  const double theta6          = theta4 * theta2;
  const double theta8          = theta4 * theta4;
  const double radial          = 1.0 + intrinsics.k1 * theta2 +
                                 intrinsics.k2 * theta4 + intrinsics.k3 * theta6 +
                                 intrinsics.k4 * theta8;
  const double distorted_theta = theta * radial;
  *pixel_x                     = intrinsics.fx * unit_x * distorted_theta + intrinsics.cx;
  *pixel_y                     = intrinsics.fy * unit_y * distorted_theta + intrinsics.cy;

  const double distance = sqrt(x * x + y * y + z * z);
  const double angular_radius =
      asin(fmin(1.0, voxel_circumradius / fmax(distance, 1e-12)));
  const double radial_derivative =
      1.0 + 3.0 * intrinsics.k1 * theta2 +
      5.0 * intrinsics.k2 * theta4 + 7.0 * intrinsics.k3 * theta6 +
      9.0 * intrinsics.k4 * theta8;
  const double tangential_derivative =
      theta > 1e-12 ? fabs(distorted_theta / sin(theta)) : 1.0;
  const double projection_derivative =
      fmax(fabs(radial_derivative), tangential_derivative);
  *pixel_radius = fmax(intrinsics.fx, intrinsics.fy) * angular_radius *
                  projection_derivative;
  return true;
}

__global__ void TransformProjectPointsKernel(const float3* world_points,
                                             int point_count,
                                             FaceTransform transform,
                                             ProjectionIntrinsics intrinsics,
                                             double voxel_circumradius,
                                             double max_distance_sq,
                                             int max_radius,
                                             float3* filtered_points,
                                             int* center_x,
                                             int* center_y,
                                             int* radius,
                                             int* filtered_count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }

  const float3 world = world_points[i];
  const double wx    = static_cast<double>(world.x);
  const double wy    = static_cast<double>(world.y);
  const double wz    = static_cast<double>(world.z);
  const double x     = transform.rotation[0] * wx + transform.rotation[1] * wy +
                       transform.rotation[2] * wz + transform.translation[0];
  const double y     = transform.rotation[3] * wx + transform.rotation[4] * wy +
                       transform.rotation[5] * wz + transform.translation[1];
  const double z     = transform.rotation[6] * wx + transform.rotation[7] * wy +
                       transform.rotation[8] * wz + transform.translation[2];
  if (!isfinite(x) || !isfinite(y) || !isfinite(z) || z <= 0.0 ||
      x * x + y * y + z * z > max_distance_sq) {
    return;
  }

  double u                = 0.0;
  double v                = 0.0;
  double projected_radius = 0.0;
  if (!ProjectPoint(intrinsics,
                    x,
                    y,
                    z,
                    voxel_circumradius,
                    &u,
                    &v,
                    &projected_radius) ||
      u + projected_radius < 0.0 || u - projected_radius >= intrinsics.width ||
      v + projected_radius < 0.0 || v - projected_radius >= intrinsics.height) {
    return;
  }

  const int output_index = atomicAdd(filtered_count, 1);
  filtered_points[output_index] =
      make_float3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
  // For non-negative image coordinates llround(u - 0.5), used by the CPU
  // path, is equivalent to floor(u). Keeping the same convention avoids
  // half-pixel visibility differences between the two implementations.
  center_x[output_index] = static_cast<int>(floor(u));
  center_y[output_index] = static_cast<int>(floor(v));
  const int r            = static_cast<int>(ceil(projected_radius));
  radius[output_index]   = min(max(r, 0), max_radius);
}

__global__ void SplatMinDepthKernel(const float3* points,
                                    const int* center_x,
                                    const int* center_y,
                                    const int* radius,
                                    int point_count,
                                    int width,
                                    int height,
                                    unsigned int* flat_depth_bits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const int cx              = center_x[i];
  const int cy              = center_y[i];
  const int r               = radius[i];
  const float z             = points[i].z;
  const unsigned int z_bits = __float_as_uint(z);
  for (int dy = -r; dy <= r; ++dy) {
    for (int dx = -r; dx <= r; ++dx) {
      if (dx * dx + dy * dy > r * r) {
        continue;
      }
      const int x = cx + dx;
      const int y = cy + dy;
      if (x < 0 || x >= width || y < 0 || y >= height) {
        continue;
      }
      atomicMin(flat_depth_bits + y * width + x, z_bits);
    }
  }
}

__global__ void CountFiniteKernel(const unsigned int* flat_depth_bits,
                                  int pixel_count,
                                  unsigned int inf_bits,
                                  int* count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pixel_count && flat_depth_bits[i] != inf_bits) {
    atomicAdd(count, 1);
  }
}

__global__ void FillBitsKernel(unsigned int* values,
                               int count,
                               unsigned int bits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    values[i] = bits;
  }
}

__global__ void DenseDepthToFloatKernel(const unsigned int* flat_depth_bits,
                                        int pixel_count,
                                        unsigned int inf_bits,
                                        float* depth) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixel_count) {
    return;
  }
  depth[i] = flat_depth_bits[i] == inf_bits ? 0.0f
                                            : __uint_as_float(flat_depth_bits[i]);
}

__global__ void MarkVisiblePointsKernel(const float3* points,
                                        const int* center_x,
                                        const int* center_y,
                                        const int* radius,
                                        int point_count,
                                        int width,
                                        int height,
                                        const unsigned int* flat_depth_bits,
                                        unsigned char* visible) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const int cx              = center_x[i];
  const int cy              = center_y[i];
  const int r               = radius[i];
  const unsigned int z_bits = __float_as_uint(points[i].z);
  for (int dy = -r; dy <= r; ++dy) {
    for (int dx = -r; dx <= r; ++dx) {
      if (dx * dx + dy * dy > r * r) {
        continue;
      }
      const int x = cx + dx;
      const int y = cy + dy;
      if (x < 0 || x >= width || y < 0 || y >= height) {
        continue;
      }
      if (flat_depth_bits[y * width + x] == z_bits) {
        visible[i] = 1;
        return;
      }
    }
  }
}

__global__ void SparseCenterDepthKernel(const float3* points,
                                        const int* center_x,
                                        const int* center_y,
                                        const unsigned char* visible,
                                        int point_count,
                                        int width,
                                        int height,
                                        unsigned int* flat_sparse_bits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count || visible[i] == 0) {
    return;
  }
  const int x = center_x[i];
  const int y = center_y[i];
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }
  atomicMin(flat_sparse_bits + y * width + x, __float_as_uint(points[i].z));
}

__global__ void CountVisiblePointsKernel(const unsigned char* visible,
                                         int point_count,
                                         int* count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < point_count && visible[i] != 0) {
    atomicAdd(count, 1);
  }
}

// A small weighted reservoir keeps observations spread over the full image
// sequence without retaining every visible pixel. RGB stays byte-exact, while
// quality and source-frame identity are compact enough for very large clouds.
constexpr int kConsensusCandidateCount  = 12;
constexpr float kConsensusColorDistance = 35.0f;
// A single high-quality projection must agree more closely with the dominant
// multi-view color before it may replace that cluster's representative.  The
// wider consensus threshold above is intentionally tolerant of exposure
// changes; reusing it here lets orange/red foreground leaks pass as neutral
// wall colors with similar luminance.
constexpr float kSharpBestColorDistance = 20.0f;
constexpr int kSharpCandidateCount      = 3;

struct alignas(4) PackedColorCandidate {
  uchar4 color{};
  uint16_t quality    = 0;
  uint16_t view_index = 0;
  float reservoir_key = 0.0f;
};
static_assert(sizeof(PackedColorCandidate) == 12);

// NVS-style appearance hypotheses kept independently from the statistical
// reservoir.  The score combines geometric quality with local source-image
// detail, while RGB remains an untouched real source pixel.
struct alignas(4) PackedSharpCandidate {
  uchar4 color{};
  uint16_t score      = 0;
  uint16_t view_index = 0;
};
static_assert(sizeof(PackedSharpCandidate) == 8);

__device__ float LoadBgrLuminance(const unsigned char* source_bgr,
                                  int source_width,
                                  int x,
                                  int y) {
  const size_t offset =
      (static_cast<size_t>(y) * source_width + x) * 3;
  return 0.114f * static_cast<float>(source_bgr[offset]) +
         0.587f * static_cast<float>(source_bgr[offset + 1]) +
         0.299f * static_cast<float>(source_bgr[offset + 2]);
}

__device__ float ComputeLocalSharpness(const unsigned char* source_bgr,
                                       int source_width,
                                       int source_height,
                                       int x,
                                       int y) {
  if (x <= 0 || x + 1 >= source_width || y <= 0 ||
      y + 1 >= source_height) {
    return 0.0f;
  }
  const float center = LoadBgrLuminance(source_bgr, source_width, x, y);
  const float left   = LoadBgrLuminance(source_bgr, source_width, x - 1, y);
  const float right  = LoadBgrLuminance(source_bgr, source_width, x + 1, y);
  const float up     = LoadBgrLuminance(source_bgr, source_width, x, y - 1);
  const float down   = LoadBgrLuminance(source_bgr, source_width, x, y + 1);
  const float gradient =
      0.5f * (fabsf(right - left) + fabsf(down - up));
  const float laplacian =
      0.25f * fabsf(4.0f * center - left - right - up - down);
  const float detail_signal = 0.7f * gradient + 0.3f * laplacian;
  // Soft normalization prevents a single saturated edge from overwhelming
  // depth, distance, and normal quality.
  return detail_signal / (detail_signal + 24.0f);
}

__device__ uint32_t MixCandidateBits(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

}  // namespace

bool HasCudaDevice() {
  int count                = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  return status == cudaSuccess && count > 0;
}

struct CudaDepthRenderer::Impl {
  float3* world_points           = nullptr;
  float3* world_normals          = nullptr;
  size_t point_count             = 0;
  unsigned char* visibility_mask = nullptr;
  int visibility_mask_width      = 0;
  int visibility_mask_height     = 0;
  unsigned char* color_image     = nullptr;
  size_t color_image_capacity    = 0;
  // Four quality-weighted reservoir samples per point are enough to reject a
  // minority moving object or bad occlusion projection statistically. The
  // final RGB still comes from one real observation rather than an average.
  PackedColorCandidate* consensus_candidates = nullptr;
  // Keep several sharp real observations independently of the statistical
  // reservoir.  If the first is a moving object or occlusion leak, another
  // crisp inlier can be used instead of a random reservoir sample.
  PackedSharpCandidate* sharp_candidates = nullptr;
  // Weighted log-luminance from non-clipped observations. It adjusts only the
  // brightness of the selected sharp sample, never its spatial RGB detail.
  float2* exposure_log_luminance   = nullptr;
  unsigned char* color_view_counts = nullptr;

  ~Impl() {
    cudaFree(world_points);
    cudaFree(world_normals);
    cudaFree(visibility_mask);
    cudaFree(color_image);
    cudaFree(consensus_candidates);
    cudaFree(sharp_candidates);
    cudaFree(exposure_log_luminance);
    cudaFree(color_view_counts);
  }
};

CudaDepthRenderer::CudaDepthRenderer(
    const std::vector<DepthWorldPoint>& world_points)
    : impl_(std::make_unique<Impl>()) {
  if (!HasCudaDevice()) {
    throw std::runtime_error(
        "CUDA depth rendering was requested but no CUDA device is available.");
  }
  impl_->point_count = world_points.size();
  if (world_points.empty()) {
    return;
  }

  static_assert(sizeof(DepthWorldPoint) == sizeof(float3));
  CheckCuda(cudaMalloc(&impl_->world_points, world_points.size() * sizeof(float3)),
            "cudaMalloc(world_points)");
  CheckCuda(cudaMemcpy(impl_->world_points,
                       world_points.data(),
                       world_points.size() * sizeof(float3),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(world_points)");
}

CudaDepthRenderer::CudaDepthRenderer(
    const std::vector<DepthWorldPoint>& world_points,
    const std::vector<DepthWorldNormal>& world_normals)
    : CudaDepthRenderer(world_points) {
  if (world_normals.size() != world_points.size()) {
    throw std::invalid_argument("Point and normal counts must match");
  }
  if (world_normals.empty()) {
    return;
  }
  static_assert(sizeof(DepthWorldNormal) == sizeof(float3));
  CheckCuda(cudaMalloc(&impl_->world_normals,
                       world_normals.size() * sizeof(float3)),
            "cudaMalloc(world_normals)");
  CheckCuda(cudaMemcpy(impl_->world_normals,
                       world_normals.data(),
                       world_normals.size() * sizeof(float3),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(world_normals)");
}

void CudaDepthRenderer::SetVisibilityMask(const uint8_t* data,
                                          int width,
                                          int height,
                                          size_t row_stride) {
  if (data == nullptr || width <= 0 || height <= 0 ||
      row_stride < static_cast<size_t>(width)) {
    throw std::invalid_argument("Invalid visibility mask");
  }
  if (impl_->visibility_mask_width != width ||
      impl_->visibility_mask_height != height) {
    cudaFree(impl_->visibility_mask);
    impl_->visibility_mask = nullptr;
    CheckCuda(cudaMalloc(&impl_->visibility_mask,
                         static_cast<size_t>(width) * height),
              "cudaMalloc(visibility_mask)");
    impl_->visibility_mask_width  = width;
    impl_->visibility_mask_height = height;
  }
  CheckCuda(cudaMemcpy2D(impl_->visibility_mask,
                         static_cast<size_t>(width),
                         data,
                         row_stride,
                         static_cast<size_t>(width),
                         height,
                         cudaMemcpyHostToDevice),
            "cudaMemcpy2D(visibility_mask)");
}

void CudaDepthRenderer::ClearVisibilityMask() {
  cudaFree(impl_->visibility_mask);
  impl_->visibility_mask        = nullptr;
  impl_->visibility_mask_width  = 0;
  impl_->visibility_mask_height = 0;
}

__global__ void FinalizeFusedColorsKernel(
    const PackedColorCandidate* consensus_candidates,
    const PackedSharpCandidate* sharp_candidates,
    const float2* exposure_log_luminance,
    const unsigned char* color_view_counts,
    int point_count,
    bool smooth_fusion,
    uchar4* output);

void CudaDepthRenderer::ResetFusedColors() const {
  if (impl_->point_count == 0) {
    return;
  }
  if (impl_->consensus_candidates == nullptr) {
    CheckCuda(cudaMalloc(&impl_->consensus_candidates,
                         impl_->point_count * kConsensusCandidateCount *
                             sizeof(PackedColorCandidate)),
              "cudaMalloc(consensus_candidates)");
  }
  if (impl_->sharp_candidates == nullptr) {
    CheckCuda(cudaMalloc(&impl_->sharp_candidates,
                         impl_->point_count * kSharpCandidateCount *
                             sizeof(PackedSharpCandidate)),
              "cudaMalloc(sharp_candidates)");
  }
  if (impl_->color_view_counts == nullptr) {
    CheckCuda(cudaMalloc(&impl_->color_view_counts,
                         impl_->point_count * sizeof(unsigned char)),
              "cudaMalloc(color_view_counts)");
  }
  if (impl_->exposure_log_luminance == nullptr) {
    CheckCuda(cudaMalloc(&impl_->exposure_log_luminance,
                         impl_->point_count * sizeof(float2)),
              "cudaMalloc(exposure_log_luminance)");
  }
  CheckCuda(cudaMemset(impl_->consensus_candidates,
                       0,
                       impl_->point_count * kConsensusCandidateCount *
                           sizeof(PackedColorCandidate)),
            "cudaMemset(consensus_candidates)");
  CheckCuda(cudaMemset(impl_->sharp_candidates,
                       0,
                       impl_->point_count * kSharpCandidateCount *
                           sizeof(PackedSharpCandidate)),
            "cudaMemset(sharp_candidates)");
  CheckCuda(cudaMemset(impl_->color_view_counts,
                       0,
                       impl_->point_count * sizeof(unsigned char)),
            "cudaMemset(color_view_counts)");
  CheckCuda(cudaMemset(impl_->exposure_log_luminance,
                       0,
                       impl_->point_count * sizeof(float2)),
            "cudaMemset(exposure_log_luminance)");
}

std::vector<FusedPointColor> CudaDepthRenderer::DownloadFusedColors(
    bool smooth_fusion) const {
  std::vector<FusedPointColor> result(impl_->point_count);
  if (impl_->point_count == 0 || impl_->consensus_candidates == nullptr ||
      impl_->sharp_candidates == nullptr ||
      impl_->exposure_log_luminance == nullptr ||
      impl_->color_view_counts == nullptr) {
    return result;
  }
  static_assert(sizeof(FusedPointColor) == sizeof(uchar4));
  DeviceBuffer<uchar4> device_output(impl_->point_count);
  constexpr int threads = 256;
  const int blocks =
      static_cast<int>((impl_->point_count + threads - 1) / threads);
  FinalizeFusedColorsKernel<<<blocks, threads>>>(
      impl_->consensus_candidates,
      impl_->sharp_candidates,
      impl_->exposure_log_luminance,
      impl_->color_view_counts,
      static_cast<int>(impl_->point_count),
      smooth_fusion,
      device_output.get());
  CheckCuda(cudaGetLastError(), "FinalizeFusedColorsKernel");
  CheckCuda(cudaMemcpy(result.data(),
                       device_output.get(),
                       result.size() * sizeof(FusedPointColor),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(fused colors)");
  return result;
}

__global__ void CollectDepthVisiblePointsKernel(
    const float3* world_points,
    const float3* world_normals,
    int point_count,
    int source_offset,
    FaceTransform transform,
    ProjectionIntrinsics intrinsics,
    double max_distance_sq,
    int source_width,
    int source_height,
    const unsigned char* visibility_mask,
    int visibility_mask_width,
    int visibility_mask_height,
    int width,
    int height,
    const unsigned int* flat_depth_bits,
    unsigned int inf_bits,
    float visibility_tolerance,
    VisiblePointProjection* output,
    int* output_count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const float3 world = world_points[i];
  const double wx    = static_cast<double>(world.x);
  const double wy    = static_cast<double>(world.y);
  const double wz    = static_cast<double>(world.z);
  const double x_cam = transform.rotation[0] * wx +
                       transform.rotation[1] * wy +
                       transform.rotation[2] * wz + transform.translation[0];
  const double y_cam = transform.rotation[3] * wx +
                       transform.rotation[4] * wy +
                       transform.rotation[5] * wz + transform.translation[1];
  const double z_cam = transform.rotation[6] * wx +
                       transform.rotation[7] * wy +
                       transform.rotation[8] * wz + transform.translation[2];
  if (!isfinite(x_cam) || !isfinite(y_cam) || !isfinite(z_cam) ||
      z_cam <= 0.0 ||
      x_cam * x_cam + y_cam * y_cam + z_cam * z_cam > max_distance_sq) {
    return;
  }
  if (world_normals != nullptr) {
    const float3 world_normal   = world_normals[i];
    const double nx             = static_cast<double>(world_normal.x);
    const double ny             = static_cast<double>(world_normal.y);
    const double nz             = static_cast<double>(world_normal.z);
    const double normal_norm_sq = nx * nx + ny * ny + nz * nz;
    if (isfinite(nx) && isfinite(ny) && isfinite(nz) &&
        normal_norm_sq >= 1e-12) {
      const double nx_cam = transform.rotation[0] * nx +
                            transform.rotation[1] * ny +
                            transform.rotation[2] * nz;
      const double ny_cam = transform.rotation[3] * nx +
                            transform.rotation[4] * ny +
                            transform.rotation[5] * nz;
      const double nz_cam = transform.rotation[6] * nx +
                            transform.rotation[7] * ny +
                            transform.rotation[8] * nz;
      if (nx_cam * -x_cam + ny_cam * -y_cam + nz_cam * -z_cam <= 0.0) {
        return;
      }
    }
  }
  double u              = 0.0;
  double v              = 0.0;
  double ignored_radius = 0.0;
  if (!ProjectPoint(intrinsics,
                    x_cam,
                    y_cam,
                    z_cam,
                    0.0,
                    &u,
                    &v,
                    &ignored_radius) ||
      u < 0.0 || u >= width || v < 0.0 || v >= height) {
    return;
  }
  const int x = static_cast<int>(floor(u));
  const int y = static_cast<int>(floor(v));
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }
  const unsigned int depth_bits = flat_depth_bits[y * width + x];
  if (depth_bits == inf_bits) {
    return;
  }
  const float depth = __uint_as_float(depth_bits);
  if (fabsf(depth - static_cast<float>(z_cam)) > visibility_tolerance) {
    return;
  }
  const double source_x = u * static_cast<double>(source_width) / width;
  const double source_y = v * static_cast<double>(source_height) / height;
  if (source_x < 0.0 || source_x >= source_width || source_y < 0.0 ||
      source_y >= source_height) {
    return;
  }
  const int source_pixel_x = static_cast<int>(floor(source_x));
  const int source_pixel_y = static_cast<int>(floor(source_y));
  if (visibility_mask != nullptr &&
      (source_pixel_x < 0 || source_pixel_x >= visibility_mask_width ||
       source_pixel_y < 0 || source_pixel_y >= visibility_mask_height ||
       visibility_mask[source_pixel_y * visibility_mask_width +
                       source_pixel_x] == 0)) {
    return;
  }
  const int output_index            = atomicAdd(output_count, 1);
  VisiblePointProjection& candidate = output[output_index];
  candidate.point_index             = static_cast<uint32_t>(source_offset + i);
  candidate.pixel_x                 = source_pixel_x;
  candidate.pixel_y                 = source_pixel_y;
  candidate.distance                = static_cast<float>(
      sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam));
}

__global__ void FuseDepthVisibleColorsKernel(
    const float3* world_points,
    const float3* world_normals,
    int point_count,
    int source_offset,
    FaceTransform transform,
    ProjectionIntrinsics intrinsics,
    double max_distance_sq,
    int source_width,
    int source_height,
    const unsigned char* source_bgr,
    const unsigned char* visibility_mask,
    int visibility_mask_width,
    int visibility_mask_height,
    int width,
    int height,
    const unsigned int* flat_depth_bits,
    unsigned int inf_bits,
    float visibility_tolerance,
    int max_view_count,
    int view_index,
    PackedColorCandidate* consensus_candidates,
    PackedSharpCandidate* sharp_candidates,
    float2* exposure_log_luminance,
    unsigned char* color_view_counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }

  const float3 world = world_points[i];
  const double wx    = static_cast<double>(world.x);
  const double wy    = static_cast<double>(world.y);
  const double wz    = static_cast<double>(world.z);
  const double x_cam = transform.rotation[0] * wx +
                       transform.rotation[1] * wy +
                       transform.rotation[2] * wz + transform.translation[0];
  const double y_cam = transform.rotation[3] * wx +
                       transform.rotation[4] * wy +
                       transform.rotation[5] * wz + transform.translation[1];
  const double z_cam = transform.rotation[6] * wx +
                       transform.rotation[7] * wy +
                       transform.rotation[8] * wz + transform.translation[2];
  if (!isfinite(x_cam) || !isfinite(y_cam) || !isfinite(z_cam) ||
      z_cam <= 0.0 ||
      x_cam * x_cam + y_cam * y_cam + z_cam * z_cam > max_distance_sq) {
    return;
  }

  const double distance_double =
      sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam);
  const float distance = static_cast<float>(distance_double);
  float normal_weight  = 1.0f;
  if (world_normals != nullptr) {
    const float3 world_normal   = world_normals[i];
    const double nx             = static_cast<double>(world_normal.x);
    const double ny             = static_cast<double>(world_normal.y);
    const double nz             = static_cast<double>(world_normal.z);
    const double normal_norm_sq = nx * nx + ny * ny + nz * nz;
    if (isfinite(nx) && isfinite(ny) && isfinite(nz) &&
        normal_norm_sq >= 1e-12) {
      const double nx_cam = transform.rotation[0] * nx +
                            transform.rotation[1] * ny +
                            transform.rotation[2] * nz;
      const double ny_cam = transform.rotation[3] * nx +
                            transform.rotation[4] * ny +
                            transform.rotation[5] * nz;
      const double nz_cam = transform.rotation[6] * nx +
                            transform.rotation[7] * ny +
                            transform.rotation[8] * nz;
      const double facing =
          nx_cam * -x_cam + ny_cam * -y_cam + nz_cam * -z_cam;
      if (facing <= 0.0) {
        return;
      }
      const double cosine = fmin(
          1.0, fmax(0.0, facing / (sqrt(normal_norm_sq) * distance_double)));
      constexpr float kRadiansToDegrees = 57.29577951308232f;
      const float angle =
          static_cast<float>(acos(cosine)) * kRadiansToDegrees;
      // Recovered Lixel normal Gaussian: sigma = 45 degrees.
      normal_weight = __expf(-(angle * angle) / 4050.0f);
    }
  }

  double u              = 0.0;
  double v              = 0.0;
  double ignored_radius = 0.0;
  if (!ProjectPoint(intrinsics,
                    x_cam,
                    y_cam,
                    z_cam,
                    0.0,
                    &u,
                    &v,
                    &ignored_radius) ||
      u < 0.0 || u >= width || v < 0.0 || v >= height) {
    return;
  }
  const int x = static_cast<int>(floor(u));
  const int y = static_cast<int>(floor(v));
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }
  const unsigned int depth_bits = flat_depth_bits[y * width + x];
  if (depth_bits == inf_bits) {
    return;
  }
  const float depth = __uint_as_float(depth_bits);
  if (fabsf(depth - static_cast<float>(z_cam)) > visibility_tolerance) {
    return;
  }

  const double source_x = u * static_cast<double>(source_width) / width;
  const double source_y = v * static_cast<double>(source_height) / height;
  if (source_x < 0.0 || source_x >= source_width || source_y < 0.0 ||
      source_y >= source_height) {
    return;
  }
  const int source_pixel_x = static_cast<int>(floor(source_x));
  const int source_pixel_y = static_cast<int>(floor(source_y));
  if (visibility_mask != nullptr &&
      (source_pixel_x < 0 || source_pixel_x >= visibility_mask_width ||
       source_pixel_y < 0 || source_pixel_y >= visibility_mask_height ||
       visibility_mask[source_pixel_y * visibility_mask_width +
                       source_pixel_x] == 0)) {
    return;
  }

  const size_t image_offset =
      (static_cast<size_t>(source_pixel_y) * source_width + source_pixel_x) * 3;
  const unsigned char blue  = source_bgr[image_offset];
  const unsigned char green = source_bgr[image_offset + 1];
  const unsigned char red   = source_bgr[image_offset + 2];

  const float distance_delta = distance - 0.5f;
  float distance_weight =
      __expf(-(distance_delta * distance_delta) / 72.0f);
  // Preserve useful weight for distant surfaces. This is the far-distance
  // branch visible in the recovered Lixel kernel.
  if (distance >= 11.80646f) {
    distance_weight = fmaxf(distance_weight, 2.0f / distance);
  }
  float weight = fminf(1.0f, fmaxf(0.0f, distance_weight * normal_weight));
  const bool clipped_exposure =
      (blue > 250 && green > 250 && red > 250) ||
      (blue < 5 && green < 5 && red < 5);
  if (clipped_exposure) {
    // Lixel lowers fully blown/black samples instead of discarding them.
    weight *= 0.1f;
  }
  if (!(weight > 0.0f) || !isfinite(weight)) {
    return;
  }

  const int point_index = source_offset + i;
  if (!clipped_exposure) {
    // A geometric mean in luminance space is stable across multiplicative
    // exposure changes. RGB is deliberately not accumulated, so fine texture
    // continues to come from exactly one best-aligned source image.
    const float luminance = 0.114f * static_cast<float>(blue) +
                            0.587f * static_cast<float>(green) +
                            0.299f * static_cast<float>(red);
    float2 exposure       = exposure_log_luminance[point_index];
    exposure.x += weight * logf(luminance + 1.0f);
    exposure.y += weight;
    exposure_log_luminance[point_index] = exposure;
  }

  // Preserve several real source pixels ranked by geometry and local image
  // detail.  This mirrors NVS's separation between stable multi-view
  // appearance and a small set of high-frequency view hypotheses.
  bool sharpness_neighborhood_valid = true;
  if (visibility_mask != nullptr) {
    sharpness_neighborhood_valid =
        source_pixel_x > 0 && source_pixel_y > 0 &&
        source_pixel_x + 1 < visibility_mask_width &&
        source_pixel_y + 1 < visibility_mask_height &&
        visibility_mask[source_pixel_y * visibility_mask_width +
                        source_pixel_x - 1] != 0 &&
        visibility_mask[source_pixel_y * visibility_mask_width +
                        source_pixel_x + 1] != 0 &&
        visibility_mask[(source_pixel_y - 1) * visibility_mask_width +
                        source_pixel_x] != 0 &&
        visibility_mask[(source_pixel_y + 1) * visibility_mask_width +
                        source_pixel_x] != 0;
  }
  const float local_sharpness =
      sharpness_neighborhood_valid
          ? ComputeLocalSharpness(source_bgr,
                                  source_width,
                                  source_height,
                                  source_pixel_x,
                                  source_pixel_y)
          : 0.0f;
  const float sharp_score_float =
      weight * (0.75f + 0.25f * local_sharpness);
  const uint16_t sharp_score = static_cast<uint16_t>(
      min(65535, max(1, __float2int_rn(sharp_score_float * 65535.0f))));
  PackedSharpCandidate sharp_candidate;
  sharp_candidate.color = make_uchar4(blue, green, red, 0);
  sharp_candidate.score = sharp_score;
  sharp_candidate.view_index =
      static_cast<uint16_t>(min(65535, max(0, view_index)));
  PackedSharpCandidate* point_sharp_candidates =
      sharp_candidates + point_index * kSharpCandidateCount;
  for (int slot = 0; slot < kSharpCandidateCount; ++slot) {
    if (sharp_score <= point_sharp_candidates[slot].score) {
      continue;
    }
    for (int shift = kSharpCandidateCount - 1; shift > slot; --shift) {
      point_sharp_candidates[shift] = point_sharp_candidates[shift - 1];
    }
    point_sharp_candidates[slot] = sharp_candidate;
    break;
  }
  // Efraimidis-Spirakis weighted reservoir sampling. Quality affects the
  // chance of retention, while the deterministic hash prevents adjacent
  // frames from monopolizing all four slots.
  // Use one temporal sampling sequence for the whole cloud. Nearby surface
  // points therefore prefer the same well-spaced source frames without the
  // hard boundaries produced by quantized spatial cells. Twelve retained
  // observations give moving foreground substantially less chance to win the
  // floor consensus than the previous four/eight-sample reservoirs.
  const uint32_t mixed = MixCandidateBits(
      static_cast<uint32_t>(view_index) * 0x85ebca6bu + 0x9e3779b9u);
  const float uniform =
      (static_cast<float>(mixed & 0x00ffffffu) + 1.0f) / 16777217.0f;
  const float reservoir_key = logf(uniform) / fmaxf(weight, 1e-6f);
  const uint16_t quality    = static_cast<uint16_t>(
      min(65535, max(1, __float2int_rn(weight * 65535.0f))));
  PackedColorCandidate* point_candidates =
      consensus_candidates + point_index * kConsensusCandidateCount;
  int replacement_slot = -1;
  float lowest_key     = FLT_MAX;
  for (int slot = 0; slot < kConsensusCandidateCount; ++slot) {
    const PackedColorCandidate candidate = point_candidates[slot];
    if (candidate.quality == 0) {
      replacement_slot = slot;
      break;
    }
    if (candidate.reservoir_key < lowest_key) {
      lowest_key       = candidate.reservoir_key;
      replacement_slot = slot;
    }
  }
  if (replacement_slot >= 0 &&
      (point_candidates[replacement_slot].quality == 0 ||
       reservoir_key > lowest_key)) {
    PackedColorCandidate candidate;
    candidate.color                    = make_uchar4(blue, green, red, 0);
    candidate.quality                  = quality;
    candidate.view_index               = static_cast<uint16_t>(view_index);
    candidate.reservoir_key            = reservoir_key;
    point_candidates[replacement_slot] = candidate;
  }
  const unsigned char old_count = color_view_counts[point_index];
  if (old_count < static_cast<unsigned char>(max_view_count)) {
    color_view_counts[point_index] = old_count + 1;
  }
}

__global__ void FinalizeFusedColorsKernel(
    const PackedColorCandidate* consensus_candidates,
    const PackedSharpCandidate* sharp_candidates,
    const float2* exposure_log_luminance,
    const unsigned char* color_view_counts,
    int point_count,
    bool smooth_fusion,
    uchar4* output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const unsigned char count = color_view_counts[i];
  if (count == 0) {
    output[i] = make_uchar4(0, 0, 0, 0);
    return;
  }

  const float2 exposure = exposure_log_luminance[i];
  const float target_luma =
      exposure.y > 0.0f ? __expf(exposure.x / exposure.y) - 1.0f : 0.0f;
  const PackedColorCandidate* candidates =
      consensus_candidates + i * kConsensusCandidateCount;
  int slots[kConsensusCandidateCount];
  float weights[kConsensusCandidateCount];
  float3 normalized_colors[kConsensusCandidateCount];
  int candidate_count = 0;
  for (int slot = 0; slot < kConsensusCandidateCount; ++slot) {
    const PackedColorCandidate candidate = candidates[slot];
    if (candidate.quality == 0) {
      continue;
    }
    const float blue  = static_cast<float>(candidate.color.x);
    const float green = static_cast<float>(candidate.color.y);
    const float red   = static_cast<float>(candidate.color.z);
    const float luma  = 0.114f * blue + 0.587f * green + 0.299f * red;
    float gain        = 1.0f;
    if (target_luma > 0.0f && luma > 1.0f) {
      gain = fminf(1.333333f, fmaxf(0.75f, target_luma / luma));
    }
    slots[candidate_count] = slot;
    weights[candidate_count] =
        static_cast<float>(candidate.quality) / 65535.0f;
    normalized_colors[candidate_count] =
        make_float3(blue * gain, green * gain, red * gain);
    ++candidate_count;
  }
  if (candidate_count == 0) {
    output[i] = make_uchar4(0, 0, 0, 0);
    return;
  }

  // Find the weighted color medoid. The first comparison is support count so
  // one geometrically excellent but statistically isolated projection cannot
  // beat several mutually consistent observations.
  constexpr float kDistanceSquared =
      kConsensusColorDistance * kConsensusColorDistance;
  constexpr float kSharpBestDistanceSquared =
      kSharpBestColorDistance * kSharpBestColorDistance;
  int medoid_index           = 0;
  int medoid_support_count   = 0;
  float medoid_support       = -1.0f;
  float medoid_distance_cost = FLT_MAX;
  for (int candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    int support_count   = 0;
    float support       = 0.0f;
    float distance_cost = 0.0f;
    const float3 color  = normalized_colors[candidate_index];
    for (int other_index = 0; other_index < candidate_count; ++other_index) {
      const float3 other      = normalized_colors[other_index];
      const float blue_delta  = color.x - other.x;
      const float green_delta = color.y - other.y;
      const float red_delta   = color.z - other.z;
      const float distance_squared =
          (blue_delta * blue_delta + green_delta * green_delta +
           red_delta * red_delta) /
          3.0f;
      if (distance_squared <= kDistanceSquared) {
        ++support_count;
        // Retain geometric quality as a tie breaker without allowing it to
        // outweigh the number of independent observations.
        const float support_weight =
            0.25f + 0.75f * weights[other_index];
        support += support_weight;
        distance_cost += support_weight * distance_squared;
      }
    }
    if (support_count > medoid_support_count ||
        (support_count == medoid_support_count &&
         (support > medoid_support ||
          (support == medoid_support &&
           distance_cost < medoid_distance_cost)))) {
      medoid_index         = candidate_index;
      medoid_support_count = support_count;
      medoid_support       = support;
      medoid_distance_cost = distance_cost;
    }
  }

  // Three observations can establish a 2:1 majority. Four observations need
  // at least 3:1; a 2:2 split is deliberately marked uncertain and omitted.
  if (candidate_count >= 3 &&
      medoid_support_count * 5 < candidate_count * 3) {
    output[i] = make_uchar4(0, 0, 0, 0);
    return;
  }

  // Use the sharpest real source pixel inside the winning cluster, never an
  // average of RGB values.
  int selected_index        = medoid_index;
  float selected_quality    = -1.0f;
  const float3 medoid_color = normalized_colors[medoid_index];
  for (int candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    const float3 color      = normalized_colors[candidate_index];
    const float blue_delta  = color.x - medoid_color.x;
    const float green_delta = color.y - medoid_color.y;
    const float red_delta   = color.z - medoid_color.z;
    const float distance_squared =
        (blue_delta * blue_delta + green_delta * green_delta +
         red_delta * red_delta) /
        3.0f;
    if (distance_squared <= kSharpBestDistanceSquared &&
        weights[candidate_index] > selected_quality) {
      selected_index   = candidate_index;
      selected_quality = weights[candidate_index];
    }
  }

  const PackedColorCandidate selected_candidate =
      candidates[slots[selected_index]];
  float selected_blue  = static_cast<float>(selected_candidate.color.x);
  float selected_green = static_cast<float>(selected_candidate.color.y);
  float selected_red   = static_cast<float>(selected_candidate.color.z);

  if (smooth_fusion) {
    // Average only the exposure-normalized observations that agree tightly
    // with the winning medoid. This removes per-point source-view switching
    // while retaining robust rejection of foreground and occlusion leaks.
    float weight_sum = 0.0f;
    float blue_sum   = 0.0f;
    float green_sum  = 0.0f;
    float red_sum    = 0.0f;
    for (int candidate_index = 0; candidate_index < candidate_count;
         ++candidate_index) {
      const float3 color      = normalized_colors[candidate_index];
      const float blue_delta  = color.x - medoid_color.x;
      const float green_delta = color.y - medoid_color.y;
      const float red_delta   = color.z - medoid_color.z;
      const float distance_squared =
          (blue_delta * blue_delta + green_delta * green_delta +
           red_delta * red_delta) /
          3.0f;
      if (distance_squared > kSharpBestDistanceSquared) {
        continue;
      }
      const float sample_weight = 0.25f + 0.75f * weights[candidate_index];
      weight_sum += sample_weight;
      blue_sum += sample_weight * color.x;
      green_sum += sample_weight * color.y;
      red_sum += sample_weight * color.z;
    }
    if (weight_sum > 0.0f) {
      selected_blue  = blue_sum / weight_sum;
      selected_green = green_sum / weight_sum;
      selected_red   = red_sum / weight_sum;
    }
  } else {
    // Select the sharpest source observation that agrees tightly with the
    // stable multi-view appearance.  Keeping three hypotheses avoids losing
    // detail when the highest-scoring one is a person or an occlusion leak.
    const PackedSharpCandidate* point_sharp_candidates =
        sharp_candidates + i * kSharpCandidateCount;
    for (int sharp_index = 0; sharp_index < kSharpCandidateCount;
         ++sharp_index) {
      const PackedSharpCandidate sharp_candidate =
          point_sharp_candidates[sharp_index];
      if (sharp_candidate.score == 0) {
        continue;
      }
      const float sharp_blue  = static_cast<float>(sharp_candidate.color.x);
      const float sharp_green = static_cast<float>(sharp_candidate.color.y);
      const float sharp_red   = static_cast<float>(sharp_candidate.color.z);
      const float sharp_luma =
          0.114f * sharp_blue + 0.587f * sharp_green + 0.299f * sharp_red;
      float sharp_gain = 1.0f;
      if (target_luma > 0.0f && sharp_luma > 1.0f) {
        sharp_gain =
            fminf(1.333333f, fmaxf(0.75f, target_luma / sharp_luma));
      }
      const float blue_delta  = sharp_blue * sharp_gain - medoid_color.x;
      const float green_delta = sharp_green * sharp_gain - medoid_color.y;
      const float red_delta   = sharp_red * sharp_gain - medoid_color.z;
      const float distance_squared =
          (blue_delta * blue_delta + green_delta * green_delta +
           red_delta * red_delta) /
          3.0f;
      if (distance_squared <= kSharpBestDistanceSquared) {
        selected_blue  = sharp_blue;
        selected_green = sharp_green;
        selected_red   = sharp_red;
        break;
      }
    }
  }

  const float selected_luma = 0.114f * selected_blue +
                              0.587f * selected_green +
                              0.299f * selected_red;
  float exposure_gain       = 1.0f;
  if (!smooth_fusion && target_luma > 0.0f && selected_luma > 1.0f) {
    // Keep the correction conservative so different material reflectances or
    // small occlusion errors cannot create large brightness halos.
    exposure_gain =
        fminf(1.333333f, fmaxf(0.75f, target_luma / selected_luma));
  }
  const unsigned char blue = static_cast<unsigned char>(
      min(255, max(0, __float2int_rn(selected_blue * exposure_gain))));
  const unsigned char green = static_cast<unsigned char>(
      min(255, max(0, __float2int_rn(selected_green * exposure_gain))));
  const unsigned char red = static_cast<unsigned char>(
      min(255, max(0, __float2int_rn(selected_red * exposure_gain))));
  output[i] = make_uchar4(blue, green, red, count);
}

CudaDepthRenderer::~CudaDepthRenderer() = default;

size_t CudaDepthRenderer::PointCount() const { return impl_->point_count; }

DepthRenderResult CudaDepthRenderer::RenderProjection(
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
    int color_view_index) const {
  DepthRenderResult result;
  result.width  = width;
  result.height = height;
  if (copy_depth) {
    result.depth.assign(static_cast<size_t>(result.width) * result.height, 0.0f);
  }
  if (visible_points != nullptr) {
    visible_points->clear();
  }
  const bool fuse_colors = color_image != nullptr;
  if (fuse_colors &&
      (visible_points != nullptr || color_image_width <= 0 ||
       color_image_height <= 0 || max_view_count <= 0 ||
       max_view_count > 255 || color_view_index < 0 ||
       color_view_index > 65535 ||
       color_image_row_stride < static_cast<size_t>(color_image_width) * 3 ||
       visibility_source_width != color_image_width ||
       visibility_source_height != color_image_height)) {
    throw std::invalid_argument("Invalid CUDA color-fusion inputs");
  }
  if (impl_->point_count == 0) {
    return result;
  }

  if (fuse_colors) {
    if (impl_->consensus_candidates == nullptr ||
        impl_->sharp_candidates == nullptr ||
        impl_->exposure_log_luminance == nullptr ||
        impl_->color_view_counts == nullptr) {
      ResetFusedColors();
    }
    const size_t required_color_bytes =
        static_cast<size_t>(color_image_width) * color_image_height * 3;
    if (impl_->color_image_capacity < required_color_bytes) {
      cudaFree(impl_->color_image);
      impl_->color_image = nullptr;
      CheckCuda(cudaMalloc(&impl_->color_image, required_color_bytes),
                "cudaMalloc(color_image)");
      impl_->color_image_capacity = required_color_bytes;
    }
    CheckCuda(cudaMemcpy2D(impl_->color_image,
                           static_cast<size_t>(color_image_width) * 3,
                           color_image,
                           color_image_row_stride,
                           static_cast<size_t>(color_image_width) * 3,
                           color_image_height,
                           cudaMemcpyHostToDevice),
              "cudaMemcpy2D(color_image)");
  }

  const int pixel_count       = result.width * result.height;
  const int threads           = 256;
  const int pixel_blocks      = (pixel_count + threads - 1) / threads;
  const unsigned int inf_bits = 0x7f800000u;
  const int max_radius =
      static_cast<int>(std::ceil(std::hypot(result.width, result.height)));
  const size_t chunk_capacity =
      std::min(impl_->point_count, static_cast<size_t>(gpu_chunk_points));
  DeviceBuffer<float3> d_filtered_points(chunk_capacity);
  DeviceBuffer<int> d_center_x(chunk_capacity);
  DeviceBuffer<int> d_center_y(chunk_capacity);
  DeviceBuffer<int> d_radius(chunk_capacity);
  DeviceBuffer<int> d_filtered_count(1);
  const bool collect_visibility = visible_points != nullptr;
  DeviceBuffer<unsigned int> d_flat_depth(pixel_count);
  DeviceBuffer<float> d_depth(copy_depth ? pixel_count : 0);
  DeviceBuffer<int> d_count(1);

  FaceTransform transform{};
  std::copy(rotation.begin(), rotation.end(), transform.rotation);
  std::copy(translation.begin(), translation.end(), transform.translation);
  ProjectionIntrinsics intrinsics{};
  intrinsics.model  = fisheye ? ProjectionModel::kOpenCVFisheye
                              : ProjectionModel::kPerspective;
  intrinsics.width  = width;
  intrinsics.height = height;
  intrinsics.fx     = projection_params[0];
  intrinsics.fy     = projection_params[1];
  intrinsics.cx     = projection_params[2];
  intrinsics.cy     = projection_params[3];
  intrinsics.k1     = projection_params[4];
  intrinsics.k2     = projection_params[5];
  intrinsics.k3     = projection_params[6];
  intrinsics.k4     = projection_params[7];
  const double voxel_circumradius_double =
      static_cast<double>(voxel_size) * std::sqrt(3.0) * 0.5;
  const double max_distance_sq =
      max_distance > 0.0f
          ? (max_distance + voxel_circumradius_double) *
                (max_distance + voxel_circumradius_double)
          : std::numeric_limits<double>::infinity();

  auto transform_and_project_chunk = [&](size_t offset, int chunk_count) {
    CheckCuda(cudaMemset(d_filtered_count.get(), 0, sizeof(int)),
              "cudaMemset(filtered_count)");
    const int chunk_blocks = (chunk_count + threads - 1) / threads;
    TransformProjectPointsKernel<<<chunk_blocks, threads>>>(
        impl_->world_points + offset,
        chunk_count,
        transform,
        intrinsics,
        voxel_circumradius_double,
        max_distance_sq,
        max_radius,
        d_filtered_points.get(),
        d_center_x.get(),
        d_center_y.get(),
        d_radius.get(),
        d_filtered_count.get());
    CheckCuda(cudaGetLastError(), "TransformProjectPointsKernel");

    int filtered_count = 0;
    CheckCuda(cudaMemcpy(&filtered_count,
                         d_filtered_count.get(),
                         sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(filtered_count)");
    if (filtered_count == 0) {
      return 0;
    }
    return filtered_count;
  };

  FillBitsKernel<<<pixel_blocks, threads>>>(d_flat_depth.get(), pixel_count, inf_bits);
  CheckCuda(cudaGetLastError(), "FillBitsKernel(flat_depth)");
  for (size_t offset = 0; offset < impl_->point_count; offset += chunk_capacity) {
    const int chunk_count = static_cast<int>(
        std::min(chunk_capacity, impl_->point_count - offset));
    const int filtered_count = transform_and_project_chunk(offset, chunk_count);
    if (filtered_count == 0) {
      continue;
    }
    const int filtered_blocks = (filtered_count + threads - 1) / threads;
    SplatMinDepthKernel<<<filtered_blocks, threads>>>(d_filtered_points.get(),
                                                      d_center_x.get(),
                                                      d_center_y.get(),
                                                      d_radius.get(),
                                                      filtered_count,
                                                      result.width,
                                                      result.height,
                                                      d_flat_depth.get());
    CheckCuda(cudaGetLastError(), "SplatMinDepthKernel");
  }

  CheckCuda(cudaMemset(d_count.get(), 0, sizeof(int)), "cudaMemset(count)");
  CountFiniteKernel<<<pixel_blocks, threads>>>(
      d_flat_depth.get(), pixel_count, inf_bits, d_count.get());
  CheckCuda(cudaGetLastError(), "CountFiniteKernel");
  CheckCuda(cudaMemcpy(&result.contributing_count,
                       d_count.get(),
                       sizeof(int),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(contributing_count)");

  if (!sparse_mode) {
    if (copy_depth) {
      DenseDepthToFloatKernel<<<pixel_blocks, threads>>>(
          d_flat_depth.get(), pixel_count, inf_bits, d_depth.get());
      CheckCuda(cudaGetLastError(), "DenseDepthToFloatKernel");
    }
    result.auxiliary_count = result.contributing_count;
  } else {
    DeviceBuffer<unsigned char> d_visible(chunk_capacity);
    DeviceBuffer<unsigned int> d_sparse(pixel_count);
    FillBitsKernel<<<pixel_blocks, threads>>>(d_sparse.get(), pixel_count, inf_bits);
    CheckCuda(cudaGetLastError(), "FillBitsKernel(sparse)");
    CheckCuda(cudaMemset(d_count.get(), 0, sizeof(int)), "cudaMemset(count sparse)");
    for (size_t offset = 0; offset < impl_->point_count; offset += chunk_capacity) {
      const int chunk_count = static_cast<int>(
          std::min(chunk_capacity, impl_->point_count - offset));
      const int filtered_count = transform_and_project_chunk(offset, chunk_count);
      if (filtered_count == 0) {
        continue;
      }
      const int filtered_blocks = (filtered_count + threads - 1) / threads;
      CheckCuda(cudaMemset(d_visible.get(),
                           0,
                           static_cast<size_t>(filtered_count) * sizeof(unsigned char)),
                "cudaMemset(visible)");
      MarkVisiblePointsKernel<<<filtered_blocks, threads>>>(d_filtered_points.get(),
                                                            d_center_x.get(),
                                                            d_center_y.get(),
                                                            d_radius.get(),
                                                            filtered_count,
                                                            result.width,
                                                            result.height,
                                                            d_flat_depth.get(),
                                                            d_visible.get());
      CheckCuda(cudaGetLastError(), "MarkVisiblePointsKernel");
      CountVisiblePointsKernel<<<filtered_blocks, threads>>>(
          d_visible.get(), filtered_count, d_count.get());
      CheckCuda(cudaGetLastError(), "CountVisiblePointsKernel");
      SparseCenterDepthKernel<<<filtered_blocks, threads>>>(d_filtered_points.get(),
                                                            d_center_x.get(),
                                                            d_center_y.get(),
                                                            d_visible.get(),
                                                            filtered_count,
                                                            result.width,
                                                            result.height,
                                                            d_sparse.get());
      CheckCuda(cudaGetLastError(), "SparseCenterDepthKernel");
    }
    CheckCuda(cudaMemcpy(&result.auxiliary_count,
                         d_count.get(),
                         sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(visible_count)");
    if (copy_depth) {
      DenseDepthToFloatKernel<<<pixel_blocks, threads>>>(
          d_sparse.get(), pixel_count, inf_bits, d_depth.get());
      CheckCuda(cudaGetLastError(), "SparseDepthToFloatKernel");
    }
  }

  if (collect_visibility) {
    DeviceBuffer<VisiblePointProjection> d_visible_points(chunk_capacity);
    DeviceBuffer<int> d_visible_count(1);
    for (size_t offset = 0; offset < impl_->point_count; offset += chunk_capacity) {
      const int chunk_count = static_cast<int>(
          std::min(chunk_capacity, impl_->point_count - offset));
      CheckCuda(cudaMemset(d_visible_count.get(), 0, sizeof(int)),
                "cudaMemset(visible candidate count)");
      const int chunk_blocks = (chunk_count + threads - 1) / threads;
      CollectDepthVisiblePointsKernel<<<chunk_blocks, threads>>>(
          impl_->world_points + offset,
          impl_->world_normals != nullptr ? impl_->world_normals + offset : nullptr,
          chunk_count,
          static_cast<int>(offset),
          transform,
          intrinsics,
          max_distance_sq,
          visibility_source_width,
          visibility_source_height,
          impl_->visibility_mask,
          impl_->visibility_mask_width,
          impl_->visibility_mask_height,
          result.width,
          result.height,
          d_flat_depth.get(),
          inf_bits,
          visibility_tolerance,
          d_visible_points.get(),
          d_visible_count.get());
      CheckCuda(cudaGetLastError(), "CollectDepthVisiblePointsKernel");
      int visible_count = 0;
      CheckCuda(cudaMemcpy(&visible_count,
                           d_visible_count.get(),
                           sizeof(int),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy(visible candidate count)");
      if (visible_count == 0) {
        continue;
      }
      const size_t old_size = visible_points->size();
      visible_points->resize(old_size + static_cast<size_t>(visible_count));
      CheckCuda(cudaMemcpy(visible_points->data() + old_size,
                           d_visible_points.get(),
                           static_cast<size_t>(visible_count) *
                               sizeof(VisiblePointProjection),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy(visible candidates)");
    }
    result.auxiliary_count = static_cast<int>(visible_points->size());
  } else if (fuse_colors) {
    for (size_t offset = 0; offset < impl_->point_count; offset += chunk_capacity) {
      const int chunk_count = static_cast<int>(
          std::min(chunk_capacity, impl_->point_count - offset));
      const int chunk_blocks = (chunk_count + threads - 1) / threads;
      FuseDepthVisibleColorsKernel<<<chunk_blocks, threads>>>(
          impl_->world_points + offset,
          impl_->world_normals != nullptr ? impl_->world_normals + offset
                                          : nullptr,
          chunk_count,
          static_cast<int>(offset),
          transform,
          intrinsics,
          max_distance_sq,
          color_image_width,
          color_image_height,
          impl_->color_image,
          impl_->visibility_mask,
          impl_->visibility_mask_width,
          impl_->visibility_mask_height,
          result.width,
          result.height,
          d_flat_depth.get(),
          inf_bits,
          visibility_tolerance,
          max_view_count,
          color_view_index,
          impl_->consensus_candidates,
          impl_->sharp_candidates,
          impl_->exposure_log_luminance,
          impl_->color_view_counts);
      CheckCuda(cudaGetLastError(), "FuseDepthVisibleColorsKernel");
    }
  }

  if (copy_depth) {
    CheckCuda(cudaMemcpy(result.depth.data(),
                         d_depth.get(),
                         pixel_count * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(depth)");
  }
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  return result;
}

DepthRenderResult CudaDepthRenderer::Render(
    const std::array<double, 9>& rotation,
    const std::array<double, 3>& translation,
    const FaceIntrinsics& intrinsics,
    float voxel_size,
    int gpu_chunk_points,
    float max_distance,
    bool sparse_mode) const {
  return RenderProjection(rotation,
                          translation,
                          intrinsics.width,
                          intrinsics.height,
                          {intrinsics.focal,
                           intrinsics.focal,
                           intrinsics.cx,
                           intrinsics.cy,
                           0.0,
                           0.0,
                           0.0,
                           0.0},
                          false,
                          voxel_size,
                          gpu_chunk_points,
                          max_distance,
                          sparse_mode,
                          0,
                          0,
                          0.0f,
                          nullptr,
                          true,
                          nullptr,
                          0,
                          0,
                          0,
                          0,
                          -1);
}

DepthRenderResult CudaDepthRenderer::RenderFisheye(
    const std::array<double, 9>& rotation,
    const std::array<double, 3>& translation,
    const OpenCVFisheyeIntrinsics& intrinsics,
    float voxel_size,
    int gpu_chunk_points,
    float max_distance,
    bool sparse_mode) const {
  return RenderProjection(rotation,
                          translation,
                          intrinsics.width,
                          intrinsics.height,
                          {intrinsics.fx,
                           intrinsics.fy,
                           intrinsics.cx,
                           intrinsics.cy,
                           intrinsics.k1,
                           intrinsics.k2,
                           intrinsics.k3,
                           intrinsics.k4},
                          true,
                          voxel_size,
                          gpu_chunk_points,
                          max_distance,
                          sparse_mode,
                          0,
                          0,
                          0.0f,
                          nullptr,
                          true,
                          nullptr,
                          0,
                          0,
                          0,
                          0,
                          -1);
}

FisheyeVisibilityResult CudaDepthRenderer::RenderFisheyeVisibility(
    const std::array<double, 9>& rotation,
    const std::array<double, 3>& translation,
    const OpenCVFisheyeIntrinsics& intrinsics,
    int source_width,
    int source_height,
    float voxel_size,
    int gpu_chunk_points,
    float max_distance,
    float visibility_tolerance,
    bool copy_depth) const {
  if (source_width <= 0 || source_height <= 0 ||
      visibility_tolerance < 0.0f) {
    throw std::invalid_argument("Invalid fisheye visibility options");
  }
  FisheyeVisibilityResult result;
  DepthRenderResult depth = RenderProjection(
      rotation,
      translation,
      intrinsics.width,
      intrinsics.height,
      {intrinsics.fx,
       intrinsics.fy,
       intrinsics.cx,
       intrinsics.cy,
       intrinsics.k1,
       intrinsics.k2,
       intrinsics.k3,
       intrinsics.k4},
      true,
      voxel_size,
      gpu_chunk_points,
      max_distance,
      false,
      source_width,
      source_height,
      visibility_tolerance,
      &result.visible_points,
      copy_depth,
      nullptr,
      0,
      0,
      0,
      0,
      -1);
  result.depth              = std::move(depth.depth);
  result.width              = depth.width;
  result.height             = depth.height;
  result.contributing_count = depth.contributing_count;
  return result;
}

FisheyeColoringResult CudaDepthRenderer::RenderFisheyeColor(
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
    bool copy_depth) const {
  if (bgr_data == nullptr || source_width <= 0 || source_height <= 0 ||
      source_row_stride < static_cast<size_t>(source_width) * 3 ||
      visibility_tolerance < 0.0f || max_view_count <= 0 ||
      max_view_count > 255 || view_index < 0 || view_index > 65535) {
    throw std::invalid_argument("Invalid fisheye color-fusion options");
  }
  FisheyeColoringResult result;
  DepthRenderResult depth = RenderProjection(
      rotation,
      translation,
      intrinsics.width,
      intrinsics.height,
      {intrinsics.fx,
       intrinsics.fy,
       intrinsics.cx,
       intrinsics.cy,
       intrinsics.k1,
       intrinsics.k2,
       intrinsics.k3,
       intrinsics.k4},
      true,
      voxel_size,
      gpu_chunk_points,
      max_distance,
      false,
      source_width,
      source_height,
      visibility_tolerance,
      nullptr,
      copy_depth,
      bgr_data,
      source_width,
      source_height,
      source_row_stride,
      max_view_count,
      view_index);
  result.depth              = std::move(depth.depth);
  result.width              = depth.width;
  result.height             = depth.height;
  result.contributing_count = depth.contributing_count;
  return result;
}

}  // namespace xsfm_post
