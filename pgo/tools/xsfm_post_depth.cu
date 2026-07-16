#include "xsfm_post_depth.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace xsfm_post {

struct FaceIntrinsics {
  int width = 0;
  int height = 0;
  double focal = 0.0;
  double cx = 0.0;
  double cy = 0.0;
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

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const { return data_; }

 private:
  T* data_ = nullptr;
};

struct FaceTransform {
  double rotation[9];
  double translation[3];
};

__global__ void TransformFilterPointsKernel(const float3* world_points,
                                            int point_count,
                                            FaceTransform transform,
                                            double x_min,
                                            double x_max,
                                            double y_min,
                                            double y_max,
                                            double left_margin,
                                            double right_margin,
                                            double top_margin,
                                            double bottom_margin,
                                            double voxel_circumradius,
                                            double max_distance_sq,
                                            double focal,
                                            double cx,
                                            double cy,
                                            int width,
                                            int height,
                                            float3* filtered_points,
                                            int* filtered_count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }

  const float3 world = world_points[i];
  const double wx = static_cast<double>(world.x);
  const double wy = static_cast<double>(world.y);
  const double wz = static_cast<double>(world.z);
  const double x = transform.rotation[0] * wx + transform.rotation[1] * wy +
                   transform.rotation[2] * wz + transform.translation[0];
  const double y = transform.rotation[3] * wx + transform.rotation[4] * wy +
                   transform.rotation[5] * wz + transform.translation[1];
  const double z = transform.rotation[6] * wx + transform.rotation[7] * wy +
                   transform.rotation[8] * wz + transform.translation[2];
  if (!isfinite(x) || !isfinite(y) || !isfinite(z) ||
      z <= -voxel_circumradius || x * x + y * y + z * z > max_distance_sq ||
      x - x_min * z < -left_margin || -x + x_max * z < -right_margin ||
      y - y_min * z < -top_margin || -y + y_max * z < -bottom_margin || z <= 0.0) {
    return;
  }

  const double u = focal * x / z + cx;
  const double v = focal * y / z + cy;
  const double radius = focal * voxel_circumradius / z;
  if (u + radius < 0.0 || u - radius >= width ||
      v + radius < 0.0 || v - radius >= height) {
    return;
  }

  const int output_index = atomicAdd(filtered_count, 1);
  filtered_points[output_index] =
      make_float3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

__global__ void ProjectPointsKernel(const float3* points,
                                    int point_count,
                                    float focal,
                                    float cx,
                                    float cy,
                                    float voxel_circumradius,
                                    int max_radius,
                                    int* center_x,
                                    int* center_y,
                                    int* radius) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const float3 p = points[i];
  const float u = focal * p.x / p.z + cx;
  const float v = focal * p.y / p.z + cy;
  center_x[i] = __float2int_rn(u - 0.5f);
  center_y[i] = __float2int_rn(v - 0.5f);
  const int r = static_cast<int>(ceilf(focal * voxel_circumradius / p.z));
  radius[i] = min(max(r, 0), max_radius);
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
  const int cx = center_x[i];
  const int cy = center_y[i];
  const int r = radius[i];
  const float z = points[i].z;
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
  const int cx = center_x[i];
  const int cy = center_y[i];
  const int r = radius[i];
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

}  // namespace

bool HasCudaDevice() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  return status == cudaSuccess && count > 0;
}

struct CudaDepthRenderer::Impl {
  float3* world_points = nullptr;
  size_t point_count = 0;

  ~Impl() { cudaFree(world_points); }
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

  std::vector<float3> upload_points;
  upload_points.reserve(world_points.size());
  for (const DepthWorldPoint& point : world_points) {
    upload_points.push_back(make_float3(point.x, point.y, point.z));
  }
  CheckCuda(cudaMalloc(&impl_->world_points, upload_points.size() * sizeof(float3)),
            "cudaMalloc(world_points)");
  CheckCuda(cudaMemcpy(impl_->world_points,
                       upload_points.data(),
                       upload_points.size() * sizeof(float3),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(world_points)");
}

CudaDepthRenderer::~CudaDepthRenderer() = default;

size_t CudaDepthRenderer::PointCount() const { return impl_->point_count; }

DepthRenderResult CudaDepthRenderer::Render(
    const std::array<double, 9>& rotation,
    const std::array<double, 3>& translation,
    const FaceIntrinsics& intrinsics,
    float voxel_size,
    int gpu_chunk_points,
    float max_distance,
    bool sparse_mode) const {
  DepthRenderResult result;
  result.width = intrinsics.width;
  result.height = intrinsics.height;
  result.depth.assign(static_cast<size_t>(result.width) * result.height, 0.0f);
  if (impl_->point_count == 0) {
    return result;
  }

  const int pixel_count = result.width * result.height;
  const int threads = 256;
  const int pixel_blocks = (pixel_count + threads - 1) / threads;
  const unsigned int inf_bits = 0x7f800000u;
  const int max_radius =
      static_cast<int>(std::ceil(std::hypot(result.width, result.height)));
  const float voxel_circumradius = voxel_size * std::sqrt(3.0f) * 0.5f;

  const size_t chunk_capacity =
      std::min(impl_->point_count, static_cast<size_t>(gpu_chunk_points));
  DeviceBuffer<float3> d_filtered_points(chunk_capacity);
  DeviceBuffer<int> d_center_x(chunk_capacity);
  DeviceBuffer<int> d_center_y(chunk_capacity);
  DeviceBuffer<int> d_radius(chunk_capacity);
  DeviceBuffer<int> d_filtered_count(1);
  DeviceBuffer<unsigned int> d_flat_depth(pixel_count);
  DeviceBuffer<float> d_depth(pixel_count);
  DeviceBuffer<int> d_count(1);

  FaceTransform transform{};
  std::copy(rotation.begin(), rotation.end(), transform.rotation);
  std::copy(translation.begin(), translation.end(), transform.translation);
  const double voxel_circumradius_double =
      static_cast<double>(voxel_size) * std::sqrt(3.0) * 0.5;
  const double x_min = (0.0 - intrinsics.cx) / intrinsics.focal;
  const double x_max =
      (static_cast<double>(intrinsics.width) - intrinsics.cx) / intrinsics.focal;
  const double y_min = (0.0 - intrinsics.cy) / intrinsics.focal;
  const double y_max =
      (static_cast<double>(intrinsics.height) - intrinsics.cy) / intrinsics.focal;
  const double left_margin =
      voxel_circumradius_double * std::sqrt(1.0 + x_min * x_min);
  const double right_margin =
      voxel_circumradius_double * std::sqrt(1.0 + x_max * x_max);
  const double top_margin =
      voxel_circumradius_double * std::sqrt(1.0 + y_min * y_min);
  const double bottom_margin =
      voxel_circumradius_double * std::sqrt(1.0 + y_max * y_max);
  const double max_distance_sq =
      max_distance > 0.0f
          ? (max_distance + voxel_circumradius_double) *
                (max_distance + voxel_circumradius_double)
          : std::numeric_limits<double>::infinity();

  auto transform_and_project_chunk = [&](size_t offset, int chunk_count) {
    CheckCuda(cudaMemset(d_filtered_count.get(), 0, sizeof(int)),
              "cudaMemset(filtered_count)");
    const int chunk_blocks = (chunk_count + threads - 1) / threads;
    TransformFilterPointsKernel<<<chunk_blocks, threads>>>(
        impl_->world_points + offset,
        chunk_count,
        transform,
        x_min,
        x_max,
        y_min,
        y_max,
        left_margin,
        right_margin,
        top_margin,
        bottom_margin,
        voxel_circumradius_double,
        max_distance_sq,
        intrinsics.focal,
        intrinsics.cx,
        intrinsics.cy,
        intrinsics.width,
        intrinsics.height,
        d_filtered_points.get(),
        d_filtered_count.get());
    CheckCuda(cudaGetLastError(), "TransformFilterPointsKernel");

    int filtered_count = 0;
    CheckCuda(cudaMemcpy(&filtered_count,
                         d_filtered_count.get(),
                         sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(filtered_count)");
    if (filtered_count == 0) {
      return 0;
    }
    const int filtered_blocks = (filtered_count + threads - 1) / threads;
    ProjectPointsKernel<<<filtered_blocks, threads>>>(
        d_filtered_points.get(),
        filtered_count,
        static_cast<float>(intrinsics.focal),
        static_cast<float>(intrinsics.cx),
        static_cast<float>(intrinsics.cy),
        voxel_circumradius,
        max_radius,
        d_center_x.get(),
        d_center_y.get(),
        d_radius.get());
    CheckCuda(cudaGetLastError(), "ProjectPointsKernel");
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
    DenseDepthToFloatKernel<<<pixel_blocks, threads>>>(
        d_flat_depth.get(), pixel_count, inf_bits, d_depth.get());
    CheckCuda(cudaGetLastError(), "DenseDepthToFloatKernel");
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
    DenseDepthToFloatKernel<<<pixel_blocks, threads>>>(
        d_sparse.get(), pixel_count, inf_bits, d_depth.get());
    CheckCuda(cudaGetLastError(), "SparseDepthToFloatKernel");
  }

  CheckCuda(cudaMemcpy(result.depth.data(),
                       d_depth.get(),
                       pixel_count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(depth)");
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  return result;
}

}  // namespace xsfm_post
