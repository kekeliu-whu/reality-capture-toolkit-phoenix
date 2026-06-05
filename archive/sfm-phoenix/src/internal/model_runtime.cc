#include "sfm_phoenix/internal/model_runtime.h"

#include "sfm_phoenix/internal/trt_engine.h"

#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdexcept>

namespace sfm_phoenix {

namespace {

int SelectPhoenixEngineBatchBucket(const int batch_size) {
  if (batch_size <= 1) {
    return 1;
  }
  if (batch_size <= 4) {
    return 4;
  }
  return 8;
}

std::string PhoenixBatchEngineName(const std::string& base_name,
                                   const int batch_size) {
  return base_name + "_b" + std::to_string(batch_size) + ".engine";
}

std::filesystem::path EnsureRuntimeEngine(
    const std::string& onnx_filename,
    const std::string& engine_filename,
    const TrtBuildOptions& build_options) {
  const auto runtime_dir = GetExecutableDirectory();
  const auto onnx_path = runtime_dir / onnx_filename;
  const auto engine_path = runtime_dir / engine_filename;

  if (std::filesystem::exists(engine_path)) {
    return engine_path;
  }

  if (!std::filesystem::exists(onnx_path)) {
    throw std::runtime_error("Bundled ONNX model not found: " +
                             onnx_path.string());
  }

  if (!BuildSerializedEngine(onnx_path, engine_path, build_options)) {
    throw std::runtime_error("Failed to build TensorRT engine from: " +
                             onnx_path.string());
  }

  spdlog::info("Built TensorRT engine: {}", engine_path.string());
  return engine_path;
}

}  // namespace

std::filesystem::path GetExecutableDirectory() {
#ifdef _WIN32
  std::wstring buffer(MAX_PATH, L'\0');
  DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                    static_cast<DWORD>(buffer.size()));
  while (length == buffer.size()) {
    buffer.resize(buffer.size() * 2);
    length = GetModuleFileNameW(nullptr, buffer.data(),
                                static_cast<DWORD>(buffer.size()));
  }
  if (length == 0) {
    throw std::runtime_error("GetModuleFileNameW failed");
  }
  buffer.resize(length);
  return std::filesystem::path(buffer).parent_path();
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path EnsurePhoenixBackboneEngine(const int batch_size) {
  const int engine_batch = SelectPhoenixEngineBatchBucket(batch_size);
  TrtBuildOptions options;
  options.enable_fp16 = false; // ALIKED backbone is more stable in FP32 (some ops produce NaNs in FP16)
  options.builder_optimization_level = 3;
  // Cap max spatial resolution at 1024 to keep workspace requirements within
  // 8 GB. Phoenix typical usage is --Phoenix.max_edge 1024; higher resolutions
  // cause /Div node tactics to require >15 GB workspace on this GPU.
  constexpr int kBackboneMaxEdge = 1024;
  options.profile_shapes = {
    {"image", {1, 3, 320, 320}, {engine_batch, 3, kBackboneMaxEdge, kBackboneMaxEdge},
     {engine_batch, 3, kBackboneMaxEdge, kBackboneMaxEdge}},
  };
  return EnsureRuntimeEngine(
      "aliked_backbone.onnx",
      PhoenixBatchEngineName("aliked_backbone", engine_batch),
      options);
}

std::filesystem::path EnsurePhoenixSddhEngine() {
  TrtBuildOptions options;
  options.enable_fp16 = true;
  options.builder_optimization_level = 3;
  options.profile_shapes = {
      {"feature_map", {1, 128, 320, 320}, {1, 128, 1024, 1024}, {1, 128, 1600, 1600}},
      {"keypoints_wh", {100, 2}, {5000, 2}, {5000, 2}},
      {"feature_map_hw", {2}, {2}, {2}},
  };
  return EnsureRuntimeEngine("aliked_sddh.onnx", "aliked_sddh.engine", options);
}

std::filesystem::path EnsurePhoenixLightGlueEngine(const int batch_size) {
  const int engine_batch = SelectPhoenixEngineBatchBucket(batch_size);
  TrtBuildOptions options;
  options.enable_fp16 = true;
  options.builder_optimization_level = 3;
  options.detailed_profiling = true;  // Allows trtexec --exportLayerInfo
  // Phoenix feature matching uses separate engines for batch buckets
  // {1, 4, 8} to avoid loading the widest LightGlue context for batch=1.
  options.profile_shapes = {
      {"kpts0", {1, 100, 2}, {engine_batch, 5000, 2},
       {engine_batch, 5000, 2}},
      {"desc0", {1, 100, 128}, {engine_batch, 5000, 128},
       {engine_batch, 5000, 128}},
      {"kpts1", {1, 100, 2}, {engine_batch, 5000, 2},
       {engine_batch, 5000, 2}},
      {"desc1", {1, 100, 128}, {engine_batch, 5000, 128},
       {engine_batch, 5000, 128}},
  };
  return EnsureRuntimeEngine(
      "lightglue.onnx",
      PhoenixBatchEngineName("lightglue", engine_batch),
      options);
}

std::filesystem::path EnsureRetrievalEngine(
    const std::filesystem::path& onnx_path) {
  auto engine_path = onnx_path;
  engine_path.replace_extension(".engine");

  if (std::filesystem::exists(engine_path)) {
    return engine_path;
  }

  if (!std::filesystem::exists(onnx_path)) {
    throw std::runtime_error("Retrieval ONNX model not found: " +
                             onnx_path.string());
  }

  spdlog::info("Building retrieval TRT engine from {}", onnx_path.string());

  TrtBuildOptions options;
  options.enable_fp16 = true;
  options.builder_optimization_level = 3;
  options.profile_shapes = {
      {"pixel_values",
       {1, 3, 224, 224},
       {16, 3, 224, 224},
       {64, 3, 224, 224}},
  };

  if (!BuildSerializedEngine(onnx_path, engine_path, options)) {
    throw std::runtime_error(
        "Failed to build retrieval TRT engine from: " + onnx_path.string());
  }

  spdlog::info("Built retrieval TRT engine: {}", engine_path.string());
  return engine_path;
}

}  // namespace sfm_phoenix