#include "sfm_phoenix/internal/model_runtime.h"

#include "sfm_phoenix/internal/trt_engine.h"

#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdexcept>

namespace sfm_phoenix {

namespace {

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

std::filesystem::path EnsurePhoenixBackboneEngine() {
  TrtBuildOptions options;
  options.enable_fp16 = false;
  options.builder_optimization_level = 3;
  options.profile_shapes = {
      {"image", {1, 3, 320, 320}, {1, 3, 1600, 1600}, {1, 3, 1600, 1600}},
  };
  return EnsureRuntimeEngine(
      "aliked_backbone.onnx", "aliked_backbone.engine", options);
}

std::filesystem::path EnsurePhoenixSddhEngine() {
  TrtBuildOptions options;
  options.enable_fp16 = true;
  options.builder_optimization_level = 3;
  options.profile_shapes = {
      {"feature_map", {1, 128, 320, 320}, {1, 128, 1600, 1600}, {1, 128, 1600, 1600}},
      {"keypoints_wh", {100, 2}, {5000, 2}, {5000, 2}},
      {"feature_map_hw", {2}, {2}, {2}},
  };
  return EnsureRuntimeEngine("aliked_sddh.onnx", "aliked_sddh.engine", options);
}

std::filesystem::path EnsurePhoenixLightGlueEngine() {
  TrtBuildOptions options;
  options.enable_fp16 = true;
  options.builder_optimization_level = 3;
  options.profile_shapes = {
      {"kpts0", {1, 100, 2}, {1, 5000, 2}, {1, 5000, 2}},
      {"desc0", {1, 100, 128}, {1, 5000, 128}, {1, 5000, 128}},
      {"kpts1", {1, 100, 2}, {1, 5000, 2}, {1, 5000, 2}},
      {"desc1", {1, 100, 128}, {1, 5000, 128}, {1, 5000, 128}},
  };
  return EnsureRuntimeEngine("lightglue.onnx", "lightglue.engine", options);
}

}  // namespace sfm_phoenix