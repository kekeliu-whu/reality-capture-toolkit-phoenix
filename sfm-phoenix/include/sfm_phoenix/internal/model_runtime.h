#pragma once

#include <filesystem>

namespace sfm_phoenix {

std::filesystem::path GetExecutableDirectory();

std::filesystem::path EnsurePhoenixBackboneEngine(int batch_size);

std::filesystem::path EnsurePhoenixSddhEngine();

std::filesystem::path EnsurePhoenixLightGlueEngine(int batch_size);

/// Build (or load cached) a TensorRT engine from an arbitrary ONNX path.
/// Engine file is placed alongside the ONNX with a .engine extension.
std::filesystem::path EnsureRetrievalEngine(
    const std::filesystem::path& onnx_path);

}  // namespace sfm_phoenix