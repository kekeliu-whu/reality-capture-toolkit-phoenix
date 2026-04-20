#pragma once

#include <filesystem>

namespace sfm_phoenix {

std::filesystem::path GetExecutableDirectory();

std::filesystem::path EnsurePhoenixBackboneEngine();

std::filesystem::path EnsurePhoenixSddhEngine();

std::filesystem::path EnsurePhoenixLightGlueEngine();

}  // namespace sfm_phoenix