#include "sfm_phoenix/internal/trt_engine.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace sfm_phoenix {

// --------------------------------------------------------------------------
// Simple TRT logger
// --------------------------------------------------------------------------
class TrtLoggerImpl : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            spdlog::warn("[TRT] {}", msg);
        }
    }
};

static TrtLoggerImpl g_trt_logger;

namespace {

std::string DimsToString(const nvinfer1::Dims& dims) {
    std::ostringstream stream;
    stream << '[';
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << dims.d[i];
    }
    stream << ']';
    return stream.str();
}

nvinfer1::Dims MakeDims(const std::vector<int>& shape) {
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(shape.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = shape[i];
    }
    return dims;
}

}  // namespace

// Ensure TRT plugins are registered (one-time init)
static bool g_plugins_initialized = []() {
    initLibNvInferPlugins(&g_trt_logger, "");
    return true;
}();

bool BuildSerializedEngine(const std::filesystem::path& onnx_path,
                          const std::filesystem::path& engine_path,
                          const TrtBuildOptions& options) {
    if (!std::filesystem::exists(onnx_path)) {
        spdlog::error("Cannot find ONNX model: {}", onnx_path.string());
        return false;
    }

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(g_trt_logger));
    if (!builder) {
        spdlog::error("Failed to create TensorRT builder");
        return false;
    }

    constexpr uint32_t kExplicitBatchFlag =
        1U << static_cast<uint32_t>(
                  nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(kExplicitBatchFlag));
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, g_trt_logger));
    if (!network || !config || !parser) {
        spdlog::error("Failed to create TensorRT network/config/parser");
        return false;
    }

    if (!parser->parseFromFile(
            onnx_path.string().c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int index = 0; index < parser->getNbErrors(); ++index) {
            spdlog::error("[TRT] {}", parser->getError(index)->desc());
        }
        spdlog::error("Failed to parse ONNX model: {}", onnx_path.string());
        return false;
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               options.workspace_bytes);
    config->setBuilderOptimizationLevel(options.builder_optimization_level);
    if (options.enable_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    if (!options.profile_shapes.empty()) {
        auto* profile = builder->createOptimizationProfile();
        if (profile == nullptr) {
            spdlog::error("Failed to create TensorRT optimization profile");
            return false;
        }
        for (const auto& profile_shape : options.profile_shapes) {
            const auto min_dims = MakeDims(profile_shape.min_shape);
            const auto opt_dims = MakeDims(profile_shape.opt_shape);
            const auto max_dims = MakeDims(profile_shape.max_shape);
            const bool ok =
                profile->setDimensions(profile_shape.name.c_str(),
                                       nvinfer1::OptProfileSelector::kMIN,
                                       min_dims) &&
                profile->setDimensions(profile_shape.name.c_str(),
                                       nvinfer1::OptProfileSelector::kOPT,
                                       opt_dims) &&
                profile->setDimensions(profile_shape.name.c_str(),
                                       nvinfer1::OptProfileSelector::kMAX,
                                       max_dims);
            if (!ok) {
                spdlog::error("Failed to set TensorRT profile for {}",
                              profile_shape.name);
                return false;
            }
        }
        config->addOptimizationProfile(profile);
    }

    spdlog::info("Building missing TensorRT engine: {} -> {}",
                 onnx_path.string(),
                 engine_path.string());
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        spdlog::error("TensorRT engine build failed for {}", onnx_path.string());
        return false;
    }

    std::filesystem::create_directories(engine_path.parent_path());
    std::ofstream stream(engine_path, std::ios::binary);
    if (!stream.is_open()) {
        spdlog::error("Cannot write engine file: {}", engine_path.string());
        return false;
    }
    stream.write(static_cast<const char*>(serialized->data()),
                 static_cast<std::streamsize>(serialized->size()));
    stream.close();
    return true;
}

// --------------------------------------------------------------------------
// CudaBuffer
// --------------------------------------------------------------------------
CudaBuffer::~CudaBuffer() {
    if (ptr) cudaFree(ptr);
}

CudaBuffer::CudaBuffer(CudaBuffer&& o) noexcept : ptr(o.ptr), size(o.size) {
    o.ptr = nullptr;
    o.size = 0;
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& o) noexcept {
    if (this != &o) {
        if (ptr) cudaFree(ptr);
        ptr = o.ptr;
        size = o.size;
        o.ptr = nullptr;
        o.size = 0;
    }
    return *this;
}

void CudaBuffer::resize(size_t bytes) {
    if (bytes <= size) return;
    if (ptr) cudaFree(ptr);
    auto err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                 cudaGetErrorString(err));
    }
    size = bytes;
}

// --------------------------------------------------------------------------
// TrtEngine (TensorRT 10.x: tensor-name API, enqueueV3)
// --------------------------------------------------------------------------
bool TrtEngine::load(const std::string& engine_path) {
    // Read serialised engine
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        spdlog::error("Cannot open engine: {}", engine_path);
        return false;
    }
    auto fsize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(fsize);
    file.read(data.data(), fsize);
    file.close();

    logger_ = &g_trt_logger;
    runtime_.reset(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) return false;

    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) {
        spdlog::error("Failed to deserialise engine: {}", engine_path);
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) return false;

    int nb = engine_->getNbIOTensors();
    spdlog::info("Loaded TRT engine: {} ({} IO tensors)", engine_path, nb);

    // Log precision for each IO tensor
    auto dtype_str = [](nvinfer1::DataType dt) -> const char* {
        switch (dt) {
            case nvinfer1::DataType::kFLOAT: return "FP32";
            case nvinfer1::DataType::kHALF:  return "FP16";
            case nvinfer1::DataType::kINT8:  return "INT8";
            case nvinfer1::DataType::kINT32: return "INT32";
            case nvinfer1::DataType::kINT64: return "INT64";
            case nvinfer1::DataType::kBOOL:  return "BOOL";
            default:                         return "OTHER";
        }
    };
    for (int i = 0; i < nb; ++i) {
        const char* tname = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(tname);
        auto dtype = engine_->getTensorDataType(tname);
        const char* dir = (mode == nvinfer1::TensorIOMode::kINPUT)
                          ? "in" : "out";
        spdlog::info("  [{}] {} dtype={}", dir, tname, dtype_str(dtype));
    }
    return true;
}

size_t TrtEngine::tensor_bytes(const std::string& name) const {
    auto dims = context_->getTensorShape(name.c_str());
    int64_t count = 1;
    for (int d = 0; d < dims.nbDims; ++d) {
        count *= dims.d[d];
    }
    // Determine element size from data type
    auto dtype = engine_->getTensorDataType(name.c_str());
    size_t elem_size = sizeof(float);  // default float32
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: elem_size = 4; break;
        case nvinfer1::DataType::kHALF:  elem_size = 2; break;
        case nvinfer1::DataType::kINT32: elem_size = 4; break;
        case nvinfer1::DataType::kINT64: elem_size = 8; break;
        case nvinfer1::DataType::kINT8:  elem_size = 1; break;
        case nvinfer1::DataType::kBOOL:  elem_size = 1; break;
        default: elem_size = 4; break;
    }
    return static_cast<size_t>(count) * elem_size;
}

void TrtEngine::ensure_buffer(const std::string& name, size_t bytes) {
    if (bytes > 0) {
        buffers_[name].resize(bytes);
    }
}

void TrtEngine::set_input_shape(const std::string& name,
                                const std::vector<int>& shape) {
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(shape.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = shape[i];
    }
    if (!context_->setInputShape(name.c_str(), dims)) {
        const auto max_dims =
            engine_->getProfileShape(name.c_str(),
                                     0,
                                     nvinfer1::OptProfileSelector::kMAX);
        throw std::runtime_error("TensorRT input shape for '" + name +
                                 "'=" + DimsToString(dims) +
                                 " exceeds profile max " +
                                 DimsToString(max_dims));
    }
    ensure_buffer(name, tensor_bytes(name));
}

void TrtEngine::set_input(const std::string& name, const void* host_data,
                          size_t bytes, cudaStream_t stream) {
    ensure_buffer(name, bytes);
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(buffers_[name].ptr, host_data, bytes,
                              cudaMemcpyHostToDevice, stream);
    } else {
        err = cudaMemcpy(buffers_[name].ptr, host_data, bytes,
                         cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") +
                                 cudaGetErrorString(err));
    }
}

void TrtEngine::set_device_input(const std::string& name, void* device_ptr) {
    external_ptrs_[name] = device_ptr;
}

void TrtEngine::clear_device_inputs() {
    external_ptrs_.clear();
}

bool TrtEngine::infer(cudaStream_t stream) {
    // Set tensor addresses for all IO tensors
    int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* tname = engine_->getIOTensorName(i);
        std::string sname(tname);
        auto ext_it = external_ptrs_.find(sname);
        if (ext_it != external_ptrs_.end()) {
            // Use externally-managed device pointer
            context_->setTensorAddress(tname, ext_it->second);
        } else {
            ensure_buffer(sname, tensor_bytes(sname));
            context_->setTensorAddress(tname, buffers_[sname].ptr);
        }
    }

    bool ok;
    if (stream) {
        ok = context_->enqueueV3(stream);
    } else {
        cudaStream_t tmp;
        cudaStreamCreate(&tmp);
        ok = context_->enqueueV3(tmp);
        cudaStreamSynchronize(tmp);
        cudaStreamDestroy(tmp);
    }
    return ok;
}

void TrtEngine::get_output(const std::string& name, void* host_data,
                           size_t bytes, cudaStream_t stream) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        throw std::runtime_error("Unknown tensor: " + name);
    }
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(host_data, it->second.ptr, bytes,
                              cudaMemcpyDeviceToHost, stream);
    } else {
        err = cudaMemcpy(host_data, it->second.ptr, bytes,
                         cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") +
                                 cudaGetErrorString(err));
    }
}

void* TrtEngine::device_ptr(const std::string& name) {
    return buffers_[name].ptr;
}

const void* TrtEngine::device_ptr(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        throw std::runtime_error("Unknown tensor: " + name);
    }
    return it->second.ptr;
}

std::vector<int> TrtEngine::shape(const std::string& name) const {
    auto dims = context_->getTensorShape(name.c_str());
    std::vector<int> s(dims.nbDims);
    for (int i = 0; i < dims.nbDims; ++i) s[i] = dims.d[i];
    return s;
}

std::vector<int> TrtEngine::max_profile_shape(const std::string& name) const {
    auto dims = engine_->getProfileShape(name.c_str(),
                                         0,
                                         nvinfer1::OptProfileSelector::kMAX);
    std::vector<int> s(dims.nbDims);
    for (int i = 0; i < dims.nbDims; ++i) {
        s[i] = dims.d[i];
    }
    return s;
}

int64_t TrtEngine::element_count(const std::string& name) const {
    auto dims = context_->getTensorShape(name.c_str());
    int64_t count = 1;
    for (int d = 0; d < dims.nbDims; ++d) count *= dims.d[d];
    return count;
}

}  // namespace sfm_phoenix
