#include "sfm_phoenix/internal/trt_engine.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <cassert>
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

}  // namespace

// Ensure TRT plugins are registered (one-time init)
static bool g_plugins_initialized = []() {
    initLibNvInferPlugins(&g_trt_logger, "");
    return true;
}();

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
