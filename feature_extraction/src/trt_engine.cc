#include "feature_extraction/trt_engine.h"

#include <NvInfer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace feature_extraction {

// --------------------------------------------------------------------------
// Simple TRT logger
// --------------------------------------------------------------------------
class TrtLoggerImpl : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

static TrtLoggerImpl g_trt_logger;

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
// TrtEngine
// --------------------------------------------------------------------------
bool TrtEngine::load(const std::string& engine_path) {
    // Read serialised engine
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Cannot open engine: " << engine_path << std::endl;
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
        std::cerr << "Failed to deserialise engine: " << engine_path
                  << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) return false;

    // Build binding index map
    int nb = engine_->getNbBindings();
    buffers_.resize(nb);
    shapes_.resize(nb);
    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getBindingName(i);
        binding_index_[name] = i;
        shapes_[i] = engine_->getBindingDimensions(i);
    }

    std::cout << "Loaded TRT engine: " << engine_path << " (" << nb
              << " bindings)" << std::endl;
    return true;
}

int TrtEngine::get_index(const std::string& name) const {
    auto it = binding_index_.find(name);
    if (it == binding_index_.end()) {
        throw std::runtime_error("Unknown binding: " + name);
    }
    return it->second;
}

size_t TrtEngine::binding_bytes(int idx) const {
    auto dims = context_->getBindingDimensions(idx);
    int64_t count = 1;
    for (int d = 0; d < dims.nbDims; ++d) {
        count *= dims.d[d];
    }
    // Assume float32
    return static_cast<size_t>(count) * sizeof(float);
}

void TrtEngine::ensure_buffer(int idx) {
    size_t bytes = binding_bytes(idx);
    if (bytes > 0) {
        buffers_[idx].resize(bytes);
    }
}

void TrtEngine::set_input_shape(const std::string& name,
                                const std::vector<int>& shape) {
    int idx = get_index(name);
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(shape.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = shape[i];
    }
    context_->setBindingDimensions(idx, dims);
    shapes_[idx] = dims;
    ensure_buffer(idx);
}

void TrtEngine::set_input(const std::string& name, const void* host_data,
                          size_t bytes) {
    int idx = get_index(name);
    ensure_buffer(idx);
    auto err = cudaMemcpy(buffers_[idx].ptr, host_data, bytes,
                          cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") +
                                 cudaGetErrorString(err));
    }
}

bool TrtEngine::infer(cudaStream_t stream) {
    // Collect device pointers
    std::vector<void*> ptrs(buffers_.size());
    for (size_t i = 0; i < buffers_.size(); ++i) {
        ensure_buffer(static_cast<int>(i));
        ptrs[i] = buffers_[i].ptr;
    }

    bool ok;
    if (stream) {
        ok = context_->enqueueV2(ptrs.data(), stream, nullptr);
    } else {
        // Create a temporary stream
        cudaStream_t tmp;
        cudaStreamCreate(&tmp);
        ok = context_->enqueueV2(ptrs.data(), tmp, nullptr);
        cudaStreamSynchronize(tmp);
        cudaStreamDestroy(tmp);
    }
    return ok;
}

void TrtEngine::get_output(const std::string& name, void* host_data,
                           size_t bytes) const {
    int idx = get_index(name);
    auto err = cudaMemcpy(host_data, buffers_[idx].ptr, bytes,
                          cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") +
                                 cudaGetErrorString(err));
    }
}

void* TrtEngine::device_ptr(const std::string& name) {
    int idx = get_index(name);
    return buffers_[idx].ptr;
}

const void* TrtEngine::device_ptr(const std::string& name) const {
    int idx = get_index(name);
    return buffers_[idx].ptr;
}

std::vector<int> TrtEngine::shape(const std::string& name) const {
    int idx = get_index(name);
    auto dims = context_->getBindingDimensions(idx);
    std::vector<int> s(dims.nbDims);
    for (int i = 0; i < dims.nbDims; ++i) s[i] = dims.d[i];
    return s;
}

int64_t TrtEngine::element_count(const std::string& name) const {
    int idx = get_index(name);
    auto dims = context_->getBindingDimensions(idx);
    int64_t count = 1;
    for (int d = 0; d < dims.nbDims; ++d) count *= dims.d[d];
    return count;
}

}  // namespace feature_extraction
