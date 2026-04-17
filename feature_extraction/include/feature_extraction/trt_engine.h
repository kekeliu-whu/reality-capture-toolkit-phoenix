#pragma once

#include <NvInfer.h>

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace feature_extraction {

/// RAII wrapper for a CUDA device buffer.
struct CudaBuffer {
    void* ptr = nullptr;
    size_t size = 0;  // in bytes

    CudaBuffer() = default;
    ~CudaBuffer();
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& o) noexcept;
    CudaBuffer& operator=(CudaBuffer&& o) noexcept;

    /// Ensure buffer has at least `bytes` allocated.
    void resize(size_t bytes);
};

/// TensorRT engine wrapper (TensorRT 10.x API).
///
/// Manages loading a serialised engine, creating an execution context, and
/// running inference with GPU buffers.
class TrtEngine {
public:
    TrtEngine() = default;
    ~TrtEngine() = default;

    /// Load a serialised TensorRT engine from file.
    bool load(const std::string& engine_path);

    /// Set the shape of a dynamic input tensor.
    /// Must be called before infer() for every dynamic-shape input.
    void set_input_shape(const std::string& name,
                         const std::vector<int>& shape);

    /// Copy host data to a named input tensor's GPU buffer.
    void set_input(const std::string& name, const void* host_data,
                   size_t bytes);

    /// Run inference synchronously on the given CUDA stream.
    bool infer(cudaStream_t stream = nullptr);

    /// Copy a named output tensor's GPU buffer to host memory.
    void get_output(const std::string& name, void* host_data,
                    size_t bytes) const;

    /// Get the raw GPU pointer for a named tensor (input or output).
    void* device_ptr(const std::string& name);
    const void* device_ptr(const std::string& name) const;

    /// Get the current shape of a binding (after set_input_shape / infer).
    std::vector<int> shape(const std::string& name) const;

    /// Get the element count of a binding.
    int64_t element_count(const std::string& name) const;

private:
    nvinfer1::ILogger* logger_ = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Tensor name → GPU buffer.
    std::unordered_map<std::string, CudaBuffer> buffers_;

    void ensure_buffer(const std::string& name, size_t bytes);
    size_t tensor_bytes(const std::string& name) const;
};

}  // namespace feature_extraction
