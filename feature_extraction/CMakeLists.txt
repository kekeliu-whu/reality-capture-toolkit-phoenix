cmake_minimum_required(VERSION 3.18)
project(feature_extraction VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

# TensorRT — search common install locations
# Set TensorRT_ROOT via cmake -DTensorRT_ROOT=... or environment variable
if(NOT TensorRT_ROOT)
    if(DEFINED ENV{TensorRT_ROOT})
        set(TensorRT_ROOT $ENV{TensorRT_ROOT})
    elseif(WIN32)
        # Common Windows install paths
        file(GLOB _trt_candidates "C:/TensorRT*" "C:/Program Files/NVIDIA/TensorRT*")
        list(LENGTH _trt_candidates _n)
        if(_n GREATER 0)
            list(GET _trt_candidates 0 TensorRT_ROOT)
        endif()
    else()
        set(TensorRT_ROOT "/usr/local/TensorRT")
    endif()
endif()

if(TensorRT_ROOT)
    message(STATUS "TensorRT root: ${TensorRT_ROOT}")
    set(TRT_INCLUDE_DIR "${TensorRT_ROOT}/include")
    if(WIN32)
        set(TRT_LIB_DIR "${TensorRT_ROOT}/lib")
    else()
        set(TRT_LIB_DIR "${TensorRT_ROOT}/lib")
    endif()
else()
    message(WARNING "TensorRT_ROOT not set. Set -DTensorRT_ROOT=<path> or TensorRT_ROOT env var.")
endif()

find_library(NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR})
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TRT_LIB_DIR})
find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TRT_LIB_DIR})

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "Could not find nvinfer library. Set TensorRT_ROOT.")
endif()

# --------------------------------------------------------------------------
# Library: feature_extraction
# --------------------------------------------------------------------------
set(FE_SOURCES
    src/trt_engine.cc
    src/aliked_trt.cc
    src/lightglue_trt.cc
    src/feature_pipeline.cc
)
set(FE_CUDA_SOURCES
    src/dkd_cuda.cu
)

add_library(feature_extraction STATIC ${FE_SOURCES} ${FE_CUDA_SOURCES})

target_include_directories(feature_extraction PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(feature_extraction PUBLIC
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${NVONNXPARSER_LIB}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

if(MSVC)
    target_compile_options(feature_extraction PRIVATE /MP /utf-8 /W3)
    target_compile_options(feature_extraction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
else()
    target_compile_options(feature_extraction PRIVATE -Wall -Wextra)
endif()

# --------------------------------------------------------------------------
# Demo executable
# --------------------------------------------------------------------------
add_executable(demo_feature_matching examples/demo_pair.cc)
target_link_libraries(demo_feature_matching PRIVATE feature_extraction)

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
install(TARGETS feature_extraction demo_feature_matching
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/feature_extraction DESTINATION include)
