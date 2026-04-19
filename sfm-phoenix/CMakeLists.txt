cmake_minimum_required(VERSION 3.18)
project(feature_extraction VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;86;89" CACHE STRING "CUDA architectures")

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(_vcpkg_root "")
if(DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_TOOLCHAIN_FILE}")
    get_filename_component(_vcpkg_scripts_dir "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
    get_filename_component(_vcpkg_root "${_vcpkg_scripts_dir}/../.." ABSOLUTE)
elseif(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}")
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_root)
elseif(EXISTS "D:/vcpkg")
    set(_vcpkg_root "D:/vcpkg")
endif()

if(_vcpkg_root)
    set(_vcpkg_triplet "x64-windows")
    if(DEFINED VCPKG_TARGET_TRIPLET AND NOT VCPKG_TARGET_TRIPLET STREQUAL "")
        set(_vcpkg_triplet "${VCPKG_TARGET_TRIPLET}")
    endif()

    set(_vcpkg_installed "${_vcpkg_root}/installed/${_vcpkg_triplet}")
    if(EXISTS "${_vcpkg_installed}")
        list(PREPEND CMAKE_PREFIX_PATH
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
            "${_vcpkg_installed}"
            "${_vcpkg_installed}/share/opencv4"
            "${_vcpkg_installed}/share/colmap")
        list(PREPEND CMAKE_MODULE_PATH
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
            "${_vcpkg_installed}/share/cudss"
            "${_vcpkg_installed}/share/jxr"
            "${_vcpkg_installed}/share/colmap/cmake")
    endif()
endif()

# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(TARGET OpenMP::OpenMP_CXX AND NOT TARGET OpenMP::OpenMP_C)
    add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)
    target_link_libraries(OpenMP::OpenMP_C INTERFACE OpenMP::OpenMP_CXX)
endif()

set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
find_package(gflags CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(COLMAP CONFIG REQUIRED)

# TensorRT — search common install locations
# Set TensorRT_ROOT via cmake -DTensorRT_ROOT=... or environment variable
if(NOT TensorRT_ROOT)
    if(DEFINED ENV{TensorRT_ROOT})
        set(TensorRT_ROOT $ENV{TensorRT_ROOT})
    elseif(WIN32)
        # Common Windows install paths
        file(GLOB _trt_candidates
            "C:/TensorRT*"
            "C:/Program Files/TensorRT*"
            "C:/Program Files/NVIDIA/TensorRT*")
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

# TensorRT 10.x uses versioned names: nvinfer_10, nvinfer_plugin_10, nvonnxparser_10
find_library(NVINFER_LIB NAMES nvinfer nvinfer_10 HINTS ${TRT_LIB_DIR})
find_library(NVINFER_PLUGIN_LIB NAMES nvinfer_plugin nvinfer_plugin_10 HINTS ${TRT_LIB_DIR})
find_library(NVONNXPARSER_LIB NAMES nvonnxparser nvonnxparser_10 HINTS ${TRT_LIB_DIR})

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "Could not find nvinfer library. Set TensorRT_ROOT.")
endif()

# --------------------------------------------------------------------------
# Library: feature_extraction
# --------------------------------------------------------------------------
set(FE_SOURCES
    src/trt_engine.cc
    src/aliked_trt.cc
    src/colmap_bridge.cc
    src/lightglue_trt.cc
    src/feature_pipeline.cc
)
set(FE_CUDA_SOURCES
    src/dkd_cuda.cu
    src/preprocess_cuda.cu
)

add_library(feature_extraction STATIC ${FE_SOURCES} ${FE_CUDA_SOURCES})

target_include_directories(feature_extraction PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${COLMAP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(feature_extraction PUBLIC
    colmap::colmap
    OpenMP::OpenMP_CXX
    gflags::gflags
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${NVONNXPARSER_LIB}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

if(MSVC)
    target_compile_options(feature_extraction PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /utf-8 /W3>)
    target_compile_options(feature_extraction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
else()
    target_compile_options(feature_extraction PRIVATE -Wall -Wextra)
endif()

# --------------------------------------------------------------------------
# Demo executable
# --------------------------------------------------------------------------
add_executable(demo_feature_matching examples/demo_pair.cc)
target_link_libraries(demo_feature_matching PRIVATE feature_extraction)

add_executable(phoenix_feature_extractor tools/phoenix_feature_extractor.cc)
target_link_libraries(phoenix_feature_extractor PRIVATE feature_extraction)

add_executable(phoenix_matcher tools/phoenix_matcher.cc)
target_link_libraries(phoenix_matcher PRIVATE feature_extraction)

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
install(TARGETS
    feature_extraction
    demo_feature_matching
    phoenix_feature_extractor
    phoenix_matcher
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/feature_extraction DESTINATION include)
