cmake_minimum_required(VERSION 3.18)
project(sfm_phoenix VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;86;89" CACHE STRING "CUDA architectures")

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(_vcpkg_root "")
set(_vcpkg_triplet "x64-windows")
if(DEFINED VCPKG_TARGET_TRIPLET AND NOT VCPKG_TARGET_TRIPLET STREQUAL "")
    set(_vcpkg_triplet "${VCPKG_TARGET_TRIPLET}")
endif()

set(_vcpkg_candidate_roots "")
if(DEFINED VCPKG_INSTALLED_DIR AND NOT VCPKG_INSTALLED_DIR STREQUAL "")
    get_filename_component(_vcpkg_installed_root "${VCPKG_INSTALLED_DIR}" DIRECTORY)
    list(APPEND _vcpkg_candidate_roots "${_vcpkg_installed_root}")
endif()

if(DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_TOOLCHAIN_FILE}")
    get_filename_component(_vcpkg_scripts_dir "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
    get_filename_component(_vcpkg_from_toolchain "${_vcpkg_scripts_dir}/../.." ABSOLUTE)
    list(APPEND _vcpkg_candidate_roots "${_vcpkg_from_toolchain}")
endif()

if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}")
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_env_root)
    list(APPEND _vcpkg_candidate_roots "${_vcpkg_env_root}")
endif()

list(APPEND _vcpkg_candidate_roots "D:/vcpkg")
list(REMOVE_DUPLICATES _vcpkg_candidate_roots)

foreach(_candidate_root IN LISTS _vcpkg_candidate_roots)
    set(_candidate_installed "${_candidate_root}/installed/${_vcpkg_triplet}")
    if(EXISTS "${_candidate_installed}/share/colmap" AND EXISTS "${_candidate_installed}/share/cudss")
        set(_vcpkg_root "${_candidate_root}")
        set(_vcpkg_installed "${_candidate_installed}")
        break()
    endif()
endforeach()

if(_vcpkg_installed)
    message(STATUS "Using vcpkg installed tree: ${_vcpkg_installed}")
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
find_package(spdlog CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(TBB CONFIG REQUIRED)

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
# Library: sfm_phoenix
# --------------------------------------------------------------------------
set(FE_SOURCES
    src/internal/trt_engine.cc
    src/internal/model_runtime.cc
    src/extractors/aliked.cc
    src/bridges/colmap_bridge.cc
    src/matchers/lightglue.cc
    src/pipelines/feature_pipeline.cc
)
set(FE_CUDA_SOURCES
    src/internal/dkd_cuda.cu
    src/internal/preprocess_cuda.cu
)

add_library(sfm_phoenix STATIC ${FE_SOURCES} ${FE_CUDA_SOURCES})

target_include_directories(sfm_phoenix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../migration/include
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${COLMAP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sfm_phoenix PUBLIC
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
    target_compile_options(sfm_phoenix PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /utf-8 /W3>)
    target_compile_options(sfm_phoenix PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd9002>)
else()
    target_compile_options(sfm_phoenix PRIVATE -Wall -Wextra)
endif()

# --------------------------------------------------------------------------
# Demo executable
# --------------------------------------------------------------------------
add_library(phoenix_logging STATIC ../migration/src/logging.cc)
target_include_directories(phoenix_logging PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../migration/include
)
target_compile_definitions(phoenix_logging PRIVATE
    $<$<CONFIG:RelWithDebInfo>:MIGRATION_PLAINTEXT_LOGGING=1>
)
target_link_libraries(phoenix_logging PUBLIC
    spdlog::spdlog_header_only
    OpenSSL::Crypto
)

if(MSVC)
    target_compile_options(phoenix_logging PRIVATE /utf-8 /W3)
endif()

add_executable(demo_feature_matching examples/demo_pair.cc)
target_link_libraries(demo_feature_matching PRIVATE sfm_phoenix phoenix_logging)

add_executable(phoenix
    tools/phoenix_main.cc
    tools/phoenix/phoenix_common.cc
    tools/phoenix/phoenix_extractor.cc
    tools/phoenix/phoenix_matcher.cc
    tools/phoenix/phoenix_automatic.cc)
target_link_libraries(phoenix PRIVATE sfm_phoenix phoenix_logging TBB::tbb)

set(PHOENIX_RUNTIME_OUTPUT_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")

set(PHOENIX_RUNTIME_ASSETS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/aliked_backbone.onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/models/aliked_sddh.onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/models/dinov3_vitb16_pretrain_lvd1689m.onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/models/lightglue.onnx
)

function(copy_phoenix_runtime_assets target_name)
    set(_copy_commands)

    foreach(asset IN LISTS PHOENIX_RUNTIME_ASSETS)
        if(EXISTS "${asset}")
            get_filename_component(_asset_name "${asset}" NAME)
            list(APPEND _copy_commands
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${asset}"
                        "${PHOENIX_RUNTIME_OUTPUT_DIR}")
        else()
            message(WARNING "[phoenix] asset not found, skipping: ${asset}")
        endif()
    endforeach()

    if(_vcpkg_installed AND EXISTS "${_vcpkg_installed}/tools/crashpad_handler.exe")
        list(APPEND _copy_commands
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_vcpkg_installed}/tools/crashpad_handler.exe"
                    "${PHOENIX_RUNTIME_OUTPUT_DIR}/crashpad_handler.exe")
    endif()

    if(_vcpkg_installed AND EXISTS "${_vcpkg_installed}/bin")
        file(GLOB _vcpkg_runtime_dlls "${_vcpkg_installed}/bin/*.dll")
        foreach(_dll IN LISTS _vcpkg_runtime_dlls)
            list(APPEND _copy_commands
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_dll}"
                        "${PHOENIX_RUNTIME_OUTPUT_DIR}")
        endforeach()
    endif()

    if(_vcpkg_installed AND EXISTS "${_vcpkg_installed}/tools/colmap")
        list(APPEND _copy_commands
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                    "${_vcpkg_installed}/tools/colmap/"
                    "${PHOENIX_RUNTIME_OUTPUT_DIR}/")
    endif()

    if(WIN32 AND TensorRT_ROOT AND EXISTS "${TensorRT_ROOT}")
        set(_trt_bin_dir "${TensorRT_ROOT}/bin")
        file(GLOB _trt_dlls "${_trt_bin_dir}/*.dll")
        if(_trt_dlls)
            foreach(_dll IN LISTS _trt_dlls)
                get_filename_component(_dll_name "${_dll}" NAME)
                list(APPEND _copy_commands
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${_dll}"
                            "${PHOENIX_RUNTIME_OUTPUT_DIR}")
            endforeach()
        else()
            message(WARNING "[phoenix] No DLLs found in TRT bin dir: ${_trt_bin_dir}")
        endif()
    else()
        message(STATUS "[phoenix] TRT DLL copy skipped: WIN32=${WIN32} TensorRT_ROOT=${TensorRT_ROOT}")
    endif()

    if(_copy_commands)
        add_custom_command(TARGET ${target_name} POST_BUILD
            ${_copy_commands}
            COMMENT "[phoenix] copy runtime assets")
    endif()
endfunction()

copy_phoenix_runtime_assets(demo_feature_matching)
copy_phoenix_runtime_assets(phoenix)

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
install(TARGETS
    sfm_phoenix
    phoenix_logging
    demo_feature_matching
    phoenix
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${PHOENIX_RUNTIME_ASSETS} DESTINATION bin)
install(DIRECTORY include/sfm_phoenix DESTINATION include)
