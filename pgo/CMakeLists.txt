cmake_minimum_required(VERSION 3.10)
project(pgo)

include(CTest)

if(CMAKE_TOOLCHAIN_FILE)
    include(${CMAKE_TOOLCHAIN_FILE})
endif(CMAKE_TOOLCHAIN_FILE)

add_definitions(-D_USE_MATH_DEFINES -DPROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
  # COLMAP pulls in OpenImageIO's header-only fmt definition, while
  # migration/spdlog link the compiled fmt library. Keep PGO translation
  # units on the compiled fmt mode to avoid duplicate fmt symbols.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/UFMT_HEADER_ONLY>)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Ob3 /MD /Zi")
  add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/Zi>)
  add_link_options($<$<CONFIG:Release>:/DEBUG>)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VTK CONFIG QUIET)
find_package(PCL REQUIRED COMPONENTS common io filters registration visualization)
find_package(Boost CONFIG COMPONENTS program_options)
find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)
find_package(PDAL CONFIG REQUIRED)
find_package(Glog REQUIRED)
find_package(PROJ CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(colmap CONFIG REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be provided by the build script")
endif()

# xsfm_post_depth.cu is compiled by a custom NVCC command, so translate the
# standard CMake architecture list into matching NVCC -gencode arguments.
set(PGO_CUDA_GENCODE_FLAGS)
foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(CUDA_ARCH MATCHES "^([0-9]+)-virtual$")
    list(APPEND PGO_CUDA_GENCODE_FLAGS
      "-gencode=arch=compute_${CMAKE_MATCH_1},code=compute_${CMAKE_MATCH_1}")
  elseif(CUDA_ARCH MATCHES "^([0-9]+)-real$")
    list(APPEND PGO_CUDA_GENCODE_FLAGS
      "-gencode=arch=compute_${CMAKE_MATCH_1},code=sm_${CMAKE_MATCH_1}")
  elseif(CUDA_ARCH MATCHES "^[0-9]+$")
    list(APPEND PGO_CUDA_GENCODE_FLAGS
      "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
      "-gencode=arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}")
  else()
    message(FATAL_ERROR "Unsupported CUDA architecture: ${CUDA_ARCH}")
  endif()
endforeach()
message(STATUS "PGO CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

include_directories(
  src
  src/thirdparty/BTC
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(../migration migration_build)

if(TARGET colmap::colmap)
  set(PGO_COLMAP_TARGET colmap::colmap)
elseif(TARGET colmap)
  set(PGO_COLMAP_TARGET colmap)
else()
  message(FATAL_ERROR "The required colmap package did not export colmap::colmap or colmap target.")
endif()


set(
  PGO_PRIVATE_SRCS
  src/map/optimizer.cc
  src/map/pgo_runner.cc
  src/map/utils.cc
  src/io/read_write.cc
  src/io/local_enu_transformer.cc
  src/thirdparty/BTC/BTC.cpp

  src/common/msg_conversions.h
  src/common/types.h
  src/factor/gps_factor.h
  src/factor/gravity_factor.h
  src/factor/local_parameterization_se3.h
  src/factor/pose_graph_edge_factor.h
  src/factor/rtk_factor.h
  src/io/read_write.h
  src/io/local_enu_transformer.h
  src/map/optimizer.h
  src/map/pgo_runner.h
  src/map/utils.h
  src/thirdparty/BTC/BTC.h
)

set(PGO_PRIVATE_LIBS
  migration_core
  glog::glog
  ${PCL_LIBRARIES}
  pcl_visualization
  ${GFLAGS_LIBRARIES}
  ${CERES_LIBRARIES}
  OpenMP::OpenMP_CXX
  $<TARGET_NAME:pdalcpp>
  PROJ::proj
)
if(VTK_FOUND)
  list(APPEND PGO_PRIVATE_LIBS ${VTK_LIBRARIES})
endif()

set(PGO_XSFM_TOOL_LIBS
  migration_core
  ${GFLAGS_LIBRARIES}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  $<TARGET_NAME:pdalcpp>
  PROJ::proj
  spdlog::spdlog_header_only
  OpenMP::OpenMP_CXX
  ${PGO_COLMAP_TARGET}
)

add_executable(
  xsfm_process_point_cloud
  tools/xsfm_process_point_cloud.cc
)
target_link_libraries(xsfm_process_point_cloud ${PGO_XSFM_TOOL_LIBS})
if(NOT DEFINED VCPKG_INSTALLED_DIR OR "${VCPKG_INSTALLED_DIR}" STREQUAL "")
  set(VCPKG_INSTALLED_DIR "F:/Library/vcpkg/installed")
endif()
set(CRASHPAD_HANDLER_SOURCE "${VCPKG_INSTALLED_DIR}/x64-windows/tools/crashpad_handler.exe")
if(NOT EXISTS "${CRASHPAD_HANDLER_SOURCE}")
  set(CRASHPAD_HANDLER_SOURCE "${VCPKG_INSTALLED_DIR}/x64-windows/tools/crashpad/crashpad_handler.exe")
endif()
add_custom_command(TARGET xsfm_process_point_cloud POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:xsfm_process_point_cloud>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${VCPKG_INSTALLED_DIR}/x64-windows/share/proj/proj.db"
    "$<TARGET_FILE_DIR:xsfm_process_point_cloud>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CRASHPAD_HANDLER_SOURCE}"
    "$<TARGET_FILE_DIR:xsfm_process_point_cloud>"
)

set(PGO_XCOLOR_TOOL_LIBS
  ${PGO_XSFM_TOOL_LIBS}
  ${OpenCV_LIBS}
)

add_library(
  pgo_xcolor_lib
  STATIC
  tools/xcolor_lib.cc
  tools/xcolor_lib.h
)
target_link_libraries(pgo_xcolor_lib ${PGO_XCOLOR_TOOL_LIBS})

add_executable(
  xcolor
  tools/xcolor_main.cc
)
target_link_libraries(xcolor pgo_xcolor_lib ${PGO_XCOLOR_TOOL_LIBS})

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  add_executable(
    xcolor_lib_test
    tools/xcolor_lib_test.cc
  )
  target_link_libraries(xcolor_lib_test pgo_xcolor_lib GTest::gtest_main)
  add_test(NAME xcolor_lib_test COMMAND xcolor_lib_test)
endif()

set(XSFM_POST_DEPTH_OBJ "${CMAKE_CURRENT_BINARY_DIR}/xsfm_post_depth_$<CONFIG>.obj")
add_custom_command(
  OUTPUT "${XSFM_POST_DEPTH_OBJ}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
    -std=c++17
    ${PGO_CUDA_GENCODE_FLAGS}
    -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/xsfm_post_depth.cu"
    -I "${CMAKE_CURRENT_SOURCE_DIR}/tools"
    -I "${CUDAToolkit_INCLUDE_DIRS}"
    -Xcompiler=/EHsc
    -Xcompiler=/MD
    -o "${XSFM_POST_DEPTH_OBJ}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/xsfm_post_depth.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/xsfm_post_depth.h"
  VERBATIM
)
set_source_files_properties(
  "${XSFM_POST_DEPTH_OBJ}"
  PROPERTIES
    GENERATED TRUE
    EXTERNAL_OBJECT TRUE
)
# xcolor reuses the same CUDA renderer to build depth directly in original
# OPENCV_FISHEYE views. Keeping one object avoids a second CUDA implementation.
target_sources(pgo_xcolor_lib PRIVATE "${XSFM_POST_DEPTH_OBJ}")
target_link_libraries(pgo_xcolor_lib CUDA::cudart)
add_executable(
  xsfm_post
  tools/xsfm_post.cc
  tools/xsfm_post_depth.h
  "${XSFM_POST_DEPTH_OBJ}"
)
target_link_libraries(
  xsfm_post
  ${PGO_XSFM_TOOL_LIBS}
  ${OpenCV_LIBS}
  CUDA::cudart
)
add_library(
  pgo_private
  STATIC
  ${PGO_PRIVATE_SRCS}
)
target_link_libraries(pgo_private ${PGO_PRIVATE_LIBS})

function(add_private_tool name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} pgo_private)
endfunction()

add_private_tool(slam_post src/pgo_main.cc)
