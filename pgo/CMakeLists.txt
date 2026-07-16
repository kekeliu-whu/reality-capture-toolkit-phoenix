cmake_minimum_required(VERSION 3.10)
project(pgo)

if(CMAKE_TOOLCHAIN_FILE)
    include(${CMAKE_TOOLCHAIN_FILE})
endif(CMAKE_TOOLCHAIN_FILE)

add_definitions(-D_USE_MATH_DEFINES -DPROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
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
find_package(OpenCV QUIET)
find_package(VTK CONFIG QUIET)
find_package(PCL REQUIRED COMPONENTS common io filters registration visualization)
find_package(Boost CONFIG COMPONENTS program_options)
find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)
find_package(PDAL CONFIG REQUIRED)
find_package(Glog REQUIRED)
find_package(PROJ CONFIG REQUIRED)
find_package(CUDAToolkit QUIET)

include_directories(
  src
  src/thirdparty/BTC
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(../migration migration_build)

set(PGO_COLMAP_TARGET "")
if(TARGET colmap::colmap)
  set(PGO_COLMAP_TARGET colmap::colmap)
elseif(TARGET colmap)
  set(PGO_COLMAP_TARGET colmap)
else()
  find_package(colmap CONFIG QUIET)
  if(TARGET colmap::colmap)
    set(PGO_COLMAP_TARGET colmap::colmap)
  elseif(TARGET colmap)
    set(PGO_COLMAP_TARGET colmap)
  endif()
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

if(PGO_COLMAP_TARGET AND OpenCV_FOUND AND CUDAToolkit_FOUND)
  set(PGO_XSFM_TOOL_LIBS
    ${PCL_LIBRARIES}
    $<TARGET_NAME:pdalcpp>
    spdlog::spdlog_header_only
    OpenMP::OpenMP_CXX
    ${PGO_COLMAP_TARGET}
  )

  set(XSFM_POST_DEPTH_OBJ "${CMAKE_CURRENT_BINARY_DIR}/xsfm_post_depth_$<CONFIG>.obj")
  add_custom_command(
    OUTPUT "${XSFM_POST_DEPTH_OBJ}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
      -std=c++17
      -gencode=arch=compute_75,code=compute_75
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
else()
  message(
    WARNING
    "Skipping xsfm_post build in pgo: COLMAP, OpenCV, or CUDA Toolkit was not found."
  )
endif()

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
