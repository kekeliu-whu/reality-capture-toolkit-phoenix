cmake_minimum_required(VERSION 3.10)
project(pgo)

if(CMAKE_TOOLCHAIN_FILE)
    include(${CMAKE_TOOLCHAIN_FILE})
endif(CMAKE_TOOLCHAIN_FILE)

add_definitions(-D_USE_MATH_DEFINES -DPROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
  add_compile_options(/MP)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Ob3 /MD /Zi")
  add_compile_options($<$<CONFIG:Release>:/Zi>)
  add_link_options($<$<CONFIG:Release>:/DEBUG>)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
find_package(VTK CONFIG QUIET)
find_package(PCL REQUIRED COMPONENTS common io filters registration visualization)
find_package(Boost CONFIG COMPONENTS program_options)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PDAL CONFIG REQUIRED)
find_package(Glog REQUIRED)
find_package(PROJ CONFIG REQUIRED)

include_directories(
  src
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_subdirectory(../migration migration_build)


set(
  PGO_PRIVATE_SRCS
  src/BTC.cpp
  src/map/optimizer.cc
  src/map/pgo_runner.cc
  src/map/utils.cc
  src/io/read_write.cc
  src/io/local_enu_transformer.cc
  src/common/msg_conversions.h
  src/common/types.h
  src/factor/gps_factor.h
  src/factor/gravity_factor.h
  src/factor/local_parameterization_se3.h
  src/factor/pose_graph_edge_factor.h
  src/factor/rtk_factor.h
  src/io/read_write.h
  src/io/local_enu_transformer.h
  src/BTC.h
  src/map/optimizer.h
  src/map/pgo_runner.h
  src/map/utils.h
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
  opencv_core
  opencv_highgui
  opencv_calib3d
  PROJ::proj
)
if(VTK_FOUND)
  list(APPEND PGO_PRIVATE_LIBS ${VTK_LIBRARIES})
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
add_private_tool(process_point_cloud_std src/tools/process_point_cloud_std.cc)
add_private_tool(calib_camera_pose src/tools/calib_camera_pose.cc)
