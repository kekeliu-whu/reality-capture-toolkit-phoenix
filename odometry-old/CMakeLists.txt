cmake_minimum_required(VERSION 3.10)
project(odometry-old)

if(CMAKE_TOOLCHAIN_FILE)
    include(${CMAKE_TOOLCHAIN_FILE})
endif(CMAKE_TOOLCHAIN_FILE)

if(OPT_ASAN)
  add_definitions(-D_DISABLE_VECTOR_ANNOTATION)
  add_definitions(-D_DISABLE_STRING_ANNOTATION)
  add_compile_options(/fsanitize=address)
endif()

if(MSVC)
  add_compile_options(/MP /bigobj /utf-8)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob1")
endif()

add_definitions(-D_USE_MATH_DEFINES)
add_definitions(-DGLOG_USE_GLOG_EXPORT)
add_definitions(-DROOT_DIR="D:/ProjectX/project-3d/reality-capture-toolkit/odometry-old/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost COMPONENTS program_options)
find_package(OpenCV CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(Glog REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(mimalloc CONFIG REQUIRED)
find_package(crashpad CONFIG REQUIRED)
find_package(PDAL CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

include_directories(
  include
  src
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
)

add_subdirectory(../migration migration_build)


set(
  PGO_PRIVATE_SRCS
  include/voxel_map_util.cc
  include/use-ikfom.cc
  src/preprocess.cc
  src/imu_processing.cc
)

set(PGO_PRIVATE_LIBS
  mimalloc
  migration_core
  ${GFLAGS_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  glog::glog
  absl::flat_hash_map
  absl::flat_hash_set
  rapidjson
  crashpad::crashpad
  $<TARGET_NAME:pdalcpp>
  fmt::fmt
  ${OpenCV_LIBS}
  spdlog::spdlog_header_only
)

function(add_private_tool name)
  add_executable(${name} ${ARGN} ${PGO_PRIVATE_SRCS})
  target_link_libraries(${name} ${PGO_PRIVATE_LIBS})
endfunction()

add_private_tool(slam_core_main src/laser_mapping.cc)
add_private_tool(smooth_cloud src/smooth_cloud.cc)
add_private_tool(colorize_cloud_general src/colorize_cloud_general.cc)
add_private_tool(gravity_align src/gravity_align.cc)
