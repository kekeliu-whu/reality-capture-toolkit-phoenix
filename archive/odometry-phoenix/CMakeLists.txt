cmake_minimum_required(VERSION 3.20)
project(odometry_phoenix LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/MP)
endif()

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(Glog REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(PDAL REQUIRED)
find_package(OpenMP REQUIRED)

add_compile_definitions(
  NOMINMAX
  GLOG_USE_GLOG_EXPORT
  GLOG_USE_GFLAGS
  GLOG_NO_ABBREVIATED_SEVERITIES
  MP_EN
)

add_subdirectory(thirdparty/xmap xmap_build)
add_subdirectory(../../migration migration_build)

add_library(odometry_phoenix_core STATIC
  src/ieskf/ieskf.cpp
  src/ieskf/ieskf_predict.cpp
  src/ieskf/ieskf_update.cpp
  src/ieskf/state_group.cpp
  src/sensor_fusion/lidar_fusion.cpp
  src/sensor_fusion/static_fusion.cpp
  src/sensor_fusion/base_fusion.cpp
  src/lio.cpp
  src/common/common_lib.cpp
  src/method/lio_init.cpp
  src/method/preprocess.cpp
  src/method/io_utils.cpp
  src/method/uniform_sampling.cpp
  src/method/undistort.cpp
)

target_include_directories(odometry_phoenix_core PUBLIC
  interface
  src
  thirdparty/xmap/include
)

target_link_libraries(odometry_phoenix_core PUBLIC
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  yaml-cpp::yaml-cpp
  glog::glog
  spdlog::spdlog
  xmap_lib
  OpenMP::OpenMP_CXX
)

add_executable(odometry_phoenix app/main.cpp)
target_link_libraries(odometry_phoenix PRIVATE
  odometry_phoenix_core
  migration_core
  pdalcpp
  gflags::gflags
  glog::glog
  spdlog::spdlog
  yaml-cpp::yaml-cpp
  OpenMP::OpenMP_CXX
)

# Keep the packaged executable name stable for run-hs.ps1.
set_target_properties(odometry_phoenix PROPERTIES
  OUTPUT_NAME "REALTIME_MAPPING"
)

add_custom_command(TARGET odometry_phoenix POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/config/L2PRO.yaml"
          "$<TARGET_FILE_DIR:odometry_phoenix>/L2PRO.yaml"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/config/K1.yaml"
          "$<TARGET_FILE_DIR:odometry_phoenix>/K1.yaml"
)
