cmake_minimum_required(VERSION 3.20)
project(realtime_mapping_offline LANGUAGES C CXX)

option(OFFLINE_ASAN "Build the offline runner with AddressSanitizer" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/MP /bigobj /utf-8)
  add_compile_definitions(NOMINMAX GLOG_USE_GLOG_EXPORT MP_EN)
  if(OFFLINE_ASAN)
    add_compile_options(/fsanitize=address)
    add_link_options(/INFERASANLIBS)
  endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(PDAL CONFIG REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog CONFIG REQUIRED)

set(REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(LIO_CORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../lio_core")

# Reuse the repository's protobuf sensor readers and incremental LAS writer.
add_subdirectory("${REPOSITORY_ROOT}/migration" migration_build)
add_subdirectory("${LIO_CORE_ROOT}/thirdparty/xmap" xmap_build)
add_subdirectory("${LIO_CORE_ROOT}/thirdparty/lslog" lslog_build)

add_library(realtime_lio_core STATIC
  ${LIO_CORE_ROOT}/src/ieskf/ieskf.cpp
  ${LIO_CORE_ROOT}/src/ieskf/ieskf_predict.cpp
  ${LIO_CORE_ROOT}/src/ieskf/ieskf_update.cpp
  ${LIO_CORE_ROOT}/src/ieskf/state_group.cpp
  ${LIO_CORE_ROOT}/src/sensor_fusion/lidar_fusion.cpp
  ${LIO_CORE_ROOT}/src/sensor_fusion/static_fusion.cpp
  ${LIO_CORE_ROOT}/src/sensor_fusion/base_fusion.cpp
  ${LIO_CORE_ROOT}/src/lio.cpp
  ${LIO_CORE_ROOT}/src/common/common_lib.cpp
  ${LIO_CORE_ROOT}/src/method/lio_init.cpp
  ${LIO_CORE_ROOT}/src/method/preprocess.cpp
  ${LIO_CORE_ROOT}/src/method/io_utils.cpp
  ${LIO_CORE_ROOT}/src/method/uniform_sampling.cpp
  ${LIO_CORE_ROOT}/src/method/undistort.cpp
)

target_include_directories(realtime_lio_core PUBLIC
  ${LIO_CORE_ROOT}/interface
  ${LIO_CORE_ROOT}/src
  ${LIO_CORE_ROOT}/thirdparty/sophus
  ${LIO_CORE_ROOT}/thirdparty/xmap/include
  ${LIO_CORE_ROOT}/thirdparty/lslog/include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(realtime_lio_core PUBLIC
  xmap_lib
  xgrids::log
  OpenMP::OpenMP_CXX
  ${PCL_LIBRARIES}
  yaml-cpp::yaml-cpp
)

add_executable(realtime_mapping main.cpp)
target_link_libraries(realtime_mapping PRIVATE
  realtime_lio_core
  migration_core
  gflags::gflags
  spdlog::spdlog
  pdalcpp
)

add_custom_command(TARGET realtime_mapping POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/config/K1.yaml"
    "$<TARGET_FILE_DIR:realtime_mapping>/K1.yaml"
)
