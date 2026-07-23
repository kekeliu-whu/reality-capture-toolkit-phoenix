cmake_minimum_required(VERSION 3.20)
project(realtime_mapping_offline LANGUAGES C CXX)

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

set(LIO_CORE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../lio_core")

add_subdirectory("${LIO_CORE_PATH}/thirdparty/xmap" xmap_build)
add_subdirectory("${LIO_CORE_PATH}/thirdparty/lslog" lslog_build)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../migration" migration_build)

add_library(realtime_lio_core STATIC
  "${LIO_CORE_PATH}/src/ieskf/ieskf.cpp"
  "${LIO_CORE_PATH}/src/ieskf/ieskf_predict.cpp"
  "${LIO_CORE_PATH}/src/ieskf/ieskf_update.cpp"
  "${LIO_CORE_PATH}/src/ieskf/state_group.cpp"
  "${LIO_CORE_PATH}/src/sensor_fusion/lidar_fusion.cpp"
  "${LIO_CORE_PATH}/src/sensor_fusion/static_fusion.cpp"
  "${LIO_CORE_PATH}/src/sensor_fusion/base_fusion.cpp"
  "${LIO_CORE_PATH}/src/lio.cpp"
  "${LIO_CORE_PATH}/src/common/common_lib.cpp"
  "${LIO_CORE_PATH}/src/method/lio_init.cpp"
  "${LIO_CORE_PATH}/src/method/preprocess.cpp"
  "${LIO_CORE_PATH}/src/method/io_utils.cpp"
  "${LIO_CORE_PATH}/src/method/uniform_sampling.cpp"
  "${LIO_CORE_PATH}/src/method/undistort.cpp"
)

target_include_directories(realtime_lio_core PUBLIC
  "${LIO_CORE_PATH}/interface"
  "${LIO_CORE_PATH}/src"
  "${LIO_CORE_PATH}/thirdparty/sophus"
  "${LIO_CORE_PATH}/thirdparty/xmap/include"
  "${LIO_CORE_PATH}/thirdparty/lslog/include"
)

target_link_libraries(realtime_lio_core PUBLIC
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  yaml-cpp::yaml-cpp
  glog::glog
  spdlog::spdlog
  xmap_lib
  xgrids::log
  OpenMP::OpenMP_CXX
)

add_executable(realtime_mapping main.cpp)
target_link_libraries(realtime_mapping PRIVATE
  realtime_lio_core
  migration_core
  pdalcpp
  gflags::gflags
  glog::glog
  spdlog::spdlog
  yaml-cpp::yaml-cpp
  OpenMP::OpenMP_CXX
)

set_target_properties(realtime_mapping PROPERTIES
  OUTPUT_NAME "REALTIME_MAPPING"
)

add_custom_command(TARGET realtime_mapping POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/config/L2PRO.yaml"
          "$<TARGET_FILE_DIR:realtime_mapping>/L2PRO.yaml"
)
