cmake_minimum_required(VERSION 3.10)
project(pgo)

if(CMAKE_TOOLCHAIN_FILE)
    include(${CMAKE_TOOLCHAIN_FILE})
endif(CMAKE_TOOLCHAIN_FILE)

add_definitions(-D_USE_MATH_DEFINES)

if(MSVC)
  add_compile_options(/MP)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob1")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters registration)
find_package(Boost CONFIG COMPONENTS program_options)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PDAL CONFIG REQUIRED)
if(WIN32)
  find_package(Glog REQUIRED)
endif()

include_directories(
  src
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_subdirectory(../migration migration_build)

add_executable(
  pgo_main
  src/pgo_main.cc
  src/map/optimizer.cc
  src/map/pgo_runner.cc
  src/map/utils.cc
  src/io/read_write.cc
)

target_link_libraries(
  pgo_main
  migration_core
  ${GFLAGS_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  glog::glog
  OpenMP::OpenMP_CXX
  $<TARGET_NAME:pdalcpp>
)

add_executable(
  process_point_cloud_std
  src/tools/process_point_cloud_std.cc
  src/io/read_write.cc
  src/map/utils.cc
)

target_link_libraries(
  process_point_cloud_std
  migration_core
  glog::glog
  ${PCL_LIBRARIES}
  opencv_core
  opencv_highgui
  opencv_calib3d
  ${GFLAGS_LIBRARIES}
  OpenMP::OpenMP_CXX
  $<TARGET_NAME:pdalcpp>
)

add_executable(
  process_point_cloud_s10
  src/tools/process_point_cloud_s10.cc
  src/io/read_write.cc
  src/map/utils.cc
)

target_link_libraries(
  process_point_cloud_s10
  migration_core
  glog::glog
  ${PCL_LIBRARIES}
  opencv_core
  opencv_highgui
  opencv_calib3d
  ${GFLAGS_LIBRARIES}
  OpenMP::OpenMP_CXX
  $<TARGET_NAME:pdalcpp>
)

add_executable(
  calib_camera_pose
  src/tools/calib_camera_pose.cc
  src/io/read_write.cc
)

target_link_libraries(
  calib_camera_pose
  migration_core
  glog::glog
  ${PCL_LIBRARIES}
  opencv_core
  opencv_highgui
  opencv_calib3d
  $<TARGET_NAME:pdalcpp>
)
