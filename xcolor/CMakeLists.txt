cmake_minimum_required(VERSION 3.10)
project(xcolor)

if(CMAKE_TOOLCHAIN_FILE)
    include(${CMAKE_TOOLCHAIN_FILE})
endif(CMAKE_TOOLCHAIN_FILE)

if(OPT_ASAN)
  add_definitions(-D_DISABLE_VECTOR_ANNOTATION)
  add_definitions(-D_DISABLE_STRING_ANNOTATION)
  add_compile_options(/fsanitize=address)
endif()

if(MSVC)
  add_compile_options(/MP)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob1")
endif()

add_definitions(-D_USE_MATH_DEFINES)
add_definitions(-DGLOG_USE_GLOG_EXPORT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost COMPONENTS program_options)
find_package(colmap REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(Glog REQUIRED)
find_package(GTest CONFIG REQUIRED)

include_directories(
  src
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_subdirectory(../migration migration_build)

add_executable(
  xcolor
  src/xcolor_main.cc
  src/core/xcolor_lib.cc
  src/io/read_write.cc
)

target_link_libraries(
  xcolor
  migration_core
  ${GFLAGS_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  glog::glog
  colmap::colmap
  absl::flat_hash_map
  absl::flat_hash_set
  ${OpenCV_LIBS}
)

add_executable(
  xsfm
  src/xsfm_main.cc
  src/core/xsfm_lib.cc
  src/common/histogram.cc
  src/io/read_write.cc
)

target_link_libraries(
  xsfm
  migration_core
  ${GFLAGS_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  glog::glog
  colmap::colmap
  absl::flat_hash_map
  absl::flat_hash_set
  ${OpenCV_LIBS}
  rapidjson
)

add_executable(
  xsfm_reset_s20_cameras
  src/xsfm_reset_s20_cameras.cc
)

target_link_libraries(
  xsfm_reset_s20_cameras
  migration_core
  ${GFLAGS_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  glog::glog
  colmap::colmap
  absl::flat_hash_map
  absl::flat_hash_set
  ${OpenCV_LIBS}
  rapidjson
)
